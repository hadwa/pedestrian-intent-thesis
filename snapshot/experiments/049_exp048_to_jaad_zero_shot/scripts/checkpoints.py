"""Authenticate and load the frozen Exp048 models used by Exp049."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import torch

from contracts import (
    EXP048_CHECKPOINT_ROOT,
    EXP048_ROOT,
    EXPECTED_PARAMETERS,
    SEEDS,
    VARIANTS,
    CheckpointRecord,
    sha256_file,
)


def _load_module(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import immutable Exp048 module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _exp048_core():
    return _load_module(
        "exp049_immutable_exp048_core",
        EXP048_ROOT / "scripts" / "exp048_core.py",
    )


def _exp048_biu_v1_core():
    return _load_module(
        "exp049_immutable_exp048_biu_v1_core",
        EXP048_ROOT / "scripts" / "biu_v1_amendment_core.py",
    )


def authenticate_checkpoint(
    checkpoint_path: Path,
    marker_path: Path,
    variant: str,
    seed: int,
) -> CheckpointRecord:
    checkpoint_path = Path(checkpoint_path)
    marker_path = Path(marker_path)
    if variant not in VARIANTS or int(seed) not in SEEDS:
        raise ValueError(f"Unapproved checkpoint identity: {variant}{seed}.")
    if not checkpoint_path.is_file() or not marker_path.is_file():
        raise ValueError(f"Missing checkpoint or completion marker for {variant}{seed}.")

    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("phase") != "train":
        raise ValueError(f"Invalid completion phase for {variant}{seed}.")
    if marker.get("variant") != variant or int(marker.get("seed", -1)) != int(seed):
        raise ValueError(f"Completion marker identity mismatch for {variant}{seed}.")
    matches = [
        value
        for name, value in marker.get("artifacts", {}).items()
        if Path(name).name == checkpoint_path.name
    ]
    if len(matches) != 1:
        raise ValueError(f"Completion marker does not uniquely authenticate {checkpoint_path.name}.")
    actual_sha = sha256_file(checkpoint_path)
    if actual_sha != matches[0]:
        raise ValueError(f"Checkpoint hash mismatch for {variant}{seed}.")

    total_parameters = int(marker.get("total_parameters", -1))
    if total_parameters != EXPECTED_PARAMETERS[variant]:
        raise ValueError(f"Parameter-count drift for {variant}{seed}.")
    return CheckpointRecord(
        variant=variant,
        seed=int(seed),
        checkpoint_path=checkpoint_path.resolve(),
        marker_path=marker_path.resolve(),
        checkpoint_sha256=actual_sha,
        total_parameters=total_parameters,
    )


def load_frozen_model(record: CheckpointRecord, device: str | torch.device):
    if record.variant == "F":
        model = _exp048_biu_v1_core().build_f_model(record.seed)
    else:
        model = _exp048_core().build_model(record.variant, record.seed)
    payload = torch.load(record.checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("variant") != record.variant or int(payload.get("seed", -1)) != record.seed:
        raise ValueError(f"Checkpoint payload identity mismatch for {record.variant}{record.seed}.")
    if int(payload.get("total_parameters", -1)) != record.total_parameters:
        raise ValueError(f"Checkpoint parameter metadata drift for {record.variant}{record.seed}.")
    model.load_state_dict(payload["model_state"], strict=True)
    actual_parameters = sum(parameter.numel() for parameter in model.parameters())
    if actual_parameters != record.total_parameters:
        raise ValueError(f"Instantiated parameter-count drift for {record.variant}{record.seed}.")
    model.requires_grad_(False)
    return model.to(device).eval()


def authenticate_exp048_checkpoints() -> list[CheckpointRecord]:
    records = []
    for variant in VARIANTS:
        for seed in SEEDS:
            records.append(
                authenticate_checkpoint(
                    EXP048_CHECKPOINT_ROOT / f"{variant}_seed{seed}.best.pt",
                    EXP048_ROOT / f".train_{variant}_seed{seed}.complete.json",
                    variant,
                    seed,
                )
            )
    expected = {(variant, seed) for variant in VARIANTS for seed in SEEDS}
    actual = {(record.variant, record.seed) for record in records}
    if actual != expected or len(records) != len(expected):
        raise ValueError("Checkpoint authentication did not produce exactly 18 identities.")
    return records
