#!/usr/bin/env python3
"""Cross-dataset transfer of the frozen ten-percent BIU correction to PIE."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
REPO_ROOT = ROOT.parents[1]
EXP084 = REPO_ROOT / "experiments/084_pedcmt_biu_transfer"
EXP084_SCRIPTS = EXP084 / "scripts"
EXP090 = REPO_ROOT / "experiments/090_pedcmt_direct_biu_fusion"
if str(EXP084_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(EXP084_SCRIPTS))
if str(EXP090 / "scripts") not in sys.path:
    sys.path.insert(0, str(EXP090 / "scripts"))

import exp084_data as EXP084_DATA  # noqa: E402
from pedcmt_model import PedCMTTransfer  # noqa: E402


def _load_runner(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE84 = _load_runner(EXP084 / "scripts/run_experiment.py", "exp084_runner_for_exp093")
BASE90 = _load_runner(EXP090 / "scripts/run_experiment.py", "exp090_runner_for_exp093")

PARENT_CHECKPOINTS = EXP084 / "checkpoints"
PARENT_PREDICTIONS = EXP084 / "predictions"
CHECKPOINTS = ROOT / "checkpoints"
CACHE = ROOT / "cache/trainval"
LOGS = ROOT / "logs"
METRICS = ROOT / "metrics"
PREDICTIONS = ROOT / "predictions"
VALIDATION_COMPLETE_MARKER = ROOT / "VALIDATION_COMPLETE.json"
SEEDS = (42, 43, 44)
BIU_CORRECTION_SCALE = 0.10
PARENT_CHECKPOINT_SHA256 = {
    42: "2afa77b4bb55b7f6cc28e85af547771406ab7e6b7ac9fa9f63d04168298c9efb",
    43: "63c16d30ddd7403049762c79c82120e21aa89897b700dbaa9db6010a697e4bbb",
    44: "f822c331fe334690ad7856252bb617218434545dd3ec0c63eeabdbb25e81d2b0",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def authenticate_parent_checkpoints() -> None:
    for seed, expected in PARENT_CHECKPOINT_SHA256.items():
        path = PARENT_CHECKPOINTS / f"A_seed{seed}.best.pt"
        if not path.is_file():
            raise FileNotFoundError(f"Missing Exp084 A checkpoint: {path}")
        observed = sha256(path)
        if observed != expected:
            raise ValueError(f"Exp084 A seed {seed} SHA drift: {observed}")


def scaled_logits(parent_logits: np.ndarray, correction: np.ndarray) -> np.ndarray:
    parent = np.asarray(parent_logits, dtype=np.float64)
    delta = np.asarray(correction, dtype=np.float64)
    if parent.shape != delta.shape:
        raise ValueError("Parent logits and BIU corrections must align")
    return parent + BIU_CORRECTION_SCALE * delta


def sigmoid(logits: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(logits, dtype=np.float64), -80.0, 80.0)
    return 1.0 / (1.0 + np.exp(-values))


def assert_probability_parity(
    observed: np.ndarray,
    expected: np.ndarray,
    *,
    tolerance: float = 2e-6,
) -> float:
    difference = float(
        np.max(np.abs(np.asarray(observed, dtype=np.float64) - np.asarray(expected, dtype=np.float64)))
    )
    if difference > tolerance:
        raise RuntimeError(f"Exp084 parent reproduction failed: {difference:.3g}")
    return difference


def load_trainval() -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    population = EXP084_DATA.load_population("trainval", allow_test=False)
    arrays = {
        "bbox": population.bbox,
        "ego": population.ego,
        "biu_union": np.load(EXP084 / "cache/trainval/biu_union_std.npy", mmap_mode="r"),
    }
    if population.rows["split"].value_counts().to_dict() != {"val": 3619, "train": 2366}:
        raise RuntimeError("Exp084 train/validation population drifted")
    expected = {"bbox": (5985, 16, 4), "biu_union": (5985, 135)}
    for name, shape in expected.items():
        if arrays[name].shape != shape:
            raise RuntimeError(f"Exp084 {name} shape drifted: {arrays[name].shape}")
    if arrays["ego"].shape[:2] != (5985, 16):
        raise RuntimeError(f"Exp084 ego shape drifted: {arrays['ego'].shape}")
    return population.rows, arrays


def load_parent(seed: int, ego_dim: int, device: torch.device) -> PedCMTTransfer:
    authenticate_parent_checkpoints()
    checkpoint = torch.load(PARENT_CHECKPOINTS / f"A_seed{seed}.best.pt", map_location=device, weights_only=False)
    model = PedCMTTransfer(variant="A", ego_dim=ego_dim).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def extract_parent_outputs(
    model: PedCMTTransfer,
    bbox: np.ndarray,
    ego: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(np.asarray(bbox, dtype=np.float32)),
            torch.from_numpy(np.asarray(ego, dtype=np.float32)),
        ),
        batch_size=256,
        shuffle=False,
        num_workers=0,
    )
    logits: list[np.ndarray] = []
    anchors: list[np.ndarray] = []
    captured: list[Tensor] = []

    def capture(_module: nn.Module, inputs: tuple[Tensor, ...]) -> None:
        captured.append(inputs[0].detach())

    handle = model.endpoint.register_forward_pre_hook(capture)
    try:
        with torch.inference_mode():
            for bbox_batch, ego_batch in loader:
                captured.clear()
                output = model(bbox_batch.to(device), ego_batch.to(device))[0]
                if len(captured) != 1 or captured[0].shape[1] != 256:
                    raise RuntimeError("Exp084 full parent anchor capture failed")
                logits.append(output.cpu().numpy())
                anchors.append(captured[0].cpu().numpy())
    finally:
        handle.remove()
    return np.concatenate(logits), np.concatenate(anchors)


def prepare(device_name: str) -> None:
    authenticate_parent_checkpoints()
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    rows, arrays = load_trainval()
    val_indices = np.flatnonzero(rows["split"].eq("val").to_numpy())
    CACHE.mkdir(parents=True, exist_ok=True)
    parity: dict[str, float] = {}
    for seed in SEEDS:
        parent = load_parent(seed, arrays["ego"].shape[-1], device)
        logits, anchor = extract_parent_outputs(parent, arrays["bbox"], arrays["ego"], device)
        saved = pd.read_csv(PARENT_PREDICTIONS / f"validation_A_seed{seed}.csv")
        if not np.array_equal(rows.iloc[val_indices]["sequence_id"].astype(str), saved["sequence_id"].astype(str)):
            raise RuntimeError(f"Exp084 validation order drifted for seed {seed}")
        probability = sigmoid(logits[val_indices].reshape(-1))
        difference = assert_probability_parity(probability, saved["probability"].to_numpy())
        np.save(CACHE / f"parent_seed{seed}_logits.npy", logits.astype(np.float32))
        np.save(CACHE / f"parent_seed{seed}_anchor.npy", anchor.astype(np.float32))
        parity[str(seed)] = difference
    audit = {
        "parent_experiment": str(EXP084),
        "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "train_windows": 2366,
        "validation_windows": 3619,
        "biu_union_dimension": 135,
        "correction_scale_transferred_from_jaad": BIU_CORRECTION_SCALE,
        "parent_probability_max_abs_difference": parity,
        "test_cache_accessed": False,
    }
    atomic_json(METRICS / "source_and_parent_audit.json", audit)
    atomic_json(ROOT / "PREPARATION_COMPLETE.json", audit)


def _cached_parent(seed: int) -> tuple[np.ndarray, np.ndarray]:
    logits = np.load(CACHE / f"parent_seed{seed}_logits.npy", mmap_mode="r")
    anchor = np.load(CACHE / f"parent_seed{seed}_anchor.npy", mmap_mode="r")
    if logits.shape != (5985, 1) or anchor.shape != (5985, 256):
        raise RuntimeError(f"Exp093 cached parent outputs drifted for seed {seed}")
    return logits, anchor


def _values(indices: np.ndarray, rows: pd.DataFrame, logits: np.ndarray, anchor: np.ndarray, union: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "parent_logits": np.asarray(logits[indices], dtype=np.float32),
        "parent_anchor": np.asarray(anchor[indices], dtype=np.float32),
        "extra": np.asarray(union[indices], dtype=np.float32),
        "labels": rows.iloc[indices]["label"].to_numpy(dtype=np.float32),
    }


def train_one(seed: int, device_name: str, *, smoke: bool = False) -> dict[str, Any]:
    if not (ROOT / "PREPARATION_COMPLETE.json").is_file():
        prepare(device_name)
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    rows, arrays = load_trainval()
    logits, anchor = _cached_parent(seed)
    train_base = np.flatnonzero(rows["split"].eq("train").to_numpy())
    balanced = BASE84.balanced_indices(rows.iloc[train_base]["label"].to_numpy(), seed)
    train_indices = train_base[balanced]
    val_indices = np.flatnonzero(rows["split"].eq("val").to_numpy())
    stem = f"{'smoke_' if smoke else ''}R_seed{seed}"
    result = BASE90.fit_branch(
        "C",
        _values(train_indices, rows, logits, anchor, arrays["biu_union"]),
        _values(val_indices, rows, logits, anchor, arrays["biu_union"]),
        seed=seed,
        device=device,
        checkpoint_path=CHECKPOINTS / f"{stem}.best.pt",
        history_path=LOGS / f"{stem}.history.csv",
        parent_sha256=PARENT_CHECKPOINT_SHA256[seed],
        max_epochs=1 if smoke else BASE90.MAX_EPOCHS,
        patience=1 if smoke else BASE90.PATIENCE,
    )
    correction = result.pop("correction")
    result.pop("probability")
    parent_logits = np.asarray(logits[val_indices], dtype=np.float64).reshape(-1)
    probability = sigmoid(scaled_logits(parent_logits, correction))
    output = rows.iloc[val_indices][["sequence_id", "set_id", "video_id", "ped_id", "label"]].copy()
    output["parent_probability"] = sigmoid(parent_logits)
    output["full_biu_correction_logit"] = correction
    output["scaled_biu_correction_logit"] = BIU_CORRECTION_SCALE * correction
    output["probability"] = probability
    output["prediction"] = (probability >= 0.5).astype(np.int64)
    PREDICTIONS.mkdir(parents=True, exist_ok=True)
    output.to_csv(PREDICTIONS / f"validation_{stem}.csv", index=False)
    result["inference_correction_scale"] = BIU_CORRECTION_SCALE
    atomic_json(LOGS / f"{stem}.complete.json", result)
    return result


def train_all(device_name: str) -> None:
    if not (ROOT / "PREPARATION_COMPLETE.json").is_file():
        prepare(device_name)
    for seed in SEEDS:
        if (CHECKPOINTS / f"R_seed{seed}.best.pt").is_file() and (LOGS / f"R_seed{seed}.complete.json").is_file():
            print(f"Skipping complete R seed {seed}", flush=True)
            continue
        print(json.dumps(train_one(seed, device_name), indent=2), flush=True)
    missing = [seed for seed in SEEDS if not (CHECKPOINTS / f"R_seed{seed}.best.pt").is_file()]
    if missing:
        raise RuntimeError(f"Missing PIE correction checkpoints: {missing}")
    atomic_json(ROOT / "TRAINING_COMPLETE.json", {"variant": "R10", "seeds": list(SEEDS)})


def validate() -> None:
    if not (ROOT / "TRAINING_COMPLETE.json").is_file():
        raise RuntimeError("Complete all three correction fits before validation")
    rows, _ = load_trainval()
    val = rows.loc[rows["split"].eq("val")].reset_index(drop=True)
    labels = val["label"].to_numpy(dtype=np.int64)
    metric_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        saved = pd.read_csv(PREDICTIONS / f"validation_R_seed{seed}.csv")
        if not np.array_equal(val["sequence_id"].astype(str), saved["sequence_id"].astype(str)):
            raise RuntimeError(f"Exp093 validation order drifted for seed {seed}")
        metric_rows.append({"seed": seed, "variant": "A_parent", **BASE84.fixed_metrics(labels, saved["parent_probability"])})
        metric_rows.append({"seed": seed, "variant": "R10", **BASE84.fixed_metrics(labels, saved["probability"])})
    results = pd.DataFrame(metric_rows)
    METRICS.mkdir(parents=True, exist_ok=True)
    results.to_csv(METRICS / "validation_per_seed_results.csv", index=False)
    summary = results.groupby("variant")[["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "fp", "fn", "high_conf_fp", "high_conf_fn"]].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary.reset_index().to_csv(METRICS / "validation_summary_results.csv", index=False)
    atomic_json(
        VALIDATION_COMPLETE_MARKER,
        {"scope": "PIE validation only", "scale": BIU_CORRECTION_SCALE, "test_cache_accessed": False},
    )


def _load_branch(seed: int, device: torch.device):
    checkpoint = torch.load(CHECKPOINTS / f"R_seed{seed}.best.pt", map_location=device, weights_only=False)
    if checkpoint.get("parent_sha256") != PARENT_CHECKPOINT_SHA256[seed]:
        raise RuntimeError(f"PIE correction seed {seed} parent lineage mismatch")
    model = BASE90.DirectFusionModel("C").to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def final_evaluate(device_name: str) -> None:
    if not VALIDATION_COMPLETE_MARKER.is_file():
        raise RuntimeError("Final set03 is locked until validation completion")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    population = EXP084_DATA.load_population("test", allow_test=True)
    stats = np.load(EXP084 / "cache/trainval/training_stats.npz")
    union, _ = EXP084_DATA.build_union(
        population.biu_v1,
        population.biu_v2,
        population.biu_v1_names,
        population.biu_v2_names,
    )
    union = BASE84._apply_stats(union, stats["union_mean"], stats["union_std"])
    labels = population.rows["label"].to_numpy(dtype=np.int64)
    metric_rows: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    for seed in SEEDS:
        parent = load_parent(seed, population.ego.shape[-1], device)
        parent_logits, parent_anchor = extract_parent_outputs(parent, population.bbox, population.ego, device)
        parent_logits = parent_logits.reshape(-1)
        parent_probability = sigmoid(parent_logits)
        saved_parent = pd.read_csv(PARENT_PREDICTIONS / f"test_A_seed{seed}.csv")
        if not np.array_equal(population.rows["sequence_id"].astype(str), saved_parent["sequence_id"].astype(str)):
            raise RuntimeError(f"Exp084 set03 order drifted for seed {seed}")
        assert_probability_parity(parent_probability, saved_parent["probability"].to_numpy())
        branch = _load_branch(seed, device)
        values = {
            "parent_logits": parent_logits.reshape(-1, 1).astype(np.float32),
            "parent_anchor": parent_anchor.astype(np.float32),
            "extra": union.astype(np.float32),
            "labels": labels.astype(np.float32),
        }
        _, _, correction = BASE90.evaluate_branch(branch, values, device)
        probability = sigmoid(scaled_logits(parent_logits, correction))
        metric_rows.append({"seed": seed, "variant": "A_parent", **BASE84.fixed_metrics(labels, parent_probability)})
        metric_rows.append({"seed": seed, "variant": "R10", **BASE84.fixed_metrics(labels, probability)})
        before = parent_probability >= 0.5
        after = probability >= 0.5
        transitions.append(
            {
                "seed": seed,
                "wrong_to_correct": int(np.sum((before != labels) & (after == labels))),
                "correct_to_wrong": int(np.sum((before == labels) & (after != labels))),
                "fp_to_tn": int(np.sum((labels == 0) & before & ~after)),
                "fn_to_tp": int(np.sum((labels == 1) & ~before & after)),
                "tn_to_fp": int(np.sum((labels == 0) & ~before & after)),
                "tp_to_fn": int(np.sum((labels == 1) & before & ~after)),
            }
        )
        output = population.rows[["sequence_id", "set_id", "video_id", "ped_id", "label"]].copy()
        output["parent_probability"] = parent_probability
        output["full_biu_correction_logit"] = correction
        output["scaled_biu_correction_logit"] = BIU_CORRECTION_SCALE * correction
        output["probability"] = probability
        output["prediction"] = after.astype(np.int64)
        PREDICTIONS.mkdir(parents=True, exist_ok=True)
        output.to_csv(PREDICTIONS / f"test_R10_seed{seed}.csv", index=False)

    results = pd.DataFrame(metric_rows)
    METRICS.mkdir(parents=True, exist_ok=True)
    results.to_csv(METRICS / "per_seed_results.csv", index=False)
    results.to_csv(METRICS / "fixed_0p5_results.csv", index=False)
    summary = results.groupby("variant")[["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "fp", "fn", "high_conf_fp", "high_conf_fn"]].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary.reset_index().to_csv(METRICS / "summary_results.csv", index=False)
    deltas: list[dict[str, Any]] = []
    for seed in SEEDS:
        indexed = results.loc[results["seed"].eq(seed)].set_index("variant")
        deltas.append(
            {"seed": seed, **{f"delta_{metric}": float(indexed.loc["R10", metric] - indexed.loc["A_parent", metric]) for metric in ("accuracy", "f1", "roc_auc", "pr_auc")}}
        )
    pd.DataFrame(deltas).to_csv(METRICS / "paired_deltas.csv", index=False)
    pd.DataFrame(transitions).to_csv(METRICS / "error_transitions_fixed_0p5.csv", index=False)
    references = pd.read_csv(EXP084 / "metrics/summary_results.csv")
    references = references.loc[references["variant"].isin(["A", "B", "E"])].copy()
    references["source"] = "Exp084 saved same-cohort result"
    references.to_csv(METRICS / "exp084_reference_results.csv", index=False)
    atomic_json(
        ROOT / "EXPERIMENT_COMPLETE.json",
        {"scale": BIU_CORRECTION_SCALE, "test_population": "PIE set03", "test_windows": int(len(labels)), "training_runs": 3},
    )


def smoke(device_name: str) -> None:
    result = train_one(42, device_name, smoke=True)
    atomic_json(LOGS / "SMOKE_COMPLETE.json", {"device": device_name, "result": result})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "smoke", "train", "train-all", "validate", "evaluate"))
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.device)
    elif args.command == "smoke":
        smoke(args.device)
    elif args.command == "train":
        if args.seed is None:
            raise SystemExit("train requires --seed")
        print(json.dumps(train_one(args.seed, args.device), indent=2))
    elif args.command == "train-all":
        train_all(args.device)
    elif args.command == "validate":
        validate()
    else:
        final_evaluate(args.device)


if __name__ == "__main__":
    main()
