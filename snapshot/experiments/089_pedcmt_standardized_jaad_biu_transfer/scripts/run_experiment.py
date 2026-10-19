#!/usr/bin/env python3
"""Exp089 wrapper around the frozen Exp084 PedCMT training implementation."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
EXP_DIR = SCRIPT_DIR.parent
REPO_ROOT = EXP_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import exp089_data as data  # noqa: E402


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load frozen source: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


EXP084_SCRIPTS = REPO_ROOT / "experiments/084_pedcmt_biu_transfer/scripts"
model_module = _load("exp089_frozen_pedcmt_model", EXP084_SCRIPTS / "pedcmt_model.py")
previous = {name: sys.modules.get(name) for name in ("exp084_data", "pedcmt_model")}
try:
    sys.modules["exp084_data"] = data
    sys.modules["pedcmt_model"] = model_module
    core = _load("exp089_frozen_exp084_runner", EXP084_SCRIPTS / "run_experiment.py")
finally:
    for name, module in previous.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


CACHE_DIR = EXP_DIR / "cache"
TRAINVAL_DERIVED = CACHE_DIR / "trainval"
CHECKPOINT_DIR = EXP_DIR / "checkpoints"
PREDICTION_DIR = EXP_DIR / "predictions"
METRICS_DIR = EXP_DIR / "metrics"
LOG_DIR = EXP_DIR / "logs"
VARIANTS = tuple("ABCDE")
SEEDS = (42, 43, 44)
MAX_EPOCHS = 200
PATIENCE = 25
BATCH_SIZE = 64


def _bind_local_contract() -> None:
    bindings = {
        "EXP_DIR": EXP_DIR,
        "CACHE_DIR": CACHE_DIR,
        "TRAINVAL_DERIVED": TRAINVAL_DERIVED,
        "CHECKPOINT_DIR": CHECKPOINT_DIR,
        "PREDICTION_DIR": PREDICTION_DIR,
        "METRICS_DIR": METRICS_DIR,
        "LOG_DIR": LOG_DIR,
        "SEEDS": SEEDS,
        "VARIANTS": VARIANTS,
        "MAX_EPOCHS": MAX_EPOCHS,
        "PATIENCE": PATIENCE,
        "BATCH_SIZE": BATCH_SIZE,
        "load_population": data.load_population,
        "build_event_bbox_targets": data.build_event_bbox_targets,
        "build_union": data.build_union,
        "standardize_from_training": data.standardize_from_training,
    }
    for name, value in bindings.items():
        setattr(core, name, value)


_bind_local_contract()
balanced_indices = core.balanced_indices
fixed_metrics = core.fixed_metrics
multitask_loss = core.multitask_loss
is_better_checkpoint = core.is_better_checkpoint


def assert_test_gate(checkpoint_dir: Path) -> None:
    expected = [
        checkpoint_dir / f"{variant}_seed{seed}.best.pt"
        for variant in VARIANTS
        for seed in SEEDS
    ]
    missing = [path.name for path in expected if not path.is_file()]
    if missing:
        raise RuntimeError(
            f"final JAAD test is locked until all 15 frozen checkpoints exist; missing {len(missing)}"
        )


core.assert_test_gate = assert_test_gate


def prepare() -> None:
    _bind_local_contract()
    core.prepare()
    trainval_pose = data.audit_pose_sources(include_test=False)
    audit = {
        "manifest": data.audit_manifests(include_test=False),
        "trainval_pose": trainval_pose,
        "test_cache_accessed": False,
        "parent_runner": str(EXP084_SCRIPTS / "run_experiment.py"),
        "parent_model": str(EXP084_SCRIPTS / "pedcmt_model.py"),
    }
    core._atomic_json(METRICS_DIR / "jaad_invariant_audit.json", audit)


def smoke(device_name: str) -> None:
    _bind_local_contract()
    core.smoke(device_name)


def train_one(variant: str, seed: int, *, device_name: str):
    _bind_local_contract()
    return core.train_one(variant, seed, device_name=device_name)


def train_all(device_name: str) -> None:
    _bind_local_contract()
    core.train_all(device_name)


def evaluate(device_name: str) -> None:
    _bind_local_contract()
    assert_test_gate(CHECKPOINT_DIR)
    core.evaluate(device_name)
    marker = {
        "dataset": "JAAD-beh",
        "protocol": "standardized_16f_tte30_60",
        "test_population": 1501,
        "runs": 15,
        "threshold": 0.5,
        "pose_sources": [str(data.PRIMARY_POSE), str(data.SUPPLEMENT_POSE)],
    }
    core._atomic_json(EXP_DIR / "EXPERIMENT_COMPLETE.json", marker)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "smoke", "train", "train-all", "evaluate"))
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "smoke":
        smoke(args.device)
    elif args.command == "train":
        if args.variant is None or args.seed is None:
            raise SystemExit("train requires --variant and --seed")
        print(json.dumps(train_one(args.variant, args.seed, device_name=args.device), indent=2))
    elif args.command == "train-all":
        train_all(args.device)
    else:
        evaluate(args.device)


if __name__ == "__main__":
    main()
