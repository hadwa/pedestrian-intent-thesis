#!/usr/bin/env python3
"""Evaluate the frozen ten-percent Exp091 BIU correction."""

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


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
REPO_ROOT = ROOT.parents[1]
EXP090 = REPO_ROOT / "experiments/090_pedcmt_direct_biu_fusion"
EXP091 = REPO_ROOT / "experiments/091_pedcmt_natural_sampling_calibration"
BRANCH_CHECKPOINTS = EXP091 / "checkpoints"
METRICS = ROOT / "metrics"
PREDICTIONS = ROOT / "predictions"
VALIDATION_GATE_MARKER = ROOT / "VALIDATION_GATE_PASSED.json"
SEEDS = (42, 43, 44)
BIU_CORRECTION_SCALE = 0.10
BRANCH_CHECKPOINT_SHA256 = {
    42: "e45ef5131a0ed27fc78a128e82a56bd72cd3a29198aad69b2dc2cab6d242aa42",
    43: "9f1c4659b38ffe4121d1a5c8283d0ceafc6e4ecff8a0bd6793fa1d3391824a89",
    44: "708bc00739f85ceb05f4a4d64cf80682d236e4237758c09273349d4af59946fc",
}


def _load_exp090_runner():
    path = EXP090 / "scripts/run_experiment.py"
    spec = importlib.util.spec_from_file_location("exp090_runner_for_exp092", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = _load_exp090_runner()


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


def authenticate_branch_checkpoints() -> None:
    BASE.authenticate_parent_checkpoints()
    for seed, expected in BRANCH_CHECKPOINT_SHA256.items():
        path = BRANCH_CHECKPOINTS / f"N_seed{seed}.best.pt"
        if not path.is_file():
            raise FileNotFoundError(f"Missing Exp091 checkpoint: {path}")
        observed = sha256(path)
        if observed != expected:
            raise ValueError(f"Exp091 N seed {seed} SHA drift: {observed}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint.get("parent_sha256") != BASE.PARENT_CHECKPOINT_SHA256[seed]:
            raise RuntimeError(f"Exp091 N seed {seed} parent lineage mismatch")


def scaled_logits(parent_logits: np.ndarray, correction: np.ndarray) -> np.ndarray:
    parent = np.asarray(parent_logits, dtype=np.float64)
    delta = np.asarray(correction, dtype=np.float64)
    if parent.shape != delta.shape:
        raise ValueError("Parent logits and BIU corrections must align")
    return parent + BIU_CORRECTION_SCALE * delta


def sigmoid(logits: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(logits, dtype=np.float64), -80.0, 80.0)
    return 1.0 / (1.0 + np.exp(-values))


def probability_to_logits(probabilities: np.ndarray) -> np.ndarray:
    probability = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-8, 1.0 - 1e-8)
    return np.log(probability) - np.log1p(-probability)


def evaluate_gate(results: pd.DataFrame) -> dict[str, Any]:
    if set(results["variant"]) != {"A_parent", "N10"} or len(results) != 6:
        raise ValueError("Gate requires A_parent and N10 for all three seeds")
    means = results.groupby("variant").mean(numeric_only=True)
    criteria = {
        "pr_auc_gain_at_least_0p005": float(means.loc["N10", "pr_auc"])
        >= float(means.loc["A_parent", "pr_auc"]) + 0.005,
        "f1_within_0p005": float(means.loc["N10", "f1"])
        >= float(means.loc["A_parent", "f1"]) - 0.005,
        "accuracy_within_0p005": float(means.loc["N10", "accuracy"])
        >= float(means.loc["A_parent", "accuracy"]) - 0.005,
        "no_high_confidence_error_increase": float(
            means.loc["N10", "high_conf_fp"] + means.loc["N10", "high_conf_fn"]
        )
        <= float(
            means.loc["A_parent", "high_conf_fp"] + means.loc["A_parent", "high_conf_fn"]
        ),
    }
    return {
        "passed": bool(all(criteria.values())),
        "criteria": {name: bool(value) for name, value in criteria.items()},
        "scale": BIU_CORRECTION_SCALE,
        "mean_metrics": {
            variant: {
                metric: float(means.loc[variant, metric])
                for metric in ("accuracy", "f1", "roc_auc", "pr_auc", "high_conf_fp", "high_conf_fn")
            }
            for variant in ("A_parent", "N10")
        },
    }


def prepare() -> None:
    authenticate_branch_checkpoints()
    audit = {
        "source_parent": str(BASE.PARENT),
        "source_branch": str(EXP091),
        "branch_checkpoint_sha256": BRANCH_CHECKPOINT_SHA256,
        "biu_correction_scale": BIU_CORRECTION_SCALE,
        "training_performed": False,
        "test_cache_accessed": False,
    }
    atomic_json(METRICS / "lineage_audit.json", audit)
    atomic_json(ROOT / "PREPARATION_COMPLETE.json", audit)


def validate() -> dict[str, Any]:
    if not (ROOT / "PREPARATION_COMPLETE.json").is_file():
        prepare()
    rows, _ = BASE.load_trainval()
    val = rows.loc[rows["split"].eq("val")].reset_index(drop=True)
    metric_rows: list[dict[str, Any]] = []
    PREDICTIONS.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        saved = pd.read_csv(EXP091 / f"predictions/validation_N_seed{seed}.csv")
        if not np.array_equal(val["sequence_id"].astype(str), saved["sequence_id"].astype(str)):
            raise RuntimeError(f"Exp091 validation order drifted for seed {seed}")
        labels = val["label"].to_numpy(dtype=np.int64)
        parent_probability = saved["parent_probability"].to_numpy(dtype=np.float64)
        parent_logits = probability_to_logits(parent_probability)
        correction = saved["correction_logit"].to_numpy(dtype=np.float64)
        probability = sigmoid(scaled_logits(parent_logits, correction))
        metric_rows.append({"seed": seed, "variant": "A_parent", **BASE.fixed_metrics(labels, parent_probability)})
        metric_rows.append({"seed": seed, "variant": "N10", **BASE.fixed_metrics(labels, probability)})
        output = saved[["sequence_id", "track_id", "video_id", "ped_id", "label"]].copy()
        output["parent_probability"] = parent_probability
        output["full_biu_correction_logit"] = correction
        output["scaled_biu_correction_logit"] = BIU_CORRECTION_SCALE * correction
        output["probability"] = probability
        output["prediction"] = (probability >= 0.5).astype(np.int64)
        output.to_csv(PREDICTIONS / f"validation_N10_seed{seed}.csv", index=False)

    results = pd.DataFrame(metric_rows)
    METRICS.mkdir(parents=True, exist_ok=True)
    results.to_csv(METRICS / "validation_per_seed_results.csv", index=False)
    summary = results.groupby("variant")[["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "fp", "fn", "high_conf_fp", "high_conf_fn"]].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary.reset_index().to_csv(METRICS / "validation_summary_results.csv", index=False)
    report = evaluate_gate(results)
    atomic_json(METRICS / "validation_gate_report.json", report)
    if report["passed"]:
        atomic_json(VALIDATION_GATE_MARKER, report)
    else:
        atomic_json(ROOT / "VALIDATION_ONLY_COMPLETE.json", report)
    print(json.dumps(report, indent=2))
    return report


def _load_branch(seed: int, device: torch.device):
    authenticate_branch_checkpoints()
    checkpoint = torch.load(BRANCH_CHECKPOINTS / f"N_seed{seed}.best.pt", map_location=device, weights_only=False)
    model = BASE.DirectFusionModel("C").to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def final_evaluate(device_name: str) -> None:
    if not VALIDATION_GATE_MARKER.is_file():
        raise RuntimeError("Final test is locked because the validation gate has not passed")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    rows, arrays = BASE.load_test()
    labels = rows["label"].to_numpy(dtype=np.int64)
    metric_rows: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        parent = BASE.load_parent_model(seed, device)
        parent_logits, parent_anchor = BASE.extract_parent_outputs(parent, arrays["bbox"], arrays["ego"], device)
        parent_logits = parent_logits.reshape(-1)
        parent_probability = sigmoid(parent_logits)
        saved_parent = pd.read_csv(BASE.PARENT_PREDICTIONS / f"test_A_seed{seed}.csv")
        if not np.array_equal(rows["sequence_id"].astype(str), saved_parent["sequence_id"].astype(str)):
            raise RuntimeError(f"Final parent order drifted for seed {seed}")
        if float(np.max(np.abs(parent_probability - saved_parent["probability"].to_numpy()))) > 1e-6:
            raise RuntimeError(f"Final parent probability reproduction failed for seed {seed}")
        branch = _load_branch(seed, device)
        values = {
            "parent_logits": parent_logits.reshape(-1, 1).astype(np.float32),
            "parent_anchor": parent_anchor.astype(np.float32),
            "extra": np.asarray(arrays["biu_union"], dtype=np.float32),
            "labels": labels.astype(np.float32),
        }
        _, _, correction = BASE.evaluate_branch(branch, values, device)
        probability = sigmoid(scaled_logits(parent_logits, correction))
        metric_rows.append({"seed": seed, "variant": "A_parent", **BASE.fixed_metrics(labels, parent_probability)})
        metric_rows.append({"seed": seed, "variant": "N10", **BASE.fixed_metrics(labels, probability)})
        before = parent_probability >= 0.5
        after = probability >= 0.5
        transition_rows.append(
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
        output = rows[["sequence_id", "track_id", "video_id", "ped_id", "label"]].copy()
        output["parent_probability"] = parent_probability
        output["full_biu_correction_logit"] = correction
        output["scaled_biu_correction_logit"] = BIU_CORRECTION_SCALE * correction
        output["probability"] = probability
        output["prediction"] = after.astype(np.int64)
        PREDICTIONS.mkdir(parents=True, exist_ok=True)
        output.to_csv(PREDICTIONS / f"test_N10_seed{seed}.csv", index=False)

    results = pd.DataFrame(metric_rows)
    METRICS.mkdir(parents=True, exist_ok=True)
    results.to_csv(METRICS / "fixed_0p5_results.csv", index=False)
    results.to_csv(METRICS / "per_seed_results.csv", index=False)
    summary = results.groupby("variant")[["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "fp", "fn", "high_conf_fp", "high_conf_fn"]].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary.reset_index().to_csv(METRICS / "summary_results.csv", index=False)
    paired = []
    for seed in SEEDS:
        indexed = results.loc[results["seed"].eq(seed)].set_index("variant")
        paired.append(
            {
                "seed": seed,
                **{f"delta_{metric}": float(indexed.loc["N10", metric] - indexed.loc["A_parent", metric]) for metric in ("accuracy", "f1", "roc_auc", "pr_auc")},
            }
        )
    pd.DataFrame(paired).to_csv(METRICS / "paired_deltas.csv", index=False)
    pd.DataFrame(transition_rows).to_csv(METRICS / "error_transitions_fixed_0p5.csv", index=False)
    atomic_json(
        ROOT / "EXPERIMENT_COMPLETE.json",
        {
            "scale": BIU_CORRECTION_SCALE,
            "seeds": list(SEEDS),
            "test_windows": int(len(rows)),
            "training_performed": False,
            "validation_gate_passed_before_test": True,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "validate", "evaluate"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "validate":
        validate()
    else:
        final_evaluate(args.device)


if __name__ == "__main__":
    main()
