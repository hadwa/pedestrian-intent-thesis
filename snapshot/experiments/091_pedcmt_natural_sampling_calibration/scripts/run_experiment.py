#!/usr/bin/env python3
"""Validation-only natural-sampling and calibration audit for Exp090."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import StratifiedGroupKFold


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
REPO_ROOT = ROOT.parents[1]
EXP090 = REPO_ROOT / "experiments/090_pedcmt_direct_biu_fusion"
EXP090_RUNNER = EXP090 / "scripts/run_experiment.py"
CHECKPOINTS = ROOT / "checkpoints"
LOGS = ROOT / "logs"
METRICS = ROOT / "metrics"
PREDICTIONS = ROOT / "predictions"
SEEDS = (42, 43, 44)


def _load_exp090_runner():
    spec = importlib.util.spec_from_file_location("exp090_runner_for_exp091", EXP090_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {EXP090_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = _load_exp090_runner()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def natural_training_indices(rows: pd.DataFrame) -> np.ndarray:
    """Return every training window exactly once in its original order."""
    indices = np.flatnonzero(rows["split"].eq("train").to_numpy()).astype(np.int64)
    if len(indices) != len(np.unique(indices)):
        raise RuntimeError("Natural training indices contain duplicates")
    return indices


def _clipped_logits(probabilities: np.ndarray) -> np.ndarray:
    probability = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-8, 1.0 - 1e-8)
    return np.log(probability) - np.log1p(-probability)


def apply_logit_intercept(probabilities: np.ndarray, intercept: float) -> np.ndarray:
    shifted = _clipped_logits(probabilities) + float(intercept)
    return 1.0 / (1.0 + np.exp(-shifted))


def fit_logit_intercept(labels: np.ndarray, probabilities: np.ndarray) -> float:
    """Fit one calibration-in-the-large offset with slope fixed to one."""
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    logits = _clipped_logits(probabilities).reshape(-1)
    if len(y) != len(logits) or len(y) == 0:
        raise ValueError("Labels and probabilities must be non-empty and aligned")
    if not np.isin(y, [0.0, 1.0]).all() or len(np.unique(y)) != 2:
        raise ValueError("Intercept calibration requires both binary classes")

    low, high = -40.0, 40.0
    target = float(y.mean())
    for _ in range(120):
        middle = (low + high) / 2.0
        mean_probability = float(np.mean(1.0 / (1.0 + np.exp(-(logits + middle)))))
        if mean_probability < target:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


def group_crossfit_intercept(
    labels: np.ndarray,
    probabilities: np.ndarray,
    groups: np.ndarray,
    *,
    n_splits: int = 5,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Cross-fit intercept calibration while holding out complete tracks."""
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    probability = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    group = np.asarray(groups).reshape(-1)
    if not (len(y) == len(probability) == len(group)):
        raise ValueError("Cross-fit inputs must align")
    unique_groups = np.unique(group)
    splits = min(int(n_splits), len(unique_groups))
    if splits < 2:
        raise ValueError("Cross-fitting requires at least two groups")

    calibrated = np.full(len(y), np.nan, dtype=np.float64)
    audit_rows: list[dict[str, Any]] = []
    splitter = StratifiedGroupKFold(n_splits=splits, shuffle=True, random_state=42)
    for fold, (fit_indices, held_indices) in enumerate(
        splitter.split(probability.reshape(-1, 1), y, groups=group), start=1
    ):
        intercept = fit_logit_intercept(y[fit_indices], probability[fit_indices])
        calibrated[held_indices] = apply_logit_intercept(probability[held_indices], intercept)
        fit_groups = frozenset(group[fit_indices].astype(str))
        held_groups = frozenset(group[held_indices].astype(str))
        if not fit_groups.isdisjoint(held_groups):
            raise RuntimeError("Calibration group leakage detected")
        for sample_index in held_indices:
            audit_rows.append(
                {
                    "sample_index": int(sample_index),
                    "fold": fold,
                    "intercept": intercept,
                    "fit_groups": fit_groups,
                    "held_out_groups": held_groups,
                    "fit_positive_fraction": float(y[fit_indices].mean()),
                    "held_positive_fraction": float(y[held_indices].mean()),
                }
            )
    if not np.isfinite(calibrated).all():
        raise RuntimeError("Cross-fitting did not produce every held-out probability")
    return calibrated, pd.DataFrame(audit_rows).sort_values("sample_index").reset_index(drop=True)


def _ece(labels: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    probability = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for index in range(bins):
        if index == bins - 1:
            selected = (probability >= boundaries[index]) & (probability <= boundaries[index + 1])
        else:
            selected = (probability >= boundaries[index]) & (probability < boundaries[index + 1])
        if selected.any():
            total += float(selected.mean()) * abs(float(probability[selected].mean() - y[selected].mean()))
    return total


def validation_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float | int]:
    values = BASE.fixed_metrics(labels, probabilities)
    probability = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-8, 1.0 - 1e-8)
    values.update(
        {
            "brier": float(brier_score_loss(labels, probability)),
            "log_loss": float(log_loss(labels, probability, labels=[0, 1])),
            "ece_10bin": _ece(labels, probability),
        }
    )
    return values


def calibration_line(labels: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    model.fit(_clipped_logits(probabilities).reshape(-1, 1), np.asarray(labels, dtype=np.int64))
    return float(model.intercept_[0]), float(model.coef_[0, 0])


def prepare() -> None:
    BASE.authenticate_parent_checkpoints()
    rows, arrays = BASE.load_trainval()
    for seed in SEEDS:
        BASE._prepared_parent(seed)
        source_prediction = EXP090 / f"predictions/validation_C_seed{seed}.csv"
        if not source_prediction.is_file():
            raise FileNotFoundError(f"Missing Exp090 validation prediction: {source_prediction}")
    indices = natural_training_indices(rows)
    labels = rows.iloc[indices]["label"].to_numpy(dtype=np.int64)
    if len(indices) != 1552 or dict(zip(*np.unique(labels, return_counts=True))) != {0: 553, 1: 999}:
        raise RuntimeError("Natural JAAD training population drifted")
    audit = {
        "source_experiment": str(EXP090),
        "parent_experiment": str(BASE.PARENT),
        "variant": "C BIU-v1/v2 union direct residual",
        "training_sampling": "natural; every train window exactly once",
        "train_windows": 1552,
        "train_class_counts": {"not_cross": 553, "cross": 999},
        "validation_windows": 274,
        "validation_tracks": int(rows.loc[rows["split"].eq("val"), "track_id"].nunique()),
        "biu_union_shape": list(arrays["biu_union"].shape),
        "test_cache_accessed": False,
    }
    atomic_json(METRICS / "source_and_sampling_audit.json", audit)
    atomic_json(ROOT / "PREPARATION_COMPLETE.json", audit)


def _values(indices: np.ndarray, logits: np.ndarray, anchor: np.ndarray, extra: np.ndarray, rows: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "parent_logits": np.asarray(logits[indices], dtype=np.float32),
        "parent_anchor": np.asarray(anchor[indices], dtype=np.float32),
        "extra": np.asarray(extra[indices], dtype=np.float32),
        "labels": rows.iloc[indices]["label"].to_numpy(dtype=np.float32),
    }


def train_one(seed: int, device_name: str, *, smoke: bool = False) -> dict[str, Any]:
    if not (ROOT / "PREPARATION_COMPLETE.json").is_file():
        prepare()
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    rows, arrays = BASE.load_trainval()
    logits, anchor = BASE._prepared_parent(seed)
    train_indices = natural_training_indices(rows)
    val_indices = np.flatnonzero(rows["split"].eq("val").to_numpy())
    stem = f"{'smoke_' if smoke else ''}N_seed{seed}"
    result = BASE.fit_branch(
        "C",
        _values(train_indices, logits, anchor, arrays["biu_union"], rows),
        _values(val_indices, logits, anchor, arrays["biu_union"], rows),
        seed=seed,
        device=device,
        checkpoint_path=CHECKPOINTS / f"{stem}.best.pt",
        history_path=LOGS / f"{stem}.history.csv",
        parent_sha256=BASE.PARENT_CHECKPOINT_SHA256[seed],
        max_epochs=1 if smoke else BASE.MAX_EPOCHS,
        patience=1 if smoke else BASE.PATIENCE,
    )
    output = rows.iloc[val_indices][["sequence_id", "track_id", "video_id", "ped_id", "label"]].copy()
    output["parent_probability"] = torch.sigmoid(torch.from_numpy(np.asarray(logits[val_indices]))).numpy().reshape(-1)
    output["probability"] = result.pop("probability")
    output["prediction"] = (output["probability"] >= 0.5).astype(np.int64)
    output["correction_logit"] = result.pop("correction")
    PREDICTIONS.mkdir(parents=True, exist_ok=True)
    output.to_csv(PREDICTIONS / f"validation_{stem}.csv", index=False)
    result["training_sampling"] = "natural_no_resampling"
    atomic_json(LOGS / f"{stem}.complete.json", result)
    return result


def train_all(device_name: str) -> None:
    if not (ROOT / "PREPARATION_COMPLETE.json").is_file():
        prepare()
    for seed in SEEDS:
        if (CHECKPOINTS / f"N_seed{seed}.best.pt").is_file() and (LOGS / f"N_seed{seed}.complete.json").is_file():
            print(f"Skipping complete N seed {seed}", flush=True)
            continue
        print(json.dumps(train_one(seed, device_name), indent=2), flush=True)
    missing = [seed for seed in SEEDS if not (CHECKPOINTS / f"N_seed{seed}.best.pt").is_file()]
    if missing:
        raise RuntimeError(f"Missing natural-sampling checkpoints: {missing}")
    atomic_json(ROOT / "TRAINING_COMPLETE.json", {"variant": "N", "seeds": list(SEEDS)})


def _saved_probabilities(seed: int) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    rows, _ = BASE.load_trainval()
    val = rows.loc[rows["split"].eq("val")].reset_index(drop=True)
    logits, _ = BASE._prepared_parent(seed)
    val_indices = np.flatnonzero(rows["split"].eq("val").to_numpy())
    probabilities = {
        "A_parent": torch.sigmoid(torch.from_numpy(np.asarray(logits[val_indices]))).numpy().reshape(-1),
        "C_balanced": pd.read_csv(EXP090 / f"predictions/validation_C_seed{seed}.csv")["probability"].to_numpy(dtype=np.float64),
        "N_natural": pd.read_csv(PREDICTIONS / f"validation_N_seed{seed}.csv")["probability"].to_numpy(dtype=np.float64),
    }
    if any(len(value) != len(val) for value in probabilities.values()):
        raise RuntimeError(f"Validation probability alignment failed for seed {seed}")
    return val, probabilities


def analyze() -> dict[str, Any]:
    if not (ROOT / "TRAINING_COMPLETE.json").is_file():
        raise RuntimeError("Complete natural-sampling training before analysis")
    metric_rows: list[dict[str, Any]] = []
    intercept_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    reliability_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        val, probabilities = _saved_probabilities(seed)
        labels = val["label"].to_numpy(dtype=np.int64)
        groups = val["track_id"].astype(str).to_numpy()
        for variant, raw_probability in probabilities.items():
            raw_intercept, raw_slope = calibration_line(labels, raw_probability)
            metric_rows.append(
                {
                    "seed": seed,
                    "variant": variant,
                    "policy": "fixed_0p5_raw",
                    "calibration_intercept": raw_intercept,
                    "calibration_slope": raw_slope,
                    **validation_metrics(labels, raw_probability),
                }
            )
            calibrated, fold_audit = group_crossfit_intercept(labels, raw_probability, groups)
            calibrated_intercept, calibrated_slope = calibration_line(labels, calibrated)
            metric_rows.append(
                {
                    "seed": seed,
                    "variant": variant,
                    "policy": "fixed_0p5_group_crossfit_intercept",
                    "calibration_intercept": calibrated_intercept,
                    "calibration_slope": calibrated_slope,
                    **validation_metrics(labels, calibrated),
                }
            )
            for fold, fold_rows in fold_audit.groupby("fold"):
                intercept_rows.append(
                    {
                        "seed": seed,
                        "variant": variant,
                        "fold": int(fold),
                        "intercept": float(fold_rows["intercept"].iloc[0]),
                        "held_out_samples": int(len(fold_rows)),
                        "held_out_tracks": int(len(fold_rows["held_out_groups"].iloc[0])),
                    }
                )
            frame = val[["sequence_id", "track_id", "video_id", "ped_id", "label"]].copy()
            frame["seed"] = seed
            frame["variant"] = variant
            frame["raw_probability"] = raw_probability
            frame["calibrated_probability"] = calibrated
            prediction_frames.append(frame)
            for policy, values in (("raw", raw_probability), ("group_crossfit_intercept", calibrated)):
                fraction, mean_prediction = calibration_curve(labels, values, n_bins=10, strategy="uniform")
                for bin_index, (observed, predicted) in enumerate(zip(fraction, mean_prediction)):
                    reliability_rows.append(
                        {
                            "seed": seed,
                            "variant": variant,
                            "policy": policy,
                            "bin": bin_index,
                            "mean_probability": float(predicted),
                            "observed_positive_fraction": float(observed),
                        }
                    )

    metrics = pd.DataFrame(metric_rows)
    METRICS.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(METRICS / "validation_per_seed_results.csv", index=False)
    summary = metrics.groupby(["variant", "policy"])[
        ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "brier", "log_loss", "ece_10bin", "fp", "fn", "high_conf_fp", "high_conf_fn"]
    ].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary.reset_index().to_csv(METRICS / "validation_summary_results.csv", index=False)
    pd.DataFrame(intercept_rows).to_csv(METRICS / "crossfit_intercepts.csv", index=False)
    pd.DataFrame(reliability_rows).to_csv(METRICS / "reliability_bins.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(PREDICTIONS / "validation_calibration_predictions.csv", index=False)

    raw_means = metrics.loc[metrics["policy"].eq("fixed_0p5_raw")].groupby("variant").mean(numeric_only=True)
    criteria = {
        "retains_pr_auc_gain_over_parent": float(raw_means.loc["N_natural", "pr_auc"] - raw_means.loc["A_parent", "pr_auc"]) >= 0.005,
        "f1_within_0p005_of_parent": float(raw_means.loc["N_natural", "f1"]) >= float(raw_means.loc["A_parent", "f1"]) - 0.005,
        "accuracy_within_0p005_of_parent": float(raw_means.loc["N_natural", "accuracy"]) >= float(raw_means.loc["A_parent", "accuracy"]) - 0.005,
        "no_high_confidence_error_increase": float(raw_means.loc["N_natural", "high_conf_fp"] + raw_means.loc["N_natural", "high_conf_fn"]) <= float(raw_means.loc["A_parent", "high_conf_fp"] + raw_means.loc["A_parent", "high_conf_fn"]),
    }
    report = {
        "passed": bool(all(criteria.values())),
        "criteria": criteria,
        "scope": "validation-only diagnostic; final JAAD cohort never loaded",
        "test_cache_accessed": False,
    }
    atomic_json(METRICS / "validation_gate_report.json", report)
    atomic_json(ROOT / "VALIDATION_ONLY_COMPLETE.json", report)
    print(json.dumps(report, indent=2))
    return report


def smoke(device_name: str) -> None:
    result = train_one(42, device_name, smoke=True)
    atomic_json(LOGS / "SMOKE_COMPLETE.json", {"device": device_name, "result": result})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "smoke", "train", "train-all", "analyze"))
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "smoke":
        smoke(args.device)
    elif args.command == "train":
        if args.seed is None:
            raise SystemExit("train requires --seed")
        print(json.dumps(train_one(args.seed, args.device), indent=2))
    elif args.command == "train-all":
        train_all(args.device)
    else:
        analyze()


if __name__ == "__main__":
    main()
