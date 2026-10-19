#!/usr/bin/env python3
"""Saved-prediction-only evidence analysis for completed Exp089."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score


SCRIPT_DIR = Path(__file__).resolve().parent
EXP_DIR = SCRIPT_DIR.parent
PREDICTIONS = EXP_DIR / "predictions"
METRICS = EXP_DIR / "metrics"
VARIANTS = tuple("ABCDE")
SEEDS = (42, 43, 44)
COMPARISONS = (("A", "C"), ("A", "D"), ("A", "E"), ("B", "C"), ("B", "D"), ("B", "E"))


def metric_values(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float | int]:
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    p = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if y.shape != p.shape or not np.isfinite(p).all():
        raise ValueError("Labels and probabilities must be finite and aligned")
    prediction = (p >= 0.5).astype(np.int64)
    return {
        "accuracy": float(accuracy_score(y, prediction)),
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "fp": int(np.sum((y == 0) & (prediction == 1))),
        "fn": int(np.sum((y == 1) & (prediction == 0))),
    }


def classification_values_at_threshold(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, float | int]:
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    p = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if y.shape != p.shape or not np.isfinite(p).all():
        raise ValueError("Labels and probabilities must be finite and aligned")
    prediction = (p >= threshold).astype(np.int64)
    return {
        "accuracy": float(accuracy_score(y, prediction)),
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "fp": int(np.sum((y == 0) & (prediction == 1))),
        "fn": int(np.sum((y == 1) & (prediction == 0))),
    }


def select_validation_f1_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    """Select an attainable threshold using validation labels only.

    Ties prefer the threshold closest to 0.5, then the higher threshold. This
    diagnostic is deliberately separate from the frozen fixed-0.5 benchmark.
    """
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    p = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if y.shape != p.shape or not np.isfinite(p).all():
        raise ValueError("Labels and probabilities must be finite and aligned")
    candidates = np.unique(np.concatenate(([0.0], p, [1.0])))
    scored = (
        (
            float(f1_score(y, p >= threshold, zero_division=0)),
            -abs(float(threshold) - 0.5),
            float(threshold),
        )
        for threshold in candidates
    )
    return max(scored)[2]


def cluster_bootstrap_delta(
    labels: np.ndarray,
    left_probability: np.ndarray,
    right_probability: np.ndarray,
    clusters: np.ndarray,
    *,
    repeats: int,
    seed: int,
) -> dict[str, float | int]:
    y = np.asarray(labels, dtype=np.int64)
    left = np.asarray(left_probability, dtype=np.float64)
    right = np.asarray(right_probability, dtype=np.float64)
    group = np.asarray(clusters).astype(str)
    if not (len(y) == len(left) == len(right) == len(group)):
        raise ValueError("Bootstrap inputs are not aligned")
    unique = np.unique(group)
    indices = {name: np.flatnonzero(group == name) for name in unique}
    rng = np.random.default_rng(seed)
    observed: dict[str, list[float]] = {name: [] for name in ("accuracy", "f1", "roc_auc", "pr_auc")}
    for _ in range(repeats):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        take = np.concatenate([indices[name] for name in sampled])
        if np.unique(y[take]).size != 2:
            continue
        left_metrics = metric_values(y[take], left[take])
        right_metrics = metric_values(y[take], right[take])
        for metric in observed:
            observed[metric].append(float(right_metrics[metric]) - float(left_metrics[metric]))
    if not observed["f1"]:
        raise ValueError("No valid clustered bootstrap repetitions")
    result: dict[str, float | int] = {
        "bootstrap_seed": int(seed),
        "bootstrap_repeats_requested": int(repeats),
        "valid_bootstrap_repeats": int(len(observed["f1"])),
        "cluster_count": int(len(unique)),
    }
    for metric, values in observed.items():
        result[f"{metric}_delta_low"] = float(np.quantile(values, 0.025))
        result[f"{metric}_delta_high"] = float(np.quantile(values, 0.975))
    return result


def _load_predictions() -> tuple[pd.DataFrame, dict[tuple[str, int], np.ndarray]]:
    reference: pd.DataFrame | None = None
    probabilities: dict[tuple[str, int], np.ndarray] = {}
    identity = ["sequence_id", "video_id", "ped_id", "label"]
    for variant in VARIANTS:
        for seed in SEEDS:
            path = PREDICTIONS / f"test_{variant}_seed{seed}.csv"
            if not path.is_file():
                raise FileNotFoundError(f"Missing frozen prediction artifact: {path}")
            frame = pd.read_csv(path)
            if reference is None:
                reference = frame[identity].copy()
                if reference["sequence_id"].duplicated().any():
                    raise ValueError("Prediction population contains duplicate sequence IDs")
            elif not frame[identity].equals(reference):
                raise ValueError(f"Prediction cohort drift in {path}")
            probabilities[(variant, seed)] = frame["probability"].to_numpy(dtype=np.float64)
    assert reference is not None
    return reference, probabilities


def _load_validation_predictions() -> tuple[pd.DataFrame, dict[tuple[str, int], np.ndarray]]:
    reference: pd.DataFrame | None = None
    probabilities: dict[tuple[str, int], np.ndarray] = {}
    identity = ["sequence_id", "label"]
    for variant in VARIANTS:
        for seed in SEEDS:
            path = PREDICTIONS / f"validation_{variant}_seed{seed}.csv"
            if not path.is_file():
                raise FileNotFoundError(f"Missing frozen validation prediction artifact: {path}")
            frame = pd.read_csv(path)
            if reference is None:
                reference = frame[identity].copy()
                if reference["sequence_id"].duplicated().any():
                    raise ValueError("Validation population contains duplicate sequence IDs")
            elif not frame[identity].equals(reference):
                raise ValueError(f"Validation prediction cohort drift in {path}")
            probabilities[(variant, seed)] = frame["probability"].to_numpy(dtype=np.float64)
    assert reference is not None
    return reference, probabilities


def build_analysis(*, bootstrap_repeats: int = 2000) -> dict[str, pd.DataFrame]:
    population, probabilities = _load_predictions()
    validation_population, validation_probabilities = _load_validation_predictions()
    labels = population["label"].to_numpy(dtype=np.int64)
    validation_labels = validation_population["label"].to_numpy(dtype=np.int64)
    clusters = (population["video_id"].astype(str) + ":" + population["ped_id"].astype(str)).to_numpy()
    per_seed_rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        for seed in SEEDS:
            per_seed_rows.append(
                {"variant": variant, "seed": seed, **metric_values(labels, probabilities[(variant, seed)])}
            )
    per_seed = pd.DataFrame(per_seed_rows)

    delta_rows: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []
    for left, right in COMPARISONS:
        for seed in SEEDS:
            left_metrics = per_seed.query("variant == @left and seed == @seed").iloc[0]
            right_metrics = per_seed.query("variant == @right and seed == @seed").iloc[0]
            delta_rows.append(
                {
                    "comparison": f"{left}->{right}",
                    "seed": seed,
                    **{
                        f"delta_{metric}": float(right_metrics[metric] - left_metrics[metric])
                        for metric in ("accuracy", "f1", "roc_auc", "pr_auc", "fp", "fn")
                    },
                }
            )
            left_prediction = probabilities[(left, seed)] >= 0.5
            right_prediction = probabilities[(right, seed)] >= 0.5
            left_correct = left_prediction == labels
            right_correct = right_prediction == labels
            transition_rows.append(
                {
                    "comparison": f"{left}->{right}",
                    "seed": seed,
                    "left_wrong_right_correct": int(np.sum(~left_correct & right_correct)),
                    "left_correct_right_wrong": int(np.sum(left_correct & ~right_correct)),
                    "fp_corrected_to_tn": int(np.sum((labels == 0) & left_prediction & ~right_prediction)),
                    "fn_corrected_to_tp": int(np.sum((labels == 1) & ~left_prediction & right_prediction)),
                    "tn_harmed_to_fp": int(np.sum((labels == 0) & ~left_prediction & right_prediction)),
                    "tp_harmed_to_fn": int(np.sum((labels == 1) & left_prediction & ~right_prediction)),
                }
            )
    paired_deltas = pd.DataFrame(delta_rows)
    transitions = pd.DataFrame(transition_rows)
    summary_rows: list[dict[str, Any]] = []
    for comparison, group in paired_deltas.groupby("comparison", sort=False):
        row: dict[str, Any] = {"comparison": comparison}
        for metric in ("accuracy", "f1", "roc_auc", "pr_auc", "fp", "fn"):
            values = group[f"delta_{metric}"]
            row[f"{metric}_delta_mean"] = float(values.mean())
            row[f"{metric}_delta_std"] = float(values.std(ddof=1))
            if metric in {"accuracy", "f1", "roc_auc", "pr_auc"}:
                row[f"{metric}_favorable_seeds"] = int((values > 0).sum())
        summary_rows.append(row)
    paired_summary = pd.DataFrame(summary_rows)

    ensemble_probability = {
        variant: np.mean([probabilities[(variant, seed)] for seed in SEEDS], axis=0)
        for variant in VARIANTS
    }
    ensemble_metrics = pd.DataFrame(
        [{"variant": variant, **metric_values(labels, values)} for variant, values in ensemble_probability.items()]
    )
    bootstrap_rows: list[dict[str, Any]] = []
    for index, (left, right) in enumerate(COMPARISONS):
        left_metrics = metric_values(labels, ensemble_probability[left])
        right_metrics = metric_values(labels, ensemble_probability[right])
        intervals = cluster_bootstrap_delta(
            labels,
            ensemble_probability[left],
            ensemble_probability[right],
            clusters,
            repeats=bootstrap_repeats,
            seed=20260901 + index,
        )
        bootstrap_rows.append(
            {
                "comparison": f"{left}->{right}",
                **{
                    f"ensemble_{metric}_delta": float(right_metrics[metric]) - float(left_metrics[metric])
                    for metric in ("accuracy", "f1", "roc_auc", "pr_auc", "fp", "fn")
                },
                **intervals,
            }
        )
    ensemble_bootstrap = pd.DataFrame(bootstrap_rows)

    threshold_rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        for seed in SEEDS:
            threshold = select_validation_f1_threshold(
                validation_labels, validation_probabilities[(variant, seed)]
            )
            validation_selected = classification_values_at_threshold(
                validation_labels, validation_probabilities[(variant, seed)], threshold
            )
            test_selected = classification_values_at_threshold(
                labels, probabilities[(variant, seed)], threshold
            )
            test_fixed = classification_values_at_threshold(
                labels, probabilities[(variant, seed)], 0.5
            )
            threshold_rows.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "validation_f1_threshold": threshold,
                    "validation_f1_at_selected_threshold": validation_selected["f1"],
                    "test_f1_at_validation_threshold": test_selected["f1"],
                    "test_f1_at_fixed_0p5": test_fixed["f1"],
                    "test_f1_delta_vs_fixed_0p5": float(test_selected["f1"])
                    - float(test_fixed["f1"]),
                    "test_accuracy_at_validation_threshold": test_selected["accuracy"],
                    "test_fp_at_validation_threshold": test_selected["fp"],
                    "test_fn_at_validation_threshold": test_selected["fn"],
                }
            )
    validation_threshold_diagnostic = pd.DataFrame(threshold_rows)
    validation_threshold_summary = (
        validation_threshold_diagnostic.groupby("variant", sort=False)
        .agg(
            threshold_mean=("validation_f1_threshold", "mean"),
            threshold_std=("validation_f1_threshold", "std"),
            test_f1_validation_threshold_mean=("test_f1_at_validation_threshold", "mean"),
            test_f1_fixed_0p5_mean=("test_f1_at_fixed_0p5", "mean"),
            test_f1_delta_mean=("test_f1_delta_vs_fixed_0p5", "mean"),
        )
        .reset_index()
    )
    return {
        "per_seed": per_seed,
        "paired_deltas": paired_deltas,
        "paired_summary": paired_summary,
        "transitions": transitions,
        "ensemble_metrics": ensemble_metrics,
        "ensemble_bootstrap": ensemble_bootstrap,
        "validation_threshold_diagnostic": validation_threshold_diagnostic,
        "validation_threshold_summary": validation_threshold_summary,
    }


def main() -> None:
    outputs = build_analysis()
    METRICS.mkdir(parents=True, exist_ok=True)
    filenames = {
        "per_seed": "verified_per_seed_results.csv",
        "paired_deltas": "complete_paired_deltas.csv",
        "paired_summary": "complete_paired_summary.csv",
        "transitions": "complete_error_transitions_fixed_0p5.csv",
        "ensemble_metrics": "ensemble_metrics.csv",
        "ensemble_bootstrap": "ensemble_cluster_bootstrap.csv",
        "validation_threshold_diagnostic": "validation_threshold_diagnostic.csv",
        "validation_threshold_summary": "validation_threshold_summary.csv",
    }
    for name, frame in outputs.items():
        frame.to_csv(METRICS / filenames[name], index=False)
    print(outputs["paired_summary"].to_string(index=False))
    print(outputs["ensemble_bootstrap"].to_string(index=False))


if __name__ == "__main__":
    main()
