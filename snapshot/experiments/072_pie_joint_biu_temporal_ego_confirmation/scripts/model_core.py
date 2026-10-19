"""Frozen estimator, metrics, and paired analyses for Exp072."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


THRESHOLD = 0.5
SEEDS = (42, 43, 44)


def build_candidate(seed: int, labels: np.ndarray) -> Pipeline:
    if int(seed) not in SEEDS:
        raise ValueError("Exp072 seed must be 42, 43, or 44")
    labels = np.asarray(labels, dtype=int)
    if labels.ndim != 1 or not len(labels) or not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("training labels must be a nonempty binary vector")
    positives = int((labels == 1).sum())
    negatives = int((labels == 0).sum())
    if positives == 0 or negatives == 0:
        raise ValueError("training labels require both classes")
    estimator = XGBClassifier(
        n_estimators=250,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=int(seed),
        n_jobs=4,
        scale_pos_weight=float(negatives / positives),
    )
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", estimator)])


def _validate_vectors(labels: np.ndarray, probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if labels.ndim != 1 or probabilities.ndim != 1 or not len(labels) or len(labels) != len(probabilities):
        raise ValueError("labels and probabilities must be aligned nonempty vectors")
    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("labels must be binary")
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("probabilities must be finite in [0,1]")
    return labels, probabilities


def calibration_error(labels: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    labels, probabilities = _validate_vectors(labels, probabilities)
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    assignments = np.minimum(np.digitize(probabilities, edges[1:-1], right=True), int(bins) - 1)
    result = 0.0
    for index in range(int(bins)):
        mask = assignments == index
        if mask.any():
            result += float(mask.mean()) * abs(float(labels[mask].mean()) - float(probabilities[mask].mean()))
    return float(result)


def compute_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float | int]:
    labels, probabilities = _validate_vectors(labels, probabilities)
    predictions = (probabilities >= THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    negatives = int((labels == 0).sum())
    positives = int((labels == 1).sum())
    high_fp = int(np.sum((labels == 0) & (probabilities >= 0.9)))
    high_fn = int(np.sum((labels == 1) & (probabilities <= 0.1)))
    return {
        "n": int(len(labels)),
        "positive_count": positives,
        "negative_count": negatives,
        "threshold": THRESHOLD,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probabilities)) if len(np.unique(labels)) == 2 else float("nan"),
        "pr_auc": float(average_precision_score(labels, probabilities)),
        "fpr": float(fp / negatives) if negatives else float("nan"),
        "fnr": float(fn / positives) if positives else float("nan"),
        "brier": float(brier_score_loss(labels, probabilities)),
        "log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
        "ece_10bin": calibration_error(labels, probabilities, bins=10),
        "fp_ge_0p9_count": high_fp,
        "fn_le_0p1_count": high_fn,
        "fp_ge_0p9_rate": float(high_fp / negatives) if negatives else float("nan"),
        "fn_le_0p1_rate": float(high_fn / positives) if positives else float("nan"),
    }


def validation_decision(per_seed_deltas: pd.DataFrame) -> dict[str, Any]:
    required = {"seed", "delta_pr_auc", "delta_f1", "delta_roc_auc"}
    if required.difference(per_seed_deltas.columns):
        raise ValueError("validation deltas are missing required columns")
    frame = per_seed_deltas.loc[:, sorted(required)].copy()
    if set(frame["seed"].astype(int)) != set(SEEDS) or frame["seed"].duplicated().any() or len(frame) != 3:
        raise ValueError("validation gate requires exactly seeds 42,43,44")
    values = frame.loc[:, ["delta_pr_auc", "delta_f1", "delta_roc_auc"]].to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("validation deltas must be finite")
    mean_pr = float(frame["delta_pr_auc"].mean())
    mean_f1 = float(frame["delta_f1"].mean())
    mean_roc = float(frame["delta_roc_auc"].mean())
    checks = {
        "mean_pr_auc_delta_ge_0p005": mean_pr >= 0.005,
        "mean_f1_delta_ge_0p005": mean_f1 >= 0.005,
        "mean_roc_auc_delta_ge_minus_0p002": mean_roc >= -0.002,
        "all_seed_pr_auc_delta_ge_minus_0p005": bool((frame["delta_pr_auc"] >= -0.005).all()),
        "all_seed_f1_delta_ge_minus_0p005": bool((frame["delta_f1"] >= -0.005).all()),
    }
    return {
        "status": "advance" if all(checks.values()) else "no_advance",
        "checks": checks,
        "mean_deltas": {"pr_auc": mean_pr, "f1": mean_f1, "roc_auc": mean_roc},
        "minimum_seed_deltas": {
            "pr_auc": float(frame["delta_pr_auc"].min()),
            "f1": float(frame["delta_f1"].min()),
        },
    }


def paired_deltas(aligned: pd.DataFrame, scope: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"seed", "label", "probability", "anchor_probability"}
    if required.difference(aligned.columns) or aligned.empty:
        raise ValueError("aligned probabilities are incomplete")
    rows: list[dict[str, Any]] = []
    deltas: list[dict[str, Any]] = []
    for seed, group in aligned.groupby("seed", sort=True):
        labels = group["label"].to_numpy(int)
        candidate = compute_metrics(labels, group["probability"].to_numpy(float))
        anchor = compute_metrics(labels, group["anchor_probability"].to_numpy(float))
        rows.extend(
            [
                {"scope": scope, "seed": int(seed), "model": "candidate", **candidate},
                {"scope": scope, "seed": int(seed), "model": "anchor", **anchor},
            ]
        )
        delta = {"scope": scope, "seed": int(seed)}
        for metric in (
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "pr_auc",
            "fpr",
            "fnr",
            "brier",
            "log_loss",
            "ece_10bin",
            "fp_ge_0p9_count",
            "fn_le_0p1_count",
            "fp_ge_0p9_rate",
            "fn_le_0p1_rate",
        ):
            delta[f"delta_{metric}"] = float(candidate[metric] - anchor[metric])
        deltas.append(delta)
    if set(int(value) for value in aligned["seed"].unique()) != set(SEEDS):
        # Unit tests may intentionally exercise one seed; real phase checks enforce all three.
        if aligned["seed"].nunique() != 1:
            raise ValueError("paired metrics require one seed or the exact Exp072 seed set")
    return pd.DataFrame(rows), pd.DataFrame(deltas)


def _error_type(label: int, prediction: int) -> str:
    if int(label) == int(prediction):
        return "correct"
    return "fp" if int(label) == 0 else "fn"


def error_transitions(aligned: pd.DataFrame) -> pd.DataFrame:
    required = {
        "seed",
        "sequence_id",
        "set_id",
        "track_id",
        "video_id",
        "label",
        "probability",
        "anchor_probability",
    }
    if required.difference(aligned.columns) or aligned.empty:
        raise ValueError("aligned transition rows are incomplete")
    output = aligned.copy()
    output["candidate_prediction"] = (output["probability"].astype(float) >= THRESHOLD).astype(int)
    output["anchor_prediction"] = (output["anchor_probability"].astype(float) >= THRESHOLD).astype(int)
    candidate_correct = output["candidate_prediction"].eq(output["label"].astype(int))
    anchor_correct = output["anchor_prediction"].eq(output["label"].astype(int))
    output["transition"] = np.select(
        [anchor_correct & candidate_correct, ~anchor_correct & candidate_correct, anchor_correct & ~candidate_correct],
        ["both_correct", "anchor_wrong_candidate_correct", "anchor_correct_candidate_wrong"],
        default="both_wrong",
    )
    output["anchor_error_type"] = [
        _error_type(label, prediction)
        for label, prediction in zip(output["label"], output["anchor_prediction"])
    ]
    output["candidate_error_type"] = [
        _error_type(label, prediction)
        for label, prediction in zip(output["label"], output["candidate_prediction"])
    ]
    return output


def cluster_bootstrap_deltas(
    aligned: pd.DataFrame,
    *,
    metrics: Iterable[str] = ("f1", "pr_auc", "roc_auc"),
    replicates: int = 10000,
    seed: int = 42,
) -> pd.DataFrame:
    if aligned["seed"].nunique() != 1 or "track_id" not in aligned:
        raise ValueError("bootstrap requires one aligned seed and track_id")
    metric_names = tuple(metrics)
    if not metric_names or set(metric_names).difference({"f1", "pr_auc", "roc_auc"}):
        raise ValueError("unsupported bootstrap metric")
    groups = [group.index.to_numpy() for _, group in aligned.groupby("track_id", sort=True)]
    if not groups:
        raise ValueError("bootstrap requires at least one track")
    rng = np.random.default_rng(int(seed))
    samples = {name: [] for name in metric_names}
    for _ in range(int(replicates)):
        selected = rng.integers(0, len(groups), size=len(groups))
        indices = np.concatenate([groups[index] for index in selected])
        sample = aligned.loc[indices]
        labels = sample["label"].to_numpy(int)
        candidate = compute_metrics(labels, sample["probability"].to_numpy(float))
        anchor = compute_metrics(labels, sample["anchor_probability"].to_numpy(float))
        for name in metric_names:
            samples[name].append(float(candidate[name] - anchor[name]))
    rows = []
    for name in metric_names:
        values = np.asarray(samples[name], dtype=float)
        rows.append(
            {
                "seed": int(aligned["seed"].iloc[0]),
                "metric": name,
                "replicates": int(replicates),
                "bootstrap_seed": int(seed),
                "mean_delta": float(values.mean()),
                "ci_low": float(np.quantile(values, 0.025)),
                "ci_high": float(np.quantile(values, 0.975)),
            }
        )
    return pd.DataFrame(rows)
