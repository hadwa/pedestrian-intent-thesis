"""Common fixed-threshold and paired evaluation for Exp076/077."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
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


THRESHOLD = 0.5


def compute_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float | int]:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if labels.ndim != 1 or probabilities.shape != labels.shape or not len(labels):
        raise ValueError("metric vectors must be aligned and nonempty")
    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("metric labels must be binary")
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("metric probabilities must be finite in [0,1]")
    predictions = (probabilities >= THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    positive_count = int((labels == 1).sum())
    negative_count = int((labels == 0).sum())
    return {
        "n": int(len(labels)),
        "positive_count": positive_count,
        "negative_count": negative_count,
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
        "fpr": float(fp / negative_count) if negative_count else float("nan"),
        "fnr": float(fn / positive_count) if positive_count else float("nan"),
        "brier": float(brier_score_loss(labels, probabilities)),
        "log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
        "fp_ge_0p9_count": int(np.sum((labels == 0) & (probabilities >= 0.9))),
        "fn_le_0p1_count": int(np.sum((labels == 1) & (probabilities <= 0.1))),
    }


def paired_results(frame: pd.DataFrame, *, scope: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"seed", "sequence_id", "label", "one_stage_probability", "two_stage_probability"}
    if required.difference(frame.columns) or frame.empty or frame.duplicated(["seed", "sequence_id"]).any():
        raise ValueError("paired prediction frame is invalid")
    metric_rows = []
    delta_rows = []
    delta_metrics = (
        "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "fpr", "fnr",
        "brier", "log_loss", "fp_ge_0p9_count", "fn_le_0p1_count",
    )
    for seed, group in frame.groupby("seed", sort=True):
        labels = group["label"].to_numpy(int)
        one = compute_metrics(labels, group["one_stage_probability"].to_numpy(float))
        two = compute_metrics(labels, group["two_stage_probability"].to_numpy(float))
        metric_rows.extend(
            [
                {"scope": scope, "seed": int(seed), "model": "one_stage", **one},
                {"scope": scope, "seed": int(seed), "model": "two_stage", **two},
            ]
        )
        delta = {"scope": scope, "seed": int(seed)}
        for name in delta_metrics:
            delta[f"delta_{name}"] = float(two[name] - one[name])
        delta_rows.append(delta)
    return pd.DataFrame(metric_rows), pd.DataFrame(delta_rows)


def summarize(per_seed: pd.DataFrame) -> pd.DataFrame:
    id_columns = {
        "scope", "seed", "model", "n", "positive_count", "negative_count", "threshold",
        "tn", "fp", "fn", "tp",
    }
    metric_columns = [name for name in per_seed.columns if name not in id_columns]
    rows: list[dict[str, Any]] = []
    for (scope, model), group in per_seed.groupby(["scope", "model"], sort=True):
        row: dict[str, Any] = {"scope": scope, "model": model, "seed_count": int(group["seed"].nunique())}
        for name in metric_columns:
            row[f"{name}_mean"] = float(group[name].mean())
            row[f"{name}_std"] = float(group[name].std(ddof=1)) if len(group) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)
