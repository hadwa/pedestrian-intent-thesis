"""Reusable binary metrics with validation-only threshold selection."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score


def _as_arrays(y_true: Any, y_prob: Any) -> tuple[np.ndarray, np.ndarray]:
    y_true_arr = np.asarray(y_true).astype(int)
    y_prob_arr = np.asarray(y_prob).astype(float)
    if y_true_arr.shape[0] != y_prob_arr.shape[0]:
        raise ValueError(f"y_true and y_prob must have the same length, got {len(y_true_arr)} and {len(y_prob_arr)}.")
    if y_true_arr.size == 0:
        raise ValueError("Cannot compute metrics for an empty split.")
    return y_true_arr, y_prob_arr


def _safe_auc(metric_name: str, func: Any, y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        warnings.warn(f"{metric_name} is undefined because y_true contains only one class; returning nan.", RuntimeWarning)
        return float("nan")
    return float(func(y_true, y_prob))


def compute_binary_metrics(y_true: Any, y_prob: Any, threshold: float) -> dict[str, float]:
    """Compute binary metrics from probabilities and a fixed threshold."""
    y_true_arr, y_prob_arr = _as_arrays(y_true, y_prob)
    y_pred = (y_prob_arr >= float(threshold)).astype(int)
    return {
        "f1": float(f1_score(y_true_arr, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true_arr, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true_arr, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true_arr, y_pred)),
        "roc_auc": _safe_auc("roc_auc", roc_auc_score, y_true_arr, y_prob_arr),
        "pr_auc": _safe_auc("pr_auc", average_precision_score, y_true_arr, y_prob_arr),
        "threshold": float(threshold),
    }


def find_best_threshold_on_validation(y_true: Any, y_prob: Any, metric: str = "f1") -> float:
    """Select a threshold using validation data only."""
    if metric != "f1":
        raise ValueError(f"Unsupported threshold-selection metric {metric!r}; currently only 'f1' is implemented.")
    y_true_arr, y_prob_arr = _as_arrays(y_true, y_prob)
    candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], y_prob_arr)))
    best_threshold = 0.5
    best_score = -1.0
    for threshold in candidates:
        y_pred = (y_prob_arr >= threshold).astype(int)
        score = f1_score(y_true_arr, y_pred, zero_division=0)
        if score > best_score or (score == best_score and abs(float(threshold) - 0.5) < abs(best_threshold - 0.5)):
            best_score = float(score)
            best_threshold = float(threshold)
    return best_threshold


def evaluate_with_validation_threshold(val_y: Any, val_prob: Any, test_y: Any, test_prob: Any) -> dict[str, dict[str, float]]:
    """Choose threshold on validation, then evaluate validation and test once."""
    threshold = find_best_threshold_on_validation(val_y, val_prob, metric="f1")
    return {
        "validation": compute_binary_metrics(val_y, val_prob, threshold),
        "test": compute_binary_metrics(test_y, test_prob, threshold),
    }
