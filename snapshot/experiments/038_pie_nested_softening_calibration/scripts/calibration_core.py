"""Pure calibration, metrics, and invariants for Exp038."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


THRESHOLD = 0.5
EPSILON = 1e-12
TEMPERATURE_BOUNDS = (1.0, 100.0)


@dataclass(frozen=True)
class TemperatureFit:
    temperature: float
    identity_log_loss: float
    fitted_log_loss: float
    optimizer_temperature: float
    optimizer_success: bool


def _validate_probability(probability: np.ndarray) -> np.ndarray:
    values = np.asarray(probability, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("probabilities must be a nonempty finite vector")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("probabilities must lie in [0,1]")
    return values


def _validate_labels(labels: np.ndarray, size: int, *, require_both: bool = False) -> np.ndarray:
    values = np.asarray(labels, dtype=int)
    if values.ndim != 1 or len(values) != size or not set(np.unique(values)).issubset({0, 1}):
        raise ValueError("labels must be an aligned binary vector")
    if require_both and len(np.unique(values)) != 2:
        raise ValueError("temperature fitting requires both label classes")
    return values


def apply_temperature(
    probability: np.ndarray, temperature: float, epsilon: float = EPSILON
) -> np.ndarray:
    values = _validate_probability(probability)
    temperature = float(temperature)
    if not np.isfinite(temperature) or not TEMPERATURE_BOUNDS[0] <= temperature <= TEMPERATURE_BOUNDS[1]:
        raise ValueError("temperature must be finite in [1,100]")
    if not 0.0 < float(epsilon) < 0.5:
        raise ValueError("probability epsilon is invalid")
    clipped = np.clip(values, epsilon, 1.0 - epsilon)
    logit = np.log(clipped) - np.log1p(-clipped)
    return expit(logit / temperature)


def binary_log_loss(labels: np.ndarray, probability: np.ndarray) -> float:
    values = _validate_probability(probability)
    targets = _validate_labels(labels, len(values))
    clipped = np.clip(values, EPSILON, 1.0 - EPSILON)
    return float(-np.mean(targets * np.log(clipped) + (1 - targets) * np.log1p(-clipped)))


def fit_softening_temperature(labels: np.ndarray, probability: np.ndarray) -> TemperatureFit:
    values = _validate_probability(probability)
    targets = _validate_labels(labels, len(values), require_both=True)
    order = np.lexsort((targets, values))
    values = values[order]
    targets = targets[order]
    identity_loss = binary_log_loss(targets, values)

    result = minimize_scalar(
        lambda temperature: binary_log_loss(targets, apply_temperature(values, temperature)),
        method="bounded",
        bounds=TEMPERATURE_BOUNDS,
        options={"xatol": 1e-12},
    )
    if not result.success or not np.isfinite(result.x) or not np.isfinite(result.fun):
        raise RuntimeError("temperature optimizer failed")
    optimizer_temperature = float(result.x)
    optimizer_loss = float(result.fun)
    if identity_loss - optimizer_loss <= 1e-12:
        temperature = 1.0
        fitted_loss = identity_loss
    else:
        temperature = optimizer_temperature
        fitted_loss = optimizer_loss
    return TemperatureFit(
        temperature=temperature,
        identity_log_loss=identity_loss,
        fitted_log_loss=fitted_loss,
        optimizer_temperature=optimizer_temperature,
        optimizer_success=True,
    )


def calibration_error(labels: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    values = _validate_probability(probability)
    targets = _validate_labels(labels, len(values))
    assignments = np.minimum(np.digitize(values, np.linspace(0, 1, bins + 1)[1:-1], right=True), bins - 1)
    return float(
        sum(
            (mask.sum() / len(values)) * abs(float(targets[mask].mean()) - float(values[mask].mean()))
            for index in range(bins)
            if (mask := assignments == index).any()
        )
    )


def compute_metrics(labels: np.ndarray, probability: np.ndarray) -> dict[str, float | int]:
    values = _validate_probability(probability)
    targets = _validate_labels(labels, len(values))
    prediction = (values >= THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(targets, prediction, labels=[0, 1]).ravel()
    negatives = int((targets == 0).sum())
    positives = int((targets == 1).sum())
    fp_tail = int(((targets == 0) & (values >= 0.9)).sum())
    fn_tail = int(((targets == 1) & (values <= 0.1)).sum())
    return {
        "n": int(len(values)),
        "positive_count": positives,
        "threshold": THRESHOLD,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "accuracy": float(accuracy_score(targets, prediction)),
        "precision": float(precision_score(targets, prediction, zero_division=0)),
        "recall": float(recall_score(targets, prediction, zero_division=0)),
        "f1_fixed_0p5": float(f1_score(targets, prediction, zero_division=0)),
        "fpr": float(fp / negatives) if negatives else float("nan"),
        "fnr": float(fn / positives) if positives else float("nan"),
        "pr_auc": float(average_precision_score(targets, values)),
        "roc_auc": float(roc_auc_score(targets, values)) if len(np.unique(targets)) == 2 else float("nan"),
        "brier": float(brier_score_loss(targets, values)),
        "log_loss": binary_log_loss(targets, values),
        "ece": calibration_error(targets, values),
        "fp_ge_0p9_count": fp_tail,
        "fn_le_0p1_count": fn_tail,
        "fp_ge_0p9_rate_all_negatives": float(fp_tail / negatives) if negatives else float("nan"),
        "fn_le_0p1_rate_all_positives": float(fn_tail / positives) if positives else float("nan"),
    }


def verify_softening_invariants(
    raw_probability: np.ndarray, calibrated_probability: np.ndarray, labels: np.ndarray
) -> dict[str, bool]:
    raw = _validate_probability(raw_probability)
    calibrated = _validate_probability(calibrated_probability)
    targets = _validate_labels(labels, len(raw))
    if len(calibrated) != len(raw):
        raise ValueError("calibrated probabilities are misaligned")
    if np.any(np.abs(calibrated - 0.5) > np.abs(raw - 0.5) + 1e-14):
        raise ValueError("softening direction invariant failed")
    if not np.array_equal(raw >= 0.5, calibrated >= 0.5):
        raise ValueError("fixed-0.5 prediction invariant failed")
    if not np.array_equal(np.argsort(raw, kind="stable"), np.argsort(calibrated, kind="stable")):
        raise ValueError("ranking invariant failed")
    raw_metrics = compute_metrics(targets, raw)
    calibrated_metrics = compute_metrics(targets, calibrated)
    for metric in ("tn", "fp", "fn", "tp"):
        if raw_metrics[metric] != calibrated_metrics[metric]:
            raise ValueError("confusion invariant failed")
    for metric in ("fp_ge_0p9_count", "fn_le_0p1_count"):
        if calibrated_metrics[metric] > raw_metrics[metric]:
            raise ValueError("high-confidence tail invariant failed")
    return {"softening": True, "prediction": True, "ranking": True, "confusion": True, "tails": True}
