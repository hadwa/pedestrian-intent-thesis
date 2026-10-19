"""Teacher filtering policies for later KD experiments."""

from __future__ import annotations

import pandas as pd


def filter_by_teacher_confidence(predictions: pd.DataFrame, min_confidence: float) -> pd.DataFrame:
    """Keep samples where teacher confidence is at least min_confidence."""
    if "teacher_prob" not in predictions.columns:
        raise ValueError("predictions must contain teacher_prob.")
    confidence = predictions["teacher_prob"].sub(0.5).abs() * 2.0
    return predictions[confidence >= min_confidence].copy()


def filter_teacher_correct_samples(predictions: pd.DataFrame) -> pd.DataFrame:
    """Keep samples where teacher_pred equals the hard label."""
    required = {"label", "teacher_pred"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"predictions missing columns: {sorted(missing)}")
    return predictions[predictions["label"].astype(int) == predictions["teacher_pred"].astype(int)].copy()
