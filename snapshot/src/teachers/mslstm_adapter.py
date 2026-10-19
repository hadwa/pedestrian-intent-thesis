"""Adapter boundary for the external MSLSTM-PID teacher repository.

This module must not modify files under external/MSLSTM-PID directly. It maps the
frozen sequence CSVs into whatever input format the teacher repo requires and keeps
teacher logits aligned by sequence_id.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

TEACHER_PREDICTION_COLUMNS = ["sequence_id", "split", "label", "teacher_logit", "teacher_prob", "teacher_pred"]


def prepare_mslstm_inputs(sequence_csv: Path, feature_roots: dict[str, Path], active_streams: list[str], output_dir: Path) -> Path:
    """Prepare teacher input files from frozen sequence CSVs.

    Adapt this function after inspecting the MSLSTM-PID data API. The output must
    preserve sequence_id so exported predictions can be joined back exactly.
    """
    if not sequence_csv.exists():
        raise FileNotFoundError(f"Missing sequence CSV: {sequence_csv}")
    missing_roots = [name for name in active_streams if name not in feature_roots]
    if missing_roots:
        raise ValueError(f"Missing feature_roots entries for active streams: {missing_roots}")
    output_dir.mkdir(parents=True, exist_ok=True)
    raise NotImplementedError(
        "MSLSTM-PID input preparation requires inspecting external/MSLSTM-PID. "
        "Implement this boundary without changing the external repo."
    )


def run_mslstm_training(config_path: Path) -> Path:
    """Run teacher training through the external repo and return a checkpoint path."""
    raise NotImplementedError(
        "Connect this wrapper to the MSLSTM-PID training command/API after cloning "
        "external/MSLSTM-PID and inspecting its expected config format."
    )


def export_teacher_logits(checkpoint_path: Path, sequence_csv: Path, output_path: Path) -> pd.DataFrame:
    """Export teacher logits/probabilities aligned to sequence_id."""
    raise NotImplementedError(
        "Connect this wrapper to the MSLSTM-PID inference/export path. The saved CSV "
        f"must include columns: {TEACHER_PREDICTION_COLUMNS}."
    )


def load_teacher_predictions(path: Path) -> pd.DataFrame:
    """Load exported teacher predictions and validate required columns."""
    if not path.exists():
        raise FileNotFoundError(f"Missing teacher prediction file: {path}")
    df = pd.read_csv(path)
    missing = [col for col in TEACHER_PREDICTION_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Teacher prediction file {path} is missing columns: {missing}")
    if df["sequence_id"].duplicated().any():
        raise ValueError(f"Teacher prediction file {path} contains duplicate sequence_id values.")
    return df
