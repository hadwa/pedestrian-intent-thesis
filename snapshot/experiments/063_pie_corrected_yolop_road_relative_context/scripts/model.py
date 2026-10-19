"""Authenticated direct reuse of the exact Exp062 context-correction model."""

from __future__ import annotations

import importlib
from pathlib import Path

from .contracts import ExperimentConfig, sha256_file


REPO_ROOT = Path(__file__).resolve().parents[3]
EXP062_MODEL_SOURCE = (
    REPO_ROOT
    / "experiments/062_hybrid_tree_anchor_context_correction_skl152/scripts/context_model.py"
)


def authenticate_model_source() -> str:
    """Fail if the inherited model/training implementation changes by one byte."""
    expected = ExperimentConfig().exp062_model_source_sha256
    if not EXP062_MODEL_SOURCE.is_file() or sha256_file(EXP062_MODEL_SOURCE) != expected:
        raise RuntimeError("Exp062 context-model source authentication failed")
    return expected


authenticate_model_source()
_source = importlib.import_module(
    "experiments.062_hybrid_tree_anchor_context_correction_skl152.scripts.context_model"
)

HybridContextCorrector = _source.HybridContextCorrector
CorrectionOutput = _source.CorrectionOutput
TrainingResult = _source.TrainingResult
predict_context_corrector = _source.predict_context_corrector
train_one_seed = _source.train_one_seed


__all__ = [
    "CorrectionOutput",
    "HybridContextCorrector",
    "TrainingResult",
    "authenticate_model_source",
    "predict_context_corrector",
    "train_one_seed",
]
