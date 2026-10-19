from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts/evaluation.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("exp076_evaluation", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_fixed_threshold_metrics_include_confident_errors() -> None:
    evaluation = _load_module()
    labels = np.asarray([0, 0, 1, 1])
    probabilities = np.asarray([0.1, 0.95, 0.05, 0.9])

    metrics = evaluation.compute_metrics(labels, probabilities)

    assert metrics["tn"] == 1
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert metrics["tp"] == 1
    assert metrics["f1"] == 0.5
    assert metrics["fp_ge_0p9_count"] == 1
    assert metrics["fn_le_0p1_count"] == 1


def test_paired_results_align_models_by_seed_and_sequence() -> None:
    evaluation = _load_module()
    frame = pd.DataFrame(
        {
            "seed": [42, 42, 42, 42],
            "sequence_id": ["a", "b", "c", "d"],
            "label": [0, 0, 1, 1],
            "one_stage_probability": [0.1, 0.7, 0.3, 0.8],
            "two_stage_probability": [0.1, 0.4, 0.6, 0.8],
        }
    )

    per_seed, deltas = evaluation.paired_results(frame, scope="validation")

    assert set(per_seed["model"]) == {"one_stage", "two_stage"}
    assert deltas.loc[0, "delta_f1"] > 0
    assert deltas.loc[0, "scope"] == "validation"
