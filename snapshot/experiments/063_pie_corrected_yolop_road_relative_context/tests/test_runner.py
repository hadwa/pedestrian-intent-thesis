from __future__ import annotations

import importlib
import json
from pathlib import Path
import subprocess
import sys

import pytest


def _module():
    return importlib.import_module(
        "experiments.063_pie_corrected_yolop_road_relative_context.scripts.runner"
    )


def test_extraction_runtime_import_does_not_require_xgboost() -> None:
    code = """
import builtins
real_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == 'xgboost' or name.startswith('xgboost.'):
        raise ModuleNotFoundError('xgboost intentionally unavailable in extraction runtime')
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded
import importlib
importlib.import_module('experiments.063_pie_corrected_yolop_road_relative_context.scripts.runner')
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[3],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr


def _summary(runner, *, f1: float, pr: float, roc: float, hc: float, per_seed):
    return runner.MetricBundle(
        f1=f1,
        pr_auc=pr,
        roc_auc=roc,
        harmonic=runner.harmonic_f1_pr(f1, pr),
        high_confidence_errors=hc,
        per_seed_harmonic=tuple(per_seed),
        invariants_pass=True,
    )


def test_expected_run_grid_is_exactly_twelve_unique_jobs() -> None:
    runner = _module()
    keys = runner.expected_run_keys()
    assert len(keys) == 12
    assert len(set(keys)) == 12
    assert keys[0] == ("corrected_grid", "rf_v1_bbox", 42)
    assert keys[-1] == ("road_relative", "xgb_v2_bbox", 44)


def test_candidate_selection_uses_frozen_tie_break_order() -> None:
    runner = _module()
    anchor = _summary(
        runner,
        f1=0.70,
        pr=0.74,
        roc=0.90,
        hc=20,
        per_seed=(0.71, 0.71, 0.71),
    )
    grid = _summary(
        runner,
        f1=0.72,
        pr=0.76,
        roc=0.91,
        hc=19,
        per_seed=(0.73, 0.73, 0.73),
    )
    road = _summary(
        runner,
        f1=0.73,
        pr=0.77,
        roc=0.92,
        hc=18,
        per_seed=(0.74, 0.74, 0.74),
    )
    winner, decisions = runner.select_validation_candidate(
        {"corrected_grid__rf_v1_bbox": grid, "road_relative__rf_v1_bbox": road},
        {"corrected_grid__rf_v1_bbox": anchor, "road_relative__rf_v1_bbox": anchor},
    )
    assert all(decision.eligible for decision in decisions.values())
    assert winner == "road_relative__rf_v1_bbox"


def test_no_eligible_candidate_writes_validation_only_marker(tmp_path: Path) -> None:
    runner = _module()
    anchor = _summary(
        runner,
        f1=0.70,
        pr=0.74,
        roc=0.90,
        hc=20,
        per_seed=(0.71, 0.71, 0.71),
    )
    weak = _summary(
        runner,
        f1=0.69,
        pr=0.73,
        roc=0.89,
        hc=21,
        per_seed=(0.69, 0.69, 0.69),
    )
    winner, decisions = runner.select_validation_candidate(
        {"corrected_grid__rf_v1_bbox": weak},
        {"corrected_grid__rf_v1_bbox": anchor},
    )
    marker = runner.write_validation_decision(tmp_path, winner, decisions, {})
    assert marker.name == "VALIDATION_ONLY_COMPLETE"
    assert not (tmp_path / "FROZEN_FINAL_MODEL.json").exists()
    payload = json.loads(marker.read_text(encoding="utf-8"))
    assert payload["set03_accessed"] is False


def test_training_completion_requires_all_twelve_markers(tmp_path: Path) -> None:
    runner = _module()
    for representation, anchor, seed in runner.expected_run_keys()[:-1]:
        (tmp_path / runner.run_marker_name(representation, anchor, seed)).write_text(
            json.dumps(
                {
                    "representation": representation,
                    "anchor_name": anchor,
                    "seed": seed,
                    "artifacts": {},
                }
            )
        )
    with pytest.raises(RuntimeError, match="12"):
        runner.assert_all_runs_complete(tmp_path)
