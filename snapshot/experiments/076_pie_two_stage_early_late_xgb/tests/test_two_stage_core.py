from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
CORE_PATH = ROOT / "scripts/two_stage_core.py"


def _load_core():
    spec = importlib.util.spec_from_file_location("exp076_two_stage_core", CORE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _synthetic_pose() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base = np.asarray(
        [
            [50, 15], [47, 13], [53, 13], [44, 14], [56, 14],
            [42, 30], [58, 30], [38, 48], [62, 48], [36, 68], [64, 68],
            [45, 62], [55, 62], [44, 88], [57, 88], [42, 116], [60, 116],
        ],
        dtype=float,
    )
    coords = np.stack([base + np.asarray([2.0 * index, 0.2 * index]) for index in range(8)])
    confidence = np.full((8, 17), 0.9, dtype=float)
    bboxes = np.asarray([[20 + index, 5, 85 + index, 130] for index in range(8)], dtype=float)
    return coords, confidence, bboxes


def test_extract_half_features_matches_frozen_joint_schema() -> None:
    core = _load_core()
    definitions = ROOT.parents[0] / "012_pie_biu_v2_30_60_benchmark/metrics/pie_biu_v2_feature_set_definitions.csv"
    contract = core.load_joint_contract(definitions)
    coords, confidence, bboxes = _synthetic_pose()

    features = core.extract_half_features(coords, confidence, bboxes, contract=contract)

    assert tuple(features) == contract["joint"]
    assert len(features) == 161
    assert features["bbox_valid_frame_count"] == pytest.approx(8.0)
    assert features["biu_valid_frame_count"] == pytest.approx(8.0)
    assert np.isfinite(np.asarray(list(features.values()), dtype=float)).all()


def test_early_view_is_invariant_to_late_frame_changes() -> None:
    core = _load_core()
    definitions = ROOT.parents[0] / "012_pie_biu_v2_30_60_benchmark/metrics/pie_biu_v2_feature_set_definitions.csv"
    contract = core.load_joint_contract(definitions)
    coords8, confidence8, bboxes8 = _synthetic_pose()
    coords = np.concatenate([coords8, coords8 + np.asarray([30.0, 5.0])])
    confidence = np.concatenate([confidence8, confidence8 * 0.8])
    bboxes = np.concatenate([bboxes8, bboxes8 + np.asarray([30.0, 5.0, 30.0, 5.0])])

    first = core.extract_window_views(coords, confidence, bboxes, contract=contract)
    changed_coords = coords.copy()
    changed_coords[8:] += 500.0
    changed_bboxes = bboxes.copy()
    changed_bboxes[8:] += np.asarray([400.0, 200.0, 400.0, 200.0])
    second = core.extract_window_views(changed_coords, confidence, changed_bboxes, contract=contract)

    assert first["early"] == second["early"]
    assert first["full"] != second["full"]
    assert tuple(first["portable"]) == contract["portable"]
    assert len(first["portable"]) == 182


def test_track_group_oof_has_complete_coverage_without_group_overlap() -> None:
    core = _load_core()
    rng = np.random.default_rng(7)
    groups = np.repeat([f"track-{index}" for index in range(8)], 3)
    labels = np.repeat([0, 0, 0, 0, 1, 1, 1, 1], 3)
    values = rng.normal(size=(len(labels), 5))
    values[:, 0] += labels * 1.5

    probabilities, audit = core.group_oof_probabilities(
        values,
        labels,
        groups,
        seed=42,
        n_estimators=4,
        n_splits=4,
    )

    assert probabilities.shape == (len(labels),)
    assert np.isfinite(probabilities).all()
    assert np.logical_and(probabilities >= 0.0, probabilities <= 1.0).all()
    assert audit["covered_rows"] == len(labels)
    assert audit["duplicate_rows"] == 0
    assert all(fold["group_overlap"] == 0 for fold in audit["folds"])


def test_two_stage_model_splits_tree_budget_and_predicts_once_per_row() -> None:
    core = _load_core()
    rng = np.random.default_rng(11)
    groups = np.repeat([f"track-{index}" for index in range(10)], 4)
    labels = np.repeat([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], 4)
    early = rng.normal(size=(len(labels), 6))
    full = rng.normal(size=(len(labels), 9))
    early[:, 0] += labels
    full[:, 0] += labels * 1.5

    bundle, audit = core.fit_two_stage(
        early,
        full,
        labels,
        groups,
        seed=42,
        total_estimators=10,
        n_splits=5,
    )
    probabilities = core.predict_two_stage(bundle, early, full)

    assert bundle.stage1_estimators == 5
    assert bundle.stage2_estimators == 5
    assert bundle.stage2_feature_count == full.shape[1] + 1
    assert audit["oof"]["covered_rows"] == len(labels)
    assert probabilities.shape == (len(labels),)
    assert np.logical_and(probabilities >= 0.0, probabilities <= 1.0).all()


def test_cohort_parity_rejects_reordered_or_relabelled_rows() -> None:
    core = _load_core()
    reference = pd.DataFrame(
        {"sequence_id": ["a", "b"], "track_id": ["t1", "t2"], "label": [0, 1]}
    )
    core.assert_cohort_parity(reference, reference.copy())

    with pytest.raises(ValueError, match="cohort parity"):
        core.assert_cohort_parity(reference, reference.iloc[::-1].reset_index(drop=True))
    changed = reference.copy()
    changed.loc[1, "label"] = 0
    with pytest.raises(ValueError, match="cohort parity"):
        core.assert_cohort_parity(reference, changed)
