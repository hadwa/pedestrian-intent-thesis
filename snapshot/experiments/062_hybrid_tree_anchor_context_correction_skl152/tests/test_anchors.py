"""Executable contracts for the Exp062 parity-checked tree anchors."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.anchors import (  # noqa: E402
    ANCHOR_FEATURE_DIMENSIONS,
    anchor_complexity,
    anchor_feature_metadata,
    build_anchor,
    build_grouped_oof,
    fit_full_anchor,
    verify_anchor_parity,
)
from scripts.contracts import sha256_file  # noqa: E402


def _synthetic_grouped_data(
    feature_count: int = ANCHOR_FEATURE_DIMENSIONS["rf_v1_bbox"],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create ten mixed-class tracks, enough for five stratified group folds."""
    rng = np.random.RandomState(19)
    track_ids = np.repeat(np.array([f"track-{index}" for index in range(10)]), 4)
    labels = np.tile(np.array([0, 1, 0, 1], dtype=np.int64), 10)
    features = rng.normal(size=(len(labels), feature_count))
    features[:, 0] += labels * 1.5
    features[0, 1] = np.nan
    return features, labels, track_ids


def _parity_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq-0", "seq-1", "seq-2", "seq-3"],
            "label": [0, 1, 0, 1],
            "probability": [0.1, 0.9, 0.2, 0.8],
        }
    )


def test_rf_v1_bbox_exact_configuration() -> None:
    """Catches any drift from the frozen historical RF parameters."""
    model = build_anchor("rf_v1_bbox", seed=42, labels=np.array([0, 1]))
    rf = model.named_steps["classifier"]

    assert rf.n_estimators == 300
    assert rf.min_samples_leaf == 2
    assert rf.class_weight == "balanced_subsample"
    assert rf.random_state == 42
    assert rf.n_jobs == -1
    assert model.named_steps["imputer"].strategy == "median"


def test_xgb_v2_bbox_exact_configuration_and_class_balance() -> None:
    """Catches XGBoost hyperparameter or train-label balance drift."""
    model = build_anchor("xgb_v2_bbox", seed=43, labels=np.array([0, 0, 0, 1]))
    xgb = model.named_steps["classifier"]

    assert xgb.n_estimators == 250
    assert xgb.max_depth == 4
    assert xgb.learning_rate == 0.05
    assert xgb.subsample == 0.8
    assert xgb.colsample_bytree == 0.8
    assert xgb.reg_lambda == 1.0
    assert xgb.objective == "binary:logistic"
    assert xgb.eval_metric == "logloss"
    assert xgb.random_state == 43
    assert xgb.n_jobs == 4
    assert xgb.scale_pos_weight == 3.0


def test_fit_full_anchor_serializes_hash_and_native_rf_complexity(tmp_path: Path) -> None:
    """Catches reporting pre-close hashes or non-native RF tree accounting."""
    features, labels, _ = _synthetic_grouped_data()
    destination = tmp_path / "rf_seed42.joblib"

    result = fit_full_anchor(features, labels, "rf_v1_bbox", seed=42, path=destination)
    complexity = anchor_complexity(result.model)

    assert destination.is_file()
    assert result.sha256 == sha256_file(destination)
    assert result.size_bytes == destination.stat().st_size
    assert result.complexity == complexity
    assert complexity.tree_count == 300
    assert complexity.node_count >= complexity.tree_count
    assert complexity.leaf_count >= complexity.tree_count


def test_fit_full_xgb_anchor_serializes_hash_and_native_complexity(tmp_path: Path) -> None:
    """Catches missing persisted XGBoost metadata or non-native complexity counts."""
    features, labels, _ = _synthetic_grouped_data(
        ANCHOR_FEATURE_DIMENSIONS["xgb_v2_bbox"]
    )
    destination = tmp_path / "xgb_seed42.joblib"

    result = fit_full_anchor(features, labels, "xgb_v2_bbox", seed=42, path=destination)

    assert result.path == destination
    assert result.sha256 == sha256_file(destination)
    assert result.size_bytes == destination.stat().st_size
    assert result.complexity.tree_count == 250
    assert result.complexity.node_count >= result.complexity.tree_count
    assert result.complexity.leaf_count >= result.complexity.tree_count


@pytest.mark.parametrize(
    ("anchor_name", "feature_count"),
    [("rf_v1_bbox", 124), ("xgb_v2_bbox", 108)],
)
def test_public_fit_and_oof_boundaries_reject_wrong_anchor_feature_dimensions(
    tmp_path: Path, anchor_name: str, feature_count: int
) -> None:
    """Catches an anchor fit on a matrix not matching its frozen feature ABI."""
    features, labels, track_ids = _synthetic_grouped_data(feature_count)

    with pytest.raises(ValueError, match="feature dimension"):
        fit_full_anchor(features, labels, anchor_name, seed=42, path=tmp_path / "full.joblib")
    with pytest.raises(ValueError, match="feature dimension"):
        build_grouped_oof(
            features,
            labels,
            track_ids,
            anchor_name,
            seed=42,
            output_dir=tmp_path / "folds",
        )


def test_anchor_feature_metadata_exposes_frozen_dimensions() -> None:
    """Catches a runner-facing anchor metadata view that omits the input ABI."""
    assert anchor_feature_metadata("rf_v1_bbox") == {
        "anchor_name": "rf_v1_bbox",
        "feature_dimension": 125,
        "feature_set": "biu_v1_plus_bbox",
    }
    assert anchor_feature_metadata("xgb_v2_bbox")["feature_dimension"] == 109


def test_oof_assigns_each_row_once_without_group_leakage_and_persists_folds(
    tmp_path: Path,
) -> None:
    """Catches overlap, dropped rows, or a track appearing on both fold sides."""
    features, labels, track_ids = _synthetic_grouped_data()

    output_dir = tmp_path / "rf_folds"
    output_dir.mkdir()
    result = build_grouped_oof(
        features, labels, track_ids, "rf_v1_bbox", seed=42, output_dir=output_dir
    )

    assert np.all(result.assignment_count == 1)
    assert set(result.fold_assignment.tolist()) == {1, 2, 3, 4, 5}
    assert result.probabilities.shape == labels.shape
    assert np.isfinite(result.probabilities).all()
    assert result.fold_audit["shared_track_count"].eq(0).all()
    assert result.fold_audit["train_rows"].add(result.fold_audit["holdout_rows"]).eq(len(labels)).all()
    assert result.fold_audit["train_unique_tracks"].add(
        result.fold_audit["holdout_unique_tracks"]
    ).eq(len(np.unique(track_ids))).all()
    assert len(result.fold_artifacts) == 5
    assert {artifact.fold for artifact in result.fold_artifacts} == {1, 2, 3, 4, 5}
    for artifact in result.fold_artifacts:
        assert artifact.path == output_dir / f"rf_v1_bbox__seed42__fold{artifact.fold}.joblib"
        assert artifact.path.is_file()
        assert artifact.sha256 == sha256_file(artifact.path)
        assert artifact.size_bytes == artifact.path.stat().st_size
        assert artifact.complexity.tree_count == 300
        assert artifact.holdout_rows == int((result.fold_assignment == artifact.fold).sum())
        assert artifact.shared_track_count == 0


def test_oof_rejects_nonempty_or_missing_artifact_directories(tmp_path: Path) -> None:
    """Catches fold artifact overwrite or an ambiguous caller output directory."""
    features, labels, track_ids = _synthetic_grouped_data()
    missing = tmp_path / "missing"
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "existing.joblib").touch()

    with pytest.raises(ValueError, match="existing empty"):
        build_grouped_oof(
            features, labels, track_ids, "rf_v1_bbox", seed=42, output_dir=missing
        )
    with pytest.raises(ValueError, match="empty"):
        build_grouped_oof(
            features, labels, track_ids, "rf_v1_bbox", seed=42, output_dir=occupied
        )


@pytest.mark.parametrize(
    ("mutate", "expected_failure"),
    [
        (lambda frame: frame.assign(sequence_id=frame["sequence_id"].iloc[::-1].to_numpy()), "sequence_id"),
        (lambda frame: frame.assign(label=[1, 1, 0, 1]), "label"),
        (lambda frame: frame.assign(probability=[0.6, 0.9, 0.2, 0.8]), "fixed_0.5_class"),
        (lambda frame: frame.assign(probability=[0.100000101, 0.9, 0.2, 0.8]), "probability"),
    ],
    ids=("swapped_sequence_id", "changed_label", "changed_fixed_class", "probability_over_tolerance"),
)
def test_validation_parity_fails_closed_on_any_identity_or_prediction_drift(
    mutate: object, expected_failure: str
) -> None:
    """Catches silent acceptance of all four required parity violations."""
    historical = _parity_frame()
    fresh = mutate(historical.copy())

    result = verify_anchor_parity(fresh, historical)

    assert not result.passed
    assert expected_failure in result.failed_checks


def test_validation_parity_records_exact_metric_and_probability_agreement() -> None:
    """Catches success reports that omit the fixed-threshold metric comparisons."""
    historical = _parity_frame()

    result = verify_anchor_parity(historical.copy(), historical)

    assert result.passed
    assert result.max_probability_error == 0.0
    assert result.max_metric_error == 0.0
    assert result.id_match
    assert result.label_match
    assert result.class_match
