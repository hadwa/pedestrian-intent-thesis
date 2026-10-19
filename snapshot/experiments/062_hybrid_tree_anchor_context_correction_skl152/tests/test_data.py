"""Integration contracts for the Exp062 authenticated train/validation loader."""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.contracts import ExperimentConfig  # noqa: E402
from scripts.data import (  # noqa: E402
    ContextScaler,
    TrainValData,
    YOLOP_CHANNELS,
    fit_context_scaler,
    load_authenticated_trainval,
    read_historical_validation_prefix,
    train_row_manifest_sha256,
    transform_context,
    write_context_scaler,
)


def test_authenticated_trainval_has_exact_cohorts_and_shapes() -> None:
    """Catches source, cohort, split-order, shape, or finite-value drift."""
    data = load_authenticated_trainval(ExperimentConfig())

    assert data.rows.shape[0] == 5985
    assert data.train_mask.sum() == 2366
    assert data.val_mask.sum() == 3619
    assert data.rows.loc[data.train_mask, "split"].eq("train").all()
    assert data.rows.loc[data.val_mask, "split"].eq("val").all()
    assert data.summary_v1.shape == (5985, 125)
    assert data.summary_v2_core.shape == (5985, 109)
    assert data.ego.shape == (5985, 16, 3)
    assert data.yolop.shape == (5985, 16, 20)
    assert data.rows.loc[data.train_mask, "track_id"].notna().all()
    assert data.rows["cache_index"].to_numpy().tolist() == list(range(5985))
    assert set(data.rows.loc[data.train_mask, "set_id"]) == {"set01", "set02", "set06"}
    assert set(data.rows.loc[data.val_mask, "set_id"]) == {"set04", "set05"}
    assert set(data.labels.tolist()) == {0, 1}
    assert np.isfinite(data.summary_v1).all()
    assert np.isfinite(data.summary_v2_core).all()
    assert np.isfinite(data.ego).all()
    assert np.isfinite(data.yolop).all()


def test_anchor_summaries_are_reconstructed_from_exp012_not_exp055_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catches silently restoring the known-nonparity Exp055 summary arrays."""
    import scripts.data as data_module

    original_load_array = data_module._load_array

    def contexts_only(filename: str, expected_shape: tuple[int, ...]) -> np.ndarray:
        if filename.startswith("summary_"):
            pytest.fail("Exp055 summary cache must not supply Exp062 anchor arrays")
        return original_load_array(filename, expected_shape)

    monkeypatch.setattr(data_module, "_load_array", contexts_only)
    data = load_authenticated_trainval(ExperimentConfig())

    assert data.summary_v1.shape == (5985, 125)
    assert data.summary_v2_core.shape == (5985, 109)


def test_exp012_anchor_cache_cannot_bypass_definition_authentication() -> None:
    """Catches returning cached anchors when the frozen Exp012 definition changes."""
    import scripts.data as data_module

    config = ExperimentConfig()
    rows, _, _, _ = data_module._validate_rows(
        pd.read_csv(data_module.EXP055_TRAINVAL / "rows.csv"), config
    )
    previous_cache = data_module._EXP012_ANCHOR_CACHE
    data_module._EXP012_ANCHOR_CACHE = None
    try:
        expected_v1, expected_v2 = data_module._reconstruct_from_exp012(rows, config)
        cached_v1, cached_v2 = data_module._reconstruct_from_exp012(rows, config)

        np.testing.assert_array_equal(cached_v1, expected_v1)
        np.testing.assert_array_equal(cached_v2, expected_v2)
        for altered_config in (
            replace(config, exp012_feature_definition_sha256="0" * 64),
            replace(config, exp012_feature_definition_path="unexpected-definition.csv"),
        ):
            with pytest.raises(RuntimeError, match="feature-definition"):
                data_module._reconstruct_from_exp012(rows, altered_config)
    finally:
        data_module._EXP012_ANCHOR_CACHE = previous_cache


def test_context_layout_and_frame_order_are_frozen() -> None:
    """Catches reordered YOLOP channels or a nonconsecutive 16-frame window."""
    data = load_authenticated_trainval(ExperimentConfig())

    assert YOLOP_CHANNELS == (
        "drivable_cell_0", "drivable_cell_1", "drivable_cell_2",
        "drivable_cell_3", "drivable_cell_4", "drivable_cell_5",
        "drivable_cell_6", "drivable_cell_7", "drivable_cell_8",
        "lane_cell_0", "lane_cell_1", "lane_cell_2", "lane_cell_3",
        "lane_cell_4", "lane_cell_5", "lane_cell_6", "lane_cell_7",
        "lane_cell_8", "vehicle_count", "pedestrian_count",
    )
    frames = data.rows["start_frame"].to_numpy()[:, None] + np.arange(16)
    np.testing.assert_array_equal(frames[:, -1], data.rows["end_frame"].to_numpy())


def synthetic_trainval_with_shifted_validation() -> TrainValData:
    rows = pd.DataFrame(
        {
            "sequence_id": ["train-a", "train-b", "val-a", "val-b"],
            "split": ["train", "train", "val", "val"],
            "label": [0, 1, 0, 1],
            "track_id": ["track-a", "track-b", "track-c", "track-d"],
        }
    )
    ego = np.array(
        [
            [[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]],
            [[5.0, 6.0, 7.0], [7.0, 8.0, 9.0]],
            [[101.0, 102.0, 103.0], [103.0, 104.0, 105.0]],
            [[105.0, 106.0, 107.0], [107.0, 108.0, 109.0]],
        ]
    )
    yolop = np.repeat(ego[..., :1], 20, axis=2)
    train_mask = rows["split"].eq("train").to_numpy()
    return TrainValData(
        rows=rows,
        summary_v1=np.zeros((4, 125)),
        summary_v2_core=np.zeros((4, 109)),
        ego=ego,
        yolop=yolop,
        labels=rows["label"].to_numpy(),
        train_mask=train_mask,
        val_mask=~train_mask,
    )


def test_scaler_uses_training_rows_only() -> None:
    """Catches validation rows influencing context normalization statistics."""
    data = synthetic_trainval_with_shifted_validation()
    scaler = fit_context_scaler(data)

    np.testing.assert_allclose(scaler.ego_mean, data.ego[data.train_mask].mean((0, 1)))
    np.testing.assert_allclose(scaler.yolop_mean, data.yolop[data.train_mask].mean((0, 1)))
    assert not np.allclose(scaler.ego_mean, data.ego.mean((0, 1)))
    transformed = transform_context(data, scaler)
    np.testing.assert_allclose(transformed.ego[data.train_mask].mean((0, 1)), 0.0, atol=1e-7)


def test_scaler_persistence_binds_derived_train_manifest(tmp_path: Path) -> None:
    """Catches a scaler marker bound to caller-supplied rather than actual train rows."""
    data = synthetic_trainval_with_shifted_validation()
    scaler = fit_context_scaler(data)
    destination = tmp_path / "context_scaler.json"

    payload = write_context_scaler(destination, data, scaler)

    assert json.loads(destination.read_text(encoding="utf-8")) == payload
    assert payload["train_row_manifest_sha256"] == train_row_manifest_sha256(data)
    assert payload["ego_mean"] == scaler.ego_mean.tolist()


def test_historical_reader_is_bounded_to_validation_prefix() -> None:
    """Catches test-row parsing or accepting a non-approved anchor filename."""
    path = (
        ROOT.parent
        / "012_pie_biu_v2_30_60_benchmark/predictions/biu_v1_plus_bbox__rf__seed42.csv"
    )
    source_rows = pd.read_csv(
        ROOT.parent / "055_compact_residual_biu_lstm/cache/trainval/rows.csv"
    )
    expected_validation = source_rows.loc[
        source_rows["split"].eq("val"), ["sequence_id", "split", "label"]
    ].reset_index(drop=True)
    frame = read_historical_validation_prefix(path, "rf_v1_bbox", 42, expected_validation)

    assert frame.shape == (3619, 4)
    assert frame.columns.tolist() == ["sequence_id", "split", "label", "probability"]
    assert frame["split"].eq("val").all()
    with pytest.raises(ValueError, match="approved"):
        read_historical_validation_prefix(path, "xgb_v2_bbox", 42, expected_validation)
    with pytest.raises(ValueError, match="3,619"):
        read_historical_validation_prefix(path, "rf_v1_bbox", 42, expected_validation, expected_rows=1)
    with pytest.raises(ValueError, match="cohort"):
        read_historical_validation_prefix(
            path, "rf_v1_bbox", 42, expected_validation.iloc[::-1].reset_index(drop=True)
        )
    mismatched_labels = expected_validation.copy()
    mismatched_labels.loc[0, "label"] = 1 - int(mismatched_labels.loc[0, "label"])
    with pytest.raises(ValueError, match="cohort"):
        read_historical_validation_prefix(path, "rf_v1_bbox", 42, mismatched_labels)


def test_authentication_rejects_hash_drift_before_array_load(monkeypatch: pytest.MonkeyPatch) -> None:
    """Catches opening a train/validation cache after its manifest hash drifts."""
    import scripts.data as data_module

    monkeypatch.setattr(data_module, "sha256_file", lambda _path: "drifted")
    monkeypatch.setattr(data_module.np, "load", lambda *_args, **_kwargs: pytest.fail("array opened"))

    with pytest.raises(RuntimeError, match="authentication failed"):
        load_authenticated_trainval(ExperimentConfig())
