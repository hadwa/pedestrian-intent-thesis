from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _data_module():
    return importlib.import_module(
        "experiments.063_pie_corrected_yolop_road_relative_context.scripts.data"
    )


def _model_module():
    return importlib.import_module(
        "experiments.063_pie_corrected_yolop_road_relative_context.scripts.model"
    )


def test_align_scene_parts_restores_manifest_order(tmp_path: Path) -> None:
    data = _data_module()
    rows = pd.DataFrame(
        {
            "sequence_id": ("s2", "s1"),
            "set_id": ("set01", "set01"),
            "video_id": ("video_0001", "video_0001"),
        }
    )
    grid = np.stack(
        (
            np.full((16, 20), 1.0, np.float32),
            np.full((16, 20), 2.0, np.float32),
        )
    )
    road = grid + 10
    with (tmp_path / "set01_video_0001.npz").open("wb") as handle:
        np.savez_compressed(
            handle,
            sequence_id=np.asarray(("s1", "s2")),
            corrected_grid=grid,
            road_relative=road,
        )
    aligned = data.align_scene_parts(rows, tmp_path, "corrected_grid")
    assert aligned.shape == (2, 16, 20)
    np.testing.assert_allclose(aligned[0], 2.0)
    np.testing.assert_allclose(aligned[1], 1.0)


def test_align_scene_parts_rejects_duplicate_or_missing_ids(tmp_path: Path) -> None:
    data = _data_module()
    rows = pd.DataFrame(
        {
            "sequence_id": ("s1", "s2"),
            "set_id": ("set01", "set01"),
            "video_id": ("video_0001", "video_0001"),
        }
    )
    values = np.zeros((2, 16, 20), np.float32)
    with (tmp_path / "set01_video_0001.npz").open("wb") as handle:
        np.savez_compressed(
            handle,
            sequence_id=np.asarray(("s1", "s1")),
            corrected_grid=values,
            road_relative=values,
        )
    with pytest.raises(ValueError, match="duplicate|identity"):
        data.align_scene_parts(rows, tmp_path, "road_relative")


def test_scaler_uses_training_rows_only() -> None:
    data = _data_module()
    ego = np.zeros((4, 16, 3), dtype=np.float32)
    scene = np.zeros((4, 16, 20), dtype=np.float32)
    ego[:2] = 2.0
    ego[2:] = 100.0
    scene[:2] = np.arange(20, dtype=np.float32)
    scene[2:] = 1000.0
    train_mask = np.asarray((True, True, False, False))
    scaler = data.fit_context_scaler(ego, scene, train_mask)
    np.testing.assert_allclose(scaler.ego_mean, 2.0)
    np.testing.assert_allclose(scaler.scene_mean, np.arange(20, dtype=np.float32))
    transformed_ego, transformed_scene = data.transform_context(ego, scene, scaler)
    np.testing.assert_allclose(transformed_ego[:2], 0.0)
    np.testing.assert_allclose(transformed_scene[:2], 0.0)


def test_model_is_exact_authenticated_exp062_5932_parameter_source() -> None:
    model_module = _model_module()
    model_module.authenticate_model_source()
    model = model_module.HybridContextCorrector()
    assert sum(parameter.numel() for parameter in model.parameters()) == 5932


def test_zero_initialized_model_reproduces_anchor_probabilities() -> None:
    torch = pytest.importorskip("torch")
    model_module = _model_module()
    model = model_module.HybridContextCorrector().eval()
    anchors = torch.tensor((0.2, 0.7), dtype=torch.float32)
    with torch.no_grad():
        output = model(
            torch.zeros(2, 16, 3),
            torch.zeros(2, 16, 20),
            anchors,
        )
    np.testing.assert_allclose(torch.sigmoid(output.final_logit).numpy(), anchors.numpy(), atol=1e-7)
