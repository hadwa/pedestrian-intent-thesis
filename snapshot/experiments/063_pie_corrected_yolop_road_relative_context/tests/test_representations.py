from __future__ import annotations

import importlib

import numpy as np
import pytest


contracts = importlib.import_module(
    "experiments.063_pie_corrected_yolop_road_relative_context.scripts.contracts"
)


def _module():
    return importlib.import_module(
        "experiments.063_pie_corrected_yolop_road_relative_context.scripts.representations"
    )


def _box(cx: float, bottom: float, height: float = 20.0, width: float = 10.0):
    return np.asarray(
        [cx - width / 2, bottom - height, cx + width / 2, bottom],
        dtype=np.float32,
    )


def test_corrected_grid_uses_foreground_fraction_and_historical_order() -> None:
    representations = _module()
    road = np.ones((100, 100), dtype=np.uint8)
    lane = np.zeros((100, 100), dtype=np.uint8)
    detections = np.asarray(
        [
            [45, 65, 55, 75, 0.9, 0],
            [40, 60, 60, 80, 0.9, 2],
        ],
        dtype=np.float32,
    )
    result = representations.corrected_grid_features(
        road, lane, detections, _box(50, 70)
    )
    assert result.shape == (20,)
    np.testing.assert_allclose(result[:9], 1.0)
    np.testing.assert_allclose(result[9:18], 0.0)
    np.testing.assert_allclose(result[18:], (0.1, 0.1))


def test_motion_toward_vertical_road_boundary_is_positive() -> None:
    representations = _module()
    index = {name: i for i, name in enumerate(contracts.ROAD_RELATIVE_CHANNELS)}
    road = np.zeros((100, 100), dtype=np.uint8)
    road[:, 50:] = 1
    lane = np.zeros_like(road)
    bboxes = np.stack((_box(40, 70), _box(44, 70)))
    result = representations.road_relative_sequence(
        [road, road], [lane, lane], bboxes
    )
    assert result.shape == (2, 20)
    assert result[1, index["motion_toward_drivable_area"]] > 0.9
    assert result[1, index["motion_parallel_to_road"]] < 0.1
    assert result[1, index["road_signed_distance_change_h"]] > 0


def test_motion_parallel_to_vertical_road_boundary_is_detected() -> None:
    representations = _module()
    index = {name: i for i, name in enumerate(contracts.ROAD_RELATIVE_CHANNELS)}
    road = np.zeros((100, 100), dtype=np.uint8)
    road[:, 50:] = 1
    lane = np.zeros_like(road)
    bboxes = np.stack((_box(40, 65), _box(40, 70)))
    result = representations.road_relative_sequence(
        [road, road], [lane, lane], bboxes
    )
    assert result[1, index["motion_parallel_to_road"]] > 0.9
    assert abs(result[1, index["motion_toward_drivable_area"]]) < 0.1


def test_lane_tangent_and_parallel_motion_are_sign_invariant() -> None:
    representations = _module()
    index = {name: i for i, name in enumerate(contracts.ROAD_RELATIVE_CHANNELS)}
    road = np.ones((100, 100), dtype=np.uint8)
    lane = np.zeros_like(road)
    lane[20:90, 50] = 1
    bboxes = np.stack((_box(45, 60), _box(45, 65)))
    result = representations.road_relative_sequence(
        [road, road], [lane, lane], bboxes
    )
    assert result[1, index["lane_geometry_valid"]] == 1
    assert result[1, index["lane_tangent_y"]] > 0.9
    assert result[1, index["motion_parallel_to_lane"]] > 0.9


def test_first_frame_temporal_channels_are_zero() -> None:
    representations = _module()
    road = np.zeros((100, 100), dtype=np.uint8)
    road[:, 50:] = 1
    lane = np.zeros_like(road)
    result = representations.road_relative_sequence(
        [road, road], [lane, lane], np.stack((_box(40, 70), _box(42, 70)))
    )
    np.testing.assert_allclose(result[0, 13:20], 0.0)


def test_invalid_geometry_is_finite_zero_with_validity_flags() -> None:
    representations = _module()
    index = {name: i for i, name in enumerate(contracts.ROAD_RELATIVE_CHANNELS)}
    road = np.ones((50, 50), dtype=np.uint8)
    lane = np.zeros_like(road)
    result = representations.road_relative_sequence(
        [road], [lane], np.stack((_box(25, 40, height=10),))
    )
    assert np.isfinite(result).all()
    assert result[0, index["road_foot_inside"]] == 1
    assert result[0, index["road_boundary_valid"]] == 0
    assert result[0, index["lane_geometry_valid"]] == 0
    assert result[0, index["road_signed_boundary_distance_h"]] == 0


def test_inputs_must_have_matching_lengths_and_valid_bboxes() -> None:
    representations = _module()
    mask = np.zeros((20, 20), dtype=np.uint8)
    with pytest.raises(ValueError, match="matching lengths"):
        representations.road_relative_sequence([mask], [mask, mask], np.stack((_box(10, 15),)))
    with pytest.raises(ValueError, match="positive width and height"):
        representations.road_relative_sequence(
            [mask], [mask], np.asarray([[1, 1, 1, 1]], dtype=np.float32)
        )
