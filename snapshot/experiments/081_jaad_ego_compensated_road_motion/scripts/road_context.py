"""Matched uncompensated and ego-compensated road features for Exp081."""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Sequence

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
ACTIVE_ROOT = SCRIPT_DIR.parents[2]


def _load_contract():
    path = SCRIPT_DIR / "contract.py"
    spec = importlib.util.spec_from_file_location("exp081_contract_for_road", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load Exp081 contract from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CONTRACT = _load_contract()
REPRESENTATIONS_PATH = CONTRACT.resolve_source("exp063_representations")
CONTRACT.resolve_source("exp063_contracts")
if str(ACTIVE_ROOT) not in sys.path:
    sys.path.insert(0, str(ACTIVE_ROOT))
REPRESENTATIONS = importlib.import_module(
    "experiments.063_pie_corrected_yolop_road_relative_context.scripts.representations"
)
if Path(REPRESENTATIONS.__file__).resolve() != REPRESENTATIONS_PATH:
    raise RuntimeError("Authenticated Exp063 representation import drifted")

ROAD_CHANNELS = tuple(CONTRACT.load_registration()["road_features"]["channels"])
if ROAD_CHANNELS != tuple(REPRESENTATIONS.ROAD_RELATIVE_CHANNELS):
    raise RuntimeError("Exp081 road channel ABI differs from authenticated Exp063")
ROAD_INDEX = {name: position for position, name in enumerate(ROAD_CHANNELS)}


def index(name: str) -> int:
    return ROAD_INDEX[name]


def _validate_inputs(
    drivable_masks: Sequence[np.ndarray],
    lane_masks: Sequence[np.ndarray],
    bboxes: np.ndarray,
) -> np.ndarray:
    values = np.asarray(bboxes, dtype=np.float64)
    if len(drivable_masks) != 16 or len(lane_masks) != 16 or values.shape != (16, 4):
        raise ValueError("Exp081 road input must contain 16 masks, lanes, and BBoxes")
    if not np.isfinite(values).all() or np.any(values[:, 2:] <= values[:, :2]):
        raise ValueError("Exp081 road BBoxes must be finite positive XYXY values")
    return values


def uncompensated_sequence(
    drivable_masks: Sequence[np.ndarray],
    lane_masks: Sequence[np.ndarray],
    bboxes: np.ndarray,
) -> np.ndarray:
    """Return the authenticated Exp063 sequence without numerical changes."""
    values = _validate_inputs(drivable_masks, lane_masks, bboxes)
    output = REPRESENTATIONS.road_relative_sequence(
        drivable_masks, lane_masks, values
    )
    result = np.asarray(output, dtype=np.float32)
    if result.shape != (16, 20) or not np.isfinite(result).all():
        raise RuntimeError("Exp081 uncompensated road contract failed")
    return result


def static_features(
    drivable_mask: np.ndarray, lane_mask: np.ndarray, bbox: np.ndarray
) -> np.ndarray:
    """Reduce one frame immediately to the authenticated Exp063 static 13-D ABI."""
    values = REPRESENTATIONS.road_relative_sequence(
        [drivable_mask], [lane_mask], np.asarray(bbox, dtype=np.float64)[None, :]
    )[0, :13]
    output = np.asarray(values, dtype=np.float32)
    if output.shape != (13,) or not np.isfinite(output).all():
        raise RuntimeError("Exp081 static road geometry contract failed")
    return output


def assemble_uncompensated(static: np.ndarray, bboxes: np.ndarray) -> np.ndarray:
    values = REPRESENTATIONS.assemble_road_relative_temporal(static, bboxes)
    output = np.asarray(values, dtype=np.float32)
    if output.shape != (16, 20) or not np.isfinite(output).all():
        raise RuntimeError("Exp081 assembled road contract failed")
    return output


def _map_points(points: np.ndarray, transforms: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack((points, np.ones(len(points), dtype=np.float64)))
    mapped = np.einsum("nij,nj->ni", transforms, homogeneous)
    if np.any(np.abs(mapped[:, 2]) < 1e-12):
        raise RuntimeError("Exp081 compensated foot point mapped to infinity")
    result = mapped[:, :2] / mapped[:, 2:3]
    if not np.isfinite(result).all():
        raise RuntimeError("Exp081 compensated foot points are nonfinite")
    return result


def _map_unit_vector(vector: np.ndarray, transform: np.ndarray) -> np.ndarray | None:
    mapped = transform[:2, :2] @ np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(mapped))
    return mapped / norm if norm > 1e-12 else None


def compensated_sequence(
    drivable_masks: Sequence[np.ndarray],
    lane_masks: Sequence[np.ndarray],
    bboxes: np.ndarray,
    to_first: np.ndarray,
) -> np.ndarray:
    """Replace only Exp063 motion channels using stabilized target feet."""
    bbox_values = _validate_inputs(drivable_masks, lane_masks, bboxes)
    transforms = np.asarray(to_first, dtype=np.float64)
    if transforms.shape != (16, 3, 3) or not np.isfinite(transforms).all():
        raise ValueError("Exp081 to-first transforms must be finite 16x3x3")
    raw = uncompensated_sequence(drivable_masks, lane_masks, bbox_values)
    return assemble_compensated(raw[:, :13], bbox_values, transforms, raw=raw)


def assemble_compensated(
    static_features: np.ndarray,
    bboxes: np.ndarray,
    to_first: np.ndarray,
    *,
    raw: np.ndarray | None = None,
) -> np.ndarray:
    """Assemble compensated temporal channels from compact static geometry."""
    bbox_values = np.asarray(bboxes, dtype=np.float64)
    static = np.asarray(static_features, dtype=np.float32)
    transforms = np.asarray(to_first, dtype=np.float64)
    if (
        static.shape != (16, 13)
        or bbox_values.shape != (16, 4)
        or transforms.shape != (16, 3, 3)
        or not np.isfinite(transforms).all()
    ):
        raise ValueError("Exp081 compact compensated inputs changed shape")
    if raw is None:
        raw = assemble_uncompensated(static, bbox_values)
    else:
        raw = np.asarray(raw, dtype=np.float32)
        if raw.shape != (16, 20):
            raise ValueError("Exp081 raw road sequence changed shape")
    output = np.zeros_like(raw)
    output[:, :13] = raw[:, :13]
    output[:, ROAD_INDEX["road_signed_distance_change_h"]] = raw[
        :, ROAD_INDEX["road_signed_distance_change_h"]
    ]

    feet = np.column_stack(
        ((bbox_values[:, 0] + bbox_values[:, 2]) / 2.0, bbox_values[:, 3])
    )
    stabilized_feet = _map_points(feet, transforms)
    for frame_index in range(1, 16):
        bbox_height = max(
            float(bbox_values[frame_index, 3] - bbox_values[frame_index, 1]), 1.0
        )
        velocity = (
            stabilized_feet[frame_index] - stabilized_feet[frame_index - 1]
        ) / bbox_height
        speed = float(np.linalg.norm(velocity))
        output[frame_index, ROAD_INDEX["foot_velocity_x_h"]] = velocity[0]
        output[frame_index, ROAD_INDEX["foot_velocity_y_h"]] = velocity[1]
        output[frame_index, ROAD_INDEX["foot_speed_h"]] = speed
        if speed <= 1e-12:
            continue
        unit_velocity = velocity / speed
        static = raw[frame_index, :13].astype(np.float64)
        if bool(static[ROAD_INDEX["road_boundary_valid"]]):
            tangent = _map_unit_vector(static[4:6], transforms[frame_index])
            offset = transforms[frame_index, :2, :2] @ static[2:4]
            if tangent is not None:
                output[frame_index, ROAD_INDEX["motion_parallel_to_road"]] = abs(
                    float(np.dot(unit_velocity, tangent))
                )
                normal = np.asarray([-tangent[1], tangent[0]])
                target = -offset if static[0] > 0.5 else offset
                if float(np.dot(normal, target)) < 0.0:
                    normal *= -1.0
                if np.linalg.norm(target) > 1e-12:
                    output[
                        frame_index, ROAD_INDEX["motion_toward_drivable_area"]
                    ] = float(np.dot(unit_velocity, normal))
        if bool(static[ROAD_INDEX["lane_geometry_valid"]]):
            lane_tangent = _map_unit_vector(static[10:12], transforms[frame_index])
            if lane_tangent is not None:
                output[frame_index, ROAD_INDEX["motion_parallel_to_lane"]] = abs(
                    float(np.dot(unit_velocity, lane_tangent))
                )
    if output.shape != (16, 20) or not np.isfinite(output).all():
        raise RuntimeError("Exp081 compensated road contract failed")
    return output


def context_columns() -> tuple[str, ...]:
    return tuple(
        name
        for channel in ROAD_CHANNELS
        for name in (
            f"road__{channel}__early_mean",
            f"road__{channel}__late_mean",
            f"road__{channel}__late_minus_early",
        )
    )


def summarize(sequence: np.ndarray) -> np.ndarray:
    values = np.asarray(sequence, dtype=np.float32)
    if values.shape != (16, 20) or not np.isfinite(values).all():
        raise ValueError("Exp081 road-relative sequence must be finite 16x20")
    early = values[:8].mean(axis=0)
    late = values[8:].mean(axis=0)
    output = np.column_stack((early, late, late - early)).reshape(-1)
    if output.shape != (60,) or not np.isfinite(output).all():
        raise RuntimeError("Exp081 road summary contract failed")
    return output
