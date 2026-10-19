"""Frozen corrected-grid and road-relative 20-D context representations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .contracts import CORRECTED_GRID_CHANNELS, ROAD_RELATIVE_CHANNELS


GRID_SIZE = 3
VEHICLE_COCO_CLASSES = frozenset((2, 3, 5, 7))
PEDESTRIAN_COCO_CLASS = 0
COUNT_RADIUS_PIXELS = 15.0 * 30.0
MIN_GEOMETRY_PIXELS = 8
MIN_PCA_RATIO = 1.5

ROAD_INDEX = {name: index for index, name in enumerate(ROAD_RELATIVE_CHANNELS)}


@dataclass(frozen=True)
class LineGeometry:
    """Nearest offset and local line orientation within the frozen footprint."""

    valid: bool
    distance: float = 0.0
    offset: np.ndarray | None = None
    tangent: np.ndarray | None = None
    inward_normal: np.ndarray | None = None


def _binary_mask(mask: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(mask)
    if values.ndim != 2 or min(values.shape) <= 0:
        raise ValueError(f"{name} must be a nonempty 2-D mask")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains non-finite values")
    return values > 0


def _validate_bbox(bbox: Sequence[float]) -> np.ndarray:
    values = np.asarray(bbox, dtype=np.float64)
    if values.shape != (4,) or not np.isfinite(values).all():
        raise ValueError("BBox must contain four finite XYXY values")
    if values[2] <= values[0] or values[3] <= values[1]:
        raise ValueError("BBox must have positive width and height")
    return values


def _foot_and_height(bbox: Sequence[float]) -> tuple[np.ndarray, float]:
    values = _validate_bbox(bbox)
    return np.asarray([(values[0] + values[2]) / 2.0, values[3]]), float(
        values[3] - values[1]
    )


def _grid_features(mask: np.ndarray, foot: np.ndarray, bbox_height: float) -> np.ndarray:
    binary = _binary_mask(mask, "grid mask")
    cell = max(1, int(bbox_height))
    half = (GRID_SIZE * cell) // 2
    height, width = binary.shape
    centre_x, centre_y = int(foot[0]), int(foot[1])
    output = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
    for row in range(GRID_SIZE):
        for column in range(GRID_SIZE):
            x1 = int(np.clip(centre_x - half + column * cell, 0, width))
            y1 = int(np.clip(centre_y - half + row * cell, 0, height))
            x2 = int(np.clip(x1 + cell, 0, width))
            y2 = int(np.clip(y1 + cell, 0, height))
            if x2 > x1 and y2 > y1:
                output[row, column] = float(binary[y1:y2, x1:x2].mean())
    return output.reshape(-1)


def _nearby_counts(detections: np.ndarray | None, foot: np.ndarray) -> np.ndarray:
    if detections is None:
        return np.zeros(2, dtype=np.float32)
    values = np.asarray(detections, dtype=np.float64)
    if values.size == 0:
        return np.zeros(2, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] < 6 or not np.isfinite(values).all():
        raise ValueError("Detections must be a finite Nx6-or-wider array")
    vehicles = 0
    pedestrians = 0
    for detection in values:
        centre = np.asarray(
            [(detection[0] + detection[2]) / 2, (detection[1] + detection[3]) / 2]
        )
        if float(np.sum((centre - foot) ** 2)) >= COUNT_RADIUS_PIXELS**2:
            continue
        class_id = int(detection[5])
        if class_id in VEHICLE_COCO_CLASSES:
            vehicles += 1
        elif class_id == PEDESTRIAN_COCO_CLASS:
            pedestrians += 1
    return np.asarray(
        [min(vehicles, 10) / 10.0, min(pedestrians, 10) / 10.0],
        dtype=np.float32,
    )


def corrected_grid_features(
    drivable_mask: np.ndarray,
    lane_mask: np.ndarray,
    detections: np.ndarray | None,
    bbox: Sequence[float],
) -> np.ndarray:
    """Compose the corrected foreground-mask version of historical 9+9+2."""
    foot, bbox_height = _foot_and_height(bbox)
    output = np.concatenate(
        (
            _grid_features(drivable_mask, foot, bbox_height),
            _grid_features(lane_mask, foot, bbox_height),
            _nearby_counts(detections, foot),
        )
    ).astype(np.float32, copy=False)
    if output.shape != (len(CORRECTED_GRID_CHANNELS),) or not np.isfinite(output).all():
        raise RuntimeError("Corrected grid composition violated its 20-D ABI")
    if np.any(output < 0) or np.any(output > 1):
        raise RuntimeError("Corrected grid values must remain in [0, 1]")
    return output


def _footprint_slices(
    shape: tuple[int, int], foot: np.ndarray, bbox_height: float
) -> tuple[slice, slice]:
    side = max(1, int(3 * bbox_height))
    half = side // 2
    centre_x, centre_y = int(foot[0]), int(foot[1])
    x1 = int(np.clip(centre_x - half, 0, shape[1]))
    y1 = int(np.clip(centre_y - half, 0, shape[0]))
    x2 = int(np.clip(x1 + side, 0, shape[1]))
    y2 = int(np.clip(y1 + side, 0, shape[0]))
    return slice(y1, y2), slice(x1, x2)


def _boundary_pixels(binary: np.ndarray) -> np.ndarray:
    boundary = np.zeros_like(binary, dtype=bool)
    horizontal = binary[:, 1:] != binary[:, :-1]
    boundary[:, 1:] |= horizontal
    boundary[:, :-1] |= horizontal
    vertical = binary[1:, :] != binary[:-1, :]
    boundary[1:, :] |= vertical
    boundary[:-1, :] |= vertical
    return boundary


def _canonical_pca_tangent(points_xy: np.ndarray) -> np.ndarray | None:
    if len(points_xy) < MIN_GEOMETRY_PIXELS:
        return None
    centred = points_xy - points_xy.mean(axis=0, keepdims=True)
    covariance = centred.T @ centred / max(len(points_xy) - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    secondary = max(float(eigenvalues[0]), 1e-12)
    if float(eigenvalues[1]) / secondary < MIN_PCA_RATIO:
        return None
    tangent = eigenvectors[:, 1].astype(np.float64)
    if tangent[0] < 0 or (abs(tangent[0]) <= 1e-12 and tangent[1] < 0):
        tangent *= -1
    norm = float(np.linalg.norm(tangent))
    return tangent / norm if norm > 0 else None


def _probe_occupancy(binary: np.ndarray, point_xy: np.ndarray) -> float:
    x = int(round(float(point_xy[0])))
    y = int(round(float(point_xy[1])))
    x1, x2 = max(0, x - 1), min(binary.shape[1], x + 2)
    y1, y2 = max(0, y - 1), min(binary.shape[0], y + 2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return float(binary[y1:y2, x1:x2].mean())


def _road_geometry(
    road: np.ndarray, foot: np.ndarray, bbox_height: float
) -> LineGeometry:
    row_slice, column_slice = _footprint_slices(road.shape, foot, bbox_height)
    footprint = road[row_slice, column_slice]
    if footprint.size == 0 or footprint.all() or not footprint.any():
        return LineGeometry(False)
    boundary = _boundary_pixels(road)
    local_y, local_x = np.nonzero(boundary[row_slice, column_slice])
    if len(local_x) < MIN_GEOMETRY_PIXELS:
        return LineGeometry(False)
    points = np.column_stack(
        (local_x + int(column_slice.start), local_y + int(row_slice.start))
    ).astype(np.float64)
    distances = np.linalg.norm(points - foot, axis=1)
    nearest_index = int(np.argmin(distances))
    nearest = points[nearest_index]
    radius = max(4.0, 0.5 * bbox_height)
    tangent = _canonical_pca_tangent(points[np.linalg.norm(points - nearest, axis=1) <= radius])
    if tangent is None:
        return LineGeometry(False)
    normal_a = np.asarray([-tangent[1], tangent[0]])
    probe_distance = max(2.0, 0.1 * bbox_height)
    occupancy_a = _probe_occupancy(road, nearest + normal_a * probe_distance)
    occupancy_b = _probe_occupancy(road, nearest - normal_a * probe_distance)
    if abs(occupancy_a - occupancy_b) <= 1e-9:
        return LineGeometry(False)
    inward = normal_a if occupancy_a > occupancy_b else -normal_a
    return LineGeometry(
        True,
        distance=float(distances[nearest_index]),
        offset=nearest - foot,
        tangent=tangent,
        inward_normal=inward,
    )


def _lane_geometry(
    lane: np.ndarray, foot: np.ndarray, bbox_height: float
) -> LineGeometry:
    row_slice, column_slice = _footprint_slices(lane.shape, foot, bbox_height)
    local_y, local_x = np.nonzero(lane[row_slice, column_slice])
    if len(local_x) < MIN_GEOMETRY_PIXELS:
        return LineGeometry(False)
    points = np.column_stack(
        (local_x + int(column_slice.start), local_y + int(row_slice.start))
    ).astype(np.float64)
    tangent = _canonical_pca_tangent(points)
    if tangent is None:
        return LineGeometry(False)
    distances = np.linalg.norm(points - foot, axis=1)
    nearest_index = int(np.argmin(distances))
    return LineGeometry(
        True,
        distance=float(distances[nearest_index]),
        offset=points[nearest_index] - foot,
        tangent=tangent,
    )


def road_relative_sequence(
    drivable_masks: Sequence[np.ndarray],
    lane_masks: Sequence[np.ndarray],
    bboxes: np.ndarray,
) -> np.ndarray:
    """Build the frozen road-relative sequence from corrected masks and BBoxes."""
    bbox_values = np.asarray(bboxes, dtype=np.float64)
    if len(drivable_masks) != len(lane_masks) or len(drivable_masks) != len(bbox_values):
        raise ValueError("Road, lane, and BBox sequences must have matching lengths")
    if len(bbox_values) == 0 or bbox_values.ndim != 2 or bbox_values.shape[1] != 4:
        raise ValueError("BBox sequence must be nonempty Nx4")
    output = np.zeros((len(bbox_values), len(ROAD_RELATIVE_CHANNELS)), dtype=np.float32)
    for frame_index, (road_mask, lane_mask, bbox) in enumerate(
        zip(drivable_masks, lane_masks, bbox_values)
    ):
        road = _binary_mask(road_mask, "drivable mask")
        lane = _binary_mask(lane_mask, "lane mask")
        if road.shape != lane.shape:
            raise ValueError("Drivable and lane masks must share exact frame shape")
        foot, bbox_height = _foot_and_height(bbox)
        scale = max(bbox_height, 1.0)
        sample_x = int(np.clip(round(float(foot[0])), 0, road.shape[1] - 1))
        sample_y = int(np.clip(round(float(foot[1])), 0, road.shape[0] - 1))
        foot_inside = float(road[sample_y, sample_x])
        output[frame_index, ROAD_INDEX["road_foot_inside"]] = foot_inside

        road_geometry = _road_geometry(road, foot, bbox_height)
        signed_distance = 0.0
        if road_geometry.valid:
            signed_distance = road_geometry.distance / scale * (1.0 if foot_inside else -1.0)
            output[frame_index, ROAD_INDEX["road_signed_boundary_distance_h"]] = signed_distance
            output[frame_index, ROAD_INDEX["road_nearest_boundary_dx_h"]] = road_geometry.offset[0] / scale
            output[frame_index, ROAD_INDEX["road_nearest_boundary_dy_h"]] = road_geometry.offset[1] / scale
            output[frame_index, ROAD_INDEX["road_boundary_tangent_x"]] = road_geometry.tangent[0]
            output[frame_index, ROAD_INDEX["road_boundary_tangent_y"]] = road_geometry.tangent[1]
            output[frame_index, ROAD_INDEX["road_boundary_valid"]] = 1.0

        lane_geometry = _lane_geometry(lane, foot, bbox_height)
        if lane_geometry.valid:
            output[frame_index, ROAD_INDEX["lane_nearest_distance_h"]] = lane_geometry.distance / scale
            output[frame_index, ROAD_INDEX["lane_nearest_dx_h"]] = lane_geometry.offset[0] / scale
            output[frame_index, ROAD_INDEX["lane_nearest_dy_h"]] = lane_geometry.offset[1] / scale
            output[frame_index, ROAD_INDEX["lane_tangent_x"]] = lane_geometry.tangent[0]
            output[frame_index, ROAD_INDEX["lane_tangent_y"]] = lane_geometry.tangent[1]
            output[frame_index, ROAD_INDEX["lane_geometry_valid"]] = 1.0

    return assemble_road_relative_temporal(output[:, :13], bbox_values)


def assemble_road_relative_temporal(
    static_features: np.ndarray, bboxes: np.ndarray
) -> np.ndarray:
    """Add observation-window temporal relations to the 13 static channels."""
    static = np.asarray(static_features, dtype=np.float64)
    bbox_values = np.asarray(bboxes, dtype=np.float64)
    if static.ndim != 2 or static.shape[1] != 13:
        raise ValueError("Road-relative static features must be an Nx13 array")
    if bbox_values.shape != (len(static), 4):
        raise ValueError("Static features and BBoxes must have matching Nx4 rows")
    output = np.zeros((len(static), len(ROAD_RELATIVE_CHANNELS)), dtype=np.float32)
    output[:, :13] = static
    for frame_index in range(1, len(static)):
        current_foot, bbox_height = _foot_and_height(bbox_values[frame_index])
        previous_foot, _ = _foot_and_height(bbox_values[frame_index - 1])
        velocity = (current_foot - previous_foot) / max(bbox_height, 1.0)
        speed = float(np.linalg.norm(velocity))
        output[frame_index, ROAD_INDEX["foot_velocity_x_h"]] = velocity[0]
        output[frame_index, ROAD_INDEX["foot_velocity_y_h"]] = velocity[1]
        output[frame_index, ROAD_INDEX["foot_speed_h"]] = speed
        road_valid = bool(static[frame_index, ROAD_INDEX["road_boundary_valid"]])
        lane_valid = bool(static[frame_index, ROAD_INDEX["lane_geometry_valid"]])
        if speed > 1e-12:
            unit_velocity = velocity / speed
            if road_valid:
                tangent = static[frame_index, 4:6]
                output[frame_index, ROAD_INDEX["motion_parallel_to_road"]] = abs(
                    float(np.dot(unit_velocity, tangent))
                )
                normal = np.asarray([-tangent[1], tangent[0]])
                offset = static[frame_index, 2:4]
                target = -offset if static[frame_index, 0] > 0.5 else offset
                if float(np.dot(normal, target)) < 0:
                    normal *= -1
                if np.linalg.norm(target) > 1e-12:
                    output[
                        frame_index, ROAD_INDEX["motion_toward_drivable_area"]
                    ] = float(np.dot(unit_velocity, normal))
            if lane_valid:
                lane_tangent = static[frame_index, 10:12]
                output[frame_index, ROAD_INDEX["motion_parallel_to_lane"]] = abs(
                    float(np.dot(unit_velocity, lane_tangent))
                )
        if road_valid and bool(static[frame_index - 1, ROAD_INDEX["road_boundary_valid"]]):
            output[frame_index, ROAD_INDEX["road_signed_distance_change_h"]] = (
                static[frame_index, ROAD_INDEX["road_signed_boundary_distance_h"]]
                - static[frame_index - 1, ROAD_INDEX["road_signed_boundary_distance_h"]]
            )
    if not np.isfinite(output).all():
        raise RuntimeError("Road-relative representation contains non-finite values")
    return output


__all__ = [
    "assemble_road_relative_temporal",
    "corrected_grid_features",
    "road_relative_sequence",
]
