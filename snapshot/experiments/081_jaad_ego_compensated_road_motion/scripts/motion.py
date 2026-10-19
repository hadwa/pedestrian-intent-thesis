"""Deterministic sparse-flow visual motion estimation for Exp081."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class MotionWindow:
    pair_transforms: np.ndarray
    to_first: np.ndarray
    proxy: np.ndarray
    valid: np.ndarray
    tracked_point_count: np.ndarray
    inlier_fraction: np.ndarray
    forward_backward_error: np.ndarray
    background_displacement: np.ndarray
    residual_displacement: np.ndarray


@dataclass(frozen=True)
class PairMotion:
    transform: np.ndarray
    valid: bool
    tracked_point_count: int
    inlier_fraction: float
    forward_backward_error: float
    background_displacement: float
    residual_displacement: float


def background_mask(
    shape: tuple[int, int], boxes: np.ndarray, expansion_fraction: float
) -> np.ndarray:
    """Create a Shi-Tomasi mask with expanded pedestrian regions removed."""
    height, width = (int(shape[0]), int(shape[1]))
    mask = np.full((height, width), 255, dtype=np.uint8)
    for box in np.asarray(boxes, dtype=float).reshape((-1, 4)):
        x1, y1, x2, y2 = box
        box_width = max(0.0, x2 - x1)
        box_height = max(0.0, y2 - y1)
        left = max(0, int(np.floor(x1 - expansion_fraction * box_width)))
        top = max(0, int(np.floor(y1 - expansion_fraction * box_height)))
        right = min(width, int(np.ceil(x2 + expansion_fraction * box_width)))
        bottom = min(height, int(np.ceil(y2 + expansion_fraction * box_height)))
        if right > left and bottom > top:
            mask[top:bottom, left:right] = 0
    return mask


def _setting(settings: Mapping[str, Any], group: str, name: str) -> Any:
    value = settings.get(group)
    if isinstance(value, Mapping) and name in value:
        return value[name]
    if name in settings:
        return settings[name]
    raise KeyError(f"Missing Exp081 motion setting: {group}.{name}")


def _identity() -> np.ndarray:
    return np.eye(3, dtype=np.float64)


def _points_in_mask(points: np.ndarray, mask: np.ndarray) -> np.ndarray:
    rounded = np.rint(points).astype(int)
    height, width = mask.shape
    inside = (
        (rounded[:, 0] >= 0)
        & (rounded[:, 0] < width)
        & (rounded[:, 1] >= 0)
        & (rounded[:, 1] < height)
    )
    allowed = np.zeros(len(points), dtype=bool)
    valid_indices = np.flatnonzero(inside)
    allowed[valid_indices] = mask[
        rounded[valid_indices, 1], rounded[valid_indices, 0]
    ] > 0
    return allowed


def _estimate_pair(
    prior_gray: np.ndarray,
    current_gray: np.ndarray,
    prior_boxes: np.ndarray,
    current_boxes: np.ndarray,
    settings: Mapping[str, Any],
) -> tuple[np.ndarray, bool, int, float, float, float, float]:
    expansion = float(settings["bbox_mask_expansion_fraction"])
    prior_mask = background_mask(prior_gray.shape, prior_boxes, expansion)
    current_mask = background_mask(current_gray.shape, current_boxes, expansion)
    detector = settings["feature_detector"]
    prior = cv2.goodFeaturesToTrack(
        prior_gray,
        mask=prior_mask,
        maxCorners=int(detector["maxCorners"]),
        qualityLevel=float(detector["qualityLevel"]),
        minDistance=float(detector["minDistance"]),
    )
    if prior is None or len(prior) < int(settings["minimum_correspondences"]):
        return _identity(), False, 0, 0.0, 0.0, 0.0, 0.0

    lk = settings["lk"]
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        30,
        0.01,
    )
    lk_kwargs = {
        "winSize": tuple(int(value) for value in lk["winSize"]),
        "maxLevel": int(lk["maxLevel"]),
        "criteria": criteria,
    }
    current, forward_status, _ = cv2.calcOpticalFlowPyrLK(
        prior_gray, current_gray, prior, None, **lk_kwargs
    )
    if current is None or forward_status is None:
        return _identity(), False, 0, 0.0, 0.0, 0.0, 0.0
    backward, backward_status, _ = cv2.calcOpticalFlowPyrLK(
        current_gray, prior_gray, current, None, **lk_kwargs
    )
    if backward is None or backward_status is None:
        return _identity(), False, 0, 0.0, 0.0, 0.0, 0.0

    prior_xy = prior.reshape((-1, 2)).astype(np.float64)
    current_xy = current.reshape((-1, 2)).astype(np.float64)
    backward_xy = backward.reshape((-1, 2)).astype(np.float64)
    errors = np.linalg.norm(prior_xy - backward_xy, axis=1)
    keep = (
        forward_status.reshape(-1).astype(bool)
        & backward_status.reshape(-1).astype(bool)
        & np.isfinite(prior_xy).all(axis=1)
        & np.isfinite(current_xy).all(axis=1)
        & np.isfinite(errors)
        & (errors <= float(settings["forward_backward_max_error_px"]))
        & _points_in_mask(prior_xy, prior_mask)
        & _points_in_mask(current_xy, current_mask)
    )
    prior_good = prior_xy[keep]
    current_good = current_xy[keep]
    tracked_count = len(prior_good)
    mean_error = float(errors[keep].mean()) if tracked_count else 0.0
    if tracked_count < int(settings["minimum_correspondences"]):
        return _identity(), False, tracked_count, 0.0, mean_error, 0.0, 0.0

    robust = settings["robust_fit"]
    cv2.setRNGSeed(0)
    affine, inliers = cv2.estimateAffinePartial2D(
        prior_good,
        current_good,
        method=cv2.RANSAC,
        ransacReprojThreshold=float(robust["reprojection_threshold_px"]),
        maxIters=int(robust["maxIters"]),
        confidence=float(robust["confidence"]),
        refineIters=int(robust["refineIters"]),
    )
    if affine is None or not np.isfinite(affine).all():
        return _identity(), False, tracked_count, 0.0, mean_error, 0.0, 0.0
    homogeneous = _identity()
    homogeneous[:2] = affine
    determinant = float(np.linalg.det(homogeneous))
    scale = float(np.hypot(homogeneous[0, 0], homogeneous[1, 0]))
    if not np.isfinite(determinant) or abs(determinant) < 1e-9 or scale <= 0.0:
        return _identity(), False, tracked_count, 0.0, mean_error, 0.0, 0.0
    inlier_fraction = float(np.asarray(inliers, dtype=float).mean()) if inliers is not None else 0.0
    inlier_mask = (
        np.asarray(inliers).reshape(-1).astype(bool)
        if inliers is not None
        else np.ones(tracked_count, dtype=bool)
    )
    source_inliers = prior_good[inlier_mask]
    target_inliers = current_good[inlier_mask]
    predicted = (
        homogeneous[:2, :2] @ source_inliers.T
    ).T + homogeneous[:2, 2]
    background_displacement = float(
        np.median(np.linalg.norm(target_inliers - source_inliers, axis=1))
    )
    residual_displacement = float(
        np.median(np.linalg.norm(target_inliers - predicted, axis=1))
    )
    return (
        homogeneous,
        True,
        tracked_count,
        inlier_fraction,
        mean_error,
        background_displacement,
        residual_displacement,
    )


def _motion_proxy(
    pair_transforms: np.ndarray,
    valid: np.ndarray,
    inlier_fraction: np.ndarray,
    final_cumulative: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    channels = np.empty((15, 4), dtype=np.float64)
    channels[:, 0] = pair_transforms[:, 0, 2] / float(width)
    channels[:, 1] = pair_transforms[:, 1, 2] / float(height)
    channels[:, 2] = np.arctan2(pair_transforms[:, 1, 0], pair_transforms[:, 0, 0])
    scales = np.hypot(pair_transforms[:, 0, 0], pair_transforms[:, 1, 0])
    channels[:, 3] = np.log(np.maximum(scales, np.finfo(float).tiny))
    early = channels[:7].mean(axis=0)
    late = channels[7:].mean(axis=0)
    summaries = np.stack((early, late, late - early), axis=1).reshape(-1)
    valid_inliers = inlier_fraction[valid]
    final_translation = np.hypot(
        final_cumulative[0, 2] / float(width),
        final_cumulative[1, 2] / float(height),
    )
    proxy = np.concatenate(
        (
            summaries,
            np.asarray(
                [
                    valid.mean(),
                    valid_inliers.mean() if len(valid_inliers) else 0.0,
                    final_translation,
                ],
                dtype=np.float64,
            ),
        )
    )
    if proxy.shape != (15,) or not np.isfinite(proxy).all():
        raise RuntimeError("Exp081 visual motion proxy contract failed")
    return proxy


def estimate_window_motion(
    frames: Sequence[np.ndarray],
    pedestrian_boxes: Sequence[np.ndarray],
    settings: Mapping[str, Any],
) -> MotionWindow:
    """Estimate 15 prior-to-current transforms for one 16-frame window."""
    if len(frames) != 16 or len(pedestrian_boxes) != 16:
        raise ValueError("Exp081 motion estimation requires exactly 16 observations")
    shapes = {tuple(frame.shape[:2]) for frame in frames}
    if len(shapes) != 1:
        raise ValueError("Exp081 motion frames must share one image shape")
    height, width = next(iter(shapes))
    pairs = [
        estimate_pair_motion(
            frames[index],
            frames[index + 1],
            pedestrian_boxes[index],
            pedestrian_boxes[index + 1],
            settings,
        )
        for index in range(15)
    ]
    return assemble_window_motion(pairs, width=width, height=height)


def estimate_pair_motion(
    prior_frame: np.ndarray,
    current_frame: np.ndarray,
    prior_boxes: np.ndarray,
    current_boxes: np.ndarray,
    settings: Mapping[str, Any],
) -> PairMotion:
    """Estimate one reusable prior-to-current background transform."""
    prior = np.asarray(prior_frame)
    current = np.asarray(current_frame)
    if (
        prior.ndim != 3
        or current.ndim != 3
        or prior.shape != current.shape
        or prior.shape[2] != 3
    ):
        raise ValueError("Exp081 pair motion requires equal HxWx3 frames")
    values = _estimate_pair(
        cv2.cvtColor(prior, cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(current, cv2.COLOR_BGR2GRAY),
        prior_boxes,
        current_boxes,
        settings,
    )
    return PairMotion(
        transform=values[0],
        valid=values[1],
        tracked_point_count=values[2],
        inlier_fraction=values[3],
        forward_backward_error=values[4],
        background_displacement=values[5],
        residual_displacement=values[6],
    )


def assemble_window_motion(
    pairs: Sequence[PairMotion], *, width: int, height: int
) -> MotionWindow:
    """Assemble 15 cached pair estimates into the frozen window contract."""
    if len(pairs) != 15 or width <= 0 or height <= 0:
        raise ValueError("Exp081 window motion requires 15 pairs and positive geometry")
    pair_transforms = np.repeat(_identity()[None, :, :], 15, axis=0)
    valid = np.zeros(15, dtype=bool)
    tracked = np.zeros(15, dtype=np.int64)
    inliers = np.zeros(15, dtype=np.float64)
    errors = np.zeros(15, dtype=np.float64)
    background = np.zeros(15, dtype=np.float64)
    residual = np.zeros(15, dtype=np.float64)
    for index, pair in enumerate(pairs):
        pair_transforms[index] = np.asarray(pair.transform, dtype=np.float64)
        valid[index] = bool(pair.valid)
        tracked[index] = int(pair.tracked_point_count)
        inliers[index] = float(pair.inlier_fraction)
        errors[index] = float(pair.forward_backward_error)
        background[index] = float(pair.background_displacement)
        residual[index] = float(pair.residual_displacement)

    to_first = np.repeat(_identity()[None, :, :], 16, axis=0)
    cumulative = _identity()
    for index, pair in enumerate(pair_transforms):
        cumulative = pair @ cumulative
        if not np.isfinite(cumulative).all() or abs(np.linalg.det(cumulative)) < 1e-9:
            raise RuntimeError("Exp081 cumulative motion transform is invalid")
        to_first[index + 1] = np.linalg.inv(cumulative)
    proxy = _motion_proxy(
        pair_transforms, valid, inliers, cumulative, width=int(width), height=int(height)
    )
    return MotionWindow(
        pair_transforms=pair_transforms,
        to_first=to_first,
        proxy=proxy,
        valid=valid,
        tracked_point_count=tracked,
        inlier_fraction=inliers,
        forward_backward_error=errors,
        background_displacement=background,
        residual_displacement=residual,
    )


def stabilize_points(points: np.ndarray, to_first: np.ndarray) -> np.ndarray:
    """Map 2-D points from their observation frame into frame-1 coordinates."""
    values = np.asarray(points, dtype=np.float64)
    original_shape = values.shape
    flat = values.reshape((-1, 2))
    homogeneous = np.column_stack((flat, np.ones(len(flat), dtype=np.float64)))
    mapped = (np.asarray(to_first, dtype=np.float64) @ homogeneous.T).T
    if np.any(np.abs(mapped[:, 2]) < 1e-12):
        raise RuntimeError("Exp081 point stabilization produced points at infinity")
    result = mapped[:, :2] / mapped[:, 2:3]
    if not np.isfinite(result).all():
        raise RuntimeError("Exp081 point stabilization produced nonfinite values")
    return result.reshape(original_shape)
