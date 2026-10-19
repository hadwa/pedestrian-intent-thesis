"""Frozen observation-only feature helpers for Exp037."""

from __future__ import annotations

import math
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TEMPORAL_BASES = (
    "lower_body_motion_magnitude_mean",
    "lead_ankle_forward_delta",
    "torso_motion_magnitude_mean",
    "torso_center_x_delta",
    "bbox_center_dx_norm",
    "bbox_scale_log_delta",
    "pose_confidence_mean",
)

OBD_COLUMNS = (
    "obd_speed_mean",
    "obd_speed_std",
    "obd_speed_first",
    "obd_speed_last",
    "obd_speed_delta",
)


def load_exp012_feature_contract(path: Path) -> dict[str, tuple[str, ...]]:
    definitions = pd.read_csv(Path(path))
    required = {"feature_set", "feature_columns_json"}
    if required.difference(definitions.columns) or definitions["feature_set"].duplicated().any():
        raise ValueError("Exp012 feature definition schema is invalid")
    rows = definitions.set_index("feature_set")
    names = ("bbox_only", "biu_v1_only", "biu_v2_core")
    if any(name not in rows.index for name in names):
        raise ValueError("Exp012 feature definition is missing a frozen family")
    contract = {
        name: tuple(json.loads(str(rows.loc[name, "feature_columns_json"]))) for name in names
    }
    expected = {"bbox_only": 53, "biu_v1_only": 72, "biu_v2_core": 56}
    for name, count in expected.items():
        if len(contract[name]) != count or len(set(contract[name])) != count:
            raise ValueError(f"Exp012 {name} feature contract changed")
    contract["joint"] = joint_feature_columns(
        contract["bbox_only"], contract["biu_v1_only"], contract["biu_v2_core"]
    )
    if len(contract["joint"]) != 161:
        raise ValueError("Exp037 joint feature contract changed")
    return contract


def joint_feature_columns(
    bbox_columns: Sequence[str], v1_columns: Sequence[str], v2_columns: Sequence[str]
) -> tuple[str, ...]:
    """Keep v1 semantics for overlaps and append only unique v2 names."""
    groups = [tuple(map(str, columns)) for columns in (bbox_columns, v1_columns, v2_columns)]
    if any(len(group) != len(set(group)) for group in groups):
        raise ValueError("feature contract contains duplicate columns")
    return tuple(sorted(set().union(*groups)))


def build_arm_columns(contract: Mapping[str, Sequence[str]]) -> dict[str, tuple[str, ...]]:
    bbox = tuple(map(str, contract["bbox_only"]))
    v1 = tuple(map(str, contract["biu_v1_only"]))
    v2 = tuple(map(str, contract["biu_v2_core"]))
    v1_semantics = set(bbox) | set(v1)
    joint = tuple(name if name in v1_semantics else f"v2__{name}" for name in contract["joint"])
    temporal = tuple(f"temporal__{base}_{stat}" for base in TEMPORAL_BASES for stat in ("early", "late", "late_minus_early"))
    columns = {
        "v2_bbox_xgb": bbox + tuple(f"v2__{name}" for name in v2),
        "v1_bbox_rf": bbox + v1,
        "joint_xgb": joint,
        "joint_rf": joint,
        "joint_temporal_xgb": joint + temporal,
        "joint_temporal_ego_xgb": joint + temporal + OBD_COLUMNS,
    }
    expected = {
        "v2_bbox_xgb": 109,
        "v1_bbox_rf": 125,
        "joint_xgb": 161,
        "joint_rf": 161,
        "joint_temporal_xgb": 182,
        "joint_temporal_ego_xgb": 187,
    }
    for arm, names in columns.items():
        if len(names) != expected[arm] or len(names) != len(set(names)):
            raise ValueError(f"Exp037 {arm} feature contract changed")
    return columns


def _bbox_summary(bboxes: np.ndarray) -> tuple[float, float]:
    if bboxes.ndim != 2 or bboxes.shape[1] != 4 or len(bboxes) < 2:
        raise ValueError("BBox sequence must contain at least two xyxy rows")
    if not np.isfinite(bboxes).all():
        raise ValueError("BBox sequence contains non-finite values")
    widths = bboxes[:, 2] - bboxes[:, 0]
    heights = bboxes[:, 3] - bboxes[:, 1]
    if np.any(widths <= 0) or np.any(heights <= 0):
        raise ValueError("BBox sequence contains non-positive dimensions")
    centers = (bboxes[:, 0] + bboxes[:, 2]) / 2.0
    center_dx_norm = float((centers[-1] - centers[0]) / widths.mean())
    areas = widths * heights
    scale_log_delta = float(math.log(max(float(areas[-1]), 1e-12) / max(float(areas[0]), 1e-12)))
    return center_dx_norm, scale_log_delta


def temporal_contrast_features(
    coords: np.ndarray,
    confidence: np.ndarray,
    bboxes: np.ndarray,
    *,
    extractor: Callable[..., Mapping[str, Any]],
) -> dict[str, float]:
    coords = np.asarray(coords, dtype=float)
    confidence = np.asarray(confidence, dtype=float)
    bboxes = np.asarray(bboxes, dtype=float)
    if coords.shape != (16, 17, 2) or confidence.shape != (16, 17) or bboxes.shape != (16, 4):
        raise ValueError("temporal contrasts require exact 16-frame COCO-17 observations")
    if not np.isfinite(coords).all() or not np.isfinite(confidence).all():
        raise ValueError("pose observation contains non-finite values")

    by_half: dict[str, dict[str, float]] = {}
    for name, indices in (("early", slice(0, 8)), ("late", slice(8, 16))):
        part_coords = coords[indices]
        part_conf = confidence[indices]
        part_bboxes = bboxes[indices]
        extracted = extractor(part_coords, part_conf, bbox_sequence=part_bboxes)
        center_dx, scale_delta = _bbox_summary(part_bboxes)
        values = {
            "lower_body_motion_magnitude_mean": float(extracted["lower_body_motion_magnitude_mean"]),
            "lead_ankle_forward_delta": float(extracted["lead_ankle_forward_delta"]),
            "torso_motion_magnitude_mean": float(extracted["torso_motion_magnitude_mean"]),
            "torso_center_x_delta": float(extracted["torso_center_x_delta"]),
            "bbox_center_dx_norm": center_dx,
            "bbox_scale_log_delta": scale_delta,
            "pose_confidence_mean": float(part_conf.mean()),
        }
        if not np.isfinite(list(values.values())).all():
            raise ValueError("temporal contrast produced a non-finite value")
        by_half[name] = values

    output: dict[str, float] = {}
    for base in TEMPORAL_BASES:
        early = by_half["early"][base]
        late = by_half["late"][base]
        output[f"{base}_early"] = early
        output[f"{base}_late"] = late
        output[f"{base}_late_minus_early"] = late - early
    return output


def obd_speed_features(speed_by_frame: Mapping[int, float], observed_frames: Sequence[int]) -> dict[str, float]:
    frames = tuple(int(frame) for frame in observed_frames)
    if len(frames) != 16 or frames != tuple(range(frames[0], frames[0] + 16)):
        raise ValueError("OBD features require 16 consecutive observed frames")
    missing = [frame for frame in frames if frame not in speed_by_frame]
    if missing:
        raise ValueError(f"missing exact observed OBD frame: {missing[0]}")
    values = np.asarray([speed_by_frame[frame] for frame in frames], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("OBD observation contains non-finite speed")
    return {
        "obd_speed_mean": float(values.mean()),
        "obd_speed_std": float(values.std(ddof=0)),
        "obd_speed_first": float(values[0]),
        "obd_speed_last": float(values[-1]),
        "obd_speed_delta": float(values[-1] - values[0]),
    }


def _validate_base_cohort(base: pd.DataFrame) -> None:
    required = {
        "sequence_id", "set_id", "track_id", "video_id", "label", "start_frame", "end_frame",
        "observation_length", "frame_step", "frame_sampling",
    }
    missing = required.difference(base.columns)
    if missing:
        raise ValueError(f"base cohort missing columns {sorted(missing)}")
    if base.empty or base["sequence_id"].duplicated().any():
        raise ValueError("base cohort sequence_id must be nonempty and unique")
    if base["track_id"].isnull().any() or not set(base["label"].astype(int)).issubset({0, 1}):
        raise ValueError("base cohort track or label is invalid")
    exact = (
        base["observation_length"].astype(int).eq(16)
        & base["frame_step"].astype(int).eq(1)
        & base["frame_sampling"].astype(str).eq("consecutive")
        & (base["end_frame"].astype(int) - base["start_frame"].astype(int)).eq(15)
    )
    if not exact.all():
        raise ValueError("base cohort is not exact 16-frame consecutive observation")


def exact_feature_join(base: pd.DataFrame, features: pd.DataFrame, *, name: str) -> pd.DataFrame:
    _validate_base_cohort(base)
    if "sequence_id" not in features or features["sequence_id"].duplicated().any():
        raise ValueError(f"{name} sequence_id must be unique")
    if set(base["sequence_id"].astype(str)) != set(features["sequence_id"].astype(str)):
        raise ValueError(f"{name} exact cohort mismatch")
    overlap = set(base.columns).intersection(features.columns).difference({"sequence_id"})
    if overlap:
        raise ValueError(f"{name} duplicates base columns {sorted(overlap)}")
    order = base["sequence_id"].astype(str).tolist()
    merged = base.merge(features, on="sequence_id", how="left", validate="one_to_one", sort=False)
    if merged["sequence_id"].astype(str).tolist() != order:
        raise ValueError(f"{name} changed canonical cohort order")
    return merged
