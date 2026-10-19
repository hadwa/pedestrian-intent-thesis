"""BIU feature extraction aligned to frozen benchmark sequence_id files.

This module re-implements the active BIU formulas from the old BIU repository in a
self-contained form so the new repository does not import or modify old code.
Feature tables are generated only from frozen sequence CSVs; this module never
regenerates windows, labels, or splits.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SEQUENCE_META_COLUMNS = [
    "sequence_id",
    "split",
    "label",
    "set_id",
    "video_id",
    "ped_id",
    "track_id",
    "start_frame",
    "end_frame",
]

BIU_BASE_FEATURES = [
    "torso_lean_angle",
    "stride_length",
    "head_yaw_angle",
    "lead_foot_elevation",
    "lead_foot_lifted",
    "arm_swing_asymmetry",
    "gaze_to_car_angle",
    "body_yaw_angle",
    "face_yaw_angle",
    "norm_shoulder_width",
]

AGGREGATIONS = ["mean", "std", "first", "last", "delta", "min", "max"]
LOW_CONFIDENCE_THRESHOLD = 0.2
_EPS = 1e-9

COCO17_NAMES = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}

REQUIRED_KEYPOINTS = [
    "nose",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_ankle",
    "right_ankle",
]


@dataclass
class FeatureBuildReport:
    """Mutable counters collected during BIU feature generation."""

    source_jsonl: str = ""
    splits_dir: str = ""
    output_dir: str = ""
    split_rows: dict[str, int] = field(default_factory=dict)
    output_rows: dict[str, int] = field(default_factory=dict)
    missing_frame_count: int = 0
    missing_frame_examples: list[dict[str, Any]] = field(default_factory=list)
    invalid_keypoint_frames: int = 0
    low_confidence_frames: int = 0
    sequence_nan_counts: dict[str, int] = field(default_factory=dict)
    feature_nan_counts: dict[str, int] = field(default_factory=dict)
    validation_messages: list[str] = field(default_factory=list)

    def add_missing_frame(self, sequence_id: str, key: tuple[str, str, str, int]) -> None:
        self.missing_frame_count += 1
        if len(self.missing_frame_examples) < 10:
            set_id, video_id, ped_id, frame_idx = key
            self.missing_frame_examples.append(
                {
                    "sequence_id": sequence_id,
                    "set_id": set_id,
                    "video_id": video_id,
                    "ped_id": ped_id,
                    "frame_idx": int(frame_idx),
                }
            )


def _to_xy(point: Any) -> np.ndarray:
    arr = np.asarray(point, dtype=float)
    return arr[:2]


def _unit(vector: Any) -> np.ndarray:
    vec = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > _EPS else np.zeros_like(vec)


def dist(p1: Any, p2: Any) -> float:
    """Euclidean distance between two keypoints."""
    return float(np.linalg.norm(_to_xy(p1) - _to_xy(p2)))


def midpoint(p1: Any, p2: Any) -> np.ndarray:
    """Midpoint between two keypoints."""
    return (_to_xy(p1) + _to_xy(p2)) / 2.0


def angle_between(v1: Any, v2: Any, *, signed: bool = False) -> float:
    """Angle between vectors in radians, matching the old active pose_utils behavior."""
    v1u = _unit(v1)
    v2u = _unit(v2)
    if signed:
        cross_z = v2u[0] * v1u[1] - v2u[1] * v1u[0]
        dot = float(np.clip(np.dot(v1u, v2u), -1.0, 1.0))
        return float(np.arctan2(cross_z, dot))
    dot = float(np.clip(np.dot(v1u, v2u), -1.0, 1.0))
    return float(np.arccos(dot))


def torso_lean_angle(left_shoulder: Any, right_shoulder: Any, left_hip: Any, right_hip: Any) -> float:
    shoulder_mid = midpoint(left_shoulder, right_shoulder)
    hip_mid = midpoint(left_hip, right_hip)
    torso_vec = hip_mid - shoulder_mid
    vertical_vec = np.array([0.0, 1.0])
    theta = angle_between(torso_vec, vertical_vec, signed=True)
    magnitude = abs(theta)
    if magnitude > np.pi / 2:
        magnitude = np.pi - magnitude
        theta = math.copysign(magnitude, theta)
    return float(theta)


def stride_length(left_ankle: Any, right_ankle: Any, *, norm_scale: float | None = None) -> float:
    value = dist(left_ankle, right_ankle)
    return float(value / norm_scale) if norm_scale is not None and norm_scale > 0 else float(value)


def head_yaw_angle(nose: Any, left_shoulder: Any, right_shoulder: Any) -> float:
    neck = midpoint(left_shoulder, right_shoulder)
    head_vec = _to_xy(nose) - neck
    return angle_between(head_vec, np.array([0.0, -1.0]), signed=True)


def lead_foot_elevation(left_ankle: Any, right_ankle: Any, *, norm_scale: float | None = None) -> tuple[float, bool]:
    elev_px = abs(float(left_ankle[1]) - float(right_ankle[1]))
    if norm_scale is not None and norm_scale > 0:
        elev = elev_px / norm_scale
        threshold = 0.05
    else:
        elev = elev_px
        threshold = 5.0
    return float(elev), bool(elev > threshold)


def arm_swing_asymmetry(left_wrist: Any, right_wrist: Any, left_shoulder: Any, right_shoulder: Any) -> float:
    delta_y = abs(_to_xy(left_wrist)[1] - _to_xy(right_wrist)[1])
    shoulder_width = dist(left_shoulder, right_shoulder)
    return float(delta_y / shoulder_width) if shoulder_width > _EPS else float("nan")


def gaze_to_car_angle(nose: Any, left_ear: Any, right_ear: Any) -> float:
    head_center = midpoint(left_ear, right_ear)
    head_forward = _to_xy(nose) - head_center
    if np.linalg.norm(head_forward) <= _EPS:
        return float("nan")
    return angle_between(head_forward, np.array([0.0, -1.0]), signed=True)


def keypoints_to_joints(keypoints: np.ndarray) -> dict[str, np.ndarray]:
    """Map COCO-17 keypoints to named joints required by BIU formulas."""
    if keypoints.shape != (17, 2):
        raise ValueError(f"Expected keypoints shape (17, 2), got {keypoints.shape}")
    return {name: keypoints[index] for name, index in COCO17_NAMES.items()}


def frame_has_invalid_required_keypoints(keypoints: np.ndarray) -> bool:
    required = keypoints[[COCO17_NAMES[name] for name in REQUIRED_KEYPOINTS], :]
    return bool(np.isnan(required).any())


def frame_has_low_confidence(kp_conf: np.ndarray | None) -> bool:
    if kp_conf is None or kp_conf.size == 0:
        return False
    required = kp_conf[[COCO17_NAMES[name] for name in REQUIRED_KEYPOINTS]]
    return bool(np.nanmin(required) < LOW_CONFIDENCE_THRESHOLD)


def compute_frame_biu_features(keypoints: np.ndarray) -> dict[str, float]:
    """Compute active old-repo BIU features for one frame."""
    if frame_has_invalid_required_keypoints(keypoints):
        return {feature: float("nan") for feature in BIU_BASE_FEATURES}

    joints = keypoints_to_joints(keypoints)
    left_shoulder = joints["left_shoulder"]
    right_shoulder = joints["right_shoulder"]
    shoulder_px = dist(left_shoulder, right_shoulder)
    height_px = dist(joints["nose"], joints["left_ankle"])

    foot_elev, foot_lifted = lead_foot_elevation(
        joints["left_ankle"],
        joints["right_ankle"],
        norm_scale=shoulder_px,
    )
    shoulder_vec = _to_xy(right_shoulder) - _to_xy(left_shoulder)
    hip_mid = midpoint(joints["left_hip"], joints["right_hip"])
    face_vec = _to_xy(joints["nose"]) - hip_mid

    return {
        "torso_lean_angle": torso_lean_angle(left_shoulder, right_shoulder, joints["left_hip"], joints["right_hip"]),
        "stride_length": stride_length(joints["left_ankle"], joints["right_ankle"], norm_scale=shoulder_px),
        "head_yaw_angle": head_yaw_angle(joints["nose"], left_shoulder, right_shoulder),
        "lead_foot_elevation": foot_elev,
        "lead_foot_lifted": float(int(foot_lifted)),
        "arm_swing_asymmetry": arm_swing_asymmetry(joints["left_wrist"], joints["right_wrist"], left_shoulder, right_shoulder),
        "gaze_to_car_angle": gaze_to_car_angle(joints["nose"], joints["left_ear"], joints["right_ear"]),
        "body_yaw_angle": angle_between(shoulder_vec, np.array([1.0, 0.0]), signed=True),
        "face_yaw_angle": angle_between(face_vec, np.array([0.0, -1.0]), signed=True),
        "norm_shoulder_width": float(shoulder_px / (height_px + 1e-6)),
    }


def aggregate_biu_sequence(frame_features: list[dict[str, float]]) -> dict[str, float]:
    """Aggregate per-frame BIUs over one frozen 16-frame sequence."""
    output: dict[str, float] = {}
    for feature in BIU_BASE_FEATURES:
        values = np.asarray([row[feature] for row in frame_features], dtype=float)
        finite_mask = np.isfinite(values)
        if finite_mask.any():
            output[f"{feature}_mean"] = float(np.nanmean(values))
            output[f"{feature}_std"] = float(np.nanstd(values, ddof=0))
            output[f"{feature}_first"] = float(values[0]) if np.isfinite(values[0]) else float("nan")
            output[f"{feature}_last"] = float(values[-1]) if np.isfinite(values[-1]) else float("nan")
            output[f"{feature}_delta"] = output[f"{feature}_last"] - output[f"{feature}_first"] if np.isfinite(output[f"{feature}_first"]) and np.isfinite(output[f"{feature}_last"]) else float("nan")
            output[f"{feature}_min"] = float(np.nanmin(values))
            output[f"{feature}_max"] = float(np.nanmax(values))
        else:
            for agg in AGGREGATIONS:
                output[f"{feature}_{agg}"] = float("nan")
    return output


def parse_keypoints(raw: Any, *, source: str) -> np.ndarray:
    arr = np.asarray(raw, dtype=float)
    if arr.shape != (17, 2):
        raise ValueError(f"{source} expected keypoints shape (17, 2), got {arr.shape}")
    return arr


def parse_conf(raw: Any) -> np.ndarray | None:
    if raw is None:
        return None
    arr = np.asarray(raw, dtype=float)
    if arr.size == 0:
        return None
    if arr.shape != (17,):
        return None
    return arr


def load_keypoint_index(jsonl_path: Path) -> dict[tuple[str, str, str, int], dict[str, Any]]:
    """Load keypoints into an index keyed by (set, video, pid, frame_idx)."""
    index: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    required = {"set", "video", "pid", "frame_idx", "keypoints", "label"}
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = required - set(row)
            if missing:
                raise ValueError(f"{jsonl_path}:{line_num} missing fields {sorted(missing)}")
            key = (str(row["set"]), str(row["video"]), str(row["pid"]), int(row["frame_idx"]))
            if key in index:
                raise ValueError(f"Duplicate JSONL frame key at {jsonl_path}:{line_num}: {key}")
            index[key] = {
                "keypoints": parse_keypoints(row["keypoints"], source=f"{jsonl_path}:{line_num}"),
                "kp_conf": parse_conf(row.get("kp_conf")),
                "label": int(row["label"]),
            }
    if not index:
        raise RuntimeError(f"No keypoint rows loaded from {jsonl_path}")
    return index


def load_keypoints_for_sequence(sequence_row: pd.Series, keypoint_index: dict[tuple[str, str, str, int], dict[str, Any]], report: FeatureBuildReport | None = None) -> list[dict[str, Any]]:
    """Load exact frozen frames for one sequence row.

    Returns one indexed JSONL frame record per raw frame. Missing required frames are
    counted in report and then raised loudly by the caller.
    """
    frames: list[dict[str, Any]] = []
    for frame_idx in range(int(sequence_row["start_frame"]), int(sequence_row["end_frame"]) + 1):
        key = (str(sequence_row["set_id"]), str(sequence_row["video_id"]), str(sequence_row["ped_id"]), frame_idx)
        item = keypoint_index.get(key)
        if item is None:
            if report is not None:
                report.add_missing_frame(str(sequence_row["sequence_id"]), key)
            continue
        frames.append(item)
    return frames


def compute_biu_features_for_sequence(frames: list[dict[str, Any]], report: FeatureBuildReport | None = None) -> dict[str, float]:
    """Compute aggregated BIU features for exactly 16 loaded JSONL frame records."""
    if len(frames) != 16:
        raise ValueError(f"Expected exactly 16 frames, got {len(frames)}")
    per_frame: list[dict[str, float]] = []
    for item in frames:
        keypoints = item["keypoints"]
        if frame_has_invalid_required_keypoints(keypoints) and report is not None:
            report.invalid_keypoint_frames += 1
        if frame_has_low_confidence(item.get("kp_conf")) and report is not None:
            report.low_confidence_frames += 1
        per_frame.append(compute_frame_biu_features(keypoints))
    return aggregate_biu_sequence(per_frame)


def split_feature_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col not in SEQUENCE_META_COLUMNS]


def build_biu_feature_table(
    sequence_csv: Path,
    keypoint_index: dict[tuple[str, str, str, int], dict[str, Any]],
    output_path: Path,
    report: FeatureBuildReport,
) -> pd.DataFrame:
    """Build one BIU feature table aligned by sequence_id."""
    sequences = pd.read_csv(sequence_csv)
    required = set(SEQUENCE_META_COLUMNS)
    missing = required - set(sequences.columns)
    if missing:
        raise ValueError(f"{sequence_csv} missing required columns: {sorted(missing)}")

    rows: list[dict[str, Any]] = []
    split_name = str(sequences["split"].iloc[0]) if not sequences.empty else sequence_csv.stem.replace("_sequences", "")
    report.split_rows[split_name] = int(len(sequences))

    for _, seq in sequences.iterrows():
        loaded_frames = load_keypoints_for_sequence(seq, keypoint_index, report)
        if len(loaded_frames) != 16:
            raise ValueError(
                f"Sequence {seq['sequence_id']} expected 16 frames but loaded {len(loaded_frames)}. "
                "Missing frame details are recorded in the report object."
            )
        features = compute_biu_features_for_sequence(loaded_frames, report)
        row = {col: seq[col] for col in SEQUENCE_META_COLUMNS}
        row.update(features)
        row["biu_valid_frame_count"] = int(16 - sum(frame_has_invalid_required_keypoints(item["keypoints"]) for item in loaded_frames))
        row["biu_low_confidence_frame_count"] = int(sum(frame_has_low_confidence(item.get("kp_conf")) for item in loaded_frames))
        rows.append(row)

    feature_df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_csv(output_path, index=False)
    report.output_rows[split_name] = int(len(feature_df))
    report.sequence_nan_counts[split_name] = int(feature_df.isna().any(axis=1).sum())
    for col in split_feature_columns(feature_df):
        report.feature_nan_counts[col] = report.feature_nan_counts.get(col, 0) + int(feature_df[col].isna().sum())
    return feature_df


def validate_feature_table(split_df: pd.DataFrame, feature_df: pd.DataFrame, split_name: str) -> list[str]:
    """Validate one BIU feature table against its frozen split CSV."""
    messages: list[str] = []
    if "sequence_id" not in feature_df.columns:
        raise ValueError(f"{split_name}: feature table missing sequence_id")
    if feature_df["sequence_id"].duplicated().any():
        raise ValueError(f"{split_name}: duplicate sequence_id values in feature table")
    if split_df["sequence_id"].duplicated().any():
        raise ValueError(f"{split_name}: duplicate sequence_id values in split CSV")
    if len(split_df) != len(feature_df):
        raise ValueError(f"{split_name}: row count mismatch split={len(split_df)} features={len(feature_df)}")

    split_ids = set(split_df["sequence_id"].astype(str))
    feature_ids = set(feature_df["sequence_id"].astype(str))
    if split_ids != feature_ids:
        missing = sorted(split_ids - feature_ids)[:5]
        extra = sorted(feature_ids - split_ids)[:5]
        raise ValueError(f"{split_name}: sequence_id mismatch missing={missing} extra={extra}")

    merged = split_df[["sequence_id", "label"]].merge(
        feature_df[["sequence_id", "label"]],
        on="sequence_id",
        suffixes=("_split", "_feature"),
        validate="one_to_one",
    )
    if not (merged["label_split"].astype(int) == merged["label_feature"].astype(int)).all():
        raise ValueError(f"{split_name}: labels do not match between split and feature table")

    feature_cols = split_feature_columns(feature_df)
    entirely_null = [col for col in feature_cols if feature_df[col].isna().all()]
    if entirely_null:
        raise ValueError(f"{split_name}: entirely-null feature columns: {entirely_null[:10]}")

    messages.append(f"{split_name}: sequence_ids complete, labels match, rows={len(feature_df)}")
    messages.append(f"{split_name}: no entirely-null feature columns")
    return messages


def validate_biu_feature_outputs(splits_dir: Path, output_dir: Path) -> list[str]:
    """Validate all generated BIU feature tables."""
    messages: list[str] = []
    for split in ["train", "val", "test"]:
        split_df = pd.read_csv(splits_dir / f"{split}_sequences.csv")
        feature_df = pd.read_csv(output_dir / f"biu_{split}.csv")
        messages.extend(validate_feature_table(split_df, feature_df, split))
    return messages
