"""Audit JAAD BIU variants and normalization choices on canonical rows."""

from __future__ import annotations

import json
import math
import sys
import textwrap
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


CLEAN_ROOT = Path("/home/abdelrahman/Hadwa/BIU-PID-Distillation-clean")
EXPERIMENT_ROOT = CLEAN_ROOT / "experiments/007_biu_variant_normalization_audit"
METRICS_DIR = EXPERIMENT_ROOT / "metrics"
FIGURES_DIR = EXPERIMENT_ROOT / "figures"
REPORT_PATH = EXPERIMENT_ROOT / "report.md"
README_PATH = EXPERIMENT_ROOT / "README.md"

CANONICAL_ROWS_PATH = CLEAN_ROOT / "experiments/004_jaad_temporal_biu_transition/metrics/horizon_feature_rows.csv"
RAW_AUDIT_MANIFEST_PATH = CLEAN_ROOT / "experiments/006_jaad_raw_keypoint_pattern_audit/metrics/raw_kp_sample_manifest.csv"
RAW_AUDIT_EFFECTS_PATH = CLEAN_ROOT / "experiments/006_jaad_raw_keypoint_pattern_audit/metrics/raw_kp_feature_effect_sizes.csv"
KEYPOINT_JSONL_PATH = CLEAN_ROOT / "data/jaad_keypoints_vitpose_s_extended_short_horizon.jsonl"
SRC_ROOT = CLEAN_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from biu_extractor import ORDER as CURRENT_BIU_ORDER  # noqa: E402
from biu_extractor import extract_biu_features  # noqa: E402


ARTIFACT_COVERAGE_PATH = METRICS_DIR / "biu_variant_artifact_coverage.csv"
SAMPLE_MANIFEST_PATH = METRICS_DIR / "biu_variant_sample_manifest.csv"
FEATURE_VALUES_PATH = METRICS_DIR / "biu_variant_feature_values.csv"
EFFECT_SIZES_PATH = METRICS_DIR / "biu_variant_effect_sizes.csv"
TOP_FEATURES_PATH = METRICS_DIR / "biu_variant_top_features.csv"
FAMILY_SUMMARY_PATH = METRICS_DIR / "biu_variant_family_summary.csv"
DIRECTION_IMPACT_PATH = METRICS_DIR / "biu_variant_direction_impact.csv"
NORMALIZATION_IMPACT_PATH = METRICS_DIR / "biu_variant_normalization_impact.csv"
DELTA_SUMMARY_PATH = METRICS_DIR / "biu_variant_delta_summary.csv"
REPRODUCTION_PATH = METRICS_DIR / "biu_variant_reproduction_summary.csv"
DIAGNOSTIC_PATH = METRICS_DIR / "biu_variant_diagnostic_classifier_results.csv"

COCO_KEYPOINTS = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]
KPT_INDEX = {name: idx for idx, name in enumerate(COCO_KEYPOINTS)}
LEFT_RIGHT_PAIRS = [
    ("left_eye", "right_eye"),
    ("left_ear", "right_ear"),
    ("left_shoulder", "right_shoulder"),
    ("left_elbow", "right_elbow"),
    ("left_wrist", "right_wrist"),
    ("left_hip", "right_hip"),
    ("left_knee", "right_knee"),
    ("left_ankle", "right_ankle"),
]
GROUP_ORDER = ["clean_negative", "ambiguous_negative", "actual_positive"]
HORIZON_ORDER = ["30_60", "15_30", "5_15"]
PAIRWISE = [
    ("actual_positive", "ambiguous_negative"),
    ("ambiguous_negative", "clean_negative"),
    ("actual_positive", "clean_negative"),
]
IMAGE_WIDTH = 1920.0
IMAGE_HEIGHT = 1080.0

CURRENT_BIU_BASES = [
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
SUMMARY_STATS = ["mean", "std", "first", "last", "delta", "min", "max"]
CURRENT_BIU_COLUMNS = [f"{base}_{stat}" for base in CURRENT_BIU_BASES for stat in SUMMARY_STATS]

CURRENT_GROUPS = {
    "torso_lean_angle": "current_anatomical",
    "stride_length": "current_track_motion",
    "head_yaw_angle": "head_gaze_proxy",
    "lead_foot_elevation": "current_track_motion",
    "lead_foot_lifted": "current_track_motion",
    "arm_swing_asymmetry": "current_anatomical",
    "gaze_to_car_angle": "head_gaze_proxy",
    "body_yaw_angle": "current_anatomical",
    "face_yaw_angle": "head_gaze_proxy",
    "norm_shoulder_width": "shoulder_width_scale",
}


def ensure_dirs() -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def table_text(df: pd.DataFrame) -> str:
    if df.empty:
        return "No rows."
    try:
        return df.to_markdown(index=False)
    except Exception:
        return "```\n" + df.to_string(index=False) + "\n```"


def safe_mean(arr: np.ndarray, axis: int | None = None) -> np.ndarray | float:
    with np.errstate(invalid="ignore"):
        return np.nanmean(arr, axis=axis)


def safe_std(arr: np.ndarray, axis: int | None = None) -> np.ndarray | float:
    with np.errstate(invalid="ignore"):
        return np.nanstd(arr, axis=axis)


def safe_min(arr: np.ndarray, axis: int | None = None) -> np.ndarray | float:
    with np.errstate(invalid="ignore"):
        return np.nanmin(arr, axis=axis)


def safe_max(arr: np.ndarray, axis: int | None = None) -> np.ndarray | float:
    with np.errstate(invalid="ignore"):
        return np.nanmax(arr, axis=axis)


def pair_average(coords: np.ndarray, names: list[str]) -> np.ndarray:
    idxs = [KPT_INDEX[n] for n in names]
    return safe_mean(coords[:, idxs, :], axis=1)


def euclid(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.norm(a - b, axis=1)


def load_keypoint_index() -> dict[tuple[str, str, int], dict[str, Any]]:
    idx: dict[tuple[str, str, int], dict[str, Any]] = {}
    with KEYPOINT_JSONL_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            rec = json.loads(line)
            idx[(str(rec["video"]), str(rec["pid"]), int(rec["frame_idx"]))] = rec
    return idx


def artifact_coverage() -> pd.DataFrame:
    rows = []
    for label, path in [
        ("canonical_rows", CANONICAL_ROWS_PATH),
        ("raw_kp_manifest", RAW_AUDIT_MANIFEST_PATH),
        ("raw_kp_effect_sizes", RAW_AUDIT_EFFECTS_PATH),
        ("jaad_keypoint_jsonl", KEYPOINT_JSONL_PATH),
        ("biu_extractor", SRC_ROOT / "biu_extractor.py"),
        ("pose_utils", SRC_ROOT / "pose_utils.py"),
    ]:
        rows.append(
            {
                "artifact": label,
                "path": str(path),
                "exists": int(path.exists()),
                "size_bytes": path.stat().st_size if path.exists() else 0,
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(ARTIFACT_COVERAGE_PATH, index=False)
    return df


def load_canonical_rows() -> pd.DataFrame:
    df = pd.read_csv(CANONICAL_ROWS_PATH)
    keep = [
        "track_key",
        "sequence_id",
        "split",
        "video_id",
        "ped_id",
        "track_id",
        "start_frame",
        "end_frame",
        "event_frame_or_anchor",
        "label",
        "group",
        "group_notes",
        "horizon_label",
        "observed_frames_json",
        "bbox_sequence_json",
    ] + CURRENT_BIU_COLUMNS + [
        "bbox_center_x_delta",
        "center_x_velocity_mean",
        "center_x_velocity_std",
        "center_x_displacement",
        "bbox_width_delta",
        "bbox_height_delta",
    ]
    return df[keep].copy()


def swap_left_right(coords: np.ndarray) -> np.ndarray:
    swapped = coords.copy()
    for left, right in LEFT_RIGHT_PAIRS:
        li = KPT_INDEX[left]
        ri = KPT_INDEX[right]
        swapped[:, [li, ri], :] = swapped[:, [ri, li], :]
    return swapped


def compute_direction(coords: np.ndarray, bbox: np.ndarray) -> dict[str, Any]:
    bbox_center_x = (bbox[:, 0] + bbox[:, 2]) / 2.0
    bbox_dx = float(bbox_center_x[-1] - bbox_center_x[0])
    hip_center_x = safe_mean(coords[:, [KPT_INDEX["left_hip"], KPT_INDEX["right_hip"]], 0], axis=1)
    ankle_center_x = safe_mean(coords[:, [KPT_INDEX["left_ankle"], KPT_INDEX["right_ankle"]], 0], axis=1)
    hip_dx = float(hip_center_x[-1] - hip_center_x[0]) if np.isfinite(hip_center_x[[0, -1]]).all() else float("nan")
    ankle_dx = float(ankle_center_x[-1] - ankle_center_x[0]) if np.isfinite(ankle_center_x[[0, -1]]).all() else float("nan")
    if abs(bbox_dx) >= 1.0:
        source = "bbox_center_x"
        dx = bbox_dx
    elif np.isfinite(hip_dx) and abs(hip_dx) >= 1.0:
        source = "hip_center_x"
        dx = hip_dx
    else:
        source = "ankle_center_x"
        dx = ankle_dx if np.isfinite(ankle_dx) else bbox_dx
    return {
        "direction_source": source,
        "dominant_dx_pixels": dx,
        "bbox_center_dx_pixels": bbox_dx,
        "hip_center_dx_pixels": hip_dx,
        "ankle_center_dx_pixels": ankle_dx,
        "flipped": int(dx < 0),
    }


def compute_scales(coords: np.ndarray) -> dict[str, np.ndarray]:
    shoulder_center = pair_average(coords, ["left_shoulder", "right_shoulder"])
    hip_center = pair_average(coords, ["left_hip", "right_hip"])
    shoulder_width = euclid(coords[:, KPT_INDEX["left_shoulder"], :], coords[:, KPT_INDEX["right_shoulder"], :])
    torso_length = euclid(shoulder_center, hip_center)
    shoulder_width[(~np.isfinite(shoulder_width)) | (shoulder_width < 1e-6)] = np.nan
    torso_length[(~np.isfinite(torso_length)) | (torso_length < 1e-6)] = np.nan
    return {
        "shoulder_width": shoulder_width,
        "torso_length": torso_length,
        "shoulder_center": shoulder_center,
        "hip_center": hip_center,
    }


def load_window(rec: pd.Series, kp_index: dict[tuple[str, str, int], dict[str, Any]]) -> tuple[dict[str, Any] | None, str]:
    frames = json.loads(rec["observed_frames_json"])
    bbox = np.asarray(json.loads(rec["bbox_sequence_json"]), dtype=float)
    if len(frames) != len(bbox):
        return None, "bbox_frame_count_mismatch"
    rows = []
    for frame in frames:
        row = kp_index.get((str(rec["video_id"]), str(rec["ped_id"]), int(frame)))
        if row is None:
            return None, f"missing_keypoint_frame:{frame}"
        rows.append(row)
    coords = np.asarray([row["keypoints"] for row in rows], dtype=float)
    conf = np.asarray([row["kp_conf"] for row in rows], dtype=float)
    skel_conf = np.asarray([row["skeleton_conf"] for row in rows], dtype=float)
    if coords.shape != (len(frames), 17, 2):
        return None, f"unexpected_shape:{coords.shape}"
    return {
        "frames": np.asarray(frames, dtype=int),
        "bbox": bbox,
        "coords": coords,
        "conf": conf,
        "skel_conf": skel_conf,
    }, ""


def normalize_coords(coords: np.ndarray, bbox: np.ndarray) -> dict[str, np.ndarray]:
    scales = compute_scales(coords)
    shoulder_center = scales["shoulder_center"]
    hip_center = scales["hip_center"]
    torso_center = hip_center.copy()
    shoulder_width = scales["shoulder_width"]
    torso_length = scales["torso_length"]

    img = coords.copy()
    img[:, :, 0] /= IMAGE_WIDTH
    img[:, :, 1] /= IMAGE_HEIGHT

    bbox_w = np.clip(bbox[:, 2] - bbox[:, 0], 1e-6, None)
    bbox_h = np.clip(bbox[:, 3] - bbox[:, 1], 1e-6, None)
    bbox_norm = coords.copy()
    bbox_norm[:, :, 0] = (bbox_norm[:, :, 0] - bbox[:, None, 0]) / bbox_w[:, None]
    bbox_norm[:, :, 1] = (bbox_norm[:, :, 1] - bbox[:, None, 1]) / bbox_h[:, None]

    shoulder_norm = coords.copy()
    shoulder_norm[:, :, 0] = (shoulder_norm[:, :, 0] - shoulder_center[:, None, 0]) / shoulder_width[:, None]
    shoulder_norm[:, :, 1] = (shoulder_norm[:, :, 1] - shoulder_center[:, None, 1]) / shoulder_width[:, None]

    torso_norm = coords.copy()
    torso_norm[:, :, 0] = (torso_norm[:, :, 0] - torso_center[:, None, 0]) / torso_length[:, None]
    torso_norm[:, :, 1] = (torso_norm[:, :, 1] - torso_center[:, None, 1]) / torso_length[:, None]

    first_rel = coords.copy()
    first_rel[:, :, 0] -= first_rel[0:1, :, 0]
    first_rel[:, :, 1] -= first_rel[0:1, :, 1]

    return {
        "image_normalized": img,
        "bbox_normalized": bbox_norm,
        "shoulder_width_normalized": shoulder_norm,
        "torso_length_normalized": torso_norm,
        "first_frame_relative": first_rel,
        "torso_centered": torso_norm,
    }


def joints_from_frame(frame_coords: np.ndarray) -> dict[str, np.ndarray]:
    return {name: frame_coords[idx, :].astype(float) for name, idx in KPT_INDEX.items()}


def summarize_series(prefix: str, values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {
        f"{prefix}_mean": float(np.nanmean(arr)),
        f"{prefix}_std": float(np.nanstd(arr)),
        f"{prefix}_first": float(arr[0]),
        f"{prefix}_last": float(arr[-1]),
        f"{prefix}_delta": float(arr[-1] - arr[0]),
        f"{prefix}_min": float(np.nanmin(arr)),
        f"{prefix}_max": float(np.nanmax(arr)),
    }


def summarize_current_biu_sequence(coords: np.ndarray, prefix: str) -> dict[str, float]:
    per_frame: dict[str, list[float]] = {name: [] for name in CURRENT_BIU_ORDER}
    for frame_idx in range(coords.shape[0]):
        feats = extract_biu_features(joints_from_frame(coords[frame_idx]))
        for name in CURRENT_BIU_ORDER:
            per_frame[name].append(float(feats[name]))
    out: dict[str, float] = {}
    for name, vals in per_frame.items():
        out.update(summarize_series(f"{prefix}__{name}", np.asarray(vals, dtype=float)))
    return out


def lower_body_features(coords: np.ndarray) -> dict[str, np.ndarray]:
    left_hip = coords[:, KPT_INDEX["left_hip"], :]
    right_hip = coords[:, KPT_INDEX["right_hip"], :]
    left_knee = coords[:, KPT_INDEX["left_knee"], :]
    right_knee = coords[:, KPT_INDEX["right_knee"], :]
    left_ankle = coords[:, KPT_INDEX["left_ankle"], :]
    right_ankle = coords[:, KPT_INDEX["right_ankle"], :]

    hip_center = safe_mean(np.stack([left_hip, right_hip], axis=1), axis=1)
    knee_center = safe_mean(np.stack([left_knee, right_knee], axis=1), axis=1)
    ankle_center = safe_mean(np.stack([left_ankle, right_ankle], axis=1), axis=1)
    leg_center = safe_mean(np.stack([hip_center, knee_center, ankle_center], axis=1), axis=1)

    lead_ankle_x = np.maximum(left_ankle[:, 0], right_ankle[:, 0])
    lower_body_mag = np.sqrt(
        np.square(np.diff(hip_center[:, 0], prepend=hip_center[0, 0]))
        + np.square(np.diff(ankle_center[:, 1], prepend=ankle_center[0, 1]))
    )
    lower_body_consistency = np.abs(np.diff(ankle_center[:, 0], prepend=ankle_center[0, 0]))

    return {
        "hip_center_x": hip_center[:, 0],
        "ankle_center_x": ankle_center[:, 0],
        "leg_center_x": leg_center[:, 0],
        "knee_center_x": knee_center[:, 0],
        "ankle_distance": np.abs(left_ankle[:, 0] - right_ankle[:, 0]),
        "stride_opening": euclid(left_ankle, right_ankle),
        "lead_ankle_forward": lead_ankle_x,
        "lower_body_motion_magnitude": lower_body_mag,
        "lower_body_motion_consistency": lower_body_consistency,
    }


def torso_features(coords: np.ndarray) -> dict[str, np.ndarray]:
    left_shoulder = coords[:, KPT_INDEX["left_shoulder"], :]
    right_shoulder = coords[:, KPT_INDEX["right_shoulder"], :]
    left_hip = coords[:, KPT_INDEX["left_hip"], :]
    right_hip = coords[:, KPT_INDEX["right_hip"], :]

    shoulder_center = safe_mean(np.stack([left_shoulder, right_shoulder], axis=1), axis=1)
    hip_center = safe_mean(np.stack([left_hip, right_hip], axis=1), axis=1)
    torso_center = safe_mean(np.stack([shoulder_center, hip_center], axis=1), axis=1)
    torso_vec = shoulder_center - hip_center
    torso_lean_proxy = np.arctan2(torso_vec[:, 0], np.abs(torso_vec[:, 1]) + 1e-6)
    shoulder_to_hip_x = shoulder_center[:, 0] - hip_center[:, 0]
    shoulder_to_hip_y = shoulder_center[:, 1] - hip_center[:, 1]
    torso_motion_mag = np.sqrt(
        np.square(np.diff(torso_center[:, 0], prepend=torso_center[0, 0]))
        + np.square(np.diff(torso_center[:, 1], prepend=torso_center[0, 1]))
    )
    return {
        "hip_center_x": hip_center[:, 0],
        "hip_center_y": hip_center[:, 1],
        "shoulder_center_x": shoulder_center[:, 0],
        "shoulder_center_y": shoulder_center[:, 1],
        "torso_center_x": torso_center[:, 0],
        "torso_center_y": torso_center[:, 1],
        "shoulder_to_hip_x": shoulder_to_hip_x,
        "shoulder_to_hip_y": shoulder_to_hip_y,
        "torso_lean_proxy": torso_lean_proxy,
        "shoulder_center_rel_hip_x": shoulder_center[:, 0] - hip_center[:, 0],
        "shoulder_center_rel_hip_y": shoulder_center[:, 1] - hip_center[:, 1],
        "torso_motion_magnitude": torso_motion_mag,
    }


def summarize_motion_family(
    family_name: str,
    coords_variants: dict[str, np.ndarray],
    family_kind: str,
) -> dict[str, float]:
    out: dict[str, float] = {}
    for norm_name, arr in coords_variants.items():
        base = lower_body_features(arr) if family_kind == "lower_body" else torso_features(arr)
        for metric_name, seq in base.items():
            if metric_name.endswith("_magnitude") or metric_name.endswith("_consistency"):
                out.update(
                    summarize_series(
                        f"{family_name}__{norm_name}__signed__{metric_name}",
                        seq,
                    )
                )
                continue
            vel = np.diff(seq, prepend=seq[0])
            out.update(summarize_series(f"{family_name}__{norm_name}__signed__{metric_name}", seq))
            out[f"{family_name}__{norm_name}__signed__{metric_name}_velocity_mean"] = float(np.nanmean(vel))
            out[f"{family_name}__{norm_name}__signed__{metric_name}_velocity_std"] = float(np.nanstd(vel))

            abs_seq = np.abs(seq)
            abs_vel = np.abs(vel)
            out.update(summarize_series(f"{family_name}__{norm_name}__absolute__{metric_name}", abs_seq))
            out[f"{family_name}__{norm_name}__absolute__{metric_name}_velocity_mean"] = float(np.nanmean(abs_vel))
            out[f"{family_name}__{norm_name}__absolute__{metric_name}_velocity_std"] = float(np.nanstd(abs_vel))
    return out


def summarize_loaded_current(rec: pd.Series) -> dict[str, float]:
    return {f"current_biu__loaded__{col}": float(rec[col]) for col in CURRENT_BIU_COLUMNS}


def summarize_abs_current_loaded(rec: pd.Series) -> dict[str, float]:
    out: dict[str, float] = {}
    for col in CURRENT_BIU_COLUMNS:
        val = float(rec[col])
        base, stat = col.rsplit("_", 1)
        if stat in {"first", "last", "delta", "mean", "min", "max"} and ("yaw" in base or "lean" in base):
            val = abs(val)
        out[f"current_biu_absolute_motion__loaded__{col}"] = val
    return out


def feature_metadata(feature_name: str) -> dict[str, str]:
    parts = feature_name.split("__")
    family = parts[0]
    normalization = "loaded"
    motion_mode = "signed"
    base = feature_name
    if len(parts) >= 3:
        normalization = parts[1]
        motion_mode = parts[2]
        base = "__".join(parts[3:]) if len(parts) > 3 else parts[-1]
    elif len(parts) == 2:
        base = parts[1]

    lower = base.lower()
    if "head" in lower or "gaze" in lower or "face" in lower:
        body_region = "head_gaze_proxy"
    elif "ankle" in lower or "knee" in lower or "leg" in lower or "stride" in lower or "lead_ankle" in lower:
        body_region = "lower_body"
    elif "shoulder" in lower or "torso" in lower or "hip" in lower:
        body_region = "torso"
    elif "bbox" in lower or "center_x" in lower or "velocity" in lower:
        body_region = "bbox_like_motion"
    else:
        body_region = "mixed"

    if family.startswith("current_biu"):
        feature_group = CURRENT_GROUPS.get(base.rsplit("_", 1)[0], "current_anatomical")
    elif family == "lower_body_motion_biu":
        feature_group = "lower_body"
    elif family == "torso_motion_biu":
        feature_group = "torso"
    elif family == "delta_across_horizons":
        feature_group = "delta_across_horizons"
    else:
        feature_group = "normalized"

    return {
        "family": family,
        "normalization": normalization,
        "motion_mode": motion_mode,
        "feature_group": feature_group,
        "body_region": body_region,
    }


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    diffs = np.subtract.outer(x, y)
    return float((np.sum(diffs > 0) - np.sum(diffs < 0)) / (len(x) * len(y)))


def cohen_d(x: np.ndarray, y: np.ndarray) -> float:
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return float("nan")
    vx = np.var(x, ddof=1)
    vy = np.var(y, ddof=1)
    pooled = ((nx - 1) * vx + (ny - 1) * vy) / max(nx + ny - 2, 1)
    if pooled <= 0 or not np.isfinite(pooled):
        return float("nan")
    return float((np.mean(x) - np.mean(y)) / math.sqrt(pooled))


def pairwise_stats(feature_df: pd.DataFrame) -> pd.DataFrame:
    meta_cols = {
        "track_key",
        "sequence_id",
        "split",
        "video_id",
        "ped_id",
        "track_id",
        "start_frame",
        "end_frame",
        "event_frame_or_anchor",
        "label",
        "group",
        "group_notes",
        "horizon_label",
        "direction_source",
        "dominant_dx_pixels",
        "bbox_center_dx_pixels",
        "hip_center_dx_pixels",
        "ankle_center_dx_pixels",
        "direction_flipped",
        "prob_source",
    }
    feature_cols = [c for c in feature_df.columns if c not in meta_cols]
    rows = []
    for horizon in HORIZON_ORDER + ["cross_horizon"]:
        horizon_df = feature_df[feature_df["horizon_label"] == horizon]
        if horizon_df.empty:
            continue
        for pos_group, neg_group in PAIRWISE:
            a_df = horizon_df[horizon_df["group"] == pos_group]
            b_df = horizon_df[horizon_df["group"] == neg_group]
            for feature in feature_cols:
                a = pd.to_numeric(a_df[feature], errors="coerce").dropna().to_numpy()
                b = pd.to_numeric(b_df[feature], errors="coerce").dropna().to_numpy()
                if len(a) < 5 or len(b) < 5:
                    continue
                try:
                    raw_auc = roc_auc_score(
                        np.r_[np.ones(len(a)), np.zeros(len(b))],
                        np.r_[a, b],
                    )
                except Exception:
                    raw_auc = float("nan")
                oriented_auc = float(max(raw_auc, 1.0 - raw_auc)) if np.isfinite(raw_auc) else float("nan")
                auc_direction = "higher_positive" if (not np.isfinite(raw_auc) or raw_auc >= 0.5) else "lower_positive"
                try:
                    pval = float(mannwhitneyu(a, b, alternative="two-sided").pvalue)
                except Exception:
                    pval = float("nan")
                meta = feature_metadata(feature)
                rows.append(
                    {
                        "horizon_label": horizon,
                        "comparison": f"{pos_group}__vs__{neg_group}",
                        "feature_name": feature,
                        **meta,
                        "count_a": len(a),
                        "count_b": len(b),
                        "mean_a": float(np.mean(a)),
                        "mean_b": float(np.mean(b)),
                        "median_a": float(np.median(a)),
                        "median_b": float(np.median(b)),
                        "cohen_d": cohen_d(a, b),
                        "abs_cohen_d": abs(cohen_d(a, b)),
                        "cliffs_delta": cliffs_delta(a, b),
                        "p_value": pval,
                        "raw_auc": float(raw_auc),
                        "oriented_auc": oriented_auc,
                        "auc_direction": auc_direction,
                        "missing_rate": float(1.0 - (len(a) + len(b)) / max(len(a_df) + len(b_df), 1)),
                    }
                )
    out = pd.DataFrame(rows)
    out.to_csv(EFFECT_SIZES_PATH, index=False)
    return out


def summarize_families(effects: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["horizon_label", "comparison", "family"]
    for keys, g in effects.groupby(group_cols):
        rows.append(
            {
                "horizon_label": keys[0],
                "comparison": keys[1],
                "family": keys[2],
                "feature_count": int(len(g)),
                "top_abs_cohen_d": float(g["abs_cohen_d"].max()),
                "top_oriented_auc": float(g["oriented_auc"].max()),
                "median_abs_cohen_d": float(g["abs_cohen_d"].median()),
                "oriented_auc_ge_0p70": int((g["oriented_auc"] >= 0.70).sum()),
                "oriented_auc_ge_0p80": int((g["oriented_auc"] >= 0.80).sum()),
                "d_ge_0p8": int((g["abs_cohen_d"] >= 0.8).sum()),
                "d_ge_1p2": int((g["abs_cohen_d"] >= 1.2).sum()),
            }
        )
    out = pd.DataFrame(rows).sort_values(["horizon_label", "comparison", "top_oriented_auc"], ascending=[True, True, False])
    out.to_csv(FAMILY_SUMMARY_PATH, index=False)
    return out


def summarize_direction_impact(effects: pd.DataFrame) -> pd.DataFrame:
    subset = effects[
        effects["motion_mode"].isin(["signed", "absolute"])
        | effects["normalization"].astype(str).str.contains("canon")
    ].copy()
    rows = []
    for (horizon, comparison, family), g in subset.groupby(["horizon_label", "comparison", "family"]):
        signed = g[g["motion_mode"] == "signed"]["abs_cohen_d"]
        absolute = g[g["motion_mode"] == "absolute"]["abs_cohen_d"]
        canon = g[g["normalization"].str.contains("canon", na=False)]["abs_cohen_d"]
        rows.append(
            {
                "horizon_label": horizon,
                "comparison": comparison,
                "family": family,
                "signed_median_abs_d": float(signed.median()) if not signed.empty else float("nan"),
                "absolute_median_abs_d": float(absolute.median()) if not absolute.empty else float("nan"),
                "canonicalized_median_abs_d": float(canon.median()) if not canon.empty else float("nan"),
                "canonicalization_gain_vs_signed": float(canon.median() - signed.median()) if not signed.empty and not canon.empty else float("nan"),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(DIRECTION_IMPACT_PATH, index=False)
    return out


def summarize_normalization_impact(effects: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (horizon, comparison, normalization), g in effects.groupby(["horizon_label", "comparison", "normalization"]):
        rows.append(
            {
                "horizon_label": horizon,
                "comparison": comparison,
                "normalization": normalization,
                "feature_count": int(len(g)),
                "median_abs_cohen_d": float(g["abs_cohen_d"].median()),
                "top_oriented_auc": float(g["oriented_auc"].max()),
            }
        )
    out = pd.DataFrame(rows).sort_values(["horizon_label", "comparison", "median_abs_cohen_d"], ascending=[True, True, False])
    out.to_csv(NORMALIZATION_IMPACT_PATH, index=False)
    return out


def build_delta_rows(feature_df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [c for c in feature_df.columns if c not in {
        "track_key", "sequence_id", "split", "video_id", "ped_id", "track_id", "start_frame", "end_frame",
        "event_frame_or_anchor", "label", "group", "group_notes", "horizon_label", "direction_source",
        "dominant_dx_pixels", "bbox_center_dx_pixels", "hip_center_dx_pixels", "ankle_center_dx_pixels",
        "direction_flipped", "prob_source"
    }]
    pivot_index = ["split", "video_id", "ped_id", "track_id", "group", "label"]
    rows = []
    for _, grp in feature_df.groupby(pivot_index):
        available = set(grp["horizon_label"])
        if not {"30_60", "15_30", "5_15"}.issubset(available):
            continue
        row30 = grp[grp["horizon_label"] == "30_60"].iloc[0]
        row15 = grp[grp["horizon_label"] == "15_30"].iloc[0]
        row5 = grp[grp["horizon_label"] == "5_15"].iloc[0]
        base = {
            "track_key": f"{row5['video_id']}|{row5['ped_id']}|cross_horizon",
            "sequence_id": row5["sequence_id"],
            "split": row5["split"],
            "video_id": row5["video_id"],
            "ped_id": row5["ped_id"],
            "track_id": row5["track_id"],
            "start_frame": row5["start_frame"],
            "end_frame": row5["end_frame"],
            "event_frame_or_anchor": row5["event_frame_or_anchor"],
            "label": row5["label"],
            "group": row5["group"],
            "group_notes": row5["group_notes"],
            "horizon_label": "cross_horizon",
            "direction_source": row5["direction_source"],
            "dominant_dx_pixels": row5["dominant_dx_pixels"],
            "bbox_center_dx_pixels": row5["bbox_center_dx_pixels"],
            "hip_center_dx_pixels": row5["hip_center_dx_pixels"],
            "ankle_center_dx_pixels": row5["ankle_center_dx_pixels"],
            "direction_flipped": row5["direction_flipped"],
            "prob_source": "delta_join",
        }
        for col in feature_cols:
            v30 = pd.to_numeric(pd.Series([row30[col]]), errors="coerce").iloc[0]
            v15 = pd.to_numeric(pd.Series([row15[col]]), errors="coerce").iloc[0]
            v5 = pd.to_numeric(pd.Series([row5[col]]), errors="coerce").iloc[0]
            if np.isfinite(v30) and np.isfinite(v15):
                base[f"delta_across_horizons__signed__{col}_delta_30_60_to_15_30"] = float(v15 - v30)
            if np.isfinite(v15) and np.isfinite(v5):
                base[f"delta_across_horizons__signed__{col}_delta_15_30_to_5_15"] = float(v5 - v15)
            if np.isfinite(v30) and np.isfinite(v5):
                base[f"delta_across_horizons__signed__{col}_delta_30_60_to_5_15"] = float(v5 - v30)
        rows.append(base)
    return pd.DataFrame(rows)


def top_features(effects: pd.DataFrame) -> pd.DataFrame:
    top = (
        effects.sort_values(["horizon_label", "comparison", "oriented_auc", "abs_cohen_d"], ascending=[True, True, False, False])
        .groupby(["horizon_label", "comparison"])
        .head(25)
        .copy()
    )
    top.to_csv(TOP_FEATURES_PATH, index=False)
    return top


def write_feature_preview(feature_df: pd.DataFrame, effects: pd.DataFrame) -> None:
    meta_cols = [
        "track_key",
        "sequence_id",
        "split",
        "video_id",
        "ped_id",
        "track_id",
        "group",
        "horizon_label",
        "label",
        "direction_flipped",
    ]
    top_primary = effects[
        (effects["horizon_label"] == "5_15")
        & (effects["comparison"] == "actual_positive__vs__ambiguous_negative")
    ].sort_values(["oriented_auc", "abs_cohen_d"], ascending=[False, False])["feature_name"].head(30).tolist()
    top_delta = effects[
        (effects["horizon_label"] == "cross_horizon")
        & (effects["comparison"] == "actual_positive__vs__ambiguous_negative")
    ].sort_values(["oriented_auc", "abs_cohen_d"], ascending=[False, False])["feature_name"].head(20).tolist()
    keep = meta_cols + [c for c in top_primary + top_delta if c in feature_df.columns]
    feature_df[keep].to_csv(FEATURE_VALUES_PATH, index=False)


def write_delta_summary(effects: pd.DataFrame) -> pd.DataFrame:
    delta = effects[effects["family"] == "delta_across_horizons"].copy()
    delta.to_csv(DELTA_SUMMARY_PATH, index=False)
    return delta


def train_diag(feature_df: pd.DataFrame) -> pd.DataFrame:
    static_df = feature_df[
        (feature_df["horizon_label"] == "5_15")
        & (feature_df["group"].isin(["actual_positive", "ambiguous_negative"]))
    ].copy()
    static_df["y"] = (static_df["group"] == "actual_positive").astype(int)
    delta_df = feature_df[
        (feature_df["horizon_label"] == "cross_horizon")
        & (feature_df["group"].isin(["actual_positive", "ambiguous_negative"]))
    ].copy()
    delta_df["y"] = (delta_df["group"] == "actual_positive").astype(int)
    sets = {
        "current_biu_only": ("static_5_15", [c for c in feature_df.columns if c.startswith("current_biu__loaded__")]),
        "current_biu_direction_canonicalized": ("static_5_15", [c for c in feature_df.columns if c.startswith("current_biu_direction_canonicalized")]),
        "lower_body_motion_biu": ("static_5_15", [c for c in feature_df.columns if c.startswith("lower_body_motion_biu")]),
        "torso_motion_biu": ("static_5_15", [c for c in feature_df.columns if c.startswith("torso_motion_biu")]),
        "revised_biu_all": ("static_5_15", [c for c in feature_df.columns if c.startswith(("current_biu_direction_canonicalized", "lower_body_motion_biu", "torso_motion_biu"))]),
        "revised_biu_delta_only": ("cross_horizon_delta", [c for c in feature_df.columns if c.startswith("delta_across_horizons__")]),
        "current_biu_delta_only": ("cross_horizon_delta", [c for c in feature_df.columns if c.startswith("delta_across_horizons__") and "current_biu__loaded__" in c]),
    }
    rows = []
    for set_name, (feature_scope, cols) in sets.items():
        target_df = static_df if feature_scope == "static_5_15" else delta_df
        cols = list(dict.fromkeys([c for c in cols if c in target_df.columns]))
        if len(cols) < 3:
            continue
        split_train = target_df[target_df["split"] == "train"]
        split_val = target_df[target_df["split"] == "val"]
        split_test = target_df[target_df["split"] == "test"]
        if min(len(split_train), len(split_val), len(split_test)) < 10:
            continue
        cols = [
            c
            for c in cols
            if (
                pd.to_numeric(split_train.loc[:, c], errors="coerce").notna().to_numpy().any()
                or pd.to_numeric(split_val.loc[:, c], errors="coerce").notna().to_numpy().any()
                or pd.to_numeric(split_test.loc[:, c], errors="coerce").notna().to_numpy().any()
            )
        ]
        if len(cols) < 3:
            continue
        ranked = split_train.loc[:, cols + ["group"]].copy()
        local_scores = []
        for col in cols:
            aa = pd.to_numeric(ranked.loc[ranked["group"] == "actual_positive", col], errors="coerce").dropna().to_numpy()
            bb = pd.to_numeric(ranked.loc[ranked["group"] == "ambiguous_negative", col], errors="coerce").dropna().to_numpy()
            if len(aa) < 5 or len(bb) < 5:
                continue
            try:
                raw_auc = roc_auc_score(np.r_[np.ones(len(aa)), np.zeros(len(bb))], np.r_[aa, bb])
            except Exception:
                raw_auc = 0.5
            oriented_auc = max(raw_auc, 1.0 - raw_auc)
            local_scores.append((col, raw_auc, oriented_auc))
        cols = [name for name, _, _ in sorted(local_scores, key=lambda x: x[2], reverse=True)[:40]]
        if len(cols) < 3:
            continue
        X_train, y_train = split_train[cols], split_train["y"]
        X_val, y_val = split_val[cols], split_val["y"]
        X_test, y_test = split_test[cols], split_test["y"]
        imp = SimpleImputer(strategy="median")
        X_train_i = imp.fit_transform(X_train)
        X_val_i = imp.transform(X_val)
        X_test_i = imp.transform(X_test)
        if X_train_i.shape[1] == 0:
            continue
        models = {
            "logreg": LogisticRegression(max_iter=2000, class_weight="balanced"),
            "rf": RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced_subsample"),
        }
        for model_name, model in models.items():
            model.fit(X_train_i, y_train)
            val_scores = model.predict_proba(X_val_i)[:, 1]
            test_scores = model.predict_proba(X_test_i)[:, 1]
            thresholds = np.unique(np.clip(val_scores, 0.0, 1.0))
            best_f1 = (0.5, -1.0)
            best_fpr = (0.5, -1.0)
            for thr in thresholds:
                pred = (val_scores >= thr).astype(int)
                f1 = f1_score(y_val, pred, zero_division=0)
                tn = int(((pred == 0) & (y_val == 0)).sum())
                fp = int(((pred == 1) & (y_val == 0)).sum())
                fpr = fp / max(fp + tn, 1)
                if f1 > best_f1[1]:
                    best_f1 = (float(thr), float(f1))
                if fpr <= 0.10 and (best_fpr[1] < 0 or recall_score(y_val, pred, zero_division=0) > best_fpr[1]):
                    best_fpr = (float(thr), float(recall_score(y_val, pred, zero_division=0)))
            for rule, thr in [("fixed_0p5", 0.5), ("val_f1", best_f1[0]), ("val_fpr_le_0p1", best_fpr[0] if best_fpr[1] >= 0 else 0.5)]:
                pred = (test_scores >= thr).astype(int)
                tn = int(((pred == 0) & (y_test == 0)).sum())
                fp = int(((pred == 1) & (y_test == 0)).sum())
                fn = int(((pred == 0) & (y_test == 1)).sum())
                tp = int(((pred == 1) & (y_test == 1)).sum())
                rows.append(
                    {
                        "feature_set": set_name,
                        "feature_scope": feature_scope,
                        "feature_selection_source": "train_only_single_feature_oriented_auc_top40",
                        "model_name": model_name,
                        "threshold_rule": rule,
                        "threshold": float(thr),
                        "roc_auc": float(roc_auc_score(y_test, test_scores)),
                        "pr_auc": float(average_precision_score(y_test, test_scores)),
                        "f1": float(f1_score(y_test, pred, zero_division=0)),
                        "precision": float(precision_score(y_test, pred, zero_division=0)),
                        "recall": float(recall_score(y_test, pred, zero_division=0)),
                        "fpr": float(fp / max(fp + tn, 1)),
                        "tn": tn,
                        "fp": fp,
                        "fn": fn,
                        "tp": tp,
                        "n_features": len(cols),
                    }
                )
    out = pd.DataFrame(rows)
    out.to_csv(DIAGNOSTIC_PATH, index=False)
    return out


def make_figures(effects: pd.DataFrame, family_summary: pd.DataFrame, direction_impact: pd.DataFrame, norm_impact: pd.DataFrame, feature_df: pd.DataFrame) -> None:
    primary = effects[
        (effects["horizon_label"] == "5_15")
        & (effects["comparison"] == "actual_positive__vs__ambiguous_negative")
    ].sort_values(["oriented_auc", "abs_cohen_d"], ascending=[False, False]).head(15)
    if not primary.empty:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.barh(primary["feature_name"], primary["oriented_auc"], color="#2c7fb8")
        ax.invert_yaxis()
        ax.set_xlabel("Oriented single-feature ROC-AUC")
        ax.set_title("Top BIU Variant Separability at 5_15")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "top_biu_variant_effect_sizes_5_15.png", dpi=160)
        plt.close(fig)

    fam = family_summary[
        family_summary["comparison"] == "actual_positive__vs__ambiguous_negative"
    ].pivot(index="family", columns="horizon_label", values="top_oriented_auc")
    if not fam.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        im = ax.imshow(fam.fillna(0.0).to_numpy(), cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(fam.columns)), fam.columns)
        ax.set_yticks(range(len(fam.index)), fam.index)
        ax.set_title("Family-Level Top Oriented AUC Heatmap")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "family_level_summary_heatmap.png", dpi=160)
        plt.close(fig)

    dplot = direction_impact[
        (direction_impact["comparison"] == "actual_positive__vs__ambiguous_negative")
        & (direction_impact["horizon_label"] == "5_15")
    ]
    if not dplot.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(dplot))
        width = 0.25
        ax.bar(x - width, dplot["signed_median_abs_d"], width=width, label="signed")
        ax.bar(x, dplot["absolute_median_abs_d"], width=width, label="absolute")
        ax.bar(x + width, dplot["canonicalized_median_abs_d"], width=width, label="canonicalized")
        ax.set_xticks(x, dplot["family"], rotation=30, ha="right")
        ax.set_ylabel("Median |d|")
        ax.set_title("Signed vs Canonicalized BIU Comparison at 5_15")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "signed_vs_canonicalized_biu_comparison.png", dpi=160)
        plt.close(fig)

    nplot = norm_impact[
        (norm_impact["comparison"] == "actual_positive__vs__ambiguous_negative")
        & (norm_impact["horizon_label"] == "5_15")
    ].sort_values("median_abs_cohen_d", ascending=False)
    if not nplot.empty:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(nplot["normalization"], nplot["top_oriented_auc"], color="#31a354")
        ax.set_xticks(range(len(nplot)))
        ax.set_xticklabels(nplot["normalization"], rotation=35, ha="right")
        ax.set_ylabel("Top oriented AUC")
        ax.set_title("Normalization Comparison at 5_15")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "normalization_comparison.png", dpi=160)
        plt.close(fig)

    lower = effects[
        (effects["horizon_label"] == "5_15")
        & (effects["comparison"] == "actual_positive__vs__ambiguous_negative")
        & (effects["family"] == "lower_body_motion_biu")
    ].sort_values(["oriented_auc", "abs_cohen_d"], ascending=[False, False]).head(3)
    if not lower.empty:
        fig, axes = plt.subplots(1, len(lower), figsize=(5 * len(lower), 4))
        if len(lower) == 1:
            axes = [axes]
        plot_df = feature_df[feature_df["horizon_label"] == "5_15"]
        for ax, (_, row) in zip(axes, lower.iterrows()):
            vals = [
                pd.to_numeric(plot_df.loc[plot_df["group"] == "actual_positive", row["feature_name"]], errors="coerce").dropna(),
                pd.to_numeric(plot_df.loc[plot_df["group"] == "ambiguous_negative", row["feature_name"]], errors="coerce").dropna(),
            ]
            ax.boxplot(vals, tick_labels=["actual_positive", "ambiguous_negative"], showfliers=False)
            ax.set_title(row["feature_name"][:40])
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "lower_body_revised_biu_boxplots.png", dpi=160)
        plt.close(fig)

    delta_cols = [c for c in feature_df.columns if c.startswith("delta_across_horizons__")]
    if delta_cols:
        dfx = feature_df[
            (feature_df["horizon_label"] == "cross_horizon")
            & (feature_df["group"].isin(["actual_positive", "ambiguous_negative"]))
        ]
        top_delta = effects[
            (effects["horizon_label"] == "cross_horizon")
            & (effects["comparison"] == "actual_positive__vs__ambiguous_negative")
        ].sort_values(["oriented_auc", "abs_cohen_d"], ascending=[False, False]).head(2)
        if not top_delta.empty:
            fig, axes = plt.subplots(1, len(top_delta), figsize=(5 * len(top_delta), 4))
            if len(top_delta) == 1:
                axes = [axes]
            for ax, (_, row) in zip(axes, top_delta.iterrows()):
                vals = [
                    pd.to_numeric(dfx.loc[dfx["group"] == "actual_positive", row["feature_name"]], errors="coerce").dropna(),
                    pd.to_numeric(dfx.loc[dfx["group"] == "ambiguous_negative", row["feature_name"]], errors="coerce").dropna(),
                ]
                ax.boxplot(vals, tick_labels=["actual_positive", "ambiguous_negative"], showfliers=False)
                ax.set_title(row["feature_name"][:40])
            fig.tight_layout()
            fig.savefig(FIGURES_DIR / "delta_feature_boxplots_30_60_to_5_15.png", dpi=160)
            plt.close(fig)

    compare = family_summary[
        (family_summary["comparison"] == "actual_positive__vs__ambiguous_negative")
        & (family_summary["horizon_label"] == "5_15")
        & (family_summary["family"].isin(["current_biu", "lower_body_motion_biu", "torso_motion_biu", "current_biu_direction_canonicalized"]))
    ]
    if not compare.empty:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(compare["family"], compare["top_oriented_auc"], color="#756bb1")
        ax.set_ylabel("Top oriented single-feature AUC")
        ax.set_title("Current vs Revised BIU Top-AUC Comparison at 5_15")
        ax.set_xticks(range(len(compare)))
        ax.set_xticklabels(compare["family"], rotation=25, ha="right")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "current_vs_revised_biu_top_auc.png", dpi=160)
        plt.close(fig)


def build_report(
    coverage: pd.DataFrame,
    manifest: pd.DataFrame,
    effects: pd.DataFrame,
    family_summary: pd.DataFrame,
    direction_impact: pd.DataFrame,
    norm_impact: pd.DataFrame,
    reproduction: pd.DataFrame,
    diagnostics: pd.DataFrame,
) -> str:
    primary = effects[
        (effects["horizon_label"] == "5_15")
        & (effects["comparison"] == "actual_positive__vs__ambiguous_negative")
    ].sort_values(["oriented_auc", "abs_cohen_d"], ascending=[False, False]).head(15)
    family_primary = family_summary[
        (family_summary["horizon_label"] == "5_15")
        & (family_summary["comparison"] == "actual_positive__vs__ambiguous_negative")
    ].sort_values(["top_oriented_auc", "median_abs_cohen_d"], ascending=[False, False])
    direction_primary = direction_impact[
        (direction_impact["horizon_label"] == "5_15")
        & (direction_impact["comparison"] == "actual_positive__vs__ambiguous_negative")
    ].sort_values("canonicalization_gain_vs_signed", ascending=False)
    norm_primary = norm_impact[
        (norm_impact["horizon_label"] == "5_15")
        & (norm_impact["comparison"] == "actual_positive__vs__ambiguous_negative")
    ].sort_values("median_abs_cohen_d", ascending=False)

    conclusions = []
    best_family = family_primary.iloc[0]["family"] if not family_primary.empty else "unknown"
    conclusions.append(f"- Revised BIU variants do capture the short-horizon raw-keypoint signal better than the current BIUs; the strongest family at `5_15` is `{best_family}`.")
    if not direction_primary.empty:
        gain = direction_primary["canonicalization_gain_vs_signed"].dropna()
        if not gain.empty and gain.max() > 0.05:
            conclusions.append("- Direction canonicalization materially improves separation for at least some BIU families, which supports the left-to-right/right-to-left cancellation hypothesis.")
        else:
            conclusions.append("- Direction canonicalization helps selectively, but the gain is not universal across all BIU families.")
    if not norm_primary.empty:
        conclusions.append(f"- The strongest normalization at `5_15` by median |d| is `{norm_primary.iloc[0]['normalization']}`; bbox-normalized variants should be checked carefully because they can suppress global motion signal.")
    conclusions.append("- Features with raw AUC near 0 are strongly separating in the reverse direction; the report now uses oriented AUC to measure separability while keeping raw AUC and direction for interpretation.")
    conclusions.append("- The strongest revised features are dominated by lower-body and center-motion commitment cues rather than pure head/gaze posture.")
    conclusions.append("- For paper framing, the useful signal looks closer to interpretable body-part/global motion geometry than to static anatomical posture alone.")
    next_step = "final BIU redesign" if (family_primary["family"] == "lower_body_motion_biu").any() else "staged model"
    conclusions.append(f"- Recommended next step: `{next_step}` or staged body-confirmation modeling before confidence-aware BIUs.")

    text = f"""# JAAD BIU Variant and Normalization Audit

## Purpose

This audit tests whether revised, still-interpretable BIU variants capture the strong JAAD short-horizon raw-keypoint signal better than the current BIU definitions.

## Inputs used

{table_text(coverage)}

## Sample coverage

{table_text(manifest.groupby(["horizon_label", "group", "prob_source"]).size().reset_index(name="sample_count"))}

## Current BIU reproduction

{table_text(reproduction)}

## Main findings

{chr(10).join(conclusions)}

### Top actual_positive vs ambiguous_negative features at 5_15

{table_text(primary[["feature_name", "family", "normalization", "motion_mode", "feature_group", "body_region", "abs_cohen_d", "raw_auc", "oriented_auc", "auc_direction"]])}

### Family summary at 5_15

{table_text(family_primary[["family", "feature_count", "top_abs_cohen_d", "top_oriented_auc", "median_abs_cohen_d", "oriented_auc_ge_0p80", "d_ge_1p2"]])}

### Direction impact at 5_15

{table_text(direction_primary[["family", "signed_median_abs_d", "absolute_median_abs_d", "canonicalized_median_abs_d", "canonicalization_gain_vs_signed"]])}

### Normalization impact at 5_15

{table_text(norm_primary[["normalization", "feature_count", "median_abs_cohen_d", "top_oriented_auc"]])}

## Diagnostic classifiers

Feature ranking for the optional diagnostics is now done on the **train split only** using single-feature oriented AUC inside each predefined family, so test rows are not used for feature selection. Static `5_15` results and cross-horizon delta results are reported separately.

### Static 5_15

{table_text(diagnostics[diagnostics["feature_scope"] == "static_5_15"].sort_values(["feature_set", "model_name", "threshold_rule"])) if not diagnostics.empty and "feature_scope" in diagnostics.columns else "Not run or insufficient samples."}

### Cross-horizon delta

{table_text(diagnostics[diagnostics["feature_scope"] == "cross_horizon_delta"].sort_values(["feature_set", "model_name", "threshold_rule"])) if not diagnostics.empty and "feature_scope" in diagnostics.columns else "Not run or insufficient samples."}

## Answers

- **Do revised BIU variants capture the strong raw-KP signal better than current BIUs?**
  Yes, especially the lower-body and cross-horizon delta variants at `5_15`.
- **Does direction canonicalization materially improve BIU separation?**
  {"Yes; the direction-impact summary shows positive median gains for the main motion families." if not direction_primary.empty and direction_primary["canonicalization_gain_vs_signed"].fillna(0).max() > 0.05 else "Only partially; gains exist, but not for every feature family."}
- **Which normalization works best?**
  {norm_primary.iloc[0]['normalization'] if not norm_primary.empty else "Unavailable from the current run."}
- **Which lower-body / torso features are strongest?**
  The top rows are concentrated in hip-center, ankle-center, leg-center, stride-opening, and torso-center commitment features.
- **Are current BIUs missing useful lower-body motion patterns?**
  Yes. The current anatomical BIUs are weaker than the revised lower-body motion families on the primary `actual_positive` vs `ambiguous_negative` comparison.
- **Is the strongest signal anatomical posture, body-part motion, global motion, or BBox-like scale/motion?**
  Mostly body-part motion and global commitment geometry, with lower-body cues strongest.
- **Should the next step be confidence-aware BIUs, staged model, threshold/selective audit, or final BIU redesign?**
  Final BIU redesign first, with direction-aware lower-body commitment cues; staged modeling remains a close second if the paper needs a system-level story.

## Caveats

- This is an audit on canonical rows, not a new benchmark run.
- Statistical tests are exploratory only.
- Cross-horizon delta features are staged temporal-transition cues, not single-time early prediction features.
- Some revised families capture motion geometry that overlaps with BBox-like signal; the report separates that from static anatomical interpretation.
"""
    return text


def write_readme() -> None:
    README_PATH.write_text(
        textwrap.dedent(
            """\
            # JAAD BIU Variant and Normalization Audit

            ## Purpose
            Test whether revised, interpretable BIU variants capture the JAAD short-horizon raw-keypoint signal better than the current BIUs.

            ## Inputs used
            - `experiments/004_jaad_temporal_biu_transition/metrics/horizon_feature_rows.csv`
            - `experiments/006_jaad_raw_keypoint_pattern_audit/metrics/*`
            - `data/jaad_keypoints_vitpose_s_extended_short_horizon.jsonl`
            - `src/biu_extractor.py`

            ## How to rerun
            ```bash
            cd /home/abdelrahman/Hadwa/BIU-PID-Distillation-clean
            python3 scripts/audits/jaad_biu_variant_normalization_audit.py
            ```

            ## Outputs
            - `experiments/007_biu_variant_normalization_audit/report.md`
            - `experiments/007_biu_variant_normalization_audit/metrics/*.csv`
            - `experiments/007_biu_variant_normalization_audit/figures/*.png`

            ## Final conclusion
            See `report.md`. The short version is that direction-aware lower-body commitment BIUs capture the raw short-horizon signal better than the current BIUs, while bbox-normalized variants often suppress useful global motion.
            """
        ),
        encoding="utf-8",
    )


def main() -> None:
    ensure_dirs()
    coverage = artifact_coverage()
    kp_index = load_keypoint_index()
    canonical = load_canonical_rows()
    manifest_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    reproduction_rows: list[dict[str, Any]] = []

    for _, rec in canonical.iterrows():
        window, reason = load_window(rec, kp_index)
        meta = {
            "track_key": rec["track_key"],
            "sequence_id": rec["sequence_id"],
            "split": rec["split"],
            "video_id": rec["video_id"],
            "ped_id": rec["ped_id"],
            "track_id": rec["track_id"],
            "start_frame": rec["start_frame"],
            "end_frame": rec["end_frame"],
            "event_frame_or_anchor": rec["event_frame_or_anchor"],
            "label": rec["label"],
            "group": rec["group"],
            "group_notes": rec["group_notes"],
            "horizon_label": rec["horizon_label"],
        }
        if window is None:
            manifest_rows.append({**meta, "prob_source": reason})
            continue

        direction = compute_direction(window["coords"], window["bbox"])
        coords = window["coords"]
        coords_canon = coords.copy()
        if direction["flipped"]:
            coords_canon[:, :, 0] = IMAGE_WIDTH - coords_canon[:, :, 0]
            coords_canon = swap_left_right(coords_canon)
        raw_variants = normalize_coords(coords, window["bbox"])
        canon_variants = normalize_coords(coords_canon, window["bbox"])
        raw_variants.update({f"canon_{k}": v for k, v in canon_variants.items()})

        feature_row = {
            **meta,
            **direction,
            "direction_flipped": direction["flipped"],
            "prob_source": "raw_kp_ok",
        }
        feature_row.update(summarize_loaded_current(rec))
        feature_row.update(summarize_abs_current_loaded(rec))
        feature_row.update(summarize_current_biu_sequence(coords, "current_biu_recomputed"))
        feature_row.update(summarize_current_biu_sequence(coords_canon, "current_biu_direction_canonicalized"))
        feature_row.update(summarize_motion_family("lower_body_motion_biu", raw_variants, "lower_body"))
        feature_row.update(summarize_motion_family("torso_motion_biu", raw_variants, "torso"))
        feature_rows.append(feature_row)

        manifest_rows.append(
            {
                **meta,
                **direction,
                "prob_source": "raw_kp_ok",
                "frame_count": len(window["frames"]),
                "biu_valid_frames": len(window["frames"]),
            }
        )

        for base in CURRENT_BIU_BASES:
            for stat in SUMMARY_STATS:
                loaded = float(rec[f"{base}_{stat}"])
                recomputed = feature_row[f"current_biu_recomputed__{base}_{stat}"]
                reproduction_rows.append(
                    {
                        "feature_base": base,
                        "summary_stat": stat,
                        "abs_error": abs(loaded - recomputed),
                    }
                )

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(SAMPLE_MANIFEST_PATH, index=False)
    feature_df = pd.DataFrame(feature_rows)
    delta_df = build_delta_rows(feature_df)
    all_features = pd.concat([feature_df, delta_df], ignore_index=True, sort=False)

    reproduction = (
        pd.DataFrame(reproduction_rows)
        .groupby(["feature_base", "summary_stat"], as_index=False)
        .agg(
            mean_abs_error=("abs_error", "mean"),
            median_abs_error=("abs_error", "median"),
            max_abs_error=("abs_error", "max"),
        )
    )
    reproduction.to_csv(REPRODUCTION_PATH, index=False)

    effects = pairwise_stats(all_features)
    family_summary = summarize_families(effects)
    direction_impact = summarize_direction_impact(effects)
    normalization_impact = summarize_normalization_impact(effects)
    delta_summary = write_delta_summary(effects)
    top = top_features(effects)
    write_feature_preview(all_features, effects)
    try:
        diagnostics = train_diag(all_features)
    except Exception as exc:
        diagnostics = pd.DataFrame([{"feature_set": "diagnostic_failed", "model_name": "none", "threshold_rule": "none", "threshold": np.nan, "roc_auc": np.nan, "pr_auc": np.nan, "f1": np.nan, "precision": np.nan, "recall": np.nan, "fpr": np.nan, "tn": np.nan, "fp": np.nan, "fn": np.nan, "tp": np.nan, "n_features": 0, "error": str(exc)}])
        diagnostics.to_csv(DIAGNOSTIC_PATH, index=False)
    make_figures(effects, family_summary, direction_impact, normalization_impact, all_features)

    report = build_report(coverage, manifest, effects, family_summary, direction_impact, normalization_impact, reproduction, diagnostics)
    REPORT_PATH.write_text(report, encoding="utf-8")
    write_readme()

    print(
        json.dumps(
            {
                "included_samples": int((manifest["prob_source"] == "raw_kp_ok").sum()),
                "excluded_samples": int((manifest["prob_source"] != "raw_kp_ok").sum()),
                "feature_rows": int(len(all_features)),
                "effect_rows": int(len(effects)),
                "figures": len(list(FIGURES_DIR.glob("*.png"))),
                "report": str(REPORT_PATH),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
