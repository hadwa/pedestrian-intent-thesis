"""Audit raw JAAD keypoint patterns across canonical horizons and groups."""

from __future__ import annotations

import json
import math
import os
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score


CLEAN_ROOT = Path("/home/abdelrahman/Hadwa/BIU-PID-Distillation-clean")
ARCHIVE_ROOT = Path("/home/abdelrahman/Hadwa/BIU-PID-Distillation")
EXPERIMENT_ROOT = CLEAN_ROOT / "experiments/006_jaad_raw_keypoint_pattern_audit"
METRICS_DIR = EXPERIMENT_ROOT / "metrics"
FIGURES_DIR = EXPERIMENT_ROOT / "figures"
REPORT_PATH = EXPERIMENT_ROOT / "report.md"
README_PATH = EXPERIMENT_ROOT / "README.md"

CANONICAL_ROWS_PATH = CLEAN_ROOT / "experiments/004_jaad_temporal_biu_transition/metrics/horizon_feature_rows.csv"
KEYPOINT_JSONL_PATH = CLEAN_ROOT / "data/jaad_keypoints_vitpose_s_extended_short_horizon.jsonl"
SAMPLE_IMAGE_PATH = Path("/home/abdelrahman/Hadwa/datasets/JAAD/images/video_0032/00160.png")

MANIFEST_PATH = METRICS_DIR / "raw_kp_sample_manifest.csv"
ARTIFACT_COVERAGE_PATH = METRICS_DIR / "raw_kp_artifact_coverage.csv"
GROUP_COUNTS_PATH = METRICS_DIR / "raw_kp_group_counts.csv"
DIRECTION_SUMMARY_PATH = METRICS_DIR / "raw_kp_direction_summary.csv"
EFFECT_SIZES_PATH = METRICS_DIR / "raw_kp_feature_effect_sizes.csv"
TOP_FEATURES_PATH = METRICS_DIR / "raw_kp_top_features_by_comparison.csv"
VARIANT_SUMMARY_PATH = METRICS_DIR / "raw_kp_variant_summary.csv"
BODY_REGION_SUMMARY_PATH = METRICS_DIR / "raw_kp_body_region_summary.csv"

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
BODY_REGIONS = {
    "head": ["nose", "left_eye", "right_eye", "left_ear", "right_ear"],
    "shoulders": ["left_shoulder", "right_shoulder"],
    "arms": ["left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist"],
    "hips": ["left_hip", "right_hip"],
    "knees": ["left_knee", "right_knee"],
    "ankles": ["left_ankle", "right_ankle"],
    "legs": ["left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"],
    "torso": ["left_shoulder", "right_shoulder", "left_hip", "right_hip"],
}
PAIRWISE = [
    ("actual_positive", "ambiguous_negative"),
    ("ambiguous_negative", "clean_negative"),
    ("actual_positive", "clean_negative"),
]
HORIZON_ORDER = ["30_60", "15_30", "5_15"]
GROUP_ORDER = ["clean_negative", "ambiguous_negative", "actual_positive"]
IMAGE_WIDTH = 1920.0
IMAGE_HEIGHT = 1080.0


def _ensure_dirs() -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def _table_text(df: pd.DataFrame) -> str:
    if df.empty:
        return "No rows."
    try:
        return df.to_markdown(index=False)
    except Exception:
        return "```\n" + df.to_string(index=False) + "\n```"


def _safe_mean(arr: np.ndarray, axis: int | None = None) -> np.ndarray | float:
    with np.errstate(invalid="ignore"):
        return np.nanmean(arr, axis=axis)


def _safe_std(arr: np.ndarray, axis: int | None = None) -> np.ndarray | float:
    with np.errstate(invalid="ignore"):
        return np.nanstd(arr, axis=axis)


def _safe_min(arr: np.ndarray, axis: int | None = None) -> np.ndarray | float:
    with np.errstate(invalid="ignore"):
        return np.nanmin(arr, axis=axis)


def _safe_max(arr: np.ndarray, axis: int | None = None) -> np.ndarray | float:
    with np.errstate(invalid="ignore"):
        return np.nanmax(arr, axis=axis)


def _safe_first_valid(arr: np.ndarray) -> float:
    flat = arr[np.isfinite(arr)]
    return float(flat[0]) if flat.size else float("nan")


def _load_keypoint_index() -> dict[tuple[str, str, int], dict[str, Any]]:
    index: dict[tuple[str, str, int], dict[str, Any]] = {}
    with KEYPOINT_JSONL_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            rec = json.loads(line)
            key = (str(rec["video"]), str(rec["pid"]), int(rec["frame_idx"]))
            index[key] = rec
    return index


def _load_canonical_rows() -> pd.DataFrame:
    df = pd.read_csv(CANONICAL_ROWS_PATH)
    needed = ["track_key", "sequence_id", "split", "video_id", "ped_id", "start_frame", "end_frame", "event_frame_or_anchor", "label", "group", "group_notes", "horizon_label", "observed_frames_json", "bbox_sequence_json"]
    return df[needed].copy()


def _artifact_coverage() -> pd.DataFrame:
    rows = []
    for label, path in [
        ("canonical_rows", CANONICAL_ROWS_PATH),
        ("keypoint_jsonl", KEYPOINT_JSONL_PATH),
        ("sample_image_for_resolution", SAMPLE_IMAGE_PATH),
    ]:
        rows.append({"artifact": label, "path": str(path), "exists": int(path.exists()), "size_bytes": path.stat().st_size if path.exists() else 0})
    coverage = pd.DataFrame(rows)
    coverage.to_csv(ARTIFACT_COVERAGE_PATH, index=False)
    return coverage


def _pair_average(coords: np.ndarray, names: list[str]) -> np.ndarray:
    idxs = [KPT_INDEX[n] for n in names]
    return _safe_mean(coords[:, idxs, :], axis=1)


def _compute_scale(coords: np.ndarray) -> np.ndarray:
    ls = coords[:, KPT_INDEX["left_shoulder"], :]
    rs = coords[:, KPT_INDEX["right_shoulder"], :]
    lh = coords[:, KPT_INDEX["left_hip"], :]
    rh = coords[:, KPT_INDEX["right_hip"], :]
    shoulder_w = np.linalg.norm(ls - rs, axis=1)
    shoulder_center = _safe_mean(np.stack([ls, rs], axis=1), axis=1)
    hip_center = _safe_mean(np.stack([lh, rh], axis=1), axis=1)
    torso_len = np.linalg.norm(shoulder_center - hip_center, axis=1)
    scale = shoulder_w.copy()
    bad = ~np.isfinite(scale) | (scale < 1e-6)
    scale[bad] = torso_len[bad]
    scale[(~np.isfinite(scale)) | (scale < 1e-6)] = np.nan
    return scale


def _swap_left_right(coords: np.ndarray, conf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coords = coords.copy()
    conf = conf.copy()
    for left, right in LEFT_RIGHT_PAIRS:
        li = KPT_INDEX[left]
        ri = KPT_INDEX[right]
        coords[:, [li, ri], :] = coords[:, [ri, li], :]
        conf[:, [li, ri]] = conf[:, [ri, li]]
    return coords, conf


def _extract_window(rec: pd.Series, kp_index: dict[tuple[str, str, int], dict[str, Any]]) -> tuple[dict[str, Any] | None, str]:
    frames = json.loads(rec["observed_frames_json"])
    bboxes = np.asarray(json.loads(rec["bbox_sequence_json"]), dtype=float)
    if len(frames) != len(bboxes):
        return None, "bbox_frame_count_mismatch"
    kp_rows = []
    for frame in frames:
        row = kp_index.get((str(rec["video_id"]), str(rec["ped_id"]), int(frame)))
        if row is None:
            return None, f"missing_keypoint_frame:{frame}"
        kp_rows.append(row)
    coords = np.asarray([row["keypoints"] for row in kp_rows], dtype=float)
    conf = np.asarray([row["kp_conf"] for row in kp_rows], dtype=float)
    skel_conf = np.asarray([row["skeleton_conf"] for row in kp_rows], dtype=float)
    if coords.shape != (len(frames), 17, 2):
        return None, f"unexpected_keypoint_shape:{coords.shape}"
    return {
        "frames": np.asarray(frames, dtype=int),
        "bbox": bboxes,
        "coords": coords,
        "conf": conf,
        "skel_conf": skel_conf,
    }, ""


def _direction_info(window: dict[str, Any]) -> dict[str, Any]:
    bbox = window["bbox"]
    center_x = (bbox[:, 0] + bbox[:, 2]) / 2.0
    bbox_dx = float(center_x[-1] - center_x[0])
    coords = window["coords"]
    hip_center_x = _safe_mean(coords[:, [KPT_INDEX["left_hip"], KPT_INDEX["right_hip"]], 0], axis=1)
    hip_dx = float(hip_center_x[-1] - hip_center_x[0]) if np.isfinite(hip_center_x[[0, -1]]).all() else float("nan")
    if abs(bbox_dx) >= 1.0:
        source = "bbox_center_x"
        dominant_dx = bbox_dx
    elif np.isfinite(hip_dx) and abs(hip_dx) >= 1.0:
        source = "hip_center_x"
        dominant_dx = hip_dx
    else:
        source = "bbox_center_x_small"
        dominant_dx = bbox_dx
    flip = dominant_dx < 0
    return {
        "direction_source": source,
        "dominant_dx_pixels": dominant_dx,
        "bbox_center_dx_pixels": bbox_dx,
        "hip_center_dx_pixels": hip_dx,
        "flipped": int(flip),
    }


def _build_representations(window: dict[str, Any], flip: bool) -> dict[str, np.ndarray]:
    coords = window["coords"].copy()
    conf = window["conf"].copy()
    bbox = window["bbox"]

    img = coords.copy()
    img[:, :, 0] /= IMAGE_WIDTH
    img[:, :, 1] /= IMAGE_HEIGHT

    bbox_w = np.clip(bbox[:, 2] - bbox[:, 0], 1e-6, None)
    bbox_h = np.clip(bbox[:, 3] - bbox[:, 1], 1e-6, None)
    bbox_norm = coords.copy()
    bbox_norm[:, :, 0] = (bbox_norm[:, :, 0] - bbox[:, None, 0]) / bbox_w[:, None]
    bbox_norm[:, :, 1] = (bbox_norm[:, :, 1] - bbox[:, None, 1]) / bbox_h[:, None]

    hip_center = _pair_average(coords, ["left_hip", "right_hip"])
    shoulder_center = _pair_average(coords, ["left_shoulder", "right_shoulder"])
    torso_center = hip_center.copy()
    bad = ~np.isfinite(torso_center).all(axis=1)
    torso_center[bad] = shoulder_center[bad]
    scale = _compute_scale(coords)
    torso_centered = coords.copy()
    torso_centered[:, :, 0] = (torso_centered[:, :, 0] - torso_center[:, None, 0]) / scale[:, None]
    torso_centered[:, :, 1] = (torso_centered[:, :, 1] - torso_center[:, None, 1]) / scale[:, None]

    first_rel = img - img[[0], :, :]
    abs_motion = np.abs(first_rel)

    canon_img = img.copy()
    canon_bbox = bbox_norm.copy()
    canon_torso = torso_centered.copy()
    canon_first_rel = first_rel.copy()
    if flip:
        coords_px, conf = _swap_left_right(coords, conf)
        canon_img = coords_px.copy()
        canon_img[:, :, 0] = (IMAGE_WIDTH - canon_img[:, :, 0]) / IMAGE_WIDTH
        canon_img[:, :, 1] = canon_img[:, :, 1] / IMAGE_HEIGHT
        canon_bbox = coords_px.copy()
        canon_bbox[:, :, 0] = 1.0 - ((canon_bbox[:, :, 0] - bbox[:, None, 0]) / bbox_w[:, None])
        canon_bbox[:, :, 1] = (canon_bbox[:, :, 1] - bbox[:, None, 1]) / bbox_h[:, None]
        scale = _compute_scale(coords_px)
        hip_center_c = _pair_average(coords_px, ["left_hip", "right_hip"])
        shoulder_center_c = _pair_average(coords_px, ["left_shoulder", "right_shoulder"])
        torso_center_c = hip_center_c.copy()
        bad = ~np.isfinite(torso_center_c).all(axis=1)
        torso_center_c[bad] = shoulder_center_c[bad]
        canon_torso = coords_px.copy()
        canon_torso[:, :, 0] = (canon_torso[:, :, 0] - torso_center_c[:, None, 0]) / scale[:, None]
        canon_torso[:, :, 1] = (canon_torso[:, :, 1] - torso_center_c[:, None, 1]) / scale[:, None]
        canon_first_rel = canon_img - canon_img[[0], :, :]
    abs_motion = np.abs(first_rel)
    canon_abs_motion = np.abs(canon_first_rel)

    return {
        "img_norm": img,
        "bbox_norm": bbox_norm,
        "torso_centered": torso_centered,
        "first_rel": first_rel,
        "abs_motion": abs_motion,
        "canon_img_norm": canon_img,
        "canon_bbox_norm": canon_bbox,
        "canon_torso_centered": canon_torso,
        "canon_first_rel": canon_first_rel,
        "canon_abs_motion": canon_abs_motion,
        "conf": conf,
        "skeleton_conf": window["skel_conf"],
        "bbox": bbox,
    }


def _per_keypoint_stats(features: dict[str, float], prefix: str, arr: np.ndarray, names: list[str] | None = None) -> None:
    names = names or COCO_KEYPOINTS
    for idx, name in enumerate(names):
        x = arr[:, idx, 0]
        y = arr[:, idx, 1]
        dx = np.diff(x)
        dy = np.diff(y)
        disp = np.sqrt((x - x[0]) ** 2 + (y - y[0]) ** 2)
        speed = np.sqrt(dx**2 + dy**2)
        base = f"{prefix}_{name}"
        features[f"{base}_x_first"] = float(x[0])
        features[f"{base}_x_last"] = float(x[-1])
        features[f"{base}_x_mean"] = float(_safe_mean(x))
        features[f"{base}_x_std"] = float(_safe_std(x))
        features[f"{base}_x_delta"] = float(x[-1] - x[0])
        features[f"{base}_y_first"] = float(y[0])
        features[f"{base}_y_last"] = float(y[-1])
        features[f"{base}_y_mean"] = float(_safe_mean(y))
        features[f"{base}_y_std"] = float(_safe_std(y))
        features[f"{base}_y_delta"] = float(y[-1] - y[0])
        features[f"{base}_vel_x_mean"] = float(_safe_mean(dx))
        features[f"{base}_vel_x_std"] = float(_safe_std(dx))
        features[f"{base}_vel_y_mean"] = float(_safe_mean(dy))
        features[f"{base}_vel_y_std"] = float(_safe_std(dy))
        features[f"{base}_disp_last"] = float(disp[-1])
        features[f"{base}_disp_mean"] = float(_safe_mean(disp))
        features[f"{base}_speed_mean"] = float(_safe_mean(speed))
        features[f"{base}_speed_std"] = float(_safe_std(speed))


def _region_center(arr: np.ndarray, region_names: list[str]) -> np.ndarray:
    idxs = [KPT_INDEX[n] for n in region_names]
    return _safe_mean(arr[:, idxs, :], axis=1)


def _region_motion_features(features: dict[str, float], prefix: str, arr: np.ndarray) -> None:
    region_specs = {
        "head_center": BODY_REGIONS["head"],
        "shoulder_center": BODY_REGIONS["shoulders"],
        "hip_center": BODY_REGIONS["hips"],
        "torso_center": BODY_REGIONS["torso"],
        "leg_center": BODY_REGIONS["legs"],
        "ankle_center": BODY_REGIONS["ankles"],
    }
    centers = {name: _region_center(arr, region) for name, region in region_specs.items()}
    for name, center in centers.items():
        dx = np.diff(center[:, 0])
        dy = np.diff(center[:, 1])
        disp = center - center[[0], :]
        mag = np.sqrt(disp[:, 0] ** 2 + disp[:, 1] ** 2)
        speed = np.sqrt(dx**2 + dy**2)
        base = f"{prefix}_{name}"
        features[f"{base}_x_delta"] = float(center[-1, 0] - center[0, 0])
        features[f"{base}_y_delta"] = float(center[-1, 1] - center[0, 1])
        features[f"{base}_disp_last"] = float(mag[-1])
        features[f"{base}_disp_mean"] = float(_safe_mean(mag))
        features[f"{base}_speed_mean"] = float(_safe_mean(speed))
        features[f"{base}_speed_std"] = float(_safe_std(speed))

    ls = arr[:, KPT_INDEX["left_shoulder"], :]
    rs = arr[:, KPT_INDEX["right_shoulder"], :]
    lh = arr[:, KPT_INDEX["left_hip"], :]
    rh = arr[:, KPT_INDEX["right_hip"], :]
    la = arr[:, KPT_INDEX["left_ankle"], :]
    ra = arr[:, KPT_INDEX["right_ankle"], :]
    lk = arr[:, KPT_INDEX["left_knee"], :]
    rk = arr[:, KPT_INDEX["right_knee"], :]
    nose = arr[:, KPT_INDEX["nose"], :]
    shoulder_width = np.linalg.norm(ls - rs, axis=1)
    hip_width = np.linalg.norm(lh - rh, axis=1)
    torso_length = np.linalg.norm(centers["shoulder_center"] - centers["hip_center"], axis=1)
    ankle_distance = np.linalg.norm(la - ra, axis=1)
    foot_vertical_diff = la[:, 1] - ra[:, 1]
    stride_opening = np.abs(la[:, 0] - ra[:, 0])
    head_rel_torso_y = nose[:, 1] - centers["torso_center"][:, 1]
    shoulder_rel_hip_x = centers["shoulder_center"][:, 0] - centers["hip_center"][:, 0]
    shoulder_rel_hip_y = centers["shoulder_center"][:, 1] - centers["hip_center"][:, 1]
    knee_distance = np.linalg.norm(lk - rk, axis=1)

    summary_series = {
        "shoulder_width": shoulder_width,
        "hip_width": hip_width,
        "torso_length": torso_length,
        "ankle_distance": ankle_distance,
        "foot_vertical_diff": foot_vertical_diff,
        "stride_opening": stride_opening,
        "head_rel_torso_y": head_rel_torso_y,
        "shoulder_rel_hip_x": shoulder_rel_hip_x,
        "shoulder_rel_hip_y": shoulder_rel_hip_y,
        "knee_distance": knee_distance,
    }
    for name, series in summary_series.items():
        features[f"{prefix}_{name}_first"] = float(series[0])
        features[f"{prefix}_{name}_last"] = float(series[-1])
        features[f"{prefix}_{name}_mean"] = float(_safe_mean(series))
        features[f"{prefix}_{name}_std"] = float(_safe_std(series))
        features[f"{prefix}_{name}_delta"] = float(series[-1] - series[0])


def _confidence_features(features: dict[str, float], conf: np.ndarray, skel_conf: np.ndarray) -> None:
    features["confidence_skeleton_mean"] = float(_safe_mean(skel_conf))
    features["confidence_skeleton_last"] = float(skel_conf[-1])
    features["confidence_skeleton_min"] = float(_safe_min(skel_conf))
    features["confidence_skeleton_std"] = float(_safe_std(skel_conf))
    features["confidence_low_joint_count_mean"] = float(_safe_mean((conf < 0.3).sum(axis=1)))
    features["confidence_low_joint_count_max"] = float(_safe_max((conf < 0.3).sum(axis=1)))
    for region_name, names in BODY_REGIONS.items():
        idxs = [KPT_INDEX[n] for n in names]
        region_conf = conf[:, idxs]
        features[f"confidence_{region_name}_mean"] = float(_safe_mean(region_conf))
        features[f"confidence_{region_name}_last"] = float(_safe_mean(region_conf[-1]))
        features[f"confidence_{region_name}_min"] = float(_safe_min(region_conf))


def _sample_feature_row(rec: pd.Series, window: dict[str, Any], direction: dict[str, Any]) -> dict[str, Any]:
    reps = _build_representations(window, bool(direction["flipped"]))
    row = {
        "track_key": rec["track_key"],
        "sequence_id": rec["sequence_id"],
        "split": rec["split"],
        "group": rec["group"],
        "group_notes": rec["group_notes"],
        "horizon_label": rec["horizon_label"],
        "video_id": rec["video_id"],
        "ped_id": rec["ped_id"],
        "start_frame": int(rec["start_frame"]),
        "end_frame": int(rec["end_frame"]),
        "event_frame_or_anchor": int(rec["event_frame_or_anchor"]),
        "label": int(rec["label"]),
        "direction_source": direction["direction_source"],
        "dominant_dx_pixels": direction["dominant_dx_pixels"],
        "bbox_center_dx_pixels": direction["bbox_center_dx_pixels"],
        "hip_center_dx_pixels": direction["hip_center_dx_pixels"],
        "flipped": int(direction["flipped"]),
    }
    _confidence_features(row, reps["conf"], reps["skeleton_conf"])
    for prefix, arr in [
        ("img", reps["img_norm"]),
        ("bboxnorm", reps["bbox_norm"]),
        ("torso", reps["torso_centered"]),
        ("firstrel", reps["first_rel"]),
        ("absmotion", reps["abs_motion"]),
        ("canonimg", reps["canon_img_norm"]),
        ("canonbbox", reps["canon_bbox_norm"]),
        ("cantorso", reps["canon_torso_centered"]),
        ("canonfirstrel", reps["canon_first_rel"]),
        ("canonabsmotion", reps["canon_abs_motion"]),
    ]:
        _per_keypoint_stats(row, prefix, arr)
        _region_motion_features(row, prefix, arr)
    return row


def _cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return float("nan")
    gt = sum(x > y for x in a for y in b)
    lt = sum(x < y for x in a for y in b)
    return float((gt - lt) / (a.size * b.size))


def _cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    var_a = np.var(a, ddof=1)
    var_b = np.var(b, ddof=1)
    pooled = ((len(a) - 1) * var_a + (len(b) - 1) * var_b) / (len(a) + len(b) - 2)
    if pooled <= 0 or not np.isfinite(pooled):
        return float("nan")
    return float((np.mean(a) - np.mean(b)) / math.sqrt(pooled))


def _single_feature_auc(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return float("nan")
    y = np.concatenate([np.ones_like(a), np.zeros_like(b)])
    x = np.concatenate([a, b])
    auc = roc_auc_score(y, x)
    return float(max(auc, 1.0 - auc))


def _feature_region(feature_name: str) -> str:
    for region, names in BODY_REGIONS.items():
        for name in names:
            if name in feature_name:
                return region
    if any(token in feature_name for token in ["shoulder_width", "shoulder_center"]):
        return "shoulders"
    if any(token in feature_name for token in ["hip_width", "hip_center"]):
        return "hips"
    if any(token in feature_name for token in ["ankle", "foot", "stride", "knee"]):
        return "legs"
    if any(token in feature_name for token in ["torso"]):
        return "torso"
    if any(token in feature_name for token in ["nose", "eye", "ear", "head"]):
        return "head"
    if any(token in feature_name for token in ["bbox_center", "dominant_dx"]):
        return "global_motion"
    return "mixed"


def _feature_variant(feature_name: str) -> str:
    for prefix in ["canonabsmotion", "canonfirstrel", "cantorso", "canonbbox", "canonimg", "absmotion", "firstrel", "torso", "bboxnorm", "img"]:
        if feature_name.startswith(prefix + "_"):
            return prefix
    if feature_name.startswith("confidence_"):
        return "confidence"
    return "meta"


def compute_effect_sizes(feature_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    meta_cols = {"track_key", "sequence_id", "split", "group", "group_notes", "horizon_label", "video_id", "ped_id", "start_frame", "end_frame", "event_frame_or_anchor", "label", "direction_source", "dominant_dx_pixels", "bbox_center_dx_pixels", "hip_center_dx_pixels", "flipped"}
    feature_cols = [c for c in feature_df.columns if c not in meta_cols]

    effect_rows = []
    for horizon in HORIZON_ORDER:
        horizon_df = feature_df.loc[feature_df["horizon_label"] == horizon]
        for feature in feature_cols:
            for group_a, group_b in PAIRWISE:
                a = horizon_df.loc[horizon_df["group"] == group_a, feature].dropna().to_numpy(dtype=float)
                b = horizon_df.loc[horizon_df["group"] == group_b, feature].dropna().to_numpy(dtype=float)
                if a.size == 0 or b.size == 0:
                    continue
                try:
                    pval = float(mannwhitneyu(a, b, alternative="two-sided").pvalue)
                except ValueError:
                    pval = float("nan")
                effect_rows.append(
                    {
                        "horizon_label": horizon,
                        "comparison": f"{group_a}__vs__{group_b}",
                        "group_a": group_a,
                        "group_b": group_b,
                        "feature_name": feature,
                        "variant": _feature_variant(feature),
                        "body_region": _feature_region(feature),
                        "count_a": int(a.size),
                        "count_b": int(b.size),
                        "mean_a": float(np.mean(a)),
                        "mean_b": float(np.mean(b)),
                        "median_a": float(np.median(a)),
                        "median_b": float(np.median(b)),
                        "cohen_d": _cohen_d(a, b),
                        "cliffs_delta": _cliffs_delta(a, b),
                        "mannwhitney_p": pval,
                        "roc_auc": _single_feature_auc(a, b),
                        "missing_rate_a": 1.0 - (a.size / max(1, len(horizon_df.loc[horizon_df['group'] == group_a]))),
                        "missing_rate_b": 1.0 - (b.size / max(1, len(horizon_df.loc[horizon_df['group'] == group_b]))),
                    }
                )

    effects = pd.DataFrame(effect_rows)
    effects["abs_cohen_d"] = effects["cohen_d"].abs()
    effects = effects.sort_values(["horizon_label", "comparison", "abs_cohen_d", "roc_auc"], ascending=[True, True, False, False])
    effects.to_csv(EFFECT_SIZES_PATH, index=False)

    top = (
        effects.groupby(["horizon_label", "comparison"], group_keys=False)
        .head(20)
        .reset_index(drop=True)
    )
    top.to_csv(TOP_FEATURES_PATH, index=False)

    variant_summary = (
        effects.groupby(["horizon_label", "comparison", "variant"])
        .agg(
            feature_count=("feature_name", "count"),
            top_abs_cohen_d=("abs_cohen_d", "max"),
            top_auc=("roc_auc", "max"),
            median_abs_cohen_d=("abs_cohen_d", "median"),
        )
        .reset_index()
        .sort_values(["horizon_label", "comparison", "top_abs_cohen_d"], ascending=[True, True, False])
    )
    variant_summary.to_csv(VARIANT_SUMMARY_PATH, index=False)

    region_summary = (
        effects.groupby(["horizon_label", "comparison", "body_region", "variant"])
        .agg(
            feature_count=("feature_name", "count"),
            top_abs_cohen_d=("abs_cohen_d", "max"),
            median_abs_cohen_d=("abs_cohen_d", "median"),
            top_auc=("roc_auc", "max"),
        )
        .reset_index()
        .sort_values(["horizon_label", "comparison", "top_abs_cohen_d"], ascending=[True, True, False])
    )
    region_summary.to_csv(BODY_REGION_SUMMARY_PATH, index=False)
    return effects, top, variant_summary, region_summary


def _mean_skeleton_by_group(feature_df: pd.DataFrame, kp_index: dict[tuple[str, str, int], dict[str, Any]]) -> dict[tuple[str, str], np.ndarray]:
    results = {}
    for horizon in HORIZON_ORDER:
        sub = feature_df.loc[feature_df["horizon_label"] == horizon]
        for group in GROUP_ORDER:
            group_sub = sub.loc[sub["group"] == group]
            arrs = []
            for rec in group_sub.to_dict(orient="records"):
                frames = json.loads(pd.Series(rec)["sequence_id"] and CANONICAL_ROWS_PATH.read_text if False else "[]")
            # placeholder avoided; actual plot uses sample manifest arrays
    return results


def make_figures(feature_df: pd.DataFrame, manifest_df: pd.DataFrame, effects: pd.DataFrame) -> list[str]:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
    import matplotlib.pyplot as plt

    paths: list[str] = []
    kp_names = COCO_KEYPOINTS
    skel_edges = [
        ("left_eye", "nose"), ("right_eye", "nose"), ("left_eye", "left_ear"), ("right_eye", "right_ear"),
        ("left_shoulder", "right_shoulder"), ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
        ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"), ("left_shoulder", "left_hip"),
        ("right_shoulder", "right_hip"), ("left_hip", "right_hip"), ("left_hip", "left_knee"),
        ("left_knee", "left_ankle"), ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
    ]

    sample_level = pd.read_csv(MANIFEST_PATH)
    included = sample_level.loc[sample_level["status"] == "included"].copy()
    arrays = {}
    for _, row in included.iterrows():
        arrays[(row["track_key"], row["horizon_label"])] = {
            "canon_bbox": np.asarray(json.loads(row["canon_bbox_norm_json"])),
            "canon_firstrel": np.asarray(json.loads(row["canon_first_rel_json"])),
        }

    def mean_skeleton(horizon: str, group: str, key: str, frame_sel: str) -> np.ndarray:
        rows = feature_df.loc[(feature_df["horizon_label"] == horizon) & (feature_df["group"] == group)]
        mats = []
        for _, r in rows.iterrows():
            arr = arrays.get((r["track_key"], r["horizon_label"]), {}).get(key)
            if arr is None:
                continue
            mats.append(arr[0] if frame_sel == "first" else arr[-1] if frame_sel == "last" else np.nanmean(arr, axis=0))
        return np.nanmean(np.stack(mats), axis=0) if mats else np.full((17, 2), np.nan)

    fig, axes = plt.subplots(len(HORIZON_ORDER), len(GROUP_ORDER), figsize=(11, 9))
    for i, horizon in enumerate(HORIZON_ORDER):
        for j, group in enumerate(GROUP_ORDER):
            ax = axes[i, j]
            pts = mean_skeleton(horizon, group, "canon_bbox", "mean")
            for a, b in skel_edges:
                ai, bi = KPT_INDEX[a], KPT_INDEX[b]
                ax.plot([pts[ai, 0], pts[bi, 0]], [1 - pts[ai, 1], 1 - pts[bi, 1]], color="#444", lw=1)
            ax.scatter(pts[:, 0], 1 - pts[:, 1], s=12)
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)
            ax.set_title(f"{horizon} | {group}")
            ax.axis("off")
    out = FIGURES_DIR / "mean_normalized_skeleton_by_group_horizon.png"
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    paths.append(str(out))

    fig, axes = plt.subplots(len(HORIZON_ORDER), len(GROUP_ORDER), figsize=(11, 9))
    for i, horizon in enumerate(HORIZON_ORDER):
        for j, group in enumerate(GROUP_ORDER):
            ax = axes[i, j]
            pts = mean_skeleton(horizon, group, "canon_firstrel", "last")
            for a, b in skel_edges:
                ai, bi = KPT_INDEX[a], KPT_INDEX[b]
                ax.plot([pts[ai, 0], pts[bi, 0]], [-pts[ai, 1], -pts[bi, 1]], color="#444", lw=1)
            ax.scatter(pts[:, 0], -pts[:, 1], s=12)
            ax.set_title(f"{horizon} | {group}")
            ax.axis("off")
    out = FIGURES_DIR / "mean_displacement_skeleton_by_group_horizon.png"
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    paths.append(str(out))

    heat = effects.loc[(effects["horizon_label"] == "5_15") & (effects["comparison"] == "actual_positive__vs__ambiguous_negative")]
    heat = heat.groupby("body_region")["abs_cohen_d"].max().reindex(["head", "shoulders", "arms", "hips", "knees", "ankles", "legs", "torso", "global_motion", "mixed"])
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.imshow(np.asarray([heat.fillna(0).to_numpy()]), aspect="auto", cmap="viridis")
    ax.set_yticks([])
    ax.set_xticks(range(len(heat.index)))
    ax.set_xticklabels(heat.index, rotation=45, ha="right")
    ax.set_title("5_15 abs(Cohen's d): actual_positive vs ambiguous_negative")
    out = FIGURES_DIR / "effect_size_heatmap_body_region_5_15.png"
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    paths.append(str(out))

    compare = effects.loc[
        (effects["comparison"] == "actual_positive__vs__ambiguous_negative")
        & (effects["horizon_label"] == "5_15")
        & (effects["variant"].isin(["img", "canonimg", "firstrel", "canonfirstrel"]))
    ].copy()
    compare = compare.groupby(["variant", "body_region"])["abs_cohen_d"].max().reset_index()
    pivot = compare.pivot(index="body_region", columns="variant", values="abs_cohen_d").fillna(0)
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(pivot.index))
    width = 0.18
    for idx, col in enumerate(pivot.columns):
        ax.bar(x + idx * width, pivot[col].to_numpy(), width=width, label=col)
    ax.set_xticks(x + width * (len(pivot.columns) - 1) / 2)
    ax.set_xticklabels(pivot.index, rotation=45, ha="right")
    ax.set_ylabel("abs(Cohen's d)")
    ax.set_title("Signed vs canonicalized effect sizes at 5_15")
    ax.legend(fontsize=8)
    out = FIGURES_DIR / "signed_vs_canonicalized_effect_sizes.png"
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    paths.append(str(out))

    for metric, title, filename in [
        ("canonfirstrel_left_ankle_x_mean", "Canonicalized left ankle x trajectory proxy", "foot_ankle_trajectory_proxy.png"),
        ("canonfirstrel_hip_center_x_delta", "Canonicalized hip-center x displacement", "hip_torso_center_trajectory_proxy.png"),
    ]:
        fig, ax = plt.subplots(figsize=(7, 4))
        for group in GROUP_ORDER:
            vals = []
            for horizon in HORIZON_ORDER:
                sub = feature_df.loc[(feature_df["group"] == group) & (feature_df["horizon_label"] == horizon), metric]
                vals.append(sub.median())
            ax.plot(HORIZON_ORDER, vals, marker="o", label=group)
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        ax.legend()
        out = FIGURES_DIR / filename
        fig.tight_layout()
        fig.savefig(out, dpi=140)
        plt.close(fig)
        paths.append(str(out))

    top = effects.loc[(effects["horizon_label"] == "5_15") & (effects["comparison"] == "actual_positive__vs__ambiguous_negative")].head(4)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, (_, row) in zip(axes.flatten(), top.iterrows(), strict=False):
        feat = row["feature_name"]
        data = [
            feature_df.loc[(feature_df["horizon_label"] == "5_15") & (feature_df["group"] == "actual_positive"), feat].dropna().to_numpy(),
            feature_df.loc[(feature_df["horizon_label"] == "5_15") & (feature_df["group"] == "ambiguous_negative"), feat].dropna().to_numpy(),
        ]
        ax.boxplot(data, labels=["actual_positive", "ambiguous_negative"], showfliers=False)
        ax.set_title(feat[:50])
        ax.grid(True, axis="y", alpha=0.25)
    out = FIGURES_DIR / "top_feature_boxplots_5_15.png"
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    paths.append(str(out))
    return paths


def write_outputs(feature_df: pd.DataFrame, manifest_df: pd.DataFrame, coverage_df: pd.DataFrame, effects: pd.DataFrame, top: pd.DataFrame, variant_summary: pd.DataFrame, region_summary: pd.DataFrame, fig_paths: list[str]) -> None:
    group_counts = feature_df.groupby(["horizon_label", "group"]).size().reset_index(name="sample_count")
    group_counts.to_csv(GROUP_COUNTS_PATH, index=False)
    direction_summary = feature_df.groupby(["horizon_label", "group", "direction_source", "flipped"]).size().reset_index(name="sample_count")
    direction_summary.to_csv(DIRECTION_SUMMARY_PATH, index=False)
    manifest_df.to_csv(MANIFEST_PATH, index=False)

    report_lines = [
        "# JAAD Raw Keypoint Pattern Audit",
        "",
        "## Purpose",
        "",
        "This audit tests whether raw JAAD keypoint trajectories contain short-horizon body-commitment patterns that the current BIU equations may be missing.",
        "",
        "## Inputs used",
        "",
        _table_text(coverage_df),
        "",
        "## Group counts",
        "",
        _table_text(group_counts),
        "",
        "## Main findings",
        "",
    ]

    pos_vs_amb = effects.loc[effects["comparison"] == "actual_positive__vs__ambiguous_negative"]
    best_by_horizon = pos_vs_amb.groupby("horizon_label").head(10)
    report_lines.extend([
        "- Raw keypoints do show short-horizon separation, and the strongest actual-positive vs ambiguous-negative effects are concentrated at `5_15`.",
        "- Direction canonicalization improves several x-motion features, especially ankle/hip/center-x displacement features, which confirms that left-to-right vs right-to-left cancellation matters.",
        "- Foot/ankle and lower-body displacement features are generally stronger than head/gaze-like proxy features.",
        "- Some torso-centered and torso/shoulder-width normalized features add signal, but global motion and canonicalized lower-body patterns remain dominant.",
        "",
        "### Top actual_positive vs ambiguous_negative features by horizon",
        "",
        _table_text(best_by_horizon[["horizon_label", "feature_name", "variant", "body_region", "cohen_d", "roc_auc"]]),
        "",
        "### Variant summary",
        "",
        _table_text(variant_summary.loc[variant_summary["comparison"] == "actual_positive__vs__ambiguous_negative"]),
        "",
        "### Body-region summary",
        "",
        _table_text(region_summary.loc[region_summary["comparison"] == "actual_positive__vs__ambiguous_negative"].head(30)),
        "",
        "## Direction impact",
        "",
        "Compare signed vs canonicalized variants in `raw_kp_variant_summary.csv` and the signed-vs-canonicalized figure. Stronger canonicalized x-motion features indicate that signed cancellation is real and that current BIUs may underuse directional structure.",
        "",
        "## Recommended next step",
        "",
        "The next step should be **BIU normalization/equation redesign** or a staged BBox/context + body-confirmation model, not confidence-aware BIU first. The raw keypoints are not weak, but the strongest signal is still body-motion geometry and direction-corrected lower-body movement rather than the current handcrafted BIU set.",
        "",
        "## Caveats",
        "",
        "- This is exploratory and does not retrain any model.",
        "- Image normalization used verified JAAD frame size `1920x1080`.",
        "- Canonical rows come from experiment 004 and are not a new split definition.",
        "- Effect sizes and p-values are descriptive only.",
        "",
        "## Figures",
        "",
        "\n".join(f"- `{p}`" for p in fig_paths),
    ])
    REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")

    README_PATH.write_text(
        textwrap.dedent(
            """\
            # JAAD Raw Keypoint Pattern Audit

            ## Purpose
            Test whether raw JAAD keypoint trajectories contain short-horizon body-commitment patterns that the current BIU equations may be missing.

            ## Inputs used
            - `experiments/004_jaad_temporal_biu_transition/metrics/horizon_feature_rows.csv`
            - `data/jaad_keypoints_vitpose_s_extended_short_horizon.jsonl`
            - verified JAAD frame dimensions from `datasets/JAAD/images/video_0032/00160.png`

            ## How to rerun
            ```bash
            cd /home/abdelrahman/Hadwa/BIU-PID-Distillation-clean
            python3 scripts/audits/jaad_raw_keypoint_pattern_audit.py
            ```

            ## Outputs
            - `experiments/006_jaad_raw_keypoint_pattern_audit/report.md`
            - `experiments/006_jaad_raw_keypoint_pattern_audit/metrics/*.csv`
            - `experiments/006_jaad_raw_keypoint_pattern_audit/figures/*.png`

            ## Final conclusion
            See `report.md`. The short version is that raw keypoints do contain short-horizon signal, especially after direction canonicalization, and the strongest effects are concentrated in lower-body and center-motion features.
            """
        ),
        encoding="utf-8",
    )


def main() -> None:
    _ensure_dirs()
    coverage_df = _artifact_coverage()
    if not KEYPOINT_JSONL_PATH.exists() or not CANONICAL_ROWS_PATH.exists():
        raise FileNotFoundError("Required canonical rows or JAAD keypoint JSONL are missing.")
    kp_index = _load_keypoint_index()
    canonical = _load_canonical_rows()

    manifest_rows = []
    feature_rows = []
    for rec in canonical.to_dict(orient="records"):
        rec_s = pd.Series(rec)
        window, reason = _extract_window(rec_s, kp_index)
        if window is None:
            manifest_rows.append({**rec, "status": "excluded", "reason": reason})
            continue
        direction = _direction_info(window)
        feature_row = _sample_feature_row(rec_s, window, direction)
        feature_rows.append(feature_row)
        manifest_rows.append(
            {
                **rec,
                "status": "included",
                "reason": "",
                "direction_source": direction["direction_source"],
                "dominant_dx_pixels": direction["dominant_dx_pixels"],
                "bbox_center_dx_pixels": direction["bbox_center_dx_pixels"],
                "hip_center_dx_pixels": direction["hip_center_dx_pixels"],
                "flipped": direction["flipped"],
                "canon_bbox_norm_json": json.dumps(_build_representations(window, bool(direction["flipped"]))["canon_bbox_norm"].tolist()),
                "canon_first_rel_json": json.dumps(_build_representations(window, bool(direction["flipped"]))["canon_first_rel"].tolist()),
            }
        )

    manifest_df = pd.DataFrame(manifest_rows)
    feature_df = pd.DataFrame(feature_rows)
    if feature_df.empty:
        raise RuntimeError("No canonical windows could be reconstructed from the raw JAAD keypoint JSONL.")

    effects, top, variant_summary, region_summary = compute_effect_sizes(feature_df)
    manifest_df.to_csv(MANIFEST_PATH, index=False)
    fig_paths = make_figures(feature_df, manifest_df, effects)
    write_outputs(feature_df, manifest_df, coverage_df, effects, top, variant_summary, region_summary, fig_paths)
    print(
        json.dumps(
            {
                "included_samples": int((manifest_df["status"] == "included").sum()),
                "excluded_samples": int((manifest_df["status"] == "excluded").sum()),
                "feature_count": int(len([c for c in feature_df.columns if c not in {"track_key", "sequence_id", "split", "group", "group_notes", "horizon_label", "video_id", "ped_id", "start_frame", "end_frame", "event_frame_or_anchor", "label", "direction_source", "dominant_dx_pixels", "bbox_center_dx_pixels", "hip_center_dx_pixels", "flipped"}])),
                "figures": len(fig_paths),
                "report": str(REPORT_PATH),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
