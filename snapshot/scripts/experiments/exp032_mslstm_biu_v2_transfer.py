"""Exp032: BIU-v2 transfer into the local MSLSTM next-frame PIE setting.

This experiment preserves the recovered local MSLSTM task lineage:
  - PIE
  - 30 consecutive observed frames
  - pred_len = 1 raw frame
  - label = behavior.cross at the target frame
  - train=set01,set02,set04 ; val=set05,set06 ; test=set03

Variant A is the fresh control in this recovered setting.
Variant B adds one small late-fusion BIU-v2 branch. BIU-v2 is derived from the same
stored MSLSTM pose measurements after deterministic de-normalization back to image pixel
coordinates before calling the frozen extractor.
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path("/home/abdelrahman/Hadwa/BIU-PID-Distillation-clean")
MSLSTM_ROOT = Path("/home/abdelrahman/Hadwa/MSLSTM-PID")
SCRIPTS_DIR = ROOT / "scripts/experiments"
SRC_DIR = ROOT / "src"
for _path in (str(SCRIPTS_DIR), str(SRC_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import jaad_biu_v2_benchmark_comparison as jaad  # noqa: E402
from biu_v2_extractor import COCO_KEYPOINTS, extract_biu_v2, feature_dictionary_df  # noqa: E402

PIE_DB_CACHE = MSLSTM_ROOT / "PIE/annotations/data_cache/pie_database.pkl"
POSE_DATA_DIR = MSLSTM_ROOT / "Extracted Poses"
TRAIN_INDEX_PATH = MSLSTM_ROOT / "checkpoints/train_index.pkl"
DEFAULT_OUT_ROOT = ROOT / "experiments/032_mslstm_biu_v2_transfer"

TRAIN_SETS = ("set01", "set02", "set04")
VAL_SETS = ("set05", "set06")
TEST_SETS = ("set03",)
SPLIT_SETS = {"train": TRAIN_SETS, "val": VAL_SETS, "test": TEST_SETS}
SPLITS = ("train", "val", "test")
SEQ_LEN = 30
PRED_LEN = 1
SEEDS = (42, 43, 44)

ACTIVE_STREAMS = (
    "bbox",
    "pose",
    "ped_action",
    "ped_look",
    "ego_speed",
    "ego_acc",
    "static_context",
)
VARIANTS = {
    "variant_a_mslstm_control": ACTIVE_STREAMS,
    "variant_b_mslstm_plus_biu_v2": ACTIVE_STREAMS,
}

INPUT_SIZE_BBOX = 4
INPUT_SIZE_POSE = 34
INPUT_SIZE_EGO_SPEED = 1
INPUT_SIZE_EGO_ACC = 2
INPUT_SIZE_PED_ACTION = 1
INPUT_SIZE_PED_LOOK = 1
NUM_SIGNALIZED_CATS = 4
NUM_INTERSECTION_CATS = 5
NUM_AGE_CATS = 4
NUM_GENDER_CATS = 3
NUM_TRAFFIC_DIR_CATS = 2
LANE_CATEGORIES = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 4, 7: 4, 8: 4}
NUM_LANE_CATS = len(set(LANE_CATEGORIES.values()))
INPUT_SIZE_STATIC = (
    NUM_SIGNALIZED_CATS
    + NUM_INTERSECTION_CATS
    + NUM_AGE_CATS
    + NUM_GENDER_CATS
    + NUM_TRAFFIC_DIR_CATS
    + NUM_LANE_CATS
)
STREAM_DIMS = {
    "bbox": INPUT_SIZE_BBOX,
    "pose": INPUT_SIZE_POSE,
    "ped_action": INPUT_SIZE_PED_ACTION,
    "ped_look": INPUT_SIZE_PED_LOOK,
    "ego_speed": INPUT_SIZE_EGO_SPEED,
    "ego_acc": INPUT_SIZE_EGO_ACC,
    "static_context": INPUT_SIZE_STATIC,
}

LSTM_HIDDEN_SIZE = 256
NUM_LSTM_LAYERS = 2
ATTENTION_DIM = 128
DROPOUT_RATE = 0.3
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
NUM_EPOCHS = 5
BIU_PROJECTION_DIM = 128
WILD_F1_DROP = 0.08
WILD_ROC_AUC_DROP = 0.05
PAPER_REFERENCE = {
    "accuracy": 0.916,
    "roc_auc": 0.970,
    "f1": 0.899,
    "precision": 0.929,
    "recall": 0.871,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class RunArtifacts:
    rows: pd.DataFrame
    arrays: dict[str, np.ndarray]
    labels: np.ndarray
    split_index: dict[str, np.ndarray]
    scalers: dict[str, float]
    biu_feature_names: list[str]
    audit: dict[str, object]


@dataclass
class FullRunContext:
    rows: pd.DataFrame
    pie_db: dict
    pose_cache: PoseCache
    scalers: dict[str, float]
    biu_cache_path: Path
    biu_feature_names: list[str]
    audit: dict[str, object]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_one_hot(index: int, num_classes: int) -> np.ndarray:
    idx = int(index)
    if idx < 0 or idx >= num_classes:
        idx = 0
    out = np.zeros(num_classes, dtype=np.float32)
    out[idx] = 1.0
    return out


def denormalize_pose_window(window_norm: np.ndarray, frame_width: int, frame_height: int) -> np.ndarray:
    coords = np.asarray(window_norm, dtype=np.float64).reshape(-1, 17, 2)
    scale = np.array([frame_width, frame_height], dtype=np.float64).reshape(1, 1, 2)
    return coords * scale


def renormalize_pose_window(window_px: np.ndarray, frame_width: int, frame_height: int) -> np.ndarray:
    coords = np.asarray(window_px, dtype=np.float64).reshape(-1, 17, 2)
    scale = np.array([frame_width, frame_height], dtype=np.float64).reshape(1, 1, 2)
    return coords / scale


def _sequence_id(set_id: str, video_id: str, ped_id: str, start_frame: int) -> str:
    return f"{set_id}|{video_id}|{ped_id}|{start_frame}"


def load_pie_database() -> dict:
    with PIE_DB_CACHE.open("rb") as handle:
        return pickle.load(handle)


def load_exp032_protocol_rows() -> pd.DataFrame:
    pie_db = load_pie_database()
    rows: list[dict[str, object]] = []
    for split, set_names in SPLIT_SETS.items():
        for set_id in set_names:
            for video_id, vdb in pie_db.get(set_id, {}).items():
                width = int(vdb.get("width", 1920) or 1920)
                height = int(vdb.get("height", 1080) or 1080)
                ped_annotations = vdb.get("ped_annotations", {})
                for ped_id, pdb in ped_annotations.items():
                    frames = sorted(int(f) for f in pdb.get("frames", []))
                    if len(frames) < SEQ_LEN + PRED_LEN:
                        continue
                    cross_seq = pdb.get("behavior", {}).get("cross", [])
                    for i in range(len(frames) - SEQ_LEN - PRED_LEN + 1):
                        start_f = frames[i]
                        obs_end_f = frames[i + SEQ_LEN - 1]
                        if obs_end_f - start_f != SEQ_LEN - 1:
                            continue
                        target_idx = i + SEQ_LEN + PRED_LEN - 1
                        target_f = frames[target_idx]
                        if target_f - obs_end_f != PRED_LEN:
                            continue
                        try:
                            ped_target_idx = pdb["frames"].index(target_f)
                            label = int(cross_seq[ped_target_idx])
                        except (ValueError, IndexError, KeyError):
                            label = 0
                        if label == -1:
                            label = 0
                        rows.append(
                            {
                                "sequence_id": _sequence_id(set_id, video_id, str(ped_id), start_f),
                                "split": split,
                                "set_id": set_id,
                                "video_id": video_id,
                                "ped_id": str(ped_id),
                                "start_frame": int(start_f),
                                "end_frame": int(obs_end_f),
                                "target_frame": int(target_f),
                                "observation_length": SEQ_LEN,
                                "pred_len": PRED_LEN,
                                "label": int(label),
                                "frame_width": width,
                                "frame_height": height,
                            }
                        )
    df = pd.DataFrame(rows).sort_values(["split", "set_id", "video_id", "ped_id", "start_frame"]).reset_index(drop=True)
    if df.empty:
        raise RuntimeError("Exp032 protocol row reconstruction produced no rows.")
    return df


def load_recovered_train_rows(
    protocol_rows: pd.DataFrame,
    index_path: Path = TRAIN_INDEX_PATH,
) -> pd.DataFrame:
    """Resolve the immutable balanced notebook index against reconstructed rows."""
    with Path(index_path).open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Recovered train index payload must be a dictionary.")
    if int(payload.get("seq_len", -1)) != SEQ_LEN or int(payload.get("pred_len", -1)) != PRED_LEN:
        raise ValueError("Recovered train index task dimensions do not match Exp032.")

    tuples = payload.get("seq_index")
    if not isinstance(tuples, list) or not tuples:
        raise ValueError("Recovered train index has no sequence rows.")
    index_df = pd.DataFrame(tuples, columns=["set_id", "video_id", "ped_id", "start_frame", "index_label"])
    index_df["ped_id"] = index_df["ped_id"].astype(str)
    index_df["start_frame"] = index_df["start_frame"].astype(int)
    index_df["index_label"] = index_df["index_label"].astype(int)
    index_df["sequence_id"] = [
        _sequence_id(row.set_id, row.video_id, row.ped_id, row.start_frame)
        for row in index_df.itertuples(index=False)
    ]
    if index_df["sequence_id"].duplicated().any():
        raise ValueError("Recovered train index contains duplicate sequence IDs.")
    if set(index_df["set_id"].unique()) != set(TRAIN_SETS):
        raise ValueError("Recovered train index set membership does not match Exp032.")
    label_counts = index_df["index_label"].value_counts().to_dict()
    if set(label_counts) != {0, 1} or label_counts[0] != label_counts[1]:
        raise ValueError("Recovered train index is not exactly class-balanced.")

    source = protocol_rows.loc[protocol_rows["split"] == "train"].copy()
    if source["sequence_id"].duplicated().any():
        raise ValueError("Reconstructed train protocol contains duplicate sequence IDs.")
    source = source.set_index("sequence_id", drop=False)
    missing = index_df.loc[~index_df["sequence_id"].isin(source.index), "sequence_id"]
    if not missing.empty:
        raise ValueError(f"Recovered train index rows are absent from reconstruction: {missing.iloc[0]}")

    selected = source.loc[index_df["sequence_id"]].reset_index(drop=True)
    expected_labels = index_df["index_label"].to_numpy(dtype=np.int64)
    actual_labels = selected["label"].to_numpy(dtype=np.int64)
    if not np.array_equal(expected_labels, actual_labels):
        raise ValueError("Recovered train index labels disagree with reconstructed labels.")
    return selected


class PoseCache:
    def __init__(self, pose_dir: Path):
        self.pose_dir = Path(pose_dir)
        self._cache: dict[tuple[str, str], dict[int, dict[str, np.ndarray]]] = {}

    def _path_for(self, set_id: str, video_id: str) -> Path:
        return self.pose_dir / set_id / f"{set_id}_{video_id}_poses.pkl"

    def get_video(self, set_id: str, video_id: str) -> dict[int, dict[str, np.ndarray]]:
        key = (set_id, video_id)
        if key not in self._cache:
            path = self._path_for(set_id, video_id)
            if not path.exists():
                raise FileNotFoundError(f"Missing pose file: {path}")
            with path.open("rb") as handle:
                payload = pickle.load(handle)
            nested_key = f"{set_id}_{video_id}"
            if nested_key not in payload:
                raise KeyError(f"Pose payload {path} missing top-level key {nested_key}")
            self._cache[key] = payload[nested_key]
        return self._cache[key]

    def get_pose(self, set_id: str, video_id: str, frame_num: int, ped_id: str) -> np.ndarray | None:
        video = self.get_video(set_id, video_id)
        frame_payload = video.get(int(frame_num), {})
        pose = frame_payload.get(str(ped_id))
        if pose is None:
            return None
        arr = np.asarray(pose, dtype=np.float32)
        if arr.shape != (34,):
            return None
        return arr


def compute_train_scalers(pie_db: dict) -> dict[str, float]:
    speed_values: list[float] = []
    accx_values: list[float] = []
    accy_values: list[float] = []
    for set_id in TRAIN_SETS:
        for _, vdb in pie_db.get(set_id, {}).items():
            for _, ego in vdb.get("vehicle_annotations", {}).items():
                speed_values.append(float(ego.get("OBD_speed", 0.0) or ego.get("GPS_speed", 0.0)))
                accx_values.append(float(ego.get("accX", 0.0)))
                accy_values.append(float(ego.get("accY", 0.0)))
    def _mean_std(values: list[float]) -> tuple[float, float]:
        arr = np.asarray(values, dtype=np.float64)
        return float(arr.mean()), float(max(arr.std(), 1e-6))
    speed_mean, speed_std = _mean_std(speed_values)
    accx_mean, accx_std = _mean_std(accx_values)
    accy_mean, accy_std = _mean_std(accy_values)
    return {
        "ego_speed_mean": speed_mean,
        "ego_speed_std": speed_std,
        "accX_mean": accx_mean,
        "accX_std": accx_std,
        "accY_mean": accy_mean,
        "accY_std": accy_std,
    }


def build_static_context(attr: dict[str, object]) -> np.ndarray:
    signalized = attr.get("signalized", 0)
    intersection = attr.get("intersection", 0)
    age = attr.get("age", 2)
    gender = attr.get("gender", 0)
    traffic_direction = int(attr.get("traffic_direction", 0))
    num_lanes = attr.get("num_lanes", 2)
    lane_cat = LANE_CATEGORIES.get(num_lanes, LANE_CATEGORIES[max(LANE_CATEGORIES.keys())])
    vec = np.concatenate(
        [
            to_one_hot(signalized, NUM_SIGNALIZED_CATS),
            to_one_hot(intersection, NUM_INTERSECTION_CATS),
            to_one_hot(age, NUM_AGE_CATS),
            to_one_hot(gender, NUM_GENDER_CATS),
            to_one_hot(traffic_direction, NUM_TRAFFIC_DIR_CATS),
            to_one_hot(lane_cat, NUM_LANE_CATS),
        ]
    ).astype(np.float32)
    if vec.shape[0] != INPUT_SIZE_STATIC:
        raise RuntimeError(f"Static context shape mismatch: {vec.shape[0]} != {INPUT_SIZE_STATIC}")
    return vec


def extract_biu_v2_feature_map(window_px: np.ndarray, image_size: tuple[int, int] = (1920, 1080)) -> dict[str, object]:
    return extract_biu_v2(window_px.astype(np.float64), kp_conf=None, bbox_sequence=None, image_size=image_size)


def extract_biu_v2_window_features(window_px: np.ndarray, image_size: tuple[int, int] = (1920, 1080)) -> tuple[np.ndarray, list[str]]:
    feats = extract_biu_v2_feature_map(window_px, image_size=image_size)
    numeric_names = [
        name
        for name, value in feats.items()
        if isinstance(value, (int, float, np.integer, np.floating))
    ]
    values = np.asarray([float(feats[name]) for name in numeric_names], dtype=np.float64)
    return values, numeric_names


def expected_biu_v2_feature_names() -> list[str]:
    dummy = np.zeros((SEQ_LEN, 17, 2), dtype=np.float64)
    names = list(extract_biu_v2_feature_map(dummy).keys())
    return names


def verify_biu_schema(names: list[str]) -> None:
    required = feature_dictionary_df()
    required = required.loc[required["required_or_optional"] == "required", "feature_name"].tolist()
    expanded_required: set[str] = set()
    for _, row in feature_dictionary_df().iterrows():
        if row["required_or_optional"] != "required":
            continue
        for stat in str(row["summary_stats"]).split(";"):
            stat = stat.strip()
            if stat:
                expanded_required.add(f"{row['feature_name']}_{stat}")
    missing_required = sorted(expanded_required.difference(names))
    if missing_required:
        raise RuntimeError(f"BIU-v2 schema mismatch: missing required features {missing_required[:10]}")
    if "direction_sign" not in names or "direction_source" not in names:
        raise RuntimeError("BIU-v2 schema mismatch: direction metadata missing from extractor output.")


def _window_from_row(
    row: pd.Series | object,
    pie_db: dict,
    pose_cache: PoseCache,
    scalers: dict[str, float],
) -> tuple[dict[str, np.ndarray], np.ndarray, int, int]:
    if hasattr(row, "_asdict"):
        row = row._asdict()
    row_d = dict(row)
    video_db = pie_db[row_d["set_id"]][row_d["video_id"]]
    ped_db = video_db["ped_annotations"][row_d["ped_id"]]
    ego_db = video_db.get("vehicle_annotations", {})
    frames = list(range(int(row_d["start_frame"]), int(row_d["end_frame"]) + 1))
    frame_to_idx = {int(frame): idx for idx, frame in enumerate(ped_db.get("frames", []))}
    static_vec = build_static_context(ped_db.get("attributes", {}))
    width = int(row_d["frame_width"])
    height = int(row_d["frame_height"])

    stream_arrays = {
        "bbox": np.zeros((SEQ_LEN, INPUT_SIZE_BBOX), dtype=np.float32),
        "pose": np.zeros((SEQ_LEN, INPUT_SIZE_POSE), dtype=np.float32),
        "ped_action": np.zeros((SEQ_LEN, INPUT_SIZE_PED_ACTION), dtype=np.float32),
        "ped_look": np.zeros((SEQ_LEN, INPUT_SIZE_PED_LOOK), dtype=np.float32),
        "ego_speed": np.zeros((SEQ_LEN, INPUT_SIZE_EGO_SPEED), dtype=np.float32),
        "ego_acc": np.zeros((SEQ_LEN, INPUT_SIZE_EGO_ACC), dtype=np.float32),
        "static_context": np.zeros((SEQ_LEN, INPUT_SIZE_STATIC), dtype=np.float32),
    }
    pose_window_norm = np.zeros((SEQ_LEN, 17, 2), dtype=np.float64)
    for t, frame_num in enumerate(frames):
        ped_frame_idx = frame_to_idx.get(frame_num, -1)
        ego = ego_db.get(frame_num, {})

        if ped_frame_idx != -1 and ped_frame_idx < len(ped_db.get("bbox", [])):
            try:
                x1, y1, x2, y2 = ped_db["bbox"][ped_frame_idx]
                cx = ((x1 + x2) / 2.0) / width
                cy = ((y1 + y2) / 2.0) / height
                bw = (x2 - x1) / width
                bh = (y2 - y1) / height
                if bw > 0 and bh > 0 and 0 <= cx <= 1 and 0 <= cy <= 1:
                    stream_arrays["bbox"][t] = np.asarray([cx, cy, bw, bh], dtype=np.float32)
            except Exception:
                pass

        pose_vec = pose_cache.get_pose(row_d["set_id"], row_d["video_id"], frame_num, row_d["ped_id"])
        if pose_vec is None:
            pose_vec = np.zeros((34,), dtype=np.float32)
        stream_arrays["pose"][t] = pose_vec
        pose_window_norm[t] = pose_vec.reshape(17, 2)

        speed = float(ego.get("OBD_speed", 0.0) or ego.get("GPS_speed", 0.0))
        stream_arrays["ego_speed"][t, 0] = (speed - scalers["ego_speed_mean"]) / scalers["ego_speed_std"]
        ax = float(ego.get("accX", 0.0))
        ay = float(ego.get("accY", 0.0))
        stream_arrays["ego_acc"][t, 0] = (ax - scalers["accX_mean"]) / scalers["accX_std"]
        stream_arrays["ego_acc"][t, 1] = (ay - scalers["accY_mean"]) / scalers["accY_std"]

        if ped_frame_idx != -1:
            action_seq = ped_db.get("behavior", {}).get("action", [])
            if ped_frame_idx < len(action_seq):
                stream_arrays["ped_action"][t, 0] = float(action_seq[ped_frame_idx])
            look_seq = ped_db.get("behavior", {}).get("look", [])
            if ped_frame_idx < len(look_seq):
                stream_arrays["ped_look"][t, 0] = float(look_seq[ped_frame_idx])

        stream_arrays["static_context"][t] = static_vec
    return stream_arrays, pose_window_norm, width, height


def assemble_exp032_windows(sample_only: bool = False, sample_limit: int = 16) -> RunArtifacts:
    pie_db = load_pie_database()
    rows = load_exp032_protocol_rows()
    if sample_only:
        rows = rows.groupby("split", group_keys=False).head(max(1, sample_limit // 3)).reset_index(drop=True)
    scalers = compute_train_scalers(pie_db)
    pose_cache = PoseCache(POSE_DATA_DIR)

    n = len(rows)
    arrays = {
        "bbox": np.zeros((n, SEQ_LEN, INPUT_SIZE_BBOX), dtype=np.float32),
        "pose": np.zeros((n, SEQ_LEN, INPUT_SIZE_POSE), dtype=np.float32),
        "ped_action": np.zeros((n, SEQ_LEN, INPUT_SIZE_PED_ACTION), dtype=np.float32),
        "ped_look": np.zeros((n, SEQ_LEN, INPUT_SIZE_PED_LOOK), dtype=np.float32),
        "ego_speed": np.zeros((n, SEQ_LEN, INPUT_SIZE_EGO_SPEED), dtype=np.float32),
        "ego_acc": np.zeros((n, SEQ_LEN, INPUT_SIZE_EGO_ACC), dtype=np.float32),
        "static_context": np.zeros((n, SEQ_LEN, INPUT_SIZE_STATIC), dtype=np.float32),
    }
    labels = rows["label"].to_numpy(dtype=np.int64)
    biu_values: list[np.ndarray] = []
    biu_input_names_ref: list[str] | None = None
    biu_output_names_ref: list[str] | None = None
    round_trip_errors: list[float] = []
    frame_dims_seen: set[tuple[int, int]] = set()
    pose_missing_frames = 0
    pose_total_frames = n * SEQ_LEN
    zero_pose_windows = 0

    for row_idx, row in enumerate(rows.itertuples(index=False)):
        stream_arrays, pose_window_norm, width, height = _window_from_row(row, pie_db, pose_cache, scalers)
        frame_dims_seen.add((width, height))
        for name in ACTIVE_STREAMS:
            arrays[name][row_idx] = stream_arrays[name]
        pose_missing_frames += int(np.sum(~np.any(stream_arrays["pose"].reshape(SEQ_LEN, 17, 2), axis=(1, 2))))

        if not np.any(pose_window_norm):
            zero_pose_windows += 1
        pose_window_px = denormalize_pose_window(pose_window_norm, width, height)
        restored_norm = renormalize_pose_window(pose_window_px, width, height)
        round_trip_errors.append(float(np.max(np.abs(restored_norm - pose_window_norm))))
        feat_map = extract_biu_v2_feature_map(pose_window_px, image_size=(width, height))
        feat_output_names = list(feat_map.keys())
        feat_values, feat_input_names = extract_biu_v2_window_features(pose_window_px, image_size=(width, height))
        if biu_output_names_ref is None:
            verify_biu_schema(feat_output_names)
            biu_output_names_ref = feat_output_names
            biu_input_names_ref = feat_input_names
        elif feat_output_names != biu_output_names_ref or feat_input_names != biu_input_names_ref:
            raise RuntimeError("BIU-v2 feature schema drifted across windows.")
        biu_values.append(np.nan_to_num(feat_values, nan=0.0, posinf=0.0, neginf=0.0))

    if biu_input_names_ref is None or biu_output_names_ref is None:
        raise RuntimeError("BIU-v2 feature extraction produced no schema.")
    arrays["biu_v2"] = np.vstack(biu_values).astype(np.float32)
    split_index = {split: rows.index[rows["split"] == split].to_numpy(dtype=np.int64) for split in SPLITS}
    audit = {
        "n_rows": int(n),
        "split_counts": {split: int((rows["split"] == split).sum()) for split in SPLITS},
        "split_positive_counts": {split: int(rows.loc[rows["split"] == split, "label"].sum()) for split in SPLITS},
        "round_trip_max_abs_error": float(max(round_trip_errors) if round_trip_errors else 0.0),
        "round_trip_mean_abs_error": float(np.mean(round_trip_errors) if round_trip_errors else 0.0),
        "frame_dimensions_seen": [list(x) for x in sorted(frame_dims_seen)],
        "pose_missing_frames": int(pose_missing_frames),
        "pose_missing_rate": float(pose_missing_frames / max(pose_total_frames, 1)),
        "zero_pose_windows": int(zero_pose_windows),
        "biu_feature_dim": int(arrays["biu_v2"].shape[1]),
        "biu_numeric_feature_names": biu_input_names_ref,
        "biu_extractor_output_names": biu_output_names_ref,
        "active_streams": list(ACTIVE_STREAMS),
        "variant_names": list(VARIANTS.keys()),
    }
    return RunArtifacts(rows=rows, arrays=arrays, labels=labels, split_index=split_index, scalers=scalers, biu_feature_names=biu_input_names_ref, audit=audit)


def validate_biu_cache_live_sample(
    cache_path: Path,
    rows: pd.DataFrame,
    feature_names: list[str],
    recompute_row: Callable[[int, pd.Series], tuple[np.ndarray, list[str]]],
    sample_size: int = 16,
) -> dict[str, object]:
    cache = np.load(cache_path, mmap_mode="r")
    expected_shape = (len(rows), len(feature_names))
    if cache.shape != expected_shape:
        raise RuntimeError(f"BIU cache shape {cache.shape} does not match {expected_shape}.")
    if len(rows) == 0 or sample_size <= 0:
        raise RuntimeError("BIU cache validation requires rows and a positive sample size.")

    indices = np.unique(np.linspace(0, len(rows) - 1, min(sample_size, len(rows)), dtype=np.int64))
    max_abs_error = 0.0
    for idx in indices:
        live_values, live_names = recompute_row(int(idx), rows.iloc[int(idx)])
        if live_names != feature_names:
            raise RuntimeError("Live BIU feature names do not match cached feature names.")
        expected = np.nan_to_num(live_values, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        error = float(np.max(np.abs(expected - np.asarray(cache[int(idx)], dtype=np.float32))))
        max_abs_error = max(max_abs_error, error)
        if error != 0.0:
            raise RuntimeError(
                f"BIU cache does not match live BIU recomputation at row {int(idx)} "
                f"(max_abs_error={error})."
            )
    return {"sample_size": int(len(indices)), "max_abs_error": max_abs_error}


def precompute_biu_cache(
    rows: pd.DataFrame,
    pie_db: dict,
    pose_cache: PoseCache,
    scalers: dict[str, float],
    out_root: Path,
) -> tuple[Path, list[str], dict[str, object]]:
    cache_dir = out_root / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "biu_v2_cache.npy"
    meta_path = cache_dir / "biu_v2_cache_meta.json"

    if cache_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if int(meta.get("n_rows", -1)) == len(rows) and meta.get("complete") is True:
            feature_names = list(meta["biu_numeric_feature_names"])

            def _recompute_cached_row(_idx: int, row: pd.Series) -> tuple[np.ndarray, list[str]]:
                _, pose_window_norm, width, height = _window_from_row(row, pie_db, pose_cache, scalers)
                window_px = denormalize_pose_window(pose_window_norm, width, height)
                return extract_biu_v2_window_features(window_px, image_size=(width, height))

            validation = validate_biu_cache_live_sample(
                cache_path,
                rows,
                feature_names,
                _recompute_cached_row,
            )
            audit = dict(meta["audit"])
            audit["cache_live_validation"] = validation
            return cache_path, feature_names, audit

    names_ref: list[str] | None = None
    output_names_ref: list[str] | None = None
    frame_dim_counts: defaultdict[tuple[int, int], int] = defaultdict(int)
    round_trip_errors: list[float] = []
    zero_pose_windows = 0
    pose_missing_frames = 0

    _, first_pose, first_w, first_h = _window_from_row(rows.iloc[0], pie_db, pose_cache, scalers)
    first_px = denormalize_pose_window(first_pose, first_w, first_h)
    first_map = extract_biu_v2_feature_map(first_px, image_size=(first_w, first_h))
    output_names_ref = list(first_map.keys())
    verify_biu_schema(output_names_ref)
    first_values, names_ref = extract_biu_v2_window_features(first_px, image_size=(first_w, first_h))
    biu_dim = len(names_ref)
    mem = np.lib.format.open_memmap(cache_path, mode="w+", dtype=np.float32, shape=(len(rows), biu_dim))

    for row_idx, row in enumerate(rows.itertuples(index=False)):
        stream_arrays, pose_window_norm, width, height = _window_from_row(row, pie_db, pose_cache, scalers)
        frame_dim_counts[(width, height)] += 1
        pose_missing_frames += int(np.sum(~np.any(stream_arrays["pose"].reshape(SEQ_LEN, 17, 2), axis=(1, 2))))
        if not np.any(pose_window_norm):
            zero_pose_windows += 1
        window_px = denormalize_pose_window(pose_window_norm, width, height)
        restored = renormalize_pose_window(window_px, width, height)
        round_trip_errors.append(float(np.max(np.abs(restored - pose_window_norm))))
        feat_map = extract_biu_v2_feature_map(window_px, image_size=(width, height))
        if list(feat_map.keys()) != output_names_ref:
            raise RuntimeError("BIU extractor output schema drifted while precomputing the full cache.")
        values, names = extract_biu_v2_window_features(window_px, image_size=(width, height))
        if names != names_ref:
            raise RuntimeError("BIU numeric schema drifted while precomputing the full cache.")
        mem[row_idx] = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    mem.flush()

    audit = {
        "n_rows": int(len(rows)),
        "round_trip_max_abs_error": float(max(round_trip_errors) if round_trip_errors else 0.0),
        "round_trip_mean_abs_error": float(np.mean(round_trip_errors) if round_trip_errors else 0.0),
        "frame_dimensions_seen": [
            {"frame_width": int(w), "frame_height": int(h), "n_windows": int(count)}
            for (w, h), count in sorted(frame_dim_counts.items())
        ],
        "pose_missing_frames": int(pose_missing_frames),
        "pose_missing_rate": float(pose_missing_frames / max(len(rows) * SEQ_LEN, 1)),
        "zero_pose_windows": int(zero_pose_windows),
        "biu_feature_dim": int(biu_dim),
        "biu_numeric_feature_names": names_ref,
        "biu_extractor_output_names": output_names_ref,
    }
    meta = {
        "complete": True,
        "n_rows": int(len(rows)),
        "biu_numeric_feature_names": names_ref,
        "audit": audit,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return cache_path, names_ref, audit


class Exp032LazyDataset(Dataset):
    def __init__(
        self,
        rows: pd.DataFrame,
        pie_db: dict,
        pose_cache: PoseCache,
        scalers: dict[str, float],
        include_biu: bool,
        biu_cache_path: Path | None = None,
    ):
        self.rows = rows.reset_index(drop=True).copy()
        self.pie_db = pie_db
        self.pose_cache = pose_cache
        self.scalers = scalers
        self.include_biu = include_biu
        self.biu_cache = np.load(biu_cache_path, mmap_mode="r") if include_biu and biu_cache_path is not None else None

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows.iloc[idx]
        stream_arrays, _, _, _ = _window_from_row(row, self.pie_db, self.pose_cache, self.scalers)
        out = {name: torch.tensor(stream_arrays[name], dtype=torch.float32) for name in ACTIVE_STREAMS}
        if self.include_biu:
            global_idx = int(row["global_index"])
            out["biu_v2"] = torch.tensor(np.asarray(self.biu_cache[global_idx], dtype=np.float32), dtype=torch.float32)
        label = torch.tensor(int(row["label"]), dtype=torch.long)
        meta = {
            "sequence_id": row["sequence_id"],
            "split": row["split"],
            "set_id": row["set_id"],
            "video_id": row["video_id"],
            "ped_id": row["ped_id"],
            "start_frame": int(row["start_frame"]),
            "end_frame": int(row["end_frame"]),
            "target_frame": int(row["target_frame"]),
            "label": int(row["label"]),
        }
        return out, label, meta


def write_pretraining_audits(out_dir: Path, artifacts: RunArtifacts) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts.rows.to_csv(out_dir / "protocol_rows.csv", index=False)
    pd.DataFrame(
        [
            {
                "split": split,
                "n_samples": artifacts.audit["split_counts"][split],
                "n_positive": artifacts.audit["split_positive_counts"][split],
                "positive_rate": artifacts.audit["split_positive_counts"][split] / max(artifacts.audit["split_counts"][split], 1),
            }
            for split in SPLITS
        ]
    ).to_csv(out_dir / "population_summary.csv", index=False)
    (out_dir / "pretraining_audit.json").write_text(json.dumps(artifacts.audit, indent=2), encoding="utf-8")
    (out_dir / "biu_numeric_feature_names.json").write_text(json.dumps(artifacts.biu_feature_names, indent=2), encoding="utf-8")
    (out_dir / "biu_extractor_output_names.json").write_text(json.dumps(artifacts.audit["biu_extractor_output_names"], indent=2), encoding="utf-8")


class StreamAttention(nn.Module):
    def __init__(self, hidden_dim: int, attention_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden_dim, attention_dim), nn.Tanh(), nn.Linear(attention_dim, 1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.net(x).squeeze(2)
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(x * weights.unsqueeze(2), dim=1)
        return context, weights


class Exp032BaselineModel(nn.Module):
    def __init__(self, input_sizes: dict[str, int]):
        super().__init__()
        self.stream_names = list(ACTIVE_STREAMS)
        self.lstm_hidden_size = LSTM_HIDDEN_SIZE
        self.lstm_output_dim = LSTM_HIDDEN_SIZE * 2
        self.lstms = nn.ModuleDict()
        self.attentions = nn.ModuleDict()
        for name in self.stream_names:
            self.lstms[name] = nn.LSTM(
                input_size=input_sizes[name],
                hidden_size=LSTM_HIDDEN_SIZE,
                num_layers=NUM_LSTM_LAYERS,
                batch_first=True,
                dropout=DROPOUT_RATE if NUM_LSTM_LAYERS > 1 else 0.0,
                bidirectional=True,
            )
            self.attentions[name] = StreamAttention(self.lstm_output_dim, ATTENTION_DIM)
        self.fusion_weights = nn.Parameter(torch.ones(len(self.stream_names)))
        self.dropout = nn.Dropout(DROPOUT_RATE)
        self.fc1 = nn.Linear(self.lstm_output_dim, max(2 * 4, self.lstm_output_dim // 2))
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(max(2 * 4, self.lstm_output_dim // 2), 2)

    def fused_context(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        contexts = []
        for name in self.stream_names:
            lstm_out, _ = self.lstms[name](x[name])
            context, _ = self.attentions[name](lstm_out)
            contexts.append(context)
        stacked = torch.stack(contexts, dim=1)
        weights = torch.softmax(self.fusion_weights, dim=0).view(1, -1, 1)
        return torch.sum(stacked * weights, dim=1)

    def forward(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        fused = self.fused_context(x)
        hidden = self.dropout(fused)
        hidden = self.relu(self.fc1(hidden))
        hidden = self.dropout(hidden)
        return self.fc2(hidden)


class Exp032BiuAugmentedModel(Exp032BaselineModel):
    def __init__(self, input_sizes: dict[str, int], biu_dim: int):
        super().__init__(input_sizes)
        self.biu_proj = nn.Sequential(nn.Linear(biu_dim, BIU_PROJECTION_DIM), nn.ReLU())
        self.fc1 = nn.Linear(self.lstm_output_dim + BIU_PROJECTION_DIM, max(2 * 4, self.lstm_output_dim // 2))

    def forward(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        fused = self.fused_context(x)
        biu_embed = self.biu_proj(x["biu_v2"])
        fused_plus_biu = torch.cat([fused, biu_embed], dim=1)
        hidden = self.dropout(fused_plus_biu)
        hidden = self.relu(self.fc1(hidden))
        hidden = self.dropout(hidden)
        return self.fc2(hidden)


def model_parameter_count(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return int(total), int(trainable)


def save_training_checkpoint(
    path: Path,
    *,
    variant: str,
    seed: int,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    best_state: dict[str, torch.Tensor],
    best_epoch: int,
    best_val_f1: float,
    history_rows: list[dict[str, object]],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "variant": variant,
            "seed": int(seed),
            "epoch": int(epoch),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_state": best_state,
            "best_epoch": int(best_epoch),
            "best_val_f1": float(best_val_f1),
            "history_rows": history_rows,
        },
        temp_path,
    )
    temp_path.replace(path)


def load_training_checkpoint(path: Path, *, expected_variant: str, expected_seed: int) -> dict[str, object]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("variant") != expected_variant or int(payload.get("seed", -1)) != int(expected_seed):
        raise ValueError("Training checkpoint identity does not match the requested run.")
    return payload


def _subset_tensors(arrays: dict[str, np.ndarray], idx: np.ndarray, include_biu: bool) -> dict[str, torch.Tensor]:
    tensors = {name: torch.tensor(arrays[name][idx], dtype=torch.float32, device=DEVICE) for name in ACTIVE_STREAMS}
    if include_biu:
        tensors["biu_v2"] = torch.tensor(arrays["biu_v2"][idx], dtype=torch.float32, device=DEVICE)
    return tensors


def _iter_batches(indices: np.ndarray, batch_size: int, shuffle: bool, seed: int) -> list[np.ndarray]:
    idx = indices.copy()
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
    return [idx[i : i + batch_size] for i in range(0, len(idx), batch_size)]


def _slice_batch(tensors: dict[str, torch.Tensor], batch_idx: np.ndarray) -> dict[str, torch.Tensor]:
    return {name: tensor[batch_idx] for name, tensor in tensors.items()}


def _predict(model: nn.Module, tensors: dict[str, torch.Tensor], indices: np.ndarray) -> np.ndarray:
    probs: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch_idx in _iter_batches(indices, BATCH_SIZE, shuffle=False, seed=0):
            batch = _slice_batch(tensors, batch_idx)
            logits = model(batch)
            probs.append(torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy())
    return np.concatenate(probs, axis=0) if probs else np.array([], dtype=np.float64)


def _classification_rows(meta: pd.DataFrame, probs: np.ndarray, threshold: float, seed: int, variant: str, threshold_rule: str) -> pd.DataFrame:
    preds = (probs >= threshold).astype(int)
    out = meta.copy().reset_index(drop=True)
    out["seed"] = seed
    out["variant"] = variant
    out["threshold_rule"] = threshold_rule
    out["threshold"] = float(threshold)
    out["probability"] = probs
    out["prediction"] = preds
    return out


def _select_only_fixed_and_val_f1(y_val: np.ndarray, p_val: np.ndarray):
    return [sel for sel in jaad._select_thresholds(y_val, p_val) if sel.rule in {"fixed_0_5", "val_f1"}]


def train_variant(variant: str, seed: int, artifacts: RunArtifacts, epochs: int = NUM_EPOCHS) -> tuple[list[dict[str, object]], dict[str, pd.DataFrame], dict[str, object]]:
    include_biu = variant == "variant_b_mslstm_plus_biu_v2"
    set_seed(seed)
    split_idx = artifacts.split_index
    train_tensors = _subset_tensors(artifacts.arrays, split_idx["train"], include_biu=include_biu)
    val_tensors = _subset_tensors(artifacts.arrays, split_idx["val"], include_biu=include_biu)
    test_tensors = _subset_tensors(artifacts.arrays, split_idx["test"], include_biu=include_biu)
    train_local_idx = np.arange(len(split_idx["train"]), dtype=np.int64)
    val_local_idx = np.arange(len(split_idx["val"]), dtype=np.int64)
    test_local_idx = np.arange(len(split_idx["test"]), dtype=np.int64)
    y_train = torch.tensor(artifacts.labels[split_idx["train"]], dtype=torch.long, device=DEVICE)
    y_val = artifacts.labels[split_idx["val"]]
    y_test = artifacts.labels[split_idx["test"]]

    if include_biu:
        model = Exp032BiuAugmentedModel(STREAM_DIMS, biu_dim=artifacts.arrays["biu_v2"].shape[1]).to(DEVICE)
    else:
        model = Exp032BaselineModel(STREAM_DIMS).to(DEVICE)

    n_pos = int((artifacts.labels[split_idx["train"]] == 1).sum())
    n_total = len(split_idx["train"])
    n_neg = n_total - n_pos
    w0, w1 = (n_total / (2.0 * n_neg), n_total / (2.0 * n_pos)) if n_pos > 0 and n_neg > 0 else (1.0, 1.0)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([w0, w1], dtype=torch.float32, device=DEVICE))
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_state = None
    best_epoch = -1
    best_val_f1 = -1.0
    history_rows = []
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        batch_losses = []
        for batch_idx in _iter_batches(train_local_idx, BATCH_SIZE, shuffle=True, seed=seed + epoch):
            x = _slice_batch(train_tensors, batch_idx)
            y = y_train[batch_idx]
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu().item()))

        p_val = _predict(model, val_tensors, val_local_idx)
        val_metrics = jaad._compute_metrics(y_val, p_val, 0.5)
        history_rows.append(
            {
                "variant": variant,
                "seed": seed,
                "epoch": epoch,
                "train_loss_mean": float(np.mean(batch_losses) if batch_losses else float("nan")),
                "val_f1_at_0p5": float(val_metrics["f1"]),
                "val_precision_at_0p5": float(val_metrics["precision"]),
                "val_recall_at_0p5": float(val_metrics["recall"]),
                "val_roc_auc": float(val_metrics["roc_auc"]),
                "val_pr_auc": float(val_metrics["pr_auc"]),
            }
        )
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = float(val_metrics["f1"])
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError(f"No best state captured for {variant} seed {seed}")
    model.load_state_dict(best_state)
    p_val = _predict(model, val_tensors, val_local_idx)
    p_test = _predict(model, test_tensors, test_local_idx)
    selection_rows = []
    predictions: dict[str, pd.DataFrame] = {}
    for sel in _select_only_fixed_and_val_f1(y_val, p_val):
        test_metrics = jaad._compute_metrics(y_test, p_test, float(sel.threshold))
        row = {
            "variant": variant,
            "seed": seed,
            "threshold_rule": sel.rule,
            "threshold_status": sel.status,
            "selected_threshold": float(sel.threshold),
            "best_epoch": int(best_epoch),
            "train_rows": int(len(split_idx["train"])),
            "val_rows": int(len(split_idx["val"])),
            "test_rows": int(len(split_idx["test"])),
            "train_positive_rate": float(artifacts.labels[split_idx["train"]].mean()),
            "val_positive_rate": float(y_val.mean()),
            "test_positive_rate": float(y_test.mean()),
            "wall_time_seconds": float(time.time() - start_time),
        }
        for prefix, metrics in (("val", sel.val_metrics), ("test", test_metrics)):
            for metric_name, value in metrics.items():
                row[f"{prefix}_{metric_name}"] = float(value)
        selection_rows.append(row)
        meta = artifacts.rows.iloc[split_idx["test"]][["sequence_id", "split", "set_id", "video_id", "ped_id", "start_frame", "end_frame", "target_frame", "label"]]
        predictions[sel.rule] = _classification_rows(meta, p_test, float(sel.threshold), seed, variant, sel.rule)

    total_params, trainable_params = model_parameter_count(model)
    run_info = {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "history": pd.DataFrame(history_rows),
    }
    return selection_rows, predictions, run_info


def _uncollate_meta(batch_meta: dict[str, object], batch_size: int) -> list[dict[str, object]]:
    rows = []
    for i in range(batch_size):
        row = {}
        for key, value in batch_meta.items():
            if torch.is_tensor(value):
                row[key] = value[i].item()
            else:
                row[key] = value[i]
        rows.append(row)
    return rows


def _predict_loader(model: nn.Module, loader: DataLoader) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    probs: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    meta_rows: list[dict[str, object]] = []
    model.eval()
    with torch.no_grad():
        for batch_x, batch_y, batch_meta in loader:
            inputs = {name: tensor.to(DEVICE) for name, tensor in batch_x.items()}
            logits = model(inputs)
            batch_probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            probs.append(batch_probs)
            labels.append(batch_y.detach().cpu().numpy())
            meta_rows.extend(_uncollate_meta(batch_meta, len(batch_probs)))
    return np.concatenate(labels), np.concatenate(probs), pd.DataFrame(meta_rows)


def prepare_full_run_context(out_root: Path) -> FullRunContext:
    pie_db = load_pie_database()
    rows = load_exp032_protocol_rows().reset_index(drop=True)
    rows["global_index"] = np.arange(len(rows), dtype=np.int64)
    recovered_train = load_recovered_train_rows(rows)
    scalers = compute_train_scalers(pie_db)
    pose_cache = PoseCache(POSE_DATA_DIR)
    biu_cache_path, biu_feature_names, biu_audit = precompute_biu_cache(rows, pie_db, pose_cache, scalers, out_root)
    audit = {
        "n_rows": int(len(rows)),
        "candidate_split_counts": {split: int((rows["split"] == split).sum()) for split in SPLITS},
        "split_counts": {
            "train": int(len(recovered_train)),
            "val": int((rows["split"] == "val").sum()),
            "test": int((rows["split"] == "test").sum()),
        },
        "split_positive_counts": {
            "train": int(recovered_train["label"].sum()),
            "val": int(rows.loc[rows["split"] == "val", "label"].sum()),
            "test": int(rows.loc[rows["split"] == "test", "label"].sum()),
        },
        "train_index_path": str(TRAIN_INDEX_PATH),
        **biu_audit,
    }
    return FullRunContext(
        rows=rows,
        pie_db=pie_db,
        pose_cache=pose_cache,
        scalers=scalers,
        biu_cache_path=biu_cache_path,
        biu_feature_names=biu_feature_names,
        audit=audit,
    )


def train_variant_lazy(
    variant: str,
    seed: int,
    context: FullRunContext,
    epochs: int = NUM_EPOCHS,
    checkpoint_dir: Path | None = None,
) -> tuple[list[dict[str, object]], dict[str, pd.DataFrame], dict[str, object]]:
    include_biu = variant == "variant_b_mslstm_plus_biu_v2"
    set_seed(seed)

    train_rows = load_recovered_train_rows(context.rows)
    val_rows = context.rows[context.rows["split"] == "val"].reset_index(drop=True)
    test_rows = context.rows[context.rows["split"] == "test"].reset_index(drop=True)
    train_dataset = Exp032LazyDataset(train_rows, context.pie_db, context.pose_cache, context.scalers, include_biu, context.biu_cache_path)
    val_dataset = Exp032LazyDataset(val_rows, context.pie_db, context.pose_cache, context.scalers, include_biu, context.biu_cache_path)
    test_dataset = Exp032LazyDataset(test_rows, context.pie_db, context.pose_cache, context.scalers, include_biu, context.biu_cache_path)
    pin_memory = DEVICE.type == "cuda"
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=pin_memory)

    if include_biu:
        model = Exp032BiuAugmentedModel(STREAM_DIMS, biu_dim=len(context.biu_feature_names)).to(DEVICE)
    else:
        model = Exp032BaselineModel(STREAM_DIMS).to(DEVICE)

    y_train_np = train_rows["label"].to_numpy(dtype=np.int64)
    n_pos = int((y_train_np == 1).sum())
    n_total = len(y_train_np)
    n_neg = n_total - n_pos
    w0, w1 = (n_total / (2.0 * n_neg), n_total / (2.0 * n_pos)) if n_pos > 0 and n_neg > 0 else (1.0, 1.0)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([w0, w1], dtype=torch.float32, device=DEVICE))
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_state = None
    best_epoch = -1
    best_val_f1 = -1.0
    history_rows = []
    start_epoch = 1
    checkpoint_path = None if checkpoint_dir is None else Path(checkpoint_dir) / f"{variant}__seed{seed}.pt"
    if checkpoint_path is not None and checkpoint_path.exists():
        checkpoint = load_training_checkpoint(checkpoint_path, expected_variant=variant, expected_seed=seed)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(DEVICE)
        best_state = checkpoint["best_state"]
        best_epoch = int(checkpoint["best_epoch"])
        best_val_f1 = float(checkpoint["best_val_f1"])
        history_rows = list(checkpoint["history_rows"])
        start_epoch = int(checkpoint["epoch"]) + 1
        print(f"Resuming {variant} seed {seed} at epoch {start_epoch}/{epochs}", flush=True)
    start_time = time.time()
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        batch_losses = []
        for batch_x, batch_y, _ in train_loader:
            inputs = {name: tensor.to(DEVICE) for name, tensor in batch_x.items()}
            labels = batch_y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu().item()))

        y_val, p_val, _ = _predict_loader(model, val_loader)
        val_metrics = jaad._compute_metrics(y_val, p_val, 0.5)
        history_rows.append(
            {
                "variant": variant,
                "seed": seed,
                "epoch": epoch,
                "train_loss_mean": float(np.mean(batch_losses) if batch_losses else float("nan")),
                "val_f1_at_0p5": float(val_metrics["f1"]),
                "val_precision_at_0p5": float(val_metrics["precision"]),
                "val_recall_at_0p5": float(val_metrics["recall"]),
                "val_roc_auc": float(val_metrics["roc_auc"]),
                "val_pr_auc": float(val_metrics["pr_auc"]),
            }
        )
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = float(val_metrics["f1"])
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if checkpoint_path is not None:
            save_training_checkpoint(
                checkpoint_path,
                variant=variant,
                seed=seed,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                best_state=best_state,
                best_epoch=best_epoch,
                best_val_f1=best_val_f1,
                history_rows=history_rows,
            )
        print(json.dumps(history_rows[-1], sort_keys=True), flush=True)

    if best_state is None:
        raise RuntimeError(f"No best state captured for {variant} seed {seed}")
    model.load_state_dict(best_state)
    y_val, p_val, _ = _predict_loader(model, val_loader)
    y_test, p_test, test_meta = _predict_loader(model, test_loader)

    selection_rows = []
    predictions: dict[str, pd.DataFrame] = {}
    for sel in _select_only_fixed_and_val_f1(y_val, p_val):
        test_metrics = jaad._compute_metrics(y_test, p_test, float(sel.threshold))
        row = {
            "variant": variant,
            "seed": seed,
            "threshold_rule": sel.rule,
            "threshold_status": sel.status,
            "selected_threshold": float(sel.threshold),
            "best_epoch": int(best_epoch),
            "train_rows": int(len(train_rows)),
            "val_rows": int(len(val_rows)),
            "test_rows": int(len(test_rows)),
            "train_positive_rate": float(train_rows["label"].mean()),
            "val_positive_rate": float(val_rows["label"].mean()),
            "test_positive_rate": float(test_rows["label"].mean()),
            "wall_time_seconds": float(time.time() - start_time),
        }
        for prefix, metrics in (("val", sel.val_metrics), ("test", test_metrics)):
            for metric_name, value in metrics.items():
                row[f"{prefix}_{metric_name}"] = float(value)
        selection_rows.append(row)
        predictions[sel.rule] = _classification_rows(test_meta, p_test, float(sel.threshold), seed, variant, sel.rule)

    total_params, trainable_params = model_parameter_count(model)
    sample_batch_x, _, _ = next(iter(test_loader))
    latency_inputs = {name: tensor.to(DEVICE) for name, tensor in sample_batch_x.items()}
    run_info = {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "history": pd.DataFrame(history_rows),
        "latency": measure_latency(model, latency_inputs),
    }
    return selection_rows, predictions, run_info


def summarize_results(results_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [c for c in results_df.columns if c.startswith("test_") or c.startswith("val_")]
    grouped = results_df.groupby(["variant", "threshold_rule"], dropna=False)[metric_cols].agg(["mean", "std"])
    grouped.columns = [f"{a}_{b}" for a, b in grouped.columns]
    return grouped.reset_index()


def build_paired_deltas(results_df: pd.DataFrame) -> pd.DataFrame:
    metric_names = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    rows = []
    for threshold_rule in sorted(results_df["threshold_rule"].unique()):
        a = results_df[(results_df["variant"] == "variant_a_mslstm_control") & (results_df["threshold_rule"] == threshold_rule)].sort_values("seed")
        b = results_df[(results_df["variant"] == "variant_b_mslstm_plus_biu_v2") & (results_df["threshold_rule"] == threshold_rule)].sort_values("seed")
        if len(a) != len(b):
            raise RuntimeError(f"Paired delta mismatch for {threshold_rule}: {len(a)} vs {len(b)} seeds")
        row = {"threshold_rule": threshold_rule}
        for metric in metric_names:
            diffs = b[f"test_{metric}"].to_numpy(dtype=float) - a[f"test_{metric}"].to_numpy(dtype=float)
            row[f"{metric}_delta_mean"] = float(diffs.mean())
            row[f"{metric}_delta_std"] = float(diffs.std(ddof=0))
            row[f"{metric}_direction_consistent"] = bool(np.all(diffs >= 0) or np.all(diffs <= 0))
            row[f"{metric}_delta_signs"] = ",".join("+" if d > 0 else "-" if d < 0 else "0" for d in diffs)
        rows.append(row)
    return pd.DataFrame(rows)


def build_parity_table(results_df: pd.DataFrame) -> pd.DataFrame:
    fixed = results_df[(results_df["variant"] == "variant_a_mslstm_control") & (results_df["threshold_rule"] == "fixed_0_5")].copy()
    if fixed.empty:
        return pd.DataFrame()
    fresh = fixed[["seed", "test_accuracy", "test_precision", "test_recall", "test_f1", "test_roc_auc", "test_pr_auc"]].mean().to_dict()
    rows = []
    for metric, ref_value in PAPER_REFERENCE.items():
        fresh_value = float(fresh[f"test_{metric}"])
        rows.append(
            {
                "metric": metric,
                "paper_reference": ref_value,
                "fresh_variant_a_mean": fresh_value,
                "absolute_delta": fresh_value - ref_value,
                "wild_inconsistency": bool((metric == "f1" and abs(fresh_value - ref_value) > WILD_F1_DROP) or (metric == "roc_auc" and abs(fresh_value - ref_value) > WILD_ROC_AUC_DROP)),
            }
        )
    return pd.DataFrame(rows)


def build_error_transitions(pred_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for seed in sorted(pred_df["seed"].unique()):
        a = pred_df[(pred_df["seed"] == seed) & (pred_df["variant"] == "variant_a_mslstm_control") & (pred_df["threshold_rule"] == "fixed_0_5")].sort_values("sequence_id")
        b = pred_df[(pred_df["seed"] == seed) & (pred_df["variant"] == "variant_b_mslstm_plus_biu_v2") & (pred_df["threshold_rule"] == "fixed_0_5")].sort_values("sequence_id")
        if a["sequence_id"].tolist() != b["sequence_id"].tolist():
            raise RuntimeError(f"Prediction alignment mismatch for fixed_0_5 seed {seed}")
        for base_row, cand_row in zip(a.itertuples(index=False), b.itertuples(index=False)):
            label = int(base_row.label)
            base_pred = int(base_row.prediction)
            cand_pred = int(cand_row.prediction)
            if label == 0:
                base_state = "FP" if base_pred == 1 else "TN"
                cand_state = "FP" if cand_pred == 1 else "TN"
            else:
                base_state = "TP" if base_pred == 1 else "FN"
                cand_state = "TP" if cand_pred == 1 else "FN"
            rows.append(
                {
                    "seed": seed,
                    "sequence_id": base_row.sequence_id,
                    "label": label,
                    "baseline_state": base_state,
                    "candidate_state": cand_state,
                    "transition": f"{base_state}->{cand_state}",
                    "baseline_probability": float(base_row.probability),
                    "candidate_probability": float(cand_row.probability),
                    "probability_shift": float(cand_row.probability - base_row.probability),
                }
            )
    return pd.DataFrame(rows)


def measure_latency(model: nn.Module, example_batch: dict[str, torch.Tensor], warmup: int = 10, repeats: int = 50) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(example_batch)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(repeats):
            _ = model(example_batch)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
    batch_size = next(iter(example_batch.values())).shape[0]
    return {
        "latency_seconds_per_batch": elapsed / repeats,
        "latency_seconds_per_sample": elapsed / repeats / batch_size,
        "batch_size": int(batch_size),
        "device": DEVICE.type,
    }


def save_outputs(
    out_root: Path,
    results_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    delta_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    parity_df: pd.DataFrame,
    transitions_df: pd.DataFrame,
    param_df: pd.DataFrame,
    latency_df: pd.DataFrame,
    histories: list[pd.DataFrame],
) -> None:
    metrics_dir = out_root / "metrics"
    predictions_dir = out_root / "predictions"
    logs_dir = out_root / "logs"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(metrics_dir / "per_seed_results.csv", index=False)
    summary_df.to_csv(metrics_dir / "summary_results.csv", index=False)
    delta_df.to_csv(metrics_dir / "paired_deltas.csv", index=False)
    results_df[results_df["threshold_rule"] == "fixed_0_5"].to_csv(metrics_dir / "fixed_0p5_results.csv", index=False)
    results_df[results_df["threshold_rule"] == "val_f1"].to_csv(metrics_dir / "val_f1_results.csv", index=False)
    results_df[results_df["threshold_rule"] == "fixed_0_5"].to_csv(metrics_dir / "published_protocol_results.csv", index=False)
    pred_df.to_csv(predictions_dir / "test_predictions.csv", index=False)
    parity_df.to_csv(metrics_dir / "published_parity_check.csv", index=False)
    transitions_df.to_csv(metrics_dir / "error_transitions_fixed_0p5.csv", index=False)
    param_df.to_csv(metrics_dir / "parameter_counts.csv", index=False)
    latency_df.to_csv(metrics_dir / "latency_summary.csv", index=False)
    pd.concat(histories, ignore_index=True).to_csv(logs_dir / "training_history.csv", index=False)


def run_experiment(mode: str, epochs: int, out_root: Path) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    metrics_dir = out_root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    histories: list[pd.DataFrame] = []
    param_rows: list[dict[str, object]] = []
    latency_rows: list[dict[str, object]] = []

    if mode == "smoke":
        artifacts = assemble_exp032_windows(sample_only=True, sample_limit=12)
        write_pretraining_audits(metrics_dir, artifacts)
        seeds = (42,)
        for variant in VARIANTS:
            for seed in seeds:
                rows, prediction_map, info = train_variant(variant, seed, artifacts, epochs=epochs)
                all_rows.extend(rows)
                histories.append(info["history"])
                param_rows.append(
                    {
                        "variant": variant,
                        "seed": seed,
                        "total_params": info["total_params"],
                        "trainable_params": info["trainable_params"],
                    }
                )
                for _, pred_df in prediction_map.items():
                    prediction_frames.append(pred_df)
                include_biu = variant == "variant_b_mslstm_plus_biu_v2"
                sample_idx = artifacts.split_index["test"][: min(BATCH_SIZE, len(artifacts.split_index["test"]))]
                batch = _subset_tensors(artifacts.arrays, sample_idx, include_biu=include_biu)
                if include_biu:
                    model = Exp032BiuAugmentedModel(STREAM_DIMS, artifacts.arrays["biu_v2"].shape[1]).to(DEVICE)
                else:
                    model = Exp032BaselineModel(STREAM_DIMS).to(DEVICE)
                latency = measure_latency(model, batch)
                latency_rows.append({"variant": variant, "seed": seed, **latency})
    else:
        context = prepare_full_run_context(out_root)
        context.rows.to_csv(metrics_dir / "protocol_rows.csv", index=False)
        pd.DataFrame(
            [
                {
                    "split": split,
                    "n_samples": context.audit["split_counts"][split],
                    "n_positive": context.audit["split_positive_counts"][split],
                    "positive_rate": context.audit["split_positive_counts"][split] / max(context.audit["split_counts"][split], 1),
                }
                for split in SPLITS
            ]
        ).to_csv(metrics_dir / "population_summary.csv", index=False)
        (metrics_dir / "pretraining_audit.json").write_text(json.dumps(context.audit, indent=2), encoding="utf-8")
        (metrics_dir / "biu_numeric_feature_names.json").write_text(json.dumps(context.biu_feature_names, indent=2), encoding="utf-8")
        (metrics_dir / "biu_extractor_output_names.json").write_text(json.dumps(context.audit["biu_extractor_output_names"], indent=2), encoding="utf-8")
        for variant in VARIANTS:
            for seed in SEEDS:
                rows, prediction_map, info = train_variant_lazy(
                    variant,
                    seed,
                    context,
                    epochs=epochs,
                    checkpoint_dir=out_root / "checkpoints",
                )
                all_rows.extend(rows)
                histories.append(info["history"])
                param_rows.append(
                    {
                        "variant": variant,
                        "seed": seed,
                        "total_params": info["total_params"],
                        "trainable_params": info["trainable_params"],
                    }
                )
                for _, pred_df in prediction_map.items():
                    prediction_frames.append(pred_df)
                latency_rows.append({"variant": variant, "seed": seed, **info["latency"]})

    results_df = pd.DataFrame(all_rows)
    pred_df = pd.concat(prediction_frames, ignore_index=True)
    summary_df = summarize_results(results_df)
    delta_df = build_paired_deltas(results_df)
    parity_df = build_parity_table(results_df)
    transitions_df = build_error_transitions(pred_df)
    param_df = pd.DataFrame(param_rows)
    latency_df = pd.DataFrame(latency_rows)
    save_outputs(out_root, results_df, summary_df, delta_df, pred_df, parity_df, transitions_df, param_df, latency_df, histories)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Exp032 MSLSTM BIU-v2 transfer experiment.")
    parser.add_argument("--mode", choices=("smoke", "train"), default="train")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_experiment(mode=args.mode, epochs=args.epochs, out_root=args.output_root)


if __name__ == "__main__":
    main()
