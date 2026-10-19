"""Frozen core for Exp050 PIE practical-behavior replacement."""

from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SEQ_LEN = 16
SEEDS = (42, 43, 44)
VARIANTS = ("A", "B", "C", "D", "E", "F")
FIXED_THRESHOLD = 0.5
NONINFERIORITY_TOLERANCE = 0.005
EPOCHS = 15
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
GAUSSIAN_NOISE_STD = 0.02
LSTM_HIDDEN_SIZE = 256
NUM_LSTM_LAYERS = 2
STREAM_CONTEXT_DIM = 512
ATTENTION_DIM = 128
DROPOUT_RATE = 0.3

VARIANT_STREAMS = {
    "A": ("bbox", "ped_action", "ped_look", "ego_speed", "ego_acc", "static_context", "yolop"),
    "B": ("bbox", "ped_action", "ped_look", "ego_speed", "ego_acc", "yolop"),
    "C": ("bbox", "ego_speed", "ego_acc", "yolop"),
    "D": ("bbox", "ego_speed", "ego_acc", "yolop", "pose"),
    "E": ("bbox", "ego_speed", "ego_acc", "yolop", "biu_v2"),
    "F": ("bbox", "ego_speed", "ego_acc", "yolop", "biu_v1"),
}
STREAM_DIMS = {
    "bbox": 4,
    "ped_action": 1,
    "ped_look": 1,
    "ego_speed": 1,
    "ego_acc": 2,
    "static_context": 23,
    "yolop": 20,
    "pose": 34,
    "biu_v2": 63,
    "biu_v1": 72,
}

EXPECTED_SPLIT_COUNTS = {"train": 2366, "val": 3619, "test": 3760}
EXPECTED_SPLIT_SETS = {
    "train": {"set01", "set02", "set06"},
    "val": {"set04", "set05"},
    "test": {"set03"},
}


def _load_biu_dependencies():
    from src.biu_v2_extractor import extract_biu_v2
    from src.features.biu_features import (
        AGGREGATIONS,
        BIU_BASE_FEATURES,
        aggregate_biu_sequence,
        compute_frame_biu_features,
        frame_has_invalid_required_keypoints,
        frame_has_low_confidence,
    )

    return {
        "extract_biu_v2": extract_biu_v2,
        "aggregations": AGGREGATIONS,
        "base_features": BIU_BASE_FEATURES,
        "aggregate": aggregate_biu_sequence,
        "frame": compute_frame_biu_features,
        "invalid": frame_has_invalid_required_keypoints,
        "low_confidence": frame_has_low_confidence,
    }


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def observed_frames(row: pd.Series | object) -> list[int]:
    start = int(row["start_frame"] if isinstance(row, pd.Series) else getattr(row, "start_frame"))
    end = int(row["end_frame"] if isinstance(row, pd.Series) else getattr(row, "end_frame"))
    return list(range(start, end + 1))


def validate_split_rows(rows: pd.DataFrame, *, allowed_splits: set[str]) -> pd.DataFrame:
    required = {
        "sequence_id", "split", "set_id", "video_id", "ped_id", "start_frame",
        "end_frame", "event_frame", "time_to_event_frames", "observation_length",
        "frame_step", "frame_sampling", "label",
    }
    missing = sorted(required.difference(rows.columns))
    if missing:
        raise ValueError(f"Split rows are missing columns: {missing}")
    checked = rows.reset_index(drop=True).copy()
    if checked.empty or checked["sequence_id"].duplicated().any():
        raise ValueError("Split rows must be nonempty with unique sequence IDs.")
    if not set(checked["split"].astype(str)).issubset(allowed_splits):
        raise ValueError("Split rows contain a disallowed population.")
    consecutive = (
        checked["observation_length"].astype(int).eq(SEQ_LEN)
        & checked["frame_step"].astype(int).eq(1)
        & checked["frame_sampling"].astype(str).eq("consecutive")
        & (checked["end_frame"].astype(int) - checked["start_frame"].astype(int)).eq(SEQ_LEN - 1)
    )
    if not consecutive.all():
        raise ValueError("Every Exp050 window must contain exactly 16 consecutive observed frames.")
    if "last_observed_frame" in checked and not checked["last_observed_frame"].astype(int).eq(
        checked["end_frame"].astype(int)
    ).all():
        raise ValueError("Last-observed frame must equal end_frame.")
    if not checked["time_to_event_frames"].astype(int).between(30, 60).all():
        raise ValueError("Exp050 TTE must remain within 30--60 frames.")
    if not (checked["event_frame"].astype(int) > checked["end_frame"].astype(int)).all():
        raise ValueError("Every event must be a future event outside the observation window.")
    if not set(checked["label"].astype(int)).issubset({0, 1}):
        raise ValueError("Labels must be binary.")
    checked["cache_index"] = np.arange(len(checked), dtype=np.int64)
    return checked


def validate_full_split(rows: pd.DataFrame) -> pd.DataFrame:
    checked = validate_split_rows(rows, allowed_splits={"train", "val", "test"})
    counts = checked["split"].value_counts().astype(int).to_dict()
    if counts != EXPECTED_SPLIT_COUNTS:
        raise ValueError(f"Frozen split counts drifted: {counts}")
    for split, expected_sets in EXPECTED_SPLIT_SETS.items():
        actual = set(checked.loc[checked["split"] == split, "set_id"].astype(str))
        if actual != expected_sets:
            raise ValueError(f"Frozen {split} set membership drifted: {sorted(actual)}")
    return checked


def build_oversampling_manifest(rows: pd.DataFrame, seed: int) -> pd.DataFrame:
    source = rows.reset_index(drop=True).copy()
    if set(source["split"].astype(str)) != {"train"}:
        raise ValueError("Oversampling accepts training rows only.")
    if "cache_index" not in source:
        source["cache_index"] = np.arange(len(source), dtype=np.int64)
    counts = source["label"].astype(int).value_counts()
    if set(counts.index.astype(int)) != {0, 1}:
        raise ValueError("Oversampling requires both classes.")
    majority = int(counts.max())
    minority_label = int(counts.idxmin())
    rng = np.random.default_rng(int(seed))
    minority_positions = np.flatnonzero(source["label"].to_numpy(dtype=int) == minority_label)
    duplicates = rng.choice(minority_positions, size=majority - int(counts.min()), replace=True)
    records: list[dict[str, object]] = []
    for row in source.itertuples(index=False):
        records.append(
            {
                "source_cache_index": int(row.cache_index),
                "sequence_id": str(row.sequence_id),
                "label": int(row.label),
                "augmented": False,
                "noise_seed": -1,
            }
        )
    for position in duplicates:
        row = source.iloc[int(position)]
        records.append(
            {
                "source_cache_index": int(row["cache_index"]),
                "sequence_id": str(row["sequence_id"]),
                "label": int(row["label"]),
                "augmented": True,
                "noise_seed": int(rng.integers(0, np.iinfo(np.int32).max)),
            }
        )
    manifest = pd.DataFrame.from_records(records)
    return manifest.iloc[rng.permutation(len(manifest))].reset_index(drop=True)


def normalized_pose_to_pixels(pose: np.ndarray, *, width: int, height: int) -> np.ndarray:
    normalized = np.asarray(pose, dtype=np.float64)
    if normalized.shape != (SEQ_LEN, 34):
        raise ValueError(f"Expected normalized pose ({SEQ_LEN},34), got {normalized.shape}.")
    coords = normalized.reshape(SEQ_LEN, 17, 2).copy()
    coords[:, :, 0] *= float(width)
    coords[:, :, 1] *= float(height)
    return coords


def normalized_bbox_to_pixels(bbox: np.ndarray, *, width: int, height: int) -> np.ndarray:
    normalized = np.asarray(bbox, dtype=np.float64)
    if normalized.shape != (SEQ_LEN, 4):
        raise ValueError(f"Expected normalized BBox ({SEQ_LEN},4), got {normalized.shape}.")
    cx, cy, bw, bh = normalized.T
    return np.column_stack(
        (
            (cx - bw / 2.0) * width,
            (cy - bh / 2.0) * height,
            (cx + bw / 2.0) * width,
            (cy + bh / 2.0) * height,
        )
    )


def extract_biu_v1_window(
    coords_xy: np.ndarray,
    kp_conf: np.ndarray | None,
) -> tuple[np.ndarray, tuple[str, ...]]:
    deps = _load_biu_dependencies()
    coords = np.asarray(coords_xy, dtype=np.float64)
    if coords.shape != (SEQ_LEN, 17, 2):
        raise ValueError(f"Expected BIU-v1 pose ({SEQ_LEN},17,2), got {coords.shape}.")
    confidence = None if kp_conf is None else np.asarray(kp_conf, dtype=np.float64)
    if confidence is not None and confidence.shape != (SEQ_LEN, 17):
        raise ValueError("Pose confidence must align with BIU-v1 pose.")
    names = tuple(
        f"{feature}_{aggregation}"
        for feature in deps["base_features"]
        for aggregation in deps["aggregations"]
    ) + ("biu_valid_frame_count", "biu_low_confidence_frame_count")
    frame_features = [deps["frame"](frame) for frame in coords]
    aggregated = deps["aggregate"](frame_features)
    values = [aggregated[name] for name in names[:-2]]
    values.append(sum(not deps["invalid"](frame) for frame in coords))
    values.append(0 if confidence is None else sum(deps["low_confidence"](frame) for frame in confidence))
    result = np.nan_to_num(np.asarray(values, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    if result.shape != (72,):
        raise RuntimeError(f"Frozen BIU-v1 must be 72-D, got {result.shape}.")
    return result.astype(np.float32), names


def extract_biu_v2_window(
    normalized_pose: np.ndarray,
    normalized_bbox: np.ndarray,
    *,
    width: int,
    height: int,
) -> tuple[np.ndarray, tuple[str, ...], str]:
    deps = _load_biu_dependencies()
    features = deps["extract_biu_v2"](
        normalized_pose_to_pixels(normalized_pose, width=width, height=height),
        kp_conf=None,
        bbox_sequence=normalized_bbox_to_pixels(normalized_bbox, width=width, height=height),
        image_size=(int(width), int(height)),
        include_optional=False,
    )
    names = tuple(
        name
        for name, value in features.items()
        if isinstance(value, (int, float, np.integer, np.floating))
    )
    values = np.nan_to_num(
        np.asarray([features[name] for name in names], dtype=np.float64),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    if values.shape != (63,):
        raise RuntimeError(f"Frozen BIU-v2 must be 63-D, got {values.shape}.")
    return values.astype(np.float32), names, str(features["direction_source"])


def augment_measurements(inputs: Mapping[str, np.ndarray], *, noise_seed: int) -> dict[str, np.ndarray]:
    output = {name: np.array(value, copy=True) for name, value in inputs.items()}
    rng = np.random.default_rng(int(noise_seed))
    for name in ("bbox", "ego_speed", "ego_acc", "yolop", "pose"):
        if name not in output:
            continue
        value = np.asarray(output[name], dtype=np.float32)
        value = value + rng.normal(0.0, GAUSSIAN_NOISE_STD, size=value.shape).astype(np.float32)
        if name == "bbox":
            value[:, :2] = np.clip(value[:, :2], 0.01, 0.99)
            value[:, 2:] = np.clip(value[:, 2:], 0.01, 1.0)
        elif name == "pose":
            value = np.clip(value, 0.0, 1.0)
        output[name] = value.astype(np.float32, copy=False)
    return output


def recompute_augmented_biu_v1(
    normalized_pose: np.ndarray,
    *,
    width: int,
    height: int,
    original_biu_v1: np.ndarray,
) -> np.ndarray:
    original = np.asarray(original_biu_v1, dtype=np.float32)
    if original.shape != (72,):
        raise ValueError("Original BIU-v1 must be 72-D.")
    recomputed, _ = extract_biu_v1_window(
        normalized_pose_to_pixels(normalized_pose, width=width, height=height), kp_conf=None
    )
    recomputed[-2:] = original[-2:]
    return recomputed


def precompute_augmented_row(
    measurements: Mapping[str, np.ndarray],
    *,
    width: int,
    height: int,
    noise_seed: int,
    base_biu_v1: np.ndarray,
) -> dict[str, np.ndarray]:
    augmented = augment_measurements(measurements, noise_seed=noise_seed)
    biu_v2, _, _ = extract_biu_v2_window(
        augmented["pose"], augmented["bbox"], width=width, height=height
    )
    biu_v1 = recompute_augmented_biu_v1(
        augmented["pose"], width=width, height=height, original_biu_v1=base_biu_v1
    )
    return {**augmented, "biu_v2": biu_v2, "biu_v1": biu_v1}


class StreamAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STREAM_CONTEXT_DIM, ATTENTION_DIM),
            nn.Tanh(),
            nn.Linear(ATTENTION_DIM, 1),
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        scores = self.net(sequence).squeeze(2)
        weights = torch.softmax(scores, dim=1)
        return torch.sum(sequence * weights.unsqueeze(2), dim=1)


class MslstmPracticalModel(nn.Module):
    def __init__(self, variant: str) -> None:
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"Unknown Exp050 variant: {variant}")
        self.variant = variant
        self.streams = VARIANT_STREAMS[variant]
        self.lstms = nn.ModuleDict()
        self.attentions = nn.ModuleDict()
        for name in self.streams:
            if name.startswith("biu_"):
                continue
            self.lstms[name] = nn.LSTM(
                STREAM_DIMS[name],
                LSTM_HIDDEN_SIZE,
                NUM_LSTM_LAYERS,
                batch_first=True,
                dropout=DROPOUT_RATE,
                bidirectional=True,
            )
            self.attentions[name] = StreamAttention()
        if variant in {"E", "F"}:
            input_dim = STREAM_DIMS[self.streams[-1]]
            self.biu_projection = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Dropout(DROPOUT_RATE),
                nn.Linear(128, STREAM_CONTEXT_DIM),
            )
        else:
            self.biu_projection = None
        self.fusion_weights = nn.Parameter(torch.ones(len(self.streams)))
        self.dropout = nn.Dropout(DROPOUT_RATE)
        self.fc1 = nn.Linear(STREAM_CONTEXT_DIM, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 2)

    def forward(self, inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        contexts: list[torch.Tensor] = []
        for name in self.streams:
            if name.startswith("biu_"):
                if self.biu_projection is None:
                    raise RuntimeError("BIU projection is unavailable.")
                contexts.append(self.biu_projection(inputs[name]))
            else:
                sequence, _ = self.lstms[name](inputs[name])
                contexts.append(self.attentions[name](sequence))
        weights = torch.softmax(self.fusion_weights, dim=0).view(1, -1, 1)
        fused = torch.sum(torch.stack(contexts, dim=1) * weights, dim=1)
        hidden = self.dropout(fused)
        hidden = self.relu(self.fc1(hidden))
        hidden = self.dropout(hidden)
        return self.fc2(hidden)


def count_parameters(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def compute_metrics(y_true: Sequence[int], probabilities: Sequence[float]) -> dict[str, float | int]:
    labels = np.asarray(y_true, dtype=np.int64)
    probs = np.asarray(probabilities, dtype=np.float64)
    predictions = (probs >= FIXED_THRESHOLD).astype(np.int64)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probs)),
        "pr_auc": float(average_precision_score(labels, probs)),
        "tn": int(((labels == 0) & (predictions == 0)).sum()),
        "fp": int(((labels == 0) & (predictions == 1)).sum()),
        "fn": int(((labels == 1) & (predictions == 0)).sum()),
        "tp": int(((labels == 1) & (predictions == 1)).sum()),
    }
