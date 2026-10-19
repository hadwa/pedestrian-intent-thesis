"""Frozen BIU-v1 interfaces for the approved Exp048 Variant F amendment."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset

from src.features.biu_features import (
    AGGREGATIONS,
    BIU_BASE_FEATURES,
    COCO17_NAMES,
    LOW_CONFIDENCE_THRESHOLD,
    REQUIRED_KEYPOINTS,
    aggregate_biu_sequence,
    compute_frame_biu_features,
    frame_has_invalid_required_keypoints,
    frame_has_low_confidence,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ROOT = REPO_ROOT / "experiments" / "048_mslstm_practical_behavior_replacement"
ORIGINAL_CORE_PATH = EXPERIMENT_ROOT / "scripts" / "exp048_core.py"

BIU_V1_FEATURE_NAMES = tuple(
    f"{feature}_{aggregation}"
    for feature in BIU_BASE_FEATURES
    for aggregation in AGGREGATIONS
) + ("biu_valid_frame_count", "biu_low_confidence_frame_count")
BIU_V1_DIM = len(BIU_V1_FEATURE_NAMES)
F_STREAMS = ("bbox", "ego_speed", "ego_acc", "yolop", "biu_v1")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_original_core():
    module_name = "exp048_original_core_for_biu_v1_amendment"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, ORIGINAL_CORE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import immutable Exp048 core: {ORIGINAL_CORE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def extract_biu_v1_window(
    coords_xy: np.ndarray,
    kp_conf: np.ndarray | None,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Extract the exact historical 70 summaries and two quality counts."""
    coords = np.asarray(coords_xy, dtype=np.float64)
    if coords.ndim != 3 or coords.shape[1:] != (17, 2) or len(coords) == 0:
        raise ValueError(f"Expected a non-empty (T,17,2) pose window, got {coords.shape}.")
    confidence = None if kp_conf is None else np.asarray(kp_conf, dtype=np.float64)
    if confidence is not None and confidence.shape != (len(coords), 17):
        raise ValueError("Pose confidence must align exactly with the pose window.")

    frame_features = [compute_frame_biu_features(frame) for frame in coords]
    aggregated = aggregate_biu_sequence(frame_features)
    valid_count = sum(not frame_has_invalid_required_keypoints(frame) for frame in coords)
    low_confidence_count = (
        0 if confidence is None else sum(frame_has_low_confidence(frame) for frame in confidence)
    )
    values = [aggregated[name] for name in BIU_V1_FEATURE_NAMES[:-2]]
    values.extend((valid_count, low_confidence_count))
    return (
        np.nan_to_num(
            np.asarray(values, dtype=np.float64),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).astype(np.float32),
        BIU_V1_FEATURE_NAMES,
    )


def recompute_augmented_biu_v1(
    normalized_pose: np.ndarray,
    *,
    width: int,
    height: int,
    original_biu_v1: np.ndarray,
) -> np.ndarray:
    """Recompute geometric cues after augmentation and retain source quality counts."""
    pose = np.asarray(normalized_pose, dtype=np.float64)
    original = np.asarray(original_biu_v1, dtype=np.float32)
    if pose.shape != (30, 34):
        raise ValueError(f"Augmented BIU-v1 requires normalized pose (30,34), got {pose.shape}.")
    if original.shape != (BIU_V1_DIM,):
        raise ValueError(f"Original BIU-v1 must be ({BIU_V1_DIM},), got {original.shape}.")
    coords = pose.reshape(30, 17, 2).copy()
    coords[:, :, 0] *= float(width)
    coords[:, :, 1] *= float(height)
    recomputed, _ = extract_biu_v1_window(coords, kp_conf=None)
    recomputed[-2:] = original[-2:]
    return recomputed


class MslstmPracticalBiuV1Model(nn.Module):
    """Practical MSLSTM core with exact 72-D BIU-v1 replacing behavior labels."""

    def __init__(self) -> None:
        super().__init__()
        original = load_original_core()
        self.streams = F_STREAMS
        self.lstms = nn.ModuleDict()
        self.attentions = nn.ModuleDict()
        for name in self.streams:
            if name == "biu_v1":
                continue
            self.lstms[name] = nn.LSTM(
                input_size=original.STREAM_DIMS[name],
                hidden_size=original.LSTM_HIDDEN_SIZE,
                num_layers=original.NUM_LSTM_LAYERS,
                batch_first=True,
                dropout=original.DROPOUT_RATE,
                bidirectional=True,
            )
            self.attentions[name] = original.StreamAttention()
        self.biu_projection = nn.Sequential(
            nn.Linear(BIU_V1_DIM, 128),
            nn.ReLU(),
            nn.Dropout(original.DROPOUT_RATE),
            nn.Linear(128, original.STREAM_CONTEXT_DIM),
        )
        self.fusion_weights = nn.Parameter(torch.ones(len(self.streams)))
        self.dropout = nn.Dropout(original.DROPOUT_RATE)
        self.fc1 = nn.Linear(original.STREAM_CONTEXT_DIM, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 2)

    def forward(self, inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        contexts = []
        for name in self.streams:
            if name == "biu_v1":
                contexts.append(self.biu_projection(inputs[name]))
                continue
            sequence, _ = self.lstms[name](inputs[name])
            context, _ = self.attentions[name](sequence)
            contexts.append(context)
        stacked = torch.stack(contexts, dim=1)
        weights = torch.softmax(self.fusion_weights, dim=0).view(1, -1, 1)
        fused = torch.sum(stacked * weights, dim=1)
        hidden = self.dropout(fused)
        hidden = self.relu(self.fc1(hidden))
        hidden = self.dropout(hidden)
        return self.fc2(hidden)


def build_f_model(seed: int) -> MslstmPracticalBiuV1Model:
    original = load_original_core()
    original.seed_everything(int(seed))
    return MslstmPracticalBiuV1Model()


class BiuV1AmendmentDataset(Dataset):
    """Aligned Exp048 dataset exposing only the approved Variant F streams."""

    _REQUIRED_ARRAYS = frozenset(
        {"bbox", "ego_speed", "ego_acc", "yolop", "exp012_pose", "biu_v1"}
    )

    def __init__(
        self,
        rows: pd.DataFrame,
        arrays: Mapping[str, np.ndarray],
        *,
        sampling_manifest: pd.DataFrame | None = None,
        training_phase: bool = True,
    ) -> None:
        self.rows = rows.reset_index(drop=True).copy()
        if self.rows.empty:
            raise ValueError("Dataset cannot be empty.")
        if training_phase and self.rows["split"].isin({"test", "set03"}).any():
            raise ValueError("Training dataset must not contain test/set03 rows.")
        if self.rows["sequence_id"].duplicated().any():
            raise ValueError("Dataset row IDs must be unique.")
        missing = sorted(self._REQUIRED_ARRAYS.difference(arrays))
        if missing:
            raise ValueError(f"Dataset arrays are missing: {missing}")
        self.arrays = arrays
        self.sampling = (
            None if sampling_manifest is None else sampling_manifest.reset_index(drop=True).copy()
        )
        self._by_cache_index = {
            int(row.cache_index): row for row in self.rows.itertuples(index=False)
        }

    def __len__(self) -> int:
        return len(self.rows) if self.sampling is None else len(self.sampling)

    def __getitem__(self, index: int):
        if self.sampling is None:
            row = self.rows.iloc[int(index)]
            cache_index = int(row["cache_index"])
            augmented = False
            noise_seed = -1
            frame_width = int(row["frame_width"])
            frame_height = int(row["frame_height"])
            sequence_id = str(row["sequence_id"])
            split = str(row["split"])
            label_value = int(row["label"])
        else:
            sampled = self.sampling.iloc[int(index)]
            cache_index = int(sampled["source_cache_index"])
            source_row = self._by_cache_index[cache_index]
            augmented = bool(sampled["augmented"])
            noise_seed = int(sampled["noise_seed"])
            frame_width = int(source_row.frame_width)
            frame_height = int(source_row.frame_height)
            sequence_id = str(source_row.sequence_id)
            split = str(source_row.split)
            label_value = int(source_row.label)

        source = {
            name: np.asarray(self.arrays[name][cache_index], dtype=np.float32)
            for name in self._REQUIRED_ARRAYS
        }
        if augmented:
            original = load_original_core()
            source = original.augment_measurements(source, noise_seed=noise_seed)
            source["biu_v1"] = recompute_augmented_biu_v1(
                source["exp012_pose"],
                width=frame_width,
                height=frame_height,
                original_biu_v1=np.asarray(self.arrays["biu_v1"][cache_index], dtype=np.float32),
            )

        features = {
            name: torch.as_tensor(np.array(source[name], copy=True), dtype=torch.float32)
            for name in F_STREAMS
        }
        label = torch.tensor(label_value, dtype=torch.long)
        metadata = {
            "sequence_id": sequence_id,
            "split": split,
            "label": label_value,
            "cache_index": cache_index,
            "augmented": augmented,
        }
        return features, label, metadata


if BIU_V1_DIM != 72:
    raise RuntimeError(f"Frozen BIU-v1 schema must be 72-D, got {BIU_V1_DIM}.")
