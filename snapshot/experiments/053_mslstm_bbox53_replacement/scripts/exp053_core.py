"""Frozen model and BBox53 utilities for Exp053."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
EXP051_CORE_PATH = ROOT.parents[0] / "051_mslstm_normalized_fusion_pie_16f_tte30_60/scripts/exp051_core.py"
ARCHIVE_BBOX_CORE_PATH = Path("/home/abdelrahman/Hadwa/BIU-PID-Distillation/src/features/bbox_features.py")


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_base = _load_module("_exp051_core_for_exp053", EXP051_CORE_PATH)
_bbox = _load_module("_archive_bbox_core_for_exp053", ARCHIVE_BBOX_CORE_PATH)
for _name in dir(_base):
    if not _name.startswith("_") and _name not in {
        "MslstmPracticalModel",
        "STREAM_DIMS",
        "VARIANTS",
        "VARIANT_STREAMS",
    }:
        globals()[_name] = getattr(_base, _name)

VARIANTS = ("G", "H", "I")
VARIANT_STREAMS = {
    "G": ("bbox53", "ego_speed", "ego_acc", "yolop"),
    "H": ("bbox53", "ego_speed", "ego_acc", "yolop", "biu_v1"),
    "I": ("bbox53", "ego_speed", "ego_acc", "yolop", "biu_v2"),
}
RAW_CONTROL_VARIANTS = {"G": "C", "H": "F", "I": "E"}
WINDOW_VECTOR_STREAMS = {"bbox53", "biu_v1", "biu_v2"}
STREAM_DIMS = dict(_base.STREAM_DIMS)
STREAM_DIMS["bbox53"] = 53

BBOX53_FEATURES = tuple(
    [
        f"bbox_{feature}_{aggregation}"
        for feature in _bbox.BBOX_BASE_FEATURES
        for aggregation in _bbox.AGGREGATIONS
    ]
    + [
        "center_x_velocity_mean",
        "center_y_velocity_mean",
        "center_x_velocity_std",
        "center_y_velocity_std",
        "center_x_displacement",
        "center_y_displacement",
        "bbox_area_change",
        "bbox_width_change",
        "bbox_height_change",
        "bbox_scale_change_ratio",
        "bbox_valid_frame_count",
    ]
)
if len(BBOX53_FEATURES) != 53:
    raise RuntimeError("Frozen BBox53 schema must contain exactly 53 features.")


def bbox53_from_normalized_bbox(
    normalized_bbox: np.ndarray,
    *,
    width: int,
    height: int,
) -> np.ndarray:
    """Apply the archived Exp012 BBox53 equations to normalized cx/cy/w/h."""
    values = np.asarray(normalized_bbox, dtype=np.float64)
    if values.shape != (SEQ_LEN, 4):
        raise ValueError(f"Expected normalized BBox shape {(SEQ_LEN, 4)}, got {values.shape}")
    cx, cy, box_width, box_height = values.T
    pixels = np.column_stack(
        (
            (cx - box_width / 2.0) * width,
            (cy - box_height / 2.0) * height,
            (cx + box_width / 2.0) * width,
            (cy + box_height / 2.0) * height,
        )
    )
    features = _bbox.compute_bbox_motion_features(pixels)
    output = np.asarray([features[name] for name in BBOX53_FEATURES], dtype=np.float64)
    if output.shape != (53,):
        raise RuntimeError(f"BBox53 extractor returned {output.shape}")
    return output


def fit_bbox53_scaler(train_values: np.ndarray) -> dict[str, np.ndarray]:
    """Fit median imputation and z-score parameters on original training rows only."""
    values = np.asarray(train_values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("BBox53 scaler expects a 2-D training matrix.")
    median = np.nanmedian(values, axis=0)
    filled = np.where(np.isfinite(values), values, median)
    mean = filled.mean(axis=0)
    std = filled.std(axis=0)
    std = np.where(std > 1e-12, std, 1.0)
    return {"median": median, "mean": mean, "std": std}


def transform_bbox53(values: np.ndarray, scaler: Mapping[str, np.ndarray]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    median = np.asarray(scaler["median"], dtype=np.float64)
    mean = np.asarray(scaler["mean"], dtype=np.float64)
    std = np.asarray(scaler["std"], dtype=np.float64)
    filled = np.where(np.isfinite(array), array, median)
    return ((filled - mean) / std).astype(np.float32)


def _projection(input_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, 128),
        nn.ReLU(),
        nn.Dropout(DROPOUT_RATE),
        nn.Linear(128, STREAM_CONTEXT_DIM),
    )


class MslstmBbox53Model(nn.Module):
    """Exp051 normalized fusion with BBox53 replacing the raw BBox LSTM."""

    def __init__(self, variant: str) -> None:
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"Unknown Exp053 variant: {variant}")
        self.variant = variant
        self.streams = VARIANT_STREAMS[variant]
        self.lstms = nn.ModuleDict()
        self.attentions = nn.ModuleDict()
        self.vector_projections = nn.ModuleDict()
        for name in self.streams:
            if name in WINDOW_VECTOR_STREAMS:
                self.vector_projections[name] = _projection(STREAM_DIMS[name])
            else:
                self.lstms[name] = nn.LSTM(
                    STREAM_DIMS[name],
                    LSTM_HIDDEN_SIZE,
                    NUM_LSTM_LAYERS,
                    batch_first=True,
                    dropout=DROPOUT_RATE,
                    bidirectional=True,
                )
                self.attentions[name] = StreamAttention()
        self.context_normalization = nn.LayerNorm(STREAM_CONTEXT_DIM, elementwise_affine=False)
        self.fusion_weights = nn.Parameter(torch.ones(len(self.streams)))
        self.dropout = nn.Dropout(DROPOUT_RATE)
        self.fc1 = nn.Linear(STREAM_CONTEXT_DIM, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 2)

    def forward(self, inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        contexts = []
        for name in self.streams:
            if name in WINDOW_VECTOR_STREAMS:
                context = self.vector_projections[name](inputs[name])
            else:
                sequence, _ = self.lstms[name](inputs[name])
                context = self.attentions[name](sequence)
            contexts.append(self.context_normalization(context))
        weights = torch.softmax(self.fusion_weights, dim=0).view(1, -1, 1)
        fused = torch.sum(torch.stack(contexts, dim=1) * weights, dim=1)
        hidden = self.dropout(fused)
        hidden = self.relu(self.fc1(hidden))
        hidden = self.dropout(hidden)
        return self.fc2(hidden)


MslstmPracticalModel = MslstmBbox53Model
