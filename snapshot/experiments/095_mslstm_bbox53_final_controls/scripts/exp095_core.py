"""Frozen model and representation utilities for Exp095."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_ROOT = ROOT.parent
EXP053_CORE_PATH = EXPERIMENTS_ROOT / "053_mslstm_bbox53_replacement/scripts/exp053_core.py"
EXP050_METADATA_PATH = (
    EXPERIMENTS_ROOT
    / "050_mslstm_practical_behavior_replacement_pie_16f_tte30_60/cache/trainval_base/metadata.json"
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_base = _load_module("_exp053_core_for_exp095", EXP053_CORE_PATH)
for _name in dir(_base):
    if not _name.startswith("_") and _name not in {
        "MslstmBbox53Model",
        "MslstmPracticalModel",
        "STREAM_DIMS",
        "VARIANTS",
        "VARIANT_STREAMS",
        "WINDOW_VECTOR_STREAMS",
    }:
        globals()[_name] = getattr(_base, _name)

VARIANTS = ("J", "K", "L")
VARIANT_STREAMS = {
    "J": ("bbox53", "ego_speed", "ego_acc", "yolop", "pose"),
    "K": ("bbox53", "ped_action", "ped_look", "ego_speed", "ego_acc", "yolop"),
    "L": ("bbox53", "ego_speed", "ego_acc", "yolop", "biu_combined"),
}
WINDOW_VECTOR_STREAMS = {"bbox53", "biu_v1", "biu_v2", "biu_combined"}
STREAM_DIMS = dict(_base.STREAM_DIMS)
STREAM_DIMS["biu_combined"] = 135


def _combined_feature_names() -> tuple[str, ...]:
    payload = json.loads(EXP050_METADATA_PATH.read_text(encoding="utf-8"))
    v1 = tuple(str(name) for name in payload["biu_v1_feature_names"])
    v2 = tuple(str(name) for name in payload["biu_v2_feature_names"])
    if len(v1) != 72 or len(v2) != 63:
        raise RuntimeError("Frozen source metadata must declare 72 BIU-v1 and 63 BIU-v2 fields.")
    return tuple(f"v1__{name}" for name in v1) + tuple(f"v2__{name}" for name in v2)


BIU_COMBINED_FEATURE_NAMES = _combined_feature_names()


def combined_biu(biu_v1: np.ndarray, biu_v2: np.ndarray) -> np.ndarray:
    """Return the lossless, source-namespaced 72+63 BIU representation."""
    v1 = np.asarray(biu_v1, dtype=np.float32)
    v2 = np.asarray(biu_v2, dtype=np.float32)
    if v1.shape[:-1] != v2.shape[:-1] or v1.shape[-1] != 72 or v2.shape[-1] != 63:
        raise ValueError(f"Expected aligned BIU shapes (...,72) and (...,63), got {v1.shape} and {v2.shape}.")
    output = np.concatenate((v1, v2), axis=-1).astype(np.float32, copy=False)
    if output.shape[-1] != 135 or not np.isfinite(output).all():
        raise RuntimeError("Combined BIU must be finite and 135-D.")
    return output


def _projection(input_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, 128),
        nn.ReLU(),
        nn.Dropout(DROPOUT_RATE),
        nn.Linear(128, STREAM_CONTEXT_DIM),
    )


class MslstmBbox53ControlModel(nn.Module):
    """Exp053 normalized fusion extended only by the three frozen controls."""

    def __init__(self, variant: str) -> None:
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"Unknown Exp095 variant: {variant}")
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


MslstmPracticalModel = MslstmBbox53ControlModel
