"""Exp051 core: Exp050 with parameter-free context normalization before fusion."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Mapping, Sequence

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
EXP050_CORE_PATH = (
    ROOT.parents[0]
    / "050_mslstm_practical_behavior_replacement_pie_16f_tte30_60"
    / "scripts"
    / "exp050_core.py"
)


def _load_exp050_core():
    spec = importlib.util.spec_from_file_location("_exp050_core_for_exp051", EXP050_CORE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load Exp050 core: {EXP050_CORE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_base = _load_exp050_core()

# Re-export the frozen Exp050 protocol and feature functions. Only the model
# class below differs.
for _name in dir(_base):
    if not _name.startswith("_") and _name != "MslstmPracticalModel":
        globals()[_name] = getattr(_base, _name)


def normalize_and_fuse_contexts(
    contexts: Sequence[torch.Tensor], fusion_weights: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize each sample/stream context, then apply the original weights."""
    if not contexts:
        raise ValueError("At least one stream context is required.")
    normalization = nn.LayerNorm(STREAM_CONTEXT_DIM, elementwise_affine=False)
    normalized = torch.stack([normalization(context) for context in contexts], dim=1)
    weights = torch.softmax(fusion_weights, dim=0).view(1, -1, 1)
    if normalized.shape[1] != weights.shape[1]:
        raise ValueError("The number of contexts must equal the number of fusion weights.")
    return torch.sum(normalized * weights, dim=1), normalized


class MslstmPracticalModel(_base.MslstmPracticalModel):
    """The frozen Exp050 model with non-affine LayerNorm before stream fusion."""

    def __init__(self, variant: str) -> None:
        super().__init__(variant)
        self.context_normalization = nn.LayerNorm(
            STREAM_CONTEXT_DIM,
            elementwise_affine=False,
        )

    def forward(self, inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        contexts: list[torch.Tensor] = []
        for name in self.streams:
            if name.startswith("biu_"):
                if self.biu_projection is None:
                    raise RuntimeError("BIU projection is unavailable.")
                context = self.biu_projection(inputs[name])
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
