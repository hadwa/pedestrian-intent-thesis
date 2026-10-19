"""Direct signed residual branches for frozen Exp089 PedCMT parents."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import Tensor, nn


REPO_ROOT = Path(__file__).resolve().parents[3]
PARENT_MODEL_SCRIPTS = REPO_ROOT / "experiments/084_pedcmt_biu_transfer/scripts"
if str(PARENT_MODEL_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(PARENT_MODEL_SCRIPTS))

from pedcmt_model import AttentionBlock, FeedForward, StreamEmbedding  # noqa: E402


MODEL_DIM = 128
HIDDEN_DIM = 256
HEADS = 8
LAYERS = 4
CORRECTION_LIMIT = 2.0


class RawPoseEncoder(nn.Module):
    """Preserve the Exp089 PedCMT raw-pose temporal branch style."""

    def __init__(self) -> None:
        super().__init__()
        self.token = nn.Parameter(torch.ones(1, 1, MODEL_DIM))
        self.embedding = StreamEmbedding(34, MODEL_DIM)
        self.attention = nn.ModuleList(
            [AttentionBlock(MODEL_DIM, HEADS) for _ in range(LAYERS)]
        )
        self.feed_forward = nn.ModuleList(
            [FeedForward(MODEL_DIM, HIDDEN_DIM) for _ in range(LAYERS)]
        )

    def forward(self, pose: Tensor) -> Tensor:
        if pose.ndim != 3 or pose.shape[1:] != (16, 34):
            raise ValueError(f"Expected raw pose [B,16,34], got {tuple(pose.shape)}")
        state = self.embedding(pose, self.token)
        for attention, feed_forward in zip(self.attention, self.feed_forward):
            state = feed_forward(attention(state))
        return state[:, 0]


class BiuUnionEncoder(nn.Module):
    """Compact fixed projector for the standardized 135-D BIU union."""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(135, MODEL_DIM),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(MODEL_DIM, MODEL_DIM),
            nn.LayerNorm(MODEL_DIM),
        )

    def forward(self, biu: Tensor) -> Tensor:
        if biu.ndim != 2 or biu.shape[1] != 135:
            raise ValueError(f"Expected BIU union [B,135], got {tuple(biu.shape)}")
        return self.network(biu)


class DirectFusionModel(nn.Module):
    """Apply a bounded signed correction to an immutable parent logit."""

    def __init__(self, variant: str) -> None:
        super().__init__()
        if variant not in {"B", "C"}:
            raise ValueError(f"Variant must be B or C, got {variant!r}")
        self.variant = variant
        self.anchor_projector = nn.Sequential(
            nn.Linear(256, MODEL_DIM),
            nn.GELU(),
            nn.LayerNorm(MODEL_DIM),
        )
        self.extra_encoder: nn.Module = RawPoseEncoder() if variant == "B" else BiuUnionEncoder()
        self.correction_head = nn.Sequential(
            nn.Linear(4 * MODEL_DIM, 64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )
        nn.init.zeros_(self.correction_head[-1].weight)
        nn.init.zeros_(self.correction_head[-1].bias)

    def forward(
        self,
        parent_logits: Tensor,
        parent_anchor: Tensor,
        extra: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if parent_logits.ndim != 2 or parent_logits.shape[1] != 1:
            raise ValueError(f"Expected parent logits [B,1], got {tuple(parent_logits.shape)}")
        if parent_anchor.ndim != 2 or parent_anchor.shape[1] != 256:
            raise ValueError(f"Expected parent anchor [B,256], got {tuple(parent_anchor.shape)}")
        if len(parent_logits) != len(parent_anchor) or len(extra) != len(parent_logits):
            raise ValueError("Parent and extra batches must align")
        anchor = self.anchor_projector(parent_anchor)
        encoded_extra = self.extra_encoder(extra)
        interaction = torch.cat(
            [anchor, encoded_extra, anchor * encoded_extra, torch.abs(anchor - encoded_extra)],
            dim=-1,
        )
        correction = CORRECTION_LIMIT * torch.tanh(self.correction_head(interaction))
        return parent_logits + correction, correction


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
