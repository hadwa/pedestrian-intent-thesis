"""PedCMT architecture reconstruction with one controlled context-token slot."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


VARIANTS = ("A", "B", "C", "D", "E")


class PositionalEncoding(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1, max_length: int = 5000):
        super().__init__()
        position = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
        encoding = torch.zeros(max_length, dim, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: Tensor) -> Tensor:
        return self.dropout(values + self.encoding[:, : values.shape[1]])


class StreamEmbedding(nn.Module):
    def __init__(self, input_dim: int, model_dim: int):
        super().__init__()
        self.embedding = nn.Linear(input_dim, model_dim)
        self.position = PositionalEncoding(model_dim)

    def forward(self, values: Tensor, cls_token: Tensor) -> Tensor:
        batch = values.shape[0]
        embedded = self.embedding(values)
        token = cls_token.expand(batch, -1, -1)
        return self.position(torch.cat([token, embedded], dim=1))


class AttentionBlock(nn.Module):
    def __init__(self, model_dim: int, heads: int, dropout: float = 0.3):
        super().__init__()
        self.attention = nn.MultiheadAttention(model_dim, heads, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(model_dim, eps=1e-5)

    def forward(self, values: Tensor) -> Tensor:
        attended, _ = self.attention(values, values, values, need_weights=False)
        return self.norm(values + self.dropout(attended))


class FeedForward(nn.Module):
    def __init__(self, model_dim: int, hidden_dim: int, dropout: float = 0.3):
        super().__init__()
        self.norm = nn.LayerNorm(model_dim, eps=1e-5)
        self.linear1 = nn.Linear(model_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, model_dim)
        self.activation = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, values: Tensor) -> Tensor:
        transformed = self.linear2(self.dropout1(self.activation(self.linear1(values))))
        return self.norm(values + self.dropout2(transformed))


class BottleneckFusion(nn.Module):
    def __init__(self, model_dim: int, bottleneck_dim: int, layers: int):
        super().__init__()
        self.model_dim = model_dim
        self.bottleneck_dim = bottleneck_dim
        self.bbox = nn.ModuleList([nn.Linear(model_dim, model_dim + bottleneck_dim)])
        self.ego = nn.ModuleList([nn.Linear(model_dim, model_dim + bottleneck_dim)])
        for _ in range(layers - 1):
            self.bbox.append(nn.Linear(model_dim + bottleneck_dim, model_dim + bottleneck_dim))
            self.ego.append(nn.Linear(model_dim + bottleneck_dim, model_dim + bottleneck_dim))
        self.dropout = nn.Dropout(0.5)
        self.activation = nn.ReLU()

    def _project(self, layer: nn.Linear, values: Tensor) -> tuple[Tensor, Tensor]:
        projected = self.dropout(self.activation(layer(values)))
        return projected[..., : self.model_dim], projected[..., -self.bottleneck_dim :]

    def forward(self, bbox: Tensor, ego: Tensor) -> Tensor:
        bbox, bbox_bottleneck = self._project(self.bbox[0], bbox)
        ego, ego_bottleneck = self._project(self.ego[0], ego)
        shared = bbox_bottleneck + ego_bottleneck
        for index in range(1, len(self.bbox)):
            bbox, bbox_bottleneck = self._project(self.bbox[index], torch.cat([bbox, shared], dim=-1))
            ego, ego_bottleneck = self._project(self.ego[index], torch.cat([ego, shared], dim=-1))
            shared = bbox_bottleneck + ego_bottleneck
        return shared


class TimeAttention(nn.Module):
    """Preserve the released PedCMT temporal reducer exactly."""

    def __init__(self, dims: int):
        super().__init__()
        self.linear1 = nn.Linear(dims, dims, bias=False)
        self.linear2 = nn.Linear(dims, 1, bias=False)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, values: Tensor) -> Tensor:
        score = self.linear2(torch.tanh(self.linear1(values)))
        weight = torch.softmax(score, dim=-1)
        weighted = weight * values
        return self.pool(weighted.transpose(1, 2)).squeeze(-1)


class VectorProjector(nn.Module):
    def __init__(self, input_dim: int, model_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, model_dim),
            nn.LayerNorm(model_dim),
        )

    def forward(self, values: Tensor) -> Tensor:
        return self.network(values).unsqueeze(1)


class PedCMTTransfer(nn.Module):
    def __init__(
        self,
        *,
        variant: str,
        ego_dim: int = 1,
        extra_dim: int | None = None,
        model_dim: int = 128,
        hidden_dim: int = 256,
        heads: int = 8,
        layers: int = 4,
        bottleneck_dim: int = 3,
        bottleneck_layers: int = 7,
    ):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant {variant!r}")
        expected_extra = {"A": None, "B": 34, "C": 72, "D": 63, "E": 135}[variant]
        if extra_dim is None:
            extra_dim = expected_extra
        if extra_dim != expected_extra:
            raise ValueError(f"Variant {variant} requires extra_dim={expected_extra}, got {extra_dim}")

        self.variant = variant
        self.model_dim = model_dim
        self.log_var_cls = nn.Parameter(torch.zeros(1))
        self.log_var_reg = nn.Parameter(torch.zeros(1))
        self.global_token = nn.Parameter(torch.ones(1, 1, model_dim))
        self.bbox_token = nn.Parameter(torch.ones(1, 1, model_dim))
        self.ego_token = nn.Parameter(torch.ones(1, 1, model_dim))
        self.bbox_embedding = StreamEmbedding(4, model_dim)
        self.ego_embedding = StreamEmbedding(ego_dim, model_dim)

        self.bbox_attention = nn.ModuleList([AttentionBlock(model_dim, heads) for _ in range(layers)])
        self.bbox_ffn = nn.ModuleList([FeedForward(model_dim, hidden_dim) for _ in range(layers)])
        self.ego_attention = nn.ModuleList([AttentionBlock(model_dim, heads) for _ in range(layers)])
        self.ego_ffn = nn.ModuleList([FeedForward(model_dim, hidden_dim) for _ in range(layers)])
        self.cross_attention = nn.ModuleList([AttentionBlock(model_dim, heads) for _ in range(layers)])
        self.cross_ffn = nn.ModuleList([FeedForward(model_dim, hidden_dim) for _ in range(layers)])

        self.pose_token: nn.Parameter | None = None
        self.pose_embedding: StreamEmbedding | None = None
        self.pose_attention: nn.ModuleList | None = None
        self.pose_ffn: nn.ModuleList | None = None
        self.vector_projector: VectorProjector | None = None
        if variant == "B":
            self.pose_token = nn.Parameter(torch.ones(1, 1, model_dim))
            self.pose_embedding = StreamEmbedding(34, model_dim)
            self.pose_attention = nn.ModuleList([AttentionBlock(model_dim, heads) for _ in range(layers)])
            self.pose_ffn = nn.ModuleList([FeedForward(model_dim, hidden_dim) for _ in range(layers)])
        elif variant in {"C", "D", "E"}:
            assert extra_dim is not None
            self.vector_projector = VectorProjector(extra_dim, model_dim)

        # `legacy_dense` exists in the released model state even though forward does not use it.
        self.legacy_dense = nn.Linear(2 * model_dim, 4)
        self.endpoint = nn.Linear(2 * model_dim, 4)
        self.bottlenecks = BottleneckFusion(model_dim, bottleneck_dim, bottleneck_layers)
        self.time_attention = TimeAttention(bottleneck_dim)
        self.intent = nn.Linear(bottleneck_dim, 1)

    def forward(
        self,
        bbox: Tensor,
        ego: Tensor,
        *,
        pose: Tensor | None = None,
        biu: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if bbox.ndim != 3 or bbox.shape[1:] != (16, 4):
            raise ValueError(f"Expected bbox [B,16,4], got {tuple(bbox.shape)}")
        if ego.ndim != 3 or ego.shape[:2] != bbox.shape[:2]:
            raise ValueError("Ego stream must align with BBox batch and time")
        batch = bbox.shape[0]
        bbox_state = self.bbox_embedding(bbox, self.bbox_token)
        ego_state = self.ego_embedding(ego, self.ego_token)
        global_state = self.global_token.expand(batch, -1, -1)

        pose_state: Tensor | None = None
        extra_state: Tensor | None = None
        if self.variant == "B":
            if pose is None or biu is not None:
                raise ValueError("Variant B requires pose and forbids BIU")
            assert self.pose_embedding is not None and self.pose_token is not None
            pose_state = self.pose_embedding(pose, self.pose_token)
        elif self.variant in {"C", "D", "E"}:
            if biu is None or pose is not None:
                raise ValueError(f"Variant {self.variant} requires BIU and forbids pose")
            assert self.vector_projector is not None
            extra_state = self.vector_projector(biu)
        elif pose is not None or biu is not None:
            raise ValueError("Variant A accepts no extra input")

        for index in range(len(self.bbox_attention)):
            bbox_state = self.bbox_attention[index](bbox_state)
            ego_state = self.ego_attention[index](ego_state)
            cross_parts = [global_state, bbox_state[:, :1], ego_state[:, :1]]
            if pose_state is not None:
                assert self.pose_attention is not None
                pose_state = self.pose_attention[index](pose_state)
                cross_parts.append(pose_state[:, :1])
            elif extra_state is not None:
                cross_parts.append(extra_state)

            cross_state = self.cross_ffn[index](self.cross_attention[index](torch.cat(cross_parts, dim=1)))
            global_state = cross_state[:, :1]
            if extra_state is not None:
                extra_state = cross_state[:, 3:4]

            bbox_state = torch.cat([global_state, bbox_state[:, 1:]], dim=1)
            ego_state = torch.cat([global_state, ego_state[:, 1:]], dim=1)
            bbox_state = self.bbox_ffn[index](bbox_state)
            ego_state = self.ego_ffn[index](ego_state)
            if pose_state is not None:
                assert self.pose_ffn is not None
                pose_state = torch.cat([global_state, pose_state[:, 1:]], dim=1)
                pose_state = self.pose_ffn[index](pose_state)

        endpoint = self.endpoint(torch.cat([bbox_state[:, 0], ego_state[:, 0]], dim=-1))
        bottleneck = torch.relu(self.time_attention(self.bottlenecks(bbox_state, ego_state)))
        intent_logit = self.intent(bottleneck)
        return intent_logit, endpoint, self.log_var_cls, self.log_var_reg


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
