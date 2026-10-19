"""Starter student models."""

from __future__ import annotations

import torch
from torch import nn


class MLPStudent(nn.Module):
    """Simple MLP for tabular or flattened sequence features."""

    def __init__(self, input_dim: int, hidden_dims: list[int] | None = None, dropout: float = 0.2) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [128, 64]
        layers: list[nn.Module] = []
        prev = input_dim
        for hidden in hidden_dims:
            layers.extend([nn.Linear(prev, hidden), nn.ReLU(), nn.Dropout(dropout)])
            prev = hidden
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class LSTMStudent(nn.Module):
    """Small LSTM for sequence features shaped [batch, time, features]."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 1, dropout: float = 0.0) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(x)
        return self.classifier(hidden[-1]).squeeze(-1)
