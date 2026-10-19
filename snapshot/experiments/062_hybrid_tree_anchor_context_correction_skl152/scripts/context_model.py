"""Fixed, bounded neural context correction for Exp062 tree-anchor scores."""

from __future__ import annotations

import copy
import os
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from .contracts import ExperimentConfig, compute_binary_metrics


LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 64
MAX_EPOCHS = 60
EARLY_STOPPING_PATIENCE = 10
GRADIENT_CLIP_NORM = 5.0
CORRECTION_PENALTY_WEIGHT = 0.01
ANCHOR_PROBABILITY_EPSILON = 1e-6
GATE_LOGIT_BOUND = 8.0


@dataclass
class CorrectionOutput:
    """Logit correction and diagnostics for one batch of frozen anchor scores."""

    final_logit: torch.Tensor
    gate: torch.Tensor
    bounded_delta: torch.Tensor
    applied_correction: torch.Tensor


@dataclass(frozen=True)
class TrainingResult:
    """Selected context-model state from a deterministic single-seed run."""

    model: "HybridContextCorrector"
    best_epoch: int
    best_score: float
    history: tuple[Mapping[str, float | int], ...]
    checkpoint_path: Path | None


class AdditiveAttention(nn.Module):
    """Learn a normalized temporal weighting for one encoded sequence."""

    def __init__(self, input_dim: int, attention_dim: int) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1),
        )

    def forward(self, sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if sequence.ndim != 3:
            raise ValueError("attention expects a (batch, frames, features) tensor")
        weights = torch.softmax(self.score(sequence).squeeze(-1), dim=1)
        context = torch.sum(sequence * weights.unsqueeze(-1), dim=1)
        return context, weights


class SequenceEncoder(nn.Module):
    """One-layer bidirectional LSTM followed by the frozen-size attention block."""

    def __init__(self, input_dim: int, hidden_dim: int, attention_dim: int) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.lstm = nn.LSTM(
            self.input_dim,
            hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.attention = AdditiveAttention(2 * hidden_dim, attention_dim)

    def forward(
        self, values: torch.Tensor, *, return_attention: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if values.ndim != 3 or values.shape[1:] != (16, self.input_dim):
            raise ValueError(
                f"sequence encoder expects (batch, 16, {self.input_dim}), got {tuple(values.shape)}"
            )
        encoded, _ = self.lstm(values)
        context, weights = self.attention(encoded)
        return (context, weights) if return_attention else context


class HybridContextCorrector(nn.Module):
    """Apply a gated, bounded logit residual to a non-differentiable tree anchor."""

    def __init__(self) -> None:
        super().__init__()
        self.ego_encoder = SequenceEncoder(input_dim=3, hidden_dim=8, attention_dim=8)
        self.yolop_encoder = SequenceEncoder(input_dim=20, hidden_dim=12, attention_dim=12)
        self.correction_head = nn.Sequential(
            nn.Linear(40, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1),
        )
        self.gate_head = nn.Sequential(
            nn.Linear(42, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

        # The first forward pass is exactly the frozen anchor prediction.
        nn.init.zeros_(self.correction_head[-1].weight)
        nn.init.zeros_(self.correction_head[-1].bias)
        nn.init.zeros_(self.gate_head[2].weight)
        nn.init.constant_(self.gate_head[2].bias, -2.0)

    def forward(
        self, ego: torch.Tensor, yolop: torch.Tensor, anchor_probability: torch.Tensor
    ) -> CorrectionOutput:
        ego_context = self.ego_encoder(ego)
        yolop_context = self.yolop_encoder(yolop)
        context = torch.cat((ego_context, yolop_context), dim=1)
        if anchor_probability.ndim != 1 or anchor_probability.shape[0] != context.shape[0]:
            raise ValueError("anchor_probability must have shape (batch,)")

        # Anchor values are an input feature, never a differentiable model branch.
        probability = anchor_probability.detach().to(
            device=context.device, dtype=context.dtype
        ).clamp(ANCHOR_PROBABILITY_EPSILON, 1.0 - ANCHOR_PROBABILITY_EPSILON)
        anchor_logit = torch.logit(probability)
        gate_logit = anchor_logit.clamp(-GATE_LOGIT_BOUND, GATE_LOGIT_BOUND)
        bounded_delta = 1.5 * torch.tanh(self.correction_head(context).squeeze(1))
        gate_input = torch.cat(
            (context, gate_logit[:, None], gate_logit.abs()[:, None]), dim=1
        )
        gate = self.gate_head(gate_input).squeeze(1)
        applied_correction = gate * bounded_delta
        return CorrectionOutput(
            final_logit=anchor_logit + applied_correction,
            gate=gate,
            bounded_delta=bounded_delta,
            applied_correction=applied_correction,
        )


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return resolved


def _as_input_tensor(
    values: np.ndarray | torch.Tensor, *, name: str, shape: tuple[int, int], device: torch.device
) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32, device=device).detach()
    if tensor.ndim != 3 or tuple(tensor.shape[1:]) != shape:
        raise ValueError(f"{name} must have shape (batch, {shape[0]}, {shape[1]})")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must contain only finite values")
    return tensor


def _as_probability_tensor(
    values: np.ndarray | torch.Tensor, *, batch_size: int, device: torch.device
) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32, device=device).detach()
    if tensor.ndim != 1 or tensor.shape[0] != batch_size:
        raise ValueError("anchor probabilities must have shape (batch,)")
    if not torch.isfinite(tensor).all():
        raise ValueError("anchor probabilities must contain only finite values")
    return tensor


def _as_label_tensor(
    values: np.ndarray | torch.Tensor, *, batch_size: int, device: torch.device
) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32, device=device).detach()
    if tensor.ndim != 1 or tensor.shape[0] != batch_size:
        raise ValueError("labels must have shape (batch,)")
    if not torch.isfinite(tensor).all() or not torch.all((tensor == 0.0) | (tensor == 1.0)):
        raise ValueError("labels must contain only binary 0/1 values")
    return tensor


def _prediction_tensors(
    model: HybridContextCorrector,
    ego: torch.Tensor,
    yolop: torch.Tensor,
    anchor_probability: torch.Tensor,
) -> dict[str, torch.Tensor]:
    output = model(ego, yolop, anchor_probability)
    return {
        "probability": torch.sigmoid(output.final_logit),
        "final_logit": output.final_logit,
        "gate": output.gate,
        "bounded_delta": output.bounded_delta,
        "applied_correction": output.applied_correction,
    }


def predict_context_corrector(
    model: HybridContextCorrector,
    ego: np.ndarray | torch.Tensor,
    yolop: np.ndarray | torch.Tensor,
    anchor_probability: np.ndarray | torch.Tensor,
    *,
    device: str | torch.device | None = None,
) -> dict[str, np.ndarray]:
    """Return corrected probabilities and diagnostics in the original input order."""
    resolved_device = _resolve_device(device)
    batch_size = int(ego.shape[0])
    ego_tensor = _as_input_tensor(ego, name="ego", shape=(16, 3), device=resolved_device)
    yolop_tensor = _as_input_tensor(yolop, name="yolop", shape=(16, 20), device=resolved_device)
    anchor_tensor = _as_probability_tensor(
        anchor_probability, batch_size=batch_size, device=resolved_device
    )
    was_training = model.training
    model.to(resolved_device)
    model.eval()
    with torch.no_grad():
        output = _prediction_tensors(model, ego_tensor, yolop_tensor, anchor_tensor)
    if was_training:
        model.train()
    return {name: value.detach().cpu().numpy().copy() for name, value in output.items()}


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary_path = Path(handle.name)
    try:
        torch.save(dict(payload), temporary_path)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def train_one_seed(
    train_ego: np.ndarray | torch.Tensor,
    train_yolop: np.ndarray | torch.Tensor,
    train_anchor_probability: np.ndarray | torch.Tensor,
    train_labels: np.ndarray | torch.Tensor,
    validation_ego: np.ndarray | torch.Tensor,
    validation_yolop: np.ndarray | torch.Tensor,
    validation_anchor_probability: np.ndarray | torch.Tensor,
    validation_labels: np.ndarray | torch.Tensor,
    *,
    seed: int,
    device: str | torch.device | None = None,
    checkpoint_path: Path | None = None,
    max_epochs: int = MAX_EPOCHS,
    patience: int = EARLY_STOPPING_PATIENCE,
) -> TrainingResult:
    """Fit one deterministic Exp062 correction model on OOF anchor probabilities."""
    config = ExperimentConfig()
    if seed not in config.seeds:
        raise ValueError("seed is outside the frozen Exp062 seed grid")
    if max_epochs != MAX_EPOCHS or patience != EARLY_STOPPING_PATIENCE:
        raise ValueError("Exp062 requires exactly 60 epochs and patience 10")
    resolved_device = _resolve_device(device)
    _seed_everything(seed)

    train_batch_size = int(train_ego.shape[0])
    validation_batch_size = int(validation_ego.shape[0])
    train_ego_tensor = _as_input_tensor(train_ego, name="train_ego", shape=(16, 3), device=resolved_device)
    train_yolop_tensor = _as_input_tensor(train_yolop, name="train_yolop", shape=(16, 20), device=resolved_device)
    train_anchor_tensor = _as_probability_tensor(
        train_anchor_probability, batch_size=train_batch_size, device=resolved_device
    )
    train_label_tensor = _as_label_tensor(
        train_labels, batch_size=train_batch_size, device=resolved_device
    )
    validation_ego_tensor = _as_input_tensor(
        validation_ego, name="validation_ego", shape=(16, 3), device=resolved_device
    )
    validation_yolop_tensor = _as_input_tensor(
        validation_yolop, name="validation_yolop", shape=(16, 20), device=resolved_device
    )
    validation_anchor_tensor = _as_probability_tensor(
        validation_anchor_probability, batch_size=validation_batch_size, device=resolved_device
    )
    validation_label_tensor = _as_label_tensor(
        validation_labels, batch_size=validation_batch_size, device=resolved_device
    )
    if train_yolop_tensor.shape[0] != train_batch_size or validation_yolop_tensor.shape[0] != validation_batch_size:
        raise ValueError("ego and YOLOP batches must have matching row counts")
    if torch.unique(validation_label_tensor).numel() != 2:
        raise ValueError("validation labels must contain both binary classes")

    model = HybridContextCorrector().to(resolved_device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    criterion = nn.BCEWithLogitsLoss()
    rng = np.random.RandomState(seed)
    best_score = float("-inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        order = rng.permutation(train_batch_size)
        total_loss = 0.0
        for start in range(0, train_batch_size, BATCH_SIZE):
            index = torch.as_tensor(order[start : start + BATCH_SIZE], dtype=torch.long, device=resolved_device)
            optimizer.zero_grad(set_to_none=True)
            output = model(
                train_ego_tensor.index_select(0, index),
                train_yolop_tensor.index_select(0, index),
                train_anchor_tensor.index_select(0, index),
            )
            labels = train_label_tensor.index_select(0, index)
            primary_loss = criterion(output.final_logit, labels)
            correction_penalty = CORRECTION_PENALTY_WEIGHT * output.applied_correction.square().mean()
            loss = primary_loss + correction_penalty
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
            optimizer.step()
            total_loss += float(loss.detach().item()) * len(index)

        model.eval()
        with torch.no_grad():
            validation_output = _prediction_tensors(
                model, validation_ego_tensor, validation_yolop_tensor, validation_anchor_tensor
            )
        validation_probability = validation_output["probability"].detach().cpu().numpy()
        validation_metric = compute_binary_metrics(
            validation_label_tensor.detach().cpu().numpy().astype(np.int64), validation_probability
        )
        score = float(validation_metric.harmonic)
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / train_batch_size,
                "validation_f1": float(validation_metric.f1),
                "validation_pr_auc": float(validation_metric.pr_auc),
                "validation_harmonic": score,
            }
        )
        # Strict comparison preserves the earlier checkpoint on an exact metric tie.
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(
                {name: value.detach().cpu() for name, value in model.state_dict().items()}
            )
            stale_epochs = 0
            if checkpoint_path is not None:
                _atomic_torch_save(
                    {
                        "seed": seed,
                        "best_epoch": best_epoch,
                        "best_validation_harmonic": best_score,
                        "model_state": best_state,
                        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
                    },
                    Path(checkpoint_path),
                )
        else:
            stale_epochs += 1
        if stale_epochs >= EARLY_STOPPING_PATIENCE:
            break

    if best_state is None:
        raise RuntimeError("training produced no validation checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    return TrainingResult(
        model=model,
        best_epoch=best_epoch,
        best_score=best_score,
        history=tuple(history),
        checkpoint_path=None if checkpoint_path is None else Path(checkpoint_path),
    )
