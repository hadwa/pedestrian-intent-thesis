"""Frozen feature transforms and compact residual BIU-LSTM for Exp055."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn


SEQ_LEN = 16
SEEDS = (42, 43, 44)
FIXED_THRESHOLD = 0.5
BIU_MODES = ("v1", "v2_core", "joint_unique")
BBOX_TEMPORAL_MODES = ("summary_only", "raw4", "motion8")
CONTEXT_MODES = ("none", "ego", "yolop", "ego_yolop")
SUMMARY_DIMS = {"v1": 125, "v2_core": 109, "joint_unique": 161}
BBOX_TEMPORAL_DIMS = {"summary_only": 0, "raw4": 4, "motion8": 8}

SUMMARY_HIDDEN_DIM = 96
SUMMARY_EMBED_DIM = 48
BBOX_LSTM_HIDDEN = 16
ANCHOR_EMBED_DIM = 48
EGO_LSTM_HIDDEN = 12
YOLOP_LSTM_HIDDEN = 16
CONTEXT_EMBED_DIM = 32
GATE_HIDDEN_DIM = 16
DROPOUT_RATE = 0.20
CONTEXT_MODALITY_DROPOUT = 0.20
MAX_CONTEXT_DELTA = 2.0

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 64
MAX_EPOCHS = 60
EARLY_STOPPING_PATIENCE = 10
GRADIENT_CLIP_NORM = 5.0
ANCHOR_LOSS_WEIGHT = 0.30
RESIDUAL_PENALTY_WEIGHT = 0.001

EXPECTED_SPLIT_COUNTS = {"train": 2366, "val": 3619, "test": 3760}

# Exact alphabetical schema recorded for Exp012's 56-D BIU-v2 core.
BIU_V2_CORE_FEATURES = (
    "ankle_center_x_delta",
    "ankle_center_x_last",
    "ankle_center_x_velocity_mean",
    "ankle_center_x_velocity_std",
    "body_yaw_angle_delta",
    "body_yaw_angle_last",
    "body_yaw_angle_mean",
    "body_yaw_angle_std",
    "head_yaw_angle_delta",
    "head_yaw_angle_last",
    "head_yaw_angle_mean",
    "head_yaw_angle_std",
    "hip_center_x_delta",
    "hip_center_x_last",
    "hip_center_x_velocity_mean",
    "hip_center_x_velocity_std",
    "knee_center_x_delta",
    "knee_center_x_last",
    "knee_center_x_velocity_mean",
    "knee_center_x_velocity_std",
    "lead_ankle_forward_delta",
    "lead_ankle_forward_last",
    "lead_ankle_forward_velocity_mean",
    "lead_ankle_forward_velocity_std",
    "lead_foot_elevation_delta",
    "lead_foot_elevation_last",
    "lead_foot_elevation_mean",
    "lead_foot_elevation_std",
    "leg_center_x_delta",
    "leg_center_x_last",
    "leg_center_x_velocity_mean",
    "leg_center_x_velocity_std",
    "lower_body_motion_magnitude_mean",
    "lower_body_motion_magnitude_std",
    "shoulder_center_x_delta",
    "shoulder_center_x_last",
    "shoulder_center_x_velocity_mean",
    "shoulder_center_x_velocity_std",
    "stride_length_delta",
    "stride_length_last",
    "stride_length_mean",
    "stride_length_std",
    "stride_opening_delta",
    "stride_opening_last",
    "stride_opening_velocity_mean",
    "stride_opening_velocity_std",
    "torso_center_x_delta",
    "torso_center_x_last",
    "torso_center_x_velocity_mean",
    "torso_center_x_velocity_std",
    "torso_lean_angle_delta",
    "torso_lean_angle_last",
    "torso_lean_angle_mean",
    "torso_lean_angle_std",
    "torso_motion_magnitude_mean",
    "torso_motion_magnitude_std",
)


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


def joint_unique_schema(
    biu_v1_names: Sequence[str], biu_v2_core_names: Sequence[str]
) -> tuple[str, ...]:
    v1 = tuple(str(name) for name in biu_v1_names)
    v2 = tuple(str(name) for name in biu_v2_core_names)
    if len(v1) != len(set(v1)) or len(v2) != len(set(v2)):
        raise ValueError("BIU schemas must contain unique feature names.")
    present = set(v1)
    return v1 + tuple(name for name in v2 if name not in present)


def select_named_features(
    values: np.ndarray, source_names: Sequence[str], selected_names: Sequence[str]
) -> np.ndarray:
    array = np.asarray(values)
    source = tuple(str(name) for name in source_names)
    if array.shape[-1] != len(source):
        raise ValueError("Feature array does not align with its source schema.")
    if len(source) != len(set(source)):
        raise ValueError("Source feature schema contains duplicates.")
    index = {name: position for position, name in enumerate(source)}
    missing = [name for name in selected_names if name not in index]
    if missing:
        raise ValueError(f"Selected features are absent from source schema: {missing}")
    return np.asarray(array[..., [index[name] for name in selected_names]])


def build_biu_representation(
    mode: str,
    biu_v1: np.ndarray,
    biu_v2: np.ndarray,
    biu_v1_names: Sequence[str],
    biu_v2_names: Sequence[str],
) -> tuple[np.ndarray, tuple[str, ...]]:
    if mode not in BIU_MODES:
        raise ValueError(f"Unknown BIU mode: {mode}")
    v1_values = np.asarray(biu_v1)
    v2_values = np.asarray(biu_v2)
    v1_names = tuple(str(name) for name in biu_v1_names)
    if v1_values.shape[:-1] != v2_values.shape[:-1] or v1_values.shape[-1] != len(v1_names):
        raise ValueError("BIU-v1 and BIU-v2 arrays must have aligned populations and schemas.")
    if mode == "v1":
        return v1_values.astype(np.float32, copy=False), v1_names
    v2_core = select_named_features(v2_values, biu_v2_names, BIU_V2_CORE_FEATURES)
    if mode == "v2_core":
        return v2_core.astype(np.float32, copy=False), BIU_V2_CORE_FEATURES
    unique_v2_names = tuple(name for name in BIU_V2_CORE_FEATURES if name not in set(v1_names))
    unique_v2 = select_named_features(v2_values, biu_v2_names, unique_v2_names)
    output = np.concatenate((v1_values, unique_v2), axis=-1)
    return output.astype(np.float32, copy=False), v1_names + unique_v2_names


def bbox_temporal_features(normalized_bbox: np.ndarray, mode: str) -> np.ndarray:
    values = np.asarray(normalized_bbox, dtype=np.float64)
    if values.shape != (SEQ_LEN, 4):
        raise ValueError(f"Expected normalized BBox shape {(SEQ_LEN, 4)}, got {values.shape}.")
    if not np.isfinite(values).all():
        raise ValueError("BBox sequence contains non-finite values.")
    if mode == "raw4":
        return values.astype(np.float32)
    if mode != "motion8":
        raise ValueError(f"BBox temporal features are unavailable for mode: {mode}")
    position = values.copy()
    position[:, 2:] = np.log(np.clip(position[:, 2:], 1e-6, None))
    delta = np.zeros_like(position)
    delta[1:] = position[1:] - position[:-1]
    return np.concatenate((position, delta), axis=1).astype(np.float32)


def fit_feature_scaler(train_values: np.ndarray) -> dict[str, np.ndarray]:
    values = np.asarray(train_values, dtype=np.float64)
    if values.ndim < 2 or values.shape[0] == 0:
        raise ValueError("Training scaler expects a nonempty array with a feature axis.")
    flattened = values.reshape(-1, values.shape[-1])
    median = np.nanmedian(np.where(np.isfinite(flattened), flattened, np.nan), axis=0)
    median = np.where(np.isfinite(median), median, 0.0)
    filled = np.where(np.isfinite(flattened), flattened, median)
    mean = filled.mean(axis=0)
    std = filled.std(axis=0)
    std = np.where(std > 1e-12, std, 1.0)
    return {"median": median, "mean": mean, "std": std}


def transform_features(
    values: np.ndarray, scaler: Mapping[str, np.ndarray]
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    median = np.asarray(scaler["median"], dtype=np.float64)
    mean = np.asarray(scaler["mean"], dtype=np.float64)
    std = np.asarray(scaler["std"], dtype=np.float64)
    if array.shape[-1] != len(mean) or median.shape != mean.shape or std.shape != mean.shape:
        raise ValueError("Feature array and scaler dimensions do not align.")
    filled = np.where(np.isfinite(array), array, median)
    return ((filled - mean) / std).astype(np.float32)


@dataclass(frozen=True, order=True)
class ModelConfig:
    biu_mode: str
    bbox_temporal_mode: str
    context_mode: str

    def __post_init__(self) -> None:
        if self.biu_mode not in BIU_MODES:
            raise ValueError(f"Unknown BIU mode: {self.biu_mode}")
        if self.bbox_temporal_mode not in BBOX_TEMPORAL_MODES:
            raise ValueError(f"Unknown BBox temporal mode: {self.bbox_temporal_mode}")
        if self.context_mode not in CONTEXT_MODES:
            raise ValueError(f"Unknown context mode: {self.context_mode}")

    @property
    def candidate_id(self) -> str:
        return f"{self.biu_mode}__{self.bbox_temporal_mode}__{self.context_mode}"

    @property
    def summary_dim(self) -> int:
        return SUMMARY_DIMS[self.biu_mode]

    def with_context(self, context_mode: str) -> "ModelConfig":
        return replace(self, context_mode=context_mode)

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


class TemporalAttention(nn.Module):
    def __init__(self, input_dim: int, attention_dim: int = 16) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1),
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.score(sequence).squeeze(-1), dim=1)
        return torch.sum(sequence * weights.unsqueeze(-1), dim=1)


class TemporalEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.attention = TemporalAttention(2 * hidden_dim)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        sequence, _ = self.lstm(values)
        return self.attention(sequence)


class CompactResidualBiuLstm(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.summary_encoder = nn.Sequential(
            nn.Linear(config.summary_dim, SUMMARY_HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(SUMMARY_HIDDEN_DIM, SUMMARY_EMBED_DIM),
            nn.GELU(),
        )
        temporal_dim = BBOX_TEMPORAL_DIMS[config.bbox_temporal_mode]
        self.bbox_encoder = (
            None if temporal_dim == 0 else TemporalEncoder(temporal_dim, BBOX_LSTM_HIDDEN)
        )
        anchor_input_dim = SUMMARY_EMBED_DIM + (0 if self.bbox_encoder is None else 2 * BBOX_LSTM_HIDDEN)
        self.anchor_encoder = nn.Sequential(
            nn.Linear(anchor_input_dim, ANCHOR_EMBED_DIM),
            nn.GELU(),
            nn.Dropout(DROPOUT_RATE),
        )
        self.anchor_classifier = nn.Linear(ANCHOR_EMBED_DIM, 1)

        self.ego_encoder = (
            TemporalEncoder(3, EGO_LSTM_HIDDEN)
            if config.context_mode in {"ego", "ego_yolop"}
            else None
        )
        self.yolop_encoder = (
            TemporalEncoder(20, YOLOP_LSTM_HIDDEN)
            if config.context_mode in {"yolop", "ego_yolop"}
            else None
        )
        context_input_dim = (
            (2 * EGO_LSTM_HIDDEN if self.ego_encoder is not None else 0)
            + (2 * YOLOP_LSTM_HIDDEN if self.yolop_encoder is not None else 0)
        )
        if context_input_dim:
            self.context_encoder = nn.Sequential(
                nn.Linear(context_input_dim, CONTEXT_EMBED_DIM),
                nn.GELU(),
                nn.Dropout(DROPOUT_RATE),
            )
            self.delta_head = nn.Linear(CONTEXT_EMBED_DIM, 1)
            self.gate_head = nn.Sequential(
                nn.Linear(ANCHOR_EMBED_DIM + CONTEXT_EMBED_DIM, GATE_HIDDEN_DIM),
                nn.GELU(),
                nn.Linear(GATE_HIDDEN_DIM, 1),
            )
            nn.init.zeros_(self.delta_head.weight)
            nn.init.zeros_(self.delta_head.bias)
            nn.init.zeros_(self.gate_head[-1].weight)
            nn.init.constant_(self.gate_head[-1].bias, -2.0)
        else:
            self.context_encoder = None
            self.delta_head = None
            self.gate_head = None

    def _expected_inputs(self) -> set[str]:
        expected = {"summary"}
        if self.bbox_encoder is not None:
            expected.add("bbox_temporal")
        if self.ego_encoder is not None:
            expected.add("ego")
        if self.yolop_encoder is not None:
            expected.add("yolop")
        return expected

    def _modality_dropout(self, context: torch.Tensor) -> torch.Tensor:
        if not self.training or CONTEXT_MODALITY_DROPOUT <= 0.0:
            return context
        keep = torch.rand(
            (context.shape[0], 1), device=context.device, dtype=context.dtype
        ) >= CONTEXT_MODALITY_DROPOUT
        return context * keep / (1.0 - CONTEXT_MODALITY_DROPOUT)

    def forward(self, inputs: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        expected = self._expected_inputs()
        if set(inputs) != expected:
            raise ValueError(
                f"Unexpected model inputs: expected {sorted(expected)}, got {sorted(inputs)}"
            )
        summary = self.summary_encoder(inputs["summary"])
        anchor_parts = [summary]
        if self.bbox_encoder is not None:
            anchor_parts.append(self.bbox_encoder(inputs["bbox_temporal"]))
        anchor_embedding = self.anchor_encoder(torch.cat(anchor_parts, dim=1))
        anchor_logit = self.anchor_classifier(anchor_embedding).squeeze(1)

        if self.context_encoder is None:
            zeros = torch.zeros_like(anchor_logit)
            return {
                "final_logit": anchor_logit,
                "anchor_logit": anchor_logit,
                "gate": zeros,
                "bounded_delta": zeros,
                "applied_correction": zeros,
            }

        context_parts = []
        if self.ego_encoder is not None:
            context_parts.append(self._modality_dropout(self.ego_encoder(inputs["ego"])))
        if self.yolop_encoder is not None:
            context_parts.append(self._modality_dropout(self.yolop_encoder(inputs["yolop"])))
        context_embedding = self.context_encoder(torch.cat(context_parts, dim=1))
        bounded_delta = MAX_CONTEXT_DELTA * torch.tanh(self.delta_head(context_embedding)).squeeze(1)
        gate = torch.sigmoid(
            self.gate_head(torch.cat((anchor_embedding, context_embedding), dim=1))
        ).squeeze(1)
        applied = gate * bounded_delta
        return {
            "final_logit": anchor_logit + applied,
            "anchor_logit": anchor_logit,
            "gate": gate,
            "bounded_delta": bounded_delta,
            "applied_correction": applied,
        }


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def training_objective(
    output: Mapping[str, torch.Tensor], labels: torch.Tensor
) -> dict[str, torch.Tensor]:
    target = labels.to(dtype=output["final_logit"].dtype)
    criterion = nn.BCEWithLogitsLoss()
    primary = criterion(output["final_logit"], target)
    anchor = criterion(output["anchor_logit"], target)
    residual = torch.mean(output["applied_correction"].square())
    total = primary + ANCHOR_LOSS_WEIGHT * anchor + RESIDUAL_PENALTY_WEIGHT * residual
    return {"total": total, "primary": primary, "anchor": anchor, "residual": residual}


def selection_score(f1: float, pr_auc: float) -> float:
    total = float(f1) + float(pr_auc)
    return 0.0 if total <= 0.0 else 2.0 * float(f1) * float(pr_auc) / total


def compute_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float | int]:
    y_true = np.asarray(labels, dtype=np.int64)
    probability = np.asarray(probabilities, dtype=np.float64)
    if y_true.ndim != 1 or probability.shape != y_true.shape:
        raise ValueError("Labels and probabilities must be aligned one-dimensional arrays.")
    predicted = (probability >= FIXED_THRESHOLD).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    result: dict[str, float | int] = {
        "accuracy": float(accuracy_score(y_true, predicted)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "pr_auc": float(average_precision_score(y_true, probability)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "high_confidence_fp": int(((y_true == 0) & (probability >= 0.9)).sum()),
        "high_confidence_fn": int(((y_true == 1) & (probability <= 0.1)).sum()),
    }
    result["selection_score"] = selection_score(float(result["f1"]), float(result["pr_auc"]))
    return result
