"""Frozen contracts for Exp047's controlled MSLSTM pose replacement."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import pickle
import random
import re
import shutil
import sys
import types
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping
from typing import Callable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.biu_v2_extractor import extract_biu_v2


EXPERIMENT_NUMBER = 47
SEQ_LEN = 30
PRED_LEN = 1
SEEDS = (42, 43, 44)
VARIANTS = ("A", "B", "C")
NONINFERIORITY_TOLERANCE = 0.005

REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ROOT = REPO_ROOT / "experiments" / "047_mslstm_vitpose_biu_pose_replacement"
EXP032_RUNNER = REPO_ROOT / "scripts" / "experiments" / "exp032_mslstm_biu_v2_transfer.py"
EXP032_PROTOCOL_ROWS = (
    REPO_ROOT
    / "experiments"
    / "032_mslstm_biu_v2_transfer"
    / "full_run_corrected_2026-08-16"
    / "metrics"
    / "protocol_rows.csv"
)
EXP012_POSE_PATH = Path(
    "/home/abdelrahman/Hadwa/Pedestrian-keypoints-crossing-intent/"
    "classification/data/data_keypoints_all.jsonl"
)
EXP012_MIRROR_PATH = Path(
    "/home/abdelrahman/Hadwa/pedestrian-intent-biu/data/data_keypoints_all.jsonl"
)
EXP012_POSE_SHA256 = "1f0907ebb33bb04c3a3434d9c89e7bd597e2b9e3d47fe30f488ffa9410054d87"
YOLO_POSE_ROOT = Path("/home/abdelrahman/Hadwa/MSLSTM-PID/Extracted Poses")
PIE_DATABASE_PATH = Path(
    "/home/abdelrahman/Hadwa/MSLSTM-PID/PIE/annotations/data_cache/pie_database.pkl"
)
TRAIN_INDEX_PATH = Path("/home/abdelrahman/Hadwa/MSLSTM-PID/checkpoints/train_index.pkl")
PIE_ANNOTATION_ROOT = Path("/home/abdelrahman/Hadwa/MSLSTM-PID/PIE/annotations")
PIE_PARSER_PATH = Path("/home/abdelrahman/Hadwa/MSLSTM-PID/PIE/utilities/pie_data.py")
SCALERS_PATH = Path("/home/abdelrahman/Hadwa/MSLSTM-PID/checkpoints/scalers.pkl")

EXPECTED_SPLIT_COUNTS = {"train": 109_934, "val": 77_288, "test": 271_078}
ACTIVE_STREAMS = (
    "bbox",
    "pose",
    "ped_action",
    "ped_look",
    "ego_speed",
    "ego_acc",
    "static_context",
)
STREAM_DIMS = {
    "bbox": 4,
    "pose": 34,
    "ped_action": 1,
    "ped_look": 1,
    "ego_speed": 1,
    "ego_acc": 2,
    "static_context": 23,
}
LSTM_HIDDEN_SIZE = 256
NUM_LSTM_LAYERS = 2
LSTM_OUTPUT_DIM = 512
ATTENTION_DIM = 128
DROPOUT_RATE = 0.3
BIU_DIM = 63
TRAIN_SETS = ("set01", "set02", "set04")
VAL_SETS = ("set05", "set06")
TEST_SETS = ("set03",)
NUM_SIGNALIZED_CATS = 4
NUM_INTERSECTION_CATS = 5
NUM_AGE_CATS = 4
NUM_GENDER_CATS = 3
NUM_TRAFFIC_DIR_CATS = 2
LANE_CATEGORIES = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 4, 7: 4, 8: 4}
NUM_LANE_CATS = len(set(LANE_CATEGORIES.values()))


@dataclass(frozen=True)
class FrozenProtocol:
    sequence_length: int = SEQ_LEN
    prediction_length: int = PRED_LEN
    seeds: tuple[int, ...] = SEEDS
    variants: tuple[str, ...] = VARIANTS
    learning_rate: float = 1e-4
    batch_size: int = 32
    epoch_limit: int = 5
    fixed_threshold: float = 0.5
    checkpoint_rule: str = "validation_f1_at_fixed_0p5"
    noninferiority_tolerance: float = NONINFERIORITY_TOLERANCE


@dataclass(frozen=True)
class SourceManifest:
    exp012_pose_path: str
    exp012_pose_sha256: str
    exp012_mirror_path: str
    yolo_pose_root: str
    pie_database_path: str
    train_index_path: str
    exp032_protocol_rows: str
    exp032_runner: str

    @classmethod
    def frozen(cls) -> "SourceManifest":
        return cls(
            exp012_pose_path=str(EXP012_POSE_PATH),
            exp012_pose_sha256=EXP012_POSE_SHA256,
            exp012_mirror_path=str(EXP012_MIRROR_PATH),
            yolo_pose_root=str(YOLO_POSE_ROOT),
            pie_database_path=str(PIE_DATABASE_PATH),
            train_index_path=str(TRAIN_INDEX_PATH),
            exp032_protocol_rows=str(EXP032_PROTOCOL_ROWS),
            exp032_runner=str(EXP032_RUNNER),
        )


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def protocol_sha256(protocol: FrozenProtocol | None = None) -> str:
    return canonical_json_sha256(asdict(protocol or FrozenProtocol()))


def source_manifest_sha256(manifest: SourceManifest | None = None) -> str:
    return canonical_json_sha256(asdict(manifest or SourceManifest.frozen()))


def verify_protocol_rows(
    rows: pd.DataFrame,
    expected_counts: Mapping[str, int] | None = EXPECTED_SPLIT_COUNTS,
) -> dict[str, object]:
    required = {
        "sequence_id",
        "split",
        "start_frame",
        "end_frame",
        "target_frame",
        "observation_length",
        "pred_len",
        "label",
    }
    missing = sorted(required.difference(rows.columns))
    if missing:
        raise ValueError(f"Protocol rows are missing columns: {missing}")
    if rows.empty:
        raise ValueError("Protocol rows are empty.")
    if rows["sequence_id"].duplicated().any():
        raise ValueError("Protocol contains duplicate sequence IDs.")
    if not (rows["observation_length"].astype(int) == SEQ_LEN).all():
        raise ValueError(f"Every observation_length must equal {SEQ_LEN}.")
    if not (rows["pred_len"].astype(int) == PRED_LEN).all():
        raise ValueError(f"Every pred_len must equal {PRED_LEN}.")
    if not ((rows["end_frame"].astype(int) - rows["start_frame"].astype(int)) == SEQ_LEN - 1).all():
        raise ValueError("Every observation must satisfy end_frame - start_frame = 29.")
    if not ((rows["target_frame"].astype(int) - rows["end_frame"].astype(int)) == PRED_LEN).all():
        raise ValueError("Every target must be the next raw frame after observation.")
    labels = set(rows["label"].astype(int).unique())
    if not labels.issubset({0, 1}):
        raise ValueError(f"Protocol labels must be binary, got {sorted(labels)}.")
    counts = {str(k): int(v) for k, v in rows.groupby("split").size().items()}
    if expected_counts is not None and counts != dict(expected_counts):
        raise ValueError(f"Protocol split counts {counts} do not match {dict(expected_counts)}.")
    return {
        "n_rows": int(len(rows)),
        "split_counts": counts,
        "observation_length": SEQ_LEN,
        "target_delta": PRED_LEN,
    }


def verify_checkpoint_selection_source(path: Path = EXP032_RUNNER) -> dict[str, object]:
    text = Path(path).read_text(encoding="utf-8")
    required_fragments = (
        "NUM_EPOCHS = 5",
        "for epoch in range(1, epochs + 1):",
        'val_metrics = jaad._compute_metrics(y_val, p_val, 0.5)',
        'if val_metrics["f1"] > best_val_f1:',
        'best_val_f1 = float(val_metrics["f1"])',
        "best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}",
    )
    missing = [fragment for fragment in required_fragments if fragment not in text]
    if missing:
        raise ValueError(f"Recovered checkpoint-selection evidence is incomplete: {missing}")
    return {
        "source_path": str(Path(path).resolve()),
        "source_sha256": sha256_file(path),
        "epoch_limit": 5,
        "selection_metric": "validation_f1_at_fixed_0p5",
        "training_runs_all_epochs": True,
    }


def write_phase_record(path: Path, payload: Mapping[str, object], resume: bool = False) -> None:
    destination = Path(path)
    normalized = json.loads(json.dumps(dict(payload), sort_keys=True))
    if destination.exists():
        if not resume:
            raise FileExistsError(f"Phase record already exists: {destination}")
        existing = json.loads(destination.read_text(encoding="utf-8"))
        if existing != normalized:
            raise ValueError(f"Existing phase record does not match requested payload: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(normalized, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(destination)


def authenticate_phase_record(path: Path, *, expected_phase: str) -> dict[str, object]:
    marker = Path(path)
    if not marker.is_file():
        raise FileNotFoundError(f"Required phase record is absent: {marker}")
    payload = json.loads(marker.read_text(encoding="utf-8"))
    if payload.get("phase") != expected_phase:
        raise ValueError(f"Phase record is {payload.get('phase')!r}, expected {expected_phase!r}.")
    if payload.get("protocol_sha256") != protocol_sha256():
        raise ValueError("Phase record protocol hash does not match the frozen Exp047 protocol.")
    if payload.get("source_manifest_sha256") != source_manifest_sha256():
        raise ValueError("Phase record source-manifest hash does not match frozen provenance.")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("Phase record artifact manifest is invalid.")
    for source, expected_hash in artifacts.items():
        artifact = Path(source)
        if not artifact.is_file() or sha256_file(artifact) != expected_hash:
            raise ValueError(f"Phase artifact hash mismatch: {artifact}")
    return payload


def normalize_exp012_pose(coords_xy: np.ndarray, width: int, height: int) -> np.ndarray:
    coords = np.asarray(coords_xy, dtype=np.float64)
    if coords.shape != (17, 2):
        raise ValueError(f"Expected pose shape (17, 2), got {coords.shape}.")
    if width <= 0 or height <= 0:
        raise ValueError("Image width and height must be positive.")
    normalized = coords.copy()
    normalized[:, 0] /= float(width)
    normalized[:, 1] /= float(height)
    return normalized.reshape(34).astype(np.float32)


def _available_joint_mask(coords_xy: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords_xy, dtype=np.float64)
    if coords.ndim != 3 or coords.shape[1:] != (17, 2):
        raise ValueError(f"Expected window shape (T, 17, 2), got {coords.shape}.")
    return np.isfinite(coords).all(axis=2) & np.any(coords != 0.0, axis=2)


def pose_joint_fraction(coords_xy: np.ndarray) -> float:
    return float(_available_joint_mask(coords_xy).mean())


def pose_window_is_valid(coords_xy: np.ndarray) -> bool:
    return bool(_available_joint_mask(coords_xy).any())


def extract_frozen_biu(
    coords_xy: np.ndarray,
    bbox_sequence: np.ndarray,
    image_size: tuple[int, int],
) -> tuple[np.ndarray, list[str], str]:
    coords = np.asarray(coords_xy, dtype=np.float64)
    bbox = np.asarray(bbox_sequence, dtype=np.float64)
    if coords.shape != (SEQ_LEN, 17, 2):
        raise ValueError(f"Expected exactly {SEQ_LEN} observed pose frames, got {coords.shape}.")
    if bbox.shape != (SEQ_LEN, 4):
        raise ValueError(f"Expected exactly {SEQ_LEN} observed BBoxes, got {bbox.shape}.")
    features = extract_biu_v2(
        coords,
        kp_conf=None,
        bbox_sequence=bbox,
        image_size=image_size,
        include_optional=False,
    )
    direction_source = str(features["direction_source"])
    names = [
        name
        for name, value in features.items()
        if isinstance(value, (int, float, np.integer, np.floating))
    ]
    forbidden_pose_confidence = ("skeleton_conf", "lower_body_conf", "torso_conf", "head_conf", "kp_conf")
    if len(names) != BIU_DIM or any(any(token in name for token in forbidden_pose_confidence) for name in names):
        raise RuntimeError(f"Frozen BIU schema must be {BIU_DIM}-D without confidence, got {names}.")
    values = np.asarray([float(features[name]) for name in names], dtype=np.float64)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return values, names, direction_source


class StreamAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(LSTM_OUTPUT_DIM, ATTENTION_DIM),
            nn.Tanh(),
            nn.Linear(ATTENTION_DIM, 1),
        )

    def forward(self, sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.net(sequence).squeeze(2)
        weights = torch.softmax(scores, dim=1)
        return torch.sum(sequence * weights.unsqueeze(2), dim=1), weights


class _MslstmFusionClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fusion_weights = nn.Parameter(torch.ones(len(ACTIVE_STREAMS)))
        self.dropout = nn.Dropout(DROPOUT_RATE)
        self.fc1 = nn.Linear(LSTM_OUTPUT_DIM, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 2)

    def _classify(self, contexts: list[torch.Tensor]) -> torch.Tensor:
        if len(contexts) != len(ACTIVE_STREAMS):
            raise RuntimeError(f"Expected seven fusion contexts, got {len(contexts)}.")
        stacked = torch.stack(contexts, dim=1)
        weights = torch.softmax(self.fusion_weights, dim=0).view(1, -1, 1)
        fused = torch.sum(stacked * weights, dim=1)
        hidden = self.dropout(fused)
        hidden = self.relu(self.fc1(hidden))
        hidden = self.dropout(hidden)
        return self.fc2(hidden)


class MslstmRawPoseModel(_MslstmFusionClassifier):
    """Variants A/B: identical architecture; only their pose arrays differ."""

    def __init__(self) -> None:
        super().__init__()
        self.lstms = nn.ModuleDict()
        self.attentions = nn.ModuleDict()
        for name in ACTIVE_STREAMS:
            self.lstms[name] = nn.LSTM(
                input_size=STREAM_DIMS[name],
                hidden_size=LSTM_HIDDEN_SIZE,
                num_layers=NUM_LSTM_LAYERS,
                batch_first=True,
                dropout=DROPOUT_RATE,
                bidirectional=True,
            )
            self.attentions[name] = StreamAttention()

    def forward(self, inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        contexts = []
        for name in ACTIVE_STREAMS:
            sequence, _ = self.lstms[name](inputs[name])
            context, _ = self.attentions[name](sequence)
            contexts.append(context)
        return self._classify(contexts)


class MslstmBiuPoseReplacementModel(_MslstmFusionClassifier):
    """Variant C: BIU projection occupies the original pose-context slot."""

    def __init__(self) -> None:
        super().__init__()
        nonpose_streams = [name for name in ACTIVE_STREAMS if name != "pose"]
        self.lstms = nn.ModuleDict()
        self.attentions = nn.ModuleDict()
        for name in nonpose_streams:
            self.lstms[name] = nn.LSTM(
                input_size=STREAM_DIMS[name],
                hidden_size=LSTM_HIDDEN_SIZE,
                num_layers=NUM_LSTM_LAYERS,
                batch_first=True,
                dropout=DROPOUT_RATE,
                bidirectional=True,
            )
            self.attentions[name] = StreamAttention()
        self.biu_projection = nn.Sequential(
            nn.Linear(BIU_DIM, ATTENTION_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(ATTENTION_DIM, LSTM_OUTPUT_DIM),
        )

    def forward(self, inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        contexts = []
        for name in ACTIVE_STREAMS:
            if name == "pose":
                contexts.append(self.biu_projection(inputs["biu_v2"]))
                continue
            sequence, _ = self.lstms[name](inputs[name])
            context, _ = self.attentions[name](sequence)
            contexts.append(context)
        return self._classify(contexts)


def count_parameters(model: nn.Module) -> tuple[int, int]:
    return (
        int(sum(parameter.numel() for parameter in model.parameters())),
        int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)),
    )


def count_pose_path_parameters(model: nn.Module) -> int:
    if isinstance(model, MslstmRawPoseModel):
        return int(
            sum(parameter.numel() for parameter in model.lstms["pose"].parameters())
            + sum(parameter.numel() for parameter in model.attentions["pose"].parameters())
        )
    if isinstance(model, MslstmBiuPoseReplacementModel):
        return int(sum(parameter.numel() for parameter in model.biu_projection.parameters()))
    raise TypeError(f"Unsupported model type: {type(model).__name__}")


def seed_everything(seed: int) -> None:
    if int(seed) not in SEEDS:
        raise ValueError(f"Seed must be one of the frozen seeds {SEEDS}, got {seed}.")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_frozen_model(variant: str, seed: int) -> nn.Module:
    if variant not in VARIANTS:
        raise ValueError(f"Variant must be one of {VARIANTS}, got {variant!r}.")
    seed_everything(seed)
    if variant in {"A", "B"}:
        return MslstmRawPoseModel()
    return MslstmBiuPoseReplacementModel()


def training_rows_only(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"sequence_id", "split", "label"}
    missing = sorted(required.difference(rows.columns))
    if missing:
        raise ValueError(f"Training manifest is missing columns: {missing}")
    train = rows.loc[rows["split"] == "train"].copy().reset_index(drop=True)
    val = rows.loc[rows["split"] == "val"].copy().reset_index(drop=True)
    if train.empty or val.empty:
        raise ValueError("Training requires nonempty train and validation cohorts.")
    if train["sequence_id"].duplicated().any() or val["sequence_id"].duplicated().any():
        raise ValueError("Train or validation cohort contains duplicate sequence IDs.")
    if set(train["sequence_id"]).intersection(val["sequence_id"]):
        raise ValueError("Train and validation cohorts overlap.")
    return train, val


def select_best_validation_epoch(history: pd.DataFrame) -> int:
    if len(history) != FrozenProtocol().epoch_limit:
        raise ValueError("Checkpoint selection requires exactly five completed epochs.")
    if history["epoch"].astype(int).tolist() != list(range(1, FrozenProtocol().epoch_limit + 1)):
        raise ValueError("Checkpoint history must contain ordered epochs 1 through 5.")
    values = history["val_f1_at_0p5"].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Validation F1 history contains non-finite values.")
    return int(history.iloc[int(np.argmax(values))]["epoch"])


def compute_binary_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(labels, dtype=np.int64).reshape(-1)
    y_prob = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if len(y_true) == 0 or len(y_true) != len(y_prob):
        raise ValueError("Labels and probabilities must be nonempty and aligned.")
    if not np.isin(y_true, (0, 1)).all() or not np.isfinite(y_prob).all():
        raise ValueError("Binary metrics require finite probabilities and labels in {0,1}.")
    threshold = FrozenProtocol().fixed_threshold
    prediction = (y_prob >= threshold).astype(np.int64)
    ranking_defined = len(np.unique(y_true)) == 2
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, prediction)),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if ranking_defined else float("nan"),
        "pr_auc": float(average_precision_score(y_true, y_prob)) if ranking_defined else float("nan"),
    }


def authenticate_training_records(records: list[Mapping[str, object]]) -> list[dict[str, object]]:
    expected = [(variant, seed) for variant in VARIANTS for seed in SEEDS]
    if len(records) != len(expected):
        raise ValueError("Final evaluation requires exactly nine training records.")
    by_identity: dict[tuple[str, int], dict[str, object]] = {}
    for source in records:
        record = dict(source)
        identity = (str(record.get("variant")), int(record.get("seed", -1)))
        if identity not in expected or identity in by_identity:
            raise ValueError(f"Invalid or duplicate training identity: {identity}")
        checkpoint = Path(str(record.get("checkpoint_path", "")))
        expected_hash = str(record.get("checkpoint_sha256", ""))
        if not checkpoint.is_file() or sha256_file(checkpoint) != expected_hash:
            raise ValueError(f"Checkpoint hash mismatch for {identity}.")
        by_identity[identity] = record
    if set(by_identity) != set(expected):
        raise ValueError("Training records do not contain the exact frozen run grid.")
    return [by_identity[identity] for identity in expected]


def classify_biu_outcome(
    *,
    b_pr_auc: float,
    b_f1: float,
    c_pr_auc: float,
    c_f1: float,
    pr_auc_favorable_all_seeds: bool,
    f1_favorable_all_seeds: bool,
) -> str:
    pr_delta = float(c_pr_auc - b_pr_auc)
    f1_delta = float(c_f1 - b_f1)
    tolerance = NONINFERIORITY_TOLERANCE
    if pr_auc_favorable_all_seeds and f1_favorable_all_seeds and pr_delta > 0 and f1_delta > 0:
        return "strong_biu_replacement_success"
    if pr_auc_favorable_all_seeds and pr_delta > 0 and f1_delta >= -tolerance:
        return "biu_ranking_success"
    if pr_delta >= -tolerance and f1_delta >= -tolerance:
        if pr_delta <= 0 or f1_delta <= 0:
            return "biu_compact_representation_success"
        return "neutral"
    if pr_delta < -tolerance and f1_delta < -tolerance:
        return "negative_biu_result"
    return "neutral"


class Exp047CachedDataset(Dataset):
    """Aligned cached windows with the pose pathway selected only by variant."""

    def __init__(
        self,
        rows: pd.DataFrame,
        arrays: Mapping[str, np.ndarray],
        *,
        variant: str,
        training_phase: bool = True,
    ) -> None:
        if variant not in VARIANTS:
            raise ValueError(f"Variant must be one of {VARIANTS}, got {variant!r}.")
        self.rows = rows.reset_index(drop=True).copy()
        if self.rows.empty:
            raise ValueError("Cached dataset cannot be empty.")
        if training_phase and self.rows["split"].isin({"test", "set03"}).any():
            raise ValueError("Training datasets must not contain test/set03 rows.")
        if self.rows["sequence_id"].duplicated().any():
            raise ValueError("Cached dataset contains duplicate sequence IDs.")
        self.arrays = arrays
        self.variant = variant
        required = {"bbox", "ped_action", "ped_look", "ego_speed", "ego_acc", "static_context"}
        required.add("yolo_pose" if variant == "A" else "exp012_pose" if variant == "B" else "biu_v2")
        missing = sorted(required.difference(arrays))
        if missing:
            raise ValueError(f"Cached dataset arrays are missing: {missing}")
        max_index = int(self.rows["cache_index"].max())
        for name in required:
            if len(arrays[name]) <= max_index:
                raise ValueError(f"Cached array {name} is shorter than the row manifest.")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows.iloc[int(index)]
        cache_index = int(row["cache_index"])
        inputs = {
            name: torch.from_numpy(
                np.array(self.arrays[name][cache_index], dtype=np.float32, copy=True)
            )
            for name in ("bbox", "ped_action", "ped_look", "ego_speed", "ego_acc")
        }
        static = np.asarray(self.arrays["static_context"][cache_index], dtype=np.float32)
        if static.shape == (STREAM_DIMS["static_context"],):
            static = np.broadcast_to(static, (SEQ_LEN, STREAM_DIMS["static_context"]))
        inputs["static_context"] = torch.as_tensor(static.copy(), dtype=torch.float32)
        if self.variant == "A":
            inputs["pose"] = torch.from_numpy(
                np.array(self.arrays["yolo_pose"][cache_index], dtype=np.float32, copy=True)
            )
        elif self.variant == "B":
            inputs["pose"] = torch.from_numpy(
                np.array(self.arrays["exp012_pose"][cache_index], dtype=np.float32, copy=True)
            )
        else:
            inputs["biu_v2"] = torch.from_numpy(
                np.array(self.arrays["biu_v2"][cache_index], dtype=np.float32, copy=True)
            )
        label = torch.tensor(int(row["label"]), dtype=torch.long)
        meta = {
            "sequence_id": str(row["sequence_id"]),
            "split": str(row["split"]),
            "label": int(row["label"]),
            "cache_index": cache_index,
        }
        return inputs, label, meta


def verify_training_cache_rows(path: Path, expected_rows: pd.DataFrame) -> dict[str, object]:
    cached = pd.read_csv(path)
    required = ["sequence_id", "split", "label", "cache_index"]
    if any(column not in cached.columns for column in required):
        raise ValueError("Training cache row manifest is incomplete.")
    if cached["split"].isin({"test", "set03"}).any():
        raise ValueError("Training cache row manifest contains test/set03 rows.")
    expected = expected_rows.loc[:, required].reset_index(drop=True).copy()
    actual = cached.loc[:, required].reset_index(drop=True).copy()
    if actual["sequence_id"].astype(str).tolist() != expected["sequence_id"].astype(str).tolist():
        raise ValueError("Training cache row order does not match the frozen manifest.")
    if actual["split"].astype(str).tolist() != expected["split"].astype(str).tolist():
        raise ValueError("Training cache split order does not match the frozen manifest.")
    if not np.array_equal(actual["label"].to_numpy(dtype=np.int64), expected["label"].to_numpy(dtype=np.int64)):
        raise ValueError("Training cache labels do not match the frozen manifest.")
    if actual["cache_index"].astype(int).tolist() != list(range(len(actual))):
        raise ValueError("Training cache indices are not contiguous in frozen row order.")
    return {"n_rows": int(len(actual)), "row_manifest_sha256": sha256_file(path)}


_STREAM_CACHE_SHAPES = {
    "bbox": (SEQ_LEN, STREAM_DIMS["bbox"]),
    "ped_action": (SEQ_LEN, STREAM_DIMS["ped_action"]),
    "ped_look": (SEQ_LEN, STREAM_DIMS["ped_look"]),
    "ego_speed": (SEQ_LEN, STREAM_DIMS["ego_speed"]),
    "ego_acc": (SEQ_LEN, STREAM_DIMS["ego_acc"]),
    "static_context": (STREAM_DIMS["static_context"],),
    "yolo_pose": (SEQ_LEN, STREAM_DIMS["pose"]),
    "exp012_pose": (SEQ_LEN, STREAM_DIMS["pose"]),
    "biu_v2": (BIU_DIM,),
    "yolo_pose_valid": (),
    "exp012_pose_valid": (),
}


def build_stream_cache(
    rows: pd.DataFrame,
    *,
    destination: Path,
    pie_db: Mapping[str, object],
    yolo_store: object,
    exp012_store: object,
    scalers: Mapping[str, float],
    biu_cache_path: Path,
    training_phase: bool,
    progress: Callable[[str], None] = print,
) -> dict[str, object]:
    selected = rows.reset_index(drop=True).copy()
    if selected.empty:
        raise ValueError("Stream cache requires at least one row.")
    if training_phase and selected["split"].isin({"test", "set03"}).any():
        raise ValueError("Training stream cache must not contain test/set03 rows.")
    if selected["sequence_id"].duplicated().any():
        raise ValueError("Stream cache rows contain duplicate sequence IDs.")
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"Stream cache already exists: {destination}")
    temporary = destination.with_name(destination.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"Partial stream cache must be audited before removal: {temporary}")
    temporary.mkdir(parents=True)
    n_rows = len(selected)
    selected["cache_index"] = np.arange(n_rows, dtype=np.int64)
    arrays: dict[str, np.memmap] = {}
    try:
        for name, trailing_shape in _STREAM_CACHE_SHAPES.items():
            dtype = np.bool_ if name.endswith("_valid") else np.float32
            arrays[name] = np.lib.format.open_memmap(
                temporary / f"{name}.npy",
                mode="w+",
                dtype=dtype,
                shape=(n_rows, *trailing_shape),
            )
        biu_source = np.load(biu_cache_path, mmap_mode="r")
        if biu_source.ndim != 2 or biu_source.shape[1] != BIU_DIM:
            raise ValueError(f"Frozen BIU cache shape is invalid: {biu_source.shape}")
        for position, row in enumerate(selected.itertuples(index=False)):
            exp047_index = int(row.exp047_index)
            if exp047_index < 0 or exp047_index >= len(biu_source):
                raise ValueError(f"BIU cache index is invalid for {row.sequence_id}: {exp047_index}")
            extracted = extract_mslstm_stream_window(
                row,
                pie_db=pie_db,
                yolo_store=yolo_store,
                exp012_store=exp012_store,
                scalers=scalers,
            )
            for name in _STREAM_CACHE_SHAPES:
                if name == "biu_v2":
                    arrays[name][position] = biu_source[exp047_index]
                else:
                    arrays[name][position] = extracted[name]
            if (position + 1) % 5_000 == 0 or position + 1 == n_rows:
                progress(f"stream_cache rows={position + 1}/{n_rows} split={row.split}")
        for array in arrays.values():
            array.flush()
        arrays.clear()
        rows_path = temporary / "rows.csv"
        selected.to_csv(rows_path, index=False)
        artifact_hashes = {
            path.name: sha256_file(path)
            for path in sorted(temporary.glob("*.npy"))
        }
        artifact_hashes[rows_path.name] = sha256_file(rows_path)
        metadata = {
            "n_rows": int(n_rows),
            "training_phase": bool(training_phase),
            "splits": selected["split"].value_counts().sort_index().astype(int).to_dict(),
            "biu_source_path": str(Path(biu_cache_path).resolve()),
            "biu_source_sha256": sha256_file(biu_cache_path),
            "artifacts": artifact_hashes,
        }
        (temporary / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary.replace(destination)
        return metadata
    except BaseException:
        arrays.clear()
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def load_stream_cache(
    root: Path,
    *,
    training_phase: bool,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, object]]:
    root = Path(root)
    metadata_path = root / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Stream cache metadata is absent: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if bool(metadata.get("training_phase")) != bool(training_phase):
        raise ValueError("Stream cache phase does not match the requested access mode.")
    artifacts = metadata.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != {
        "rows.csv",
        *(f"{name}.npy" for name in _STREAM_CACHE_SHAPES),
    }:
        raise ValueError("Stream cache artifact manifest is not exact.")
    for relative, expected_hash in artifacts.items():
        path = root / relative
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise ValueError(f"Stream cache artifact hash mismatch: {path}")
    rows = pd.read_csv(root / "rows.csv")
    if training_phase and rows["split"].isin({"test", "set03"}).any():
        raise ValueError("Training stream cache contains test/set03 rows.")
    arrays = {
        name: np.load(root / f"{name}.npy", mmap_mode="r")
        for name in _STREAM_CACHE_SHAPES
    }
    n_rows = int(metadata["n_rows"])
    if len(rows) != n_rows or any(len(array) != n_rows for array in arrays.values()):
        raise ValueError("Stream cache population lengths are inconsistent.")
    verify_training_cache_rows(root / "rows.csv", rows) if training_phase else None
    return rows, arrays, metadata


@dataclass
class FrozenTrainingResult:
    model: nn.Module
    history: pd.DataFrame
    best_epoch: int
    best_validation_f1: float
    validation_labels: np.ndarray
    validation_probabilities: np.ndarray


def _predict_loader(
    model: nn.Module,
    loader: object,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    labels: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch_inputs, batch_labels, _ in loader:
            inputs = {name: tensor.to(device) for name, tensor in batch_inputs.items()}
            logits = model(inputs)
            probabilities.append(torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy())
            labels.append(batch_labels.detach().cpu().numpy())
    if not labels:
        raise ValueError("Prediction loader produced no batches.")
    return np.concatenate(labels), np.concatenate(probabilities)


def _atomic_torch_save(payload: Mapping[str, object], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    temporary.replace(path)


def fit_frozen_epochs(
    model: nn.Module,
    *,
    train_loader: object,
    val_loader: object,
    variant: str,
    seed: int,
    device: torch.device,
    state_path: Path,
    data_generator: torch.Generator,
    progress: Callable[[str], None] = print,
) -> FrozenTrainingResult:
    if variant not in VARIANTS or seed not in SEEDS:
        raise ValueError("Training identity is outside the frozen variant/seed grid.")
    protocol = FrozenProtocol()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=protocol.learning_rate)
    criterion = nn.CrossEntropyLoss()
    state_path = Path(state_path)
    history: list[dict[str, object]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    best_f1 = -1.0
    start_epoch = 1
    if state_path.exists():
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if state.get("variant") != variant or int(state.get("seed", -1)) != seed:
            raise ValueError("Resumable training state identity does not match the requested run.")
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        for optimizer_state in optimizer.state.values():
            for name, value in optimizer_state.items():
                if torch.is_tensor(value):
                    optimizer_state[name] = value.to(device)
        history = list(state["history"])
        best_state = state["best_state"]
        best_epoch = int(state["best_epoch"])
        best_f1 = float(state["best_validation_f1"])
        start_epoch = int(state["epoch"]) + 1
        torch.set_rng_state(state["torch_rng_state"])
        data_generator.set_state(state["data_generator_state"])
        if torch.cuda.is_available() and state.get("cuda_rng_state_all") is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng_state_all"])
        progress(f"resume variant={variant} seed={seed} epoch={start_epoch}")
    for epoch in range(start_epoch, protocol.epoch_limit + 1):
        model.train()
        losses: list[float] = []
        for batch_inputs, batch_labels, _ in train_loader:
            inputs = {name: tensor.to(device) for name, tensor in batch_inputs.items()}
            labels = batch_labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        val_labels, val_probabilities = _predict_loader(model, val_loader, device)
        metrics = compute_binary_metrics(val_labels, val_probabilities)
        history.append(
            {
                "variant": variant,
                "seed": int(seed),
                "epoch": int(epoch),
                "train_loss_mean": float(np.mean(losses)),
                "val_accuracy_at_0p5": metrics["accuracy"],
                "val_precision_at_0p5": metrics["precision"],
                "val_recall_at_0p5": metrics["recall"],
                "val_f1_at_0p5": metrics["f1"],
                "val_roc_auc": metrics["roc_auc"],
                "val_pr_auc": metrics["pr_auc"],
            }
        )
        if metrics["f1"] > best_f1:
            best_f1 = float(metrics["f1"])
            best_epoch = int(epoch)
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        if best_state is None:
            raise RuntimeError("No best validation state was captured.")
        _atomic_torch_save(
            {
                "variant": variant,
                "seed": int(seed),
                "epoch": int(epoch),
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_state": best_state,
                "best_epoch": int(best_epoch),
                "best_validation_f1": float(best_f1),
                "history": history,
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "data_generator_state": data_generator.get_state(),
            },
            state_path,
        )
        progress(
            f"train variant={variant} seed={seed} epoch={epoch}/{protocol.epoch_limit} "
            f"loss={history[-1]['train_loss_mean']:.6f} val_f1={metrics['f1']:.6f}"
        )
    if len(history) != protocol.epoch_limit or best_state is None:
        raise RuntimeError("Frozen training did not complete all five epochs.")
    history_frame = pd.DataFrame(history)
    selected_epoch = select_best_validation_epoch(history_frame)
    if selected_epoch != best_epoch:
        raise RuntimeError("Saved best checkpoint disagrees with strict validation-F1 selection.")
    model.load_state_dict(best_state)
    validation_labels, validation_probabilities = _predict_loader(model, val_loader, device)
    return FrozenTrainingResult(
        model=model,
        history=history_frame,
        best_epoch=best_epoch,
        best_validation_f1=best_f1,
        validation_labels=validation_labels,
        validation_probabilities=validation_probabilities,
    )


RESULT_METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc")


def _verify_result_grid(results: pd.DataFrame) -> None:
    required = {"variant", "seed", *RESULT_METRICS}
    missing = sorted(required.difference(results.columns))
    if missing:
        raise ValueError(f"Result grid is missing columns: {missing}")
    identities = list(zip(results["variant"].astype(str), results["seed"].astype(int)))
    expected = [(variant, seed) for variant in VARIANTS for seed in SEEDS]
    if len(identities) != len(set(identities)):
        raise ValueError("Result grid contains a duplicate variant/seed identity.")
    if set(identities) != set(expected):
        raise ValueError("Result grid does not contain the exact frozen variant/seed grid.")


def summarize_per_seed_results(results: pd.DataFrame) -> pd.DataFrame:
    _verify_result_grid(results)
    records = []
    for variant in VARIANTS:
        selected = results.loc[results["variant"] == variant].sort_values("seed")
        record: dict[str, object] = {"variant": variant, "n_seeds": int(len(selected))}
        for metric in RESULT_METRICS:
            values = selected[metric].to_numpy(dtype=np.float64)
            record[f"{metric}_mean"] = float(values.mean())
            record[f"{metric}_std"] = float(values.std(ddof=1))
        records.append(record)
    return pd.DataFrame(records)


def build_paired_deltas(results: pd.DataFrame) -> pd.DataFrame:
    _verify_result_grid(results)
    indexed = results.set_index(["variant", "seed"])
    records = []
    for comparison, baseline, candidate in (
        ("A_to_B", "A", "B"),
        ("B_to_C", "B", "C"),
        ("A_to_C", "A", "C"),
    ):
        for metric in RESULT_METRICS:
            deltas = np.asarray(
                [
                    float(indexed.loc[(candidate, seed), metric])
                    - float(indexed.loc[(baseline, seed), metric])
                    for seed in SEEDS
                ],
                dtype=np.float64,
            )
            records.append(
                {
                    "comparison": comparison,
                    "baseline": baseline,
                    "candidate": candidate,
                    "metric": metric,
                    "delta_mean": float(deltas.mean()),
                    "delta_std": float(deltas.std(ddof=1)),
                    "favorable_all_seeds": bool(np.all(deltas > 0)),
                    "noninferior_all_seeds": bool(np.all(deltas >= -NONINFERIORITY_TOLERANCE)),
                    "delta_signs": ",".join("+" if value > 0 else "-" if value < 0 else "0" for value in deltas),
                    **{f"seed_{seed}_delta": float(value) for seed, value in zip(SEEDS, deltas)},
                }
            )
    return pd.DataFrame(records)


def classify_final_outcome(results: pd.DataFrame) -> str:
    _verify_result_grid(results)
    indexed = results.set_index(["variant", "seed"])

    def deltas(candidate: str, baseline: str, metric: str) -> np.ndarray:
        return np.asarray(
            [
                float(indexed.loc[(candidate, seed), metric])
                - float(indexed.loc[(baseline, seed), metric])
                for seed in SEEDS
            ],
            dtype=np.float64,
        )

    bc_pr = deltas("C", "B", "pr_auc")
    bc_f1 = deltas("C", "B", "f1")
    ab_pr = deltas("B", "A", "pr_auc")
    ab_f1 = deltas("B", "A", "f1")
    tolerance = NONINFERIORITY_TOLERANCE
    if np.all(bc_pr > 0) and np.all(bc_f1 > 0):
        return "strong_biu_replacement_success"
    if np.all(bc_pr > 0) and np.all(bc_f1 >= -tolerance):
        return "biu_ranking_success"
    if np.all(bc_pr >= -tolerance) and np.all(bc_f1 >= -tolerance):
        return "biu_compact_representation_success"
    consistently_negative = np.all(bc_pr < -tolerance) and np.all(bc_f1 < -tolerance)
    if consistently_negative and np.all(ab_pr > 0) and np.all(ab_f1 > 0):
        return "vitpose_source_success_only"
    if consistently_negative:
        return "negative_biu_result"
    return "neutral"


def build_error_transition_counts(predictions: pd.DataFrame) -> pd.DataFrame:
    required = {"sequence_id", "variant", "seed", "label", "prediction", "probability"}
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(f"Prediction table is missing transition columns: {missing}")
    records = []
    for seed in sorted(predictions["seed"].astype(int).unique()):
        for comparison, baseline, candidate in (("A_to_B", "A", "B"), ("B_to_C", "B", "C")):
            base = predictions.loc[
                (predictions["seed"] == seed) & (predictions["variant"] == baseline)
            ].sort_values("sequence_id").reset_index(drop=True)
            cand = predictions.loc[
                (predictions["seed"] == seed) & (predictions["variant"] == candidate)
            ].sort_values("sequence_id").reset_index(drop=True)
            if base["sequence_id"].tolist() != cand["sequence_id"].tolist():
                raise ValueError(f"Prediction sequence alignment drifted for {comparison} seed {seed}.")
            if not np.array_equal(base["label"].to_numpy(), cand["label"].to_numpy()):
                raise ValueError(f"Prediction labels drifted for {comparison} seed {seed}.")
            base_correct = base["prediction"].to_numpy(dtype=np.int64) == base["label"].to_numpy(dtype=np.int64)
            candidate_correct = cand["prediction"].to_numpy(dtype=np.int64) == cand["label"].to_numpy(dtype=np.int64)
            masks = {
                "wrong_to_correct": (~base_correct) & candidate_correct,
                "correct_to_wrong": base_correct & (~candidate_correct),
                "wrong_to_wrong": (~base_correct) & (~candidate_correct),
                "correct_to_correct": base_correct & candidate_correct,
            }
            for transition, mask in masks.items():
                records.append(
                    {
                        "comparison": comparison,
                        "seed": int(seed),
                        "transition": transition,
                        "count": int(mask.sum()),
                        "rate": float(mask.mean()) if len(mask) else float("nan"),
                    }
                )
    return pd.DataFrame(records)


def select_effective_cohort(
    candidate_rows: pd.DataFrame,
    train_sequence_order: list[str],
) -> pd.DataFrame:
    if candidate_rows["sequence_id"].duplicated().any():
        raise ValueError("Candidate protocol contains duplicate sequence IDs.")
    indexed = candidate_rows.set_index("sequence_id", drop=False)
    missing = [sequence_id for sequence_id in train_sequence_order if sequence_id not in indexed.index]
    if missing:
        raise ValueError(f"Frozen training sequence is absent from candidate protocol: {missing[0]}")
    train = indexed.loc[train_sequence_order].copy()
    if not (train["split"] == "train").all():
        raise ValueError("Frozen training order references a non-training sample.")
    val = candidate_rows.loc[candidate_rows["split"] == "val"].copy()
    test = candidate_rows.loc[candidate_rows["split"] == "test"].copy()
    selected = pd.concat([train.reset_index(drop=True), val, test], ignore_index=True)
    if selected["sequence_id"].duplicated().any():
        raise ValueError("Effective cohort contains duplicate sequence IDs.")
    selected["exp047_index"] = np.arange(len(selected), dtype=np.int64)
    return selected


def load_effective_cohort() -> pd.DataFrame:
    candidates = pd.read_csv(EXP032_PROTOCOL_ROWS)
    verify_protocol_rows(candidates, expected_counts={"train": 333_454, "val": 77_288, "test": 271_078})
    with TRAIN_INDEX_PATH.open("rb") as handle:
        payload = pickle.load(handle)
    if int(payload.get("seq_len", -1)) != SEQ_LEN or int(payload.get("pred_len", -1)) != PRED_LEN:
        raise ValueError("Frozen train index task dimensions do not match Exp047.")
    tuples = payload.get("seq_index")
    if not isinstance(tuples, list) or len(tuples) != EXPECTED_SPLIT_COUNTS["train"]:
        raise ValueError("Frozen train index does not contain exactly 109,934 rows.")
    train_order = [f"{set_id}|{video_id}|{str(ped_id)}|{int(start_frame)}" for set_id, video_id, ped_id, start_frame, _ in tuples]
    rows = select_effective_cohort(candidates, train_order)
    verify_protocol_rows(rows, expected_counts=EXPECTED_SPLIT_COUNTS)
    expected_labels = np.asarray([int(item[4]) for item in tuples], dtype=np.int64)
    actual_labels = rows.loc[rows["split"] == "train", "label"].to_numpy(dtype=np.int64)
    if not np.array_equal(expected_labels, actual_labels):
        raise ValueError("Frozen train-index labels disagree with protocol labels.")
    return rows


def summarize_pose_coverage(
    windows_by_split: Mapping[str, list[np.ndarray]],
    source: str,
) -> pd.DataFrame:
    records = []
    for split, windows in windows_by_split.items():
        fractions = np.asarray([pose_joint_fraction(window) for window in windows], dtype=np.float64)
        valid = np.asarray([pose_window_is_valid(window) for window in windows], dtype=bool)
        n = int(len(windows))
        records.append(
            {
                "source": source,
                "split": split,
                "n_windows": n,
                "all_zero_window_count": int((~valid).sum()),
                "all_zero_window_rate": float((~valid).mean()) if n else float("nan"),
                "valid_window_count": int(valid.sum()),
                "valid_window_rate": float(valid.mean()) if n else float("nan"),
                "mean_valid_joint_fraction": float(fractions.mean()) if n else float("nan"),
                "median_valid_joint_fraction": float(np.median(fractions)) if n else float("nan"),
                "completely_invalid_window_count": int((~valid).sum()),
            }
        )
    return pd.DataFrame(records)


class Exp012PoseStore:
    """Exact frame-keyed view of the SHA-pinned historical pose JSONL."""

    def __init__(self, records: dict[tuple[str, str, str, int], np.ndarray]) -> None:
        self._records = records

    @classmethod
    def load(cls, path: Path = EXP012_POSE_PATH) -> "Exp012PoseStore":
        records: dict[tuple[str, str, str, int], np.ndarray] = {}
        with Path(path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                payload = json.loads(line)
                key = (
                    str(payload["set"]),
                    str(payload["video"]),
                    str(payload["pid"]),
                    int(payload["frame_idx"]),
                )
                if key in records:
                    raise ValueError(f"Duplicate Exp012 pose identity at line {line_number}: {key}")
                coords = np.asarray(payload["keypoints"], dtype=np.float32)
                if coords.shape != (17, 2):
                    raise ValueError(f"Malformed Exp012 keypoints at line {line_number}: {coords.shape}")
                if not np.isfinite(coords).all():
                    raise ValueError(f"Non-finite Exp012 keypoints at line {line_number}: {key}")
                records[key] = coords
        if not records:
            raise ValueError(f"Exp012 pose source is empty: {path}")
        return cls(records)

    @classmethod
    def load_allowed_sets(cls, path: Path, allowed_sets: set[str]) -> "Exp012PoseStore":
        allowed = {str(value) for value in allowed_sets}
        if not allowed:
            raise ValueError("At least one Exp012 set must be allowed.")
        records: dict[tuple[str, str, str, int], np.ndarray] = {}
        set_prefix = re.compile(r'^\{"set"\s*:\s*"([^"]+)"')
        with Path(path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                match = set_prefix.match(line)
                if match is None:
                    raise ValueError(f"Cannot identify Exp012 set before decoding line {line_number}.")
                if match.group(1) not in allowed:
                    continue
                payload = json.loads(line)
                key = (
                    str(payload["set"]),
                    str(payload["video"]),
                    str(payload["pid"]),
                    int(payload["frame_idx"]),
                )
                if key in records:
                    raise ValueError(f"Duplicate Exp012 pose identity at line {line_number}: {key}")
                coords = np.asarray(payload["keypoints"], dtype=np.float32)
                if coords.shape != (17, 2) or not np.isfinite(coords).all():
                    raise ValueError(f"Malformed Exp012 keypoints at line {line_number}: {coords.shape}")
                records[key] = coords
        if not records:
            raise ValueError(f"Exp012 pose source has no rows for allowed sets {sorted(allowed)}.")
        return cls(records)

    def get_pose(self, set_id: str, video_id: str, ped_id: str, frame: int) -> np.ndarray | None:
        return self._records.get((str(set_id), str(video_id), str(ped_id), int(frame)))

    def get_window(
        self,
        set_id: str,
        video_id: str,
        ped_id: str,
        frames: list[int],
    ) -> tuple[np.ndarray, list[int]]:
        window = np.zeros((len(frames), 17, 2), dtype=np.float32)
        missing = []
        for index, frame in enumerate(frames):
            pose = self.get_pose(set_id, video_id, ped_id, frame)
            if pose is None:
                missing.append(int(frame))
            else:
                window[index] = pose
        return window, missing

    def __len__(self) -> int:
        return len(self._records)


class YoloPoseStore:
    def __init__(self, root: Path = YOLO_POSE_ROOT) -> None:
        self.root = Path(root)
        self._videos: dict[tuple[str, str], dict[int, dict[str, np.ndarray]]] = {}

    def _video(self, set_id: str, video_id: str) -> dict[int, dict[str, np.ndarray]]:
        key = (str(set_id), str(video_id))
        if key not in self._videos:
            path = self.root / key[0] / f"{key[0]}_{key[1]}_poses.pkl"
            with path.open("rb") as handle:
                payload = pickle.load(handle)
            nested = f"{key[0]}_{key[1]}"
            if nested not in payload:
                raise KeyError(f"YOLO pose payload lacks {nested}: {path}")
            self._videos[key] = payload[nested]
        return self._videos[key]

    def get_window(
        self,
        set_id: str,
        video_id: str,
        ped_id: str,
        frames: list[int],
    ) -> tuple[np.ndarray, list[int]]:
        video = self._video(set_id, video_id)
        window = np.zeros((len(frames), 17, 2), dtype=np.float32)
        missing = []
        for index, frame in enumerate(frames):
            raw = video.get(int(frame), {}).get(str(ped_id))
            if raw is None:
                missing.append(int(frame))
                continue
            pose = np.asarray(raw, dtype=np.float32)
            if pose.shape != (34,):
                missing.append(int(frame))
                continue
            window[index] = pose.reshape(17, 2)
        return window, missing


def load_bbox_window(
    row: pd.Series | object,
    pie_db: Mapping[str, object],
) -> tuple[np.ndarray, list[int]]:
    values = row._asdict() if hasattr(row, "_asdict") else dict(row)
    ped = pie_db[str(values["set_id"])][str(values["video_id"])]["ped_annotations"][str(values["ped_id"])]
    frame_to_index = {int(frame): index for index, frame in enumerate(ped.get("frames", []))}
    frames = list(range(int(values["start_frame"]), int(values["end_frame"]) + 1))
    bbox = np.zeros((len(frames), 4), dtype=np.float32)
    missing = []
    source = ped.get("bbox", [])
    for out_index, frame in enumerate(frames):
        source_index = frame_to_index.get(frame)
        if source_index is None or source_index >= len(source):
            missing.append(frame)
            continue
        candidate = np.asarray(source[source_index], dtype=np.float32)
        if candidate.shape != (4,) or not np.isfinite(candidate).all():
            missing.append(frame)
            continue
        bbox[out_index] = candidate
    return bbox, missing


def _one_hot(index: int, num_classes: int) -> np.ndarray:
    value = int(index)
    if value < 0 or value >= num_classes:
        value = 0
    result = np.zeros(num_classes, dtype=np.float32)
    result[value] = 1.0
    return result


def build_static_context(attributes: Mapping[str, object]) -> np.ndarray:
    lane_value = attributes.get("num_lanes", 2)
    lane_category = LANE_CATEGORIES.get(lane_value, LANE_CATEGORIES[max(LANE_CATEGORIES)])
    vector = np.concatenate(
        [
            _one_hot(int(attributes.get("signalized", 0)), NUM_SIGNALIZED_CATS),
            _one_hot(int(attributes.get("intersection", 0)), NUM_INTERSECTION_CATS),
            _one_hot(int(attributes.get("age", 2)), NUM_AGE_CATS),
            _one_hot(int(attributes.get("gender", 0)), NUM_GENDER_CATS),
            _one_hot(int(attributes.get("traffic_direction", 0)), NUM_TRAFFIC_DIR_CATS),
            _one_hot(int(lane_category), NUM_LANE_CATS),
        ]
    ).astype(np.float32)
    if vector.shape != (STREAM_DIMS["static_context"],):
        raise RuntimeError(f"Static context shape mismatch: {vector.shape}")
    return vector


def compute_train_scalers(pie_db: Mapping[str, object]) -> dict[str, float]:
    values: dict[str, list[float]] = {"ego_speed": [], "accX": [], "accY": []}
    for set_id in TRAIN_SETS:
        for video in pie_db.get(set_id, {}).values():
            for ego in video.get("vehicle_annotations", {}).values():
                values["ego_speed"].append(float(ego.get("OBD_speed", 0.0) or ego.get("GPS_speed", 0.0)))
                values["accX"].append(float(ego.get("accX", 0.0)))
                values["accY"].append(float(ego.get("accY", 0.0)))
    result: dict[str, float] = {}
    for source, output in (("ego_speed", "ego_speed"), ("accX", "accX"), ("accY", "accY")):
        array = np.asarray(values[source], dtype=np.float64)
        if array.size == 0:
            raise ValueError(f"Cannot compute frozen scaler {source} from an empty training source.")
        result[f"{output}_mean"] = float(array.mean())
        result[f"{output}_std"] = float(max(array.std(), 1e-6))
    return result


def load_pie_sets_from_xml(
    set_ids: tuple[str, ...],
    *,
    data_root: Path = PIE_ANNOTATION_ROOT,
    parser: object | None = None,
    training_phase: bool = False,
) -> dict[str, object]:
    requested = tuple(str(set_id) for set_id in set_ids)
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("PIE XML loader requires a nonempty unique set list.")
    if any(set_id not in TRAIN_SETS + VAL_SETS + TEST_SETS for set_id in requested):
        raise ValueError(f"PIE XML loader received an unknown set: {requested}")
    if training_phase and "set03" in requested:
        raise ValueError("Training-phase PIE XML loading may not request set03.")
    root = Path(data_root)
    if parser is None:
        spec = importlib.util.spec_from_file_location("exp047_frozen_pie_parser", PIE_PARSER_PATH)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot import frozen PIE parser: {PIE_PARSER_PATH}")
        module = importlib.util.module_from_spec(spec)
        inserted_cv2_stub = importlib.util.find_spec("cv2") is None and "cv2" not in sys.modules
        if inserted_cv2_stub:
            sys.modules["cv2"] = types.ModuleType("cv2")
        try:
            spec.loader.exec_module(module)
        finally:
            if inserted_cv2_stub:
                sys.modules.pop("cv2", None)
        parser = module.PIE(regen_database=True, data_path=str(root))
    database: dict[str, object] = {}
    for set_id in requested:
        directory = root / "annotations" / set_id
        video_ids = sorted(
            path.name.removesuffix("_annt.xml")
            for path in directory.glob("*_annt.xml")
        )
        if not video_ids:
            raise FileNotFoundError(f"No PIE annotation XML files found for {set_id}: {directory}")
        videos: dict[str, object] = {}
        for video_id in video_ids:
            annotations = parser._get_annotations(set_id, video_id)
            attributes = parser._get_ped_attributes(set_id, video_id)
            annotations["vehicle_annotations"] = parser._get_vehicle_attributes(set_id, video_id)
            for ped_id in annotations["ped_annotations"]:
                if ped_id not in attributes:
                    raise KeyError(f"PIE attributes missing for {set_id}/{video_id}/{ped_id}")
                annotations["ped_annotations"][ped_id]["attributes"] = attributes[ped_id]
            videos[video_id] = annotations
        database[set_id] = videos
    return database


def extract_mslstm_stream_window(
    row: pd.Series | object,
    *,
    pie_db: Mapping[str, object],
    yolo_store: object,
    exp012_store: object,
    scalers: Mapping[str, float],
) -> dict[str, np.ndarray | bool]:
    values = row._asdict() if hasattr(row, "_asdict") else dict(row)
    set_id = str(values["set_id"])
    video_id = str(values["video_id"])
    ped_id = str(values["ped_id"])
    frames = list(range(int(values["start_frame"]), int(values["end_frame"]) + 1))
    if len(frames) != SEQ_LEN:
        raise ValueError(f"Sequence {values['sequence_id']} does not have exactly {SEQ_LEN} frames.")
    width = int(values["frame_width"])
    height = int(values["frame_height"])
    video = pie_db[set_id][video_id]
    ped = video["ped_annotations"][ped_id]
    ego_by_frame = video.get("vehicle_annotations", {})
    frame_to_index = {int(frame): index for index, frame in enumerate(ped.get("frames", []))}
    bbox_pixels, _ = load_bbox_window(values, pie_db)
    bbox = np.zeros((SEQ_LEN, STREAM_DIMS["bbox"]), dtype=np.float32)
    for index, (x1, y1, x2, y2) in enumerate(bbox_pixels):
        cx = ((x1 + x2) / 2.0) / width
        cy = ((y1 + y2) / 2.0) / height
        box_width = (x2 - x1) / width
        box_height = (y2 - y1) / height
        if box_width > 0 and box_height > 0 and 0 <= cx <= 1 and 0 <= cy <= 1:
            bbox[index] = np.asarray([cx, cy, box_width, box_height], dtype=np.float32)

    yolo_xy, _ = yolo_store.get_window(set_id, video_id, ped_id, frames)
    exp012_xy, exp012_missing = exp012_store.get_window(set_id, video_id, ped_id, frames)
    if exp012_missing:
        raise ValueError(f"Exp012 training pose is missing for {values['sequence_id']}: {exp012_missing[:3]}")
    yolo_pose = np.asarray(yolo_xy, dtype=np.float32).reshape(SEQ_LEN, 34)
    exp012_pose = np.vstack(
        [normalize_exp012_pose(frame, width=width, height=height) for frame in exp012_xy]
    ).astype(np.float32)
    action = np.zeros((SEQ_LEN, 1), dtype=np.float32)
    look = np.zeros((SEQ_LEN, 1), dtype=np.float32)
    ego_speed = np.zeros((SEQ_LEN, 1), dtype=np.float32)
    ego_acc = np.zeros((SEQ_LEN, 2), dtype=np.float32)
    action_source = ped.get("behavior", {}).get("action", [])
    look_source = ped.get("behavior", {}).get("look", [])
    for index, frame in enumerate(frames):
        ped_index = frame_to_index.get(frame, -1)
        ego = ego_by_frame.get(frame, {})
        speed = float(ego.get("OBD_speed", 0.0) or ego.get("GPS_speed", 0.0))
        ego_speed[index, 0] = (speed - scalers["ego_speed_mean"]) / scalers["ego_speed_std"]
        ego_acc[index, 0] = (float(ego.get("accX", 0.0)) - scalers["accX_mean"]) / scalers["accX_std"]
        ego_acc[index, 1] = (float(ego.get("accY", 0.0)) - scalers["accY_mean"]) / scalers["accY_std"]
        if ped_index >= 0:
            if ped_index < len(action_source):
                action[index, 0] = float(action_source[ped_index])
            if ped_index < len(look_source):
                look[index, 0] = float(look_source[ped_index])
    return {
        "bbox": bbox,
        "yolo_pose": yolo_pose,
        "exp012_pose": exp012_pose,
        "ped_action": action,
        "ped_look": look,
        "ego_speed": ego_speed,
        "ego_acc": ego_acc,
        "static_context": build_static_context(ped.get("attributes", {})),
        "yolo_pose_valid": pose_window_is_valid(np.asarray(yolo_xy).reshape(SEQ_LEN, 17, 2)),
        "exp012_pose_valid": pose_window_is_valid(np.asarray(exp012_xy).reshape(SEQ_LEN, 17, 2)),
    }


@dataclass(frozen=True)
class PoseAuditResult:
    coverage: pd.DataFrame
    fallback: pd.DataFrame
    biu_feature_names: tuple[str, ...]
    n_rows: int


def compute_pose_audit(
    rows: pd.DataFrame,
    *,
    yolo_store: object,
    exp012_store: object,
    pie_db: Mapping[str, object],
    biu_cache_path: Path,
    progress_every: int = 5_000,
    progress: Callable[[str], None] = print,
) -> PoseAuditResult:
    if rows.empty:
        raise ValueError("Pose audit requires at least one row.")
    destination = Path(biu_cache_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    cache = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32, shape=(len(rows), BIU_DIM))
    fractions: dict[tuple[str, str], list[float]] = {}
    valid: dict[tuple[str, str], list[bool]] = {}
    fallback_counts: dict[str, int] = {}
    split_counts: dict[str, int] = {}
    feature_names: tuple[str, ...] | None = None
    try:
        for position, row in enumerate(rows.itertuples(index=False)):
            frames = list(range(int(row.start_frame), int(row.end_frame) + 1))
            if len(frames) != SEQ_LEN:
                raise ValueError(f"Sequence {row.sequence_id} does not contain {SEQ_LEN} observation frames.")
            yolo_window, _ = yolo_store.get_window(row.set_id, row.video_id, row.ped_id, frames)
            exp_window, exp_missing = exp012_store.get_window(row.set_id, row.video_id, row.ped_id, frames)
            if exp_missing:
                raise ValueError(f"Exp012 pose frames are absent for {row.sequence_id}: {exp_missing[:3]}")
            bbox, _ = load_bbox_window(row, pie_db)
            for source, window in (("yolo", yolo_window), ("exp012", exp_window)):
                key = (str(row.split), source)
                fractions.setdefault(key, []).append(pose_joint_fraction(window))
                valid.setdefault(key, []).append(pose_window_is_valid(window))
            values, names, direction_source = extract_frozen_biu(
                exp_window,
                bbox,
                (int(row.frame_width), int(row.frame_height)),
            )
            current_names = tuple(names)
            if feature_names is None:
                feature_names = current_names
            elif current_names != feature_names:
                raise RuntimeError(f"BIU schema drift at {row.sequence_id}.")
            cache[position] = values
            split = str(row.split)
            split_counts[split] = split_counts.get(split, 0) + 1
            if direction_source == "bbox_center_x":
                fallback_counts[split] = fallback_counts.get(split, 0) + 1
            if progress_every > 0 and ((position + 1) % progress_every == 0 or position + 1 == len(rows)):
                progress(f"pose_audit rows={position + 1}/{len(rows)} split={split}")
        cache.flush()
        del cache
        temporary.replace(destination)
    except BaseException:
        del cache
        temporary.unlink(missing_ok=True)
        raise
    coverage_rows = []
    for (split, source), source_fractions in fractions.items():
        array = np.asarray(source_fractions, dtype=np.float64)
        source_valid = np.asarray(valid[(split, source)], dtype=bool)
        coverage_rows.append(
            {
                "source": source,
                "split": split,
                "n_windows": int(len(array)),
                "all_zero_window_count": int((~source_valid).sum()),
                "all_zero_window_rate": float((~source_valid).mean()),
                "valid_window_count": int(source_valid.sum()),
                "valid_window_rate": float(source_valid.mean()),
                "mean_valid_joint_fraction": float(array.mean()),
                "median_valid_joint_fraction": float(np.median(array)),
                "completely_invalid_window_count": int((~source_valid).sum()),
            }
        )
    fallback_rows = []
    for split, count in split_counts.items():
        fallback = int(fallback_counts.get(split, 0))
        fallback_rows.append(
            {
                "split": split,
                "n_windows": int(count),
                "fallback_count": fallback,
                "fallback_rate": float(fallback / count),
            }
        )
    if feature_names is None:
        raise RuntimeError("Pose audit produced no BIU schema.")
    return PoseAuditResult(
        coverage=pd.DataFrame(coverage_rows),
        fallback=pd.DataFrame(fallback_rows),
        biu_feature_names=feature_names,
        n_rows=int(len(rows)),
    )
