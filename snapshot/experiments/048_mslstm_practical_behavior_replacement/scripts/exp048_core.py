"""Frozen contracts for Exp048's practical-behavior replacement study."""

from __future__ import annotations

import hashlib
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

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


EXPERIMENT_NUMBER = 48
SEQ_LEN = 30
PRED_LEN = 1
SEEDS = (42, 43, 44)
VARIANTS = ("A", "B", "C", "D", "E")

REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ROOT = REPO_ROOT / "experiments" / "048_mslstm_practical_behavior_replacement"
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
EXP012_POSE_SHA256 = "1f0907ebb33bb04c3a3434d9c89e7bd597e2b9e3d47fe30f488ffa9410054d87"
YOLOP_ROOT = Path("/home/abdelrahman/Hadwa/MSLSTM-PID/Extracted YOLOP")

VARIANT_STREAMS = {
    "A": ("bbox", "ped_action", "ped_look", "ego_speed", "ego_acc", "static_context", "yolop"),
    "B": ("bbox", "ped_action", "ped_look", "ego_speed", "ego_acc", "yolop"),
    "C": ("bbox", "ego_speed", "ego_acc", "yolop"),
    "D": ("bbox", "ego_speed", "ego_acc", "yolop", "pose"),
    "E": ("bbox", "ego_speed", "ego_acc", "yolop", "biu_v2"),
}
STREAM_DIMS = {
    "bbox": 4,
    "ped_action": 1,
    "ped_look": 1,
    "ego_speed": 1,
    "ego_acc": 2,
    "static_context": 23,
    "yolop": 20,
    "pose": 34,
    "biu_v2": 63,
}
LSTM_HIDDEN_SIZE = 256
NUM_LSTM_LAYERS = 2
STREAM_CONTEXT_DIM = 512
ATTENTION_DIM = 128
DROPOUT_RATE = 0.3
GAUSSIAN_NOISE_STD = 0.02


@dataclass(frozen=True)
class FrozenProtocol:
    sequence_length: int = SEQ_LEN
    prediction_length: int = PRED_LEN
    learning_rate: float = 1e-4
    batch_size: int = 32
    epoch_limit: int = 15
    fixed_threshold: float = 0.5
    checkpoint_rule: str = "validation_f1_at_fixed_0p5"
    native_train_rows: int = 333_454
    validation_rows: int = 77_288
    test_rows: int = 271_078
    oversampled_train_rows: int = 556_974
    gaussian_noise_std: float = GAUSSIAN_NOISE_STD


PROTOCOL = FrozenProtocol()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    if int(seed) not in SEEDS:
        raise ValueError(f"Seed must be one of {SEEDS}, got {seed}.")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class StreamAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STREAM_CONTEXT_DIM, ATTENTION_DIM),
            nn.Tanh(),
            nn.Linear(ATTENTION_DIM, 1),
        )

    def forward(self, sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.net(sequence).squeeze(2)
        weights = torch.softmax(scores, dim=1)
        return torch.sum(sequence * weights.unsqueeze(2), dim=1), weights


class MslstmPracticalModel(nn.Module):
    """Recovered adaptive-fusion MSLSTM with only the declared variant streams."""

    def __init__(self, variant: str) -> None:
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"Variant must be one of {VARIANTS}, got {variant!r}.")
        self.variant = variant
        self.streams = VARIANT_STREAMS[variant]
        self.lstms = nn.ModuleDict()
        self.attentions = nn.ModuleDict()
        for name in self.streams:
            if name == "biu_v2":
                continue
            self.lstms[name] = nn.LSTM(
                input_size=STREAM_DIMS[name],
                hidden_size=LSTM_HIDDEN_SIZE,
                num_layers=NUM_LSTM_LAYERS,
                batch_first=True,
                dropout=DROPOUT_RATE,
                bidirectional=True,
            )
            self.attentions[name] = StreamAttention()
        if variant == "E":
            self.biu_projection = nn.Sequential(
                nn.Linear(63, 128),
                nn.ReLU(),
                nn.Dropout(DROPOUT_RATE),
                nn.Linear(128, STREAM_CONTEXT_DIM),
            )
        else:
            self.biu_projection = None
        self.fusion_weights = nn.Parameter(torch.ones(len(self.streams)))
        self.dropout = nn.Dropout(DROPOUT_RATE)
        self.fc1 = nn.Linear(STREAM_CONTEXT_DIM, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 2)

    def forward(self, inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        contexts = []
        for name in self.streams:
            if name == "biu_v2":
                if self.biu_projection is None:
                    raise RuntimeError("BIU projection is unavailable.")
                contexts.append(self.biu_projection(inputs[name]))
            else:
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


def build_model(variant: str, seed: int) -> MslstmPracticalModel:
    seed_everything(seed)
    return MslstmPracticalModel(variant)


def count_parameters(model: nn.Module) -> tuple[int, int]:
    return (
        int(sum(parameter.numel() for parameter in model.parameters())),
        int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)),
    )


def pose_path_parameters(model: MslstmPracticalModel, variant: str) -> int:
    if variant == "D":
        return int(
            sum(parameter.numel() for parameter in model.lstms["pose"].parameters())
            + sum(parameter.numel() for parameter in model.attentions["pose"].parameters())
        )
    if variant == "E" and model.biu_projection is not None:
        return int(sum(parameter.numel() for parameter in model.biu_projection.parameters()))
    return 0


class YolopStore:
    """Fail-closed reader for historical per-video 20-D YOLOP features."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self._cache: dict[tuple[str, str], Mapping[object, object]] = {}

    def _video(self, set_id: str, video_id: str) -> Mapping[object, object]:
        identity = (str(set_id), str(video_id))
        if identity not in self._cache:
            path = self.root / f"{identity[0]}_{identity[1]}_yolop_features.pkl"
            if not path.is_file():
                raise ValueError(f"missing exact YOLOP video artifact: {path}")
            with path.open("rb") as handle:
                payload = pickle.load(handle)
            key = f"{identity[0]}_{identity[1]}"
            if not isinstance(payload, Mapping) or key not in payload or not isinstance(payload[key], Mapping):
                raise ValueError(f"malformed YOLOP video artifact: {path}")
            self._cache[identity] = payload[key]
        return self._cache[identity]

    def get_window(
        self,
        set_id: str,
        video_id: str,
        track_id: str,
        frame_ids: Sequence[int],
    ) -> np.ndarray:
        video = self._video(set_id, video_id)
        output = []
        for frame_id in frame_ids:
            frame = video.get(int(frame_id), video.get(str(int(frame_id))))
            if not isinstance(frame, Mapping) or str(track_id) not in frame:
                raise ValueError(
                    f"missing exact YOLOP frame {set_id}/{video_id}/{track_id}/{frame_id}"
                )
            vector = np.asarray(frame[str(track_id)], dtype=np.float32)
            if vector.shape != (20,) or not np.isfinite(vector).all():
                raise ValueError(
                    f"YOLOP feature must be finite 20-D at "
                    f"{set_id}/{video_id}/{track_id}/{frame_id}"
                )
            output.append(vector)
        return np.stack(output).astype(np.float32, copy=False)

    def get_window_with_availability(
        self,
        set_id: str,
        video_id: str,
        track_id: str,
        frame_ids: Sequence[int],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Reproduce the notebook's zero-fill policy while exposing every miss."""
        video = self._video(set_id, video_id)
        output = np.zeros((len(frame_ids), 20), dtype=np.float32)
        available = np.zeros(len(frame_ids), dtype=np.bool_)
        for index, frame_id in enumerate(frame_ids):
            frame = video.get(int(frame_id), video.get(str(int(frame_id))))
            if not isinstance(frame, Mapping) or str(track_id) not in frame:
                continue
            vector = np.asarray(frame[str(track_id)], dtype=np.float32)
            if vector.shape != (20,) or not np.isfinite(vector).all():
                raise ValueError(
                    f"YOLOP feature must be finite 20-D at "
                    f"{set_id}/{video_id}/{track_id}/{frame_id}"
                )
            output[index] = vector
            available[index] = True
        return output, available


def build_oversampling_manifest(rows: pd.DataFrame, seed: int) -> pd.DataFrame:
    required = {"sequence_id", "split", "label", "cache_index"}
    missing = sorted(required.difference(rows.columns))
    if missing:
        raise ValueError(f"Training rows are missing columns: {missing}")
    source = rows.reset_index(drop=True).copy()
    if source.empty or source["split"].isin({"test", "set03"}).any():
        raise ValueError("Oversampling accepts nonempty training rows only, never test/set03.")
    if source["sequence_id"].duplicated().any() or set(source["label"].astype(int)) != {0, 1}:
        raise ValueError("Oversampling requires unique IDs and both binary classes.")
    rng = np.random.default_rng(int(seed))
    counts = source["label"].astype(int).value_counts()
    majority = int(counts.max())
    minority_label = int(counts.idxmin())
    minority_positions = np.flatnonzero(source["label"].to_numpy(dtype=int) == minority_label)
    duplicate_positions = rng.choice(minority_positions, size=majority - int(counts.min()), replace=True)
    records = []
    for position, row in source.iterrows():
        records.append(
            {
                "source_cache_index": int(row["cache_index"]),
                "sequence_id": str(row["sequence_id"]),
                "label": int(row["label"]),
                "augmented": False,
                "noise_seed": -1,
            }
        )
    for position in duplicate_positions:
        row = source.iloc[int(position)]
        records.append(
            {
                "source_cache_index": int(row["cache_index"]),
                "sequence_id": str(row["sequence_id"]),
                "label": int(row["label"]),
                "augmented": True,
                "noise_seed": int(rng.integers(0, np.iinfo(np.int32).max)),
            }
        )
    manifest = pd.DataFrame.from_records(records)
    return manifest.iloc[rng.permutation(len(manifest))].reset_index(drop=True)


def augment_measurements(
    inputs: Mapping[str, np.ndarray],
    *,
    noise_seed: int,
) -> dict[str, np.ndarray]:
    output = {name: np.array(value, copy=True) for name, value in inputs.items()}
    rng = np.random.default_rng(int(noise_seed))
    for name in ("bbox", "ego_speed", "ego_acc", "yolop", "exp012_pose"):
        if name not in output:
            continue
        value = np.asarray(output[name], dtype=np.float32)
        value = value + rng.normal(0.0, GAUSSIAN_NOISE_STD, size=value.shape).astype(np.float32)
        if name == "bbox":
            value[:, :2] = np.clip(value[:, :2], 0.01, 0.99)
            value[:, 2:] = np.clip(value[:, 2:], 0.01, 1.0)
        elif name == "exp012_pose":
            value = np.clip(value, 0.0, 1.0)
        output[name] = value.astype(np.float32, copy=False)
    return output


def biu_from_normalized_measurements(
    pose: np.ndarray,
    bbox: np.ndarray,
    *,
    width: int,
    height: int,
) -> np.ndarray:
    values, _, _ = biu_with_metadata_from_normalized_measurements(
        pose,
        bbox,
        width=width,
        height=height,
    )
    return values


def biu_with_metadata_from_normalized_measurements(
    pose: np.ndarray,
    bbox: np.ndarray,
    *,
    width: int,
    height: int,
) -> tuple[np.ndarray, tuple[str, ...], str]:
    normalized_pose = np.asarray(pose, dtype=np.float64)
    normalized_bbox = np.asarray(bbox, dtype=np.float64)
    if normalized_pose.shape != (SEQ_LEN, 34) or normalized_bbox.shape != (SEQ_LEN, 4):
        raise ValueError("BIU requires aligned normalized pose (30,34) and BBox (30,4).")
    coords = normalized_pose.reshape(SEQ_LEN, 17, 2).copy()
    coords[:, :, 0] *= float(width)
    coords[:, :, 1] *= float(height)
    cx, cy, bw, bh = normalized_bbox.T
    boxes = np.column_stack(
        (
            (cx - bw / 2.0) * float(width),
            (cy - bh / 2.0) * float(height),
            (cx + bw / 2.0) * float(width),
            (cy + bh / 2.0) * float(height),
        )
    )
    features = extract_biu_v2(
        coords,
        kp_conf=None,
        bbox_sequence=boxes,
        image_size=(int(width), int(height)),
        include_optional=False,
    )
    names = [
        name
        for name, value in features.items()
        if isinstance(value, (int, float, np.integer, np.floating))
    ]
    if len(names) != 63:
        raise RuntimeError(f"Frozen BIU-v2 must contain exactly 63 numeric features, got {len(names)}.")
    values = np.nan_to_num(
        np.asarray([features[name] for name in names], dtype=np.float64),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32)
    return values, tuple(names), str(features["direction_source"])


class Exp048Dataset(Dataset):
    """Aligned stream cache with variant-specific feature exposure."""

    def __init__(
        self,
        rows: pd.DataFrame,
        arrays: Mapping[str, np.ndarray],
        *,
        variant: str,
        sampling_manifest: pd.DataFrame | None = None,
        training_phase: bool = True,
    ) -> None:
        if variant not in VARIANTS:
            raise ValueError(f"Variant must be one of {VARIANTS}, got {variant!r}.")
        self.rows = rows.reset_index(drop=True).copy()
        if self.rows.empty:
            raise ValueError("Dataset cannot be empty.")
        if training_phase and self.rows["split"].isin({"test", "set03"}).any():
            raise ValueError("Training dataset must not contain test/set03 rows.")
        if self.rows["sequence_id"].duplicated().any():
            raise ValueError("Dataset row IDs must be unique.")
        self.arrays = arrays
        self.variant = variant
        self.sampling = None if sampling_manifest is None else sampling_manifest.reset_index(drop=True).copy()
        required_arrays = {
            "bbox",
            "ego_speed",
            "ego_acc",
            "yolop",
        }
        required_arrays.update(
            "exp012_pose" if name == "pose" else name for name in VARIANT_STREAMS[variant]
        )
        if variant == "E" and self.sampling is not None and self.sampling["augmented"].any():
            required_arrays.update({"exp012_pose", "biu_v2"})
        missing = sorted(required_arrays.difference(arrays))
        if missing:
            raise ValueError(f"Dataset arrays are missing: {missing}")
        self._by_cache_index = {
            int(row.cache_index): row for row in self.rows.itertuples(index=False)
        }

    def __len__(self) -> int:
        return len(self.sampling) if self.sampling is not None else len(self.rows)

    def __getitem__(self, index: int):
        if self.sampling is None:
            row = self.rows.iloc[int(index)]
            cache_index = int(row["cache_index"])
            augmented = False
            noise_seed = -1
        else:
            sampled = self.sampling.iloc[int(index)]
            cache_index = int(sampled["source_cache_index"])
            row = self._by_cache_index[cache_index]
            augmented = bool(sampled["augmented"])
            noise_seed = int(sampled["noise_seed"])
        source = {
            name: np.asarray(values[cache_index], dtype=np.float32)
            for name, values in self.arrays.items()
            if name in {
                "bbox",
                "ped_action",
                "ped_look",
                "ego_speed",
                "ego_acc",
                "static_context",
                "yolop",
                "exp012_pose",
                "biu_v2",
            }
        }
        if augmented:
            source = augment_measurements(source, noise_seed=noise_seed)
            if self.variant == "E":
                source["biu_v2"] = biu_from_normalized_measurements(
                    source["exp012_pose"],
                    source["bbox"],
                    width=int(getattr(row, "frame_width")),
                    height=int(getattr(row, "frame_height")),
                )
        features: dict[str, torch.Tensor] = {}
        for stream in VARIANT_STREAMS[self.variant]:
            array_name = "exp012_pose" if stream == "pose" else stream
            value = np.asarray(source[array_name], dtype=np.float32)
            if stream == "static_context" and value.shape == (23,):
                value = np.broadcast_to(value, (SEQ_LEN, 23))
            features[stream] = torch.as_tensor(value.copy(), dtype=torch.float32)
        label = torch.tensor(int(getattr(row, "label")), dtype=torch.long)
        meta = {
            "sequence_id": str(getattr(row, "sequence_id")),
            "split": str(getattr(row, "split")),
            "label": int(getattr(row, "label")),
            "cache_index": cache_index,
            "augmented": augmented,
        }
        return features, label, meta


def select_best_epoch(history: pd.DataFrame) -> int:
    if len(history) != PROTOCOL.epoch_limit:
        raise ValueError(f"Checkpoint selection requires exactly {PROTOCOL.epoch_limit} epochs.")
    required = {"epoch", "val_f1_at_0p5"}
    if not required.issubset(history.columns):
        raise ValueError(f"Checkpoint history requires columns {sorted(required)}.")
    scores = history["val_f1_at_0p5"].to_numpy(dtype=float)
    if not np.isfinite(scores).all():
        raise ValueError("Validation F1 history must be finite.")
    return int(history.iloc[int(np.argmax(scores))]["epoch"])


def compute_binary_metrics(y_true: np.ndarray, y_probability: np.ndarray) -> dict[str, float]:
    labels = np.asarray(y_true, dtype=np.int64)
    probabilities = np.asarray(y_probability, dtype=np.float64)
    if labels.ndim != 1 or probabilities.shape != labels.shape or not np.isfinite(probabilities).all():
        raise ValueError("Metrics require aligned finite one-dimensional labels and probabilities.")
    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("Metrics labels must be binary.")
    prediction = (probabilities >= PROTOCOL.fixed_threshold).astype(np.int64)
    ranking_defined = len(np.unique(labels)) == 2
    return {
        "threshold": PROTOCOL.fixed_threshold,
        "accuracy": float(accuracy_score(labels, prediction)),
        "precision": float(precision_score(labels, prediction, zero_division=0)),
        "recall": float(recall_score(labels, prediction, zero_division=0)),
        "f1": float(f1_score(labels, prediction, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probabilities)) if ranking_defined else float("nan"),
        "pr_auc": float(average_precision_score(labels, probabilities)) if ranking_defined else float("nan"),
    }


def build_behavior_recovery(per_seed_results: pd.DataFrame) -> pd.DataFrame:
    required = {"variant", "seed", "f1", "pr_auc"}
    if not required.issubset(per_seed_results.columns):
        raise ValueError(f"Per-seed results require columns {sorted(required)}.")
    output = []
    for metric in ("f1", "pr_auc"):
        pivot = per_seed_results.pivot(index="seed", columns="variant", values=metric)
        if tuple(sorted(pivot.index.astype(int))) != SEEDS or not set(VARIANTS).issubset(pivot.columns):
            raise ValueError("Behavior recovery requires the exact fresh A-E, three-seed grid.")
        gt_gain = pivot["B"] - pivot["C"]
        raw_recovery = pivot["D"] - pivot["C"]
        biu_recovery = pivot["E"] - pivot["C"]
        fractions = np.where(gt_gain.to_numpy() > 0, biu_recovery / gt_gain, np.nan)
        output.append(
            {
                "metric": metric,
                "gt_behavior_gain_mean": float(gt_gain.mean()),
                "raw_pose_recovery_mean": float(raw_recovery.mean()),
                "biu_recovery_mean": float(biu_recovery.mean()),
                "biu_recovery_fraction_mean": float(np.nanmean(fractions)),
            }
        )
    return pd.DataFrame(output)


def build_paired_deltas(per_seed_results: pd.DataFrame) -> pd.DataFrame:
    required = {"variant", "seed", "f1", "pr_auc"}
    if not required.issubset(per_seed_results.columns):
        raise ValueError(f"Per-seed results require columns {sorted(required)}.")
    indexed = per_seed_results.set_index(["variant", "seed"])
    expected = {(variant, seed) for variant in VARIANTS for seed in SEEDS}
    if set(indexed.index) != expected or not indexed.index.is_unique:
        raise ValueError("Paired deltas require the exact unique A-E, three-seed grid.")
    comparisons = (
        ("A_to_B", "A", "B"),
        ("B_to_C", "B", "C"),
        ("C_to_D", "C", "D"),
        ("C_to_E", "C", "E"),
        ("D_to_E", "D", "E"),
        ("A_to_E", "A", "E"),
    )
    rows = []
    for comparison, baseline, candidate in comparisons:
        for seed in SEEDS:
            source = indexed.loc[(baseline, seed)]
            target = indexed.loc[(candidate, seed)]
            rows.append(
                {
                    "comparison": comparison,
                    "baseline": baseline,
                    "candidate": candidate,
                    "seed": seed,
                    "f1_delta": float(target["f1"] - source["f1"]),
                    "pr_auc_delta": float(target["pr_auc"] - source["pr_auc"]),
                }
            )
    return pd.DataFrame(rows)


def authenticate_training_grid(records: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    expected = [(variant, seed) for variant in VARIANTS for seed in SEEDS]
    if len(records) != len(expected):
        raise ValueError("Final evaluation requires exactly 15 training records.")
    indexed: dict[tuple[str, int], dict[str, object]] = {}
    for source in records:
        record = dict(source)
        identity = (str(record.get("variant")), int(record.get("seed", -1)))
        if identity not in expected or identity in indexed:
            raise ValueError(f"Invalid or duplicate training identity: {identity}")
        checkpoint = Path(str(record.get("checkpoint_path", "")))
        if not checkpoint.is_file() or sha256_file(checkpoint) != str(record.get("checkpoint_sha256", "")):
            raise ValueError(f"Checkpoint hash mismatch for {identity}.")
        indexed[identity] = record
    if set(indexed) != set(expected):
        raise ValueError("Training records do not contain the exact 15-run grid.")
    return [indexed[identity] for identity in expected]
