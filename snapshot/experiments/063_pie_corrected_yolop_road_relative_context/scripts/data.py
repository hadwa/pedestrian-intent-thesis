"""Exact-key scene alignment and train-only context standardization for Exp063."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .contracts import ExperimentConfig, assert_development_path, sha256_file
from .extract_trainval import EXP062_ROOT, load_authenticated_rows


EXP_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARTS_ROOT = EXP_ROOT / "cache/trainval_parts"
EXP062_EGO = EXP062_ROOT / "staged_sources/ego.npy"


@dataclass(frozen=True)
class TrainValContextData:
    """The aligned population shared by every fresh Exp063 run."""

    rows: pd.DataFrame
    labels: np.ndarray
    train_mask: np.ndarray
    val_mask: np.ndarray
    ego: np.ndarray
    scene: np.ndarray
    representation: str


@dataclass(frozen=True)
class ContextScaler:
    """Train-row-only channel statistics for one scene representation."""

    ego_mean: np.ndarray
    ego_std: np.ndarray
    scene_mean: np.ndarray
    scene_std: np.ndarray


def align_scene_parts(
    rows: pd.DataFrame,
    parts_root: Path,
    representation: str,
) -> np.ndarray:
    """Align video parts to the authenticated row manifest without dropping IDs."""
    if representation not in ExperimentConfig().representations:
        raise ValueError(f"Unknown Exp063 representation: {representation}")
    parts_root = assert_development_path(parts_root)
    required_columns = {"sequence_id", "set_id", "video_id"}
    if not required_columns.issubset(rows.columns):
        raise ValueError("Scene alignment rows are missing sequence/video identity")
    requested_ids = rows["sequence_id"].astype(str).to_numpy()
    if len(set(requested_ids)) != len(requested_ids):
        raise ValueError("Scene alignment row identity contains duplicates")
    values_by_id: dict[str, np.ndarray] = {}
    for (set_id, video_id), video_rows in rows.groupby(
        ["set_id", "video_id"], sort=False
    ):
        path = assert_development_path(parts_root / f"{set_id}_{video_id}.npz")
        if not path.is_file():
            raise FileNotFoundError(f"Missing Exp063 scene part: {path}")
        with np.load(path, allow_pickle=False) as bundle:
            part_ids = bundle["sequence_id"].astype(str)
            part_values = np.asarray(bundle[representation], dtype=np.float32)
        expected_ids = video_rows["sequence_id"].astype(str).to_numpy()
        if set(part_ids) != set(expected_ids) or len(set(part_ids)) != len(part_ids):
            raise ValueError(f"Scene part has duplicate or mismatched identity: {path}")
        if part_values.shape != (len(part_ids), 16, 20) or not np.isfinite(
            part_values
        ).all():
            raise ValueError(f"Scene part has malformed values: {path}")
        for sequence_id, value in zip(part_ids, part_values):
            if sequence_id in values_by_id:
                raise ValueError(f"Duplicate scene identity across parts: {sequence_id}")
            values_by_id[sequence_id] = value
    if set(values_by_id) != set(requested_ids):
        raise ValueError("Scene part identity does not equal the requested population")
    return np.stack([values_by_id[sequence_id] for sequence_id in requested_ids]).astype(
        np.float32, copy=False
    )


def fit_context_scaler(
    ego: np.ndarray,
    scene: np.ndarray,
    train_mask: np.ndarray,
) -> ContextScaler:
    """Fit per-channel means/stds using only original training rows and frames."""
    ego_values = np.asarray(ego, dtype=np.float32)
    scene_values = np.asarray(scene, dtype=np.float32)
    mask = np.asarray(train_mask, dtype=bool)
    if ego_values.shape[:2] != scene_values.shape[:2] or ego_values.shape[2:] != (3,):
        raise ValueError("Ego and scene arrays must share Nx16 with 3-D ego")
    if scene_values.shape[2:] != (20,) or mask.shape != (len(ego_values),) or not mask.any():
        raise ValueError("Scene/scaler arrays violate the frozen shape contract")
    if not np.isfinite(ego_values).all() or not np.isfinite(scene_values).all():
        raise ValueError("Context scaler inputs must be finite")
    ego_train = ego_values[mask]
    scene_train = scene_values[mask]
    ego_mean = ego_train.mean(axis=(0, 1))
    ego_std = ego_train.std(axis=(0, 1))
    scene_mean = scene_train.mean(axis=(0, 1))
    scene_std = scene_train.std(axis=(0, 1))
    ego_std = np.where(ego_std < 1e-8, 1.0, ego_std)
    scene_std = np.where(scene_std < 1e-8, 1.0, scene_std)
    return ContextScaler(ego_mean, ego_std, scene_mean, scene_std)


def transform_context(
    ego: np.ndarray,
    scene: np.ndarray,
    scaler: ContextScaler,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a frozen train-only scaler without refitting on validation."""
    transformed_ego = (np.asarray(ego, dtype=np.float32) - scaler.ego_mean) / scaler.ego_std
    transformed_scene = (
        np.asarray(scene, dtype=np.float32) - scaler.scene_mean
    ) / scaler.scene_std
    if not np.isfinite(transformed_ego).all() or not np.isfinite(transformed_scene).all():
        raise RuntimeError("Standardized context contains non-finite values")
    return transformed_ego.astype(np.float32), transformed_scene.astype(np.float32)


def load_trainval_context(
    representation: str,
    *,
    parts_root: Path = DEFAULT_PARTS_ROOT,
) -> TrainValContextData:
    """Load exact Exp062 rows/ego and one aligned corrected scene representation."""
    config = ExperimentConfig()
    rows = load_authenticated_rows()
    assert_development_path(EXP062_EGO)
    if not EXP062_EGO.is_file() or sha256_file(EXP062_EGO) != config.exp062_ego_sha256:
        raise RuntimeError("Exp062 ego source authentication failed")
    ego = np.load(EXP062_EGO, allow_pickle=False)
    scene = align_scene_parts(rows, parts_root, representation)
    if ego.shape != (len(rows), 16, 3) or scene.shape != (len(rows), 16, 20):
        raise RuntimeError("Exp063 aligned context shape drifted")
    split = rows["split"].astype(str)
    labels = rows["label"].to_numpy(dtype=np.int64)
    return TrainValContextData(
        rows=rows,
        labels=labels,
        train_mask=split.eq("train").to_numpy(),
        val_mask=split.eq("val").to_numpy(),
        ego=np.asarray(ego, dtype=np.float32),
        scene=scene,
        representation=representation,
    )


__all__ = [
    "ContextScaler",
    "TrainValContextData",
    "align_scene_parts",
    "fit_context_scaler",
    "load_trainval_context",
    "transform_context",
]
