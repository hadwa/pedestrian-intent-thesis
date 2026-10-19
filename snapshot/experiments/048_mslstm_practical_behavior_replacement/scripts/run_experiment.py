#!/usr/bin/env python3
"""Phase-oriented, resumable runner for the frozen Exp048 study."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_ROOT = Path(__file__).resolve().parent
CORE_PATH = SCRIPT_ROOT / "exp048_core.py"
for candidate in (REPO_ROOT, SCRIPT_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from exp048_core import (  # noqa: E402
    EXP012_POSE_PATH,
    EXP012_POSE_SHA256,
    EXP032_PROTOCOL_ROWS,
    EXPERIMENT_ROOT,
    PROTOCOL,
    SEEDS,
    VARIANTS,
    YOLOP_ROOT,
    Exp048Dataset,
    YolopStore,
    biu_with_metadata_from_normalized_measurements,
    build_behavior_recovery,
    build_model,
    build_oversampling_manifest,
    build_paired_deltas,
    compute_binary_metrics,
    count_parameters,
    pose_path_parameters,
    seed_everything,
    sha256_file,
)


EXP047_ROOT = REPO_ROOT / "experiments" / "047_mslstm_vitpose_biu_pose_replacement"
EXP047_CORE_PATH = EXP047_ROOT / "scripts" / "exp047_core.py"
EXP047_TRAINVAL_CACHE = EXP047_ROOT / "cache" / "trainval_streams"
EXP047_SET03_CACHE = EXP047_ROOT / "cache" / "set03_streams"
TRAIN_SETS = ("set01", "set02", "set04")
VAL_SETS = ("set05", "set06")
EXPECTED_TRAINVAL_COUNTS = {"train": PROTOCOL.native_train_rows, "val": PROTOCOL.validation_rows}
PREPARE_CACHE_NAME = "trainval_native_streams"

CACHE_ARRAY_SHAPES = {
    "bbox": (30, 4),
    "ped_action": (30, 1),
    "ped_look": (30, 1),
    "ego_speed": (30, 1),
    "ego_acc": (30, 2),
    "static_context": (23,),
    "yolop": (30, 20),
    "exp012_pose": (30, 34),
    "biu_v2": (63,),
    "exp012_pose_valid": (),
    "exp012_valid_joint_fraction": (),
    "bbox_direction_fallback": (),
    "yolop_valid_frame_fraction": (),
    "yolop_complete_window": (),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(payload: Mapping[str, object], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def progress_logger(path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def emit(message: str) -> None:
        line = f"{utc_now()} {message}"
        print(line, flush=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    return emit


def validate_trainval_rows(
    rows: pd.DataFrame,
    *,
    expected_counts: Mapping[str, int] = EXPECTED_TRAINVAL_COUNTS,
) -> pd.DataFrame:
    required = {
        "sequence_id",
        "split",
        "set_id",
        "label",
        "observation_length",
        "pred_len",
    }
    missing = sorted(required.difference(rows.columns))
    if missing:
        raise ValueError(f"Train/validation manifest is missing columns: {missing}")
    checked = rows.reset_index(drop=True).copy()
    if checked.empty or checked["split"].isin({"test", "set03"}).any():
        raise ValueError("Train/validation preparation must never contain test/set03 rows.")
    if set(checked["split"].astype(str)) != set(expected_counts):
        raise ValueError("Train/validation manifest has unexpected split names.")
    actual_counts = checked["split"].value_counts().astype(int).to_dict()
    if actual_counts != dict(expected_counts):
        raise ValueError(f"Train/validation counts drifted: {actual_counts}")
    if checked["sequence_id"].duplicated().any():
        raise ValueError("Train/validation sequence IDs are not unique.")
    if set(checked["label"].astype(int)) != {0, 1}:
        raise ValueError("Train/validation labels are not binary with both classes present.")
    if not (checked["observation_length"].astype(int) == 30).all():
        raise ValueError("Observation length drifted from 30 frames.")
    if not (checked["pred_len"].astype(int) == 1).all():
        raise ValueError("Recovered target drifted from the next raw frame.")
    permitted_sets = {
        "train": set(TRAIN_SETS),
        "val": set(VAL_SETS),
    }
    for split, allowed in permitted_sets.items():
        observed = set(checked.loc[checked["split"] == split, "set_id"].astype(str))
        if not observed.issubset(allowed):
            raise ValueError(f"{split} contains unexpected PIE sets: {sorted(observed - allowed)}")
    return checked


def build_reuse_index(native_rows: pd.DataFrame, reused_rows: pd.DataFrame) -> dict[str, int]:
    identity_columns = ("label", "start_frame", "end_frame", "target_frame")
    required = {"sequence_id", "cache_index", *identity_columns}
    if not required.issubset(reused_rows.columns) or not {"sequence_id", *identity_columns}.issubset(native_rows.columns):
        raise ValueError("Reuse identity columns are incomplete.")
    native = native_rows.set_index("sequence_id", drop=False)
    result: dict[str, int] = {}
    for row in reused_rows.itertuples(index=False):
        sequence_id = str(row.sequence_id)
        if sequence_id not in native.index:
            raise ValueError(f"Reused sequence is absent from native cohort: {sequence_id}")
        expected = native.loc[sequence_id]
        if any(int(getattr(row, name)) != int(expected[name]) for name in identity_columns):
            raise ValueError(f"Reused row identity disagrees with native cohort: {sequence_id}")
        if sequence_id in result:
            raise ValueError(f"Duplicate reused row identity: {sequence_id}")
        result[sequence_id] = int(row.cache_index)
    return result


def load_exp047_core():
    spec = importlib.util.spec_from_file_location("exp048_exp047_lineage", EXP047_CORE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import Exp047 lineage: {EXP047_CORE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_and_authenticate_exp047_cache() -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    metadata_path = EXP047_TRAINVAL_CACHE / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    artifacts = metadata.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("Exp047 cache lacks an artifact hash manifest.")
    for relative, expected_hash in artifacts.items():
        path = EXP047_TRAINVAL_CACHE / relative
        if not path.is_file() or sha256_file(path) != str(expected_hash):
            raise ValueError(f"Exp047 cache artifact hash mismatch: {path}")
    rows = pd.read_csv(EXP047_TRAINVAL_CACHE / "rows.csv")
    arrays = {
        name: np.load(EXP047_TRAINVAL_CACHE / f"{name}.npy", mmap_mode="r")
        for name in (
            "bbox",
            "ped_action",
            "ped_look",
            "ego_speed",
            "ego_acc",
            "static_context",
            "exp012_pose",
        )
    }
    return rows, arrays


class _UnusedYoloPoseStore:
    def get_window(self, set_id: str, video_id: str, ped_id: str, frames: list[int]):
        return np.zeros((len(frames), 17, 2), dtype=np.float32), list(frames)


def _array_dtype(name: str):
    return (
        np.bool_
        if name in {"exp012_pose_valid", "bbox_direction_fallback", "yolop_complete_window"}
        else np.float32
    )


def _open_partial_arrays(root: Path, n_rows: int, *, create: bool) -> dict[str, np.memmap]:
    result = {}
    for name, trailing in CACHE_ARRAY_SHAPES.items():
        path = root / f"{name}.npy"
        if create:
            result[name] = np.lib.format.open_memmap(
                path,
                mode="w+",
                dtype=_array_dtype(name),
                shape=(n_rows, *trailing),
            )
        else:
            array = np.load(path, mmap_mode="r+")
            if array.shape != (n_rows, *trailing) or array.dtype != _array_dtype(name):
                raise ValueError(f"Partial cache array contract drifted: {path}")
            result[name] = array
    return result


def _flush_arrays(arrays: Mapping[str, np.memmap]) -> None:
    for array in arrays.values():
        array.flush()


def _summarize_prepare(rows: pd.DataFrame, arrays: Mapping[str, np.ndarray]) -> tuple[pd.DataFrame, pd.DataFrame]:
    coverage = []
    fallback = []
    for split in ("train", "val"):
        mask = rows["split"].to_numpy() == split
        valid = np.asarray(arrays["exp012_pose_valid"])[mask].astype(bool)
        fraction = np.asarray(arrays["exp012_valid_joint_fraction"])[mask].astype(float)
        fallback_values = np.asarray(arrays["bbox_direction_fallback"])[mask].astype(bool)
        coverage.append(
            {
                "source": "historical_exp012_vitpose_s_lineage",
                "split": split,
                "n_windows": int(mask.sum()),
                "all_zero_window_count": int((~valid).sum()),
                "all_zero_window_rate": float((~valid).mean()),
                "valid_window_count": int(valid.sum()),
                "valid_window_rate": float(valid.mean()),
                "mean_valid_joint_fraction": float(fraction.mean()),
                "median_valid_joint_fraction": float(np.median(fraction)),
                "completely_invalid_window_count": int((~valid).sum()),
            }
        )
        fallback.append(
            {
                "split": split,
                "n_windows": int(mask.sum()),
                "fallback_count": int(fallback_values.sum()),
                "fallback_rate": float(fallback_values.mean()),
            }
        )
    return pd.DataFrame(coverage), pd.DataFrame(fallback)


def run_prepare(output_root: Path, *, resume: bool) -> None:
    output_root = Path(output_root)
    marker = output_root / ".prepare.complete.json"
    progress = progress_logger(output_root / "logs" / "prepare.log")
    if marker.exists():
        payload = json.loads(marker.read_text(encoding="utf-8"))
        for relative, expected_hash in payload.get("artifacts", {}).items():
            path = output_root / relative
            if not path.is_file() or sha256_file(path) != expected_hash:
                raise ValueError(f"Completed preparation artifact drifted: {path}")
        progress("phase=prepare status=already_complete_authenticated")
        return
    progress("phase=prepare status=starting set03_modalities_accessed=false")
    if sha256_file(EXP012_POSE_PATH) != EXP012_POSE_SHA256:
        raise ValueError("Frozen Exp012 pose SHA-256 drifted.")

    full_rows = pd.read_csv(EXP032_PROTOCOL_ROWS)
    trainval = full_rows.loc[full_rows["split"].isin({"train", "val"})].copy()
    trainval = pd.concat(
        [trainval.loc[trainval["split"] == "train"], trainval.loc[trainval["split"] == "val"]],
        ignore_index=True,
    )
    trainval = validate_trainval_rows(trainval)
    trainval["cache_index"] = np.arange(len(trainval), dtype=np.int64)
    train_counts = trainval.loc[trainval["split"] == "train", "label"].value_counts().astype(int).to_dict()
    if train_counts != {0: 278_487, 1: 54_967}:
        raise ValueError(f"Native training class counts drifted: {train_counts}")

    metrics_root = output_root / "metrics"
    cache_root = output_root / "cache" / PREPARE_CACHE_NAME
    partial_root = cache_root.with_name(cache_root.name + ".partial")
    metrics_root.mkdir(parents=True, exist_ok=True)
    cohort_path = metrics_root / "trainval_cohort_manifest.csv"
    if cohort_path.exists():
        prior = pd.read_csv(cohort_path)
        if prior["sequence_id"].astype(str).tolist() != trainval["sequence_id"].astype(str).tolist():
            raise ValueError("Existing train/validation cohort manifest drifted.")
    else:
        trainval.to_csv(cohort_path, index=False)

    progress("authenticating_exp047_reuse_cache")
    reused_rows, reused_arrays = _load_and_authenticate_exp047_cache()
    reuse_index = build_reuse_index(trainval, reused_rows)
    if len(reuse_index) != PROTOCOL.validation_rows + 109_934:
        raise ValueError(f"Unexpected Exp047 reuse population: {len(reuse_index)}")
    progress(f"exp047_reuse_rows={len(reuse_index)} reconstruct_rows={len(trainval)-len(reuse_index)}")

    exp047 = load_exp047_core()
    progress("loading_trainval_pie_xml_only")
    pie_db = exp047.load_pie_sets_from_xml(TRAIN_SETS + VAL_SETS, training_phase=True)
    computed_scalers = exp047.compute_train_scalers(pie_db)
    with exp047.SCALERS_PATH.open("rb") as handle:
        historical_scalers = pickle.load(handle)
    scalers = {name: float(historical_scalers[name]) for name in computed_scalers}
    for name, value in computed_scalers.items():
        if not np.isclose(value, scalers[name], rtol=1e-12, atol=1e-12):
            raise ValueError(f"Historical scaler mismatch for {name}: {value} vs {scalers[name]}")
    progress("loading_pinned_exp012_trainval_sets_only")
    exp012_store = exp047.Exp012PoseStore.load_allowed_sets(EXP012_POSE_PATH, set(TRAIN_SETS + VAL_SETS))
    yolop_store = YolopStore(YOLOP_ROOT)

    state_path = partial_root / "progress.json"
    if partial_root.exists():
        if not resume or not state_path.is_file():
            raise FileExistsError(f"Partial preparation exists; rerun with --resume: {partial_root}")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("cohort_sha256") != sha256_file(cohort_path) or int(state.get("n_rows", -1)) != len(trainval):
            raise ValueError("Partial preparation is bound to a different cohort.")
        arrays = _open_partial_arrays(partial_root, len(trainval), create=False)
        start = int(state["next_index"])
        feature_names = tuple(state.get("biu_feature_names", ())) or None
        progress(f"status=resuming next_index={start}")
    else:
        partial_root.mkdir(parents=True)
        arrays = _open_partial_arrays(partial_root, len(trainval), create=True)
        start = 0
        feature_names = None
        atomic_json(
            {
                "created_at": utc_now(),
                "n_rows": len(trainval),
                "next_index": 0,
                "cohort_sha256": sha256_file(cohort_path),
                "biu_feature_names": [],
            },
            state_path,
        )

    dummy_yolo = _UnusedYoloPoseStore()
    base_names = ("bbox", "ped_action", "ped_look", "ego_speed", "ego_acc", "static_context", "exp012_pose")
    try:
        for position in range(start, len(trainval)):
            row = trainval.iloc[position]
            sequence_id = str(row.sequence_id)
            reused_index = reuse_index.get(sequence_id)
            if reused_index is not None:
                extracted = {
                    name: np.asarray(reused_arrays[name][reused_index], dtype=np.float32)
                    for name in base_names
                }
            else:
                extracted = exp047.extract_mslstm_stream_window(
                    row,
                    pie_db=pie_db,
                    yolo_store=dummy_yolo,
                    exp012_store=exp012_store,
                    scalers=scalers,
                )
            for name in base_names:
                value = np.asarray(extracted[name], dtype=np.float32)
                if value.shape != CACHE_ARRAY_SHAPES[name] or not np.isfinite(value).all():
                    raise ValueError(f"Invalid {name} for {sequence_id}: {value.shape}")
                arrays[name][position] = value
            frames = list(range(int(row.start_frame), int(row.end_frame) + 1))
            yolop, yolop_available = yolop_store.get_window_with_availability(
                row.set_id,
                row.video_id,
                row.ped_id,
                frames,
            )
            arrays["yolop"][position] = yolop
            arrays["yolop_valid_frame_fraction"][position] = float(yolop_available.mean())
            arrays["yolop_complete_window"][position] = bool(yolop_available.all())
            pose = np.asarray(extracted["exp012_pose"], dtype=np.float32)
            joint_mask = np.isfinite(pose.reshape(30, 17, 2)).all(axis=2) & np.any(
                pose.reshape(30, 17, 2) != 0.0,
                axis=2,
            )
            arrays["exp012_pose_valid"][position] = bool(joint_mask.any())
            arrays["exp012_valid_joint_fraction"][position] = float(joint_mask.mean())
            biu, names, direction_source = biu_with_metadata_from_normalized_measurements(
                pose,
                np.asarray(extracted["bbox"], dtype=np.float32),
                width=int(row.frame_width),
                height=int(row.frame_height),
            )
            if feature_names is None:
                feature_names = names
            elif names != feature_names:
                raise RuntimeError(f"BIU schema drift at {sequence_id}")
            arrays["biu_v2"][position] = biu
            arrays["bbox_direction_fallback"][position] = direction_source == "bbox_center_x"
            if (position + 1) % 5_000 == 0 or position + 1 == len(trainval):
                _flush_arrays(arrays)
                atomic_json(
                    {
                        "updated_at": utc_now(),
                        "n_rows": len(trainval),
                        "next_index": position + 1,
                        "cohort_sha256": sha256_file(cohort_path),
                        "biu_feature_names": list(feature_names or ()),
                    },
                    state_path,
                )
                progress(
                    f"cache_rows={position+1}/{len(trainval)} split={row.split} "
                    f"reused={reused_index is not None}"
                )
        _flush_arrays(arrays)
        coverage, fallback = _summarize_prepare(trainval, arrays)
        coverage_path = metrics_root / "pose_coverage_trainval.csv"
        fallback_path = metrics_root / "bbox_direction_fallback_trainval.csv"
        yolop_path = metrics_root / "yolop_coverage_trainval.csv"
        oversampling_paths = []
        coverage.to_csv(coverage_path, index=False)
        fallback.to_csv(fallback_path, index=False)
        yolop_rows = []
        for split in ("train", "val"):
            mask = trainval["split"].to_numpy() == split
            fractions = np.asarray(arrays["yolop_valid_frame_fraction"])[mask].astype(float)
            complete = np.asarray(arrays["yolop_complete_window"])[mask].astype(bool)
            yolop_rows.append(
                {
                    "split": split,
                    "n_windows": int(mask.sum()),
                    "complete_window_count": int(complete.sum()),
                    "complete_window_rate": float(complete.mean()),
                    "incomplete_window_count": int((~complete).sum()),
                    "mean_valid_frame_fraction": float(fractions.mean()),
                    "median_valid_frame_fraction": float(np.median(fractions)),
                }
            )
        pd.DataFrame(yolop_rows).to_csv(yolop_path, index=False)
        training_rows = trainval.loc[trainval["split"] == "train"].copy()
        for seed in SEEDS:
            manifest = build_oversampling_manifest(training_rows, seed)
            if len(manifest) != PROTOCOL.oversampled_train_rows:
                raise ValueError(f"Seed {seed} oversampling population drifted: {len(manifest)}")
            path = metrics_root / f"oversampling_manifest_seed{seed}.csv.gz"
            manifest.to_csv(path, index=False, compression={"method": "gzip", "mtime": 0})
            oversampling_paths.append(path)
        rows_path = partial_root / "rows.csv"
        trainval.to_csv(rows_path, index=False)
        state_path.unlink()
        artifact_hashes = {
            path.name: sha256_file(path)
            for path in sorted(partial_root.glob("*.npy"))
        }
        artifact_hashes["rows.csv"] = sha256_file(rows_path)
        metadata = {
            "created_at": utc_now(),
            "n_rows": len(trainval),
            "splits": EXPECTED_TRAINVAL_COUNTS,
            "exp047_reused_rows": len(reuse_index),
            "reconstructed_rows": len(trainval) - len(reuse_index),
            "biu_feature_names": list(feature_names or ()),
            "set03_modality_accessed": False,
            "artifacts": artifact_hashes,
        }
        atomic_json(metadata, partial_root / "metadata.json")
        partial_root.replace(cache_root)
    finally:
        arrays.clear()

    source_paths = [
        Path(__file__).resolve(),
        CORE_PATH,
        EXP012_POSE_PATH,
        EXP032_PROTOCOL_ROWS,
        EXP047_CORE_PATH,
        EXP047_TRAINVAL_CACHE / "metadata.json",
        exp047.PIE_PARSER_PATH,
        exp047.SCALERS_PATH,
        *sorted(
            path
            for set_id in TRAIN_SETS + VAL_SETS
            for path in (exp047.PIE_ANNOTATION_ROOT / "annotations" / set_id).glob("*_annt.xml")
        ),
        *sorted(path for path in YOLOP_ROOT.glob("*_yolop_features.pkl") if path.name[:5] in TRAIN_SETS + VAL_SETS),
    ]
    source_hashes_path = metrics_root / "pretraining_source_hashes.json"
    atomic_json({str(path.resolve()): sha256_file(path) for path in source_paths}, source_hashes_path)
    artifacts = [
        Path(__file__).resolve(),
        CORE_PATH,
        cohort_path,
        coverage_path,
        fallback_path,
        yolop_path,
        source_hashes_path,
        *oversampling_paths,
        cache_root / "metadata.json",
    ]
    payload = {
        "phase": "prepare",
        "created_at": utc_now(),
        "train_rows": PROTOCOL.native_train_rows,
        "validation_rows": PROTOCOL.validation_rows,
        "oversampled_train_rows": PROTOCOL.oversampled_train_rows,
        "set03_model_input_accessed": False,
        "artifacts": {
            (str(path.relative_to(output_root)) if path.is_relative_to(output_root) else str(path)): sha256_file(path)
            for path in artifacts
        },
    }
    atomic_json(payload, marker)
    progress(f"phase=prepare status=complete marker={marker}")


def load_prepared_cache(output_root: Path) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, object]]:
    root = Path(output_root)
    marker_path = root / ".prepare.complete.json"
    if not marker_path.is_file():
        raise FileNotFoundError("Training is blocked until .prepare.complete.json exists.")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    for relative, expected_hash in marker.get("artifacts", {}).items():
        path = root / relative
        if not path.is_file() or sha256_file(path) != str(expected_hash):
            raise ValueError(f"Preparation artifact hash mismatch: {path}")
    cache_root = root / "cache" / PREPARE_CACHE_NAME
    metadata_path = cache_root / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if int(metadata.get("n_rows", -1)) != PROTOCOL.native_train_rows + PROTOCOL.validation_rows:
        raise ValueError("Prepared cache population drifted.")
    if metadata.get("splits") != EXPECTED_TRAINVAL_COUNTS:
        raise ValueError("Prepared cache split counts drifted.")
    artifacts = metadata.get("artifacts")
    expected_artifacts = {"rows.csv", *(f"{name}.npy" for name in CACHE_ARRAY_SHAPES)}
    if not isinstance(artifacts, dict) or set(artifacts) != expected_artifacts:
        raise ValueError("Prepared cache artifact universe is not exact.")
    for relative, expected_hash in artifacts.items():
        path = cache_root / relative
        if not path.is_file() or sha256_file(path) != str(expected_hash):
            raise ValueError(f"Prepared cache artifact hash mismatch: {path}")
    rows = validate_trainval_rows(pd.read_csv(cache_root / "rows.csv"))
    arrays = {
        name: np.load(cache_root / f"{name}.npy", mmap_mode="r")
        for name in CACHE_ARRAY_SHAPES
    }
    if any(len(array) != len(rows) for array in arrays.values()):
        raise ValueError("Prepared cache arrays do not align with the row population.")
    return rows, arrays, metadata


def atomic_torch_save(payload: Mapping[str, object], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    temporary.replace(path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def predict_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, float]:
    criterion = nn.CrossEntropyLoss(weight=torch.ones(2, device=device))
    labels: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    loss_sum = 0.0
    sample_count = 0
    model.eval()
    with torch.no_grad():
        for inputs, batch_labels, _ in loader:
            moved = {name: value.to(device, non_blocking=device.type == "cuda") for name, value in inputs.items()}
            batch_labels = batch_labels.to(device, non_blocking=device.type == "cuda")
            logits = model(moved)
            loss = criterion(logits, batch_labels)
            batch_size = int(batch_labels.shape[0])
            loss_sum += float(loss.item()) * batch_size
            sample_count += batch_size
            labels.append(batch_labels.cpu().numpy())
            probabilities.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
    if sample_count == 0:
        raise ValueError("Prediction loader is empty.")
    return np.concatenate(labels), np.concatenate(probabilities), loss_sum / sample_count


def _training_marker(output_root: Path, variant: str, seed: int) -> Path:
    return Path(output_root) / f".train_{variant}_seed{seed}.complete.json"


def _authenticate_training_marker(output_root: Path, variant: str, seed: int) -> dict[str, object]:
    root = Path(output_root)
    path = _training_marker(root, variant, seed)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("phase") != "train" or payload.get("variant") != variant or int(payload.get("seed", -1)) != seed:
        raise ValueError(f"Training marker identity drifted: {path}")
    if payload.get("prepare_sha256") != sha256_file(root / ".prepare.complete.json"):
        raise ValueError(f"Training marker is bound to another preparation: {path}")
    for relative, expected_hash in payload.get("artifacts", {}).items():
        artifact = root / relative
        if not artifact.is_file() or sha256_file(artifact) != str(expected_hash):
            raise ValueError(f"Training artifact hash mismatch: {artifact}")
    return payload


def run_training(
    output_root: Path,
    *,
    variant: str,
    seed: int,
    resume: bool,
    prepared: tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, object]] | None = None,
) -> None:
    root = Path(output_root)
    marker = _training_marker(root, variant, seed)
    log = progress_logger(root / "logs" / f"train_{variant}_seed{seed}.log")
    if marker.exists():
        _authenticate_training_marker(root, variant, seed)
        log("status=already_complete_authenticated")
        return
    if variant not in VARIANTS or seed not in SEEDS:
        raise ValueError("Training identity is outside the frozen A-E/42-44 grid.")
    rows, arrays, cache_metadata = prepared if prepared is not None else load_prepared_cache(root)
    train_rows = rows.loc[rows["split"] == "train"].reset_index(drop=True)
    val_rows = rows.loc[rows["split"] == "val"].reset_index(drop=True)
    sampling_path = root / "metrics" / f"oversampling_manifest_seed{seed}.csv.gz"
    sampling = pd.read_csv(sampling_path)
    if len(sampling) != PROTOCOL.oversampled_train_rows:
        raise ValueError(f"Oversampling manifest population drifted for seed {seed}.")
    if sampling["label"].value_counts().astype(int).to_dict() != {0: 278_487, 1: 278_487}:
        raise ValueError(f"Oversampling class counts drifted for seed {seed}.")
    train_dataset = Exp048Dataset(
        train_rows,
        arrays,
        variant=variant,
        sampling_manifest=sampling,
        training_phase=True,
    )
    val_dataset = Exp048Dataset(val_rows, arrays, variant=variant, training_phase=True)
    seed_everything(seed)
    generator = torch.Generator().manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(
        train_dataset,
        batch_size=PROTOCOL.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=PROTOCOL.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    model = build_model(variant, seed).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=PROTOCOL.learning_rate)
    criterion = nn.CrossEntropyLoss(weight=torch.ones(2, device=device))
    state_path = root / "checkpoints" / f"{variant}_seed{seed}.state.pt"
    history: list[dict[str, float | int]] = []
    best_f1 = -1.0
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    start_epoch = 1
    if state_path.exists():
        if not resume:
            raise FileExistsError(f"Resumable state exists; rerun with --resume: {state_path}")
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if state.get("variant") != variant or int(state.get("seed", -1)) != seed:
            raise ValueError("Resumable training-state identity drifted.")
        if state.get("prepare_sha256") != sha256_file(root / ".prepare.complete.json"):
            raise ValueError("Resumable training state is bound to another preparation.")
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        generator.set_state(state["data_generator_state"])
        torch.set_rng_state(state["torch_rng_state"])
        if torch.cuda.is_available() and state.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng_state"])
        history = list(state["history"])
        best_f1 = float(state["best_f1"])
        best_epoch = int(state["best_epoch"])
        best_state = state["best_state"]
        start_epoch = int(state["last_epoch"]) + 1
        log(f"status=resuming start_epoch={start_epoch}")
    else:
        log(
            f"phase=train status=starting variant={variant} seed={seed} device={device} "
            f"train_rows={len(train_dataset)} val_rows={len(val_dataset)}"
        )

    for epoch in range(start_epoch, PROTOCOL.epoch_limit + 1):
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_count = 0
        for inputs, batch_labels, _ in train_loader:
            moved = {name: value.to(device, non_blocking=device.type == "cuda") for name, value in inputs.items()}
            batch_labels = batch_labels.to(device, non_blocking=device.type == "cuda")
            optimizer.zero_grad(set_to_none=True)
            logits = model(moved)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
            batch_size = int(batch_labels.shape[0])
            train_loss_sum += float(loss.item()) * batch_size
            train_correct += int((logits.argmax(dim=1) == batch_labels).sum().item())
            train_count += batch_size
        val_labels, val_probabilities, val_loss = predict_loader(model, val_loader, device)
        val_metrics = compute_binary_metrics(val_labels, val_probabilities)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss_sum / train_count,
                "train_accuracy": train_correct / train_count,
                "val_loss": val_loss,
                "val_accuracy": val_metrics["accuracy"],
                "val_precision_at_0p5": val_metrics["precision"],
                "val_recall_at_0p5": val_metrics["recall"],
                "val_f1_at_0p5": val_metrics["f1"],
                "val_roc_auc": val_metrics["roc_auc"],
                "val_pr_auc": val_metrics["pr_auc"],
            }
        )
        if val_metrics["f1"] > best_f1:
            best_f1 = float(val_metrics["f1"])
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        atomic_torch_save(
            {
                "variant": variant,
                "seed": seed,
                "prepare_sha256": sha256_file(root / ".prepare.complete.json"),
                "last_epoch": epoch,
                "model_state": {name: value.detach().cpu() for name, value in model.state_dict().items()},
                "optimizer_state": optimizer.state_dict(),
                "data_generator_state": generator.get_state(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "history": history,
                "best_f1": best_f1,
                "best_epoch": best_epoch,
                "best_state": best_state,
            },
            state_path,
        )
        atomic_csv(pd.DataFrame(history), root / "logs" / f"training_history_{variant}_seed{seed}.csv")
        log(
            f"epoch={epoch}/{PROTOCOL.epoch_limit} train_loss={train_loss_sum/train_count:.6f} "
            f"train_accuracy={train_correct/train_count:.6f} val_f1_at_0p5={val_metrics['f1']:.6f} "
            f"val_pr_auc={val_metrics['pr_auc']:.6f} best_epoch={best_epoch}"
        )
    if len(history) != PROTOCOL.epoch_limit or best_state is None:
        raise RuntimeError("Training did not complete all 15 frozen epochs.")
    model.load_state_dict(best_state)
    val_labels, val_probabilities, _ = predict_loader(model, val_loader, device)
    validation_metrics = compute_binary_metrics(val_labels, val_probabilities)
    checkpoint_path = root / "checkpoints" / f"{variant}_seed{seed}.best.pt"
    total_parameters, trainable_parameters = count_parameters(model)
    atomic_torch_save(
        {
            "variant": variant,
            "seed": seed,
            "best_epoch": best_epoch,
            "best_validation_f1": best_f1,
            "model_state": best_state,
            "prepare_sha256": sha256_file(root / ".prepare.complete.json"),
            "total_parameters": total_parameters,
            "trainable_parameters": trainable_parameters,
            "pose_pathway_parameters": pose_path_parameters(model, variant),
        },
        checkpoint_path,
    )
    prediction_path = root / "predictions" / f"validation_{variant}_seed{seed}.csv"
    prediction = val_rows[
        ["sequence_id", "set_id", "video_id", "ped_id", "start_frame", "end_frame", "target_frame", "label"]
    ].copy()
    prediction["variant"] = variant
    prediction["seed"] = seed
    prediction["threshold"] = PROTOCOL.fixed_threshold
    prediction["probability"] = val_probabilities
    prediction["prediction"] = (val_probabilities >= PROTOCOL.fixed_threshold).astype(np.int64)
    atomic_csv(prediction, prediction_path)
    history_path = root / "logs" / f"training_history_{variant}_seed{seed}.csv"
    payload = {
        "phase": "train",
        "created_at": utc_now(),
        "variant": variant,
        "seed": seed,
        "device": str(device),
        "best_epoch": best_epoch,
        "best_validation_f1": best_f1,
        "validation_metrics": validation_metrics,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "pose_pathway_parameters": pose_path_parameters(model, variant),
        "prepare_sha256": sha256_file(root / ".prepare.complete.json"),
        "cache_metadata_sha256": sha256_file(root / "cache" / PREPARE_CACHE_NAME / "metadata.json"),
        "artifacts": {
            str(path.relative_to(root)): sha256_file(path)
            for path in (checkpoint_path, prediction_path, history_path)
        },
    }
    atomic_json(payload, marker)
    _authenticate_training_marker(root, variant, seed)
    state_path.unlink(missing_ok=True)
    log(f"phase=train status=complete best_epoch={best_epoch} best_val_f1={best_f1:.6f}")


def authenticate_training_grid(output_root: Path) -> list[dict[str, object]]:
    root = Path(output_root)
    records = [
        _authenticate_training_marker(root, variant, seed)
        for variant in VARIANTS
        for seed in SEEDS
    ]
    if len(records) != 15:
        raise ValueError("Training authorization requires exactly 15 complete runs.")
    return records


def run_train_all(output_root: Path, *, resume: bool) -> None:
    root = Path(output_root)
    prepared = load_prepared_cache(root)
    for variant in VARIANTS:
        for seed in SEEDS:
            run_training(root, variant=variant, seed=seed, resume=resume, prepared=prepared)
    authenticate_training_grid(root)
    marker = root / "TRAINING_COMPLETE"
    marker.write_text(f"completed_at={utc_now()}\nrun_count=15\n", encoding="utf-8")


def _load_authenticated_exp047_set03() -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, object]]:
    metadata_path = EXP047_SET03_CACHE / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    artifacts = metadata.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("Exp047 set03 cache lacks an artifact hash manifest.")
    for relative, expected_hash in artifacts.items():
        path = EXP047_SET03_CACHE / relative
        if not path.is_file() or sha256_file(path) != str(expected_hash):
            raise ValueError(f"Exp047 set03 cache artifact hash mismatch: {path}")
    rows = pd.read_csv(EXP047_SET03_CACHE / "rows.csv")
    if len(rows) != PROTOCOL.test_rows or set(rows["split"].astype(str)) != {"test"}:
        raise ValueError("Exp047 set03 population drifted.")
    arrays = {
        name: np.load(EXP047_SET03_CACHE / f"{name}.npy", mmap_mode="r")
        for name in (
            "bbox",
            "ped_action",
            "ped_look",
            "ego_speed",
            "ego_acc",
            "static_context",
            "exp012_pose",
            "biu_v2",
        )
    }
    return rows, arrays, metadata


def _build_set03_yolop(
    output_root: Path,
    rows: pd.DataFrame,
    *,
    resume: bool,
    progress,
) -> tuple[np.ndarray, dict[str, object]]:
    root = Path(output_root) / "cache" / "set03_yolop"
    metadata_path = root / "metadata.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        array_path = root / "yolop.npy"
        if sha256_file(array_path) != metadata.get("yolop_sha256"):
            raise ValueError("Set03 YOLOP cache hash drifted.")
        return np.load(array_path, mmap_mode="r"), metadata
    root.mkdir(parents=True, exist_ok=True)
    array_path = root / "yolop.npy"
    availability_path = root / "availability.npy"
    state_path = root / "progress.json"
    if state_path.exists():
        if not resume:
            raise FileExistsError("Partial set03 YOLOP cache exists; rerun evaluate with --resume.")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        start = int(state["next_index"])
        yolop = np.load(array_path, mmap_mode="r+")
        availability = np.load(availability_path, mmap_mode="r+")
    else:
        start = 0
        yolop = np.lib.format.open_memmap(
            array_path,
            mode="w+",
            dtype=np.float32,
            shape=(len(rows), 30, 20),
        )
        availability = np.lib.format.open_memmap(
            availability_path,
            mode="w+",
            dtype=np.float32,
            shape=(len(rows),),
        )
        atomic_json({"next_index": 0, "n_rows": len(rows)}, state_path)
    store = YolopStore(YOLOP_ROOT)
    for position in range(start, len(rows)):
        row = rows.iloc[position]
        frames = list(range(int(row.start_frame), int(row.end_frame) + 1))
        values, valid = store.get_window_with_availability(row.set_id, row.video_id, row.ped_id, frames)
        yolop[position] = values
        availability[position] = float(valid.mean())
        if (position + 1) % 10_000 == 0 or position + 1 == len(rows):
            yolop.flush()
            availability.flush()
            atomic_json({"next_index": position + 1, "n_rows": len(rows)}, state_path)
            progress(f"set03_yolop_rows={position+1}/{len(rows)}")
    yolop.flush()
    availability.flush()
    coverage = np.asarray(availability, dtype=float)
    metadata = {
        "created_at": utc_now(),
        "n_rows": len(rows),
        "yolop_sha256": sha256_file(array_path),
        "availability_sha256": sha256_file(availability_path),
        "complete_window_count": int((coverage == 1.0).sum()),
        "complete_window_rate": float((coverage == 1.0).mean()),
        "mean_valid_frame_fraction": float(coverage.mean()),
        "median_valid_frame_fraction": float(np.median(coverage)),
    }
    state_path.unlink()
    atomic_json(metadata, metadata_path)
    return np.load(array_path, mmap_mode="r"), metadata


def _measure_model_latency(
    model: nn.Module,
    features: Mapping[str, torch.Tensor],
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    moved = {name: value.to(device) for name, value in features.items()}
    if device.type == "cuda":
        torch.cuda.synchronize()
    with torch.no_grad():
        for _ in range(2):
            model(moved)
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(5):
            model(moved)
        if device.type == "cuda":
            torch.cuda.synchronize()
    seconds = (time.perf_counter() - started) / 5.0
    batch_size = int(next(iter(moved.values())).shape[0])
    return {
        "predictor_seconds_per_batch": seconds,
        "predictor_seconds_per_sample": seconds / batch_size,
        "latency_batch_size": batch_size,
    }


def _measure_pose_pathway_latency(
    model: nn.Module,
    variant: str,
    features: Mapping[str, torch.Tensor],
    device: torch.device,
) -> float:
    if variant not in {"D", "E"}:
        return float("nan")
    moved = {name: value.to(device) for name, value in features.items()}

    def operation():
        if variant == "D":
            sequence, _ = model.lstms["pose"](moved["pose"])
            return model.attentions["pose"](sequence)[0]
        return model.biu_projection(moved["biu_v2"])

    model.eval()
    with torch.no_grad():
        for _ in range(2):
            operation()
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(5):
            operation()
        if device.type == "cuda":
            torch.cuda.synchronize()
    seconds_per_batch = (time.perf_counter() - started) / 5.0
    return seconds_per_batch / int(next(iter(moved.values())).shape[0])


def _measure_biu_extraction_latency(rows: pd.DataFrame, arrays: Mapping[str, np.ndarray]) -> float:
    sample_count = min(100, len(rows))
    started = time.perf_counter()
    for index in range(sample_count):
        row = rows.iloc[index]
        biu_with_metadata_from_normalized_measurements(
            np.asarray(arrays["exp012_pose"][index], dtype=np.float32),
            np.asarray(arrays["bbox"][index], dtype=np.float32),
            width=int(row.frame_width),
            height=int(row.frame_height),
        )
    return (time.perf_counter() - started) / sample_count


def _summary_results(results: pd.DataFrame) -> pd.DataFrame:
    metrics = ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc")
    rows = []
    for variant in VARIANTS:
        selected = results.loc[results["variant"] == variant]
        if selected["seed"].astype(int).tolist() != list(SEEDS):
            raise ValueError(f"Summary seed order/population drifted for {variant}.")
        row: dict[str, object] = {"variant": variant, "n_seeds": len(selected)}
        for metric in metrics:
            row[f"{metric}_mean"] = float(selected[metric].mean())
            row[f"{metric}_std"] = float(selected[metric].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def _error_transitions(predictions: Mapping[tuple[str, int], pd.DataFrame]) -> pd.DataFrame:
    comparisons = (("A", "B"), ("B", "C"), ("C", "D"), ("C", "E"), ("D", "E"), ("A", "E"))
    rows = []
    for baseline, candidate in comparisons:
        for seed in SEEDS:
            left = predictions[(baseline, seed)]
            right = predictions[(candidate, seed)]
            if left["sequence_id"].tolist() != right["sequence_id"].tolist():
                raise ValueError("Error-transition prediction cohorts are not aligned.")
            left_correct = left["prediction"].to_numpy() == left["label"].to_numpy()
            right_correct = right["prediction"].to_numpy() == right["label"].to_numpy()
            rows.append(
                {
                    "comparison": f"{baseline}_to_{candidate}",
                    "seed": seed,
                    "baseline_wrong_candidate_correct": int((~left_correct & right_correct).sum()),
                    "baseline_correct_candidate_wrong": int((left_correct & ~right_correct).sum()),
                    "both_correct": int((left_correct & right_correct).sum()),
                    "both_wrong": int((~left_correct & ~right_correct).sum()),
                }
            )
    return pd.DataFrame(rows)


def run_final_evaluation(output_root: Path, *, resume: bool) -> None:
    root = Path(output_root)
    records = authenticate_training_grid(root)  # This gate executes before any set03 path is read.
    marker = root / ".evaluate.complete.json"
    log = progress_logger(root / "logs" / "final_evaluation.log")
    if marker.exists():
        payload = json.loads(marker.read_text(encoding="utf-8"))
        for relative, expected_hash in payload.get("artifacts", {}).items():
            path = root / relative
            if not path.is_file() or sha256_file(path) != str(expected_hash):
                raise ValueError(f"Evaluation artifact hash mismatch: {path}")
        log("phase=evaluate status=already_complete_authenticated")
        return
    log("phase=evaluate status=authorized training_records=15 loading_set03_once")
    exp047_rows, arrays, exp047_metadata = _load_authenticated_exp047_set03()
    protocol_rows = pd.read_csv(EXP032_PROTOCOL_ROWS)
    expected = protocol_rows.loc[protocol_rows["split"] == "test"].reset_index(drop=True)
    identity = ("sequence_id", "label", "start_frame", "end_frame", "target_frame")
    if len(expected) != PROTOCOL.test_rows or any(
        expected[name].astype(str).tolist() != exp047_rows[name].astype(str).tolist()
        for name in identity
    ):
        raise ValueError("Set03 cohort does not exactly match the frozen protocol row identity/order.")
    set03_rows = exp047_rows.reset_index(drop=True).copy()
    set03_rows["cache_index"] = np.arange(len(set03_rows), dtype=np.int64)
    yolop, yolop_metadata = _build_set03_yolop(root, set03_rows, resume=resume, progress=log)
    arrays = {**arrays, "yolop": yolop}
    metrics_root = root / "metrics"
    predictions_root = root / "predictions"
    metrics_root.mkdir(parents=True, exist_ok=True)
    predictions_root.mkdir(parents=True, exist_ok=True)
    set03_manifest_path = metrics_root / "set03_cohort_manifest.csv"
    atomic_csv(set03_rows, set03_manifest_path)
    coverage_path = metrics_root / "yolop_coverage_set03.csv"
    atomic_csv(pd.DataFrame([{"split": "test", **{key: value for key, value in yolop_metadata.items() if key.endswith("count") or key.endswith("rate") or key.endswith("fraction")}}]), coverage_path)
    exp047_pose_coverage = pd.read_csv(EXP047_ROOT / "metrics" / "pose_coverage.csv")
    pose_coverage_path = metrics_root / "pose_coverage_set03.csv"
    atomic_csv(
        exp047_pose_coverage.loc[
            (exp047_pose_coverage["split"] == "test") & (exp047_pose_coverage["source"] == "exp012")
        ].reset_index(drop=True),
        pose_coverage_path,
    )
    exp047_fallback = pd.read_csv(EXP047_ROOT / "metrics" / "bbox_direction_fallback.csv")
    fallback_path = metrics_root / "bbox_direction_fallback_set03.csv"
    atomic_csv(exp047_fallback.loc[exp047_fallback["split"] == "test"].reset_index(drop=True), fallback_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []
    efficiency = []
    stream_parameters = []
    prediction_frames: dict[tuple[str, int], pd.DataFrame] = {}
    record_by_identity = {(record["variant"], int(record["seed"])): record for record in records}
    for variant in VARIANTS:
        dataset = Exp048Dataset(set03_rows, arrays, variant=variant, training_phase=False)
        loader = DataLoader(
            dataset,
            batch_size=PROTOCOL.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )
        first_features, _, _ = next(iter(loader))
        for seed in SEEDS:
            record = record_by_identity[(variant, seed)]
            checkpoint_path = root / "checkpoints" / f"{variant}_seed{seed}.best.pt"
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            model = build_model(variant, seed)
            model.load_state_dict(checkpoint["model_state"])
            model.to(device)
            labels, probabilities, _ = predict_loader(model, loader, device)
            metric = compute_binary_metrics(labels, probabilities)
            results.append({"variant": variant, "seed": seed, **metric})
            prediction = set03_rows[
                ["sequence_id", "set_id", "video_id", "ped_id", "start_frame", "end_frame", "target_frame", "label"]
            ].copy()
            prediction["variant"] = variant
            prediction["seed"] = seed
            prediction["threshold"] = PROTOCOL.fixed_threshold
            prediction["probability"] = probabilities
            prediction["prediction"] = (probabilities >= PROTOCOL.fixed_threshold).astype(np.int64)
            prediction_path = predictions_root / f"set03_{variant}_seed{seed}.csv"
            atomic_csv(prediction, prediction_path)
            prediction_frames[(variant, seed)] = prediction
            latency = _measure_model_latency(model, first_features, device)
            pathway_latency = _measure_pose_pathway_latency(model, variant, first_features, device)
            efficiency.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "device": str(device),
                    "total_parameters": int(record["total_parameters"]),
                    "trainable_parameters": int(record["trainable_parameters"]),
                    "pose_or_biu_pathway_parameters": int(record["pose_pathway_parameters"]),
                    **latency,
                    "pose_or_biu_projection_seconds_per_sample": pathway_latency,
                    "biu_extraction_seconds_per_sample": (
                        _measure_biu_extraction_latency(set03_rows, arrays) if variant == "E" and seed == SEEDS[0] else np.nan
                    ),
                    "pose_estimator_runtime_included": False,
                    "pose_estimator_runtime_status": "unavailable_historical_cache",
                }
            )
            log(
                f"evaluated variant={variant} seed={seed} f1={metric['f1']:.6f} "
                f"pr_auc={metric['pr_auc']:.6f}"
            )
            if seed == SEEDS[0]:
                for stream in model.streams:
                    if stream == "biu_v2":
                        parameter_count = sum(value.numel() for value in model.biu_projection.parameters())
                    else:
                        parameter_count = sum(value.numel() for value in model.lstms[stream].parameters())
                        parameter_count += sum(value.numel() for value in model.attentions[stream].parameters())
                    stream_parameters.append(
                        {
                            "variant": variant,
                            "stream": stream,
                            "parameters": int(parameter_count),
                        }
                    )

    per_seed = pd.DataFrame(results)
    summary = _summary_results(per_seed)
    deltas = build_paired_deltas(per_seed)
    recovery = build_behavior_recovery(per_seed)
    transitions = _error_transitions(prediction_frames)
    summary_index = summary.set_index("variant")
    recovery_index = recovery.set_index("metric")
    d_to_e = deltas.loc[deltas["comparison"] == "D_to_E"]
    success = pd.DataFrame(
        [
            {
                "criterion": "strong_automatic_behavior_recovery",
                "satisfied": bool(
                    summary_index.loc["E", "f1_mean"] - summary_index.loc["C", "f1_mean"] > 0.005
                    and summary_index.loc["E", "pr_auc_mean"] - summary_index.loc["C", "pr_auc_mean"] > 0.005
                    and recovery_index.loc["f1", "biu_recovery_fraction_mean"] >= 0.5
                    and recovery_index.loc["pr_auc", "biu_recovery_fraction_mean"] >= 0.5
                ),
            },
            {
                "criterion": "biu_representation_success",
                "satisfied": bool((d_to_e["f1_delta"] > 0).all() and (d_to_e["pr_auc_delta"] > 0).all()),
            },
            {
                "criterion": "biu_efficiency_success",
                "satisfied": bool(
                    summary_index.loc["E", "f1_mean"] >= summary_index.loc["D", "f1_mean"] - 0.005
                    and summary_index.loc["E", "pr_auc_mean"] >= summary_index.loc["D", "pr_auc_mean"] - 0.005
                ),
            },
            {
                "criterion": "near_full_gt_behavior_recovery",
                "satisfied": bool(
                    summary_index.loc["E", "f1_mean"] >= summary_index.loc["B", "f1_mean"] - 0.005
                    and summary_index.loc["E", "pr_auc_mean"] >= summary_index.loc["B", "pr_auc_mean"] - 0.005
                ),
            },
        ]
    )
    output_frames = {
        "per_seed_results.csv": per_seed,
        "fixed_0p5_results.csv": per_seed,
        "mslstm_protocol_results.csv": per_seed,
        "summary_results.csv": summary,
        "paired_deltas.csv": deltas,
        "behavior_recovery.csv": recovery,
        "efficiency.csv": pd.DataFrame(efficiency),
        "stream_parameter_counts.csv": pd.DataFrame(stream_parameters),
        "success_criteria.csv": success,
        "error_transitions_fixed_0p5.csv": transitions,
        "historical_references.csv": pd.DataFrame(
            [
                {
                    "variant": "A",
                    "source": "ICVES paper reported final result / thesis-authoritative stream target",
                    "historical_f1": 0.8992,
                    "fresh_protocol_parity": False,
                    "note": "Historical reference only; exact paper runner/checkpoint is unavailable.",
                },
                {
                    "variant": "B",
                    "source": "Recovered executed YOLOP notebook lineage",
                    "historical_f1": np.nan,
                    "fresh_protocol_parity": False,
                    "note": "Architecture/training lineage; notebook output is incomplete and is not a controlled result.",
                },
            ]
        ),
    }
    paths = []
    for name, frame in output_frames.items():
        path = metrics_root / name
        atomic_csv(frame, path)
        paths.append(path)
    paths.extend(sorted(predictions_root.glob("set03_*.csv")))
    paths.extend((set03_manifest_path, coverage_path, pose_coverage_path, fallback_path))
    payload = {
        "phase": "evaluate",
        "created_at": utc_now(),
        "training_record_count": 15,
        "training_marker_hashes": {
            f"{variant}_{seed}": sha256_file(_training_marker(root, variant, seed))
            for variant in VARIANTS
            for seed in SEEDS
        },
        "set03_rows": PROTOCOL.test_rows,
        "set03_evaluations": 15,
        "exp047_set03_cache_metadata_sha256": sha256_file(EXP047_SET03_CACHE / "metadata.json"),
        "set03_yolop_metadata": yolop_metadata,
        "artifacts": {str(path.relative_to(root)): sha256_file(path) for path in paths},
    }
    atomic_json(payload, marker)
    log(f"phase=evaluate status=complete marker={marker}")


def run_finalize(output_root: Path) -> None:
    root = Path(output_root)
    evaluation_path = root / ".evaluate.complete.json"
    if not evaluation_path.is_file():
        raise FileNotFoundError("Finalization requires a completed final evaluation.")
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    for relative, expected_hash in evaluation.get("artifacts", {}).items():
        path = root / relative
        if not path.is_file() or sha256_file(path) != str(expected_hash):
            raise ValueError(f"Finalization found a drifted evaluation artifact: {path}")
    summary = pd.read_csv(root / "metrics" / "summary_results.csv").set_index("variant")
    deltas = pd.read_csv(root / "metrics" / "paired_deltas.csv")
    recovery = pd.read_csv(root / "metrics" / "behavior_recovery.csv").set_index("metric")
    d_to_e = deltas.loc[deltas["comparison"] == "D_to_E"]
    e_vs_d_consistent = bool((d_to_e["f1_delta"] > 0).all() and (d_to_e["pr_auc_delta"] > 0).all())
    within_margin = bool(
        summary.loc["E", "f1_mean"] >= summary.loc["D", "f1_mean"] - 0.005
        and summary.loc["E", "pr_auc_mean"] >= summary.loc["D", "pr_auc_mean"] - 0.005
    )
    recovery_fraction_f1 = float(recovery.loc["f1", "biu_recovery_fraction_mean"])
    recovery_fraction_pr = float(recovery.loc["pr_auc", "biu_recovery_fraction_mean"])
    if e_vs_d_consistent:
        outcome = "BIU representation success"
    elif within_margin:
        outcome = "BIU efficiency success"
    elif summary.loc["D", "f1_mean"] > summary.loc["E", "f1_mean"] and summary.loc["D", "pr_auc_mean"] > summary.loc["E", "pr_auc_mean"]:
        outcome = "Raw-pose success"
    elif summary.loc["E", "f1_mean"] > summary.loc["C", "f1_mean"] or summary.loc["E", "pr_auc_mean"] > summary.loc["C", "pr_auc_mean"]:
        outcome = "Partial automatic-behavior recovery"
    else:
        outcome = "Negative"
    report_path = root / "FINAL_REPORT.md"
    lines = [
        "# Exp048 Final Report",
        "",
        f"Final outcome category: **{outcome}**.",
        "",
        "The controlled comparison uses the recovered 30-frame, next-raw-frame MSLSTM task. "
        "All conclusions below are from fresh paired A-E runs; the paper F1=0.8992 is historical reference only.",
        "",
        "## Fresh Mean Results",
        "",
        "| Variant | F1 mean | PR-AUC mean | ROC-AUC mean |",
        "|---|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        row = summary.loc[variant]
        lines.append(
            f"| {variant} | {row.f1_mean:.6f} | {row.pr_auc_mean:.6f} | {row.roc_auc_mean:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"D→E favorable on both F1 and PR-AUC in all seeds: `{str(e_vs_d_consistent).lower()}`.",
            f"E is within the frozen 0.005 F1/PR-AUC margin of D: `{str(within_margin).lower()}`.",
            f"BIU recovery fraction: F1 `{recovery_fraction_f1:.6f}`, PR-AUC `{recovery_fraction_pr:.6f}`.",
            "",
            "The thesis-safe interpretation separates the pose-source-independent D↔E representation comparison "
            "from A↔E, which is an overall privileged-to-automatic pipeline comparison. BIU-v2 does not add "
            "information absent from the shared Exp012 pose measurements.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    completion = root / "EXPERIMENT_COMPLETE"
    completion.write_text(
        f"completed_at={utc_now()}\noutcome={outcome}\nfinal_report_sha256={sha256_file(report_path)}\n",
        encoding="utf-8",
    )


def status(output_root: Path) -> None:
    root = Path(output_root)
    prepare = root / ".prepare.complete.json"
    partial = root / "cache" / f"{PREPARE_CACHE_NAME}.partial" / "progress.json"
    print(f"prepare_complete={prepare.is_file()}")
    if partial.is_file():
        state = json.loads(partial.read_text(encoding="utf-8"))
        print(f"prepare_rows={state['next_index']}/{state['n_rows']}")
    for variant in VARIANTS:
        completed = sum((root / f".train_{variant}_seed{seed}.complete.json").is_file() for seed in SEEDS)
        print(f"training_{variant}={completed}/{len(SEEDS)}")
    print(f"evaluation_complete={(root / '.evaluate.complete.json').is_file()}")
    print(f"experiment_complete={(root / 'EXPERIMENT_COMPLETE').is_file()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run frozen Exp048 phases without tmux.")
    parser.add_argument("--output-root", type=Path, default=EXPERIMENT_ROOT)
    parser.add_argument("--resume", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare")
    train = subparsers.add_parser("train")
    train.add_argument("--variant", choices=VARIANTS, required=True)
    train.add_argument("--seed", choices=SEEDS, type=int, required=True)
    subparsers.add_parser("train-all")
    subparsers.add_parser("evaluate")
    subparsers.add_parser("finalize")
    subparsers.add_parser("status")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "prepare":
        run_prepare(args.output_root, resume=args.resume)
    elif args.command == "train":
        run_training(
            args.output_root,
            variant=args.variant,
            seed=args.seed,
            resume=args.resume,
        )
    elif args.command == "train-all":
        run_train_all(args.output_root, resume=args.resume)
    elif args.command == "evaluate":
        run_final_evaluation(args.output_root, resume=args.resume)
    elif args.command == "finalize":
        run_finalize(args.output_root)
    elif args.command == "status":
        status(args.output_root)
    else:
        raise NotImplementedError(f"Exp048 command is not implemented yet: {args.command}")


if __name__ == "__main__":
    main()
