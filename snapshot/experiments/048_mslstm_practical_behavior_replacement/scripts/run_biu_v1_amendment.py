#!/usr/bin/env python3
"""Authenticated sidecar runner for Exp048's approved 72-D BIU-v1 Variant F."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
from dataclasses import dataclass
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
for candidate in (REPO_ROOT, SCRIPT_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from biu_v1_amendment_core import (  # noqa: E402
    BIU_V1_DIM,
    BIU_V1_FEATURE_NAMES,
    BiuV1AmendmentDataset,
    build_f_model,
    extract_biu_v1_window,
    load_original_core,
    sha256_file,
)


EXPERIMENT_ROOT = REPO_ROOT / "experiments" / "048_mslstm_practical_behavior_replacement"
ORIGINAL_CORE_PATH = SCRIPT_ROOT / "exp048_core.py"
ORIGINAL_RUNNER_PATH = SCRIPT_ROOT / "run_experiment.py"
ORIGINAL_CORE_SHA256 = "cf19c532199a57ee8f53d23f07315d3aeadfe714ac43a7aeafebef17863da1dd"
ORIGINAL_RUNNER_SHA256 = "7de82843f4576dacc4ff02b3a61b61ddf0869338479ecc60827ea4044feec185"
EXP012_POSE_PATH = Path(
    "/home/abdelrahman/Hadwa/Pedestrian-keypoints-crossing-intent/"
    "classification/data/data_keypoints_all.jsonl"
)
EXP012_POSE_SHA256 = "1f0907ebb33bb04c3a3434d9c89e7bd597e2b9e3d47fe30f488ffa9410054d87"
ALLOWED_TRAINVAL_SETS = frozenset({"set01", "set02", "set04", "set05", "set06"})
EXPECTED_TRAINVAL_ROWS = 410_742
CACHE_NAME = "biu_v1_amendment_trainval"
AUDIT_MARKER_NAME = ".biu_v1_amendment.audit.complete.json"
F_SEEDS = (42, 43, 44)
F_EPOCHS = 15
F_BATCH_SIZE = 32
F_LEARNING_RATE = 1e-4
F_FIXED_THRESHOLD = 0.5
AF_VARIANTS = ("A", "B", "C", "D", "E", "F")
AMENDED_METRIC_FILES = (
    "per_seed_results.csv",
    "fixed_0p5_results.csv",
    "mslstm_protocol_results.csv",
    "summary_results.csv",
    "paired_deltas.csv",
    "behavior_recovery.csv",
    "efficiency.csv",
    "stream_parameter_counts.csv",
    "success_criteria.csv",
    "error_transitions_fixed_0p5.csv",
    "historical_references.csv",
    "set03_cohort_manifest.csv",
    "yolop_coverage_set03.csv",
    "pose_coverage_set03.csv",
    "bbox_direction_fallback_set03.csv",
    "biu_v1_quality_counts_set03.csv",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(payload: Mapping[str, object], path: Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(destination)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(destination)


def progress_logger(path: Path):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    def emit(message: str) -> None:
        line = f"{utc_now()} {message}"
        print(line, flush=True)
        with destination.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    return emit


@dataclass(frozen=True)
class PoseRecord:
    coords: np.ndarray
    confidence: np.ndarray | None
    label: int


def load_allowed_pose_records(
    path: Path,
    allowed_sets: set[str] | frozenset[str],
) -> dict[tuple[str, str, str, int], PoseRecord]:
    """Decode only explicitly allowed PIE sets from the SHA-pinned JSONL."""
    allowed = {str(value) for value in allowed_sets}
    if not allowed:
        raise ValueError("At least one pose set must be allowed.")
    set_prefix = re.compile(r'^\{"set"\s*:\s*"([^"]+)"')
    records: dict[tuple[str, str, str, int], PoseRecord] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            match = set_prefix.match(line)
            if match is None:
                raise ValueError(f"Cannot identify pose set before decoding line {line_number}.")
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
                raise ValueError(f"Duplicate pose identity at line {line_number}: {key}")
            coords = np.asarray(payload["keypoints"], dtype=np.float64)
            if coords.shape != (17, 2) or not np.isfinite(coords).all():
                raise ValueError(f"Malformed pose coordinates at line {line_number}: {coords.shape}")
            raw_confidence = payload.get("kp_conf")
            confidence = None if raw_confidence is None else np.asarray(raw_confidence, dtype=np.float64)
            if confidence is not None and (
                confidence.shape != (17,) or not np.isfinite(confidence).all()
            ):
                raise ValueError(f"Malformed pose confidence at line {line_number}.")
            records[key] = PoseRecord(coords, confidence, int(payload["label"]))
    if not records:
        raise ValueError(f"Pose source has no rows for allowed sets {sorted(allowed)}.")
    return records


def validate_amendment_rows(candidate: pd.DataFrame, expected: pd.DataFrame) -> pd.DataFrame:
    """Require exact original row identity, order, labels, and cache positions."""
    identity = (
        "sequence_id",
        "split",
        "set_id",
        "video_id",
        "ped_id",
        "start_frame",
        "end_frame",
        "target_frame",
        "cache_index",
    )
    missing = sorted(set(identity + ("label",)).difference(candidate.columns))
    expected_missing = sorted(set(identity + ("label",)).difference(expected.columns))
    if missing or expected_missing:
        raise ValueError(f"Amendment row schema is incomplete: candidate={missing} expected={expected_missing}")
    checked = candidate.reset_index(drop=True).copy()
    reference = expected.reset_index(drop=True)
    if len(checked) != len(reference):
        raise ValueError("Amendment row population drifted.")
    if checked["sequence_id"].astype(str).tolist() != reference["sequence_id"].astype(str).tolist():
        raise ValueError("Amendment row order drifted from the original manifest.")
    if checked["label"].astype(int).tolist() != reference["label"].astype(int).tolist():
        raise ValueError("Amendment row labels drifted from the original manifest.")
    for name in identity[1:]:
        if checked[name].astype(str).tolist() != reference[name].astype(str).tolist():
            raise ValueError(f"Amendment row identity drifted for {name}.")
    if checked["sequence_id"].duplicated().any():
        raise ValueError("Amendment row IDs are not unique.")
    return checked


def validate_biu_v1_cache_metadata(
    cache_root: Path,
    metadata: Mapping[str, object],
    *,
    expected_rows: int,
) -> None:
    if int(metadata.get("n_rows", -1)) != int(expected_rows):
        raise ValueError("BIU-v1 cache row count drifted.")
    if tuple(metadata.get("feature_names", ())) != BIU_V1_FEATURE_NAMES:
        raise ValueError("BIU-v1 cache feature schema drifted.")
    artifacts = metadata.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != {"biu_v1.npy"}:
        raise ValueError("BIU-v1 cache artifact universe must equal {'biu_v1.npy'}.")
    array_path = Path(cache_root) / "biu_v1.npy"
    if not array_path.is_file() or sha256_file(array_path) != str(artifacts["biu_v1.npy"]):
        raise ValueError("BIU-v1 cache array hash mismatch.")
    array = np.load(array_path, mmap_mode="r")
    if array.shape != (expected_rows, BIU_V1_DIM) or array.dtype != np.float32:
        raise ValueError("BIU-v1 cache array shape or dtype drifted.")


def validate_partial_cache(
    partial_root: Path,
    *,
    expected_rows: int,
    cohort_sha256: str,
) -> Mapping[str, object]:
    progress_path = Path(partial_root) / "progress.json"
    if not progress_path.is_file():
        raise ValueError("Partial BIU-v1 cache is missing bound progress metadata.")
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    if int(progress.get("n_rows", -1)) != int(expected_rows):
        raise ValueError("Partial BIU-v1 cache progress row count drifted.")
    if progress.get("cohort_sha256") != cohort_sha256:
        raise ValueError("Partial BIU-v1 cache progress is bound to another cohort.")
    if tuple(progress.get("feature_names", ())) != BIU_V1_FEATURE_NAMES:
        raise ValueError("Partial BIU-v1 cache progress schema drifted.")
    next_index = int(progress.get("next_index", -1))
    if next_index < 0 or next_index > expected_rows:
        raise ValueError("Partial BIU-v1 cache progress index is invalid.")
    array_path = Path(partial_root) / "biu_v1.npy"
    if not array_path.is_file():
        raise ValueError("Partial BIU-v1 cache array is missing.")
    array = np.load(array_path, mmap_mode="r")
    if array.shape != (expected_rows, BIU_V1_DIM) or array.dtype != np.float32:
        raise ValueError("Partial BIU-v1 cache array contract drifted.")
    return progress


def _validate_original_preparation(output_root: Path) -> tuple[dict[str, object], pd.DataFrame]:
    root = Path(output_root)
    if sha256_file(ORIGINAL_CORE_PATH) != ORIGINAL_CORE_SHA256:
        raise ValueError("Immutable original Exp048 core hash drifted.")
    if sha256_file(ORIGINAL_RUNNER_PATH) != ORIGINAL_RUNNER_SHA256:
        raise ValueError("Immutable original Exp048 runner hash drifted.")
    marker_path = root / ".prepare.complete.json"
    if not marker_path.is_file():
        raise FileNotFoundError("The original authenticated Exp048 preparation marker is missing.")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("phase") != "prepare" or marker.get("set03_model_input_accessed") is not False:
        raise ValueError("Original Exp048 preparation phase or set03 seal drifted.")
    artifacts = marker.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("Original Exp048 preparation lacks an artifact manifest.")
    for relative, expected_hash in artifacts.items():
        path = Path(relative) if Path(relative).is_absolute() else root / relative
        if not path.is_file() or sha256_file(path) != str(expected_hash):
            raise ValueError(f"Original Exp048 preparation artifact hash mismatch: {path}")
    if artifacts.get("scripts/exp048_core.py") != ORIGINAL_CORE_SHA256:
        raise ValueError("Original preparation does not bind the immutable core.")
    if artifacts.get("scripts/run_experiment.py") != ORIGINAL_RUNNER_SHA256:
        raise ValueError("Original preparation does not bind the immutable runner.")

    cache_root = root / "cache" / "trainval_native_streams"
    metadata_path = cache_root / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    cache_artifacts = metadata.get("artifacts")
    if not isinstance(cache_artifacts, Mapping) or "rows.csv" not in cache_artifacts:
        raise ValueError("Original prepared cache lacks its row binding.")
    rows_path = cache_root / "rows.csv"
    if sha256_file(rows_path) != str(cache_artifacts["rows.csv"]):
        raise ValueError("Original prepared cache row hash drifted.")
    cohort_path = root / "metrics" / "trainval_cohort_manifest.csv"
    prepared_rows = pd.read_csv(rows_path)
    cohort_rows = pd.read_csv(cohort_path)
    validate_amendment_rows(prepared_rows, cohort_rows)
    if len(prepared_rows) != EXPECTED_TRAINVAL_ROWS:
        raise ValueError("Original train/validation population drifted.")
    if prepared_rows["split"].value_counts().astype(int).to_dict() != {
        "train": 333_454,
        "val": 77_288,
    }:
        raise ValueError("Original train/validation split counts drifted.")
    if set(prepared_rows["set_id"].astype(str)) != set(ALLOWED_TRAINVAL_SETS):
        raise ValueError("Original train/validation PIE set universe drifted.")
    return dict(marker), prepared_rows


def _quality_summary(
    rows: pd.DataFrame,
    values: np.ndarray,
    *,
    splits: tuple[str, ...] = ("train", "val"),
) -> pd.DataFrame:
    result = []
    for split in splits:
        mask = rows["split"].astype(str).to_numpy() == split
        if not mask.any():
            raise ValueError(f"BIU-v1 quality summary split is empty: {split}")
        valid = np.asarray(values[mask, -2], dtype=np.float64)
        low = np.asarray(values[mask, -1], dtype=np.float64)
        result.append(
            {
                "split": split,
                "rows": int(mask.sum()),
                "valid_count_min": float(valid.min()),
                "valid_count_mean": float(valid.mean()),
                "valid_count_median": float(np.median(valid)),
                "valid_count_max": float(valid.max()),
                "valid_count_nonzero_rows": int((valid > 0).sum()),
                "low_confidence_count_min": float(low.min()),
                "low_confidence_count_mean": float(low.mean()),
                "low_confidence_count_median": float(np.median(low)),
                "low_confidence_count_max": float(low.max()),
                "low_confidence_count_nonzero_rows": int((low > 0).sum()),
            }
        )
    return pd.DataFrame(result)


def authenticate_biu_v1_audit(
    output_root: Path,
    *,
    expected_rows: int = EXPECTED_TRAINVAL_ROWS,
) -> dict[str, object]:
    root = Path(output_root)
    marker_path = root / AUDIT_MARKER_NAME
    if not marker_path.is_file():
        raise FileNotFoundError("BIU-v1 amendment audit marker is missing.")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("phase") != "biu_v1_amendment_audit":
        raise ValueError("BIU-v1 amendment audit phase drifted.")
    if int(marker.get("n_rows", -1)) != int(expected_rows):
        raise ValueError("BIU-v1 amendment audit population drifted.")
    if marker.get("set03_model_input_accessed") is not False:
        raise ValueError("BIU-v1 amendment audit violated the set03 seal.")
    prepare_path = root / ".prepare.complete.json"
    if marker.get("original_prepare_sha256") != sha256_file(prepare_path):
        raise ValueError("BIU-v1 audit is bound to another original preparation.")
    current_bindings = {
        "original_core_sha256": sha256_file(ORIGINAL_CORE_PATH),
        "original_runner_sha256": sha256_file(ORIGINAL_RUNNER_PATH),
        "amendment_core_sha256": sha256_file(SCRIPT_ROOT / "biu_v1_amendment_core.py"),
        "amendment_runner_sha256": sha256_file(Path(__file__).resolve()),
        "pose_sha256": sha256_file(EXP012_POSE_PATH),
    }
    expected_bindings = {
        "original_core_sha256": ORIGINAL_CORE_SHA256,
        "original_runner_sha256": ORIGINAL_RUNNER_SHA256,
        "pose_sha256": EXP012_POSE_SHA256,
    }
    for name, expected_hash in expected_bindings.items():
        if current_bindings[name] != expected_hash:
            raise ValueError(f"Current source hash drifted for {name}.")
    for name, current_hash in current_bindings.items():
        if marker.get(name) != current_hash:
            raise ValueError(f"BIU-v1 audit source binding drifted for {name}.")
    artifacts = marker.get("artifacts")
    expected_artifacts = {
        f"cache/{CACHE_NAME}/metadata.json",
        "metrics/biu_v1_quality_counts_trainval.csv",
    }
    if not isinstance(artifacts, Mapping) or set(artifacts) != expected_artifacts:
        raise ValueError("BIU-v1 amendment audit artifact universe drifted.")
    for relative, expected_hash in artifacts.items():
        path = root / relative
        if not path.is_file() or sha256_file(path) != str(expected_hash):
            raise ValueError(f"BIU-v1 amendment audit artifact hash mismatch: {path}")
    cache_root = root / "cache" / CACHE_NAME
    metadata = json.loads((cache_root / "metadata.json").read_text(encoding="utf-8"))
    validate_biu_v1_cache_metadata(cache_root, metadata, expected_rows=expected_rows)
    return dict(marker)


def build_biu_v1_cache(output_root: Path, *, resume: bool) -> None:
    root = Path(output_root)
    log = progress_logger(root / "logs" / "biu_v1_amendment_audit.log")
    marker_path = root / AUDIT_MARKER_NAME
    if marker_path.is_file():
        authenticate_biu_v1_audit(root)
        log("phase=audit status=already_complete_authenticated")
        return
    original_prepare, rows = _validate_original_preparation(root)
    if sha256_file(EXP012_POSE_PATH) != EXP012_POSE_SHA256:
        raise ValueError("SHA-pinned Exp012 pose source drifted.")
    cache_root = root / "cache" / CACHE_NAME
    partial_root = root / "cache" / f"{CACHE_NAME}.partial"
    cohort_path = root / "metrics" / "trainval_cohort_manifest.csv"
    cohort_sha256 = sha256_file(cohort_path)
    if cache_root.exists():
        raise FileExistsError("Complete BIU-v1 cache exists without an authenticated audit marker.")

    if partial_root.exists():
        if not resume:
            raise FileExistsError("Partial BIU-v1 cache exists; rerun audit with --resume.")
        progress = validate_partial_cache(
            partial_root,
            expected_rows=len(rows),
            cohort_sha256=cohort_sha256,
        )
        start_index = int(progress["next_index"])
        array = np.load(partial_root / "biu_v1.npy", mmap_mode="r+")
    else:
        partial_root.mkdir(parents=True)
        start_index = 0
        array = np.lib.format.open_memmap(
            partial_root / "biu_v1.npy",
            mode="w+",
            dtype=np.float32,
            shape=(len(rows), BIU_V1_DIM),
        )
        atomic_json(
            {
                "next_index": 0,
                "n_rows": len(rows),
                "cohort_sha256": cohort_sha256,
                "feature_names": list(BIU_V1_FEATURE_NAMES),
            },
            partial_root / "progress.json",
        )

    log(f"phase=audit status=loading_pose allowed_sets={sorted(ALLOWED_TRAINVAL_SETS)}")
    pose_records = load_allowed_pose_records(EXP012_POSE_PATH, ALLOWED_TRAINVAL_SETS)
    log(f"phase=audit pose_records={len(pose_records)} cache_start={start_index}/{len(rows)}")
    for position in range(start_index, len(rows)):
        row = rows.iloc[position]
        cache_index = int(row.cache_index)
        if cache_index != position:
            raise ValueError("Original cache index/order drifted during BIU-v1 construction.")
        frame_ids = range(int(row.start_frame), int(row.end_frame) + 1)
        keys = [
            (str(row.set_id), str(row.video_id), str(row.ped_id), int(frame_id))
            for frame_id in frame_ids
        ]
        if len(keys) != 30:
            raise ValueError(f"BIU-v1 row does not contain exactly 30 frames: {row.sequence_id}")
        missing = [key for key in keys if key not in pose_records]
        if missing:
            raise ValueError(f"Missing exact Exp012 pose frames for {row.sequence_id}: {missing[:3]}")
        records = [pose_records[key] for key in keys]
        coords = np.stack([record.coords for record in records])
        confidence = np.stack(
            [
                np.ones(17, dtype=np.float64) if record.confidence is None else record.confidence
                for record in records
            ]
        )
        values, names = extract_biu_v1_window(coords, confidence)
        if names != BIU_V1_FEATURE_NAMES or values.shape != (BIU_V1_DIM,):
            raise RuntimeError("Frozen BIU-v1 extraction schema drifted during cache construction.")
        array[cache_index] = values
        if (position + 1) % 5_000 == 0 or position + 1 == len(rows):
            array.flush()
            atomic_json(
                {
                    "next_index": position + 1,
                    "n_rows": len(rows),
                    "cohort_sha256": cohort_sha256,
                    "feature_names": list(BIU_V1_FEATURE_NAMES),
                },
                partial_root / "progress.json",
            )
            log(f"phase=audit cache_rows={position+1}/{len(rows)} split={row.split}")

    array.flush()
    quality_path = root / "metrics" / "biu_v1_quality_counts_trainval.csv"
    atomic_csv(_quality_summary(rows, array), quality_path)
    metadata = {
        "created_at": utc_now(),
        "n_rows": len(rows),
        "splits": {"train": 333_454, "val": 77_288},
        "feature_names": list(BIU_V1_FEATURE_NAMES),
        "pose_sha256": EXP012_POSE_SHA256,
        "cohort_sha256": cohort_sha256,
        "set03_model_input_accessed": False,
        "artifacts": {"biu_v1.npy": sha256_file(partial_root / "biu_v1.npy")},
    }
    (partial_root / "progress.json").unlink()
    atomic_json(metadata, partial_root / "metadata.json")
    partial_root.replace(cache_root)
    payload = {
        "phase": "biu_v1_amendment_audit",
        "created_at": utc_now(),
        "n_rows": len(rows),
        "set03_model_input_accessed": False,
        "original_prepare_sha256": sha256_file(root / ".prepare.complete.json"),
        "original_prepare_phase": original_prepare["phase"],
        "original_core_sha256": sha256_file(ORIGINAL_CORE_PATH),
        "original_runner_sha256": sha256_file(ORIGINAL_RUNNER_PATH),
        "amendment_core_sha256": sha256_file(SCRIPT_ROOT / "biu_v1_amendment_core.py"),
        "amendment_runner_sha256": sha256_file(Path(__file__).resolve()),
        "pose_sha256": sha256_file(EXP012_POSE_PATH),
        "artifacts": {
            f"cache/{CACHE_NAME}/metadata.json": sha256_file(cache_root / "metadata.json"),
            "metrics/biu_v1_quality_counts_trainval.csv": sha256_file(quality_path),
        },
    }
    atomic_json(payload, marker_path)
    authenticate_biu_v1_audit(root)
    log("phase=audit status=complete set03_model_input_accessed=false")


def select_f_best_epoch(history: pd.DataFrame) -> int:
    if len(history) != F_EPOCHS:
        raise ValueError(f"F checkpoint selection requires exactly {F_EPOCHS} epochs.")
    required = {"epoch", "val_f1_at_0p5"}
    if not required.issubset(history.columns):
        raise ValueError(f"F checkpoint history requires columns {sorted(required)}.")
    scores = history["val_f1_at_0p5"].to_numpy(dtype=np.float64)
    if not np.isfinite(scores).all():
        raise ValueError("F validation F1 history must be finite.")
    return int(history.iloc[int(np.argmax(scores))]["epoch"])


def authenticate_f_training_marker(output_root: Path, seed: int) -> dict[str, object]:
    root = Path(output_root)
    if int(seed) not in F_SEEDS:
        raise ValueError(f"F seed must be one of {F_SEEDS}.")
    marker_path = root / f".train_F_seed{seed}.complete.json"
    if not marker_path.is_file():
        raise FileNotFoundError(f"Variant F training marker is missing: {marker_path}")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if (
        marker.get("phase") != "train"
        or marker.get("variant") != "F"
        or int(marker.get("seed", -1)) != int(seed)
    ):
        raise ValueError(f"Variant F training marker identity drifted: {marker_path}")
    bindings = {
        "prepare_sha256": root / ".prepare.complete.json",
        "amendment_audit_sha256": root / AUDIT_MARKER_NAME,
        "cache_metadata_sha256": root / "cache" / CACHE_NAME / "metadata.json",
        "sampling_manifest_sha256": root / "metrics" / f"oversampling_manifest_seed{seed}.csv.gz",
    }
    labels = {
        "prepare_sha256": "original preparation",
        "amendment_audit_sha256": "amendment audit",
        "cache_metadata_sha256": "BIU-v1 cache metadata",
        "sampling_manifest_sha256": "oversampling manifest",
    }
    for field, path in bindings.items():
        if not path.is_file() or marker.get(field) != sha256_file(path):
            raise ValueError(f"Variant F marker {labels[field]} binding drifted.")
    artifacts = marker.get("artifacts")
    expected_artifacts = {
        f"checkpoints/F_seed{seed}.best.pt",
        f"predictions/validation_F_seed{seed}.csv",
        f"logs/training_history_F_seed{seed}.csv",
    }
    if not isinstance(artifacts, Mapping) or set(artifacts) != expected_artifacts:
        raise ValueError("Variant F training artifact universe drifted.")
    for relative, expected_hash in artifacts.items():
        path = root / relative
        if not path.is_file() or sha256_file(path) != str(expected_hash):
            raise ValueError(f"Variant F training artifact hash mismatch: {path}")
    if (
        int(marker.get("total_parameters", -1)) != 8_947_083
        or int(marker.get("trainable_parameters", -1)) != 8_947_083
        or int(marker.get("pose_pathway_parameters", -1)) != 75_392
    ):
        raise ValueError("Variant F training parameter counts drifted.")
    return dict(marker)


def validate_amended_training_identities(
    original_records: list[Mapping[str, object]],
    amendment_records: list[Mapping[str, object]],
) -> list[dict[str, object]]:
    expected = {
        *((variant, seed) for variant in ("A", "B", "C", "D", "E") for seed in F_SEEDS),
        *(("F", seed) for seed in F_SEEDS),
    }
    records = [dict(record) for record in (*original_records, *amendment_records)]
    if len(records) != 18:
        raise ValueError("Amended evaluation requires exactly 18 training records.")
    identities = [(str(record.get("variant")), int(record.get("seed", -1))) for record in records]
    if len(set(identities)) != len(identities):
        raise ValueError("Amended training records contain a duplicate identity.")
    if set(identities) != expected:
        raise ValueError("Amended training records do not contain the exact 18-run A-F grid.")
    indexed = {identity: record for identity, record in zip(identities, records)}
    return [
        indexed[(variant, seed)]
        for variant in ("A", "B", "C", "D", "E", "F")
        for seed in F_SEEDS
    ]


def load_original_runner():
    module_name = "exp048_original_runner_for_biu_v1_amendment"
    if module_name in sys.modules:
        return sys.modules[module_name]
    original_core = load_original_core()
    previous = sys.modules.get("exp048_core")
    sys.modules["exp048_core"] = original_core
    try:
        spec = importlib.util.spec_from_file_location(module_name, ORIGINAL_RUNNER_PATH)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot import immutable Exp048 runner: {ORIGINAL_RUNNER_PATH}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    finally:
        if previous is None:
            sys.modules.pop("exp048_core", None)
        else:
            sys.modules["exp048_core"] = previous
    return module


def _f_state_bindings(root: Path, seed: int) -> dict[str, str]:
    return {
        "prepare_sha256": sha256_file(root / ".prepare.complete.json"),
        "amendment_audit_sha256": sha256_file(root / AUDIT_MARKER_NAME),
        "cache_metadata_sha256": sha256_file(root / "cache" / CACHE_NAME / "metadata.json"),
        "sampling_manifest_sha256": sha256_file(
            root / "metrics" / f"oversampling_manifest_seed{seed}.csv.gz"
        ),
    }


def _validate_f_state_bindings(state: Mapping[str, object], expected: Mapping[str, str]) -> None:
    labels = {
        "prepare_sha256": "original preparation",
        "amendment_audit_sha256": "amendment audit",
        "cache_metadata_sha256": "BIU-v1 cache metadata",
        "sampling_manifest_sha256": "oversampling manifest",
    }
    for field, expected_hash in expected.items():
        if state.get(field) != expected_hash:
            raise ValueError(f"Resumable F state is bound to another {labels[field]}.")


def run_f_training(
    output_root: Path,
    *,
    seed: int,
    resume: bool,
    prepared: tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, object]] | None = None,
) -> None:
    root = Path(output_root)
    if int(seed) not in F_SEEDS:
        raise ValueError(f"F seed must be one of {F_SEEDS}.")
    marker_path = root / f".train_F_seed{seed}.complete.json"
    log = progress_logger(root / "logs" / f"train_F_seed{seed}.log")
    if marker_path.is_file():
        authenticate_f_training_marker(root, seed)
        log("status=already_complete_authenticated")
        return
    authenticate_biu_v1_audit(root)
    original = load_original_runner()
    rows, original_arrays, _ = (
        prepared if prepared is not None else original.load_prepared_cache(root)
    )
    biu_path = root / "cache" / CACHE_NAME / "biu_v1.npy"
    biu_values = np.load(biu_path, mmap_mode="r")
    if biu_values.shape != (len(rows), BIU_V1_DIM) or biu_values.dtype != np.float32:
        raise ValueError("Authenticated BIU-v1 cache does not align with original prepared rows.")
    arrays = {**original_arrays, "biu_v1": biu_values}
    train_rows = rows.loc[rows["split"] == "train"].reset_index(drop=True)
    val_rows = rows.loc[rows["split"] == "val"].reset_index(drop=True)
    sampling_path = root / "metrics" / f"oversampling_manifest_seed{seed}.csv.gz"
    sampling = pd.read_csv(sampling_path)
    if len(sampling) != 556_974:
        raise ValueError(f"F oversampling population drifted for seed {seed}.")
    if sampling["label"].value_counts().astype(int).to_dict() != {0: 278_487, 1: 278_487}:
        raise ValueError(f"F oversampling class counts drifted for seed {seed}.")
    train_dataset = BiuV1AmendmentDataset(
        train_rows,
        arrays,
        sampling_manifest=sampling,
        training_phase=True,
    )
    val_dataset = BiuV1AmendmentDataset(val_rows, arrays, training_phase=True)
    original.seed_everything(seed)
    generator = torch.Generator().manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(
        train_dataset,
        batch_size=F_BATCH_SIZE,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=F_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    model = build_f_model(seed).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=F_LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(weight=torch.ones(2, device=device))
    state_path = root / "checkpoints" / f"F_seed{seed}.state.pt"
    history: list[dict[str, float | int]] = []
    best_f1 = -1.0
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    start_epoch = 1
    bindings = _f_state_bindings(root, seed)
    if state_path.is_file():
        if not resume:
            raise FileExistsError(f"Resumable F state exists; rerun with --resume: {state_path}")
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if state.get("variant") != "F" or int(state.get("seed", -1)) != seed:
            raise ValueError("Resumable F state identity drifted.")
        _validate_f_state_bindings(state, bindings)
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
            f"phase=train status=starting variant=F seed={seed} device={device} "
            f"train_rows={len(train_dataset)} val_rows={len(val_dataset)}"
        )

    for epoch in range(start_epoch, F_EPOCHS + 1):
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_count = 0
        for inputs, batch_labels, _ in train_loader:
            moved = {
                name: value.to(device, non_blocking=device.type == "cuda")
                for name, value in inputs.items()
            }
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
        val_labels, val_probabilities, val_loss = original.predict_loader(model, val_loader, device)
        val_metrics = original.compute_binary_metrics(val_labels, val_probabilities)
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
        if float(val_metrics["f1"]) > best_f1:
            best_f1 = float(val_metrics["f1"])
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
        original.atomic_torch_save(
            {
                "variant": "F",
                "seed": seed,
                **bindings,
                "last_epoch": epoch,
                "model_state": {
                    name: value.detach().cpu() for name, value in model.state_dict().items()
                },
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
        history_path = root / "logs" / f"training_history_F_seed{seed}.csv"
        atomic_csv(pd.DataFrame(history), history_path)
        log(
            f"epoch={epoch}/{F_EPOCHS} train_loss={train_loss_sum/train_count:.6f} "
            f"train_accuracy={train_correct/train_count:.6f} "
            f"val_f1_at_0p5={val_metrics['f1']:.6f} val_pr_auc={val_metrics['pr_auc']:.6f} "
            f"best_epoch={best_epoch}"
        )
    if len(history) != F_EPOCHS or best_state is None:
        raise RuntimeError("Variant F training did not complete all 15 frozen epochs.")
    if select_f_best_epoch(pd.DataFrame(history)) != best_epoch:
        raise RuntimeError("Variant F checkpoint selection disagrees with the frozen earliest-tie rule.")
    model.load_state_dict(best_state)
    val_labels, val_probabilities, _ = original.predict_loader(model, val_loader, device)
    validation_metrics = original.compute_binary_metrics(val_labels, val_probabilities)
    checkpoint_path = root / "checkpoints" / f"F_seed{seed}.best.pt"
    total_parameters = int(sum(parameter.numel() for parameter in model.parameters()))
    trainable_parameters = int(
        sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    )
    pathway_parameters = int(sum(parameter.numel() for parameter in model.biu_projection.parameters()))
    if total_parameters != 8_947_083 or pathway_parameters != 75_392:
        raise RuntimeError("Variant F parameter counts drifted from the approved architecture.")
    original.atomic_torch_save(
        {
            "variant": "F",
            "seed": seed,
            "best_epoch": best_epoch,
            "best_validation_f1": best_f1,
            "model_state": best_state,
            **bindings,
            "total_parameters": total_parameters,
            "trainable_parameters": trainable_parameters,
            "pose_pathway_parameters": pathway_parameters,
        },
        checkpoint_path,
    )
    prediction_path = root / "predictions" / f"validation_F_seed{seed}.csv"
    prediction = val_rows[
        [
            "sequence_id",
            "set_id",
            "video_id",
            "ped_id",
            "start_frame",
            "end_frame",
            "target_frame",
            "label",
        ]
    ].copy()
    prediction["variant"] = "F"
    prediction["seed"] = seed
    prediction["threshold"] = F_FIXED_THRESHOLD
    prediction["probability"] = val_probabilities
    prediction["prediction"] = (val_probabilities >= F_FIXED_THRESHOLD).astype(np.int64)
    atomic_csv(prediction, prediction_path)
    history_path = root / "logs" / f"training_history_F_seed{seed}.csv"
    marker = {
        "phase": "train",
        "created_at": utc_now(),
        "variant": "F",
        "seed": seed,
        "device": str(device),
        "best_epoch": best_epoch,
        "best_validation_f1": best_f1,
        "validation_metrics": validation_metrics,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "pose_pathway_parameters": pathway_parameters,
        **bindings,
        "artifacts": {
            str(path.relative_to(root)): sha256_file(path)
            for path in (checkpoint_path, prediction_path, history_path)
        },
    }
    atomic_json(marker, marker_path)
    authenticate_f_training_marker(root, seed)
    state_path.unlink(missing_ok=True)
    log(f"phase=train status=complete best_epoch={best_epoch} best_val_f1={best_f1:.6f}")


def authenticate_amended_training_grid(output_root: Path) -> list[dict[str, object]]:
    root = Path(output_root)
    authenticate_biu_v1_audit(root)
    original_records = load_original_runner().authenticate_training_grid(root)
    amendment_records = [authenticate_f_training_marker(root, seed) for seed in F_SEEDS]
    return validate_amended_training_identities(original_records, amendment_records)


def run_f_train_all(output_root: Path, *, resume: bool) -> None:
    root = Path(output_root)
    authenticate_biu_v1_audit(root)
    prepared = load_original_runner().load_prepared_cache(root)
    for seed in F_SEEDS:
        run_f_training(root, seed=seed, resume=resume, prepared=prepared)
    authenticate_amended_training_grid(root)
    (root / "TRAINING_AMENDED_COMPLETE").write_text(
        f"completed_at={utc_now()}\nrun_count=18\n", encoding="utf-8"
    )


def _validate_af_results(results: pd.DataFrame) -> pd.DataFrame:
    required = {"variant", "seed", "f1", "pr_auc"}
    if not required.issubset(results.columns):
        raise ValueError(f"A-F results require columns {sorted(required)}.")
    checked = results.copy()
    identities = list(zip(checked["variant"].astype(str), checked["seed"].astype(int)))
    expected = [(variant, seed) for variant in AF_VARIANTS for seed in F_SEEDS]
    if identities != expected or len(set(identities)) != 18:
        raise ValueError("A-F results require the exact ordered 18-run grid.")
    return checked


def build_amended_paired_deltas(per_seed_results: pd.DataFrame) -> pd.DataFrame:
    checked = _validate_af_results(per_seed_results).set_index(["variant", "seed"])
    comparisons = (
        ("A_to_B", "A", "B"),
        ("B_to_C", "B", "C"),
        ("C_to_D", "C", "D"),
        ("C_to_E", "C", "E"),
        ("D_to_E", "D", "E"),
        ("A_to_E", "A", "E"),
        ("C_to_F", "C", "F"),
        ("D_to_F", "D", "F"),
        ("E_to_F", "E", "F"),
        ("A_to_F", "A", "F"),
    )
    rows = []
    for comparison, baseline, candidate in comparisons:
        for seed in F_SEEDS:
            source = checked.loc[(baseline, seed)]
            target = checked.loc[(candidate, seed)]
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


def build_amended_behavior_recovery(per_seed_results: pd.DataFrame) -> pd.DataFrame:
    checked = _validate_af_results(per_seed_results)
    output = []
    for metric in ("f1", "pr_auc"):
        pivot = checked.pivot(index="seed", columns="variant", values=metric)
        gt_gain = pivot["B"] - pivot["C"]
        raw_recovery = pivot["D"] - pivot["C"]
        biu_v2_recovery = pivot["E"] - pivot["C"]
        biu_v1_recovery = pivot["F"] - pivot["C"]
        denominator = gt_gain.to_numpy(dtype=np.float64)
        v2_fraction = np.where(denominator > 0, biu_v2_recovery / gt_gain, np.nan)
        v1_fraction = np.where(denominator > 0, biu_v1_recovery / gt_gain, np.nan)
        v2_finite = np.asarray(v2_fraction, dtype=np.float64)
        v1_finite = np.asarray(v1_fraction, dtype=np.float64)
        output.append(
            {
                "metric": metric,
                "gt_behavior_gain_mean": float(gt_gain.mean()),
                "raw_pose_recovery_mean": float(raw_recovery.mean()),
                "biu_v2_recovery_mean": float(biu_v2_recovery.mean()),
                "biu_v1_recovery_mean": float(biu_v1_recovery.mean()),
                "biu_v2_recovery_fraction_mean": (
                    float(np.nanmean(v2_finite)) if np.isfinite(v2_finite).any() else float("nan")
                ),
                "biu_v1_recovery_fraction_mean": (
                    float(np.nanmean(v1_finite)) if np.isfinite(v1_finite).any() else float("nan")
                ),
            }
        )
    return pd.DataFrame(output)


def _amended_summary(results: pd.DataFrame) -> pd.DataFrame:
    checked = _validate_af_results(results)
    metrics = ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc")
    if not set(metrics).issubset(checked.columns):
        raise ValueError("A-F summary metric columns are incomplete.")
    rows = []
    for variant in AF_VARIANTS:
        selected = checked.loc[checked["variant"] == variant]
        row: dict[str, object] = {"variant": variant, "n_seeds": 3}
        for metric in metrics:
            row[f"{metric}_mean"] = float(selected[metric].mean())
            row[f"{metric}_std"] = float(selected[metric].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def expected_amended_artifact_relatives() -> set[str]:
    relatives = {f"metrics/{name}" for name in AMENDED_METRIC_FILES}
    relatives.update(
        f"predictions/set03_{variant}_seed{seed}.csv"
        for variant in AF_VARIANTS
        for seed in F_SEEDS
    )
    relatives.update(
        {
            "cache/biu_v1_amendment_set03/metadata.json",
            "cache/set03_yolop/metadata.json",
        }
    )
    return relatives


def authenticate_amended_evaluation(output_root: Path) -> dict[str, object]:
    root = Path(output_root)
    if (root / ".evaluate.complete.json").exists():
        raise ValueError("Prohibited original 15-run evaluation marker is present.")
    marker_path = root / ".evaluate_amended.complete.json"
    if not marker_path.is_file():
        raise FileNotFoundError("Amended A-F evaluation marker is missing.")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if (
        marker.get("phase") != "evaluate_amended"
        or int(marker.get("training_record_count", -1)) != 18
        or int(marker.get("set03_evaluations", -1)) != 18
    ):
        raise ValueError("Amended A-F evaluation identity/counts drifted.")
    artifacts = marker.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != expected_amended_artifact_relatives():
        raise ValueError("Amended evaluation artifact universe is incomplete or unexpected.")
    for relative, expected_hash in artifacts.items():
        path = root / relative
        if not path.is_file() or sha256_file(path) != str(expected_hash):
            raise ValueError(f"Amended evaluation artifact hash mismatch: {path}")
    training_hashes = marker.get("training_marker_hashes")
    expected_training = {
        f"{variant}_{seed}" for variant in AF_VARIANTS for seed in F_SEEDS
    }
    if not isinstance(training_hashes, Mapping) or set(training_hashes) != expected_training:
        raise ValueError("Amended evaluation training-marker universe drifted.")
    for identity, expected_hash in training_hashes.items():
        variant, seed_text = identity.rsplit("_", 1)
        path = root / f".train_{variant}_seed{int(seed_text)}.complete.json"
        if not path.is_file() or sha256_file(path) != str(expected_hash):
            raise ValueError(f"Amended evaluation training marker drifted: {identity}")
    source_bindings = {
        "amendment_audit_sha256": root / AUDIT_MARKER_NAME,
        "amendment_core_sha256": SCRIPT_ROOT / "biu_v1_amendment_core.py",
        "amendment_runner_sha256": Path(__file__).resolve(),
    }
    for field, path in source_bindings.items():
        if marker.get(field) != sha256_file(path):
            raise ValueError(f"Amended evaluation source binding drifted: {field}")
    return dict(marker)


def _build_set03_biu_v1(
    output_root: Path,
    rows: pd.DataFrame,
    *,
    resume: bool,
    progress,
) -> tuple[np.ndarray, dict[str, object]]:
    root = Path(output_root)
    cache_root = root / "cache" / "biu_v1_amendment_set03"
    partial_root = root / "cache" / "biu_v1_amendment_set03.partial"
    metadata_path = cache_root / "metadata.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        validate_biu_v1_cache_metadata(cache_root, metadata, expected_rows=len(rows))
        return np.load(cache_root / "biu_v1.npy", mmap_mode="r"), metadata
    cohort_sha256 = sha256_file(load_original_runner().EXP047_SET03_CACHE / "rows.csv")
    if partial_root.exists():
        if not resume:
            raise FileExistsError("Partial set03 BIU-v1 cache exists; rerun evaluate with --resume.")
        state = validate_partial_cache(
            partial_root,
            expected_rows=len(rows),
            cohort_sha256=cohort_sha256,
        )
        start_index = int(state["next_index"])
        values = np.load(partial_root / "biu_v1.npy", mmap_mode="r+")
    else:
        partial_root.mkdir(parents=True)
        start_index = 0
        values = np.lib.format.open_memmap(
            partial_root / "biu_v1.npy",
            mode="w+",
            dtype=np.float32,
            shape=(len(rows), BIU_V1_DIM),
        )
        atomic_json(
            {
                "next_index": 0,
                "n_rows": len(rows),
                "cohort_sha256": cohort_sha256,
                "feature_names": list(BIU_V1_FEATURE_NAMES),
            },
            partial_root / "progress.json",
        )
    progress(f"set03_biu_v1 loading_pose cache_start={start_index}/{len(rows)}")
    pose_records = load_allowed_pose_records(EXP012_POSE_PATH, {"set03"})
    progress(f"set03_biu_v1 pose_records={len(pose_records)}")
    for position in range(start_index, len(rows)):
        row = rows.iloc[position]
        if int(row.cache_index) != position:
            raise ValueError("Set03 cache index/order drifted during BIU-v1 construction.")
        keys = [
            (str(row.set_id), str(row.video_id), str(row.ped_id), frame)
            for frame in range(int(row.start_frame), int(row.end_frame) + 1)
        ]
        if len(keys) != 30:
            raise ValueError(f"Set03 BIU-v1 row does not contain 30 frames: {row.sequence_id}")
        missing = [key for key in keys if key not in pose_records]
        if missing:
            raise ValueError(f"Missing exact set03 pose frames for {row.sequence_id}: {missing[:3]}")
        records = [pose_records[key] for key in keys]
        coords = np.stack([record.coords for record in records])
        confidence = np.stack(
            [
                np.ones(17, dtype=np.float64) if record.confidence is None else record.confidence
                for record in records
            ]
        )
        extracted, names = extract_biu_v1_window(coords, confidence)
        if names != BIU_V1_FEATURE_NAMES:
            raise RuntimeError("Set03 BIU-v1 feature ordering drifted.")
        values[position] = extracted
        if (position + 1) % 5_000 == 0 or position + 1 == len(rows):
            values.flush()
            atomic_json(
                {
                    "next_index": position + 1,
                    "n_rows": len(rows),
                    "cohort_sha256": cohort_sha256,
                    "feature_names": list(BIU_V1_FEATURE_NAMES),
                },
                partial_root / "progress.json",
            )
            progress(f"set03_biu_v1_rows={position+1}/{len(rows)}")
    values.flush()
    metadata = {
        "created_at": utc_now(),
        "n_rows": len(rows),
        "splits": {"test": len(rows)},
        "feature_names": list(BIU_V1_FEATURE_NAMES),
        "pose_sha256": EXP012_POSE_SHA256,
        "cohort_sha256": cohort_sha256,
        "set03_access_authorized_after_training_records": 18,
        "artifacts": {"biu_v1.npy": sha256_file(partial_root / "biu_v1.npy")},
    }
    (partial_root / "progress.json").unlink()
    atomic_json(metadata, partial_root / "metadata.json")
    partial_root.replace(cache_root)
    validate_biu_v1_cache_metadata(cache_root, metadata, expected_rows=len(rows))
    return np.load(cache_root / "biu_v1.npy", mmap_mode="r"), metadata


def load_amended_set03(
    output_root: Path,
    *,
    resume: bool,
    progress,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, object]]:
    original = load_original_runner()
    exp047_rows, arrays, exp047_metadata = original._load_authenticated_exp047_set03()
    protocol_rows = pd.read_csv(original.EXP032_PROTOCOL_ROWS)
    expected = protocol_rows.loc[protocol_rows["split"] == "test"].reset_index(drop=True)
    identity = ("sequence_id", "label", "start_frame", "end_frame", "target_frame")
    if len(expected) != original.PROTOCOL.test_rows or any(
        expected[name].astype(str).tolist() != exp047_rows[name].astype(str).tolist()
        for name in identity
    ):
        raise ValueError("Set03 cohort does not exactly match the frozen protocol identity/order.")
    rows = exp047_rows.reset_index(drop=True).copy()
    rows["cache_index"] = np.arange(len(rows), dtype=np.int64)
    yolop, yolop_metadata = original._build_set03_yolop(
        output_root,
        rows,
        resume=resume,
        progress=progress,
    )
    biu_v1, biu_v1_metadata = _build_set03_biu_v1(
        output_root,
        rows,
        resume=resume,
        progress=progress,
    )
    combined = {**arrays, "yolop": yolop, "biu_v1": biu_v1}
    if any(len(array) != len(rows) for array in combined.values()):
        raise ValueError("Amended set03 stream arrays do not share one population.")
    return rows, combined, {
        "exp047": exp047_metadata,
        "yolop": yolop_metadata,
        "biu_v1": biu_v1_metadata,
    }


def _measure_f_pathway_latency(
    model: nn.Module,
    features: Mapping[str, torch.Tensor],
    device: torch.device,
) -> float:
    import time

    values = features["biu_v1"].to(device)
    model.eval()
    with torch.no_grad():
        for _ in range(2):
            model.biu_projection(values)
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(5):
            model.biu_projection(values)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
    return elapsed / 5.0 / int(values.shape[0])


def _measure_biu_v1_extraction_latency(rows: pd.DataFrame, arrays: Mapping[str, np.ndarray]) -> float:
    import time

    sample_count = min(100, len(rows))
    started = time.perf_counter()
    for index in range(sample_count):
        row = rows.iloc[index]
        pose = np.asarray(arrays["exp012_pose"][index], dtype=np.float64).reshape(30, 17, 2)
        pose = pose.copy()
        pose[:, :, 0] *= float(row.frame_width)
        pose[:, :, 1] *= float(row.frame_height)
        extract_biu_v1_window(pose, kp_conf=None)
    return (time.perf_counter() - started) / sample_count


def _amended_error_transitions(
    predictions: Mapping[tuple[str, int], pd.DataFrame],
) -> pd.DataFrame:
    comparisons = (
        ("A", "B"),
        ("B", "C"),
        ("C", "D"),
        ("C", "E"),
        ("D", "E"),
        ("A", "E"),
        ("C", "F"),
        ("D", "F"),
        ("E", "F"),
        ("A", "F"),
    )
    rows = []
    for baseline, candidate in comparisons:
        for seed in F_SEEDS:
            left = predictions[(baseline, seed)]
            right = predictions[(candidate, seed)]
            if left["sequence_id"].tolist() != right["sequence_id"].tolist():
                raise ValueError("Amended error-transition cohorts are not aligned.")
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


def _amended_success_criteria(
    summary: pd.DataFrame,
    deltas: pd.DataFrame,
    recovery: pd.DataFrame,
) -> pd.DataFrame:
    indexed = summary.set_index("variant")
    recovery_index = recovery.set_index("metric")
    e_to_f = deltas.loc[deltas["comparison"] == "E_to_F"]
    d_to_e = deltas.loc[deltas["comparison"] == "D_to_E"]
    d_to_f = deltas.loc[deltas["comparison"] == "D_to_F"]
    return pd.DataFrame(
        [
            {
                "criterion": "biu_v2_automatic_behavior_recovery",
                "satisfied": bool(
                    indexed.loc["E", "f1_mean"] > indexed.loc["C", "f1_mean"] + 0.005
                    or indexed.loc["E", "pr_auc_mean"] > indexed.loc["C", "pr_auc_mean"] + 0.005
                ),
            },
            {
                "criterion": "biu_v1_automatic_behavior_recovery",
                "satisfied": bool(
                    indexed.loc["F", "f1_mean"] > indexed.loc["C", "f1_mean"] + 0.005
                    or indexed.loc["F", "pr_auc_mean"] > indexed.loc["C", "pr_auc_mean"] + 0.005
                ),
            },
            {
                "criterion": "biu_v2_over_v1_seed_consistency",
                "satisfied": bool((e_to_f["f1_delta"] < 0).all() and (e_to_f["pr_auc_delta"] < 0).all()),
            },
            {
                "criterion": "biu_v1_over_v2_seed_consistency",
                "satisfied": bool((e_to_f["f1_delta"] > 0).all() and (e_to_f["pr_auc_delta"] > 0).all()),
            },
            {
                "criterion": "biu_v2_efficiency_noninferiority_to_raw_pose",
                "satisfied": bool(
                    indexed.loc["E", "f1_mean"] >= indexed.loc["D", "f1_mean"] - 0.005
                    and indexed.loc["E", "pr_auc_mean"] >= indexed.loc["D", "pr_auc_mean"] - 0.005
                ),
            },
            {
                "criterion": "biu_v1_efficiency_noninferiority_to_raw_pose",
                "satisfied": bool(
                    indexed.loc["F", "f1_mean"] >= indexed.loc["D", "f1_mean"] - 0.005
                    and indexed.loc["F", "pr_auc_mean"] >= indexed.loc["D", "pr_auc_mean"] - 0.005
                ),
            },
            {
                "criterion": "biu_v2_beats_raw_pose_all_seeds",
                "satisfied": bool((d_to_e["f1_delta"] > 0).all() and (d_to_e["pr_auc_delta"] > 0).all()),
            },
            {
                "criterion": "biu_v1_beats_raw_pose_all_seeds",
                "satisfied": bool((d_to_f["f1_delta"] > 0).all() and (d_to_f["pr_auc_delta"] > 0).all()),
            },
            {
                "criterion": "biu_v2_recovery_fraction_at_least_half",
                "satisfied": bool(
                    recovery_index.loc["f1", "biu_v2_recovery_fraction_mean"] >= 0.5
                    and recovery_index.loc["pr_auc", "biu_v2_recovery_fraction_mean"] >= 0.5
                ),
            },
            {
                "criterion": "biu_v1_recovery_fraction_at_least_half",
                "satisfied": bool(
                    recovery_index.loc["f1", "biu_v1_recovery_fraction_mean"] >= 0.5
                    and recovery_index.loc["pr_auc", "biu_v1_recovery_fraction_mean"] >= 0.5
                ),
            },
        ]
    )


def run_amended_evaluation(output_root: Path, *, resume: bool) -> None:
    root = Path(output_root)
    records = authenticate_amended_training_grid(root)
    if len(records) != 18:
        raise ValueError("Final evaluation requires the exact 18-record A-F grid.")
    if (root / ".evaluate.complete.json").exists():
        raise ValueError("Prohibited original 15-run evaluation marker is present.")
    marker = root / ".evaluate_amended.complete.json"
    if marker.is_file():
        authenticate_amended_evaluation(root)
        return
    log = progress_logger(root / "logs" / "final_evaluation_amended.log")
    log("phase=evaluate_amended status=authorized training_records=18 loading_set03_once")
    rows, arrays, set03_metadata = load_amended_set03(root, resume=resume, progress=log)
    original = load_original_runner()
    metrics_root = root / "metrics"
    predictions_root = root / "predictions"
    metrics_root.mkdir(parents=True, exist_ok=True)
    predictions_root.mkdir(parents=True, exist_ok=True)
    atomic_csv(rows, metrics_root / "set03_cohort_manifest.csv")
    yolop_metadata = set03_metadata["yolop"]
    atomic_csv(
        pd.DataFrame(
            [
                {
                    "split": "test",
                    **{
                        key: value
                        for key, value in yolop_metadata.items()
                        if key.endswith("count") or key.endswith("rate") or key.endswith("fraction")
                    },
                }
            ]
        ),
        metrics_root / "yolop_coverage_set03.csv",
    )
    pose_coverage = pd.read_csv(original.EXP047_ROOT / "metrics" / "pose_coverage.csv")
    atomic_csv(
        pose_coverage.loc[
            (pose_coverage["split"] == "test") & (pose_coverage["source"] == "exp012")
        ].reset_index(drop=True),
        metrics_root / "pose_coverage_set03.csv",
    )
    fallback = pd.read_csv(original.EXP047_ROOT / "metrics" / "bbox_direction_fallback.csv")
    atomic_csv(
        fallback.loc[fallback["split"] == "test"].reset_index(drop=True),
        metrics_root / "bbox_direction_fallback_set03.csv",
    )
    atomic_csv(
        _quality_summary(rows.assign(split="test"), arrays["biu_v1"], splits=("test",)),
        metrics_root / "biu_v1_quality_counts_set03.csv",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    record_by_identity = {(str(record["variant"]), int(record["seed"])): record for record in records}
    results = []
    efficiency = []
    stream_parameters = []
    prediction_frames: dict[tuple[str, int], pd.DataFrame] = {}
    for variant in AF_VARIANTS:
        dataset = (
            BiuV1AmendmentDataset(rows, arrays, training_phase=False)
            if variant == "F"
            else original.Exp048Dataset(rows, arrays, variant=variant, training_phase=False)
        )
        loader = DataLoader(
            dataset,
            batch_size=F_BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )
        first_features, _, _ = next(iter(loader))
        for seed in F_SEEDS:
            record = record_by_identity[(variant, seed)]
            checkpoint_path = root / "checkpoints" / f"{variant}_seed{seed}.best.pt"
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            model = build_f_model(seed) if variant == "F" else original.build_model(variant, seed)
            model.load_state_dict(checkpoint["model_state"])
            model.to(device)
            labels, probabilities, _ = original.predict_loader(model, loader, device)
            metric = original.compute_binary_metrics(labels, probabilities)
            results.append({"variant": variant, "seed": seed, **metric})
            prediction = rows[
                [
                    "sequence_id",
                    "set_id",
                    "video_id",
                    "ped_id",
                    "start_frame",
                    "end_frame",
                    "target_frame",
                    "label",
                ]
            ].copy()
            prediction["variant"] = variant
            prediction["seed"] = seed
            prediction["threshold"] = F_FIXED_THRESHOLD
            prediction["probability"] = probabilities
            prediction["prediction"] = (probabilities >= F_FIXED_THRESHOLD).astype(np.int64)
            prediction_path = predictions_root / f"set03_{variant}_seed{seed}.csv"
            atomic_csv(prediction, prediction_path)
            prediction_frames[(variant, seed)] = prediction
            latency = original._measure_model_latency(model, first_features, device)
            if variant == "F":
                pathway_latency = _measure_f_pathway_latency(model, first_features, device)
            else:
                pathway_latency = original._measure_pose_pathway_latency(
                    model, variant, first_features, device
                )
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
                        original._measure_biu_extraction_latency(rows, arrays)
                        if variant == "E" and seed == F_SEEDS[0]
                        else _measure_biu_v1_extraction_latency(rows, arrays)
                        if variant == "F" and seed == F_SEEDS[0]
                        else np.nan
                    ),
                    "pose_estimator_runtime_included": False,
                    "pose_estimator_runtime_status": "unavailable_historical_cache",
                }
            )
            log(
                f"evaluated variant={variant} seed={seed} f1={metric['f1']:.6f} "
                f"pr_auc={metric['pr_auc']:.6f}"
            )
            if seed == F_SEEDS[0]:
                for stream in model.streams:
                    if stream in {"biu_v1", "biu_v2"}:
                        parameter_count = sum(
                            value.numel() for value in model.biu_projection.parameters()
                        )
                    else:
                        parameter_count = sum(value.numel() for value in model.lstms[stream].parameters())
                        parameter_count += sum(
                            value.numel() for value in model.attentions[stream].parameters()
                        )
                    stream_parameters.append(
                        {"variant": variant, "stream": stream, "parameters": int(parameter_count)}
                    )

    per_seed = pd.DataFrame(results)
    _validate_af_results(per_seed)
    summary = _amended_summary(per_seed)
    deltas = build_amended_paired_deltas(per_seed)
    recovery = build_amended_behavior_recovery(per_seed)
    output_frames = {
        "per_seed_results.csv": per_seed,
        "fixed_0p5_results.csv": per_seed,
        "mslstm_protocol_results.csv": per_seed,
        "summary_results.csv": summary,
        "paired_deltas.csv": deltas,
        "behavior_recovery.csv": recovery,
        "efficiency.csv": pd.DataFrame(efficiency),
        "stream_parameter_counts.csv": pd.DataFrame(stream_parameters),
        "success_criteria.csv": _amended_success_criteria(summary, deltas, recovery),
        "error_transitions_fixed_0p5.csv": _amended_error_transitions(prediction_frames),
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
                    "variant": "F",
                    "source": "AIBThings historical 72-D BIU-v1 definition",
                    "historical_f1": np.nan,
                    "fresh_protocol_parity": False,
                    "note": "Representation provenance reference only; no historical MSLSTM F result exists.",
                },
            ]
        ),
    }
    for name, frame in output_frames.items():
        atomic_csv(frame, metrics_root / name)
    expected = expected_amended_artifact_relatives()
    missing = sorted(relative for relative in expected if not (root / relative).is_file())
    if missing:
        raise RuntimeError(f"Amended evaluation package is incomplete: {missing}")
    payload = {
        "phase": "evaluate_amended",
        "created_at": utc_now(),
        "training_record_count": 18,
        "training_marker_hashes": {
            f"{variant}_{seed}": sha256_file(root / f".train_{variant}_seed{seed}.complete.json")
            for variant in AF_VARIANTS
            for seed in F_SEEDS
        },
        "set03_rows": len(rows),
        "set03_evaluations": 18,
        "amendment_audit_sha256": sha256_file(root / AUDIT_MARKER_NAME),
        "amendment_core_sha256": sha256_file(SCRIPT_ROOT / "biu_v1_amendment_core.py"),
        "amendment_runner_sha256": sha256_file(Path(__file__).resolve()),
        "artifacts": {relative: sha256_file(root / relative) for relative in sorted(expected)},
    }
    atomic_json(payload, marker)
    authenticate_amended_evaluation(root)
    log(f"phase=evaluate_amended status=complete marker={marker}")


def finalize_amended_experiment(output_root: Path) -> None:
    root = Path(output_root)
    authenticate_amended_training_grid(root)
    authenticate_amended_evaluation(root)
    summary = pd.read_csv(root / "metrics" / "summary_results.csv").set_index("variant")
    per_seed = pd.read_csv(root / "metrics" / "per_seed_results.csv")
    deltas = pd.read_csv(root / "metrics" / "paired_deltas.csv")
    recovery = pd.read_csv(root / "metrics" / "behavior_recovery.csv").set_index("metric")
    efficiency = pd.read_csv(root / "metrics" / "efficiency.csv")
    e_to_f = deltas.loc[deltas["comparison"] == "E_to_F"]
    d_to_e = deltas.loc[deltas["comparison"] == "D_to_E"]
    d_to_f = deltas.loc[deltas["comparison"] == "D_to_F"]
    v1_over_v2_consistent = bool(
        (e_to_f["f1_delta"] > 0).all() and (e_to_f["pr_auc_delta"] > 0).all()
    )
    v2_over_v1_consistent = bool(
        (e_to_f["f1_delta"] < 0).all() and (e_to_f["pr_auc_delta"] < 0).all()
    )
    v2_within_raw = bool(
        summary.loc["E", "f1_mean"] >= summary.loc["D", "f1_mean"] - 0.005
        and summary.loc["E", "pr_auc_mean"] >= summary.loc["D", "pr_auc_mean"] - 0.005
    )
    v1_within_raw = bool(
        summary.loc["F", "f1_mean"] >= summary.loc["D", "f1_mean"] - 0.005
        and summary.loc["F", "pr_auc_mean"] >= summary.loc["D", "pr_auc_mean"] - 0.005
    )
    best_biu = "BIU-v1" if (
        summary.loc["F", "f1_mean"] + summary.loc["F", "pr_auc_mean"]
        > summary.loc["E", "f1_mean"] + summary.loc["E", "pr_auc_mean"]
    ) else "BIU-v2"
    best_biu_variant = "F" if best_biu == "BIU-v1" else "E"
    if v1_over_v2_consistent and (
        (d_to_f["f1_delta"] > 0).all() and (d_to_f["pr_auc_delta"] > 0).all()
    ):
        outcome = "BIU-v1 representation success"
    elif v2_over_v1_consistent and (
        (d_to_e["f1_delta"] > 0).all() and (d_to_e["pr_auc_delta"] > 0).all()
    ):
        outcome = "BIU-v2 representation success"
    elif v1_within_raw or v2_within_raw:
        outcome = f"{best_biu} efficiency success"
    elif (
        summary.loc["D", "f1_mean"] > max(summary.loc["E", "f1_mean"], summary.loc["F", "f1_mean"])
        and summary.loc["D", "pr_auc_mean"]
        > max(summary.loc["E", "pr_auc_mean"], summary.loc["F", "pr_auc_mean"])
    ):
        outcome = "Raw-pose success"
    elif (
        summary.loc[best_biu_variant, "f1_mean"] > summary.loc["C", "f1_mean"]
        or summary.loc[best_biu_variant, "pr_auc_mean"] > summary.loc["C", "pr_auc_mean"]
    ):
        outcome = f"Partial automatic-behavior recovery ({best_biu})"
    else:
        outcome = "Negative"

    report_path = root / "FINAL_REPORT.md"
    lines = [
        "# Exp048 Final Report",
        "",
        f"Final outcome category: **{outcome}**.",
        "",
        "This controlled experiment uses the recovered 30-frame, next-raw-frame MSLSTM task. "
        "The paper F1=0.8992 remains a historical reference only. Exp048 architecture and "
        "selection decisions were frozen before this one final set03 evaluation; set03 had been "
        "inspected previously in separate post-hoc work, which is disclosed as an exposure limitation.",
        "",
        "## Fresh Mean Results",
        "",
        "| Variant | F1 mean | PR-AUC mean | ROC-AUC mean | Parameters |",
        "|---|---:|---:|---:|---:|",
    ]
    for variant in AF_VARIANTS:
        row = summary.loc[variant]
        parameter_count = int(
            efficiency.loc[efficiency["variant"] == variant, "total_parameters"].iloc[0]
        )
        lines.append(
            f"| {variant} | {row.f1_mean:.6f} | {row.pr_auc_mean:.6f} | "
            f"{row.roc_auc_mean:.6f} | {parameter_count:,} |"
        )
    lines.extend(
        [
            "",
            "## Frozen Comparisons",
            "",
            f"E→F favorable for BIU-v1 on both F1 and PR-AUC in all seeds: `{str(v1_over_v2_consistent).lower()}`.",
            f"E→F favorable for BIU-v2 on both metrics in all seeds: `{str(v2_over_v1_consistent).lower()}`.",
            f"BIU-v2 within the frozen 0.005 F1/PR-AUC margin of raw pose: `{str(v2_within_raw).lower()}`.",
            f"BIU-v1 within the frozen 0.005 F1/PR-AUC margin of raw pose: `{str(v1_within_raw).lower()}`.",
            f"BIU-v2 recovery fraction: F1 `{recovery.loc['f1', 'biu_v2_recovery_fraction_mean']:.6f}`, "
            f"PR-AUC `{recovery.loc['pr_auc', 'biu_v2_recovery_fraction_mean']:.6f}`.",
            f"BIU-v1 recovery fraction: F1 `{recovery.loc['f1', 'biu_v1_recovery_fraction_mean']:.6f}`, "
            f"PR-AUC `{recovery.loc['pr_auc', 'biu_v1_recovery_fraction_mean']:.6f}`.",
            "",
            "## Thesis-Safe Interpretation",
            "",
            "D versus E/F compares raw temporal pose against two frozen task-specific behavioral "
            "representations from the same Exp012 measurements. E versus F compares complete frozen "
            "representations, not a single-cue ablation: BIU-v1 includes its historical low-confidence "
            "count, while BIU-v2 includes its frozen direction/scale metadata. Neither BIU adds body-pose "
            "information absent from the shared pose source. A versus E/F is an overall privileged-to-automatic "
            "pipeline comparison and must not be attributed solely to BIU.",
            "",
            "## Per-Seed Fixed-0.5 Results",
            "",
            "| Variant | Seed | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in per_seed.itertuples(index=False):
        lines.append(
            f"| {row.variant} | {int(row.seed)} | {row.accuracy:.6f} | {row.precision:.6f} | "
            f"{row.recall:.6f} | {row.f1:.6f} | {row.roc_auc:.6f} | {row.pr_auc:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Provenance And Efficiency",
            "",
            "Variants D, E, and F use the same SHA-pinned historical Exp012 pose artifact: "
            f"`{EXP012_POSE_PATH}` (`{EXP012_POSE_SHA256}`). The artifact is byte-identical to "
            "the historical main-BIU cache and documented as ViTPose-S-derived, but the original "
            "checkpoint/configuration and execution log are unavailable, so historical checkpoint "
            "identity is not cryptographically verifiable.",
            "",
            "F uses the exact historical 72-D BIU-v1 schema and a fixed `72 -> 128 -> 512` projector. "
            "E uses the frozen 63-D BIU-v2 schema and `63 -> 128 -> 512`. Pose-estimator runtime is "
            "excluded from predictor/pathway latency because both use the historical cache.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (root / "EXPERIMENT_COMPLETE").write_text(
        f"completed_at={utc_now()}\noutcome={outcome}\n"
        f"final_report_sha256={sha256_file(report_path)}\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the approved Exp048 BIU-v1 sidecar amendment.")
    parser.add_argument("--output-root", type=Path, default=EXPERIMENT_ROOT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("command", choices=("audit", "train", "train-all", "evaluate", "finalize", "status"))
    parser.add_argument("--seed", type=int, choices=(42, 43, 44))
    return parser


def status(output_root: Path) -> None:
    root = Path(output_root)
    print(f"original_prepare_complete={(root / '.prepare.complete.json').is_file()}")
    partial = root / "cache" / f"{CACHE_NAME}.partial" / "progress.json"
    print(f"biu_v1_audit_complete={(root / AUDIT_MARKER_NAME).is_file()}")
    if partial.is_file():
        progress = json.loads(partial.read_text(encoding="utf-8"))
        print(f"biu_v1_cache_rows={progress['next_index']}/{progress['n_rows']}")
    for variant in ("A", "B", "C", "D", "E"):
        completed = sum(
            (root / f".train_{variant}_seed{seed}.complete.json").is_file()
            for seed in (42, 43, 44)
        )
        print(f"training_{variant}={completed}/3")
    f_completed = sum(
        (root / f".train_F_seed{seed}.complete.json").is_file() for seed in (42, 43, 44)
    )
    print(f"training_F={f_completed}/3")
    print(f"original_evaluation_present={(root / '.evaluate.complete.json').is_file()}")
    print(f"amended_evaluation_complete={(root / '.evaluate_amended.complete.json').is_file()}")
    print(f"experiment_complete={(root / 'EXPERIMENT_COMPLETE').is_file()}")


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "audit":
        build_biu_v1_cache(args.output_root, resume=args.resume)
    elif args.command == "train":
        if args.seed is None:
            raise ValueError("The F train command requires --seed.")
        run_f_training(args.output_root, seed=args.seed, resume=args.resume)
    elif args.command == "train-all":
        run_f_train_all(args.output_root, resume=args.resume)
    elif args.command == "evaluate":
        run_amended_evaluation(args.output_root, resume=args.resume)
    elif args.command == "finalize":
        finalize_amended_experiment(args.output_root)
    elif args.command == "status":
        status(args.output_root)
    else:
        raise NotImplementedError(f"Amendment phase is not implemented yet: {args.command}")


if __name__ == "__main__":
    main()
