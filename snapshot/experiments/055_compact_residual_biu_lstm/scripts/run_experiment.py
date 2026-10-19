#!/usr/bin/env python3
"""Validation-gated, resumable runner for Exp055."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import exp055_core as core  # noqa: E402


ROOT = SCRIPT_DIR.parent
EXPERIMENTS_ROOT = ROOT.parent
EXP012_ROOT = EXPERIMENTS_ROOT / "012_pie_biu_v2_30_60_benchmark"
EXP050_ROOT = EXPERIMENTS_ROOT / "050_mslstm_practical_behavior_replacement_pie_16f_tte30_60"
EXP053_ROOT = EXPERIMENTS_ROOT / "053_mslstm_bbox53_replacement"
BASE_CACHE = EXP050_ROOT / "cache/trainval_base"
EXP053_BBOX53_TRAINVAL = EXP053_ROOT / "cache/bbox53_trainval.npy"
EXP012_FEATURE_DEFINITIONS = (
    EXP012_ROOT / "metrics/pie_biu_v2_feature_set_definitions.csv"
)
PREPARE_MARKER = ROOT / ".prepare.complete.json"
ANCHOR_SELECTION_MARKER = ROOT / "SELECTED_ANCHOR.json"
FINAL_CANDIDATE_MARKER = ROOT / "FROZEN_FINAL_CANDIDATE.json"
SELECTION_TOLERANCE = 1e-6
SOURCE_ARRAY_NAMES = ("bbox", "biu_v1", "biu_v2", "ego_speed", "ego_acc", "yolop")
PREPARED_ARRAY_NAMES = (
    "summary_v1",
    "summary_v2_core",
    "summary_joint_unique",
    "bbox_raw4",
    "bbox_motion8",
    "ego",
    "yolop",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(value: Mapping[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def atomic_npy(value: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, np.asarray(value))
    temporary.replace(path)


def atomic_torch_save(value: Mapping[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(value), temporary)
    temporary.replace(path)


def log_line(path: Path, message: str) -> None:
    line = f"{utc_now()} {message}"
    print(line, flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def stage1_candidates() -> tuple[core.ModelConfig, ...]:
    return tuple(
        core.ModelConfig(biu_mode, bbox_mode, "none")
        for biu_mode in core.BIU_MODES
        for bbox_mode in core.BBOX_TEMPORAL_MODES
    )


def stage2_candidates(anchor: core.ModelConfig) -> tuple[core.ModelConfig, ...]:
    if anchor.context_mode != "none":
        raise ValueError("Stage-2 expansion requires a context-free selected anchor.")
    return tuple(anchor.with_context(context) for context in core.CONTEXT_MODES)


def select_validation_candidate(results: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    required = {
        "candidate_id",
        "seed",
        "selection_score",
        "pr_auc",
        "f1",
        "total_parameters",
    }
    missing = sorted(required.difference(results.columns))
    if missing:
        raise ValueError(f"Validation result columns are missing: {missing}")
    if results.empty:
        raise ValueError("Validation selection requires candidate results.")
    expected_seeds = set(core.SEEDS)
    for candidate_id, group in results.groupby("candidate_id", sort=False):
        if set(group["seed"].astype(int)) != expected_seeds or len(group) != len(core.SEEDS):
            raise ValueError(f"Candidate {candidate_id} does not have the frozen seed grid.")
    summary = (
        results.groupby("candidate_id", as_index=False)
        .agg(
            selection_score_mean=("selection_score", "mean"),
            selection_score_std=("selection_score", "std"),
            pr_auc_mean=("pr_auc", "mean"),
            pr_auc_std=("pr_auc", "std"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            total_parameters=("total_parameters", "max"),
        )
        .sort_values("candidate_id")
        .reset_index(drop=True)
    )
    top_score = float(summary["selection_score_mean"].max())
    eligible = summary.loc[
        summary["selection_score_mean"] >= top_score - SELECTION_TOLERANCE
    ].copy()
    eligible = eligible.sort_values(
        ["pr_auc_mean", "f1_mean", "total_parameters", "candidate_id"],
        ascending=[False, False, True, True],
        kind="mergesort",
    )
    return str(eligible.iloc[0]["candidate_id"]), summary


def validate_grid_training_contracts(contracts: list[Mapping[str, object]]) -> None:
    if not contracts:
        raise RuntimeError("Validation grid has no training contracts.")
    ignored = {"candidate", "seed"}
    signatures = {
        core.canonical_json_sha256(
            {key: value for key, value in contract.items() if key not in ignored}
        )
        for contract in contracts
    }
    if len(signatures) != 1:
        raise RuntimeError("Validation grid contains mixed training contracts.")


def validate_prepare_contract(path: Path) -> dict[str, object]:
    marker = Path(path)
    if not marker.is_file():
        raise RuntimeError("Authenticated Exp055 preparation is required.")
    payload = json.loads(marker.read_text(encoding="utf-8"))
    if payload.get("test_accessed") is not False:
        raise RuntimeError("Preparation must prove set03 exclusion.")
    if payload.get("source_invariants_verified") is not True:
        raise RuntimeError("Preparation must prove source/cohort invariant verification.")
    if int(payload.get("train_rows", -1)) != core.EXPECTED_SPLIT_COUNTS["train"]:
        raise RuntimeError("Frozen training-row count drifted.")
    if int(payload.get("val_rows", -1)) != core.EXPECTED_SPLIT_COUNTS["val"]:
        raise RuntimeError("Frozen validation-row count drifted.")
    return payload


def validate_trainval_rows(rows: pd.DataFrame) -> pd.DataFrame:
    required = {
        "sequence_id",
        "split",
        "set_id",
        "start_frame",
        "end_frame",
        "event_frame",
        "time_to_event_frames",
        "observation_length",
        "frame_step",
        "frame_sampling",
        "label",
    }
    missing = sorted(required.difference(rows.columns))
    if missing:
        raise ValueError(f"Train/validation rows are missing columns: {missing}")
    checked = rows.reset_index(drop=True).copy()
    if checked.empty or checked["sequence_id"].duplicated().any():
        raise RuntimeError("Train/validation rows must be nonempty with unique sequence IDs.")
    if not set(checked["split"].astype(str)).issubset({"train", "val"}):
        raise RuntimeError("Preparation contains a population outside train/validation.")
    if checked["set_id"].astype(str).eq("set03").any():
        raise RuntimeError("Preparation must not contain set03.")
    allowed_sets = {
        "train": {"set01", "set02", "set06"},
        "val": {"set04", "set05"},
    }
    for split, allowed in allowed_sets.items():
        actual = set(
            checked.loc[checked["split"].astype(str).eq(split), "set_id"].astype(str)
        )
        if not actual.issubset(allowed):
            raise RuntimeError(f"Frozen {split} set membership drifted: {sorted(actual)}")
    consecutive = (
        checked["observation_length"].astype(int).eq(core.SEQ_LEN)
        & checked["frame_step"].astype(int).eq(1)
        & checked["frame_sampling"].astype(str).eq("consecutive")
        & (checked["end_frame"].astype(int) - checked["start_frame"].astype(int)).eq(
            core.SEQ_LEN - 1
        )
    )
    if not consecutive.all():
        raise RuntimeError("Every Exp055 window must contain exactly 16 consecutive frames.")
    if not checked["time_to_event_frames"].astype(int).between(30, 60).all():
        raise RuntimeError("Exp055 TTE must remain within 30--60 frames.")
    if not (
        checked["event_frame"].astype(int) - checked["end_frame"].astype(int)
    ).eq(checked["time_to_event_frames"].astype(int)).all():
        raise RuntimeError("Exp055 event/end-frame TTE identity drifted.")
    if not (checked["event_frame"].astype(int) > checked["end_frame"].astype(int)).all():
        raise RuntimeError("Future event frames must remain outside the observation window.")
    if not set(checked["label"].astype(int)).issubset({0, 1}):
        raise RuntimeError("Labels must remain binary.")
    return checked


def validate_augmented_alignment(augmented_rows: pd.DataFrame, base_rows: pd.DataFrame) -> None:
    required = {"source_cache_index", "sequence_id", "label"}
    if not required.issubset(augmented_rows.columns):
        raise RuntimeError("Augmented rows lack source-alignment columns.")
    train = base_rows.loc[base_rows["split"].astype(str).eq("train")].reset_index(drop=True)
    source_index = augmented_rows["source_cache_index"].to_numpy(dtype=np.int64)
    if len(train) == 0 or np.any(source_index < 0) or np.any(source_index >= len(train)):
        raise RuntimeError("Exp055 augmented sample alignment failed.")
    referenced = train.iloc[source_index].reset_index(drop=True)
    sequence_match = (
        referenced["sequence_id"].astype(str).to_numpy()
        == augmented_rows["sequence_id"].astype(str).to_numpy()
    )
    label_match = (
        referenced["label"].astype(int).to_numpy()
        == augmented_rows["label"].astype(int).to_numpy()
    )
    if not sequence_match.all() or not label_match.all():
        raise RuntimeError("Exp055 augmented sample alignment failed.")


def assert_training_allowed(root: Path = ROOT) -> dict[str, object]:
    experiment_root = Path(root)
    payload = validate_prepare_contract(experiment_root / ".prepare.complete.json")
    artifacts = payload.get("artifacts", {})
    if not isinstance(artifacts, dict):
        raise RuntimeError("Authenticated preparation artifacts are required.")
    authenticate_artifacts(experiment_root, artifacts)
    return payload


def authenticate_artifacts(root: Path, artifacts: dict[str, str]) -> None:
    for relative, expected_hash in artifacts.items():
        path = Path(root) / relative
        if not path.is_file() or core.sha256_file(path) != expected_hash:
            raise RuntimeError(f"Artifact authentication failed: {relative}")


def _load_source_cache(path: Path) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, object]]:
    metadata_path = Path(path) / "metadata.json"
    if not metadata_path.is_file():
        raise RuntimeError(f"Missing source cache metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    required = {"rows.csv", *(f"{name}.npy" for name in SOURCE_ARRAY_NAMES)}
    artifacts = metadata.get("artifacts", {})
    if not isinstance(artifacts, dict) or not required.issubset(artifacts):
        raise RuntimeError(f"Source cache is missing authenticated inputs: {path}")
    authenticate_artifacts(Path(path), {name: artifacts[name] for name in required})
    rows = pd.read_csv(Path(path) / "rows.csv")
    arrays = {
        name: np.load(Path(path) / f"{name}.npy", mmap_mode="r")
        for name in SOURCE_ARRAY_NAMES
    }
    if any(len(array) != len(rows) for array in arrays.values()):
        raise RuntimeError(f"Source cache population is misaligned: {path}")
    return rows, arrays, metadata


def _authenticate_bbox53_sources() -> dict[str, str]:
    marker_path = EXP053_ROOT / ".prepare.complete.json"
    payload = json.loads(marker_path.read_text(encoding="utf-8"))
    if payload.get("test_accessed") is not False:
        raise RuntimeError("Exp053 preparation does not prove set03 exclusion.")
    if payload.get("source_prepare_sha256") != core.sha256_file(
        EXP050_ROOT / ".prepare.complete.json"
    ):
        raise RuntimeError("Exp053 BBox53 caches are not bound to the current Exp050 preparation.")
    required = {
        "cache/bbox53_trainval.npy",
        *(f"cache/bbox53_augmented_seed{seed}.npy" for seed in core.SEEDS),
    }
    artifacts = payload.get("artifacts", {})
    if not required.issubset(artifacts):
        raise RuntimeError("Exp053 does not authenticate all required BBox53 caches.")
    authenticate_artifacts(EXP053_ROOT, {name: artifacts[name] for name in required})
    return {name: artifacts[name] for name in sorted(required)}


def _verify_exp012_v2_schema() -> str:
    definitions = pd.read_csv(EXP012_FEATURE_DEFINITIONS)
    row = definitions.loc[definitions["feature_set"].eq("biu_v2_core")]
    if len(row) != 1:
        raise RuntimeError("Exp012 must contain one BIU-v2-core feature definition.")
    names = tuple(json.loads(row.iloc[0]["feature_columns_json"]))
    if names != core.BIU_V2_CORE_FEATURES:
        raise RuntimeError("The frozen Exp055 BIU-v2 core drifted from Exp012.")
    return core.sha256_file(EXP012_FEATURE_DEFINITIONS)


def _bbox_feature_batches(bbox: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(bbox, dtype=np.float32)
    if values.ndim != 3 or values.shape[1:] != (core.SEQ_LEN, 4):
        raise RuntimeError("Source BBox array must have shape [N,16,4].")
    raw4 = values.copy()
    motion8 = np.stack(
        [core.bbox_temporal_features(window, "motion8") for window in values], axis=0
    )
    return raw4, motion8


def _scaler_to_json(scaler: Mapping[str, np.ndarray]) -> dict[str, list[float]]:
    return {
        name: np.asarray(scaler[name], dtype=np.float64).tolist()
        for name in ("median", "mean", "std")
    }


def _scaler_from_json(value: Mapping[str, object]) -> dict[str, np.ndarray]:
    return {
        name: np.asarray(value[name], dtype=np.float64)
        for name in ("median", "mean", "std")
    }


def _build_prepared_arrays(
    source_arrays: Mapping[str, np.ndarray],
    bbox53: np.ndarray,
    schemas: Mapping[str, tuple[str, ...]],
    scalers: Mapping[str, Mapping[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    v1_names = schemas["biu_v1"]
    v2_names = schemas["biu_v2_source"]
    output: dict[str, np.ndarray] = {}
    for mode in core.BIU_MODES:
        representation, names = core.build_biu_representation(
            mode,
            source_arrays["biu_v1"],
            source_arrays["biu_v2"],
            v1_names,
            v2_names,
        )
        if names != schemas[mode]:
            raise RuntimeError(f"Prepared BIU schema drifted for mode {mode}.")
        scaled = core.transform_features(representation, scalers[f"biu_{mode}"])
        summary = np.concatenate((np.asarray(bbox53, dtype=np.float32), scaled), axis=1)
        if summary.shape[1] != core.SUMMARY_DIMS[mode]:
            raise RuntimeError(f"Prepared summary dimension drifted for mode {mode}.")
        output[f"summary_{mode}"] = summary.astype(np.float32, copy=False)
    raw4, motion8 = _bbox_feature_batches(source_arrays["bbox"])
    output["bbox_raw4"] = core.transform_features(raw4, scalers["bbox_raw4"])
    output["bbox_motion8"] = core.transform_features(motion8, scalers["bbox_motion8"])
    output["ego"] = np.concatenate(
        (
            np.asarray(source_arrays["ego_speed"], dtype=np.float32),
            np.asarray(source_arrays["ego_acc"], dtype=np.float32),
        ),
        axis=2,
    )
    output["yolop"] = np.asarray(source_arrays["yolop"], dtype=np.float32)
    if any(len(array) != len(bbox53) for array in output.values()):
        raise RuntimeError("Prepared arrays are not population-aligned.")
    return output


def _save_prepared_cache(
    path: Path,
    rows: pd.DataFrame,
    arrays: Mapping[str, np.ndarray],
    extra: Mapping[str, object],
) -> dict[str, object]:
    path.mkdir(parents=True, exist_ok=True)
    atomic_csv(rows, path / "rows.csv")
    for name in PREPARED_ARRAY_NAMES:
        atomic_npy(arrays[name], path / f"{name}.npy")
    artifacts = {"rows.csv": core.sha256_file(path / "rows.csv")}
    artifacts.update(
        {f"{name}.npy": core.sha256_file(path / f"{name}.npy") for name in PREPARED_ARRAY_NAMES}
    )
    metadata = {
        "created_at": utc_now(),
        "n_rows": int(len(rows)),
        "sequence_id_sha256": core.canonical_json_sha256(rows["sequence_id"].astype(str).tolist()),
        "artifacts": artifacts,
        **dict(extra),
    }
    atomic_json(metadata, path / "metadata.json")
    return metadata


def load_prepared_cache(
    path: Path,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, object]]:
    metadata_path = Path(path) / "metadata.json"
    if not metadata_path.is_file():
        raise RuntimeError(f"Missing prepared cache: {path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    authenticate_artifacts(Path(path), metadata.get("artifacts", {}))
    rows = pd.read_csv(Path(path) / "rows.csv")
    arrays = {
        name: np.load(Path(path) / f"{name}.npy", mmap_mode="r")
        for name in PREPARED_ARRAY_NAMES
    }
    if any(len(array) != len(rows) for array in arrays.values()):
        raise RuntimeError("Prepared cache population is misaligned.")
    return rows, arrays, metadata


def run_prepare(root: Path = ROOT) -> dict[str, object]:
    experiment_root = Path(root)
    marker = experiment_root / ".prepare.complete.json"
    if marker.is_file():
        existing = json.loads(marker.read_text(encoding="utf-8"))
        if existing.get("source_invariants_verified") is True:
            return assert_training_allowed(experiment_root)
    source_rows, source_arrays, source_metadata = _load_source_cache(BASE_CACHE)
    rows = validate_trainval_rows(source_rows)
    counts = rows["split"].value_counts().astype(int).to_dict()
    if counts != {"val": core.EXPECTED_SPLIT_COUNTS["val"], "train": core.EXPECTED_SPLIT_COUNTS["train"]}:
        raise RuntimeError(f"Frozen train/validation counts drifted: {counts}")
    bbox_hashes = _authenticate_bbox53_sources()
    exp012_schema_hash = _verify_exp012_v2_schema()
    bbox53 = np.load(EXP053_BBOX53_TRAINVAL, mmap_mode="r")
    if bbox53.shape != (len(rows), 53):
        raise RuntimeError("Exp053 BBox53 train/validation population drifted.")
    v1_names = tuple(source_metadata.get("biu_v1_feature_names", ()))
    v2_names = tuple(source_metadata.get("biu_v2_feature_names", ()))
    if len(v1_names) != 72 or len(v2_names) != 63:
        raise RuntimeError("Exp050 BIU schemas are not the frozen 72-D/63-D sources.")
    schemas: dict[str, tuple[str, ...]] = {
        "biu_v1": v1_names,
        "biu_v2_source": v2_names,
        "v1": v1_names,
        "v2_core": core.BIU_V2_CORE_FEATURES,
        "joint_unique": core.joint_unique_schema(v1_names, core.BIU_V2_CORE_FEATURES),
    }
    if len(schemas["joint_unique"]) != 108:
        raise RuntimeError("The real Exp055 joint BIU schema must contain 108 unique features.")
    train_mask = rows["split"].eq("train").to_numpy()
    scalers: dict[str, dict[str, np.ndarray]] = {}
    for mode in core.BIU_MODES:
        representation, _ = core.build_biu_representation(
            mode,
            source_arrays["biu_v1"],
            source_arrays["biu_v2"],
            v1_names,
            v2_names,
        )
        scalers[f"biu_{mode}"] = core.fit_feature_scaler(representation[train_mask])
    raw4, motion8 = _bbox_feature_batches(source_arrays["bbox"])
    scalers["bbox_raw4"] = core.fit_feature_scaler(raw4[train_mask])
    scalers["bbox_motion8"] = core.fit_feature_scaler(motion8[train_mask])
    prepared = _build_prepared_arrays(source_arrays, bbox53, schemas, scalers)
    trainval_metadata = _save_prepared_cache(
        experiment_root / "cache/trainval",
        rows,
        prepared,
        {"population": ["train", "val"], "test_accessed": False},
    )
    for seed in core.SEEDS:
        aug_rows, aug_arrays, aug_metadata = _load_source_cache(
            EXP050_ROOT / f"cache/augmented_seed{seed}"
        )
        validate_augmented_alignment(aug_rows, rows)
        if aug_metadata.get("source_base_metadata_sha256") != core.sha256_file(
            BASE_CACHE / "metadata.json"
        ):
            raise RuntimeError("Augmented cache lineage drifted from the Exp050 base cache.")
        aug_bbox53 = np.load(EXP053_ROOT / f"cache/bbox53_augmented_seed{seed}.npy", mmap_mode="r")
        aug_prepared = _build_prepared_arrays(aug_arrays, aug_bbox53, schemas, scalers)
        _save_prepared_cache(
            experiment_root / f"cache/augmented_seed{seed}",
            aug_rows,
            aug_prepared,
            {"population": ["train"], "seed": seed, "test_accessed": False},
        )
    scaler_payload = {name: _scaler_to_json(value) for name, value in scalers.items()}
    schema_payload = {name: list(value) for name, value in schemas.items()}
    atomic_json(scaler_payload, experiment_root / "metrics/train_only_scalers.json")
    atomic_json(schema_payload, experiment_root / "metrics/feature_schemas.json")
    artifacts = {
        "cache/trainval/metadata.json": core.sha256_file(experiment_root / "cache/trainval/metadata.json"),
        "metrics/train_only_scalers.json": core.sha256_file(experiment_root / "metrics/train_only_scalers.json"),
        "metrics/feature_schemas.json": core.sha256_file(experiment_root / "metrics/feature_schemas.json"),
    }
    for seed in core.SEEDS:
        relative = f"cache/augmented_seed{seed}/metadata.json"
        artifacts[relative] = core.sha256_file(experiment_root / relative)
    payload = {
        "phase": "prepare",
        "created_at": utc_now(),
        "train_rows": core.EXPECTED_SPLIT_COUNTS["train"],
        "val_rows": core.EXPECTED_SPLIT_COUNTS["val"],
        "test_accessed": False,
        "source_invariants_verified": True,
        "source_exp050_metadata_sha256": core.sha256_file(BASE_CACHE / "metadata.json"),
        "source_exp053_bbox53_hashes": bbox_hashes,
        "source_exp012_feature_definitions_sha256": exp012_schema_hash,
        "prepared_trainval_sequence_id_sha256": trainval_metadata["sequence_id_sha256"],
        "artifacts": artifacts,
    }
    atomic_json(payload, marker)
    log_line(experiment_root / "logs/prepare.log", "phase=prepare status=complete test_accessed=false")
    return payload


def authenticate_final_candidate(root: Path = ROOT) -> dict[str, object]:
    experiment_root = Path(root)
    marker = experiment_root / "FROZEN_FINAL_CANDIDATE.json"
    if not marker.is_file():
        raise RuntimeError("Final evaluation requires a frozen final candidate marker.")
    payload = json.loads(marker.read_text(encoding="utf-8"))
    try:
        candidate = core.ModelConfig(**payload["candidate"])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("Frozen final candidate identity is invalid.") from error
    if payload.get("candidate_id") != candidate.candidate_id:
        raise RuntimeError("Frozen final candidate identity drifted.")
    checkpoints = payload.get("checkpoint_artifacts", {})
    if not isinstance(checkpoints, dict) or len(checkpoints) != len(core.SEEDS):
        raise RuntimeError("Final evaluation requires three authenticated checkpoints.")
    expected_paths = {
        f"checkpoints/{candidate.candidate_id}__seed{seed}.best.pt" for seed in core.SEEDS
    }
    if set(checkpoints) != expected_paths:
        raise RuntimeError("Final evaluation requires three authenticated checkpoints.")
    prepare = assert_training_allowed(experiment_root)
    prepare_sha256 = core.sha256_file(experiment_root / ".prepare.complete.json")
    authenticate_artifacts(experiment_root, checkpoints)
    for relative in sorted(checkpoints):
        checkpoint = torch.load(experiment_root / relative, map_location="cpu", weights_only=False)
        if checkpoint.get("candidate_id") != candidate.candidate_id:
            raise RuntimeError("Frozen checkpoint candidate identity drifted.")
        if checkpoint.get("prepare_sha256") != prepare_sha256:
            raise RuntimeError("Frozen checkpoint preprocessing lineage drifted.")
        contract = checkpoint.get("training_contract", {})
        if contract.get("prepare_sha256") != prepare_sha256 or contract.get("prepare_test_accessed") is not False:
            raise RuntimeError("Frozen checkpoint training contract drifted.")
    if prepare.get("test_accessed") is not False:
        raise RuntimeError("Frozen candidate preparation must exclude set03.")
    validation_path = experiment_root / "metrics/validation_candidate_results.csv"
    if (
        not validation_path.is_file()
        or core.sha256_file(validation_path) != payload.get("validation_results_sha256")
    ):
        raise RuntimeError("Frozen validation-selection evidence failed authentication.")
    validation_results = pd.read_csv(validation_path)
    anchor = _load_selected_anchor(experiment_root)
    validate_final_selection_contract(payload, validation_results, anchor)
    return payload


def validate_final_selection_contract(
    payload: Mapping[str, object],
    validation_results: pd.DataFrame,
    anchor: core.ModelConfig,
) -> None:
    if payload.get("phase") != "freeze_final_candidate" or payload.get("test_accessed") is not False:
        raise RuntimeError("Frozen final-candidate phase/test-access contract drifted.")
    if payload.get("seeds") != list(core.SEEDS):
        raise RuntimeError("Frozen final-candidate seed contract drifted.")
    expected_candidates = stage2_candidates(anchor)
    expected_ids = {candidate.candidate_id for candidate in expected_candidates}
    if (
        set(validation_results["candidate_id"].astype(str)) != expected_ids
        or len(validation_results) != len(expected_candidates) * len(core.SEEDS)
    ):
        raise RuntimeError("Frozen final-candidate validation grid drifted.")
    selected_id, _ = select_validation_candidate(validation_results)
    if payload.get("candidate_id") != selected_id:
        raise RuntimeError("Frozen final candidate is not the validation winner.")


def build_freeze_payload(
    candidate: core.ModelConfig,
    validation_results: pd.DataFrame,
    checkpoints: dict[int, Path],
    root: Path = ROOT,
) -> dict[str, object]:
    selected = validation_results.loc[
        validation_results["candidate_id"].astype(str).eq(candidate.candidate_id)
    ].copy()
    if set(selected["seed"].astype(int)) != set(core.SEEDS) or len(selected) != len(core.SEEDS):
        raise RuntimeError("Final freeze requires the same candidate for all three seeds.")
    if set(int(seed) for seed in checkpoints) != set(core.SEEDS):
        raise RuntimeError("Final freeze requires three candidate checkpoints.")
    experiment_root = Path(root)
    checkpoint_artifacts: dict[str, str] = {}
    for seed in core.SEEDS:
        path = Path(checkpoints[seed])
        if not path.is_file():
            raise RuntimeError(f"Missing candidate checkpoint for seed {seed}.")
        try:
            relative = str(path.relative_to(experiment_root))
        except ValueError as error:
            raise RuntimeError("Candidate checkpoints must be inside the experiment root.") from error
        checkpoint_artifacts[relative] = core.sha256_file(path)
    selected_records = selected.sort_values("seed").to_dict("records")
    return {
        "candidate": candidate.to_dict(),
        "candidate_id": candidate.candidate_id,
        "seeds": list(core.SEEDS),
        "checkpoint_artifacts": checkpoint_artifacts,
        "selected_validation_results": selected_records,
        "selected_validation_results_canonical_sha256": core.canonical_json_sha256(
            selected_records
        ),
    }


def _candidate_inputs(
    arrays: Mapping[str, np.ndarray], config: core.ModelConfig, indices: np.ndarray | None = None
) -> dict[str, np.ndarray]:
    def take(name: str) -> np.ndarray:
        value = np.asarray(arrays[name], dtype=np.float32)
        return value if indices is None else value[indices]

    output = {"summary": take(f"summary_{config.biu_mode}")}
    if config.bbox_temporal_mode != "summary_only":
        output["bbox_temporal"] = take(f"bbox_{config.bbox_temporal_mode}")
    if config.context_mode in {"ego", "ego_yolop"}:
        output["ego"] = take("ego")
    if config.context_mode in {"yolop", "ego_yolop"}:
        output["yolop"] = take("yolop")
    return output


def _to_device(
    arrays: Mapping[str, np.ndarray], device: torch.device
) -> dict[str, torch.Tensor]:
    return {
        name: torch.as_tensor(np.array(value, dtype=np.float32, copy=True), device=device)
        for name, value in arrays.items()
    }


def _model_outputs(
    model: core.CompactResidualBiuLstm,
    tensors: Mapping[str, torch.Tensor],
    n_rows: int,
    batch_size: int = 512,
) -> dict[str, np.ndarray]:
    model.eval()
    collected = {name: [] for name in ("probability", "anchor_probability", "gate", "bounded_delta", "applied_correction")}
    with torch.no_grad():
        for start in range(0, n_rows, batch_size):
            batch = {name: value[start : start + batch_size] for name, value in tensors.items()}
            output = model(batch)
            collected["probability"].append(torch.sigmoid(output["final_logit"]).cpu().numpy())
            collected["anchor_probability"].append(torch.sigmoid(output["anchor_logit"]).cpu().numpy())
            for name in ("gate", "bounded_delta", "applied_correction"):
                collected[name].append(output[name].cpu().numpy())
    return {name: np.concatenate(parts) for name, parts in collected.items()}


def _training_marker(root: Path, candidate: core.ModelConfig, seed: int) -> Path:
    return Path(root) / f".train__{candidate.candidate_id}__seed{seed}.complete.json"


def _checkpoint_path(root: Path, candidate: core.ModelConfig, seed: int) -> Path:
    return Path(root) / f"checkpoints/{candidate.candidate_id}__seed{seed}.best.pt"


def run_training(
    candidate: core.ModelConfig,
    seed: int,
    *,
    root: Path = ROOT,
    device: torch.device | None = None,
    max_epochs: int = core.MAX_EPOCHS,
    early_stopping_patience: int = core.EARLY_STOPPING_PATIENCE,
) -> dict[str, object]:
    if seed not in core.SEEDS:
        raise ValueError("Training seed is outside the frozen grid.")
    if max_epochs < 1 or early_stopping_patience < 1:
        raise ValueError("Training limits must be positive.")
    experiment_root = Path(root)
    prepare_payload = assert_training_allowed(experiment_root)
    if device is None:
        if not torch.cuda.is_available():
            raise RuntimeError("Full Exp055 training requires an available CUDA device.")
        device = torch.device("cuda")
    training_contract = {
        "candidate": candidate.to_dict(),
        "seed": seed,
        "prepare_sha256": core.sha256_file(experiment_root / ".prepare.complete.json"),
        "core_sha256": core.sha256_file(Path(core.__file__)),
        "runner_sha256": core.sha256_file(Path(__file__)),
        "device_type": device.type,
        "optimizer": "AdamW",
        "learning_rate": core.LEARNING_RATE,
        "weight_decay": core.WEIGHT_DECAY,
        "batch_size": core.BATCH_SIZE,
        "max_epochs": max_epochs,
        "early_stopping_patience": early_stopping_patience,
        "gradient_clip_norm": core.GRADIENT_CLIP_NORM,
        "anchor_loss_weight": core.ANCHOR_LOSS_WEIGHT,
        "residual_penalty_weight": core.RESIDUAL_PENALTY_WEIGHT,
        "prepare_test_accessed": prepare_payload["test_accessed"],
    }
    training_contract_sha256 = core.canonical_json_sha256(training_contract)
    marker_path = _training_marker(experiment_root, candidate, seed)
    if marker_path.is_file():
        payload = json.loads(marker_path.read_text(encoding="utf-8"))
        if payload.get("training_contract_sha256") != training_contract_sha256:
            raise RuntimeError("Existing completed-run training contract drifted.")
        authenticate_artifacts(experiment_root, payload.get("artifacts", {}))
        return payload
    train_rows, train_arrays, _ = load_prepared_cache(
        experiment_root / f"cache/augmented_seed{seed}"
    )
    base_rows, base_arrays, _ = load_prepared_cache(experiment_root / "cache/trainval")
    val_indices = np.flatnonzero(base_rows["split"].astype(str).eq("val").to_numpy())
    if len(val_indices) != core.EXPECTED_SPLIT_COUNTS["val"]:
        raise RuntimeError("Validation population drifted during training.")
    train_inputs = _to_device(_candidate_inputs(train_arrays, candidate), device)
    val_inputs = _to_device(_candidate_inputs(base_arrays, candidate, val_indices), device)
    train_labels = torch.as_tensor(
        train_rows["label"].to_numpy(dtype=np.float32), dtype=torch.float32, device=device
    )
    val_labels = base_rows.iloc[val_indices]["label"].to_numpy(dtype=np.int64)
    core.seed_everything(seed)
    model = core.CompactResidualBiuLstm(candidate).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=core.LEARNING_RATE, weight_decay=core.WEIGHT_DECAY
    )
    rng = np.random.RandomState(seed)
    history: list[dict[str, object]] = []
    best_score = -1.0
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    state_path = experiment_root / f"checkpoints/{candidate.candidate_id}__seed{seed}.state.pt"
    start_epoch = 1
    if state_path.is_file():
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if state.get("candidate_id") != candidate.candidate_id or int(state.get("seed", -1)) != seed:
            raise RuntimeError("Resumable training-state identity drifted.")
        if state.get("training_contract_sha256") != training_contract_sha256:
            raise RuntimeError("Resumable training contract drifted.")
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        rng.set_state(state["numpy_rng_state"])
        torch.set_rng_state(state["torch_rng_state"])
        if device.type == "cuda" and state.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng_state"])
        history = list(state["history"])
        best_score = float(state["best_score"])
        best_epoch = int(state["best_epoch"])
        best_state = state["best_state"]
        epochs_without_improvement = int(state["epochs_without_improvement"])
        start_epoch = int(state["last_epoch"]) + 1
    log = experiment_root / f"logs/train__{candidate.candidate_id}__seed{seed}.log"
    log_line(
        log,
        f"phase=train status=starting candidate={candidate.candidate_id} seed={seed} device={device} start_epoch={start_epoch}",
    )
    started = time.perf_counter()
    n_train = len(train_labels)
    for epoch in range(start_epoch, max_epochs + 1):
        model.train()
        order = rng.permutation(n_train)
        totals = {"total": 0.0, "primary": 0.0, "anchor": 0.0, "residual": 0.0}
        correct = 0
        for start in range(0, n_train, core.BATCH_SIZE):
            index = torch.as_tensor(
                order[start : start + core.BATCH_SIZE], dtype=torch.long, device=device
            )
            batch = {name: value.index_select(0, index) for name, value in train_inputs.items()}
            labels = train_labels.index_select(0, index)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch)
            losses = core.training_objective(output, labels)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), core.GRADIENT_CLIP_NORM)
            optimizer.step()
            batch_size = len(index)
            for name in totals:
                totals[name] += float(losses[name].detach().item()) * batch_size
            correct += int(((output["final_logit"] >= 0.0) == labels.bool()).sum().item())
        validation = _model_outputs(model, val_inputs, len(val_indices))
        metrics = core.compute_metrics(val_labels, validation["probability"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": totals["total"] / n_train,
                "train_primary_loss": totals["primary"] / n_train,
                "train_anchor_loss": totals["anchor"] / n_train,
                "train_residual_penalty": totals["residual"] / n_train,
                "train_accuracy": correct / n_train,
                **{f"val_{name}": value for name, value in metrics.items()},
                "val_gate_mean": float(np.mean(validation["gate"])),
                "val_abs_correction_mean": float(np.mean(np.abs(validation["applied_correction"]))),
            }
        )
        score = float(metrics["selection_score"])
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(
                {name: value.detach().cpu() for name, value in model.state_dict().items()}
            )
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        atomic_csv(pd.DataFrame(history), experiment_root / f"logs/history__{candidate.candidate_id}__seed{seed}.csv")
        atomic_torch_save(
            {
                "candidate_id": candidate.candidate_id,
                "seed": seed,
                "training_contract": training_contract,
                "training_contract_sha256": training_contract_sha256,
                "last_epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "numpy_rng_state": rng.get_state(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                "history": history,
                "best_score": best_score,
                "best_epoch": best_epoch,
                "best_state": best_state,
                "epochs_without_improvement": epochs_without_improvement,
            },
            state_path,
        )
        log_line(
            log,
            f"epoch={epoch}/{max_epochs} train_loss={totals['total']/n_train:.6f} val_f1={float(metrics['f1']):.6f} val_pr_auc={float(metrics['pr_auc']):.6f} score={score:.6f} best_epoch={best_epoch}",
        )
        if epochs_without_improvement >= early_stopping_patience:
            break
    if best_state is None:
        raise RuntimeError("Training produced no validation checkpoint.")
    model.load_state_dict(best_state)
    model.to(device)
    validation = _model_outputs(model, val_inputs, len(val_indices))
    metrics = core.compute_metrics(val_labels, validation["probability"])
    checkpoint_path = _checkpoint_path(experiment_root, candidate, seed)
    atomic_torch_save(
        {
            "candidate": candidate.to_dict(),
            "candidate_id": candidate.candidate_id,
            "seed": seed,
            "training_contract": training_contract,
            "training_contract_sha256": training_contract_sha256,
            "best_epoch": best_epoch,
            "best_validation_selection_score": best_score,
            "validation_metrics": metrics,
            "model_state": best_state,
            "total_parameters": core.count_parameters(model),
            "prepare_sha256": core.sha256_file(experiment_root / ".prepare.complete.json"),
        },
        checkpoint_path,
    )
    prediction = base_rows.iloc[val_indices][
        [
            "sequence_id",
            "split",
            "set_id",
            "video_id",
            "ped_id",
            "start_frame",
            "end_frame",
            "event_frame",
            "time_to_event_frames",
            "label",
        ]
    ].copy()
    prediction["candidate_id"] = candidate.candidate_id
    prediction["seed"] = seed
    prediction["threshold"] = core.FIXED_THRESHOLD
    for name, value in validation.items():
        prediction[name] = value
    prediction["prediction"] = (prediction["probability"] >= core.FIXED_THRESHOLD).astype(int)
    prediction_path = experiment_root / f"predictions/validation__{candidate.candidate_id}__seed{seed}.csv"
    atomic_csv(prediction, prediction_path)
    history_path = experiment_root / f"logs/history__{candidate.candidate_id}__seed{seed}.csv"
    state_path.unlink(missing_ok=True)
    marker = {
        "phase": "train",
        "created_at": utc_now(),
        "candidate": candidate.to_dict(),
        "candidate_id": candidate.candidate_id,
        "seed": seed,
        "training_contract": training_contract,
        "training_contract_sha256": training_contract_sha256,
        "device": str(device),
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "training_seconds": time.perf_counter() - started,
        "validation_metrics": metrics,
        "total_parameters": core.count_parameters(model),
        "artifacts": {
            str(checkpoint_path.relative_to(experiment_root)): core.sha256_file(checkpoint_path),
            str(prediction_path.relative_to(experiment_root)): core.sha256_file(prediction_path),
            str(history_path.relative_to(experiment_root)): core.sha256_file(history_path),
        },
    }
    atomic_json(marker, marker_path)
    log_line(log, f"phase=train status=complete best_epoch={best_epoch} score={best_score:.6f}")
    del model, train_inputs, val_inputs, train_labels
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return marker


def _collect_validation_results(
    candidates: tuple[core.ModelConfig, ...],
    *,
    root: Path,
    expected_count: int,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    contracts: list[Mapping[str, object]] = []
    for candidate in candidates:
        for seed in core.SEEDS:
            marker_path = _training_marker(root, candidate, seed)
            if not marker_path.is_file():
                raise RuntimeError(
                    f"Selection requires {expected_count} authenticated validation runs."
                )
            payload = json.loads(marker_path.read_text(encoding="utf-8"))
            if payload.get("candidate_id") != candidate.candidate_id or int(payload.get("seed", -1)) != seed:
                raise RuntimeError("Validation training-marker identity drifted.")
            authenticate_artifacts(root, payload.get("artifacts", {}))
            contract = payload.get("training_contract")
            if not isinstance(contract, dict):
                raise RuntimeError("Validation run lacks its frozen training contract.")
            if core.canonical_json_sha256(contract) != payload.get("training_contract_sha256"):
                raise RuntimeError("Validation training-contract authentication failed.")
            contracts.append(contract)
            metrics = payload.get("validation_metrics", {})
            records.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "biu_mode": candidate.biu_mode,
                    "bbox_temporal_mode": candidate.bbox_temporal_mode,
                    "context_mode": candidate.context_mode,
                    "seed": seed,
                    "best_epoch": int(payload["best_epoch"]),
                    "epochs_completed": int(payload["epochs_completed"]),
                    "training_seconds": float(payload["training_seconds"]),
                    "total_parameters": int(payload["total_parameters"]),
                    **{name: metrics[name] for name in metrics},
                }
            )
    if len(records) != expected_count:
        raise RuntimeError(f"Selection requires {expected_count} authenticated validation runs.")
    validate_grid_training_contracts(contracts)
    return pd.DataFrame.from_records(records)


def run_stage1(root: Path = ROOT, device: torch.device | None = None) -> dict[str, object]:
    run_prepare(root=root)
    for candidate in stage1_candidates():
        for seed in core.SEEDS:
            run_training(candidate, seed, root=root, device=device)
    return freeze_selected_anchor(root=root)


def freeze_selected_anchor(root: Path = ROOT) -> dict[str, object]:
    experiment_root = Path(root)
    assert_training_allowed(experiment_root)
    candidates = stage1_candidates()
    results = _collect_validation_results(
        candidates, root=experiment_root, expected_count=len(candidates) * len(core.SEEDS)
    )
    selected_id, summary = select_validation_candidate(results)
    selected = next(candidate for candidate in candidates if candidate.candidate_id == selected_id)
    results_path = experiment_root / "metrics/stage1_validation_results.csv"
    summary_path = experiment_root / "metrics/stage1_validation_summary.csv"
    atomic_csv(results.sort_values(["candidate_id", "seed"]), results_path)
    atomic_csv(summary, summary_path)
    checkpoints = {
        seed: _checkpoint_path(experiment_root, selected, seed) for seed in core.SEEDS
    }
    payload = build_freeze_payload(selected, results, checkpoints, experiment_root)
    payload.update(
        {
            "phase": "select_anchor",
            "created_at": utc_now(),
            "selection_rule": "mean_harmonic_f1_pr_auc_then_pr_auc_then_f1_then_parameters_then_id",
            "stage1_results_sha256": core.sha256_file(results_path),
            "stage1_summary_sha256": core.sha256_file(summary_path),
        }
    )
    atomic_json(payload, experiment_root / "SELECTED_ANCHOR.json")
    log_line(
        experiment_root / "logs/development.log",
        f"phase=select_anchor status=complete candidate={selected.candidate_id}",
    )
    return payload


def _load_selected_anchor(root: Path) -> core.ModelConfig:
    marker = Path(root) / "SELECTED_ANCHOR.json"
    if not marker.is_file():
        raise RuntimeError("Stage 2 requires a frozen selected anchor.")
    payload = json.loads(marker.read_text(encoding="utf-8"))
    candidate = core.ModelConfig(**payload["candidate"])
    if candidate.context_mode != "none" or payload.get("candidate_id") != candidate.candidate_id:
        raise RuntimeError("Selected-anchor identity drifted.")
    authenticate_artifacts(Path(root), payload.get("checkpoint_artifacts", {}))
    results_path = Path(root) / "metrics/stage1_validation_results.csv"
    if not results_path.is_file() or core.sha256_file(results_path) != payload.get("stage1_results_sha256"):
        raise RuntimeError("Selected-anchor evidence failed authentication.")
    return candidate


def run_stage2(root: Path = ROOT, device: torch.device | None = None) -> dict[str, object]:
    experiment_root = Path(root)
    anchor = _load_selected_anchor(experiment_root)
    candidates = stage2_candidates(anchor)
    for candidate in candidates[1:]:
        for seed in core.SEEDS:
            run_training(candidate, seed, root=experiment_root, device=device)
    results = _collect_validation_results(
        candidates, root=experiment_root, expected_count=len(candidates) * len(core.SEEDS)
    )
    selected_id, summary = select_validation_candidate(results)
    selected = next(candidate for candidate in candidates if candidate.candidate_id == selected_id)
    results_path = experiment_root / "metrics/validation_candidate_results.csv"
    summary_path = experiment_root / "metrics/validation_candidate_summary.csv"
    atomic_csv(results.sort_values(["candidate_id", "seed"]), results_path)
    atomic_csv(summary, summary_path)
    checkpoints = {
        seed: _checkpoint_path(experiment_root, selected, seed) for seed in core.SEEDS
    }
    payload = build_freeze_payload(selected, results, checkpoints, experiment_root)
    payload.update(
        {
            "phase": "freeze_final_candidate",
            "created_at": utc_now(),
            "selection_rule": "mean_harmonic_f1_pr_auc_then_pr_auc_then_f1_then_parameters_then_id",
            "validation_results_sha256": core.sha256_file(results_path),
            "validation_summary_sha256": core.sha256_file(summary_path),
            "test_accessed": False,
        }
    )
    atomic_json(payload, experiment_root / "FROZEN_FINAL_CANDIDATE.json")
    log_line(
        experiment_root / "logs/development.log",
        f"phase=freeze_final status=complete candidate={selected.candidate_id} test_accessed=false",
    )
    return payload


def run_development(root: Path = ROOT, device: torch.device | None = None) -> dict[str, object]:
    run_stage1(root=root, device=device)
    return run_stage2(root=root, device=device)


def _validate_test_rows(rows: pd.DataFrame) -> pd.DataFrame:
    checked = rows.reset_index(drop=True).copy()
    if len(checked) != core.EXPECTED_SPLIT_COUNTS["test"]:
        raise RuntimeError("Frozen test-row count drifted.")
    if set(checked["split"].astype(str)) != {"test"} or set(checked["set_id"].astype(str)) != {"set03"}:
        raise RuntimeError("Final evaluation population must be designated set03.")
    if checked["sequence_id"].duplicated().any():
        raise RuntimeError("Final evaluation sequence IDs must be unique.")
    consecutive = (
        checked["observation_length"].astype(int).eq(core.SEQ_LEN)
        & checked["frame_step"].astype(int).eq(1)
        & checked["frame_sampling"].astype(str).eq("consecutive")
        & (checked["end_frame"].astype(int) - checked["start_frame"].astype(int)).eq(core.SEQ_LEN - 1)
    )
    if not consecutive.all() or not checked["time_to_event_frames"].astype(int).between(30, 60).all():
        raise RuntimeError("Final evaluation windows drifted from 16-frame/TTE30--60.")
    if not (checked["event_frame"].astype(int) > checked["end_frame"].astype(int)).all():
        raise RuntimeError("Final evaluation contains future leakage.")
    return checked


def validate_test_source_alignment(
    source_metadata: Mapping[str, object],
    exp050_evaluation: Mapping[str, object],
    exp053_evaluation: Mapping[str, object],
    *,
    bbox_rows: int,
) -> None:
    counts = {
        int(source_metadata.get("n_rows", -1)),
        int(exp050_evaluation.get("test_rows", -2)),
        int(exp053_evaluation.get("test_rows", -3)),
        int(bbox_rows),
    }
    hashes = {
        str(source_metadata.get("sequence_id_sha256", "")),
        str(exp050_evaluation.get("test_sequence_id_sha256", "missing-exp050")),
        str(exp053_evaluation.get("test_sequence_id_sha256", "missing-exp053")),
    }
    if counts != {core.EXPECTED_SPLIT_COUNTS["test"]} or len(hashes) != 1 or "" in hashes:
        raise RuntimeError("Exp050/Exp053 test cohort alignment failed.")


def _prepare_test_cache(root: Path) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, object]]:
    experiment_root = Path(root)
    frozen = authenticate_final_candidate(experiment_root)
    frozen_sha256 = core.sha256_file(experiment_root / "FROZEN_FINAL_CANDIDATE.json")
    prepare_sha256 = core.sha256_file(experiment_root / ".prepare.complete.json")
    path = experiment_root / "cache/test"
    if (path / "metadata.json").is_file():
        cached_rows, cached_arrays, cached_metadata = load_prepared_cache(path)
        _validate_test_rows(cached_rows)
        if (
            cached_metadata.get("candidate_id") != frozen["candidate_id"]
            or cached_metadata.get("frozen_candidate_sha256") != frozen_sha256
            or cached_metadata.get("prepare_sha256") != prepare_sha256
        ):
            raise RuntimeError("Existing Exp055 test cache lineage drifted.")
        return cached_rows, cached_arrays, cached_metadata
    source_path = EXP050_ROOT / "cache/set03_final"
    rows, source_arrays, source_metadata = _load_source_cache(source_path)
    rows = _validate_test_rows(rows)
    exp050_evaluation = json.loads(
        (EXP050_ROOT / ".evaluation.complete.json").read_text(encoding="utf-8")
    )
    exp053_evaluation = json.loads(
        (EXP053_ROOT / ".evaluation.complete.json").read_text(encoding="utf-8")
    )
    bbox_relative = "cache/bbox53_test.npy"
    expected_bbox_hash = exp053_evaluation.get("artifacts", {}).get(bbox_relative)
    bbox_path = EXP053_ROOT / bbox_relative
    if not expected_bbox_hash or core.sha256_file(bbox_path) != expected_bbox_hash:
        raise RuntimeError("Exp053 test BBox53 failed authentication.")
    bbox53 = np.load(bbox_path, mmap_mode="r")
    validate_test_source_alignment(
        source_metadata,
        exp050_evaluation,
        exp053_evaluation,
        bbox_rows=len(bbox53),
    )
    scaler_payload = json.loads(
        (experiment_root / "metrics/train_only_scalers.json").read_text(encoding="utf-8")
    )
    scalers = {name: _scaler_from_json(value) for name, value in scaler_payload.items()}
    schema_payload = json.loads(
        (experiment_root / "metrics/feature_schemas.json").read_text(encoding="utf-8")
    )
    schemas = {name: tuple(value) for name, value in schema_payload.items()}
    if (
        tuple(source_metadata.get("biu_v1_feature_names", ())) != schemas["biu_v1"]
        or tuple(source_metadata.get("biu_v2_feature_names", ())) != schemas["biu_v2_source"]
    ):
        raise RuntimeError("Final test BIU source schema drifted from training.")
    prepared = _build_prepared_arrays(source_arrays, bbox53, schemas, scalers)
    metadata = _save_prepared_cache(
        path,
        rows,
        prepared,
        {
            "population": ["test"],
            "test_accessed": True,
            "source_exp050_test_metadata_sha256": core.sha256_file(source_path / "metadata.json"),
            "source_exp053_bbox53_test_sha256": expected_bbox_hash,
            "source_sequence_id_sha256": source_metadata["sequence_id_sha256"],
            "candidate_id": frozen["candidate_id"],
            "frozen_candidate_sha256": frozen_sha256,
            "prepare_sha256": prepare_sha256,
        },
    )
    atomic_json(
        {
            "phase": "prepare_test",
            "created_at": utc_now(),
            "candidate_id": authenticate_final_candidate(experiment_root)["candidate_id"],
            "test_rows": len(rows),
            "test_sequence_id_sha256": metadata["sequence_id_sha256"],
            "artifacts": {"cache/test/metadata.json": core.sha256_file(path / "metadata.json")},
        },
        experiment_root / ".test_cache.complete.json",
    )
    return load_prepared_cache(path)


def _summary_table(results: pd.DataFrame) -> pd.DataFrame:
    metric_names = (
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
        "high_confidence_fp",
        "high_confidence_fn",
        "anchor_f1",
        "anchor_pr_auc",
        "gate_mean",
        "abs_correction_mean",
        "latency_ms_per_sample",
    )
    record: dict[str, object] = {
        "candidate_id": str(results.iloc[0]["candidate_id"]),
        "total_parameters": int(results["total_parameters"].max()),
    }
    for name in metric_names:
        record[f"{name}_mean"] = float(results[name].mean())
        record[f"{name}_std"] = float(results[name].std(ddof=1))
    return pd.DataFrame([record])


def run_final_evaluation(
    root: Path = ROOT, device: torch.device | None = None
) -> dict[str, object]:
    experiment_root = Path(root)
    frozen = authenticate_final_candidate(experiment_root)
    if device is None:
        if not torch.cuda.is_available():
            raise RuntimeError("Full Exp055 final evaluation requires an available CUDA device.")
        device = torch.device("cuda")
    candidate = core.ModelConfig(**frozen["candidate"])
    test_rows, test_arrays, test_metadata = _prepare_test_cache(experiment_root)
    labels = test_rows["label"].to_numpy(dtype=np.int64)
    inputs = _to_device(_candidate_inputs(test_arrays, candidate), device)
    records: list[dict[str, object]] = []
    prediction_artifacts: dict[str, str] = {}
    for seed in core.SEEDS:
        checkpoint_path = _checkpoint_path(experiment_root, candidate, seed)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("candidate_id") != candidate.candidate_id or int(checkpoint.get("seed", -1)) != seed:
            raise RuntimeError("Final checkpoint identity drifted.")
        model = core.CompactResidualBiuLstm(candidate).to(device)
        model.load_state_dict(checkpoint["model_state"])
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        output = _model_outputs(model, inputs, len(test_rows))
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        metrics = core.compute_metrics(labels, output["probability"])
        anchor_metrics = core.compute_metrics(labels, output["anchor_probability"])
        record = {
            "candidate_id": candidate.candidate_id,
            "seed": seed,
            "best_epoch": int(checkpoint["best_epoch"]),
            "total_parameters": core.count_parameters(model),
            **metrics,
            "anchor_f1": float(anchor_metrics["f1"]),
            "anchor_pr_auc": float(anchor_metrics["pr_auc"]),
            "anchor_roc_auc": float(anchor_metrics["roc_auc"]),
            "gate_mean": float(np.mean(output["gate"])),
            "gate_median": float(np.median(output["gate"])),
            "abs_correction_mean": float(np.mean(np.abs(output["applied_correction"]))),
            "latency_seconds": elapsed,
            "latency_ms_per_sample": 1000.0 * elapsed / len(test_rows),
            "device": str(device),
        }
        records.append(record)
        prediction = test_rows[
            [
                "sequence_id",
                "split",
                "set_id",
                "video_id",
                "ped_id",
                "start_frame",
                "end_frame",
                "event_frame",
                "time_to_event_frames",
                "label",
            ]
        ].copy()
        prediction["candidate_id"] = candidate.candidate_id
        prediction["seed"] = seed
        prediction["threshold"] = core.FIXED_THRESHOLD
        for name, value in output.items():
            prediction[name] = value
        prediction["prediction"] = (prediction["probability"] >= core.FIXED_THRESHOLD).astype(int)
        path = experiment_root / f"predictions/set03__{candidate.candidate_id}__seed{seed}.csv"
        atomic_csv(prediction, path)
        prediction_artifacts[str(path.relative_to(experiment_root))] = core.sha256_file(path)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    results = pd.DataFrame.from_records(records)
    summary = _summary_table(results)
    results_path = experiment_root / "metrics/final_per_seed_results.csv"
    summary_path = experiment_root / "metrics/final_summary_results.csv"
    atomic_csv(results, results_path)
    atomic_csv(summary, summary_path)
    row = summary.iloc[0]
    success = bool(
        row["f1_mean"] > 0.869
        and row["pr_auc_mean"] > 0.916
        and row["roc_auc_mean"] > 0.965
    )
    report = (
        "# Exp055 Final Report\n\n"
        f"Frozen candidate: `{candidate.candidate_id}`.\n\n"
        f"Mean fixed-0.5 F1: {row['f1_mean']:.6f}.  "
        f"Mean PR-AUC: {row['pr_auc_mean']:.6f}.  "
        f"Mean ROC-AUC: {row['roc_auc_mean']:.6f}.\n\n"
        f"All three predeclared targets satisfied: **{'yes' if success else 'no'}**.\n\n"
        "Set03 was accessed only after validation-only candidate selection was frozen. "
        "Prior independent set03 exposure remains a disclosed limitation.\n"
    )
    (experiment_root / "FINAL_REPORT.md").write_text(report, encoding="utf-8")
    artifacts = {
        "metrics/final_per_seed_results.csv": core.sha256_file(results_path),
        "metrics/final_summary_results.csv": core.sha256_file(summary_path),
        "FINAL_REPORT.md": core.sha256_file(experiment_root / "FINAL_REPORT.md"),
        **prediction_artifacts,
    }
    completion = {
        "phase": "complete",
        "created_at": utc_now(),
        "candidate_id": candidate.candidate_id,
        "test_rows": len(test_rows),
        "test_sequence_id_sha256": test_metadata["sequence_id_sha256"],
        "all_targets_satisfied": success,
        "artifacts": artifacts,
    }
    atomic_json(completion, experiment_root / "EXPERIMENT_COMPLETE")
    log_line(
        experiment_root / "logs/evaluate.log",
        f"phase=evaluate status=complete candidate={candidate.candidate_id} f1={row['f1_mean']:.6f} pr_auc={row['pr_auc_mean']:.6f} roc_auc={row['roc_auc_mean']:.6f}",
    )
    return completion


def run_verify(root: Path = ROOT) -> dict[str, object]:
    experiment_root = Path(root)
    prepare = assert_training_allowed(experiment_root)
    result: dict[str, object] = {
        "prepare_authenticated": True,
        "test_accessed_during_prepare": prepare["test_accessed"],
        "stage1_complete": (experiment_root / "SELECTED_ANCHOR.json").is_file(),
        "final_candidate_frozen": (experiment_root / "FROZEN_FINAL_CANDIDATE.json").is_file(),
        "experiment_complete": (experiment_root / "EXPERIMENT_COMPLETE").is_file(),
    }
    if result["final_candidate_frozen"]:
        authenticate_final_candidate(experiment_root)
    if result["experiment_complete"]:
        completion = json.loads((experiment_root / "EXPERIMENT_COMPLETE").read_text(encoding="utf-8"))
        authenticate_artifacts(experiment_root, completion.get("artifacts", {}))
    atomic_json(result, experiment_root / "metrics/verification_report.json")
    return result


def status(root: Path = ROOT) -> dict[str, object]:
    experiment_root = Path(root)
    markers = list(experiment_root.glob(".train__*.complete.json"))
    payload = {
        "prepared": (experiment_root / ".prepare.complete.json").is_file(),
        "training_runs_complete": len(markers),
        "stage1_runs_required": len(stage1_candidates()) * len(core.SEEDS),
        "selected_anchor": (experiment_root / "SELECTED_ANCHOR.json").is_file(),
        "frozen_final_candidate": (experiment_root / "FROZEN_FINAL_CANDIDATE.json").is_file(),
        "evaluated": (experiment_root / "EXPERIMENT_COMPLETE").is_file(),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase",
        choices=("prepare", "stage1", "select-anchor", "stage2", "develop", "evaluate", "verify", "status"),
    )
    args = parser.parse_args()
    if args.phase == "prepare":
        run_prepare()
    elif args.phase == "stage1":
        run_stage1()
    elif args.phase == "select-anchor":
        freeze_selected_anchor()
    elif args.phase == "stage2":
        run_stage2()
    elif args.phase == "develop":
        run_development()
    elif args.phase == "evaluate":
        run_final_evaluation()
    elif args.phase == "verify":
        print(json.dumps(run_verify(), indent=2, sort_keys=True))
    else:
        status()


if __name__ == "__main__":
    main()
