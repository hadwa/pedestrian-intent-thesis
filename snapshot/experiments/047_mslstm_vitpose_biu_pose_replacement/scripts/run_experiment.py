#!/usr/bin/env python3
"""Phase-oriented runner for Exp047.

The scientific implementation is intentionally added behind fixed subcommands;
the CLI exposes no experimental hyperparameter switches.
"""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from exp047_core import (
    EXP012_MIRROR_PATH,
    EXP012_POSE_PATH,
    EXP012_POSE_SHA256,
    EXPERIMENT_ROOT,
    PIE_DATABASE_PATH,
    PIE_ANNOTATION_ROOT,
    PIE_PARSER_PATH,
    EXP032_PROTOCOL_ROWS,
    EXP032_RUNNER,
    TRAIN_INDEX_PATH,
    SCALERS_PATH,
    SEEDS,
    TRAIN_SETS,
    VAL_SETS,
    VARIANTS,
    Exp047CachedDataset,
    Exp012PoseStore,
    FrozenProtocol,
    SourceManifest,
    YoloPoseStore,
    _atomic_torch_save,
    _predict_loader,
    authenticate_phase_record,
    authenticate_training_records,
    build_error_transition_counts,
    build_frozen_model,
    build_paired_deltas,
    build_stream_cache,
    classify_final_outcome,
    compute_binary_metrics,
    compute_train_scalers,
    compute_pose_audit,
    count_parameters,
    count_pose_path_parameters,
    extract_frozen_biu,
    fit_frozen_epochs,
    load_effective_cohort,
    load_pie_sets_from_xml,
    load_stream_cache,
    protocol_sha256,
    sha256_file,
    source_manifest_sha256,
    summarize_per_seed_results,
    training_rows_only,
    verify_checkpoint_selection_source,
    verify_protocol_rows,
    write_phase_record,
)

CORE_PATH = Path(__file__).with_name("exp047_core.py")
TRAINING_CACHE = "trainval_streams"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run frozen Exp047 phases without tmux.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=EXPERIMENT_ROOT,
        help="Output root; existing scientific artifacts are never overwritten.",
    )
    parser.add_argument("--resume", action="store_true", help="Resume only hash-authenticated phase state.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser("audit", help="Run provenance, protocol, alignment, and cache audits.")
    audit.add_argument("--limit", type=int, help="Engineering smoke limit; cannot authorize training.")
    train = subparsers.add_parser("train", help="Train one frozen variant/seed using train and validation only.")
    train.add_argument("--variant", choices=VARIANTS, required=True)
    train.add_argument("--seed", choices=SEEDS, required=True, type=int)
    subparsers.add_parser("train-all", help="Train all nine frozen runs without tmux.")
    subparsers.add_parser("evaluate", help="Perform the single authorized final set03 evaluation.")
    subparsers.add_parser("finalize", help="Publish final reports after evaluation.")
    verify = subparsers.add_parser("verify", help="Verify an authenticated phase or complete package.")
    verify.add_argument("--phase", dest="verify_phase", choices=("audit", "train", "evaluate", "complete"), required=True)
    return parser


def run_pose_audit(output_root: Path, limit: int | None) -> None:
    output_root = Path(output_root)
    metrics = output_root / "metrics"
    cache = output_root / "cache"
    logs = output_root / "logs"
    metrics.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    log_path = logs / "pose_audit.log"

    def progress(message: str) -> None:
        timestamp = datetime.now(timezone.utc).isoformat()
        line = f"{timestamp} {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    progress("phase=pose_audit status=starting")
    if sha256_file(EXP012_POSE_PATH) != EXP012_POSE_SHA256:
        raise RuntimeError("Primary Exp012 pose SHA-256 does not match the frozen value.")
    if sha256_file(EXP012_MIRROR_PATH) != EXP012_POSE_SHA256:
        raise RuntimeError("Main-BIU mirror is not byte-identical to the frozen Exp012 pose cache.")
    selection = verify_checkpoint_selection_source()
    rows = load_effective_cohort()
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be positive.")
        rows = rows.head(limit).copy()
        progress(f"mode=smoke limit={limit}; training authorization will not be created")
    progress(f"cohort_rows={len(rows)} loading_exp012_pose_store")
    exp012_store = Exp012PoseStore.load(EXP012_POSE_PATH)
    progress(f"exp012_frame_rows={len(exp012_store)} loading_pie_database")
    with PIE_DATABASE_PATH.open("rb") as handle:
        pie_db = pickle.load(handle)
    result = compute_pose_audit(
        rows,
        yolo_store=YoloPoseStore(),
        exp012_store=exp012_store,
        pie_db=pie_db,
        biu_cache_path=cache / "biu_v2_exp012_30frame.npy",
        progress_every=max(1, min(5_000, len(rows))),
        progress=progress,
    )
    coverage_path = metrics / "pose_coverage.csv"
    fallback_path = metrics / "bbox_direction_fallback.csv"
    features_path = metrics / "biu_numeric_feature_names.json"
    manifest_path = metrics / "cohort_manifest.csv"
    result.coverage.to_csv(coverage_path, index=False)
    result.fallback.to_csv(fallback_path, index=False)
    features_path.write_text(json.dumps(result.biu_feature_names, indent=2) + "\n", encoding="utf-8")
    rows.to_csv(manifest_path, index=False)
    payload = {
        "phase": "pose_audit_smoke" if limit is not None else "pose_audit",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_rows": result.n_rows,
        "protocol_sha256": protocol_sha256(FrozenProtocol()),
        "source_manifest_sha256": source_manifest_sha256(SourceManifest.frozen()),
        "runner_sha256": sha256_file(Path(__file__)),
        "checkpoint_selection": selection,
        "artifacts": {
            str(path.relative_to(output_root)): sha256_file(path)
            for path in (coverage_path, fallback_path, features_path, manifest_path, cache / "biu_v2_exp012_30frame.npy")
        },
    }
    marker = output_root / (".pose_audit.smoke.complete.json" if limit is not None else ".pose_audit.complete.json")
    write_phase_record(marker, payload)
    progress(f"phase=pose_audit status=complete marker={marker}")


def _progress_logger(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    def emit(message: str) -> None:
        line = f"{datetime.now(timezone.utc).isoformat()} {message}"
        print(line, flush=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    return emit


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _pie_xml_paths(set_ids: tuple[str, ...]) -> list[Path]:
    paths = []
    for component in ("annotations", "annotations_attributes", "annotations_vehicle"):
        for set_id in set_ids:
            paths.extend(sorted((PIE_ANNOTATION_ROOT / component / set_id).glob("*.xml")))
    if not paths:
        raise FileNotFoundError(f"No PIE XML sources found for {set_ids}.")
    return paths


def _yolo_pose_paths(set_ids: tuple[str, ...]) -> list[Path]:
    paths = []
    root = Path("/home/abdelrahman/Hadwa/MSLSTM-PID/Extracted Poses")
    for set_id in set_ids:
        paths.extend(sorted((root / set_id).glob("*.pkl")))
    if not paths:
        raise FileNotFoundError(f"No YOLO pose sources found for {set_ids}.")
    return paths


def _authenticate_pose_audit(output_root: Path) -> dict[str, object]:
    marker = output_root / ".pose_audit.complete.json"
    if not marker.is_file():
        raise FileNotFoundError(
            "The completed pose audit is required. It will not be recomputed because set03 is now sealed."
        )
    payload = json.loads(marker.read_text(encoding="utf-8"))
    if payload.get("phase") != "pose_audit" or int(payload.get("n_rows", -1)) != 458_300:
        raise ValueError("The pose-audit marker does not describe the frozen full cohort.")
    if payload.get("protocol_sha256") != protocol_sha256(FrozenProtocol()):
        raise ValueError("Pose-audit protocol hash drifted.")
    if payload.get("source_manifest_sha256") != source_manifest_sha256(SourceManifest.frozen()):
        raise ValueError("Pose-audit source manifest drifted.")
    for relative, expected_hash in payload.get("artifacts", {}).items():
        path = output_root / relative
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise ValueError(f"Pose-audit artifact hash mismatch: {path}")
    return payload


def run_pretraining_audit(output_root: Path, resume: bool) -> None:
    output_root = Path(output_root)
    progress = _progress_logger(output_root / "logs" / "pretraining_audit.log")
    marker = output_root / ".audit.complete.json"
    if marker.exists():
        if not resume:
            raise FileExistsError(f"Pretraining audit already exists: {marker}; use --resume to authenticate it.")
        authenticate_phase_record(marker, expected_phase="audit")
        progress("phase=pretraining_audit status=already_complete_authenticated")
        return
    progress("phase=pretraining_audit status=starting")
    pose_payload = _authenticate_pose_audit(output_root)
    if sha256_file(EXP012_POSE_PATH) != EXP012_POSE_SHA256:
        raise ValueError("Frozen Exp012 pose source hash drifted.")
    if sha256_file(EXP012_MIRROR_PATH) != EXP012_POSE_SHA256:
        raise ValueError("Historical main-BIU mirror is no longer byte-identical.")
    manifest_path = output_root / "metrics" / "cohort_manifest.csv"
    n_trainval = 109_934 + 77_288
    rows = pd.read_csv(manifest_path, nrows=n_trainval)
    verify_protocol_rows(rows, expected_counts={"train": 109_934, "val": 77_288})
    if rows["split"].tolist()[:109_934].count("train") != 109_934:
        raise ValueError("Training prefix order drifted.")
    if rows["split"].tolist()[109_934:].count("val") != 77_288:
        raise ValueError("Validation prefix order drifted.")
    rows = rows.reset_index(drop=True)
    rows["cache_index"] = np.arange(len(rows), dtype=np.int64)
    train_label_counts = rows.loc[rows["split"] == "train", "label"].value_counts().to_dict()
    if train_label_counts != {0: 54_967, 1: 54_967}:
        raise ValueError(f"Frozen training cohort is no longer exactly class-balanced: {train_label_counts}")
    progress(f"trainval_rows={len(rows)} parsing_trainval_xml_sets_without_set03")
    pie_db = load_pie_sets_from_xml(TRAIN_SETS + VAL_SETS, training_phase=True)
    computed_scalers = compute_train_scalers(pie_db)
    with SCALERS_PATH.open("rb") as handle:
        historical_scalers = pickle.load(handle)
    scalers = {name: float(historical_scalers[name]) for name in computed_scalers}
    for name, value in computed_scalers.items():
        if not np.isclose(value, scalers[name], rtol=1e-12, atol=1e-12):
            raise ValueError(
                f"XML-reconstructed scaler {name}={value} disagrees with historical {scalers[name]}."
            )
    progress("loading_exp012_trainval_sets_without_decoding_set03_lines")
    exp012_store = Exp012PoseStore.load_allowed_sets(EXP012_POSE_PATH, set(TRAIN_SETS + VAL_SETS))
    cache_root = output_root / "cache" / TRAINING_CACHE
    if cache_root.exists():
        cached_rows, _, _ = load_stream_cache(cache_root, training_phase=True)
        if cached_rows["sequence_id"].astype(str).tolist() != rows["sequence_id"].astype(str).tolist():
            raise ValueError("Existing training cache does not match the frozen train/validation cohort.")
        progress("training_cache status=existing_authenticated")
    else:
        build_stream_cache(
            rows,
            destination=cache_root,
            pie_db=pie_db,
            yolo_store=YoloPoseStore(),
            exp012_store=exp012_store,
            scalers=scalers,
            biu_cache_path=output_root / "cache" / "biu_v2_exp012_30frame.npy",
            training_phase=True,
            progress=progress,
        )
    cache_rows, _, cache_metadata = load_stream_cache(cache_root, training_phase=True)
    if cache_metadata["splits"] != {"train": 109_934, "val": 77_288}:
        raise ValueError(f"Training cache split counts drifted: {cache_metadata['splits']}")
    scalers_path = output_root / "metrics" / "training_scalers.json"
    scalers_path.write_text(json.dumps(scalers, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    source_hashes_path = output_root / "metrics" / "pretraining_source_hashes.json"
    source_paths = [
        EXP012_POSE_PATH,
        EXP012_MIRROR_PATH,
        EXP032_PROTOCOL_ROWS,
        EXP032_RUNNER,
        TRAIN_INDEX_PATH,
        PIE_PARSER_PATH,
        SCALERS_PATH,
        *_pie_xml_paths(TRAIN_SETS + VAL_SETS),
        *_yolo_pose_paths(TRAIN_SETS + VAL_SETS),
    ]
    source_hashes_path.write_text(
        json.dumps(
            {str(path.resolve()): sha256_file(path) for path in source_paths},
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    cache_files = sorted(path for path in cache_root.iterdir() if path.is_file())
    bound_paths = [
        Path(__file__),
        CORE_PATH,
        output_root / ".pose_audit.complete.json",
        manifest_path,
        output_root / "cache" / "biu_v2_exp012_30frame.npy",
        scalers_path,
        source_hashes_path,
        EXP032_PROTOCOL_ROWS,
        EXP032_RUNNER,
        TRAIN_INDEX_PATH,
        PIE_PARSER_PATH,
        SCALERS_PATH,
        *cache_files,
    ]
    payload = {
        "phase": "audit",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": protocol_sha256(),
        "source_manifest_sha256": source_manifest_sha256(),
        "pose_audit_sha256": sha256_file(output_root / ".pose_audit.complete.json"),
        "pose_audit_artifacts": pose_payload["artifacts"],
        "train_rows": 109_934,
        "validation_rows": 77_288,
        "set03_training_access": False,
        "pie_source_mode": "set_specific_xml_only",
        "artifacts": {str(path.resolve()): sha256_file(path) for path in bound_paths},
    }
    write_phase_record(marker, payload)
    authenticate_phase_record(marker, expected_phase="audit")
    progress(f"phase=pretraining_audit status=complete marker={marker}")


def _training_marker(output_root: Path, variant: str, seed: int) -> Path:
    return output_root / f".train_{variant}_seed{seed}.complete.json"


def _authenticate_completed_run(output_root: Path, variant: str, seed: int) -> dict[str, object]:
    payload = authenticate_phase_record(_training_marker(output_root, variant, seed), expected_phase="train")
    if payload.get("variant") != variant or int(payload.get("seed", -1)) != seed:
        raise ValueError("Training completion identity drifted.")
    if payload.get("audit_sha256") != sha256_file(output_root / ".audit.complete.json"):
        raise ValueError("Training completion is bound to a different audit authorization.")
    return payload


def run_training(output_root: Path, variant: str, seed: int, resume: bool) -> None:
    output_root = Path(output_root)
    audit_marker = output_root / ".audit.complete.json"
    authenticate_phase_record(audit_marker, expected_phase="audit")
    marker = _training_marker(output_root, variant, seed)
    if marker.exists():
        _authenticate_completed_run(output_root, variant, seed)
        _progress_logger(output_root / "logs" / f"train_{variant}_seed{seed}.log")(
            "status=already_complete_authenticated"
        )
        return
    progress = _progress_logger(output_root / "logs" / f"train_{variant}_seed{seed}.log")
    progress(f"phase=train status=starting variant={variant} seed={seed}")
    rows, arrays, cache_metadata = load_stream_cache(
        output_root / "cache" / TRAINING_CACHE,
        training_phase=True,
    )
    train_rows, val_rows = training_rows_only(rows)
    train_dataset = Exp047CachedDataset(train_rows, arrays, variant=variant, training_phase=True)
    val_dataset = Exp047CachedDataset(val_rows, arrays, variant=variant, training_phase=True)
    generator = torch.Generator().manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(
        train_dataset,
        batch_size=FrozenProtocol().batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=FrozenProtocol().batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    model = build_frozen_model(variant, seed)
    state_path = output_root / "checkpoints" / f"{variant}_seed{seed}.state.pt"
    if state_path.exists() and not resume:
        raise FileExistsError(f"Resumable state exists; rerun with --resume: {state_path}")
    started = time.time()
    result = fit_frozen_epochs(
        model,
        train_loader=train_loader,
        val_loader=val_loader,
        variant=variant,
        seed=seed,
        device=device,
        state_path=state_path,
        data_generator=generator,
        progress=progress,
    )
    validation_metrics = compute_binary_metrics(result.validation_labels, result.validation_probabilities)
    checkpoint_path = output_root / "checkpoints" / f"{variant}_seed{seed}.best.pt"
    _atomic_torch_save(
        {
            "variant": variant,
            "seed": seed,
            "best_epoch": result.best_epoch,
            "best_validation_f1": result.best_validation_f1,
            "model_state": result.model.state_dict(),
            "protocol_sha256": protocol_sha256(),
            "source_manifest_sha256": source_manifest_sha256(),
            "audit_sha256": sha256_file(audit_marker),
            "total_parameters": count_parameters(result.model)[0],
            "pose_pathway_parameters": count_pose_path_parameters(result.model),
        },
        checkpoint_path,
    )
    history_path = output_root / "logs" / f"training_history_{variant}_seed{seed}.csv"
    _atomic_csv(result.history, history_path)
    predictions_path = output_root / "predictions" / f"validation_{variant}_seed{seed}.csv"
    prediction = val_rows[[
        "sequence_id", "set_id", "video_id", "ped_id", "start_frame", "end_frame", "target_frame", "label"
    ]].copy()
    prediction["variant"] = variant
    prediction["seed"] = seed
    prediction["probability"] = result.validation_probabilities
    prediction["prediction"] = (result.validation_probabilities >= 0.5).astype(np.int64)
    prediction["correct"] = prediction["prediction"].to_numpy() == prediction["label"].to_numpy()
    _atomic_csv(prediction, predictions_path)
    payload = {
        "phase": "train",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": protocol_sha256(),
        "source_manifest_sha256": source_manifest_sha256(),
        "audit_sha256": sha256_file(audit_marker),
        "variant": variant,
        "seed": seed,
        "best_epoch": result.best_epoch,
        "best_validation_f1": result.best_validation_f1,
        "validation_metrics": validation_metrics,
        "device": str(device),
        "wall_time_seconds": float(time.time() - started),
        "cache_metadata_sha256": sha256_file(output_root / "cache" / TRAINING_CACHE / "metadata.json"),
        "artifacts": {
            str(path.resolve()): sha256_file(path)
            for path in (checkpoint_path, history_path, predictions_path, Path(__file__), CORE_PATH, audit_marker)
        },
    }
    write_phase_record(marker, payload)
    _authenticate_completed_run(output_root, variant, seed)
    progress(
        f"phase=train status=complete variant={variant} seed={seed} "
        f"best_epoch={result.best_epoch} val_f1={result.best_validation_f1:.6f} marker={marker}"
    )


def verify_training_grid(output_root: Path) -> list[dict[str, object]]:
    authenticate_phase_record(Path(output_root) / ".audit.complete.json", expected_phase="audit")
    records = []
    for variant in VARIANTS:
        for seed in SEEDS:
            payload = _authenticate_completed_run(output_root, variant, seed)
            checkpoint = output_root / "checkpoints" / f"{variant}_seed{seed}.best.pt"
            records.append(
                {
                    **payload,
                    "checkpoint_path": str(checkpoint.resolve()),
                    "checkpoint_sha256": sha256_file(checkpoint),
                }
            )
    return records


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _measure_latency(model: torch.nn.Module, inputs: dict[str, torch.Tensor], device: torch.device) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        for _ in range(2):
            model(inputs)
        _synchronize(device)
        started = time.perf_counter()
        for _ in range(5):
            model(inputs)
        _synchronize(device)
    predictor_seconds = (time.perf_counter() - started) / 5
    with torch.no_grad():
        if hasattr(model, "biu_projection"):
            operation = lambda: model.biu_projection(inputs["biu_v2"])
        else:
            def operation():
                sequence, _ = model.lstms["pose"](inputs["pose"])
                return model.attentions["pose"](sequence)[0]
        for _ in range(2):
            operation()
        _synchronize(device)
        started = time.perf_counter()
        for _ in range(5):
            operation()
        _synchronize(device)
    pose_seconds = (time.perf_counter() - started) / 5
    batch_size = int(next(iter(inputs.values())).shape[0])
    return {
        "batch_size": batch_size,
        "predictor_seconds_per_batch": predictor_seconds,
        "predictor_seconds_per_sample": predictor_seconds / batch_size,
        "pose_pathway_seconds_per_batch": pose_seconds,
        "pose_pathway_seconds_per_sample": pose_seconds / batch_size,
    }


def _prediction_frame(rows: pd.DataFrame, probabilities: np.ndarray, variant: str, seed: int) -> pd.DataFrame:
    if len(rows) != len(probabilities):
        raise ValueError("Prediction population does not align with the set03 cohort.")
    frame = rows[[
        "sequence_id", "set_id", "video_id", "ped_id", "start_frame", "end_frame", "target_frame", "label"
    ]].copy().reset_index(drop=True)
    frame["variant"] = variant
    frame["seed"] = seed
    frame["threshold"] = 0.5
    frame["probability"] = np.asarray(probabilities, dtype=np.float64)
    frame["prediction"] = (frame["probability"] >= 0.5).astype(np.int64)
    frame["correct"] = frame["prediction"].to_numpy() == frame["label"].to_numpy()
    return frame


def _measure_biu_extraction_latency(rows: pd.DataFrame, arrays: dict[str, np.ndarray]) -> dict[str, object]:
    sample_count = min(100, len(rows))
    started = time.perf_counter()
    for index in range(sample_count):
        row = rows.iloc[index]
        width = int(row.frame_width)
        height = int(row.frame_height)
        pose = np.asarray(arrays["exp012_pose"][index]).reshape(30, 17, 2).astype(np.float64)
        pose *= np.asarray([width, height], dtype=np.float64).reshape(1, 1, 2)
        normalized_bbox = np.asarray(arrays["bbox"][index], dtype=np.float64)
        cx = normalized_bbox[:, 0] * width
        cy = normalized_bbox[:, 1] * height
        bw = normalized_bbox[:, 2] * width
        bh = normalized_bbox[:, 3] * height
        bbox = np.column_stack((cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2))
        extract_frozen_biu(pose, bbox, (width, height))
    elapsed = time.perf_counter() - started
    return {
        "component": "biu_v2_extraction_from_cached_exp012_points",
        "sample_count": sample_count,
        "seconds_total": elapsed,
        "seconds_per_sample": elapsed / max(sample_count, 1),
        "pose_estimator_runtime_included": False,
    }


def run_final_evaluation(output_root: Path, resume: bool) -> None:
    output_root = Path(output_root)
    progress = _progress_logger(output_root / "logs" / "final_evaluation.log")
    audit_marker = output_root / ".audit.complete.json"
    authenticate_phase_record(audit_marker, expected_phase="audit")
    # This exact nine-run authentication must finish before any set03 row or cache is loaded.
    training_records = authenticate_training_records(verify_training_grid(output_root))
    marker = output_root / ".evaluate.complete.json"
    if marker.exists():
        if not resume:
            raise FileExistsError(f"Final evaluation already exists: {marker}")
        authenticate_phase_record(marker, expected_phase="evaluate")
        progress("phase=evaluate status=already_complete_authenticated")
        return
    temporary = output_root / ".evaluation_package.tmp"
    if temporary.exists():
        if not resume:
            raise FileExistsError(f"Unpublished evaluation transaction exists: {temporary}; use --resume.")
        shutil.rmtree(temporary)
    (temporary / "metrics").mkdir(parents=True)
    (temporary / "predictions").mkdir(parents=True)
    progress("phase=evaluate status=authorized checkpoints=9 set03_access=starting")
    full_manifest = pd.read_csv(output_root / "metrics" / "cohort_manifest.csv")
    test_rows = full_manifest.loc[full_manifest["split"] == "test"].reset_index(drop=True)
    verify_protocol_rows(test_rows, expected_counts={"test": 271_078})
    test_rows["cache_index"] = np.arange(len(test_rows), dtype=np.int64)
    test_cache = output_root / "cache" / "set03_streams"
    if not test_cache.exists():
        pie_db = load_pie_sets_from_xml(TEST_SETS, training_phase=False)
        exp012_store = Exp012PoseStore.load_allowed_sets(EXP012_POSE_PATH, {"set03"})
        scalers = json.loads((output_root / "metrics" / "training_scalers.json").read_text(encoding="utf-8"))
        build_stream_cache(
            test_rows,
            destination=test_cache,
            pie_db=pie_db,
            yolo_store=YoloPoseStore(),
            exp012_store=exp012_store,
            scalers=scalers,
            biu_cache_path=output_root / "cache" / "biu_v2_exp012_30frame.npy",
            training_phase=False,
            progress=progress,
        )
    cached_rows, arrays, _ = load_stream_cache(test_cache, training_phase=False)
    if cached_rows["sequence_id"].astype(str).tolist() != test_rows["sequence_id"].astype(str).tolist():
        raise ValueError("set03 cache row order drifted from the frozen test manifest.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result_rows: list[dict[str, object]] = []
    common_rows: list[dict[str, object]] = []
    parameter_rows: list[dict[str, object]] = []
    latency_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    common_mask = np.asarray(arrays["yolo_pose_valid"], dtype=bool) & np.asarray(
        arrays["exp012_pose_valid"], dtype=bool
    )
    for record in training_records:
        variant = str(record["variant"])
        seed = int(record["seed"])
        checkpoint_path = Path(str(record["checkpoint_path"]))
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("variant") != variant or int(checkpoint.get("seed", -1)) != seed:
            raise ValueError(f"Best checkpoint identity mismatch: {checkpoint_path}")
        if checkpoint.get("audit_sha256") != sha256_file(audit_marker):
            raise ValueError(f"Best checkpoint audit binding mismatch: {checkpoint_path}")
        model = build_frozen_model(variant, seed).to(device)
        model.load_state_dict(checkpoint["model_state"])
        dataset = Exp047CachedDataset(test_rows, arrays, variant=variant, training_phase=False)
        loader = DataLoader(
            dataset,
            batch_size=FrozenProtocol().batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )
        labels, probabilities = _predict_loader(model, loader, device)
        if not np.array_equal(labels.astype(np.int64), test_rows["label"].to_numpy(dtype=np.int64)):
            raise ValueError(f"set03 labels drifted during evaluation for {variant}{seed}.")
        metrics = compute_binary_metrics(labels, probabilities)
        result_rows.append({"variant": variant, "seed": seed, **metrics})
        common_metrics = compute_binary_metrics(labels[common_mask], probabilities[common_mask])
        common_rows.append(
            {
                "variant": variant,
                "seed": seed,
                "n_common_valid": int(common_mask.sum()),
                "common_valid_rate": float(common_mask.mean()),
                **common_metrics,
            }
        )
        prediction = _prediction_frame(test_rows, probabilities, variant, seed)
        prediction_frames.append(prediction[["sequence_id", "variant", "seed", "label", "prediction", "probability"]])
        _atomic_csv(prediction, temporary / "predictions" / f"set03_{variant}_seed{seed}.csv")
        total, trainable = count_parameters(model)
        parameter_rows.append(
            {
                "variant": variant,
                "seed": seed,
                "total_parameters": total,
                "trainable_parameters": trainable,
                "pose_pathway_parameters": count_pose_path_parameters(model),
            }
        )
        first_inputs, _, _ = next(iter(loader))
        device_inputs = {name: tensor.to(device) for name, tensor in first_inputs.items()}
        latency_rows.append({"variant": variant, "seed": seed, "device": str(device), **_measure_latency(model, device_inputs, device)})
        progress(f"set03_prediction status=complete variant={variant} seed={seed}")
    results = pd.DataFrame(result_rows).sort_values(["variant", "seed"]).reset_index(drop=True)
    summary = summarize_per_seed_results(results)
    paired = build_paired_deltas(results)
    predictions_minimal = pd.concat(prediction_frames, ignore_index=True)
    transitions = build_error_transition_counts(predictions_minimal)
    outcome = classify_final_outcome(results)
    validation_rows = []
    for record in training_records:
        validation_rows.append(
            {
                "variant": record["variant"],
                "seed": record["seed"],
                "best_epoch": record["best_epoch"],
                **{f"validation_{name}": value for name, value in record["validation_metrics"].items()},
            }
        )
    efficiency = pd.DataFrame(latency_rows)
    biu_latency = _measure_biu_extraction_latency(test_rows, arrays)
    output_frames = {
        "per_seed_results.csv": results,
        "fixed_0p5_results.csv": results,
        "published_protocol_results.csv": results,
        "summary_results.csv": summary,
        "paired_deltas.csv": paired,
        "common_valid_pose_analysis.csv": pd.DataFrame(common_rows),
        "error_transitions_fixed_0p5.csv": transitions,
        "parameter_counts.csv": pd.DataFrame(parameter_rows),
        "latency_summary.csv": efficiency,
        "val_f1_results.csv": pd.DataFrame(validation_rows),
    }
    for name, frame in output_frames.items():
        _atomic_csv(frame, temporary / "metrics" / name)
    evaluation_summary = {
        "outcome_category": outcome,
        "device": str(device),
        "set03_rows": int(len(test_rows)),
        "common_valid_rows": int(common_mask.sum()),
        "biu_extraction_latency": biu_latency,
        "yolo_pose_estimator_runtime": "unavailable_historical_cache",
        "vitpose_s_estimator_runtime": "unavailable_historical_cache",
        "noninferiority_tolerance": 0.005,
    }
    (temporary / "metrics" / "evaluation_summary.json").write_text(
        json.dumps(evaluation_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    produced = sorted(path for path in temporary.rglob("*") if path.is_file())
    for source in produced:
        relative = source.relative_to(temporary)
        destination = output_root / relative
        if destination.exists():
            raise FileExistsError(f"Final evaluation refuses to overwrite: {destination}")
    published = []
    for source in produced:
        relative = source.relative_to(temporary)
        destination = output_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        source.replace(destination)
        published.append(destination)
    shutil.rmtree(temporary)
    payload = {
        "phase": "evaluate",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": protocol_sha256(),
        "source_manifest_sha256": source_manifest_sha256(),
        "audit_sha256": sha256_file(audit_marker),
        "training_markers": {
            f"{variant}{seed}": sha256_file(_training_marker(output_root, variant, seed))
            for variant in VARIANTS for seed in SEEDS
        },
        "outcome_category": outcome,
        "artifacts": {
            str(path.resolve()): sha256_file(path)
            for path in [
                *published,
                *sorted(path for path in test_cache.iterdir() if path.is_file()),
                *_pie_xml_paths(TEST_SETS),
                *_yolo_pose_paths(TEST_SETS),
                Path(__file__),
                CORE_PATH,
                audit_marker,
            ]
        },
    }
    write_phase_record(marker, payload)
    authenticate_phase_record(marker, expected_phase="evaluate")
    progress(f"phase=evaluate status=complete outcome={outcome} marker={marker}")


def run_finalize(output_root: Path) -> None:
    output_root = Path(output_root)
    evaluate_marker = output_root / ".evaluate.complete.json"
    evaluation = authenticate_phase_record(evaluate_marker, expected_phase="evaluate")
    required_metrics = (
        "per_seed_results.csv",
        "summary_results.csv",
        "paired_deltas.csv",
        "fixed_0p5_results.csv",
        "common_valid_pose_analysis.csv",
        "error_transitions_fixed_0p5.csv",
        "parameter_counts.csv",
        "latency_summary.csv",
        "evaluation_summary.json",
    )
    paths = [output_root / "metrics" / name for name in required_metrics]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Cannot finalize; required outputs are absent: {missing}")
    results = pd.read_csv(output_root / "metrics" / "per_seed_results.csv")
    summary = pd.read_csv(output_root / "metrics" / "summary_results.csv")
    paired = pd.read_csv(output_root / "metrics" / "paired_deltas.csv")
    _ = summarize_per_seed_results(results)
    _ = build_paired_deltas(results)
    lines = [
        "# Exp047 Results",
        "",
        f"Final outcome category: `{evaluation['outcome_category']}`.",
        "",
        "The frozen task uses 30 observed frames and predicts the next raw frame. Checkpoints were selected by validation F1 at fixed 0.5 after all five epochs; set03 was evaluated once after all nine checkpoints authenticated.",
        "",
        "## Mean Results",
        "",
        "```csv",
        summary.to_csv(index=False).strip(),
        "```",
        "",
        "## Paired Deltas",
        "",
        "```csv",
        paired.to_csv(index=False).strip(),
        "```",
        "",
        "A->B is the pose-source effect, B->C is the BIU representation/replacement effect using identical Exp012 pose measurements, and A->C is the overall pose-pipeline effect.",
    ]
    results_path = output_root / "RESULTS.md"
    if results_path.exists():
        raise FileExistsError(f"Results report already exists: {results_path}")
    results_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    completion = output_root / "EXPERIMENT_COMPLETE"
    payload = {
        "phase": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": protocol_sha256(),
        "source_manifest_sha256": source_manifest_sha256(),
        "evaluate_sha256": sha256_file(evaluate_marker),
        "outcome_category": evaluation["outcome_category"],
        "artifacts": {
            str(path.resolve()): sha256_file(path)
            for path in [evaluate_marker, results_path, *paths]
        },
    }
    write_phase_record(completion, payload)
    authenticate_phase_record(completion, expected_phase="complete")
    _progress_logger(output_root / "logs" / "finalize.log")(
        f"phase=finalize status=complete marker={completion}"
    )


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "audit":
        if args.limit is not None:
            run_pose_audit(args.output_root, args.limit)
        else:
            run_pretraining_audit(args.output_root, args.resume)
        return
    if args.command == "train":
        run_training(args.output_root, args.variant, args.seed, args.resume)
        return
    if args.command == "train-all":
        progress = _progress_logger(Path(args.output_root) / "logs" / "train_all.log")
        progress("phase=train_all status=starting")
        for variant in VARIANTS:
            for seed in SEEDS:
                run_training(args.output_root, variant, seed, args.resume)
        verify_training_grid(args.output_root)
        progress("phase=train_all status=complete runs=9")
        return
    if args.command == "evaluate":
        run_final_evaluation(args.output_root, args.resume)
        return
    if args.command == "finalize":
        run_finalize(args.output_root)
        return
    if args.command == "verify":
        if args.verify_phase == "audit":
            authenticate_phase_record(Path(args.output_root) / ".audit.complete.json", expected_phase="audit")
        elif args.verify_phase == "train":
            verify_training_grid(args.output_root)
        elif args.verify_phase == "evaluate":
            authenticate_phase_record(Path(args.output_root) / ".evaluate.complete.json", expected_phase="evaluate")
        elif args.verify_phase == "complete":
            authenticate_phase_record(Path(args.output_root) / ".evaluate.complete.json", expected_phase="evaluate")
            authenticate_phase_record(Path(args.output_root) / "EXPERIMENT_COMPLETE", expected_phase="complete")
        else:
            raise SystemExit(f"Verification phase {args.verify_phase!r} is not implemented yet.")
        return
    raise SystemExit(f"Exp047 phase '{args.command}' is not implemented yet; no scientific work was run.")


if __name__ == "__main__":
    main()
