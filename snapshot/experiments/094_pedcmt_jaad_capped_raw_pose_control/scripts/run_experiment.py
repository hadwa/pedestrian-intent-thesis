#!/usr/bin/env python3
"""Run the frozen Exp094 JAAD capped raw-pose control."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
REPO_ROOT = ROOT.parents[1]
EXP089 = REPO_ROOT / "experiments/089_pedcmt_standardized_jaad_biu_transfer"
EXP090 = REPO_ROOT / "experiments/090_pedcmt_direct_biu_fusion"
EXP092 = REPO_ROOT / "experiments/092_pedcmt_capped_biu_correction"
CHECKPOINTS = ROOT / "checkpoints"
LOGS = ROOT / "logs"
METRICS = ROOT / "metrics"
PREDICTIONS = ROOT / "predictions"
SEEDS = (42, 43, 44)
CORRECTION_SCALE = 0.10
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20_260_902
PARENT_REPRODUCTION_TOLERANCE = 1e-6
COPIED_SOURCE_TOLERANCE = 1e-12


def _load_exp090_runner():
    path = EXP090 / "scripts/run_experiment.py"
    spec = importlib.util.spec_from_file_location("exp090_runner_for_exp094", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = _load_exp090_runner()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def natural_training_indices(rows: pd.DataFrame) -> np.ndarray:
    indices = np.flatnonzero(rows["split"].eq("train").to_numpy()).astype(np.int64)
    if len(indices) != len(np.unique(indices)):
        raise RuntimeError("Natural training indices contain duplicates")
    return indices


def scaled_probability(
    parent_logits: np.ndarray,
    correction: np.ndarray,
    scale: float = CORRECTION_SCALE,
) -> np.ndarray:
    parent = np.asarray(parent_logits, dtype=np.float64).reshape(-1)
    delta = np.asarray(correction, dtype=np.float64).reshape(-1)
    if parent.shape != delta.shape or not np.isfinite(parent).all() or not np.isfinite(delta).all():
        raise ValueError("Parent logits and corrections must be aligned and finite")
    logits = np.clip(parent + float(scale) * delta, -80.0, 80.0)
    return 1.0 / (1.0 + np.exp(-logits))


def probability_to_logits(probability: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(probability, dtype=np.float64).reshape(-1), 1e-8, 1.0 - 1e-8)
    return np.log(values) - np.log1p(-values)


def maximum_cap_probability_difference(
    parent_logits: np.ndarray,
    correction: np.ndarray,
    probability: np.ndarray,
) -> float:
    expected = scaled_probability(parent_logits, correction)
    observed = np.asarray(probability, dtype=np.float64).reshape(-1)
    if expected.shape != observed.shape:
        raise ValueError("Saved capped probabilities are not aligned")
    return float(np.max(np.abs(expected - observed)))


def source_probability_parity(maximum_difference: float, *, mode: str) -> bool:
    """Apply the correct tolerance for recomputed versus directly copied probabilities."""
    if mode == "reproduced":
        tolerance = PARENT_REPRODUCTION_TOLERANCE
    elif mode == "copied":
        tolerance = COPIED_SOURCE_TOLERANCE
    else:
        raise ValueError(f"Unknown source parity mode: {mode}")
    return bool(np.isfinite(maximum_difference) and maximum_difference <= tolerance)


def assert_aligned(reference: pd.DataFrame, candidate: pd.DataFrame) -> None:
    for column in ("sequence_id", "label"):
        if column not in reference or column not in candidate:
            raise RuntimeError(f"Missing alignment column: {column}")
        if not np.array_equal(reference[column].to_numpy(), candidate[column].to_numpy()):
            raise RuntimeError(f"Prediction order or {column} drifted")
    if "track_id" in reference and "track_id" in candidate:
        if not np.array_equal(reference["track_id"].to_numpy(), candidate["track_id"].to_numpy()):
            raise RuntimeError("Prediction order or track_id drifted")


def require_evaluation_ready(root: Path) -> None:
    for marker in ("TRAINING_COMPLETE.json", "VALIDATION_COMPLETE.json"):
        if not (root / marker).is_file():
            raise RuntimeError(f"Missing {marker}; test evaluation is locked")


def development_population_audit(
    rows: pd.DataFrame, raw_pose: np.ndarray
) -> dict[str, object]:
    split_counts = rows["split"].value_counts().to_dict()
    train = rows.loc[rows["split"].eq("train")]
    class_counts = train["label"].value_counts().to_dict()
    if split_counts != {"train": 1552, "val": 274}:
        raise RuntimeError(f"JAAD development population drifted: {split_counts}")
    if class_counts != {1: 999, 0: 553}:
        raise RuntimeError(f"JAAD natural class counts drifted: {class_counts}")
    if raw_pose.shape != (1826, 16, 34) or not np.isfinite(raw_pose).all():
        raise RuntimeError(f"JAAD raw pose drifted: {raw_pose.shape}")
    return {
        "train_windows": 1552,
        "validation_windows": 274,
        "train_class_counts": {"not_cross": 553, "cross": 999},
        "raw_pose_shape": [1826, 16, 34],
        "test_accessed": False,
    }


def _source_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}


def _development_sources() -> list[Path]:
    paths = [
        EXP089 / "cache/trainval_source/rows.csv",
        EXP089 / "cache/trainval/pose_std.npy",
    ]
    for seed in SEEDS:
        paths.extend(
            [
                EXP089 / f"checkpoints/A_seed{seed}.best.pt",
                EXP092 / f"predictions/validation_N10_seed{seed}.csv",
            ]
        )
    return paths


def prepare(device_name: str = "cpu") -> None:
    del device_name  # Preparation reuses authenticated cached development outputs.
    BASE.authenticate_parent_checkpoints()
    rows, arrays = BASE.load_trainval()
    audit = development_population_audit(rows, arrays["raw_pose"])
    parent_cache: dict[str, dict[str, Any]] = {}
    for seed in SEEDS:
        logits, anchor = BASE._prepared_parent(seed)
        if not np.isfinite(logits).all() or not np.isfinite(anchor).all():
            raise RuntimeError(f"Non-finite cached parent output for seed {seed}")
        parent_cache[str(seed)] = {
            "logits_shape": list(logits.shape),
            "anchor_shape": list(anchor.shape),
        }
    audit.update(
        {
            "sources": [_source_record(path) for path in _development_sources()],
            "parent_cache": parent_cache,
            "parent_checkpoint_sha256": BASE.PARENT_CHECKPOINT_SHA256,
            "training_sampler": "natural_each_window_once_per_epoch",
            "correction_scale": CORRECTION_SCALE,
            "test_predictions_hashed": False,
            "test_arrays_loaded": False,
        }
    )
    atomic_json(METRICS / "development_lineage_audit.json", audit)
    atomic_json(ROOT / "PREPARATION_COMPLETE.json", audit)


def _values(
    rows: pd.DataFrame,
    logits: np.ndarray,
    anchor: np.ndarray,
    raw_pose: np.ndarray,
    indices: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        "parent_logits": np.asarray(logits[indices], dtype=np.float32),
        "parent_anchor": np.asarray(anchor[indices], dtype=np.float32),
        "extra": np.asarray(raw_pose[indices], dtype=np.float32),
        "labels": rows.iloc[indices]["label"].to_numpy(dtype=np.float32),
    }


def _checkpoint_ready(seed: int) -> bool:
    return (CHECKPOINTS / f"P_seed{seed}.best.pt").is_file() and (
        LOGS / f"P_seed{seed}.complete.json"
    ).is_file()


def train_one(seed: int, device_name: str, *, smoke: bool = False) -> dict[str, Any]:
    if not (ROOT / "PREPARATION_COMPLETE.json").is_file():
        raise RuntimeError("Run prepare before training")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    rows, arrays = BASE.load_trainval()
    development_population_audit(rows, arrays["raw_pose"])
    logits, anchor = BASE._prepared_parent(seed)
    train_indices = natural_training_indices(rows)
    val_indices = np.flatnonzero(rows["split"].eq("val").to_numpy()).astype(np.int64)
    if len(train_indices) != 1552 or len(np.unique(train_indices)) != 1552:
        raise RuntimeError("Natural training population is not exactly 1,552 unique windows")
    stem = f"{'smoke_' if smoke else ''}P_seed{seed}"
    result = BASE.fit_branch(
        "B",
        _values(rows, logits, anchor, arrays["raw_pose"], train_indices),
        _values(rows, logits, anchor, arrays["raw_pose"], val_indices),
        seed=seed,
        device=device,
        checkpoint_path=CHECKPOINTS / f"{stem}.best.pt",
        history_path=LOGS / f"{stem}.history.csv",
        parent_sha256=BASE.PARENT_CHECKPOINT_SHA256[seed],
        max_epochs=1 if smoke else BASE.MAX_EPOCHS,
        patience=1 if smoke else BASE.PATIENCE,
    )
    full_correction = result.pop("correction")
    result.pop("probability")
    parent_logits = np.asarray(logits[val_indices], dtype=np.float64).reshape(-1)
    probability = scaled_probability(parent_logits, full_correction)
    output = rows.iloc[val_indices][
        ["sequence_id", "track_id", "video_id", "ped_id", "label"]
    ].copy()
    output["parent_probability"] = scaled_probability(parent_logits, np.zeros_like(parent_logits))
    output["full_raw_pose_correction_logit"] = full_correction
    output["scaled_raw_pose_correction_logit"] = CORRECTION_SCALE * full_correction
    output["probability"] = probability
    output["prediction"] = (probability >= 0.5).astype(np.int64)
    PREDICTIONS.mkdir(parents=True, exist_ok=True)
    output.to_csv(PREDICTIONS / f"validation_{stem.replace('P_', 'P10_')}.csv", index=False)
    result.update(
        {
            "sampling": "natural",
            "unique_train_windows": int(len(np.unique(train_indices))),
            "train_windows_per_epoch": int(len(train_indices)),
            "correction_scale_for_saved_validation_prediction": CORRECTION_SCALE,
        }
    )
    atomic_json(LOGS / f"{stem}.complete.json", result)
    return result


def train_all(device_name: str) -> None:
    if not (ROOT / "PREPARATION_COMPLETE.json").is_file():
        prepare("cpu")
    for seed in SEEDS:
        if _checkpoint_ready(seed):
            print(f"Skipping complete P seed {seed}", flush=True)
            continue
        print(json.dumps(train_one(seed, device_name), indent=2), flush=True)
    missing = [seed for seed in SEEDS if not _checkpoint_ready(seed)]
    if missing:
        raise RuntimeError(f"Missing completed P checkpoints: {missing}")
    checkpoints = {str(seed): sha256(CHECKPOINTS / f"P_seed{seed}.best.pt") for seed in SEEDS}
    atomic_json(
        ROOT / "TRAINING_COMPLETE.json",
        {
            "seeds": list(SEEDS),
            "variant": "P_natural_raw_pose_residual",
            "checkpoints_sha256": checkpoints,
            "test_accessed": False,
        },
    )


def smoke(device_name: str) -> None:
    if not (ROOT / "PREPARATION_COMPLETE.json").is_file():
        prepare("cpu")
    result = train_one(42, device_name, smoke=True)
    atomic_json(LOGS / "SMOKE_COMPLETE.json", {"device": device_name, "result": result})


def build_aligned_three_arm(
    base: pd.DataFrame,
    pose: pd.DataFrame,
    union: pd.DataFrame,
    *,
    seed: int,
) -> pd.DataFrame:
    assert_aligned(base, pose)
    assert_aligned(base, union)
    columns = [column for column in ("sequence_id", "track_id", "video_id", "ped_id", "label") if column in base]
    output = base[columns].copy()
    output.insert(0, "seed", seed)
    output["A_probability"] = base["probability"].to_numpy(dtype=np.float64)
    output["P10_probability"] = pose["probability"].to_numpy(dtype=np.float64)
    output["U10_probability"] = union["probability"].to_numpy(dtype=np.float64)
    return output


def _metric_rows(aligned: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for seed, frame in aligned.groupby("seed", sort=True):
        labels = frame["label"].to_numpy(dtype=np.int64)
        for arm in ("A", "P10", "U10"):
            rows.append(
                {
                    "seed": int(seed),
                    "variant": arm,
                    **BASE.fixed_metrics(labels, frame[f"{arm}_probability"].to_numpy()),
                }
            )
    return pd.DataFrame(rows)


def paired_deltas(aligned: pd.DataFrame) -> pd.DataFrame:
    comparisons = (
        ("U10_minus_A", "A", "U10"),
        ("P10_minus_A", "A", "P10"),
        ("U10_minus_P10", "P10", "U10"),
    )
    rows: list[dict[str, Any]] = []
    for seed, frame in aligned.groupby("seed", sort=True):
        labels = frame["label"].to_numpy(dtype=np.int64)
        metrics = {
            arm: BASE.fixed_metrics(labels, frame[f"{arm}_probability"].to_numpy())
            for arm in ("A", "P10", "U10")
        }
        for name, reference, candidate in comparisons:
            rows.append(
                {
                    "comparison": name,
                    "seed": int(seed),
                    **{
                        f"delta_{metric}": float(metrics[candidate][metric] - metrics[reference][metric])
                        for metric in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc")
                    },
                }
            )
    return pd.DataFrame(rows)


def _summary(metrics: pd.DataFrame) -> pd.DataFrame:
    selected = [
        "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc",
        "tn", "fp", "fn", "tp", "high_conf_fp", "high_conf_fn",
    ]
    result = metrics.groupby("variant")[selected].agg(["mean", "std"])
    result.columns = [f"{metric}_{stat}" for metric, stat in result.columns]
    return result.reset_index()


def _error_transitions(aligned: pd.DataFrame) -> pd.DataFrame:
    comparisons = (
        ("U10_minus_A", "A", "U10"),
        ("P10_minus_A", "A", "P10"),
        ("U10_minus_P10", "P10", "U10"),
    )
    rows: list[dict[str, Any]] = []
    for seed, frame in aligned.groupby("seed", sort=True):
        labels = frame["label"].to_numpy(dtype=np.int64)
        for name, reference, candidate in comparisons:
            before = frame[f"{reference}_probability"].to_numpy() >= 0.5
            after = frame[f"{candidate}_probability"].to_numpy() >= 0.5
            before_correct = before == labels
            after_correct = after == labels
            rows.append(
                {
                    "comparison": name,
                    "seed": int(seed),
                    "wrong_to_correct": int(np.sum(~before_correct & after_correct)),
                    "correct_to_wrong": int(np.sum(before_correct & ~after_correct)),
                    "fp_to_tn": int(np.sum((labels == 0) & before & ~after)),
                    "fn_to_tp": int(np.sum((labels == 1) & ~before & after)),
                    "tn_to_fp": int(np.sum((labels == 0) & ~before & after)),
                    "tp_to_fn": int(np.sum((labels == 1) & before & ~after)),
                }
            )
    return pd.DataFrame(rows)


def validate() -> None:
    missing = [seed for seed in SEEDS if not _checkpoint_ready(seed)]
    if missing or not (ROOT / "TRAINING_COMPLETE.json").is_file():
        raise RuntimeError(f"Validation requires TRAINING_COMPLETE and all checkpoints; missing {missing}")
    rows, _ = BASE.load_trainval()
    expected = rows.loc[rows["split"].eq("val")].reset_index(drop=True)
    aligned_frames: list[pd.DataFrame] = []
    for seed in SEEDS:
        pose = pd.read_csv(PREDICTIONS / f"validation_P10_seed{seed}.csv")
        union = pd.read_csv(EXP092 / f"predictions/validation_N10_seed{seed}.csv")
        assert_aligned(expected, pose)
        assert_aligned(expected, union)
        parent_difference = float(
            np.max(np.abs(pose["parent_probability"] - union["parent_probability"]))
        )
        if not source_probability_parity(parent_difference, mode="reproduced"):
            raise RuntimeError(f"Validation parent parity failed for seed {seed}: {parent_difference}")
        base = expected[["sequence_id", "track_id", "video_id", "ped_id", "label"]].copy()
        base["probability"] = pose["parent_probability"].to_numpy()
        aligned_frames.append(build_aligned_three_arm(base, pose, union, seed=seed))
    aligned = pd.concat(aligned_frames, ignore_index=True)
    metrics = _metric_rows(aligned)
    METRICS.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(METRICS / "validation_per_seed_results.csv", index=False)
    _summary(metrics).to_csv(METRICS / "validation_summary_results.csv", index=False)
    paired_deltas(aligned).to_csv(METRICS / "validation_paired_deltas.csv", index=False)
    atomic_json(
        ROOT / "VALIDATION_COMPLETE.json",
        {
            "seeds": list(SEEDS),
            "arms": ["A", "P10", "U10"],
            "correction_scale": CORRECTION_SCALE,
            "selection_performed": False,
            "performance_gate": False,
            "test_accessed": False,
        },
    )


def _load_pose_branch(seed: int, device: torch.device):
    checkpoint = torch.load(
        CHECKPOINTS / f"P_seed{seed}.best.pt", map_location=device, weights_only=False
    )
    if checkpoint.get("parent_sha256") != BASE.PARENT_CHECKPOINT_SHA256[seed]:
        raise RuntimeError(f"P seed {seed} is not bound to the authenticated parent")
    model = BASE.DirectFusionModel("B").to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def _load_union_branch(seed: int, device: torch.device):
    path = REPO_ROOT / f"experiments/091_pedcmt_natural_sampling_calibration/checkpoints/N_seed{seed}.best.pt"
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("parent_sha256") != BASE.PARENT_CHECKPOINT_SHA256[seed]:
        raise RuntimeError(f"U seed {seed} is not bound to the authenticated parent")
    model = BASE.DirectFusionModel("C").to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def _benchmark_branch(
    model: torch.nn.Module,
    parent_logits: np.ndarray,
    parent_anchor: np.ndarray,
    extra: np.ndarray,
    device: torch.device,
    *,
    arm: str,
    seed: int,
    checkpoint_path: Path,
    training_seconds: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for batch_size, warmups, repetitions in ((1, 100, 1000), (256, 20, 200)):
        logits = torch.from_numpy(np.asarray(parent_logits[:batch_size], dtype=np.float32)).to(device)
        anchor = torch.from_numpy(np.asarray(parent_anchor[:batch_size], dtype=np.float32)).to(device)
        inputs = torch.from_numpy(np.asarray(extra[:batch_size], dtype=np.float32)).to(device)
        model.eval()
        with torch.inference_mode():
            for _ in range(warmups):
                model(logits, anchor, inputs)
            if device.type == "cuda":
                torch.cuda.synchronize()
            samples: list[float] = []
            for _ in range(repetitions):
                started = time.perf_counter()
                model(logits, anchor, inputs)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                samples.append((time.perf_counter() - started) * 1000.0)
        values = np.asarray(samples, dtype=np.float64)
        rows.append(
            {
                "arm": arm,
                "seed": seed,
                "trainable_parameters": BASE.count_trainable_parameters(model),
                "checkpoint_size_bytes": checkpoint_path.stat().st_size,
                "training_seconds": training_seconds,
                "timing_scope": "correction_branch_from_cached_parent_anchor_and_representation",
                "device": str(device),
                "batch_size": batch_size,
                "warmups": warmups,
                "repetitions": repetitions,
                "latency_mean_ms": float(values.mean()),
                "latency_median_ms": float(np.median(values)),
                "latency_p95_ms": float(np.percentile(values, 95)),
                "per_sample_mean_ms": float(values.mean() / batch_size),
            }
        )
    return rows


def evaluate(device_name: str) -> None:
    require_evaluation_ready(ROOT)
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    rows, arrays = BASE.load_test()
    labels = rows["label"].to_numpy(dtype=np.int64)
    aligned_frames: list[pd.DataFrame] = []
    evaluation_sources: list[dict[str, Any]] = []
    efficiency_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        parent = BASE.load_parent_model(seed, device)
        parent_logits, parent_anchor = BASE.extract_parent_outputs(
            parent, arrays["bbox"], arrays["ego"], device
        )
        parent_logits_flat = parent_logits.reshape(-1)
        parent_probability = scaled_probability(
            parent_logits_flat, np.zeros_like(parent_logits_flat)
        )
        saved_parent = pd.read_csv(BASE.PARENT_PREDICTIONS / f"test_A_seed{seed}.csv")
        assert_aligned(rows, saved_parent)
        parent_difference = float(
            np.max(np.abs(parent_probability - saved_parent["probability"].to_numpy()))
        )
        if parent_difference > 1e-6:
            raise RuntimeError(f"Final parent reproduction failed for seed {seed}: {parent_difference}")
        branch = _load_pose_branch(seed, device)
        values = {
            "parent_logits": parent_logits.astype(np.float32),
            "parent_anchor": parent_anchor.astype(np.float32),
            "extra": np.asarray(arrays["raw_pose"], dtype=np.float32),
            "labels": labels.astype(np.float32),
        }
        _, _, correction = BASE.evaluate_branch(branch, values, device)
        probability = scaled_probability(parent_logits_flat, correction)
        pose = rows[["sequence_id", "track_id", "video_id", "ped_id", "label"]].copy()
        pose["parent_logit"] = parent_logits_flat
        pose["parent_probability"] = parent_probability
        pose["full_raw_pose_correction_logit"] = correction
        pose["scaled_raw_pose_correction_logit"] = CORRECTION_SCALE * correction
        pose["probability"] = probability
        pose["prediction"] = (probability >= 0.5).astype(np.int64)
        PREDICTIONS.mkdir(parents=True, exist_ok=True)
        pose_path = PREDICTIONS / f"test_P10_seed{seed}.csv"
        pose.to_csv(pose_path, index=False)
        union_path = EXP092 / f"predictions/test_N10_seed{seed}.csv"
        union = pd.read_csv(union_path)
        assert_aligned(rows, union)
        union_parent_difference = float(
            np.max(np.abs(parent_probability - union["parent_probability"].to_numpy()))
        )
        if not source_probability_parity(union_parent_difference, mode="reproduced"):
            raise RuntimeError(f"Exp092 U10 parent parity failed for seed {seed}: {union_parent_difference}")
        base = rows[["sequence_id", "track_id", "video_id", "ped_id", "label"]].copy()
        base["probability"] = parent_probability
        aligned = build_aligned_three_arm(base, pose, union, seed=seed)
        aligned.to_csv(PREDICTIONS / f"test_aligned_A_P10_U10_seed{seed}.csv", index=False)
        aligned_frames.append(aligned)
        evaluation_sources.extend(
            [
                _source_record(BASE.PARENT_PREDICTIONS / f"test_A_seed{seed}.csv"),
                _source_record(union_path),
            ]
        )
        if seed == 42:
            p_log = json.loads((LOGS / f"P_seed{seed}.complete.json").read_text(encoding="utf-8"))
            efficiency_rows.extend(
                _benchmark_branch(
                    branch,
                    parent_logits,
                    parent_anchor,
                    arrays["raw_pose"],
                    device,
                    arm="P10",
                    seed=seed,
                    checkpoint_path=CHECKPOINTS / f"P_seed{seed}.best.pt",
                    training_seconds=float(p_log["elapsed_seconds"]),
                )
            )
            union_model = _load_union_branch(seed, device)
            union_checkpoint = REPO_ROOT / f"experiments/091_pedcmt_natural_sampling_calibration/checkpoints/N_seed{seed}.best.pt"
            union_log_path = REPO_ROOT / f"experiments/091_pedcmt_natural_sampling_calibration/logs/N_seed{seed}.complete.json"
            union_log = json.loads(union_log_path.read_text(encoding="utf-8"))
            efficiency_rows.extend(
                _benchmark_branch(
                    union_model,
                    parent_logits,
                    parent_anchor,
                    arrays["biu_union"],
                    device,
                    arm="U10",
                    seed=seed,
                    checkpoint_path=union_checkpoint,
                    training_seconds=float(union_log["elapsed_seconds"]),
                )
            )
    all_aligned = pd.concat(aligned_frames, ignore_index=True)
    metrics = _metric_rows(all_aligned)
    METRICS.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(METRICS / "fixed_0p5_results.csv", index=False)
    metrics.to_csv(METRICS / "per_seed_results.csv", index=False)
    _summary(metrics).to_csv(METRICS / "summary_results.csv", index=False)
    paired_deltas(all_aligned).to_csv(METRICS / "paired_deltas.csv", index=False)
    _error_transitions(all_aligned).to_csv(
        METRICS / "error_transitions_fixed_0p5.csv", index=False
    )
    pd.DataFrame(efficiency_rows).to_csv(METRICS / "efficiency.csv", index=False)
    test_paths = [
        EXP089 / "cache/test_source/rows.csv",
        EXP089 / "cache/test_source/bbox.npy",
        EXP089 / "cache/test_source/ego_speed.npy",
        EXP089 / "cache/test_source/pose.npy",
    ]
    evaluation_audit = {
        "test_windows": int(len(rows)),
        "test_sources": [_source_record(path) for path in test_paths],
        "source_predictions": evaluation_sources,
        "parent_reproduction_tolerance": PARENT_REPRODUCTION_TOLERANCE,
        "test_access_started_after_training_and_validation_markers": True,
    }
    atomic_json(METRICS / "evaluation_lineage_audit.json", evaluation_audit)
    atomic_json(
        ROOT / "EXPERIMENT_COMPLETE.json",
        {
            "scale": CORRECTION_SCALE,
            "threshold": 0.5,
            "seeds": list(SEEDS),
            "test_windows": int(len(rows)),
            "checkpoint_sha256": {
                str(seed): sha256(CHECKPOINTS / f"P_seed{seed}.best.pt") for seed in SEEDS
            },
            "source_prediction_sha256": {
                Path(item["path"]).name: item["sha256"] for item in evaluation_sources
            },
            "result_reported_regardless_of_direction": True,
            "retrospective_matched_control": True,
        },
    )


def _ensemble_predictions(aligned: pd.DataFrame) -> pd.DataFrame:
    if set(aligned["seed"].unique()) != set(SEEDS):
        raise RuntimeError("Ensemble requires exact seeds 42, 43, and 44")
    counts = aligned.groupby("sequence_id")["seed"].nunique()
    if not counts.eq(len(SEEDS)).all():
        raise RuntimeError("Each sequence must have exactly three seed predictions")
    for column in ("track_id", "label"):
        if aligned.groupby("sequence_id")[column].nunique().max() != 1:
            raise RuntimeError(f"Ensemble {column} differs across seeds")
    return (
        aligned.groupby("sequence_id", sort=False)
        .agg(
            track_id=("track_id", "first"),
            label=("label", "first"),
            A_probability=("A_probability", "mean"),
            P10_probability=("P10_probability", "mean"),
            U10_probability=("U10_probability", "mean"),
        )
        .reset_index()
    )


def _bootstrap_metric(labels: np.ndarray, probability: np.ndarray, metric: str) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    probability = np.asarray(probability, dtype=np.float64)
    if metric == "f1":
        return float(BASE.f1_score(labels, probability >= 0.5, zero_division=0))
    if metric == "roc_auc":
        return float(BASE.roc_auc_score(labels, probability))
    if metric == "pr_auc":
        return float(BASE.average_precision_score(labels, probability))
    raise ValueError(f"Unsupported bootstrap metric: {metric}")


def ensemble_cluster_bootstrap(
    aligned: pd.DataFrame,
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    ensemble = _ensemble_predictions(aligned)
    groups = {
        str(track): indices.to_numpy(dtype=np.int64)
        for track, indices in ensemble.groupby("track_id", sort=False).groups.items()
    }
    track_ids = np.asarray(list(groups), dtype=object)
    if len(track_ids) < 2:
        raise RuntimeError("Track-cluster bootstrap requires at least two tracks")
    labels = ensemble["label"].to_numpy(dtype=np.int64)
    probabilities = {
        arm: ensemble[f"{arm}_probability"].to_numpy(dtype=np.float64)
        for arm in ("A", "P10", "U10")
    }
    comparisons = (
        ("P10_minus_A", "A", "P10"),
        ("U10_minus_A", "A", "U10"),
        ("U10_minus_P10", "P10", "U10"),
    )
    metrics = ("f1", "roc_auc", "pr_auc")
    observed = {
        (name, metric): _bootstrap_metric(labels, probabilities[candidate], metric)
        - _bootstrap_metric(labels, probabilities[reference], metric)
        for name, reference, candidate in comparisons
        for metric in metrics
    }
    samples = {(name, metric): [] for name, _, _ in comparisons for metric in metrics}
    rng = np.random.default_rng(seed)
    for _ in range(replicates):
        selected_tracks = rng.choice(track_ids, size=len(track_ids), replace=True)
        indices = np.concatenate([groups[str(track)] for track in selected_tracks])
        sampled_labels = labels[indices]
        for name, reference, candidate in comparisons:
            for metric in metrics:
                try:
                    delta = _bootstrap_metric(
                        sampled_labels, probabilities[candidate][indices], metric
                    ) - _bootstrap_metric(
                        sampled_labels, probabilities[reference][indices], metric
                    )
                except ValueError:
                    delta = float("nan")
                samples[(name, metric)].append(delta)
    rows: list[dict[str, Any]] = []
    for name, _, _ in comparisons:
        for metric in metrics:
            values = np.asarray(samples[(name, metric)], dtype=np.float64)
            finite = values[np.isfinite(values)]
            if len(finite) == 0:
                raise RuntimeError(f"No finite bootstrap values for {name} {metric}")
            rows.append(
                {
                    "comparison": name,
                    "metric": metric,
                    "observed_ensemble_delta": observed[(name, metric)],
                    "bootstrap_mean": float(finite.mean()),
                    "ci_2p5": float(np.percentile(finite, 2.5)),
                    "ci_97p5": float(np.percentile(finite, 97.5)),
                    "replicates_requested": int(replicates),
                    "replicates_finite": int(len(finite)),
                    "rng_seed": int(seed),
                    "unique_tracks": int(len(track_ids)),
                }
            )
    return pd.DataFrame(rows)


def _load_all_aligned() -> pd.DataFrame:
    frames = []
    for seed in SEEDS:
        path = PREDICTIONS / f"test_aligned_A_P10_U10_seed{seed}.csv"
        if not path.is_file():
            raise RuntimeError(f"Missing aligned test prediction: {path}")
        frame = pd.read_csv(path)
        if not frame["seed"].eq(seed).all() or len(frame) != 1501:
            raise RuntimeError(f"Aligned test prediction drifted for seed {seed}")
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def analyze() -> None:
    if not (ROOT / "EXPERIMENT_COMPLETE.json").is_file():
        raise RuntimeError("Run evaluate before analyze")
    aligned = _load_all_aligned()
    bootstrap = ensemble_cluster_bootstrap(aligned)
    METRICS.mkdir(parents=True, exist_ok=True)
    bootstrap.to_csv(METRICS / "ensemble_cluster_bootstrap.csv", index=False)
    atomic_json(
        METRICS / "analysis_complete.json",
        {
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "cluster": "track_id",
            "ensemble": "mean_probability_across_three_seeds",
        },
    )


def _format_metric(value: float) -> str:
    return f"{value:.4f}"


def report() -> None:
    if not (METRICS / "ensemble_cluster_bootstrap.csv").is_file():
        raise RuntimeError("Run analyze before report")
    summary = pd.read_csv(METRICS / "summary_results.csv").set_index("variant")
    per_seed = pd.read_csv(METRICS / "paired_deltas.csv")
    bootstrap = pd.read_csv(METRICS / "ensemble_cluster_bootstrap.csv")
    transitions = pd.read_csv(METRICS / "error_transitions_fixed_0p5.csv")
    u_f1 = float(summary.loc["U10", "f1_mean"])
    a_f1 = float(summary.loc["A", "f1_mean"])
    p_f1 = float(summary.loc["P10", "f1_mean"])
    criterion = u_f1 > a_f1 and u_f1 > p_f1
    favorable_u_a = int(
        (per_seed.loc[per_seed["comparison"].eq("U10_minus_A"), "delta_f1"] > 0).sum()
    )
    favorable_u_p = int(
        (per_seed.loc[per_seed["comparison"].eq("U10_minus_P10"), "delta_f1"] > 0).sum()
    )
    lines = [
        "# Exp094 Results",
        "",
        "## Status",
        "",
        "Exp094 is a retrospective matched control conceived after the Exp092 JAAD test result was already known. It is not pristine confirmation. The frozen setup was evaluated without a performance gate, and this closes the planned neural-control line regardless of outcome.",
        "",
        "## Mean fixed-threshold results",
        "",
        "| Arm | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ("A", "P10", "U10"):
        row = summary.loc[arm]
        lines.append(
            "| " + arm + " | " + " | ".join(
                _format_metric(float(row[f"{metric}_mean"]))
                for metric in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc")
            ) + " |"
        )
    lines.extend(
        [
            "",
            "## Primary interpretation",
            "",
            (
                "U10 satisfies the prespecified mean-F1 criterion: it exceeds both A and P10."
                if criterion
                else "U10 does not satisfy the prespecified mean-F1 criterion of exceeding both A and P10."
            ),
            "",
            f"U10 improves F1 over A in {favorable_u_a}/3 seeds and over P10 in {favorable_u_p}/3 seeds.",
            "",
            "The three-seed ensemble track-cluster bootstrap gives:",
            "",
            "| Comparison | Metric | Observed delta | 95% interval | Excludes zero |",
            "|---|---|---:|---:|:---:|",
        ]
    )
    for row in bootstrap.itertuples(index=False):
        excludes = bool(row.ci_2p5 > 0 or row.ci_97p5 < 0)
        lines.append(
            f"| {row.comparison} | {row.metric} | {row.observed_ensemble_delta:.4f} | "
            f"[{row.ci_2p5:.4f}, {row.ci_97p5:.4f}] | {'yes' if excludes else 'no'} |"
        )
    lines.extend(["", "## Fixed-0.5 error transitions", "", "| Comparison | Wrong to correct | Correct to wrong | FP to TN | FN to TP | TN to FP | TP to FN |", "|---|---:|---:|---:|---:|---:|---:|"])
    for comparison, frame in transitions.groupby("comparison", sort=False):
        totals = frame[["wrong_to_correct", "correct_to_wrong", "fp_to_tn", "fn_to_tp", "tn_to_fp", "tp_to_fn"]].sum()
        lines.append(
            f"| {comparison} | {int(totals.wrong_to_correct)} | {int(totals.correct_to_wrong)} | "
            f"{int(totals.fp_to_tn)} | {int(totals.fn_to_tp)} | {int(totals.tn_to_fp)} | {int(totals.tp_to_fn)} |"
        )
    lines.extend(
        [
            "",
            "## Limits",
            "",
            "Raw pose and BIUs originate from the same pose measurements; this comparison concerns representation and fusion behavior, not access to different source information. No result here establishes that BIUs universally replace raw pose or that a 0.10 cap is universally optimal. The result is specific to this frozen parent, JAAD cohort, observation/TTE protocol, and fixed-0.5 operating point.",
            "",
        ]
    )
    (ROOT / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def verify() -> dict[str, Any]:
    for marker in (
        "PREPARATION_COMPLETE.json",
        "TRAINING_COMPLETE.json",
        "VALIDATION_COMPLETE.json",
        "EXPERIMENT_COMPLETE.json",
    ):
        if not (ROOT / marker).is_file():
            raise RuntimeError(f"Missing phase marker {marker}")
    marker_paths = [
        ROOT / "PREPARATION_COMPLETE.json",
        ROOT / "TRAINING_COMPLETE.json",
        ROOT / "VALIDATION_COMPLETE.json",
        ROOT / "EXPERIMENT_COMPLETE.json",
    ]
    marker_order = [path.stat().st_mtime_ns for path in marker_paths]
    if marker_order != sorted(marker_order):
        raise RuntimeError("Phase marker modification order is invalid")
    aligned = _load_all_aligned()
    recomputed = _metric_rows(aligned).sort_values(["seed", "variant"]).reset_index(drop=True)
    saved = pd.read_csv(METRICS / "per_seed_results.csv").sort_values(["seed", "variant"]).reset_index(drop=True)
    if not np.array_equal(recomputed[["seed", "variant"]].to_numpy(), saved[["seed", "variant"]].to_numpy()):
        raise RuntimeError("Saved metric rows do not align")
    metric_columns = [
        "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc",
        "tn", "fp", "fn", "tp", "high_conf_fp", "high_conf_fn",
    ]
    maximum_metric_difference = float(
        np.max(
            np.abs(
                recomputed[metric_columns].to_numpy(dtype=np.float64)
                - saved[metric_columns].to_numpy(dtype=np.float64)
            )
        )
    )
    maximum_cap_difference = 0.0
    maximum_a_parity = 0.0
    maximum_u_parity = 0.0
    for seed in SEEDS:
        pose = pd.read_csv(PREDICTIONS / f"test_P10_seed{seed}.csv")
        cap_difference = maximum_cap_probability_difference(
            pose["parent_logit"].to_numpy(),
            pose["full_raw_pose_correction_logit"].to_numpy(),
            pose["probability"].to_numpy(),
        )
        maximum_cap_difference = max(maximum_cap_difference, cap_difference)
        frame = aligned.loc[aligned["seed"].eq(seed)].reset_index(drop=True)
        source_a = pd.read_csv(BASE.PARENT_PREDICTIONS / f"test_A_seed{seed}.csv")
        source_u = pd.read_csv(EXP092 / f"predictions/test_N10_seed{seed}.csv")
        assert_aligned(frame, source_a)
        assert_aligned(frame, source_u)
        maximum_a_parity = max(
            maximum_a_parity,
            float(np.max(np.abs(frame["A_probability"] - source_a["probability"]))),
        )
        maximum_u_parity = max(
            maximum_u_parity,
            float(np.max(np.abs(frame["U10_probability"] - source_u["probability"]))),
        )
    tolerances = {
        "metric": 1e-12,
        "cap_probability": 1e-7,
        "reproduced_parent_probability": PARENT_REPRODUCTION_TOLERANCE,
        "copied_source_probability": COPIED_SOURCE_TOLERANCE,
    }
    checks = {
        "metric_recomputation": maximum_metric_difference <= tolerances["metric"],
        "p10_cap_recomputation": maximum_cap_difference <= tolerances["cap_probability"],
        "a_source_parity": source_probability_parity(maximum_a_parity, mode="reproduced"),
        "u10_source_parity": source_probability_parity(maximum_u_parity, mode="copied"),
        "phase_marker_order": True,
        "all_three_arms_aligned": True,
    }
    result = {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "maximum_absolute_metric_difference": maximum_metric_difference,
        "maximum_cap_probability_difference": maximum_cap_difference,
        "maximum_A_source_probability_difference": maximum_a_parity,
        "maximum_U10_source_probability_difference": maximum_u_parity,
        "tolerances": tolerances,
    }
    atomic_json(METRICS / "verification.json", result)
    if not result["passed"]:
        raise RuntimeError(f"Exp094 verification failed: {result}")
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("prepare", "smoke", "train", "train-all", "validate", "evaluate", "analyze", "report", "verify"),
    )
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.device)
    elif args.command == "smoke":
        smoke(args.device)
    elif args.command == "train":
        if args.seed is None:
            raise SystemExit("train requires --seed")
        print(json.dumps(train_one(args.seed, args.device), indent=2))
    elif args.command == "train-all":
        train_all(args.device)
    elif args.command == "validate":
        validate()
    elif args.command == "evaluate":
        evaluate(args.device)
    elif args.command == "analyze":
        analyze()
    elif args.command == "report":
        report()
    else:
        verify()


if __name__ == "__main__":
    main()
