#!/usr/bin/env python3
"""Run the validation-gated Exp090 direct fusion comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import Tensor, nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
REPO_ROOT = ROOT.parents[1]
PARENT = REPO_ROOT / "experiments/089_pedcmt_standardized_jaad_biu_transfer"
PARENT_MODEL_SCRIPTS = REPO_ROOT / "experiments/084_pedcmt_biu_transfer/scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(PARENT_MODEL_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(PARENT_MODEL_SCRIPTS))

from direct_fusion_model import DirectFusionModel, count_trainable_parameters  # noqa: E402
from pedcmt_model import PedCMTTransfer  # noqa: E402


PARENT_SOURCE = PARENT / "cache/trainval_source"
PARENT_DERIVED = PARENT / "cache/trainval"
PARENT_CHECKPOINTS = PARENT / "checkpoints"
PARENT_PREDICTIONS = PARENT / "predictions"
CACHE = ROOT / "cache/trainval"
CHECKPOINTS = ROOT / "checkpoints"
LOGS = ROOT / "logs"
METRICS = ROOT / "metrics"
PREDICTIONS = ROOT / "predictions"
VALIDATION_GATE_MARKER = ROOT / "VALIDATION_GATE_PASSED.json"
SEEDS = (42, 43, 44)
VARIANTS = ("B", "C")
MAX_EPOCHS = 200
PATIENCE = 25
BATCH_SIZE = 64
PARENT_CHECKPOINT_SHA256 = {
    42: "b5338a27d66686afad84a811d5d4f55adf55ff9432c036a41a59aed54db37d7b",
    43: "d3b1642136c7deb6354da535f11b152198ec4c036ec7d875223fc70f7ef9043c",
    44: "c84b8ba274a413f08e338d024ada6644b14f689b572439c76bdc336fe431a77b",
}


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


def atomic_checkpoint(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def authenticate_parent_checkpoints() -> None:
    for seed, expected in PARENT_CHECKPOINT_SHA256.items():
        path = PARENT_CHECKPOINTS / f"A_seed{seed}.best.pt"
        if not path.is_file():
            raise FileNotFoundError(f"Missing Exp089 parent checkpoint: {path}")
        observed = sha256(path)
        if observed != expected:
            raise ValueError(f"Exp089 A seed {seed} SHA drift: {observed}")


def load_parent_model(seed: int, device: torch.device) -> PedCMTTransfer:
    authenticate_parent_checkpoints()
    checkpoint = torch.load(
        PARENT_CHECKPOINTS / f"A_seed{seed}.best.pt",
        map_location=device,
        weights_only=False,
    )
    model = PedCMTTransfer(variant="A", ego_dim=1).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def extract_parent_outputs(
    model: PedCMTTransfer,
    bbox: np.ndarray,
    ego: np.ndarray,
    device: torch.device,
    *,
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract immutable logits and the full final BBox/ego class-token pair."""
    dataset = TensorDataset(
        torch.from_numpy(np.array(bbox, dtype=np.float32, copy=True)),
        torch.from_numpy(np.array(ego, dtype=np.float32, copy=True)),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    logits: list[np.ndarray] = []
    anchors: list[np.ndarray] = []
    captured: list[Tensor] = []

    def capture(_module: nn.Module, inputs: tuple[Tensor, ...]) -> None:
        captured.append(inputs[0].detach())

    model.eval()
    handle = model.endpoint.register_forward_pre_hook(capture)
    try:
        with torch.inference_mode():
            for bbox_batch, ego_batch in loader:
                captured.clear()
                output = model(bbox_batch.to(device), ego_batch.to(device))[0]
                if len(captured) != 1 or captured[0].shape[1] != 256:
                    raise RuntimeError("Full parent anchor hook did not fire exactly once")
                logits.append(output.cpu().numpy())
                anchors.append(captured[0].cpu().numpy())
    finally:
        handle.remove()
    return np.concatenate(logits), np.concatenate(anchors)


def load_trainval() -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    rows = pd.read_csv(PARENT_SOURCE / "rows.csv")
    arrays = {
        "bbox": np.load(PARENT_SOURCE / "bbox.npy", mmap_mode="r"),
        "ego": np.load(PARENT_SOURCE / "ego_speed.npy", mmap_mode="r"),
        "raw_pose": np.load(PARENT_DERIVED / "pose_std.npy", mmap_mode="r"),
        "biu_union": np.load(PARENT_DERIVED / "biu_union_std.npy", mmap_mode="r"),
    }
    if rows["split"].value_counts().to_dict() != {"train": 1552, "val": 274}:
        raise ValueError("Exp089 train/validation cohort drifted")
    expected = {
        "bbox": (1826, 16, 4),
        "ego": (1826, 16, 1),
        "raw_pose": (1826, 16, 34),
        "biu_union": (1826, 135),
    }
    for name, values in arrays.items():
        if values.shape != expected[name] or not np.isfinite(values).all():
            raise ValueError(f"Exp089 {name} cache drifted: {values.shape}")
    return rows, arrays


def load_test() -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    source = PARENT / "cache/test_source"
    rows = pd.read_csv(source / "rows.csv")
    stats = np.load(PARENT_DERIVED / "training_stats.npz")
    pose = np.asarray(np.load(source / "pose.npy"), dtype=np.float32)
    v1 = np.asarray(np.load(source / "biu_v1.npy"), dtype=np.float32)
    v2 = np.asarray(np.load(source / "biu_v2.npy"), dtype=np.float32)
    raw_pose = ((pose - stats["pose_mean"]) / stats["pose_std"]).astype(np.float32)
    union = np.concatenate([v1, v2], axis=1)
    biu_union = ((union - stats["union_mean"]) / stats["union_std"]).astype(np.float32)
    arrays = {
        "bbox": np.asarray(np.load(source / "bbox.npy"), dtype=np.float32),
        "ego": np.asarray(np.load(source / "ego_speed.npy"), dtype=np.float32),
        "raw_pose": raw_pose,
        "biu_union": biu_union,
    }
    if len(rows) != 1501 or rows["split"].unique().tolist() != ["test"]:
        raise ValueError("Exp089 final JAAD cohort drifted")
    if any(len(values) != len(rows) or not np.isfinite(values).all() for values in arrays.values()):
        raise ValueError("Final JAAD arrays are incomplete or non-finite")
    return rows, arrays


def balanced_indices(labels: np.ndarray, seed: int) -> np.ndarray:
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    if not np.array_equal(np.unique(y), np.array([0, 1])):
        raise ValueError("Balanced sampling requires both binary classes")
    rng = np.random.default_rng(seed)
    by_class = [np.flatnonzero(y == value) for value in (0, 1)]
    target = max(map(len, by_class))
    selected: list[np.ndarray] = []
    for indices in by_class:
        if len(indices) < target:
            indices = np.concatenate(
                [indices, rng.choice(indices, size=target - len(indices), replace=True)]
            )
        selected.append(indices)
    result = np.concatenate(selected)
    rng.shuffle(result)
    return result.astype(np.int64)


def is_better_checkpoint(new_loss: float, new_epoch: int, best_loss: float, best_epoch: int) -> bool:
    return new_loss < best_loss or (new_loss == best_loss and new_epoch < best_epoch)


def fixed_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float | int]:
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    probability = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    prediction = (probability >= 0.5).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, prediction)),
        "precision": float(precision_score(y, prediction, zero_division=0)),
        "recall": float(recall_score(y, prediction, zero_division=0)),
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probability)),
        "pr_auc": float(average_precision_score(y, probability)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "high_conf_fp": int(np.sum((y == 0) & (probability >= 0.9))),
        "high_conf_fn": int(np.sum((y == 1) & (probability <= 0.1))),
    }


def _dataset(values: dict[str, np.ndarray]) -> TensorDataset:
    return TensorDataset(
        torch.from_numpy(np.array(values["parent_logits"], dtype=np.float32, copy=True)),
        torch.from_numpy(np.array(values["parent_anchor"], dtype=np.float32, copy=True)),
        torch.from_numpy(np.array(values["extra"], dtype=np.float32, copy=True)),
        torch.from_numpy(np.array(values["labels"], dtype=np.float32, copy=True)).reshape(-1, 1),
    )


def evaluate_branch(
    model: DirectFusionModel,
    values: dict[str, np.ndarray],
    device: torch.device,
    *,
    batch_size: int = 256,
) -> tuple[float, np.ndarray, np.ndarray]:
    loader = DataLoader(_dataset(values), batch_size=batch_size, shuffle=False, num_workers=0)
    probabilities: list[np.ndarray] = []
    corrections: list[np.ndarray] = []
    loss_sum = 0.0
    count = 0
    model.eval()
    with torch.inference_mode():
        for parent_logits, parent_anchor, extra, labels in loader:
            logits, correction = model(
                parent_logits.to(device), parent_anchor.to(device), extra.to(device)
            )
            labels = labels.to(device)
            loss_sum += float(F.binary_cross_entropy_with_logits(logits, labels, reduction="sum").item())
            count += len(labels)
            probabilities.append(torch.sigmoid(logits).cpu().numpy().reshape(-1))
            corrections.append(correction.cpu().numpy().reshape(-1))
    return loss_sum / count, np.concatenate(probabilities), np.concatenate(corrections)


def fit_branch(
    variant: str,
    train_values: dict[str, np.ndarray],
    val_values: dict[str, np.ndarray],
    *,
    seed: int,
    device: torch.device,
    checkpoint_path: Path,
    history_path: Path,
    parent_sha256: str,
    max_epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
) -> dict[str, Any]:
    if variant not in VARIANTS:
        raise ValueError("Only frozen variants B and C may be trained")
    seed_all(seed)
    model = DirectFusionModel(variant).to(device)
    optimizer = Adam(model.parameters(), lr=5e-4, betas=(0.9, 0.98), eps=1e-6)
    loader = DataLoader(
        _dataset(train_values),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
    )
    best_loss = float("inf")
    best_epoch = max_epochs + 1
    stale = 0
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    initial_loss, initial_probability, initial_correction = evaluate_branch(model, val_values, device)
    if not np.array_equal(initial_correction, np.zeros_like(initial_correction)):
        raise RuntimeError("Direct branch did not preserve the parent at initialization")

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss_sum = 0.0
        count = 0
        for parent_logits, parent_anchor, extra, labels in loader:
            parent_logits = parent_logits.to(device)
            parent_anchor = parent_anchor.to(device)
            extra = extra.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(parent_logits, parent_anchor, extra)[0]
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss in {variant} seed {seed} epoch {epoch}")
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.item()) * len(labels)
            count += len(labels)

        val_loss, probability, correction = evaluate_branch(model, val_values, device)
        result_metrics = fixed_metrics(val_values["labels"], probability)
        row = {
            "variant": variant,
            "seed": seed,
            "epoch": epoch,
            "train_classification_loss": train_loss_sum / count,
            "validation_classification_loss": val_loss,
            "validation_correction_mean": float(np.mean(correction)),
            "validation_correction_abs_mean": float(np.mean(np.abs(correction))),
            "validation_positive_correction_fraction": float(np.mean(correction > 1e-7)),
            "validation_negative_correction_fraction": float(np.mean(correction < -1e-7)),
            **{f"validation_{key}": value for key, value in result_metrics.items()},
        }
        history.append(row)
        history_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(history).to_csv(history_path, index=False)
        if is_better_checkpoint(val_loss, epoch, best_loss, best_epoch):
            best_loss, best_epoch, stale = val_loss, epoch, 0
            atomic_checkpoint(
                checkpoint_path,
                {
                    "model_state": model.state_dict(),
                    "variant": variant,
                    "seed": seed,
                    "epoch": epoch,
                    "validation_classification_loss": val_loss,
                    "trainable_parameters": count_trainable_parameters(model),
                    "parent_sha256": parent_sha256,
                    "architecture": "full 256-D parent anchor; 128-D extra; bounded signed direct residual",
                    "initial_validation_classification_loss": initial_loss,
                    "initial_parent_probability_sha256": hashlib.sha256(initial_probability.tobytes()).hexdigest(),
                },
            )
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0 or stale == 0:
            print(
                f"{variant}_seed{seed} epoch={epoch:03d} val_cls={val_loss:.6f} "
                f"val_f1={result_metrics['f1']:.6f} val_pr={result_metrics['pr_auc']:.6f} "
                f"corr={row['validation_correction_mean']:.4f}",
                flush=True,
            )
        if stale >= patience:
            break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    val_loss, probability, correction = evaluate_branch(model, val_values, device)
    return {
        "variant": variant,
        "seed": seed,
        "epochs_completed": len(history),
        "best_epoch": int(checkpoint["epoch"]),
        "best_validation_classification_loss": float(val_loss),
        "elapsed_seconds": time.perf_counter() - started,
        "trainable_parameters": count_trainable_parameters(model),
        "probability": probability,
        "correction": correction,
    }


def _prepared_parent(seed: int) -> tuple[np.ndarray, np.ndarray]:
    logits = np.load(CACHE / f"parent_seed{seed}_logits.npy", mmap_mode="r")
    anchor = np.load(CACHE / f"parent_seed{seed}_anchor.npy", mmap_mode="r")
    if logits.shape != (1826, 1) or anchor.shape != (1826, 256):
        raise ValueError(f"Prepared parent outputs drifted for seed {seed}")
    return logits, anchor


def prepare(device_name: str) -> None:
    authenticate_parent_checkpoints()
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    rows, arrays = load_trainval()
    val_indices = np.flatnonzero(rows["split"].eq("val").to_numpy())
    CACHE.mkdir(parents=True, exist_ok=True)
    parity: dict[str, float] = {}
    for seed in SEEDS:
        parent = load_parent_model(seed, device)
        logits, anchor = extract_parent_outputs(parent, arrays["bbox"], arrays["ego"], device)
        saved = pd.read_csv(PARENT_PREDICTIONS / f"validation_A_seed{seed}.csv")
        expected_ids = rows.iloc[val_indices]["sequence_id"].astype(str).to_numpy()
        if not np.array_equal(expected_ids, saved["sequence_id"].astype(str).to_numpy()):
            raise RuntimeError(f"Exp089 validation sequence order drifted for seed {seed}")
        probability = torch.sigmoid(torch.from_numpy(logits[val_indices])).numpy().reshape(-1)
        difference = float(np.max(np.abs(probability - saved["probability"].to_numpy())))
        if difference > 1e-6:
            raise RuntimeError(f"Exp089 parent reproduction failed for seed {seed}: {difference:.3g}")
        np.save(CACHE / f"parent_seed{seed}_logits.npy", logits.astype(np.float32))
        np.save(CACHE / f"parent_seed{seed}_anchor.npy", anchor.astype(np.float32))
        parity[str(seed)] = difference
    audit = {
        "parent_experiment": str(PARENT),
        "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "train_windows": 1552,
        "validation_windows": 274,
        "raw_pose_shape": list(arrays["raw_pose"].shape),
        "biu_union_shape": list(arrays["biu_union"].shape),
        "parent_anchor_dimension": 256,
        "parent_probability_max_abs_difference": parity,
        "test_cache_accessed": False,
    }
    atomic_json(METRICS / "source_and_parent_audit.json", audit)
    atomic_json(ROOT / "PREPARATION_COMPLETE.json", audit)


def _split_values(
    rows: pd.DataFrame,
    parent_logits: np.ndarray,
    parent_anchor: np.ndarray,
    extra: np.ndarray,
    split: str,
    *,
    seed: int | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    indices = np.flatnonzero(rows["split"].eq(split).to_numpy())
    if split == "train" and seed is not None:
        indices = indices[balanced_indices(rows.iloc[indices]["label"].to_numpy(), seed)]
    return indices, {
        "parent_logits": np.asarray(parent_logits[indices], dtype=np.float32),
        "parent_anchor": np.asarray(parent_anchor[indices], dtype=np.float32),
        "extra": np.asarray(extra[indices], dtype=np.float32),
        "labels": rows.iloc[indices]["label"].to_numpy(dtype=np.float32),
    }


def train_one(variant: str, seed: int, device_name: str, *, smoke: bool = False) -> dict[str, Any]:
    if not (ROOT / "PREPARATION_COMPLETE.json").is_file():
        raise RuntimeError("Run prepare before training")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    rows, arrays = load_trainval()
    logits, anchor = _prepared_parent(seed)
    extra = arrays["raw_pose" if variant == "B" else "biu_union"]
    _, train_values = _split_values(rows, logits, anchor, extra, "train", seed=seed)
    val_indices, val_values = _split_values(rows, logits, anchor, extra, "val")
    stem = f"{'smoke_' if smoke else ''}{variant}_seed{seed}"
    result = fit_branch(
        variant,
        train_values,
        val_values,
        seed=seed,
        device=device,
        checkpoint_path=CHECKPOINTS / f"{stem}.best.pt",
        history_path=LOGS / f"{stem}.history.csv",
        parent_sha256=PARENT_CHECKPOINT_SHA256[seed],
        max_epochs=1 if smoke else MAX_EPOCHS,
        patience=1 if smoke else PATIENCE,
    )
    output = rows.iloc[val_indices][["sequence_id", "track_id", "video_id", "ped_id", "label"]].copy()
    output["parent_probability"] = torch.sigmoid(torch.from_numpy(np.asarray(logits[val_indices]))).numpy().reshape(-1)
    output["probability"] = result.pop("probability")
    output["prediction"] = (output["probability"] >= 0.5).astype(np.int64)
    output["correction_logit"] = result.pop("correction")
    PREDICTIONS.mkdir(parents=True, exist_ok=True)
    output.to_csv(PREDICTIONS / f"validation_{stem}.csv", index=False)
    atomic_json(LOGS / f"{stem}.complete.json", result)
    return result


def assert_training_gate(checkpoint_dir: Path) -> None:
    missing = [
        f"{variant}_seed{seed}.best.pt"
        for variant in VARIANTS
        for seed in SEEDS
        if not (checkpoint_dir / f"{variant}_seed{seed}.best.pt").is_file()
    ]
    if missing:
        raise RuntimeError(f"Final analysis requires all six branch checkpoints; missing {len(missing)}")


def train_all(device_name: str) -> None:
    if not (ROOT / "PREPARATION_COMPLETE.json").is_file():
        prepare(device_name)
    for variant in VARIANTS:
        for seed in SEEDS:
            checkpoint = CHECKPOINTS / f"{variant}_seed{seed}.best.pt"
            marker = LOGS / f"{variant}_seed{seed}.complete.json"
            if checkpoint.is_file() and marker.is_file():
                print(f"Skipping complete {variant} seed {seed}", flush=True)
                continue
            print(json.dumps(train_one(variant, seed, device_name), indent=2), flush=True)
    assert_training_gate(CHECKPOINTS)
    atomic_json(ROOT / "TRAINING_COMPLETE.json", {"variants": list(VARIANTS), "seeds": list(SEEDS)})


def _validation_results() -> pd.DataFrame:
    rows, _ = load_trainval()
    val_indices = np.flatnonzero(rows["split"].eq("val").to_numpy())
    labels = rows.iloc[val_indices]["label"].to_numpy(dtype=np.int64)
    results: list[dict[str, Any]] = []
    for seed in SEEDS:
        logits, _ = _prepared_parent(seed)
        parent_probability = torch.sigmoid(torch.from_numpy(np.asarray(logits[val_indices]))).numpy().reshape(-1)
        results.append(
            {
                "variant": "A",
                "seed": seed,
                **fixed_metrics(labels, parent_probability),
                "positive_correction_fraction": 0.0,
                "negative_correction_fraction": 0.0,
            }
        )
        for variant in VARIANTS:
            saved = pd.read_csv(PREDICTIONS / f"validation_{variant}_seed{seed}.csv")
            if not np.array_equal(
                rows.iloc[val_indices]["sequence_id"].astype(str).to_numpy(),
                saved["sequence_id"].astype(str).to_numpy(),
            ):
                raise RuntimeError(f"Validation prediction order drifted for {variant} seed {seed}")
            probability = saved["probability"].to_numpy(dtype=np.float64)
            correction = saved["correction_logit"].to_numpy(dtype=np.float64)
            results.append(
                {
                    "variant": variant,
                    "seed": seed,
                    **fixed_metrics(labels, probability),
                    "positive_correction_fraction": float(np.mean(correction > 1e-7)),
                    "negative_correction_fraction": float(np.mean(correction < -1e-7)),
                }
            )
    return pd.DataFrame(results)


def evaluate_validation_gate(results: pd.DataFrame) -> dict[str, Any]:
    required = {"A", "B", "C"}
    if set(results["variant"]) != required or len(results) != 9:
        raise ValueError("Validation gate requires A/B/C for all three seeds")
    means = results.groupby("variant").mean(numeric_only=True)
    c_pr = float(means.loc["C", "pr_auc"])
    criteria = {
        "pr_auc_margin": c_pr - float(means.loc["A", "pr_auc"]) >= 0.005
        and c_pr - float(means.loc["B", "pr_auc"]) >= 0.005,
        "f1_noninferiority": float(means.loc["C", "f1"]) >= float(means.loc["A", "f1"]) - 0.005
        and float(means.loc["C", "f1"]) >= float(means.loc["B", "f1"]) - 0.005,
        "accuracy_noninferiority": float(means.loc["C", "accuracy"])
        >= float(means.loc["A", "accuracy"]) - 0.005
        and float(means.loc["C", "accuracy"]) >= float(means.loc["B", "accuracy"]) - 0.005,
        "pr_auc_seed_consistency": True,
        "signed_corrections": True,
        "high_confidence_error_limit": float(
            means.loc["C", "high_conf_fp"] + means.loc["C", "high_conf_fn"]
        )
        <= min(
            float(means.loc["A", "high_conf_fp"] + means.loc["A", "high_conf_fn"]),
            float(means.loc["B", "high_conf_fp"] + means.loc["B", "high_conf_fn"]),
        )
        + 5.0,
    }
    for seed in SEEDS:
        seed_rows = results[results["seed"].eq(seed)].set_index("variant")
        if not (
            seed_rows.loc["C", "pr_auc"] > seed_rows.loc["A", "pr_auc"]
            and seed_rows.loc["C", "pr_auc"] > seed_rows.loc["B", "pr_auc"]
        ):
            criteria["pr_auc_seed_consistency"] = False
        if not (
            seed_rows.loc["C", "positive_correction_fraction"] >= 0.01
            and seed_rows.loc["C", "negative_correction_fraction"] >= 0.01
        ):
            criteria["signed_corrections"] = False
    return {
        "passed": bool(all(criteria.values())),
        "criteria": {name: bool(value) for name, value in criteria.items()},
        "mean_metrics": {
            variant: {
                metric: float(means.loc[variant, metric])
                for metric in ("accuracy", "f1", "roc_auc", "pr_auc", "high_conf_fp", "high_conf_fn")
            }
            for variant in ("A", "B", "C")
        },
    }


def validation_gate() -> dict[str, Any]:
    assert_training_gate(CHECKPOINTS)
    results = _validation_results()
    METRICS.mkdir(parents=True, exist_ok=True)
    results.to_csv(METRICS / "validation_per_seed_results.csv", index=False)
    summary = results.groupby("variant")[["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "high_conf_fp", "high_conf_fn"]].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary.reset_index().to_csv(METRICS / "validation_summary_results.csv", index=False)
    report = evaluate_validation_gate(results)
    atomic_json(METRICS / "validation_gate_report.json", report)
    if report["passed"]:
        atomic_json(VALIDATION_GATE_MARKER, report)
    else:
        atomic_json(ROOT / "VALIDATION_ONLY_COMPLETE.json", report)
    print(json.dumps(report, indent=2))
    return report


def _branch_from_checkpoint(variant: str, seed: int, device: torch.device) -> DirectFusionModel:
    checkpoint = torch.load(
        CHECKPOINTS / f"{variant}_seed{seed}.best.pt", map_location=device, weights_only=False
    )
    if checkpoint.get("parent_sha256") != PARENT_CHECKPOINT_SHA256[seed]:
        raise RuntimeError(f"{variant} seed {seed} is not bound to its authenticated parent")
    model = DirectFusionModel(variant).to(device)
    model.load_state_dict(checkpoint["model_state"])
    return model


def final_evaluate(device_name: str) -> None:
    if not VALIDATION_GATE_MARKER.is_file():
        raise RuntimeError("Final test is locked because the validation gate has not passed")
    assert_training_gate(CHECKPOINTS)
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    rows, arrays = load_test()
    labels = rows["label"].to_numpy(dtype=np.int64)
    metric_rows: list[dict[str, Any]] = []
    delta_rows: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []
    latency_rows: list[dict[str, Any]] = []
    probabilities: dict[tuple[str, int], np.ndarray] = {}

    for seed in SEEDS:
        parent = load_parent_model(seed, device)
        parent_logits, parent_anchor = extract_parent_outputs(
            parent, arrays["bbox"], arrays["ego"], device
        )
        parent_probability = torch.sigmoid(torch.from_numpy(parent_logits)).numpy().reshape(-1)
        saved_parent = pd.read_csv(PARENT_PREDICTIONS / f"test_A_seed{seed}.csv")
        if not np.array_equal(rows["sequence_id"].astype(str), saved_parent["sequence_id"].astype(str)):
            raise RuntimeError(f"Final parent sequence order drifted for seed {seed}")
        max_difference = float(
            np.max(np.abs(parent_probability - saved_parent["probability"].to_numpy(dtype=np.float64)))
        )
        if max_difference > 1e-6:
            raise RuntimeError(f"Final parent reproduction failed for seed {seed}: {max_difference:.3g}")
        probabilities[("A", seed)] = parent_probability
        metric_rows.append({"variant": "A", "seed": seed, **fixed_metrics(labels, parent_probability)})

        for variant, extra_name in (("B", "raw_pose"), ("C", "biu_union")):
            model = _branch_from_checkpoint(variant, seed, device)
            values = {
                "parent_logits": parent_logits,
                "parent_anchor": parent_anchor,
                "extra": arrays[extra_name],
                "labels": labels.astype(np.float32),
            }
            _, probability, correction = evaluate_branch(model, values, device)
            probabilities[(variant, seed)] = probability
            metric_rows.append({"variant": variant, "seed": seed, **fixed_metrics(labels, probability)})
            output = rows[["sequence_id", "track_id", "video_id", "ped_id", "label"]].copy()
            output["parent_probability"] = parent_probability
            output["probability"] = probability
            output["prediction"] = (probability >= 0.5).astype(np.int64)
            output["correction_logit"] = correction
            PREDICTIONS.mkdir(parents=True, exist_ok=True)
            output.to_csv(PREDICTIONS / f"test_{variant}_seed{seed}.csv", index=False)

            parent_tensor = torch.from_numpy(parent_logits[:1]).to(device)
            anchor_tensor = torch.from_numpy(parent_anchor[:1]).to(device)
            extra_tensor = torch.from_numpy(np.asarray(arrays[extra_name][:1], dtype=np.float32)).to(device)
            model.eval()
            with torch.inference_mode():
                for _ in range(10):
                    model(parent_tensor, anchor_tensor, extra_tensor)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                started = time.perf_counter()
                for _ in range(100):
                    model(parent_tensor, anchor_tensor, extra_tensor)
                if device.type == "cuda":
                    torch.cuda.synchronize()
            latency_rows.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "trainable_branch_parameters": count_trainable_parameters(model),
                    "branch_latency_ms_batch1": (time.perf_counter() - started) * 10.0,
                    "parent_probability_reproduction_max_abs_difference": max_difference,
                }
            )

    metrics_frame = pd.DataFrame(metric_rows)
    for left, right in (("A", "B"), ("A", "C"), ("B", "C")):
        for seed in SEEDS:
            before = probabilities[(left, seed)]
            after = probabilities[(right, seed)]
            left_row = metrics_frame.query("variant == @left and seed == @seed").iloc[0]
            right_row = metrics_frame.query("variant == @right and seed == @seed").iloc[0]
            delta_rows.append(
                {
                    "comparison": f"{left}->{right}",
                    "seed": seed,
                    **{
                        f"delta_{metric}": float(right_row[metric] - left_row[metric])
                        for metric in ("accuracy", "f1", "roc_auc", "pr_auc")
                    },
                }
            )
            before_prediction = before >= 0.5
            after_prediction = after >= 0.5
            before_correct = before_prediction == labels
            after_correct = after_prediction == labels
            transition_rows.append(
                {
                    "comparison": f"{left}->{right}",
                    "seed": seed,
                    "wrong_to_correct": int(np.sum(~before_correct & after_correct)),
                    "correct_to_wrong": int(np.sum(before_correct & ~after_correct)),
                    "fp_to_tn": int(np.sum((labels == 0) & before_prediction & ~after_prediction)),
                    "fn_to_tp": int(np.sum((labels == 1) & ~before_prediction & after_prediction)),
                    "tn_to_fp": int(np.sum((labels == 0) & ~before_prediction & after_prediction)),
                    "tp_to_fn": int(np.sum((labels == 1) & before_prediction & ~after_prediction)),
                }
            )

    METRICS.mkdir(parents=True, exist_ok=True)
    metrics_frame.to_csv(METRICS / "fixed_0p5_results.csv", index=False)
    metrics_frame.to_csv(METRICS / "per_seed_results.csv", index=False)
    summary = metrics_frame.groupby("variant")[["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "fp", "fn", "high_conf_fp", "high_conf_fn"]].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary.reset_index().to_csv(METRICS / "summary_results.csv", index=False)
    pd.DataFrame(delta_rows).to_csv(METRICS / "paired_deltas.csv", index=False)
    pd.DataFrame(transition_rows).to_csv(METRICS / "error_transitions_fixed_0p5.csv", index=False)
    pd.DataFrame(latency_rows).to_csv(METRICS / "efficiency.csv", index=False)
    atomic_json(
        ROOT / "EXPERIMENT_COMPLETE.json",
        {
            "test_windows": 1501,
            "variants": ["A", "B", "C"],
            "seeds": list(SEEDS),
            "threshold": 0.5,
            "validation_gate_passed_before_test": True,
        },
    )


def smoke(device_name: str) -> None:
    if not (ROOT / "PREPARATION_COMPLETE.json").is_file():
        prepare(device_name)
    runs = [train_one(variant, 42, device_name, smoke=True) for variant in VARIANTS]
    atomic_json(LOGS / "SMOKE_COMPLETE.json", {"device": device_name, "runs": runs})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "smoke", "train", "train-all", "gate", "evaluate"))
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.device)
    elif args.command == "smoke":
        smoke(args.device)
    elif args.command == "train":
        if args.variant is None or args.seed is None:
            raise SystemExit("train requires --variant and --seed")
        result = train_one(args.variant, args.seed, args.device)
        print(json.dumps(result, indent=2))
    elif args.command == "train-all":
        train_all(args.device)
    elif args.command == "gate":
        validation_gate()
    else:
        final_evaluate(args.device)


if __name__ == "__main__":
    main()
