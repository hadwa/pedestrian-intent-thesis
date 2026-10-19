#!/usr/bin/env python3
"""Frozen Exp084 preparation, training, and one-shot set03 evaluation."""

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
from torch import Tensor
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from exp084_data import (  # noqa: E402
    build_event_bbox_targets,
    build_union,
    load_population,
    standardize_from_training,
)
from pedcmt_model import PedCMTTransfer, VARIANTS, count_parameters  # noqa: E402


EXP_DIR = SCRIPT_DIR.parent
CACHE_DIR = EXP_DIR / "cache"
TRAINVAL_DERIVED = CACHE_DIR / "trainval"
CHECKPOINT_DIR = EXP_DIR / "checkpoints"
PREDICTION_DIR = EXP_DIR / "predictions"
METRICS_DIR = EXP_DIR / "metrics"
LOG_DIR = EXP_DIR / "logs"
SEEDS = (42, 43, 44)
EXTRA_DIMS = {"A": None, "B": 34, "C": 72, "D": 63, "E": 135}
MAX_EPOCHS = 200
PATIENCE = 25
BATCH_SIZE = 64


def balanced_indices(labels: np.ndarray, seed: int) -> np.ndarray:
    """Return deterministic positive oversampling indices for training only."""
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    classes = np.unique(y)
    if not np.array_equal(classes, np.array([0, 1])):
        raise ValueError(f"Expected binary labels with both classes, got {classes.tolist()}")
    rng = np.random.default_rng(seed)
    by_class = [np.flatnonzero(y == value) for value in (0, 1)]
    target = max(map(len, by_class))
    selected = []
    for indices in by_class:
        if len(indices) < target:
            extra = rng.choice(indices, size=target - len(indices), replace=True)
            indices = np.concatenate([indices, extra])
        selected.append(indices)
    result = np.concatenate(selected)
    rng.shuffle(result)
    return result.astype(np.int64)


def fixed_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float | int]:
    """Fixed-0.5 classification metrics; ranking metrics use raw probabilities."""
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    p = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if y.shape != p.shape or not np.isfinite(p).all():
        raise ValueError("Labels and finite probabilities must be aligned")
    predicted = (p >= 0.5).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, predicted)),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "high_conf_fp": int(np.sum((y == 0) & (p >= 0.9))),
        "high_conf_fn": int(np.sum((y == 1) & (p <= 0.1))),
    }


def multitask_loss(
    logits: Tensor,
    endpoints: Tensor,
    labels: Tensor,
    targets: Tensor,
    log_var_cls: Tensor,
    log_var_reg: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    classification = F.binary_cross_entropy_with_logits(logits, labels)
    regression = F.mse_loss(endpoints, targets)
    total = (
        torch.exp(-log_var_cls) * classification
        + log_var_cls
        + torch.exp(-log_var_reg) * regression
        + log_var_reg
    ).mean()
    return total, classification, regression


def assert_test_gate(checkpoint_dir: Path) -> None:
    expected = [checkpoint_dir / f"{variant}_seed{seed}.best.pt" for variant in VARIANTS for seed in SEEDS]
    missing = [path.name for path in expected if not path.is_file()]
    if missing:
        raise RuntimeError(
            f"set03 is locked until all 15 frozen checkpoints exist; missing {len(missing)}: "
            + ", ".join(missing)
        )


def is_better_checkpoint(new_loss: float, new_epoch: int, best_loss: float, best_epoch: int) -> bool:
    return new_loss < best_loss or (new_loss == best_loss and new_epoch < best_epoch)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_checkpoint(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prepare() -> None:
    """Prepare train/validation-only targets and standardized features."""
    population = load_population("trainval", allow_test=False)
    rows = population.rows
    train_mask = rows["split"].eq("train").to_numpy()
    if set(rows["split"].unique()) != {"train", "val"}:
        raise ValueError("Frozen trainval population contains unexpected memberships")

    union, union_names = build_union(
        population.biu_v1,
        population.biu_v2,
        population.biu_v1_names,
        population.biu_v2_names,
    )
    v1_std, v1_stats = standardize_from_training(population.biu_v1, train_mask)
    v2_std, v2_stats = standardize_from_training(population.biu_v2, train_mask)
    union_std, union_stats = standardize_from_training(union, train_mask)

    flat_pose = population.pose.reshape(-1, population.pose.shape[-1])
    pose_mask = np.repeat(train_mask, population.pose.shape[1])
    pose_std_flat, pose_stats = standardize_from_training(flat_pose, pose_mask)
    pose_std = pose_std_flat.reshape(population.pose.shape)
    endpoints = build_event_bbox_targets(rows)

    TRAINVAL_DERIVED.mkdir(parents=True, exist_ok=True)
    outputs = {
        "endpoint.npy": endpoints,
        "pose_std.npy": pose_std,
        "biu_v1_std.npy": v1_std,
        "biu_v2_std.npy": v2_std,
        "biu_union_std.npy": union_std,
    }
    for filename, values in outputs.items():
        np.save(TRAINVAL_DERIVED / filename, np.asarray(values, dtype=np.float32))
    np.savez(
        TRAINVAL_DERIVED / "training_stats.npz",
        pose_mean=pose_stats.mean,
        pose_std=pose_stats.std,
        v1_mean=v1_stats.mean,
        v1_std=v1_stats.std,
        v2_mean=v2_stats.mean,
        v2_std=v2_stats.std,
        union_mean=union_stats.mean,
        union_std=union_stats.std,
    )
    (TRAINVAL_DERIVED / "union_feature_names.json").write_text(
        json.dumps(list(union_names), indent=2) + "\n", encoding="utf-8"
    )

    source_files = [
        population.source_dir / name
        for name in ("rows.csv", "metadata.json", "bbox.npy", "ego_speed.npy", "pose.npy", "biu_v1.npy", "biu_v2.npy")
    ]
    models = {
        variant: count_parameters(PedCMTTransfer(variant=variant, ego_dim=population.ego.shape[-1]))
        for variant in VARIANTS
    }
    audit = {
        "prepared_at_unix": time.time(),
        "source_directory": str(population.source_dir),
        "source_sha256": {path.name: _sha256(path) for path in source_files},
        "sequence_id_sha256": hashlib.sha256("\n".join(rows["sequence_id"]).encode()).hexdigest(),
        "population": {
            "total": int(len(rows)),
            "train": int(train_mask.sum()),
            "validation": int((~train_mask).sum()),
            "train_positive": int(rows.loc[train_mask, "label"].sum()),
            "validation_positive": int(rows.loc[~train_mask, "label"].sum()),
        },
        "dimensions": {"bbox": 4, "ego": int(population.ego.shape[-1]), "pose": 34, "biu_v1": 72, "biu_v2": 63, "union": 135},
        "trainable_parameters": models,
        "test_cache_accessed": False,
    }
    _atomic_json(METRICS_DIR / "source_audit.json", audit)
    _atomic_json(EXP_DIR / "PREPARATION_COMPLETE.json", audit)
    print(json.dumps(audit, indent=2))


class ArrayDataset(Dataset):
    def __init__(
        self,
        bbox: np.ndarray,
        ego: np.ndarray,
        labels: np.ndarray,
        endpoints: np.ndarray,
        extra: np.ndarray | None,
    ):
        self.bbox = torch.from_numpy(np.asarray(bbox, dtype=np.float32))
        self.ego = torch.from_numpy(np.asarray(ego, dtype=np.float32))
        self.labels = torch.from_numpy(np.asarray(labels, dtype=np.float32)).view(-1, 1)
        self.endpoints = torch.from_numpy(np.asarray(endpoints, dtype=np.float32))
        self.extra = None if extra is None else torch.from_numpy(np.asarray(extra, dtype=np.float32))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[Tensor, ...]:
        extra = torch.empty(0) if self.extra is None else self.extra[index]
        return self.bbox[index], self.ego[index], self.labels[index], self.endpoints[index], extra


def _extra_for_variant(variant: str, derived: Path) -> np.ndarray | None:
    filename = {
        "A": None,
        "B": "pose_std.npy",
        "C": "biu_v1_std.npy",
        "D": "biu_v2_std.npy",
        "E": "biu_union_std.npy",
    }[variant]
    return None if filename is None else np.load(derived / filename)


def _forward(model: PedCMTTransfer, variant: str, bbox: Tensor, ego: Tensor, extra: Tensor) -> tuple[Tensor, ...]:
    if variant == "B":
        return model(bbox, ego, pose=extra)
    if variant in {"C", "D", "E"}:
        return model(bbox, ego, biu=extra)
    return model(bbox, ego)


def _evaluate_loader(
    model: PedCMTTransfer, loader: DataLoader, variant: str, device: torch.device
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total_cls = 0.0
    total_n = 0
    probabilities: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    with torch.inference_mode():
        for bbox, ego, label, _endpoint, extra in loader:
            bbox, ego, label, extra = bbox.to(device), ego.to(device), label.to(device), extra.to(device)
            logits = _forward(model, variant, bbox, ego, extra)[0]
            loss = F.binary_cross_entropy_with_logits(logits, label, reduction="sum")
            total_cls += float(loss.item())
            total_n += len(label)
            probabilities.append(torch.sigmoid(logits).cpu().numpy().reshape(-1))
            labels.append(label.cpu().numpy().reshape(-1))
    return total_cls / total_n, np.concatenate(labels), np.concatenate(probabilities)


def train_one(
    variant: str,
    seed: int,
    *,
    device_name: str,
    max_epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
    max_train_batches: int | None = None,
    artifact_prefix: str = "",
) -> dict[str, Any]:
    if variant not in VARIANTS or seed not in SEEDS:
        raise ValueError("Run is outside the frozen A--E / seed42--44 grid")
    if not (EXP_DIR / "PREPARATION_COMPLETE.json").is_file():
        raise RuntimeError("Run prepare before training")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    _seed_everything(seed)
    population = load_population("trainval", allow_test=False)
    rows = population.rows
    endpoints = np.load(TRAINVAL_DERIVED / "endpoint.npy")
    extra = _extra_for_variant(variant, TRAINVAL_DERIVED)
    train_base = np.flatnonzero(rows["split"].eq("train").to_numpy())
    val_indices = np.flatnonzero(rows["split"].eq("val").to_numpy())
    local_balanced = balanced_indices(rows.iloc[train_base]["label"].to_numpy(), seed)
    train_indices = train_base[local_balanced]

    def dataset(indices: np.ndarray) -> ArrayDataset:
        return ArrayDataset(
            population.bbox[indices],
            population.ego[indices],
            rows.iloc[indices]["label"].to_numpy(),
            endpoints[indices],
            None if extra is None else extra[indices],
        )

    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(dataset(train_indices), batch_size=BATCH_SIZE, shuffle=True, generator=generator, num_workers=0)
    val_loader = DataLoader(dataset(val_indices), batch_size=256, shuffle=False, num_workers=0)
    model = PedCMTTransfer(variant=variant, ego_dim=population.ego.shape[-1]).to(device)
    optimizer = Adam(model.parameters(), lr=5e-4, betas=(0.9, 0.98), eps=1e-6)

    stem = f"{artifact_prefix}{variant}_seed{seed}"
    checkpoint_path = CHECKPOINT_DIR / f"{stem}.best.pt"
    history_path = LOG_DIR / f"{stem}.history.csv"
    marker_path = LOG_DIR / f"{stem}.complete.json"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    best_epoch = max_epochs + 1
    stale = 0
    history: list[dict[str, Any]] = []
    started = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_total = train_cls = train_reg = 0.0
        seen = 0
        for batch_number, (bbox, ego, labels, target, extra_batch) in enumerate(train_loader, start=1):
            bbox, ego, labels, target, extra_batch = (
                bbox.to(device), ego.to(device), labels.to(device), target.to(device), extra_batch.to(device)
            )
            optimizer.zero_grad(set_to_none=True)
            logits, endpoint, lv_cls, lv_reg = _forward(model, variant, bbox, ego, extra_batch)
            loss, cls_loss, reg_loss = multitask_loss(logits, endpoint, labels, target, lv_cls, lv_reg)
            loss.backward()
            optimizer.step()
            batch_size = len(labels)
            train_total += float(loss.item()) * batch_size
            train_cls += float(cls_loss.item()) * batch_size
            train_reg += float(reg_loss.item()) * batch_size
            seen += batch_size
            if max_train_batches is not None and batch_number >= max_train_batches:
                break

        val_cls, val_labels, val_probabilities = _evaluate_loader(model, val_loader, variant, device)
        row = {
            "variant": variant,
            "seed": seed,
            "epoch": epoch,
            "train_total_loss": train_total / seen,
            "train_classification_loss": train_cls / seen,
            "train_endpoint_loss": train_reg / seen,
            "validation_classification_loss": val_cls,
            **{f"validation_{key}": value for key, value in fixed_metrics(val_labels, val_probabilities).items()},
        }
        history.append(row)
        pd.DataFrame(history).to_csv(history_path, index=False)
        print(
            f"{stem} epoch={epoch:03d} train={row['train_total_loss']:.6f} "
            f"val_cls={val_cls:.6f} val_f1={row['validation_f1']:.6f} "
            f"val_pr_auc={row['validation_pr_auc']:.6f}",
            flush=True,
        )
        if is_better_checkpoint(val_cls, epoch, best_loss, best_epoch):
            best_loss, best_epoch, stale = val_cls, epoch, 0
            _atomic_checkpoint(
                checkpoint_path,
                {
                    "model_state": model.state_dict(),
                    "variant": variant,
                    "seed": seed,
                    "epoch": epoch,
                    "validation_classification_loss": val_cls,
                    "parameter_count": count_parameters(model),
                    "design": "Exp084 frozen PedCMT transfer",
                },
            )
        else:
            stale += 1
        if stale >= patience:
            break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    val_cls, val_labels, val_probabilities = _evaluate_loader(model, val_loader, variant, device)
    prediction_path = PREDICTION_DIR / f"validation_{stem}.csv"
    prediction = rows.iloc[val_indices][["sequence_id", "label"]].copy()
    prediction["probability"] = val_probabilities
    prediction["prediction"] = (val_probabilities >= 0.5).astype(np.int64)
    prediction.to_csv(prediction_path, index=False)
    result = {
        "variant": variant,
        "seed": seed,
        "device": str(device),
        "epochs_completed": len(history),
        "best_epoch": int(checkpoint["epoch"]),
        "best_validation_classification_loss": float(val_cls),
        "elapsed_seconds": time.perf_counter() - started,
        "checkpoint": str(checkpoint_path),
        "validation_predictions": str(prediction_path),
        "parameter_count": int(checkpoint["parameter_count"]),
        **{f"validation_{key}": value for key, value in fixed_metrics(val_labels, val_probabilities).items()},
    }
    _atomic_json(marker_path, result)
    return result


def smoke(device_name: str) -> None:
    prepare()
    results = []
    for variant in VARIANTS:
        results.append(
            train_one(
                variant,
                42,
                device_name=device_name,
                max_epochs=1,
                patience=1,
                max_train_batches=1,
                artifact_prefix="smoke_",
            )
        )
    _atomic_json(LOG_DIR / "SMOKE_COMPLETE.json", {"device": device_name, "runs": results})


def train_all(device_name: str) -> None:
    for variant in VARIANTS:
        for seed in SEEDS:
            marker = LOG_DIR / f"{variant}_seed{seed}.complete.json"
            checkpoint = CHECKPOINT_DIR / f"{variant}_seed{seed}.best.pt"
            if marker.is_file() and checkpoint.is_file():
                print(f"Skipping completed {variant} seed {seed}", flush=True)
                continue
            train_one(variant, seed, device_name=device_name)
    assert_test_gate(CHECKPOINT_DIR)
    _atomic_json(EXP_DIR / "TRAINING_COMPLETE.json", {"variants": list(VARIANTS), "seeds": list(SEEDS)})


def _apply_stats(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    transformed = (np.asarray(values, dtype=np.float32) - mean) / std
    if not np.isfinite(transformed).all():
        raise ValueError("Test standardization produced non-finite values")
    return transformed.astype(np.float32)


def evaluate(device_name: str) -> None:
    """One-shot final set03 evaluation, accessible only after all runs finish."""
    assert_test_gate(CHECKPOINT_DIR)
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    population = load_population("test", allow_test=True)
    stats = np.load(TRAINVAL_DERIVED / "training_stats.npz")
    union, _ = build_union(
        population.biu_v1,
        population.biu_v2,
        population.biu_v1_names,
        population.biu_v2_names,
    )
    extras: dict[str, np.ndarray | None] = {
        "A": None,
        "B": _apply_stats(population.pose, stats["pose_mean"], stats["pose_std"]),
        "C": _apply_stats(population.biu_v1, stats["v1_mean"], stats["v1_std"]),
        "D": _apply_stats(population.biu_v2, stats["v2_mean"], stats["v2_std"]),
        "E": _apply_stats(union, stats["union_mean"], stats["union_std"]),
    }
    labels = population.rows["label"].to_numpy(dtype=np.int64)
    dummy_targets = np.zeros((len(labels), 4), dtype=np.float32)
    per_seed: list[dict[str, Any]] = []
    all_predictions: dict[tuple[str, int], np.ndarray] = {}
    latency_rows: list[dict[str, Any]] = []

    for variant in VARIANTS:
        for seed in SEEDS:
            model = PedCMTTransfer(variant=variant, ego_dim=population.ego.shape[-1]).to(device)
            checkpoint = torch.load(CHECKPOINT_DIR / f"{variant}_seed{seed}.best.pt", map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model_state"])
            dataset = ArrayDataset(population.bbox, population.ego, labels, dummy_targets, extras[variant])
            loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0)
            _, observed_labels, probabilities = _evaluate_loader(model, loader, variant, device)
            if not np.array_equal(observed_labels.astype(np.int64), labels):
                raise RuntimeError("Prediction order changed during final evaluation")
            all_predictions[(variant, seed)] = probabilities
            row = {"variant": variant, "seed": seed, **fixed_metrics(labels, probabilities)}
            per_seed.append(row)
            output = population.rows[["sequence_id", "set_id", "video_id", "ped_id", "label"]].copy()
            output["probability"] = probabilities
            output["prediction"] = (probabilities >= 0.5).astype(np.int64)
            output.to_csv(PREDICTION_DIR / f"test_{variant}_seed{seed}.csv", index=False)

            first = next(iter(loader))
            bbox, ego, _label, _target, extra = first
            bbox, ego, extra = bbox[:1].to(device), ego[:1].to(device), extra[:1].to(device)
            model.eval()
            with torch.inference_mode():
                for _ in range(10):
                    _forward(model, variant, bbox, ego, extra)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                started = time.perf_counter()
                for _ in range(100):
                    _forward(model, variant, bbox, ego, extra)
                if device.type == "cuda":
                    torch.cuda.synchronize()
            latency_rows.append(
                {"variant": variant, "seed": seed, "predictor_latency_ms_batch1": (time.perf_counter() - started) * 10, "parameters": count_parameters(model)}
            )

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    per_seed_frame = pd.DataFrame(per_seed)
    per_seed_frame.to_csv(METRICS_DIR / "per_seed_results.csv", index=False)
    per_seed_frame.to_csv(METRICS_DIR / "fixed_0p5_results.csv", index=False)
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "fp", "fn", "high_conf_fp", "high_conf_fn"]
    summary_rows = []
    for variant, group in per_seed_frame.groupby("variant", sort=False):
        row: dict[str, Any] = {"variant": variant}
        for metric in metrics:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1))
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(METRICS_DIR / "summary_results.csv", index=False)

    comparisons = [("A", "B"), ("A", "C"), ("A", "D"), ("A", "E"), ("B", "C"), ("C", "D"), ("C", "E"), ("D", "E")]
    delta_rows = []
    transition_rows = []
    for left, right in comparisons:
        for seed in SEEDS:
            left_row = per_seed_frame.query("variant == @left and seed == @seed").iloc[0]
            right_row = per_seed_frame.query("variant == @right and seed == @seed").iloc[0]
            delta_rows.append(
                {"comparison": f"{left}->{right}", "seed": seed, **{f"delta_{metric}": float(right_row[metric] - left_row[metric]) for metric in ("f1", "roc_auc", "pr_auc")}}
            )
            left_correct = (all_predictions[(left, seed)] >= 0.5).astype(int) == labels
            right_correct = (all_predictions[(right, seed)] >= 0.5).astype(int) == labels
            transition_rows.append(
                {"comparison": f"{left}->{right}", "seed": seed, "left_wrong_right_correct": int(np.sum(~left_correct & right_correct)), "left_correct_right_wrong": int(np.sum(left_correct & ~right_correct))}
            )
    pd.DataFrame(delta_rows).to_csv(METRICS_DIR / "paired_deltas.csv", index=False)
    pd.DataFrame(transition_rows).to_csv(METRICS_DIR / "error_transitions_fixed_0p5.csv", index=False)
    pd.DataFrame(latency_rows).to_csv(METRICS_DIR / "efficiency.csv", index=False)
    _atomic_json(EXP_DIR / "EXPERIMENT_COMPLETE.json", {"test_population": "set03", "runs": 15, "threshold": 0.5})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "smoke", "train", "train-all", "evaluate"))
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "smoke":
        smoke(args.device)
    elif args.command == "train":
        if args.variant is None or args.seed is None:
            raise SystemExit("train requires --variant and --seed")
        print(json.dumps(train_one(args.variant, args.seed, device_name=args.device), indent=2))
    elif args.command == "train-all":
        train_all(args.device)
    else:
        evaluate(args.device)


if __name__ == "__main__":
    main()
