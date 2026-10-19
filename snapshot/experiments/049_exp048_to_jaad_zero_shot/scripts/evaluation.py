"""Frozen Exp048-to-JAAD inference and saved-prediction-only reporting."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset

from cache_gate import authenticate_audit_marker, authenticate_feature_cache, sequence_id_sha256
from checkpoints import authenticate_exp048_checkpoints, load_frozen_model
from contracts import EXPERIMENT_ROOT, FIXED_THRESHOLD, SEEDS, VARIANTS, VARIANT_STREAMS, sha256_file
from jaad_protocol import attach_target_labels, build_observation_manifest, load_selected_jaad_database, read_official_split


INFERENCE_BATCH_SIZE = 32


def compute_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float | int]:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if labels.ndim != 1 or probabilities.shape != labels.shape or not np.isfinite(probabilities).all():
        raise ValueError("Metrics require aligned finite one-dimensional arrays.")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Metrics require both binary classes.")
    predictions = (probabilities >= FIXED_THRESHOLD).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "threshold": FIXED_THRESHOLD,
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "pr_auc": float(average_precision_score(labels, probabilities)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def probability_diagnostics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float | int]:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predictions = (probabilities >= FIXED_THRESHOLD).astype(np.int64)
    fp_values = probabilities[(labels == 0) & (predictions == 1)]
    fn_values = probabilities[(labels == 1) & (predictions == 0)]

    def statistic(values: np.ndarray, operation, default: float = float("nan")) -> float:
        return float(operation(values)) if len(values) else default

    return {
        "probability_mean": float(probabilities.mean()),
        "probability_median": float(np.median(probabilities)),
        "probability_std": float(probabilities.std(ddof=0)),
        "probability_min": float(probabilities.min()),
        "probability_max": float(probabilities.max()),
        "fp_count": int(len(fp_values)),
        "fp_probability_mean": statistic(fp_values, np.mean),
        "fp_probability_median": statistic(fp_values, np.median),
        "fp_probability_min": statistic(fp_values, np.min),
        "fp_probability_max": statistic(fp_values, np.max),
        "fn_count": int(len(fn_values)),
        "fn_probability_mean": statistic(fn_values, np.mean),
        "fn_probability_median": statistic(fn_values, np.median),
        "fn_probability_min": statistic(fn_values, np.min),
        "fn_probability_max": statistic(fn_values, np.max),
        "high_confidence_fp": int(np.sum((labels == 0) & (probabilities >= 0.9))),
        "high_confidence_fn": int(np.sum((labels == 1) & (probabilities <= 0.1))),
    }


class FrozenInferenceDataset(Dataset):
    def __init__(self, rows: pd.DataFrame, arrays: Mapping[str, np.ndarray], variant: str):
        self.rows = rows.reset_index(drop=True)
        self.arrays = arrays
        self.variant = variant
        if variant not in VARIANTS or self.rows["sequence_id"].duplicated().any():
            raise ValueError("Invalid frozen inference dataset identity.")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        features = {}
        for stream in VARIANT_STREAMS[self.variant]:
            array_name = "exp012_pose" if stream == "pose" else stream
            value = np.asarray(self.arrays[array_name][index], dtype=np.float32)
            if stream == "static_context":
                value = np.broadcast_to(value, (30, 23)).copy()
            features[stream] = torch.as_tensor(value.copy(), dtype=torch.float32)
        return features, torch.tensor(int(self.rows.iloc[index]["label"]), dtype=torch.long)


def model_state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        value = tensor.detach().cpu().contiguous().numpy()
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def predict(model, loader: DataLoader, device: torch.device) -> np.ndarray:
    output = []
    model.eval()
    with torch.no_grad():
        for inputs, _ in loader:
            moved = {name: value.to(device, non_blocking=device.type == "cuda") for name, value in inputs.items()}
            output.append(torch.softmax(model(moved), dim=1)[:, 1].cpu().numpy())
    if not output:
        raise ValueError("Frozen inference loader is empty.")
    return np.concatenate(output).astype(np.float64)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _prediction_paths(variant: str, seed: int) -> tuple[Path, Path]:
    root = EXPERIMENT_ROOT / "predictions"
    prediction = root / f"jaad_test_{variant}_seed{seed}.csv"
    marker = root / f".jaad_test_{variant}_seed{seed}.complete.json"
    return prediction, marker


def _load_authenticated_prediction(variant: str, seed: int, checkpoint_sha: str) -> pd.DataFrame | None:
    prediction_path, marker_path = _prediction_paths(variant, seed)
    if not prediction_path.exists() and not marker_path.exists():
        return None
    if not prediction_path.is_file() or not marker_path.is_file():
        raise ValueError(f"Partial prediction artifact for {variant}{seed}.")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if (
        marker.get("phase") != "zero_shot_inference"
        or marker.get("variant") != variant
        or int(marker.get("seed", -1)) != seed
        or marker.get("checkpoint_sha256") != checkpoint_sha
        or marker.get("prediction_sha256") != sha256_file(prediction_path)
    ):
        raise ValueError(f"Prediction authentication failed for {variant}{seed}.")
    return pd.read_csv(prediction_path)


def run_evaluation(device_name: str = "cuda:0") -> dict[str, object]:
    audit = authenticate_audit_marker(EXPERIMENT_ROOT)
    records = authenticate_exp048_checkpoints()
    video_ids = read_official_split("test")
    database = load_selected_jaad_database(video_ids, include_cross=True)
    manifest = build_observation_manifest(database, video_ids, "test")
    labeled = attach_target_labels(manifest, database)
    if labeled["label"].isna().any():
        raise ValueError("Frozen JAAD cohort contains unavailable target labels.")
    labeled["label"] = labeled["label"].astype(np.int64)
    sequence_ids = labeled["sequence_id"].astype(str).tolist()
    if sequence_id_sha256(sequence_ids) != str(audit["sequence_id_sha256"]):
        raise ValueError("Final labeled cohort identity differs from AUDIT_COMPLETE.")
    _, arrays = authenticate_feature_cache(EXPERIMENT_ROOT / "cache" / "test_features", sequence_ids)
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for final inference but unavailable.")
    prediction_frames = {}
    for run_index, record in enumerate(records, start=1):
        frame = _load_authenticated_prediction(record.variant, record.seed, record.checkpoint_sha256)
        if frame is None:
            dataset = FrozenInferenceDataset(labeled, arrays, record.variant)
            loader = DataLoader(dataset, batch_size=INFERENCE_BATCH_SIZE, shuffle=False, num_workers=0)
            model = load_frozen_model(record, device)
            before = model_state_sha256(model)
            probabilities = predict(model, loader, device)
            after = model_state_sha256(model)
            if before != after:
                raise RuntimeError(f"Model state changed during inference: {record.variant}{record.seed}")
            frame = labeled[
                ["sequence_id", "video_id", "ped_id", "start_frame", "end_frame", "target_frame", "observation_frames", "label"]
            ].copy()
            frame["observation_frames"] = frame["observation_frames"].map(
                lambda values: json.dumps(list(values), separators=(",", ":"))
            )
            frame["variant"] = record.variant
            frame["seed"] = record.seed
            frame["threshold"] = FIXED_THRESHOLD
            frame["probability"] = probabilities
            frame["prediction"] = (probabilities >= FIXED_THRESHOLD).astype(np.int64)
            frame["checkpoint_sha256"] = record.checkpoint_sha256
            prediction_path, marker_path = _prediction_paths(record.variant, record.seed)
            _atomic_csv(frame, prediction_path)
            marker = {
                "phase": "zero_shot_inference",
                "variant": record.variant,
                "seed": record.seed,
                "checkpoint_sha256": record.checkpoint_sha256,
                "prediction_sha256": sha256_file(prediction_path),
                "row_count": len(frame),
                "sequence_id_sha256": sequence_id_sha256(frame["sequence_id"].astype(str).tolist()),
                "model_state_sha256_before": before,
                "model_state_sha256_after": after,
            }
            marker_path.write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if frame["sequence_id"].astype(str).tolist() != sequence_ids:
            raise ValueError(f"Prediction cohort/order drift for {record.variant}{record.seed}.")
        if not np.array_equal(frame["label"].to_numpy(dtype=np.int64), labeled["label"].to_numpy(dtype=np.int64)):
            raise ValueError(f"Prediction label drift for {record.variant}{record.seed}.")
        prediction_frames[(record.variant, record.seed)] = frame
        print(f"inference_progress={run_index}/18 identity={record.variant}{record.seed}", flush=True)
    return finalize_from_predictions(prediction_frames)


def _paired_deltas(per_seed: pd.DataFrame) -> pd.DataFrame:
    comparisons = (("A", "B"), ("B", "C"), ("C", "D"), ("C", "E"), ("C", "F"), ("D", "E"), ("D", "F"), ("E", "F"), ("A", "E"), ("A", "F"))
    indexed = per_seed.set_index(["variant", "seed"])
    rows = []
    for left, right in comparisons:
        for seed in SEEDS:
            for metric in ("f1", "pr_auc", "roc_auc"):
                rows.append({"comparison": f"{left}_to_{right}", "seed": seed, "metric": metric, "delta": float(indexed.loc[(right, seed), metric] - indexed.loc[(left, seed), metric])})
    return pd.DataFrame(rows)


def _behavior_recovery(per_seed: pd.DataFrame) -> pd.DataFrame:
    indexed = per_seed.set_index(["variant", "seed"])
    rows = []
    for seed in SEEDS:
        for metric in ("f1", "pr_auc", "roc_auc"):
            value = lambda variant: float(indexed.loc[(variant, seed), metric])
            gain = value("B") - value("C")
            row = {"seed": seed, "metric": metric, "gt_behavior_gain": gain, "raw_pose_recovery": value("D") - value("C"), "biu_v2_recovery": value("E") - value("C"), "biu_v1_recovery": value("F") - value("C")}
            row["raw_pose_recovery_fraction"] = row["raw_pose_recovery"] / gain if gain > 0 else np.nan
            row["biu_v2_recovery_fraction"] = row["biu_v2_recovery"] / gain if gain > 0 else np.nan
            row["biu_v1_recovery_fraction"] = row["biu_v1_recovery"] / gain if gain > 0 else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def finalize_from_predictions(predictions: Mapping[tuple[str, int], pd.DataFrame]) -> dict[str, object]:
    expected = {(variant, seed) for variant in VARIANTS for seed in SEEDS}
    if set(predictions) != expected:
        raise ValueError("Finalization requires the exact 18 prediction identities.")
    per_seed_rows = []
    diagnostic_rows = []
    high_confidence = []
    for (variant, seed), frame in sorted(predictions.items()):
        labels = frame["label"].to_numpy(dtype=np.int64)
        probabilities = frame["probability"].to_numpy(dtype=np.float64)
        metrics = compute_metrics(labels, probabilities)
        per_seed_rows.append({"variant": variant, "seed": seed, **metrics})
        diagnostic_rows.append({"variant": variant, "seed": seed, **probability_diagnostics(labels, probabilities)})
        mask = ((labels == 0) & (probabilities >= 0.9)) | ((labels == 1) & (probabilities <= 0.1))
        cases = frame.loc[mask, ["sequence_id", "video_id", "ped_id", "target_frame", "label", "probability", "prediction"]].copy()
        cases["variant"] = variant
        cases["seed"] = seed
        cases["error_type"] = np.where(cases["label"].to_numpy() == 0, "FP", "FN")
        high_confidence.append(cases)
    metrics_root = EXPERIMENT_ROOT / "metrics"
    per_seed = pd.DataFrame(per_seed_rows)
    _atomic_csv(per_seed, metrics_root / "per_seed_results.csv")
    _atomic_csv(pd.DataFrame(diagnostic_rows), metrics_root / "probability_diagnostics.csv")
    _atomic_csv(pd.concat(high_confidence, ignore_index=True), metrics_root / "high_confidence_errors.csv")
    numeric_metrics = ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc")
    summary_rows = []
    for variant in VARIANTS:
        group = per_seed[per_seed["variant"] == variant]
        for metric in numeric_metrics:
            summary_rows.append({"variant": variant, "metric": metric, "mean": float(group[metric].mean()), "std": float(group[metric].std(ddof=1))})
    _atomic_csv(pd.DataFrame(summary_rows), metrics_root / "summary_results.csv")
    _atomic_csv(_paired_deltas(per_seed), metrics_root / "paired_deltas.csv")
    _atomic_csv(_behavior_recovery(per_seed), metrics_root / "behavior_recovery.csv")
    first = next(iter(predictions.values()))
    cohort = pd.DataFrame([{"rows": len(first), "positive_labels": int(first["label"].sum()), "negative_labels": int((first["label"] == 0).sum()), "videos": int(first["video_id"].nunique()), "tracks": int(first[["video_id", "ped_id"]].drop_duplicates().shape[0]), "observation_frames": 30, "prediction_offset_raw_frames": 1, "threshold": FIXED_THRESHOLD}])
    _atomic_csv(cohort, metrics_root / "cohort_summary.csv")
    coverage = pd.DataFrame([{"stream": name, "complete_windows": len(first), "total_windows": len(first), "coverage_fraction": 1.0} for name in ("bbox", "ped_action", "ped_look", "ego_speed", "ego_acc", "static_context", "yolop", "raw_pose", "biu_v2", "biu_v1")])
    _atomic_csv(coverage, metrics_root / "modality_coverage.csv")
    historical = pd.DataFrame([{"source": "Samalouty thesis historical reference only", "accuracy": 0.9186, "precision": 0.8290, "recall": 0.9112, "f1": 0.8616, "roc_auc": 0.9505, "threshold": 0.6621, "protocol_parity": False}])
    _atomic_csv(historical, metrics_root / "historical_reference.csv")
    artifacts = {}
    for path in sorted(metrics_root.glob("*.csv")):
        artifacts[str(path)] = sha256_file(path)
    for variant, seed in sorted(expected):
        path, marker = _prediction_paths(variant, seed)
        artifacts[str(path)] = sha256_file(path)
        artifacts[str(marker)] = sha256_file(marker)
    completion = {"phase": "zero_shot_evaluation", "complete": True, "prediction_count": 18, "threshold": FIXED_THRESHOLD, "artifacts": artifacts}
    marker_path = EXPERIMENT_ROOT / "EXPERIMENT_COMPLETE"
    marker_path.write_text(json.dumps(completion, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return completion


def finalize_saved_predictions() -> dict[str, object]:
    authenticate_audit_marker(EXPERIMENT_ROOT)
    frames = {}
    for record in authenticate_exp048_checkpoints():
        frame = _load_authenticated_prediction(
            record.variant, record.seed, record.checkpoint_sha256
        )
        if frame is None:
            raise FileNotFoundError(
                f"Missing saved prediction for {record.variant}{record.seed}."
            )
        frames[(record.variant, record.seed)] = frame
    return finalize_from_predictions(frames)
