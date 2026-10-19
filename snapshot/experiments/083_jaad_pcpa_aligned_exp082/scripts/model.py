"""Frozen Exp083 XGBoost training and validation."""

from __future__ import annotations

import importlib.util
import hashlib
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


SCRIPT_DIR = Path(__file__).resolve().parent


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CONTRACT = _load(SCRIPT_DIR / "contract.py", "exp083_contract_for_model")
FEATURES = _load(SCRIPT_DIR / "features.py", "exp083_features_for_model")
PARENT_MODEL = FEATURES.PARENT_MODEL


def build_xgb(seed: int, labels: np.ndarray):
    registration = CONTRACT.load_registration()
    if int(seed) not in registration["seeds"]:
        raise ValueError("Exp083 seed is not frozen")
    pipeline = PARENT_MODEL.build_xgb(int(seed), np.asarray(labels, dtype=int))
    params = pipeline.named_steps["model"].get_params()
    frozen = registration["xgboost"]
    for name in (
        "n_estimators",
        "max_depth",
        "learning_rate",
        "subsample",
        "colsample_bytree",
        "reg_lambda",
        "n_jobs",
        "objective",
        "eval_metric",
    ):
        if params[name] != frozen[name]:
            raise RuntimeError(f"Exp083 XGBoost parameter changed: {name}")
    return pipeline


def _validated(labels: np.ndarray, probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if (
        labels.ndim != 1
        or probabilities.shape != labels.shape
        or set(np.unique(labels)) != {0, 1}
        or not np.isfinite(probabilities).all()
        or np.any((probabilities < 0.0) | (probabilities > 1.0))
    ):
        raise ValueError("Exp083 metrics require aligned binary probabilities")
    return labels, probabilities


def metrics_at_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    labels, probabilities = _validated(labels, probabilities)
    threshold = float(threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Exp083 classification threshold must be in [0,1]")
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(
            precision_score(labels, predictions, zero_division=0)
        ),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "pr_auc": float(average_precision_score(labels, probabilities)),
        "brier": float(brier_score_loss(labels, probabilities)),
        "log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "high_confidence_fp": int(
            ((labels == 0) & (probabilities >= 0.9)).sum()
        ),
        "high_confidence_fn": int(
            ((labels == 1) & (probabilities <= 0.1)).sum()
        ),
    }


def select_validation_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    labels, probabilities = _validated(labels, probabilities)
    unique = np.unique(probabilities)
    midpoints = (unique[:-1] + unique[1:]) / 2.0
    candidates = np.unique(np.concatenate(([0.0], midpoints, [1.0])))
    scored = [
        (
            float(
                f1_score(
                    labels,
                    (probabilities >= threshold).astype(int),
                    zero_division=0,
                )
            ),
            float(threshold),
        )
        for threshold in candidates
    ]
    best_f1 = max(score for score, _ in scored)
    tied = [threshold for score, threshold in scored if score == best_f1]
    return min(tied, key=lambda threshold: (abs(threshold - 0.5), threshold))


def load_development_cache(
    root: Path | str,
    cohort: str,
    split: str,
) -> dict[str, np.ndarray]:
    if cohort not in {"jaad_beh", "jaad_all"}:
        raise ValueError("Exp083 cohort is invalid")
    if split not in {"train", "val"}:
        raise ValueError("Exp083 development loader refuses test access")
    path = Path(root) / "cache" / f"{cohort}_{split}_features.npz"
    with np.load(path, allow_pickle=False) as archive:
        expected = {
            "sequence_id",
            "source_sequence_id",
            "track_id",
            "label",
            "X",
            "predictor_columns",
        }
        if set(archive.files) != expected:
            raise RuntimeError("Exp083 feature cache ABI changed")
        result = {name: np.asarray(archive[name]) for name in expected}
    columns = tuple(result["predictor_columns"].astype(str))
    if columns != FEATURES.predictor_columns():
        raise RuntimeError("Exp083 feature-cache columns changed")
    rows = len(result["sequence_id"])
    if (
        result["X"].shape != (rows, 242)
        or result["label"].shape != (rows,)
        or not np.isfinite(result["X"]).all()
        or len(set(result["sequence_id"].astype(str))) != rows
    ):
        raise RuntimeError("Exp083 feature cache values changed")
    return result


def load_test_cache(root: Path | str, cohort: str) -> dict[str, np.ndarray]:
    if cohort not in {"jaad_beh", "jaad_all"}:
        raise ValueError("Exp083 cohort is invalid")
    path = Path(root) / "cache" / f"{cohort}_test_features.npz"
    with np.load(path, allow_pickle=False) as archive:
        result = {name: np.asarray(archive[name]) for name in archive.files}
    expected = {
        "sequence_id",
        "source_sequence_id",
        "track_id",
        "label",
        "X",
        "predictor_columns",
    }
    if set(result) != expected:
        raise RuntimeError("Exp083 test feature cache ABI changed")
    if tuple(result["predictor_columns"].astype(str)) != FEATURES.predictor_columns():
        raise RuntimeError("Exp083 test feature columns changed")
    rows = len(result["sequence_id"])
    if (
        result["X"].shape != (rows, 242)
        or result["label"].shape != (rows,)
        or not np.isfinite(result["X"]).all()
        or len(set(result["sequence_id"].astype(str))) != rows
    ):
        raise RuntimeError("Exp083 test feature cache values changed")
    return result


def summarize_results(
    rows: pd.DataFrame,
    *,
    metrics: tuple[str, ...],
) -> pd.DataFrame:
    records = []
    for keys, group in rows.groupby(
        ["cohort", "threshold_policy"], sort=True
    ):
        record = {"cohort": keys[0], "threshold_policy": keys[1], "seeds": len(group)}
        for metric in metrics:
            record[f"{metric}_mean"] = float(group[metric].mean())
            record[f"{metric}_std"] = float(group[metric].std(ddof=1))
        records.append(record)
    return pd.DataFrame(records)


def _row_digest(cache: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for sequence_id, label in zip(cache["sequence_id"], cache["label"]):
        digest.update(f"{str(sequence_id)}\t{int(label)}\n".encode("utf-8"))
    return digest.hexdigest()


def _atomic_joblib(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".joblib", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        joblib.dump(value, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run_train_validate(root: Path | str) -> dict[str, Any]:
    root = Path(root)
    marker = CONTRACT.authenticate_phase(root, "prepare-development")
    if marker.get("test_accessed") is not False:
        raise RuntimeError("Exp083 prepare-development accessed test data")
    if any(root.glob("cache/*_test_features.npz")):
        raise RuntimeError("Exp083 test feature cache exists before training freeze")

    outputs = []
    metric_rows = []
    threshold_rows = []
    prediction_parts = []
    model_records = []
    fixed_threshold = float(CONTRACT.load_registration()["fixed_threshold"])
    for cohort in ("jaad_beh", "jaad_all"):
        train = load_development_cache(root, cohort, "train")
        validation = load_development_cache(root, cohort, "val")
        labels_train = train["label"].astype(int)
        labels_validation = validation["label"].astype(int)
        for seed in CONTRACT.load_registration()["seeds"]:
            pipeline = build_xgb(int(seed), labels_train)
            pipeline.fit(train["X"], labels_train)
            probabilities = pipeline.predict_proba(validation["X"])[:, 1]
            selected = select_validation_threshold(
                labels_validation, probabilities
            )
            for policy, threshold in (
                ("fixed_0p5", fixed_threshold),
                ("validation_selected", selected),
            ):
                metric_rows.append(
                    {
                        "cohort": cohort,
                        "seed": int(seed),
                        "split": "validation",
                        "threshold_policy": policy,
                        **metrics_at_threshold(
                            labels_validation, probabilities, threshold
                        ),
                    }
                )
            threshold_rows.append(
                {
                    "cohort": cohort,
                    "seed": int(seed),
                    "threshold": selected,
                    "validation_f1": metrics_at_threshold(
                        labels_validation, probabilities, selected
                    )["f1"],
                }
            )
            prediction_parts.append(
                pd.DataFrame(
                    {
                        "cohort": cohort,
                        "split": "validation",
                        "seed": int(seed),
                        "sequence_id": validation["sequence_id"].astype(str),
                        "source_sequence_id": validation[
                            "source_sequence_id"
                        ].astype(str),
                        "track_id": validation["track_id"].astype(str),
                        "label": labels_validation,
                        "probability": probabilities,
                        "prediction_fixed_0p5": (
                            probabilities >= fixed_threshold
                        ).astype(int),
                        "validation_selected_threshold": selected,
                        "prediction_validation_selected": (
                            probabilities >= selected
                        ).astype(int),
                    }
                )
            )
            model_path = root / "models" / f"{cohort}_seed{seed}.joblib"
            _atomic_joblib(model_path, pipeline)
            outputs.append(model_path)
            record = CONTRACT.artifact_record(model_path)
            record.update(
                {
                    "cohort": cohort,
                    "seed": int(seed),
                    "training_rows": len(labels_train),
                    "training_row_digest": _row_digest(train),
                }
            )
            model_records.append(record)
            print(
                f"Exp083 trained cohort={cohort} seed={seed} "
                f"validation_f1_0p5={metric_rows[-2]['f1']:.6f} "
                f"validation_pr_auc={metric_rows[-2]['pr_auc']:.6f}",
                flush=True,
            )

    metrics = pd.DataFrame(metric_rows)
    thresholds = pd.DataFrame(threshold_rows)
    predictions = pd.concat(prediction_parts, ignore_index=True)
    metrics_path = root / "metrics/validation_per_seed_results.csv"
    thresholds_path = root / "metrics/validation_thresholds.csv"
    predictions_path = root / "predictions/validation_all_seeds.csv"
    models_path = root / "metrics/trained_model_manifest.json"
    CONTRACT.atomic_write_text(metrics_path, metrics.to_csv(index=False))
    CONTRACT.atomic_write_text(thresholds_path, thresholds.to_csv(index=False))
    CONTRACT.atomic_write_text(predictions_path, predictions.to_csv(index=False))
    CONTRACT.atomic_write_json(
        models_path,
        {
            "experiment": "083_jaad_pcpa_aligned_exp082",
            "test_accessed": False,
            "models": model_records,
        },
    )
    outputs.extend([metrics_path, thresholds_path, predictions_path, models_path])
    phase = CONTRACT.publish_phase(
        root,
        "train-validate",
        outputs,
        test_accessed=False,
        test_features_accessed=False,
        test_metrics_accessed=False,
        models=len(model_records),
        validation_prediction_rows=len(predictions),
    )
    return {
        "marker": str(phase),
        "models": len(model_records),
        "validation_prediction_rows": len(predictions),
    }


def run_evaluate(root: Path | str) -> dict[str, Any]:
    root = Path(root)
    prepared = CONTRACT.authenticate_phase(root, "prepare-test")
    CONTRACT.authenticate_phase(root, "train-validate")
    if (
        prepared.get("test_features_accessed") is not True
        or prepared.get("test_metrics_accessed") is not False
    ):
        raise RuntimeError("Exp083 prepare-test state changed")

    thresholds = pd.read_csv(root / "metrics/validation_thresholds.csv")
    metric_rows = []
    prediction_parts = []
    fixed = float(CONTRACT.load_registration()["fixed_threshold"])
    for cohort in ("jaad_beh", "jaad_all"):
        cache = load_test_cache(root, cohort)
        labels = cache["label"].astype(int)
        for seed in CONTRACT.load_registration()["seeds"]:
            selected_row = thresholds.loc[
                thresholds["cohort"].eq(cohort)
                & thresholds["seed"].eq(int(seed))
            ]
            if len(selected_row) != 1:
                raise RuntimeError("Exp083 validation threshold identity changed")
            selected = float(selected_row.iloc[0]["threshold"])
            model_path = root / "models" / f"{cohort}_seed{seed}.joblib"
            pipeline = joblib.load(model_path)
            probabilities = pipeline.predict_proba(cache["X"])[:, 1]
            for policy, threshold in (
                ("fixed_0p5", fixed),
                ("validation_selected", selected),
            ):
                metric_rows.append(
                    {
                        "cohort": cohort,
                        "seed": int(seed),
                        "split": "test",
                        "threshold_policy": policy,
                        **metrics_at_threshold(labels, probabilities, threshold),
                    }
                )
            prediction_parts.append(
                pd.DataFrame(
                    {
                        "cohort": cohort,
                        "split": "test",
                        "seed": int(seed),
                        "sequence_id": cache["sequence_id"].astype(str),
                        "source_sequence_id": cache[
                            "source_sequence_id"
                        ].astype(str),
                        "track_id": cache["track_id"].astype(str),
                        "label": labels,
                        "probability": probabilities,
                        "prediction_fixed_0p5": (probabilities >= fixed).astype(int),
                        "validation_selected_threshold": selected,
                        "prediction_validation_selected": (
                            probabilities >= selected
                        ).astype(int),
                    }
                )
            )
            print(
                f"Exp083 evaluated cohort={cohort} seed={seed} "
                f"test_f1_0p5={metric_rows[-2]['f1']:.6f} "
                f"test_pr_auc={metric_rows[-2]['pr_auc']:.6f}",
                flush=True,
            )

    per_seed = pd.DataFrame(metric_rows)
    metric_names = (
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
        "brier",
        "log_loss",
        "tn",
        "fp",
        "fn",
        "tp",
        "high_confidence_fp",
        "high_confidence_fn",
    )
    summary = summarize_results(per_seed, metrics=metric_names)
    predictions = pd.concat(prediction_parts, ignore_index=True)
    paths = {
        "per_seed": root / "metrics/per_seed_results.csv",
        "summary": root / "metrics/summary_results.csv",
        "fixed": root / "metrics/fixed_0p5_results.csv",
        "selected": root / "metrics/validation_selected_threshold_results.csv",
        "predictions": root / "predictions/test_all_seeds.csv",
    }
    CONTRACT.atomic_write_text(paths["per_seed"], per_seed.to_csv(index=False))
    CONTRACT.atomic_write_text(paths["summary"], summary.to_csv(index=False))
    CONTRACT.atomic_write_text(
        paths["fixed"],
        per_seed.loc[per_seed["threshold_policy"].eq("fixed_0p5")].to_csv(
            index=False
        ),
    )
    CONTRACT.atomic_write_text(
        paths["selected"],
        per_seed.loc[
            per_seed["threshold_policy"].eq("validation_selected")
        ].to_csv(index=False),
    )
    CONTRACT.atomic_write_text(
        paths["predictions"], predictions.to_csv(index=False)
    )
    phase = CONTRACT.publish_phase(
        root,
        "evaluate",
        list(paths.values()),
        test_accessed=True,
        test_features_accessed=True,
        test_metrics_accessed=True,
        cohorts=["jaad_beh", "jaad_all"],
        models=6,
    )
    return {
        "marker": str(phase),
        "per_seed_rows": len(per_seed),
        "prediction_rows": len(predictions),
    }
