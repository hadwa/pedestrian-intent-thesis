"""Run supervised BIU-only tabular student baselines.

This suite trains classical tabular models on the frozen BIU feature tables only.
It never regenerates splits, never changes labels, and selects thresholds on the
validation split only.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.benchmark.metrics import compute_binary_metrics, find_best_threshold_on_validation
from src.students.datasets import StudentLoaderReport, load_preprocessed_frames
from src.students.train_student import read_yaml


def _positive_logit(prob: np.ndarray) -> np.ndarray:
    clipped = np.clip(prob.astype(float), 1e-7, 1.0 - 1e-7)
    return np.log(clipped / (1.0 - clipped))


def _predict_positive_proba(model: Any, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)
        if proba.ndim == 2:
            return np.asarray(proba[:, 1], dtype=float)
        return np.asarray(proba, dtype=float).ravel()
    if hasattr(model, "decision_function"):
        score = np.asarray(model.decision_function(x), dtype=float)
        return 1.0 / (1.0 + np.exp(-score))
    raise TypeError(f"Model {type(model).__name__} does not expose predict_proba or decision_function.")


def _sanitize_model_name(name: str) -> str:
    return name.lower().replace(" ", "_").replace("-", "_")


def _build_model_specs(random_seed: int, positive_rate: float) -> tuple[list[tuple[str, Any]], list[str]]:
    specs: list[tuple[str, Any]] = []
    warnings: list[str] = []

    specs.append(
        (
            "logistic_regression",
            Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            max_iter=1000,
                            class_weight="balanced",
                            random_state=random_seed,
                            solver="lbfgs",
                        ),
                    ),
                ]
            ),
        )
    )

    specs.append(
        (
            "random_forest",
            RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=random_seed,
            ),
        )
    )

    if importlib.util.find_spec("lightgbm") is not None:
        from lightgbm import LGBMClassifier

        specs.append(
            (
                "lightgbm",
                LGBMClassifier(
                    objective="binary",
                    n_estimators=500,
                    learning_rate=0.05,
                    num_leaves=31,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    class_weight="balanced",
                    random_state=random_seed,
                    n_jobs=-1,
                    verbose=-1,
                ),
            )
        )
    else:
        warnings.append("LightGBM is not installed; skipped lightgbm. Add lightgbm to requirements and install it to run this model.")

    if importlib.util.find_spec("xgboost") is not None:
        from xgboost import XGBClassifier

        scale_pos_weight = (1.0 - positive_rate) / positive_rate if positive_rate > 0 else 1.0
        specs.append(
            (
                "xgboost",
                XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    n_estimators=500,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    min_child_weight=1.0,
                    reg_lambda=1.0,
                    tree_method="hist",
                    random_state=random_seed,
                    n_jobs=-1,
                    scale_pos_weight=scale_pos_weight,
                ),
            )
        )
    else:
        warnings.append("XGBoost is not installed; skipped xgboost. Add xgboost to requirements and install it to run this model.")

    warnings.append("MLP was not run in this first tabular suite; Logistic Regression, Random Forest, LightGBM, and XGBoost were prioritized.")
    return specs, warnings


def _write_predictions(
    output_path: Path,
    model_name: str,
    split: str,
    frame: pd.DataFrame,
    prob: np.ndarray,
    threshold: float,
) -> None:
    pred = (prob >= threshold).astype(int)
    out = pd.DataFrame(
        {
            "sequence_id": frame["sequence_id"].astype(str).to_numpy(),
            "split": split,
            "label": frame["label"].astype(int).to_numpy(),
            "model_name": model_name,
            "student_logit": _positive_logit(prob),
            "student_prob": prob.astype(float),
            "student_pred": pred.astype(int),
            "selected_threshold": float(threshold),
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)


def _metrics_row(
    model_name: str,
    feature_count: int,
    row_counts: dict[str, int],
    random_seed: int,
    threshold: float,
    val_metrics: dict[str, float],
    test_metrics: dict[str, float],
) -> dict[str, Any]:
    return {
        "model_name": model_name,
        "feature_set": "biu",
        "train_rows": row_counts["train"],
        "val_rows": row_counts["val"],
        "test_rows": row_counts["test"],
        "feature_count": feature_count,
        "val_threshold": float(threshold),
        "val_f1": val_metrics["f1"],
        "val_precision": val_metrics["precision"],
        "val_recall": val_metrics["recall"],
        "val_accuracy": val_metrics["accuracy"],
        "val_roc_auc": val_metrics["roc_auc"],
        "val_pr_auc": val_metrics["pr_auc"],
        "test_f1": test_metrics["f1"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "test_accuracy": test_metrics["accuracy"],
        "test_roc_auc": test_metrics["roc_auc"],
        "test_pr_auc": test_metrics["pr_auc"],
        "random_seed": random_seed,
    }


def _write_report(
    path: Path,
    config_path: Path,
    metrics_df: pd.DataFrame,
    loader_report: StudentLoaderReport,
    warnings: list[str],
    timings: dict[str, float],
    output_files: list[str],
) -> None:
    lines = [
        "# Student BIU Supervised Report",
        "",
        "## Summary",
        "",
        "Ran supervised BIU-only tabular student baselines on frozen PIE benchmark splits.",
        "No KD was implemented or run.",
        "The old repository was not modified.",
        "",
        "## Config",
        "",
        f"- Config: `{config_path}`",
        "- Feature set: `biu`",
        "- Frozen splits: `data/splits/train_sequences.csv`, `data/splits/val_sequences.csv`, `data/splits/test_sequences.csv`",
        "- Feature tables: `data/processed/features/biu_train.csv`, `biu_val.csv`, `biu_test.csv`",
        "",
        "## Loader Validation",
        "",
    ]
    for msg in loader_report.validation_messages:
        lines.append(f"- {msg}")
    lines.extend(
        [
            "",
            "## Preprocessing",
            "",
            "- Train-only median imputation was fit on train and applied to val/test.",
            "- Logistic Regression uses train-fitted standard scaling inside its pipeline.",
            "- Tree models use train-only imputation and no scaling.",
            "- No rows were dropped.",
            "",
            "## Row And Feature Counts",
            "",
            f"- Train rows: `{loader_report.row_counts.get('train')}`",
            f"- Validation rows: `{loader_report.row_counts.get('val')}`",
            f"- Test rows: `{loader_report.row_counts.get('test')}`",
            f"- Feature count: `{loader_report.feature_count}`",
            "",
            "## NaN Counts",
            "",
            f"- Before imputation: `{loader_report.nan_before}`",
            f"- After imputation: `{loader_report.nan_after}`",
            "",
            "## Models Run",
            "",
        ]
    )
    for model_name in metrics_df["model_name"].tolist():
        lines.append(f"- `{model_name}`")
    lines.extend(["", "## Warnings / Skips", ""])
    for warning in warnings:
        lines.append(f"- {warning}")
    lines.extend(["", "## Metrics", ""])
    lines.append("```text")
    lines.append(metrics_df.to_string(index=False))
    lines.append("```")
    lines.extend(["", "## Training Runtime Seconds", ""])
    for model_name, seconds in timings.items():
        lines.append(f"- `{model_name}`: `{seconds:.3f}`")
    lines.extend(["", "## Output Files", ""])
    for output_file in output_files:
        lines.append(f"- `{output_file}`")
    lines.extend(
        [
            "",
            "## Next Recommended Step",
            "",
            "Review `results/student_biu_supervised_metrics.csv` and the validation/test prediction files.",
            "If the supervised BIU baseline looks sane, the next controlled step is to add the BIU+BBox and BIU+Keypoint+BBox feature tables/loaders before comparing student ablations.",
            "Do not tune on test.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_suite(config_path: Path, output_dir: Path) -> pd.DataFrame:
    config = read_yaml(config_path)
    training_cfg = config.get("training", {})
    random_seed = int(training_cfg.get("seed", 42))

    loader_report = StudentLoaderReport(config_path=str(config_path), student_name=str(config.get("student_name", "BIU")))
    frames, feature_columns, _metadata_columns, _medians = load_preprocessed_frames(config, report=loader_report, normalize=False)

    x_train = frames["train"][feature_columns].astype(float).to_numpy()
    y_train = frames["train"]["label"].astype(int).to_numpy()
    x_val = frames["val"][feature_columns].astype(float).to_numpy()
    y_val = frames["val"]["label"].astype(int).to_numpy()
    x_test = frames["test"][feature_columns].astype(float).to_numpy()
    y_test = frames["test"]["label"].astype(int).to_numpy()

    model_specs, warnings = _build_model_specs(random_seed=random_seed, positive_rate=float(np.mean(y_train)))
    rows: list[dict[str, Any]] = []
    timings: dict[str, float] = {}
    output_files: list[str] = []

    for model_name, model in model_specs:
        start = time.perf_counter()
        model.fit(x_train, y_train)
        timings[model_name] = time.perf_counter() - start

        val_prob = _predict_positive_proba(model, x_val)
        test_prob = _predict_positive_proba(model, x_test)
        threshold = find_best_threshold_on_validation(y_val, val_prob, metric="f1")
        val_metrics = compute_binary_metrics(y_val, val_prob, threshold)
        test_metrics = compute_binary_metrics(y_test, test_prob, threshold)
        rows.append(_metrics_row(model_name, len(feature_columns), loader_report.row_counts, random_seed, threshold, val_metrics, test_metrics))

        safe_name = _sanitize_model_name(model_name)
        val_path = output_dir / f"student_biu_supervised_val_predictions_{safe_name}.csv"
        test_path = output_dir / f"student_biu_supervised_test_predictions_{safe_name}.csv"
        _write_predictions(val_path, model_name, "val", frames["val"], val_prob, threshold)
        _write_predictions(test_path, model_name, "test", frames["test"], test_prob, threshold)
        output_files.extend([str(val_path), str(test_path)])

    metrics_df = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "student_biu_supervised_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    output_files.insert(0, str(metrics_path))

    report_path = Path("STUDENT_BIU_SUPERVISED_REPORT.md")
    _write_report(report_path, config_path, metrics_df, loader_report, warnings, timings, output_files)
    return metrics_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run supervised BIU-only tabular student model suite.")
    parser.add_argument("--config", type=Path, default=Path("configs/student_biu.yaml"))
    parser.add_argument("--output_dir", type=Path, default=Path("results"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics_df = run_suite(args.config, args.output_dir)
    print(json.dumps({"models": metrics_df["model_name"].tolist(), "metrics_csv": str(args.output_dir / "student_biu_supervised_metrics.csv"), "report": "STUDENT_BIU_SUPERVISED_REPORT.md"}, indent=2))


if __name__ == "__main__":
    main()
