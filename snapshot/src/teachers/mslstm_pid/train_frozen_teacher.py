"""Train frozen-row T1/T2 tabular teacher baselines."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from src.benchmark.metrics import compute_binary_metrics, find_best_threshold_on_validation
from src.students.datasets import StudentLoaderReport, load_preprocessed_frames
from src.students.train_biu_supervised_suite import _build_model_specs, _predict_positive_proba, _sanitize_model_name


def _build_config(variant_name: str, tables_dir: Path) -> dict[str, Any]:
    feature_name = f"teacher_{variant_name.lower()}"
    return {
        "student_name": f"MSLSTM_FROZEN_{variant_name}",
        "splits_dir": "data/splits",
        "features": [feature_name],
        "feature_tables": {
            feature_name: {
                "train": str(tables_dir / f"mslstm_frozen_{variant_name}_train.csv"),
                "val": str(tables_dir / f"mslstm_frozen_{variant_name}_val.csv"),
                "test": str(tables_dir / f"mslstm_frozen_{variant_name}_test.csv"),
            }
        },
        "training": {"seed": 42},
    }


def _write_predictions(path: Path, variant: str, model_name: str, frame: pd.DataFrame, prob: np.ndarray, threshold: float) -> None:
    pred = (prob >= threshold).astype(int)
    out = pd.DataFrame(
        {
            "sequence_id": frame["sequence_id"].astype(str).to_numpy(),
            "label": frame["label"].astype(int).to_numpy(),
            "p_cross": prob.astype(float),
            "pred": pred.astype(int),
            "threshold": float(threshold),
            "model_name": model_name,
            "variant_name": variant,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def _metrics_row(variant: str, model_name: str, feature_count: int, row_counts: dict[str, int], threshold: float, val_metrics: dict[str, float], test_metrics: dict[str, float], val_cm: tuple[int, int, int, int], test_cm: tuple[int, int, int, int]) -> dict[str, Any]:
    return {
        "variant_name": variant,
        "model_name": model_name,
        "train_rows": row_counts["train"],
        "val_rows": row_counts["val"],
        "test_rows": row_counts["test"],
        "feature_count": feature_count,
        "selected_threshold": float(threshold),
        "positive_ratio_train": float("nan"),
        "positive_ratio_val": float("nan"),
        "positive_ratio_test": float("nan"),
        "val_f1": val_metrics["f1"],
        "val_precision": val_metrics["precision"],
        "val_recall": val_metrics["recall"],
        "val_accuracy": val_metrics["accuracy"],
        "val_roc_auc": val_metrics["roc_auc"],
        "val_pr_auc": val_metrics["pr_auc"],
        "val_tn": val_cm[0],
        "val_fp": val_cm[1],
        "val_fn": val_cm[2],
        "val_tp": val_cm[3],
        "test_f1": test_metrics["f1"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "test_accuracy": test_metrics["accuracy"],
        "test_roc_auc": test_metrics["roc_auc"],
        "test_pr_auc": test_metrics["pr_auc"],
        "test_tn": test_cm[0],
        "test_fp": test_cm[1],
        "test_fn": test_cm[2],
        "test_tp": test_cm[3],
    }


def _confusion_tuple(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> tuple[int, int, int, int]:
    pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true.astype(int), pred.astype(int), labels=[0, 1]).ravel()
    return int(tn), int(fp), int(fn), int(tp)


def _run_variant(variant: str, tables_dir: Path, output_dir: Path) -> tuple[pd.DataFrame, dict[str, Any], StudentLoaderReport]:
    config = _build_config(variant, tables_dir)
    report = StudentLoaderReport(config_path=f"in_memory:{variant}", student_name=config["student_name"])
    frames, feature_columns, _metadata_columns, _medians = load_preprocessed_frames(config, report=report, normalize=False)
    x_train = frames["train"][feature_columns].astype(float).to_numpy()
    y_train = frames["train"]["label"].astype(int).to_numpy()
    x_val = frames["val"][feature_columns].astype(float).to_numpy()
    y_val = frames["val"]["label"].astype(int).to_numpy()
    x_test = frames["test"][feature_columns].astype(float).to_numpy()
    y_test = frames["test"]["label"].astype(int).to_numpy()

    model_specs, warnings = _build_model_specs(random_seed=42, positive_rate=float(np.mean(y_train)))
    rows: list[dict[str, Any]] = []
    timings: dict[str, float] = {}
    for model_name, model in model_specs:
        start = time.perf_counter()
        model.fit(x_train, y_train)
        timings[model_name] = time.perf_counter() - start
        val_prob = _predict_positive_proba(model, x_val)
        test_prob = _predict_positive_proba(model, x_test)
        threshold = find_best_threshold_on_validation(y_val, val_prob, metric="f1")
        val_metrics = compute_binary_metrics(y_val, val_prob, threshold)
        test_metrics = compute_binary_metrics(y_test, test_prob, threshold)
        val_cm = _confusion_tuple(y_val, val_prob, threshold)
        test_cm = _confusion_tuple(y_test, test_prob, threshold)
        row = _metrics_row(variant, model_name, len(feature_columns), report.row_counts, threshold, val_metrics, test_metrics, val_cm, test_cm)
        row["positive_ratio_train"] = float(np.mean(y_train))
        row["positive_ratio_val"] = float(np.mean(y_val))
        row["positive_ratio_test"] = float(np.mean(y_test))
        rows.append(row)
        safe_model = _sanitize_model_name(model_name)
        _write_predictions(output_dir / f"val_predictions_{variant}_{safe_model}.csv", variant, model_name, frames["val"], val_prob, threshold)
        _write_predictions(output_dir / f"test_predictions_{variant}_{safe_model}.csv", variant, model_name, frames["test"], test_prob, threshold)
    return pd.DataFrame(rows), {"warnings": warnings, "timings": timings}, report


def write_reports(path: Path, metrics_df: pd.DataFrame, variant_meta: dict[str, dict[str, Any]], reports: dict[str, StudentLoaderReport]) -> None:
    lines = [
        "# MSLSTM Frozen T1/T2 Supervised Report",
        "",
        "## Protocol",
        "",
        "- Frozen observation length: `16` consecutive frames.",
        "- Frozen labels from existing split CSVs.",
        "- Positive windows are the existing frozen `30-60` TTE benchmark rows.",
        "- This is not the original local MSLSTM-PID `pred_len=1` recurrent protocol.",
        "",
        "## Included Streams",
        "",
        "- `T1`: bbox, ego_speed, ego_acc, ego_gyro",
        "- `T2`: T1 + traffic_light + sanitized static scene context" + (" + YOLOP" if variant_meta["T2"]["include_yolop"] else " (YOLOP skipped)"),
        "- Sanitized static scene context fields: `signalized`, `intersection`, `traffic_direction`, `num_lanes_bucket`",
        "- YOLOP usage in this run: `" + ("included" if variant_meta["T2"]["include_yolop"] else "skipped") + "`",
        "",
        "## Excluded Streams",
        "",
        "- `ped_action`, `ped_look`, `ped_occlusion`: manual pedestrian behavior/visibility annotations excluded from the main teacher.",
        "- `pose`: excluded from the main teacher.",
        "- static `age` and `gender`: excluded from the main teacher.",
        "- static future/event metadata such as `crossing`, `intention_prob`, `crossing_point`, `critical_point`, `exp_start_point`: excluded.",
        "",
        "## Metrics",
        "",
        "```text",
        metrics_df.to_string(index=False),
        "```",
        "",
        "## Class Imbalance Handling",
        "",
        "- `logistic_regression`: `class_weight=balanced` with train-only imputation and train-fitted scaling in the model pipeline.",
        "- `random_forest`: `class_weight=balanced_subsample` with train-only imputation.",
        "- `lightgbm`: `class_weight=balanced` with train-only imputation.",
        "- `xgboost`: train-set `scale_pos_weight` derived from the frozen train positive rate.",
        "- Threshold selection used validation F1 only; the selected threshold was then applied once to test.",
        "",
        "## Best By Variant",
        "",
    ]
    for variant in ["T1", "T2"]:
        subset = metrics_df.loc[metrics_df["variant_name"] == variant].copy()
        by_val = subset.loc[subset["val_f1"].idxmax()]
        by_test = subset.loc[subset["test_f1"].idxmax()]
        lines.append(f"- `{variant}` best validation F1: `{by_val['model_name']}` val_f1=`{by_val['val_f1']:.6f}` test_f1=`{by_val['test_f1']:.6f}`")
        lines.append(f"- `{variant}` best test F1: `{by_test['model_name']}` test_f1=`{by_test['test_f1']:.6f}` test_pr_auc=`{by_test['test_pr_auc']:.6f}`")
    lines.extend(["", "## Loader Diagnostics", ""])
    for variant, report in reports.items():
        lines.append(f"- `{variant}` row counts: `{report.row_counts}` feature_count=`{report.feature_count}`")
        lines.append(f"- `{variant}` NaNs before imputation: `{report.nan_before}`")
        lines.append(f"- `{variant}` NaNs after imputation: `{report.nan_after}`")
    lines.extend(["", "## Interpretation", ""])
    lines.append("These variants test whether frozen-row context streams provide stronger early anticipation than lightweight pose/BIU-centric students under the strict 30-60 TTE setup.")
    lines.append("`T2` is the main scene-context teacher in this run because it includes traffic-light state, sanitized static scene context, and YOLOP scene/context summaries.")
    lines.append("They are adapted tabular teacher baselines, not a reproduction of the original local MSLSTM-PID recurrent experiment.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_comparison_report(path: Path, metrics_df: pd.DataFrame) -> None:
    student_biu_kp_bbox = pd.read_csv("results/student_biu_kp_bbox_supervised_metrics.csv")
    student_biu_bbox = pd.read_csv("results/student_biu_bbox_supervised_metrics.csv")
    best_student_f1 = student_biu_kp_bbox.loc[student_biu_kp_bbox["test_f1"].idxmax()]
    best_student_pr = student_biu_bbox.loc[student_biu_bbox["test_pr_auc"].idxmax()]
    best_t1 = metrics_df.loc[metrics_df["variant_name"] == "T1"].sort_values(["test_f1", "val_f1"], ascending=False).iloc[0]
    best_t2 = metrics_df.loc[metrics_df["variant_name"] == "T2"].sort_values(["test_f1", "val_f1"], ascending=False).iloc[0]
    lines = [
        "# Teacher Vs Student Frozen Comparison Report",
        "",
        "## Reference Rows",
        "",
        f"- Best F1 student baseline (`BIU+KP+BBox`): `{best_student_f1['model_name']}` test_f1=`{best_student_f1['test_f1']:.6f}` test_pr_auc=`{best_student_f1['test_pr_auc']:.6f}`",
        f"- Best PR-AUC student baseline (`BIU+BBox`): `{best_student_pr['model_name']}` test_f1=`{best_student_pr['test_f1']:.6f}` test_pr_auc=`{best_student_pr['test_pr_auc']:.6f}`",
        f"- Best `T1`: `{best_t1['model_name']}` test_f1=`{best_t1['test_f1']:.6f}` test_pr_auc=`{best_t1['test_pr_auc']:.6f}`",
        f"- Best `T2`: `{best_t2['model_name']}` test_f1=`{best_t2['test_f1']:.6f}` test_pr_auc=`{best_t2['test_pr_auc']:.6f}`",
        "",
        "## Answers",
        "",
        f"- `T1` {'beats' if best_t1['test_f1'] > best_student_f1['test_f1'] else 'does not beat'} the best-F1 lightweight student baseline on test F1.",
        f"- `T1` {'beats' if best_t1['test_pr_auc'] > best_student_pr['test_pr_auc'] else 'does not beat'} the best-PR-AUC lightweight student baseline on test PR-AUC.",
        f"- `T2` {'improves over' if best_t2['test_f1'] > best_t1['test_f1'] else 'does not improve over'} `T1` on test F1.",
        f"- Scene context {'helps' if best_t2['test_f1'] > best_t1['test_f1'] or best_t2['test_pr_auc'] > best_t1['test_pr_auc'] else 'does not clearly help'} under the strict 30-60 TTE setup based on `T2` vs `T1`.",
        "- `T2` inputs included in this run: `bbox`, `ego_speed`, `ego_acc`, `ego_gyro`, `traffic_light`, sanitized static scene context (`signalized`, `intersection`, `traffic_direction`, `num_lanes_bucket`), and YOLOP scene/context summaries.",
        "- The adapted MSLSTM-style context teacher is suitable as a Stage 1/2 benchmark only if it improves or remains competitive without relying on excluded annotation-rich streams.",
        "- Limitations: tabular summaries are not the original recurrent MSLSTM architecture, no KD is applied here, and T1/T2 only cover the first frozen-row context-teacher pass.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train frozen-row T1/T2 tabular teacher baselines.")
    parser.add_argument("--tables_dir", type=Path, default=Path("data/processed/teacher"))
    parser.add_argument("--output_dir", type=Path, default=Path("results/teacher_mslstm_frozen_T1_T2"))
    parser.add_argument("--report_path", type=Path, default=Path("MSLSTM_FROZEN_T1_T2_SUPERVISED_REPORT.md"))
    parser.add_argument("--comparison_report_path", type=Path, default=Path("TEACHER_VS_STUDENT_FROZEN_COMPARISON_REPORT.md"))
    parser.add_argument("--include_yolop", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_frames: list[pd.DataFrame] = []
    variant_meta: dict[str, dict[str, Any]] = {"T1": {"include_yolop": False}, "T2": {"include_yolop": args.include_yolop}}
    reports: dict[str, StudentLoaderReport] = {}
    payload: dict[str, Any] = {"variants": {}}
    for variant in ["T1", "T2"]:
        df, meta, report = _run_variant(variant, args.tables_dir, args.output_dir)
        metrics_frames.append(df)
        reports[variant] = report
        payload["variants"][variant] = {"warnings": meta["warnings"], "timings": meta["timings"]}
    metrics_df = pd.concat(metrics_frames, ignore_index=True)
    metrics_path = args.output_dir / "teacher_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    write_reports(args.report_path, metrics_df, variant_meta, reports)
    write_comparison_report(args.comparison_report_path, metrics_df)
    payload["metrics_csv"] = str(metrics_path)
    payload["report"] = str(args.report_path)
    payload["comparison_report"] = str(args.comparison_report_path)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
