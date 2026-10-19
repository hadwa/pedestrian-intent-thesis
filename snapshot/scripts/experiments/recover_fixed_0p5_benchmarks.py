from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


ROOT = Path("/home/abdelrahman/Hadwa/BIU-PID-Distillation-clean")
EXP030_ROOT = ROOT / "experiments/030_final_confidence_benchmark"
EXP018_ROOT = ROOT / "experiments/018_mslstm_retrain_benchmark"
DOCS_ROOT = ROOT / "docs"

EXP030_METRICS_DIR = EXP030_ROOT / "metrics"
EXP018_METRICS_DIR = EXP018_ROOT / "metrics"
EXP018_PREDICTIONS_DIR = EXP018_ROOT / "predictions"

EXP030_HISTORICAL_PER_SEED = EXP030_METRICS_DIR / "per_seed_representation_results.csv"
EXP030_HISTORICAL_COMPARISONS = EXP030_METRICS_DIR / "comparison_summary.csv"
EXP030_TRANSITION_COUNTS = EXP030_ROOT / "transition_audit" / "transition_count_summary.csv"

EXP030_FIXED_PER_SEED = EXP030_METRICS_DIR / "fixed_0p5_benchmark_per_seed.csv"
EXP030_FIXED_SUMMARY = EXP030_METRICS_DIR / "fixed_0p5_benchmark_summary.csv"
EXP030_FIXED_DELTAS = EXP030_METRICS_DIR / "fixed_0p5_comparison_deltas.csv"
EXP030_FIXED_VS_VAL = EXP030_METRICS_DIR / "fixed_0p5_vs_val_f1_comparison.csv"

EXP018_RESULTS = EXP018_METRICS_DIR / "mslstm_retrain_results.csv"
EXP018_RESULTS_MEAN_STD = EXP018_METRICS_DIR / "mslstm_retrain_results_mean_std.csv"
EXP018_REFERENCE_ANCHORS = EXP018_METRICS_DIR / "reference_tabular_anchors.csv"
EXP018_STREAM_CONFIG = EXP018_METRICS_DIR / "stream_configuration.csv"
EXP018_FIXED_SUMMARY = EXP018_METRICS_DIR / "fixed_0p5_baseline_summary.csv"

EXP030_DOC = DOCS_ROOT / "EXP030_FIXED_0P5_BENCHMARK.md"
EXP018_DOC = DOCS_ROOT / "EXP018_FIXED_0P5_MSLSTM_BASELINE.md"
THRESHOLD_AUDIT_DOC = DOCS_ROOT / "THRESHOLD_PROVENANCE_AUDIT.md"
HEADLINE_DOC = DOCS_ROOT / "THESIS_HEADLINE_RESULT_PROVENANCE.md"

EXISTING_FIXED_ANCHOR_AUDIT = ROOT / "docs/aibthings_pie_30_60_fixed_vs_selected_results.csv"

COMPARISONS = [
    ("kp_xy", "kp_xyc", "KP-XY -> KP-XYC"),
    ("biu_v2", "biu_v2_plus_global_conf", "BIU-v2 -> BIU-v2 + global confidence"),
    ("biu_v2", "biu_v2_plus_reliability", "BIU-v2 -> BIU-v2 + BIU-specific reliability"),
    ("bbox", "bbox_plus_biu_v2", "BBox -> BBox + BIU-v2"),
    ("bbox_plus_biu_v2", "bbox_plus_biu_v2_plus_reliability", "BBox + BIU-v2 -> BBox + BIU-v2 + reliability"),
]

TRANSITION_COMPARISONS = {
    "KP-XY -> KP-XYC",
    "BIU-v2 -> BIU-v2 + BIU-specific reliability",
    "BBox + BIU-v2 -> BBox + BIU-v2 + reliability",
}


def _safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return float("nan")


def _safe_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        return float(average_precision_score(y_true, y_score))
    except ValueError:
        return float("nan")


def _std(series: pd.Series) -> float:
    if len(series) <= 1:
        return 0.0
    return float(series.std(ddof=1))


def _comparison_metric_sign(a: float, b: float, eps: float = 1e-12) -> bool:
    if abs(a) <= eps and abs(b) <= eps:
        return True
    return (a > eps and b > eps) or (a < -eps and b < -eps)


def _parse_selected_prediction_name(path: Path) -> tuple[str, str, str]:
    stem = path.name.removesuffix("_selected_predictions.json")
    dataset, horizon_a, horizon_b, representation = stem.split("_", 3)
    return dataset, f"{horizon_a}_{horizon_b}", representation


def load_selected_prediction_payload(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise TypeError(f"Expected list payload in {path}, got {type(payload)!r}")
    return payload


def recover_seed_metrics(
    dataset: str,
    horizon: str,
    representation: str,
    payload: dict,
    source_path: Path,
) -> dict:
    pred_df = pd.DataFrame(payload["test_predictions"])
    if pred_df.empty:
        raise ValueError(f"No test_predictions in {source_path} seed {payload.get('seed')}")
    y_true = pred_df["label"].astype(int).to_numpy()
    y_score = pred_df["probability"].astype(float).to_numpy()
    y_pred = (y_score >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "dataset": dataset,
        "horizon": horizon,
        "representation": representation,
        "seed": int(payload["seed"]),
        "model_name": payload["model_name"],
        "threshold_rule": "fixed_0p5",
        "threshold": 0.5,
        "historical_threshold_rule": "val_f1",
        "historical_selected_threshold": float(payload["selected_threshold"]),
        "test_rows": int(len(pred_df)),
        "positive_rate": float(pred_df["label"].mean()),
        "test_accuracy": float(accuracy_score(y_true, y_pred)),
        "test_precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "test_recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "test_f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "test_roc_auc": _safe_roc_auc(y_true, y_score),
        "test_pr_auc": _safe_pr_auc(y_true, y_score),
        "test_tn": int(tn),
        "test_fp": int(fp),
        "test_fn": int(fn),
        "test_tp": int(tp),
        "source_prediction_file": str(source_path),
    }


def recover_all_per_seed() -> tuple[pd.DataFrame, dict[tuple[str, str, str, int], pd.DataFrame]]:
    rows: list[dict] = []
    prediction_frames: dict[tuple[str, str, str, int], pd.DataFrame] = {}
    for path in sorted(EXP030_METRICS_DIR.glob("*_selected_predictions.json")):
        dataset, horizon, representation = _parse_selected_prediction_name(path)
        for payload in load_selected_prediction_payload(path):
            rows.append(recover_seed_metrics(dataset, horizon, representation, payload, path))
            pred_df = pd.DataFrame(payload["test_predictions"]).copy()
            pred_df["prediction_fixed_0p5"] = (pred_df["probability"].astype(float) >= 0.5).astype(int)
            prediction_frames[(dataset, horizon, representation, int(payload["seed"]))] = pred_df
    per_seed = pd.DataFrame(rows).sort_values(["dataset", "horizon", "representation", "seed"]).reset_index(drop=True)
    return per_seed, prediction_frames


def build_fixed_0p5_summary(per_seed: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "test_rows",
        "positive_rate",
        "test_accuracy",
        "test_precision",
        "test_recall",
        "test_f1",
        "test_roc_auc",
        "test_pr_auc",
    ]
    rows = []
    for (dataset, horizon, representation), group in per_seed.groupby(["dataset", "horizon", "representation"], sort=True):
        row = {
            "dataset": dataset,
            "horizon": horizon,
            "representation": representation,
            "threshold_rule": "fixed_0p5",
        }
        for col in metric_cols:
            row[f"{col}_mean"] = float(group[col].mean())
            row[f"{col}_std"] = _std(group[col])
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["dataset", "horizon", "representation"]).reset_index(drop=True)


def build_comparison_deltas(per_seed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, horizon in sorted(per_seed[["dataset", "horizon"]].drop_duplicates().itertuples(index=False, name=None)):
        subset = per_seed[(per_seed["dataset"] == dataset) & (per_seed["horizon"] == horizon)]
        for baseline, candidate, label in COMPARISONS:
            base = subset[subset["representation"] == baseline].sort_values("seed")
            cand = subset[subset["representation"] == candidate].sort_values("seed")
            if base.empty or cand.empty:
                continue
            merged = base.merge(cand, on="seed", suffixes=("_baseline", "_candidate"))
            row = {
                "dataset": dataset,
                "horizon": horizon,
                "comparison": label,
                "baseline": baseline,
                "candidate": candidate,
                "threshold_rule": "fixed_0p5",
            }
            for metric in ["accuracy", "roc_auc", "pr_auc", "f1", "precision", "recall"]:
                base_col = f"test_{metric}_baseline"
                cand_col = f"test_{metric}_candidate"
                delta = merged[cand_col] - merged[base_col]
                row[f"{metric}_baseline_mean"] = float(merged[base_col].mean())
                row[f"{metric}_baseline_std"] = _std(merged[base_col])
                row[f"{metric}_candidate_mean"] = float(merged[cand_col].mean())
                row[f"{metric}_candidate_std"] = _std(merged[cand_col])
                row[f"{metric}_delta_mean"] = float(delta.mean())
                row[f"{metric}_delta_std"] = _std(delta)
                row[f"{metric}_direction_consistent"] = bool(
                    np.all(np.sign(delta.to_numpy()) == np.sign(delta.iloc[0])) if len(delta) else True
                )
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["dataset", "horizon", "comparison"]).reset_index(drop=True)


def build_fixed_vs_val_f1(deltas_fixed: pd.DataFrame, historical: pd.DataFrame) -> pd.DataFrame:
    merged = deltas_fixed.merge(
        historical,
        on=["dataset", "horizon", "comparison", "baseline", "candidate"],
        suffixes=("_fixed", "_historical"),
        how="inner",
    )
    rows = []
    for rec in merged.itertuples(index=False):
        for metric in ["accuracy", "roc_auc", "pr_auc", "f1", "precision", "recall"]:
            fixed_delta = getattr(rec, f"{metric}_delta_mean_fixed")
            val_delta = getattr(rec, f"{metric}_delta_mean_historical")
            rows.append(
                {
                    "dataset": rec.dataset,
                    "horizon": rec.horizon,
                    "comparison": rec.comparison,
                    "metric": metric,
                    "fixed_0p5_delta": float(fixed_delta),
                    "val_f1_delta": float(val_delta),
                    "same_qualitative_conclusion": _comparison_metric_sign(float(fixed_delta), float(val_delta)),
                }
            )
    return pd.DataFrame(rows).sort_values(["dataset", "horizon", "comparison", "metric"]).reset_index(drop=True)


def build_prediction_pairs_for_transition_audit(
    prediction_frames: dict[tuple[str, str, str, int], pd.DataFrame]
) -> pd.DataFrame:
    rows = []
    for dataset, horizon in sorted({(d, h) for d, h, _, _ in prediction_frames.keys()}):
        for baseline, candidate, label in COMPARISONS:
            if label not in TRANSITION_COMPARISONS:
                continue
            for seed in [42, 43, 44]:
                base_key = (dataset, horizon, baseline, seed)
                cand_key = (dataset, horizon, candidate, seed)
                if base_key not in prediction_frames or cand_key not in prediction_frames:
                    continue
                base = prediction_frames[base_key][["sequence_id", "label", "track_id", "probability", "prediction_fixed_0p5"]].copy()
                cand = prediction_frames[cand_key][["sequence_id", "label", "track_id", "probability", "prediction_fixed_0p5"]].copy()
                merged = base.merge(
                    cand,
                    on=["sequence_id", "track_id"],
                    suffixes=("_baseline", "_candidate"),
                    how="inner",
                )
                if len(merged) != len(base) or len(merged) != len(cand):
                    raise RuntimeError(f"Prediction mismatch for {dataset} {horizon} {label} seed {seed}")
                if not merged["label_baseline"].equals(merged["label_candidate"]):
                    raise RuntimeError(f"Label mismatch for {dataset} {horizon} {label} seed {seed}")
                for rec in merged.itertuples(index=False):
                    rows.append(
                        {
                            "dataset": dataset,
                            "horizon": horizon,
                            "comparison": label,
                            "threshold_regime": "fixed_0p5",
                            "seed": seed,
                            "sequence_id": rec.sequence_id,
                            "track_id": rec.track_id,
                            "baseline_label": int(rec.label_baseline),
                            "baseline_prob": float(rec.probability_baseline),
                            "candidate_prob": float(rec.probability_candidate),
                            "baseline_pred": int(rec.prediction_fixed_0p5_baseline),
                            "candidate_pred": int(rec.prediction_fixed_0p5_candidate),
                        }
                    )
    return pd.DataFrame(rows)


def _transition_label(y_true: int, baseline_pred: int, candidate_pred: int) -> tuple[str, str]:
    if y_true == 0 and baseline_pred == 1:
        baseline_state = "FP"
        transition = "FP->TN" if candidate_pred == 0 else "FP->FP"
    elif y_true == 1 and baseline_pred == 0:
        baseline_state = "FN"
        transition = "FN->TP" if candidate_pred == 1 else "FN->FN"
    elif y_true == 0 and baseline_pred == 0:
        baseline_state = "TN"
        transition = "TN->FP" if candidate_pred == 1 else "TN->TN"
    else:
        baseline_state = "TP"
        transition = "TP->FN" if candidate_pred == 0 else "TP->TP"
    return baseline_state, transition


def verify_transition_counts_against_audit(prediction_pairs: pd.DataFrame, audit_path: Path) -> None:
    audit = pd.read_csv(audit_path)
    audit = audit[audit["threshold_regime"] == "fixed_0p5"].copy()

    actual_rows = []
    for rec in prediction_pairs.itertuples(index=False):
        baseline_state, transition = _transition_label(rec.baseline_label, rec.baseline_pred, rec.candidate_pred)
        actual_rows.append(
            {
                "dataset": rec.dataset,
                "horizon": rec.horizon,
                "comparison": rec.comparison,
                "threshold_regime": rec.threshold_regime,
                "seed": rec.seed,
                "baseline_state": baseline_state,
                "transition_label": transition,
            }
        )
    actual = pd.DataFrame(actual_rows)
    actual_counts = (
        actual.groupby(
            ["dataset", "horizon", "comparison", "threshold_regime", "seed", "baseline_state", "transition_label"],
            sort=True,
        )
        .size()
        .reset_index(name="count")
    )
    actual_pop = (
        actual.groupby(
            ["dataset", "horizon", "comparison", "threshold_regime", "seed", "baseline_state"],
            sort=True,
        )
        .size()
        .reset_index(name="baseline_population")
    )
    actual_counts = actual_counts.merge(
        actual_pop,
        on=["dataset", "horizon", "comparison", "threshold_regime", "seed", "baseline_state"],
        how="left",
    )

    merged = actual_counts.merge(
        audit[
            [
                "dataset",
                "horizon",
                "comparison",
                "threshold_regime",
                "seed",
                "baseline_state",
                "transition_label",
                "count",
                "baseline_population",
            ]
        ],
        on=["dataset", "horizon", "comparison", "threshold_regime", "seed", "baseline_state", "transition_label"],
        suffixes=("_actual", "_audit"),
        how="outer",
        indicator=True,
    )
    if not (merged["_merge"] == "both").all():
        raise RuntimeError(f"Transition audit row mismatch:\n{merged.loc[merged['_merge'] != 'both'].to_string(index=False)}")
    mismatched = merged[
        (merged["count_actual"] != merged["count_audit"])
        | (merged["baseline_population_actual"] != merged["baseline_population_audit"])
    ]
    if not mismatched.empty:
        raise RuntimeError(f"Transition audit count mismatch:\n{mismatched.to_string(index=False)}")


def _table_string(df: pd.DataFrame, cols: Iterable[str] | None = None) -> str:
    if cols is not None:
        df = df.loc[:, list(cols)]
    return df.to_string(index=False)


def write_exp030_doc(
    per_seed: pd.DataFrame,
    summary: pd.DataFrame,
    deltas: pd.DataFrame,
    fixed_vs_val: pd.DataFrame,
) -> None:
    lines = [
        "# Exp030 Fixed-0.5 Benchmark Recovery",
        "",
        "Historical primary analysis: `val_f1` selected thresholds from the original final confidence benchmark.",
        "Recovered thesis-comparability analysis: fixed `0.5`, recomputed from the saved selected-prediction artifacts only.",
        "Threshold-independent `ROC-AUC` and `PR-AUC` come from the same saved probabilities and are therefore unchanged by threshold policy.",
        "",
        "## Fixed-0.5 Summary Means",
        "",
        "```text",
        _table_string(summary, ["dataset", "horizon", "representation", "test_accuracy_mean", "test_precision_mean", "test_recall_mean", "test_f1_mean", "test_roc_auc_mean", "test_pr_auc_mean"]),
        "```",
        "",
        "## Fixed-0.5 Paired Deltas",
        "",
        "```text",
        _table_string(deltas, ["dataset", "horizon", "comparison", "accuracy_delta_mean", "roc_auc_delta_mean", "pr_auc_delta_mean", "f1_delta_mean", "precision_delta_mean", "recall_delta_mean"]),
        "```",
        "",
        "## Fixed-0.5 versus Historical val_f1",
        "",
        "```text",
        _table_string(fixed_vs_val, ["dataset", "horizon", "comparison", "metric", "fixed_0p5_delta", "val_f1_delta", "same_qualitative_conclusion"]),
        "```",
        "",
        "## Historical-primary reminder",
        "",
        "- Original Exp030 benchmark files remain unchanged and continue to represent the historical primary `val_f1` analysis.",
        "- The new fixed-`0.5` files are recovery artifacts for thesis comparability only.",
    ]
    EXP030_DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_fixed_0p5_anchor_rows() -> pd.DataFrame:
    if not EXISTING_FIXED_ANCHOR_AUDIT.exists():
        return pd.DataFrame()
    anchors = pd.read_csv(EXISTING_FIXED_ANCHOR_AUDIT)
    keep = anchors["feature_set"].isin(["bbox", "biu_bbox", "teacher_t2"]) & (anchors["policy"] == "fixed_0_5")
    return anchors.loc[keep].copy()


def build_exp018_fixed_summary() -> pd.DataFrame:
    mslstm = pd.read_csv(EXP018_RESULTS)
    mslstm_fixed = mslstm[mslstm["threshold_rule"] == "fixed_0_5"].copy()
    mslstm_fixed["family"] = "mslstm"
    mslstm_fixed["model_name"] = mslstm_fixed["variant"]
    mslstm_fixed["representation"] = mslstm_fixed["variant"]
    mslstm_fixed["threshold"] = 0.5
    mslstm_fixed["notes"] = "existing Exp018 fixed_0_5 benchmark row"
    deep_cols = [
        "family",
        "model_name",
        "representation",
        "streams",
        "seed",
        "best_epoch",
        "threshold_rule",
        "threshold",
        "test_rows",
        "test_accuracy",
        "test_precision",
        "test_recall",
        "test_f1",
        "test_roc_auc",
        "test_pr_auc",
        "notes",
    ]
    deep = mslstm_fixed[deep_cols].copy()

    mean_std_rows = []
    for variant, group in deep.groupby("representation", sort=True):
        mean_std_rows.append(
            {
                "family": "mslstm",
                "model_name": variant,
                "representation": variant,
                "streams": group["streams"].iloc[0],
                "seed": "mean_std",
                "best_epoch": "",
                "threshold_rule": "fixed_0_5",
                "threshold": 0.5,
                "test_rows": int(group["test_rows"].iloc[0]),
                "test_accuracy": f"{group['test_accuracy'].mean():.6f} +/- {_std(group['test_accuracy']):.6f}",
                "test_precision": f"{group['test_precision'].mean():.6f} +/- {_std(group['test_precision']):.6f}",
                "test_recall": f"{group['test_recall'].mean():.6f} +/- {_std(group['test_recall']):.6f}",
                "test_f1": f"{group['test_f1'].mean():.6f} +/- {_std(group['test_f1']):.6f}",
                "test_roc_auc": f"{group['test_roc_auc'].mean():.6f} +/- {_std(group['test_roc_auc']):.6f}",
                "test_pr_auc": f"{group['test_pr_auc'].mean():.6f} +/- {_std(group['test_pr_auc']):.6f}",
                "notes": "mean +/- std across seeds 42/43/44",
            }
        )
    deep = pd.concat([deep, pd.DataFrame(mean_std_rows)], ignore_index=True)

    anchors = _load_fixed_0p5_anchor_rows()
    anchor_rows = []
    if not anchors.empty:
        anchor_map = {
            ("bbox", "random_forest"): ("bbox rf", "bbox", "same frozen PIE 30_60 protocol fixed_0_5 archive-derived row"),
            ("biu_bbox", "random_forest"): ("biu_bbox rf", "biu_bbox", "same frozen PIE 30_60 protocol fixed_0_5 archive-derived row"),
            ("teacher_t2", "xgboost"): ("teacher_t2 xgboost", "teacher_t2", "same frozen PIE 30_60 protocol fixed_0_5 archive-derived row"),
        }
        for (feature_set, model_name), (label, representation, note) in anchor_map.items():
            row = anchors[(anchors["feature_set"] == feature_set) & (anchors["model_name"] == model_name)]
            if row.empty:
                anchor_rows.append(
                    {
                        "family": "tabular_anchor",
                        "model_name": label,
                        "representation": representation,
                        "streams": "",
                        "seed": "42",
                        "best_epoch": "",
                        "threshold_rule": "unavailable_fixed_0p5",
                        "threshold": 0.5,
                        "test_rows": np.nan,
                        "test_accuracy": np.nan,
                        "test_precision": np.nan,
                        "test_recall": np.nan,
                        "test_f1": np.nan,
                        "test_roc_auc": np.nan,
                        "test_pr_auc": np.nan,
                        "notes": "fixed_0p5 anchor not found in repository evidence",
                    }
                )
                continue
            rec = row.iloc[0]
            anchor_rows.append(
                {
                    "family": "tabular_anchor",
                    "model_name": label,
                    "representation": representation,
                    "streams": "",
                    "seed": "42",
                    "best_epoch": "",
                    "threshold_rule": "fixed_0_5",
                    "threshold": 0.5,
                    "test_rows": np.nan,
                    "test_accuracy": float(rec["test_accuracy"]),
                    "test_precision": float(rec["test_precision"]),
                    "test_recall": float(rec["test_recall"]),
                    "test_f1": float(rec["test_f1"]),
                    "test_roc_auc": float(rec["test_roc_auc"]),
                    "test_pr_auc": float(rec["test_pr_auc"]),
                    "notes": note,
                }
            )
    all_rows = pd.concat([deep, pd.DataFrame(anchor_rows)], ignore_index=True)
    return all_rows


def pose_source_verdict() -> tuple[str, str]:
    verdict = "CONDITIONAL"
    reason = (
        "Exp018 pose and the PIE BIU-v2 benchmark both read the same cached keypoint file "
        "`/home/abdelrahman/Hadwa/pedestrian-intent-biu/data/data_keypoints_all.jsonl`, so the current "
        "measurement source is matched at the file level. However, repository evidence does not explicitly pin "
        "that PIE JSONL to a ViTPose-S extraction lineage; Exp018 documentation instead references the older "
        "`Offline Extraction/yolo_pose_extraction.ipynb` normalization path. Reuse is exact only if the new "
        "MSLSTM+BIU experiment uses this same cached JSONL rather than an explicitly re-extracted ViTPose-S corpus."
    )
    return verdict, reason


def write_exp018_doc(summary_csv: pd.DataFrame) -> None:
    mean_row = summary_csv[(summary_csv["family"] == "mslstm") & (summary_csv["representation"] == "mslstm_bbox_only") & (summary_csv["seed"] == "mean_std")].iloc[0]
    hist = pd.read_csv(EXP018_RESULTS_MEAN_STD)
    hist_row = hist[(hist["variant"] == "mslstm_bbox_only") & (hist["threshold_rule"] == "val_f1")].iloc[0]
    verdict, reason = pose_source_verdict()
    lines = [
        "# Exp018 Fixed-0.5 MSLSTM Baseline",
        "",
        "Historical headline:",
        f"- `mslstm_bbox_only` under `val_f1`: test F1 `{hist_row['test_f1_mean']:.6f} +/- {hist_row['test_f1_std']:.6f}`.",
        "",
        "Fixed-0.5 controlled baseline:",
        f"- `mslstm_bbox_only` under fixed `0.5`: test F1 `{mean_row['test_f1']}`.",
        "",
        "## Per-seed fixed-0.5 MSLSTM rows",
        "",
        "```text",
        _table_string(summary_csv[(summary_csv["family"] == "mslstm") & (summary_csv["seed"] != "mean_std")],
                      ["model_name", "seed", "best_epoch", "test_accuracy", "test_precision", "test_recall", "test_f1", "test_roc_auc", "test_pr_auc"]),
        "```",
        "",
        "## Mean/std fixed-0.5 MSLSTM rows",
        "",
        "```text",
        _table_string(summary_csv[(summary_csv["family"] == "mslstm") & (summary_csv["seed"] == "mean_std")],
                      ["model_name", "test_accuracy", "test_precision", "test_recall", "test_f1", "test_roc_auc", "test_pr_auc"]),
        "```",
        "",
        "## Fixed-0.5 tabular anchors",
        "",
        "```text",
        _table_string(summary_csv[summary_csv["family"] == "tabular_anchor"],
                      ["model_name", "threshold_rule", "test_accuracy", "test_precision", "test_recall", "test_f1", "test_roc_auc", "test_pr_auc", "notes"]),
        "```",
        "",
        "## Reuse verdict for the upcoming MSLSTM+BIU experiment",
        "",
        f"- Verdict: `{verdict}`",
        f"- Reason: {reason}",
    ]
    EXP018_DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_once(path: Path, marker: str, content: str) -> None:
    text = path.read_text(encoding="utf-8")
    if marker in text:
        return
    if not text.endswith("\n"):
        text += "\n"
    text += "\n" + content.strip() + "\n"
    path.write_text(text, encoding="utf-8")


def update_provenance_docs(exp018_summary: pd.DataFrame) -> None:
    bbox_mean = exp018_summary[
        (exp018_summary["family"] == "mslstm")
        & (exp018_summary["representation"] == "mslstm_bbox_only")
        & (exp018_summary["seed"] == "mean_std")
    ].iloc[0]
    verdict, _ = pose_source_verdict()
    append_once(
        THRESHOLD_AUDIT_DOC,
        "### Exp030 fixed-0.5 recovery (2026-08-13)",
        """
### Exp030 fixed-0.5 recovery (2026-08-13)

- Historical primary remains `val_f1`.
- Fixed-`0.5` benchmark rows were recovered from the saved selected-prediction artifacts only.
- New thesis-comparability files:
  - `experiments/030_final_confidence_benchmark/metrics/fixed_0p5_benchmark_per_seed.csv`
  - `experiments/030_final_confidence_benchmark/metrics/fixed_0p5_benchmark_summary.csv`
  - `experiments/030_final_confidence_benchmark/metrics/fixed_0p5_comparison_deltas.csv`
  - `experiments/030_final_confidence_benchmark/metrics/fixed_0p5_vs_val_f1_comparison.csv`
- `ROC-AUC` and `PR-AUC` are unchanged because they are threshold-independent and come from the same saved probabilities.
        """,
    )
    append_once(
        HEADLINE_DOC,
        "Recovered fixed-`0.5` MSLSTM baseline",
        f"""
## Recovered fixed-`0.5` MSLSTM baseline

- Historical `Exp018` MSLSTM headline `~0.779 +/- 0.009` is the `val_f1` row.
- Recovered fixed-`0.5` `mslstm_bbox_only` baseline: `{bbox_mean['test_f1']}`.
- Recovery note: `docs/EXP018_FIXED_0P5_MSLSTM_BASELINE.md`
- Direct reuse for the upcoming `MSLSTM+BIU-v2` experiment: `{verdict}`.
        """,
    )


def main() -> int:
    per_seed, prediction_frames = recover_all_per_seed()
    summary = build_fixed_0p5_summary(per_seed)
    deltas = build_comparison_deltas(per_seed)
    historical = pd.read_csv(EXP030_HISTORICAL_COMPARISONS)
    fixed_vs_val = build_fixed_vs_val_f1(deltas, historical)
    prediction_pairs = build_prediction_pairs_for_transition_audit(prediction_frames)
    verify_transition_counts_against_audit(prediction_pairs, EXP030_TRANSITION_COUNTS)

    per_seed.to_csv(EXP030_FIXED_PER_SEED, index=False)
    summary.to_csv(EXP030_FIXED_SUMMARY, index=False)
    deltas.to_csv(EXP030_FIXED_DELTAS, index=False)
    fixed_vs_val.to_csv(EXP030_FIXED_VS_VAL, index=False)
    write_exp030_doc(per_seed, summary, deltas, fixed_vs_val)

    exp018_summary = build_exp018_fixed_summary()
    exp018_summary.to_csv(EXP018_FIXED_SUMMARY, index=False)
    write_exp018_doc(exp018_summary)
    update_provenance_docs(exp018_summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
