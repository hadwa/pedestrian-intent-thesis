from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


SEED = 12345
RNG = np.random.default_rng(SEED)

REPO = Path("/home/abdelrahman/Hadwa/BIU-PID-Distillation-clean")
ARCHIVE = Path("/home/abdelrahman/Hadwa/BIU-PID-Distillation")
EXP_DIR = REPO / "experiments/024_jaad_adaptive_early_decision"
FIG_DIR = EXP_DIR / "figures"

TRACK_011 = REPO / "experiments/011_jaad_staged_bbox_biu_v2/metrics/staged_track_table.csv"
RESULTS_011 = REPO / "experiments/011_jaad_staged_bbox_biu_v2/metrics/staged_results.csv"
OPS_011 = REPO / "experiments/011_jaad_staged_bbox_biu_v2/metrics/staged_operating_points.csv"
MODELSEL_011 = REPO / "experiments/011_jaad_staged_bbox_biu_v2/metrics/staged_model_selection.csv"

TRACK_021 = REPO / "experiments/021_jaad_corrected_15_30_rebuild/metrics/corrected_staged_track_table.csv"
RESULTS_021 = REPO / "experiments/021_jaad_corrected_15_30_rebuild/metrics/corrected_staged_results.csv"
MODELSEL_021 = REPO / "experiments/021_jaad_corrected_15_30_rebuild/metrics/corrected_staged_model_selection.csv"

RUNTIME_TABLE = REPO / "experiments/aibthings_downstream_runtime/runtime_table.csv"


@dataclass
class AdaptiveSelection:
    epsilon: float
    constraint_satisfied: bool
    l1: float
    u1: float
    l2: float
    u2: float
    t3: float
    mean_tte: float
    median_tte: float
    val_f1: float
    val_precision: float
    val_recall: float
    val_bal_acc: float
    val_pr_auc: float
    val_roc_auc: float
    finalized_30_60_pct: float
    finalized_15_30_pct: float
    finalized_5_15_pct: float
    avg_stages: float
    stage3_deferrals: int
    width_sum: float


def safe_auc(y_true: np.ndarray, y_score: np.ndarray, metric: str) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    if metric == "roc":
        return float(roc_auc_score(y_true, y_score))
    if metric == "pr":
        return float(average_precision_score(y_true, y_score))
    raise ValueError(metric)


def best_f1_threshold(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    candidates = np.unique(np.concatenate(([0.0], y_score, [1.0])))
    best_t = 0.5
    best_f1 = -1.0
    for t in candidates:
        pred = (y_score >= t).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1 or (np.isclose(f1, best_f1) and t < best_t):
            best_f1 = f1
            best_t = float(t)
    return best_t, best_f1


def compute_metrics(df: pd.DataFrame, pred_col: str, score_col: str) -> dict[str, float]:
    y_true = df["label"].to_numpy(dtype=int)
    y_pred = df[pred_col].to_numpy(dtype=int)
    y_score = df[score_col].to_numpy(dtype=float)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out = {
        "n_tracks": len(df),
        "accuracy": float((y_pred == y_true).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": safe_auc(y_true, y_score, "roc"),
        "pr_auc": safe_auc(y_true, y_score, "pr"),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "overall_fpr": float(fp / (fp + tn)) if (fp + tn) else float("nan"),
        "mean_decision_tte": float(df["decision_tte"].mean()),
        "median_decision_tte": float(df["decision_tte"].median()),
        "mean_correct_positive_warning_tte": float(df.loc[(df["label"] == 1) & (df[pred_col] == 1), "decision_tte"].mean()),
        "mean_correct_negative_decision_tte": float(df.loc[(df["label"] == 0) & (df[pred_col] == 0), "decision_tte"].mean()),
        "avg_stages_evaluated": float(df["stages_evaluated"].mean()),
    }
    for stage in ["30_60", "15_30_corrected", "5_15"]:
        stage_mask = df["exit_stage"] == stage
        out[f"finalized_{stage}_pct"] = float(stage_mask.mean())
        out[f"correct_at_{stage}_pct"] = float(((stage_mask) & (df[pred_col] == df["label"])).mean())
        out[f"fp_at_{stage}_pct"] = float(((stage_mask) & (df["label"] == 0) & (df[pred_col] == 1)).mean())
        out[f"fn_at_{stage}_pct"] = float(((stage_mask) & (df["label"] == 1) & (df[pred_col] == 0)).mean())
    for group_name in ["ambiguous_negative", "clean_negative"]:
        sub = df[df["group"] == group_name]
        out[f"{group_name}_fpr"] = float(((sub[pred_col] == 1) & (sub["label"] == 0)).mean()) if len(sub) else float("nan")
    pos = df[df["group"] == "actual_positive"]
    out["actual_positive_recall"] = float((pos[pred_col] == 1).mean()) if len(pos) else float("nan")
    return out


def compute_metrics_from_arrays(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    groups: np.ndarray,
    decision_tte: np.ndarray,
    exit_stage: np.ndarray,
    stages_evaluated: np.ndarray,
    include_auc: bool = False,
) -> dict[str, float]:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    out = {
        "accuracy": float((y_pred == y_true).mean()),
        "balanced_accuracy": float((recall + specificity) / 2.0),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": safe_auc(y_true, y_score, "roc") if include_auc else float("nan"),
        "pr_auc": safe_auc(y_true, y_score, "pr") if include_auc else float("nan"),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "mean_decision_tte": float(decision_tte.mean()),
        "median_decision_tte": float(np.median(decision_tte)),
        "avg_stages_evaluated": float(stages_evaluated.mean()),
        "finalized_30_60_pct": float((exit_stage == "30_60").mean()),
        "finalized_15_30_corrected_pct": float((exit_stage == "15_30_corrected").mean()),
        "finalized_5_15_pct": float((exit_stage == "5_15").mean()),
    }
    return out


def load_track_table(path: Path, horizon_label: str, feature_set: str, model_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    mask = (
        (df["horizon_label"] == horizon_label)
        & (df["feature_set"] == feature_set)
        & (df["model_name"] == model_name)
    )
    out = df.loc[mask].copy()
    if out["track_key"].duplicated().any():
        raise ValueError(f"duplicate track rows in {path} for {horizon_label}/{feature_set}/{model_name}")
    return out


def attach_horizon(df: pd.DataFrame, prefix: str, horizon_label: str) -> pd.DataFrame:
    cols = {
        "sequence_id": f"{prefix}_sequence_id",
        "prob": f"{prefix}_prob",
        "tte_frames": f"{prefix}_tte",
        "start_frame": f"{prefix}_start_frame",
    }
    base = df.rename(columns=cols)[["track_key", "split", "label", "group", *cols.values()]].copy()
    base[f"{prefix}_horizon"] = horizon_label
    return base


def build_common_track_frame() -> tuple[pd.DataFrame, pd.DataFrame]:
    track_011 = pd.read_csv(TRACK_011)
    track_021 = pd.read_csv(TRACK_021)

    bbox_30 = load_track_table(TRACK_011, "30_60", "bbox_only", "logreg")
    bbox_5 = load_track_table(TRACK_011, "5_15", "bbox_only", "logreg")
    bbox_15 = load_track_table(TRACK_021, "15_30_corrected", "bbox_only", "logreg")
    staged_5 = load_track_table(TRACK_011, "5_15", "biu_v2_core_plus_bbox", "rf")

    df30 = attach_horizon(bbox_30, "bbox30", "30_60")
    df15 = attach_horizon(bbox_15, "bbox15", "15_30_corrected")
    df5 = attach_horizon(bbox_5, "bbox5", "5_15")
    dfst = staged_5.rename(columns={"sequence_id": "stage2_sequence_id", "prob": "stage2_prob", "tte_frames": "stage2_tte"})[
        ["track_key", "split", "stage2_sequence_id", "stage2_prob", "stage2_tte"]
    ].copy()

    merged = df30.merge(df15, on=["track_key", "split", "label", "group"], how="outer", indicator="m15")
    merged = merged.merge(df5, on=["track_key", "split", "label", "group"], how="outer", indicator="m5")
    merged = merged.merge(dfst, on=["track_key", "split"], how="left")

    conditions = [
        merged["bbox30_sequence_id"].notna() & merged["bbox15_sequence_id"].notna() & merged["bbox5_sequence_id"].notna(),
        merged["bbox30_sequence_id"].isna(),
        merged["bbox15_sequence_id"].isna(),
        merged["bbox5_sequence_id"].isna(),
    ]
    reasons = ["common_all_horizons", "missing_30_60", "missing_15_30_corrected", "missing_5_15"]
    merged["availability"] = np.select(conditions, reasons, default="partial_other")

    common = merged[merged["availability"] == "common_all_horizons"].copy()
    missing = merged[merged["availability"] != "common_all_horizons"].copy()
    return common, missing


def run_single(df: pd.DataFrame, score_col: str, threshold: float, method: str, stage: str) -> pd.DataFrame:
    out = df.copy()
    out["method"] = method
    out["final_score"] = out[score_col]
    out["final_pred"] = (out[score_col] >= threshold).astype(int)
    out["exit_stage"] = stage
    out["decision_tte"] = out[f"{score_col.split('_prob')[0]}_tte"]
    out["stages_evaluated"] = {"30_60": 1, "15_30_corrected": 2, "5_15": 3}[stage]
    return out


def run_staged_011(df: pd.DataFrame, t1: float, t2: float) -> pd.DataFrame:
    out = df.copy()
    out["method"] = "staged_011_primary"
    stage1_pos = out["bbox30_prob"] >= t1
    has_stage2 = out["stage2_prob"].notna()
    stage2_pos = out["stage2_prob"] >= t2
    out["final_pred"] = np.where(stage1_pos, np.where(has_stage2, stage2_pos, 1), 0).astype(int)
    out["final_score"] = np.where(stage1_pos & has_stage2, out["stage2_prob"], out["bbox30_prob"])
    out["exit_stage"] = np.where(stage1_pos & has_stage2, "5_15", "30_60")
    out["decision_tte"] = np.where(stage1_pos & has_stage2, out["stage2_tte"], out["bbox30_tte"])
    out["stages_evaluated"] = np.where(stage1_pos & has_stage2, 2, 1)
    return out


def run_adaptive(df: pd.DataFrame, method: str, l1: float, u1: float, l2: float, u2: float, t3: float) -> pd.DataFrame:
    out = df.copy()
    out["method"] = method
    stage1_neg = out["bbox30_prob"] <= l1
    stage1_pos = out["bbox30_prob"] >= u1
    undecided1 = ~(stage1_neg | stage1_pos)

    stage2_neg = out["bbox15_prob"] <= l2
    stage2_pos = out["bbox15_prob"] >= u2
    undecided2 = undecided1 & ~(stage2_neg | stage2_pos)

    out["final_pred"] = np.where(
        stage1_neg,
        0,
        np.where(
            stage1_pos,
            1,
            np.where(
                stage2_neg,
                0,
                np.where(stage2_pos, 1, (out["bbox5_prob"] >= t3).astype(int)),
            ),
        ),
    ).astype(int)
    out["final_score"] = np.where(
        stage1_neg | stage1_pos,
        out["bbox30_prob"],
        np.where(stage2_neg | stage2_pos, out["bbox15_prob"], out["bbox5_prob"]),
    )
    out["exit_stage"] = np.where(
        stage1_neg | stage1_pos,
        "30_60",
        np.where(stage2_neg | stage2_pos, "15_30_corrected", "5_15"),
    )
    out["decision_tte"] = np.where(
        out["exit_stage"] == "30_60",
        out["bbox30_tte"],
        np.where(out["exit_stage"] == "15_30_corrected", out["bbox15_tte"], out["bbox5_tte"]),
    )
    out["stages_evaluated"] = np.where(out["exit_stage"] == "30_60", 1, np.where(out["exit_stage"] == "15_30_corrected", 2, 3))
    return out


def evaluate_adaptive_arrays(
    y_true: np.ndarray,
    groups: np.ndarray,
    p30: np.ndarray,
    tte30: np.ndarray,
    p15: np.ndarray,
    tte15: np.ndarray,
    p5: np.ndarray,
    tte5: np.ndarray,
    l1: float,
    u1: float,
    l2: float,
    u2: float,
    t3: float,
    include_auc: bool = False,
) -> dict[str, object]:
    stage1_neg = p30 <= l1
    stage1_pos = p30 >= u1
    undecided1 = ~(stage1_neg | stage1_pos)

    stage2_neg = p15 <= l2
    stage2_pos = p15 >= u2

    final_pred = np.where(
        stage1_neg,
        0,
        np.where(
            stage1_pos,
            1,
            np.where(stage2_neg, 0, np.where(stage2_pos, 1, (p5 >= t3).astype(int))),
        ),
    ).astype(int)
    exit_stage = np.where(
        stage1_neg | stage1_pos,
        "30_60",
        np.where(stage2_neg | stage2_pos, "15_30_corrected", "5_15"),
    )
    final_score = np.where(
        exit_stage == "30_60",
        p30,
        np.where(exit_stage == "15_30_corrected", p15, p5),
    )
    decision_tte = np.where(exit_stage == "30_60", tte30, np.where(exit_stage == "15_30_corrected", tte15, tte5))
    stages_evaluated = np.where(exit_stage == "30_60", 1, np.where(exit_stage == "15_30_corrected", 2, 3))
    metrics = compute_metrics_from_arrays(y_true, final_pred, final_score, groups, decision_tte, exit_stage, stages_evaluated, include_auc=include_auc)
    metrics.update(
        {
            "final_pred": final_pred,
            "final_score": final_score,
            "exit_stage": exit_stage,
            "decision_tte": decision_tte,
            "stages_evaluated": stages_evaluated,
        }
    )
    return metrics


def summarize_adaptive(
    df: pd.DataFrame,
    epsilon: float,
    constraint_satisfied: bool,
    l1: float,
    u1: float,
    l2: float,
    u2: float,
    t3: float,
) -> AdaptiveSelection:
    metrics = compute_metrics(df, "final_pred", "final_score")
    return AdaptiveSelection(
        epsilon=epsilon,
        constraint_satisfied=constraint_satisfied,
        l1=l1,
        u1=u1,
        l2=l2,
        u2=u2,
        t3=t3,
        mean_tte=metrics["mean_decision_tte"],
        median_tte=metrics["median_decision_tte"],
        val_f1=metrics["f1"],
        val_precision=metrics["precision"],
        val_recall=metrics["recall"],
        val_bal_acc=metrics["balanced_accuracy"],
        val_pr_auc=metrics["pr_auc"],
        val_roc_auc=metrics["roc_auc"],
        finalized_30_60_pct=metrics["finalized_30_60_pct"],
        finalized_15_30_pct=metrics["finalized_15_30_corrected_pct"],
        finalized_5_15_pct=metrics["finalized_5_15_pct"],
        avg_stages=metrics["avg_stages_evaluated"],
        stage3_deferrals=int((df["exit_stage"] == "5_15").sum()),
        width_sum=(u1 - l1) + (u2 - l2),
    )


def bootstrap_deltas(a: pd.DataFrame, b: pd.DataFrame, n_boot: int = 10_000) -> pd.DataFrame:
    if list(a["track_key"]) != list(b["track_key"]):
        raise ValueError("bootstrap inputs must share the same track order")

    def compute_metric_from_arrays(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        groups: np.ndarray,
        decision_tte: np.ndarray,
        stages: np.ndarray,
        metric_name: str,
    ) -> float:
        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        if metric_name == "f1":
            denom = (2 * tp + fp + fn)
            return float((2 * tp) / denom) if denom else 0.0
        if metric_name == "recall":
            denom = tp + fn
            return float(tp / denom) if denom else 0.0
        if metric_name == "precision":
            denom = tp + fp
            return float(tp / denom) if denom else 0.0
        if metric_name == "ambiguous_negative_fpr":
            mask = groups == "ambiguous_negative"
            return float(((y_pred[mask] == 1) & (y_true[mask] == 0)).mean()) if mask.any() else float("nan")
        if metric_name == "mean_decision_tte":
            return float(decision_tte.mean())
        if metric_name == "avg_stages_evaluated":
            return float(stages.mean())
        raise ValueError(metric_name)

    a_y = a["label"].to_numpy(dtype=int)
    a_pred = a["final_pred"].to_numpy(dtype=int)
    a_group = a["group"].to_numpy(dtype=object)
    a_tte = a["decision_tte"].to_numpy(dtype=float)
    a_stages = a["stages_evaluated"].to_numpy(dtype=float)

    b_y = b["label"].to_numpy(dtype=int)
    b_pred = b["final_pred"].to_numpy(dtype=int)
    b_group = b["group"].to_numpy(dtype=object)
    b_tte = b["decision_tte"].to_numpy(dtype=float)
    b_stages = b["stages_evaluated"].to_numpy(dtype=float)

    idx = np.arange(len(a))
    samples = RNG.integers(0, len(a), size=(n_boot, len(a)))
    metric_names = [
        "f1",
        "recall",
        "precision",
        "ambiguous_negative_fpr",
        "mean_decision_tte",
        "avg_stages_evaluated",
    ]
    rows = []
    for metric_name in metric_names:
        base_a = compute_metric_from_arrays(a_y, a_pred, a_group, a_tte, a_stages, metric_name)
        base_b = compute_metric_from_arrays(b_y, b_pred, b_group, b_tte, b_stages, metric_name)
        deltas = np.empty(n_boot, dtype=float)
        for i in range(n_boot):
            sample = samples[i]
            deltas[i] = (
                compute_metric_from_arrays(a_y[sample], a_pred[sample], a_group[sample], a_tte[sample], a_stages[sample], metric_name)
                - compute_metric_from_arrays(b_y[sample], b_pred[sample], b_group[sample], b_tte[sample], b_stages[sample], metric_name)
            )
        rows.append(
            {
                "metric": metric_name,
                "delta_observed": base_a - base_b,
                "ci_low": float(np.quantile(deltas, 0.025)),
                "ci_high": float(np.quantile(deltas, 0.975)),
                "p_value": float(2 * min((deltas <= 0).mean(), (deltas >= 0).mean())),
                "bootstrap_seed": SEED,
                "n_bootstrap": n_boot,
            }
        )
    return pd.DataFrame(rows)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    common, missing = build_common_track_frame()
    common = common.sort_values(["split", "track_key"]).reset_index(drop=True)
    missing = missing.sort_values(["split", "track_key"]).reset_index(drop=True)

    missing.to_csv(EXP_DIR / "missing_track_audit.csv", index=False)

    val = common[common["split"] == "val"].copy()
    test = common[common["split"] == "test"].copy()

    val_y = val["label"].to_numpy(dtype=int)
    val_groups = val["group"].to_numpy(dtype=object)
    val_p30 = val["bbox30_prob"].to_numpy(dtype=float)
    val_t30 = val["bbox30_tte"].to_numpy(dtype=float)
    val_p15 = val["bbox15_prob"].to_numpy(dtype=float)
    val_t15 = val["bbox15_tte"].to_numpy(dtype=float)
    val_p5 = val["bbox5_prob"].to_numpy(dtype=float)
    val_t5 = val["bbox5_tte"].to_numpy(dtype=float)

    t30, valf1_30 = best_f1_threshold(val["label"].to_numpy(), val["bbox30_prob"].to_numpy())
    t15, valf1_15 = best_f1_threshold(val["label"].to_numpy(), val["bbox15_prob"].to_numpy())
    t5, valf1_5 = best_f1_threshold(val["label"].to_numpy(), val["bbox5_prob"].to_numpy())

    ops_011 = pd.read_csv(OPS_011)
    op_011 = ops_011[
        (ops_011["config"] == "staged|bbox30_60->biu_v2_core_plus_bbox@5_15")
        & (ops_011["recall_target"] == 0.95)
    ].iloc[0]

    model_selection = pd.DataFrame(
        [
            {"method": "single_bbox_30_60", "selected_model": "logreg", "threshold_rule": "val_f1", "threshold": t30, "source": str(MODELSEL_011)},
            {"method": "single_bbox_15_30_corrected", "selected_model": "logreg", "threshold_rule": "val_f1", "threshold": t15, "source": str(MODELSEL_021)},
            {"method": "single_bbox_5_15", "selected_model": "logreg", "threshold_rule": "val_f1", "threshold": t5, "source": str(MODELSEL_011)},
            {"method": "staged_011_primary", "selected_model": "bbox30_60=logreg; stage2=rf", "threshold_rule": "exp011_recall_0p95", "threshold": float(op_011["t2"]), "source": str(OPS_011)},
        ]
    )
    model_selection.to_csv(EXP_DIR / "model_selection.csv", index=False)

    grid = np.round(np.arange(0.05, 0.96, 0.05), 2)
    search_rows = []
    for l1 in grid:
        for u1 in grid[grid > l1]:
            for l2 in grid:
                for u2 in grid[grid > l2]:
                    metrics = evaluate_adaptive_arrays(
                        val_y,
                        val_groups,
                        val_p30,
                        val_t30,
                        val_p15,
                        val_t15,
                        val_p5,
                        val_t5,
                        float(l1),
                        float(u1),
                        float(l2),
                        float(u2),
                        float(t5),
                    )
                    search_rows.append(
                        {
                            "l1": float(l1),
                            "u1": float(u1),
                            "l2": float(l2),
                            "u2": float(u2),
                            "t3": float(t5),
                            "val_f1": metrics["f1"],
                            "val_precision": metrics["precision"],
                            "val_recall": metrics["recall"],
                            "val_balanced_accuracy": metrics["balanced_accuracy"],
                            "val_pr_auc": metrics["pr_auc"],
                            "val_roc_auc": metrics["roc_auc"],
                            "mean_decision_tte": metrics["mean_decision_tte"],
                            "median_decision_tte": metrics["median_decision_tte"],
                            "avg_stages_evaluated": metrics["avg_stages_evaluated"],
                            "finalized_30_60_pct": metrics["finalized_30_60_pct"],
                            "finalized_15_30_corrected_pct": metrics["finalized_15_30_corrected_pct"],
                            "finalized_5_15_pct": metrics["finalized_5_15_pct"],
                            "stage3_deferrals": int((metrics["exit_stage"] == "5_15").sum()),
                            "width_sum": float((u1 - l1) + (u2 - l2)),
                        }
                    )
    search_df = pd.DataFrame(search_rows)
    search_df.to_csv(EXP_DIR / "threshold_search_validation.csv", index=False)

    selected_rows: list[AdaptiveSelection] = []
    for eps in [0.00, 0.01, 0.02]:
        cutoff = valf1_5 - eps - 1e-12
        eligible = search_df[search_df["val_f1"] >= cutoff].copy()
        constraint_satisfied = not eligible.empty
        if eligible.empty:
            eligible = search_df.copy()
        eligible = eligible.sort_values(
            ["mean_decision_tte", "val_f1", "stage3_deferrals", "width_sum"],
            ascending=[False, False, True, False],
        )
        row = eligible.iloc[0]
        chosen_val = run_adaptive(val, f"adaptive_eps_{eps:.2f}", float(row["l1"]), float(row["u1"]), float(row["l2"]), float(row["u2"]), float(row["t3"]))
        selected_rows.append(
            summarize_adaptive(
                chosen_val,
                eps,
                constraint_satisfied,
                float(row["l1"]),
                float(row["u1"]),
                float(row["l2"]),
                float(row["u2"]),
                float(row["t3"]),
            )
        )
    selected_df = pd.DataFrame([s.__dict__ for s in selected_rows])
    selected_df.to_csv(EXP_DIR / "selected_operating_points.csv", index=False)

    methods: list[pd.DataFrame] = [
        run_single(val, "bbox30_prob", t30, "single_bbox_30_60", "30_60"),
        run_single(test, "bbox30_prob", t30, "single_bbox_30_60", "30_60"),
        run_single(val, "bbox15_prob", t15, "single_bbox_15_30_corrected", "15_30_corrected"),
        run_single(test, "bbox15_prob", t15, "single_bbox_15_30_corrected", "15_30_corrected"),
        run_single(val, "bbox5_prob", t5, "single_bbox_5_15", "5_15"),
        run_single(test, "bbox5_prob", t5, "single_bbox_5_15", "5_15"),
        run_staged_011(val, float(op_011["t1"]), float(op_011["t2"])),
        run_staged_011(test, float(op_011["t1"]), float(op_011["t2"])),
    ]

    for selection in selected_rows:
        methods.append(run_adaptive(val, f"adaptive_eps_{selection.epsilon:.2f}", selection.l1, selection.u1, selection.l2, selection.u2, selection.t3))
        methods.append(run_adaptive(test, f"adaptive_eps_{selection.epsilon:.2f}", selection.l1, selection.u1, selection.l2, selection.u2, selection.t3))

    track_level = pd.concat(methods, ignore_index=True)
    track_level["was_correct"] = (track_level["final_pred"] == track_level["label"]).astype(int)
    track_level.to_csv(EXP_DIR / "track_level_decisions.csv", index=False)

    metrics_rows = []
    stage_exit_rows = []
    for (method, split), sub in track_level.groupby(["method", "split"]):
        metrics = compute_metrics(sub, "final_pred", "final_score")
        metrics_rows.append({"method": method, "split": split, **metrics})
        for stage, stage_sub in sub.groupby("exit_stage"):
            stage_exit_rows.append(
                {
                    "method": method,
                    "split": split,
                    "exit_stage": stage,
                    "count": len(stage_sub),
                    "pct": len(stage_sub) / len(sub),
                    "fp_count": int(((stage_sub["label"] == 0) & (stage_sub["final_pred"] == 1)).sum()),
                    "fn_count": int(((stage_sub["label"] == 1) & (stage_sub["final_pred"] == 0)).sum()),
                    "correct_count": int((stage_sub["final_pred"] == stage_sub["label"]).sum()),
                }
            )
    metrics_df = pd.DataFrame(metrics_rows)
    stage_exit_df = pd.DataFrame(stage_exit_rows)
    metrics_df.to_csv(EXP_DIR / "metrics_summary.csv", index=False)
    stage_exit_df.to_csv(EXP_DIR / "stage_exit_counts.csv", index=False)

    adaptive_test = track_level[(track_level["method"] == "adaptive_eps_0.01") & (track_level["split"] == "test")].sort_values("track_key")
    baseline_methods = ["single_bbox_5_15", "single_bbox_30_60", "single_bbox_15_30_corrected", "staged_011_primary"]
    bootstrap_frames = []
    for base in baseline_methods:
        base_test = track_level[(track_level["method"] == base) & (track_level["split"] == "test")].sort_values("track_key")
        df = bootstrap_deltas(adaptive_test, base_test)
        df.insert(0, "comparison", f"adaptive_eps_0.01_minus_{base}")
        bootstrap_frames.append(df)
    bootstrap_df = pd.concat(bootstrap_frames, ignore_index=True)
    bootstrap_df.to_csv(EXP_DIR / "bootstrap_comparisons.csv", index=False)

    runtime_df = pd.read_csv(RUNTIME_TABLE)
    bbox_runtime = runtime_df[(runtime_df["dataset"] == "JAAD") & (runtime_df["feature_set"] == "bbox")].iloc[0]
    per_stage_ms = float(bbox_runtime["downstream_mean_lower_bound_ms_per_sample"])
    runtime_rows = []
    for split in ["val", "test"]:
        for method in ["single_bbox_30_60", "single_bbox_15_30_corrected", "single_bbox_5_15", "staged_011_primary", "adaptive_eps_0.00", "adaptive_eps_0.01", "adaptive_eps_0.02"]:
            sub = track_level[(track_level["split"] == split) & (track_level["method"] == method)]
            runtime_rows.append(
                {
                    "method": method,
                    "split": split,
                    "n_tracks": len(sub),
                    "avg_stages_evaluated": float(sub["stages_evaluated"].mean()),
                    "estimated_classifier_lower_bound_ms_per_track": float(sub["stages_evaluated"].mean() * per_stage_ms),
                    "normalized_stage_cost_units": float(sub["stages_evaluated"].mean()),
                    "source_runtime_row": str(RUNTIME_TABLE),
                    "note": "Uses JAAD bbox classifier replay lower bound from the existing runtime benchmark; excludes bbox extraction and all upstream perception.",
                }
            )
    pd.DataFrame(runtime_rows).to_csv(EXP_DIR / "runtime_cost_estimate.csv", index=False)

    used_files = [
        str(TRACK_011),
        str(RESULTS_011),
        str(OPS_011),
        str(MODELSEL_011),
        str(TRACK_021),
        str(RESULTS_021),
        str(MODELSEL_021),
        str(RUNTIME_TABLE),
    ]
    write_text(EXP_DIR / "files_used.txt", "\n".join(used_files) + "\n")
    write_text(
        EXP_DIR / "commands_run.txt",
        "python3 scripts/experiments/jaad_adaptive_early_decision.py\n",
    )

    primary = selected_df[selected_df["epsilon"] == 0.01].iloc[0]
    test_metrics = metrics_df[(metrics_df["method"] == "adaptive_eps_0.01") & (metrics_df["split"] == "test")].iloc[0]
    late_metrics = metrics_df[(metrics_df["method"] == "single_bbox_5_15") & (metrics_df["split"] == "test")].iloc[0]

    verdict = (
        f"# Verdict\n\n"
        f"- Selected adaptive policy: `L1={primary.l1:.2f}, U1={primary.u1:.2f}, L2={primary.l2:.2f}, U2={primary.u2:.2f}, T3={primary.t3:.4f}` under the `<= 0.01` validation-F1 loss constraint.\n"
        f"- Test F1: `{test_metrics['f1']:.4f}` vs late-only 5_15 bbox `{late_metrics['f1']:.4f}` on the matched common-track population.\n"
        f"- Mean decision TTE: `{test_metrics['mean_decision_tte']:.2f}` frames; finalized at 30_60 / 15_30 / 5_15 = `{100*test_metrics['finalized_30_60_pct']:.1f}% / {100*test_metrics['finalized_15_30_corrected_pct']:.1f}% / {100*test_metrics['finalized_5_15_pct']:.1f}%`.\n"
        f"- Interpretation: this is an earliest-confident BBox-only policy, not a replacement for the stronger late-only 5_15 bbox model. Its value is earlier commitment of easy cases on a validation-constrained operating point.\n"
    )
    write_text(EXP_DIR / "verdict.md", verdict)

    readme = (
        "# Experiment 024: JAAD adaptive early decision\n\n"
        "## Question\n\n"
        "Can a validation-constrained multi-horizon BBox policy finalize easy tracks earlier while retaining nearly the same validation F1 as the late-only 5_15 BBox reference?\n\n"
        "## Inputs used\n\n"
        f"- `{TRACK_011}`\n"
        f"- `{TRACK_021}`\n"
        f"- `{OPS_011}`\n"
        f"- `{RUNTIME_TABLE}`\n\n"
        "## How to rerun\n\n"
        "```bash\n"
        "cd /home/abdelrahman/Hadwa/BIU-PID-Distillation-clean\n"
        "python3 scripts/experiments/jaad_adaptive_early_decision.py\n"
        "```\n\n"
        "## Outputs\n\n"
        "- `model_selection.csv`\n"
        "- `threshold_search_validation.csv`\n"
        "- `selected_operating_points.csv`\n"
        "- `track_level_decisions.csv`\n"
        "- `stage_exit_counts.csv`\n"
        "- `metrics_summary.csv`\n"
        "- `bootstrap_comparisons.csv`\n"
        "- `missing_track_audit.csv`\n"
        "- `runtime_cost_estimate.csv`\n"
        "- `figures/`\n"
        "- `verdict.md`\n\n"
        "## Final note\n\n"
        "This experiment uses saved canonical track-level probabilities only; no classifier is retrained here.\n"
    )
    write_text(EXP_DIR / "README.md", readme)

    # Figures
    test_sel = metrics_df[(metrics_df["split"] == "test") & (metrics_df["method"].isin(["single_bbox_30_60", "single_bbox_15_30_corrected", "single_bbox_5_15", "adaptive_eps_0.00", "adaptive_eps_0.01", "adaptive_eps_0.02", "staged_011_primary"]))].copy()

    plt.figure(figsize=(7, 4))
    plt.scatter(test_sel["mean_decision_tte"], test_sel["f1"])
    for _, row in test_sel.iterrows():
        plt.annotate(row["method"], (row["mean_decision_tte"], row["f1"]), fontsize=7)
    plt.xlabel("Mean decision TTE")
    plt.ylabel("Test F1")
    plt.title("Adaptive vs baseline F1 / earliness")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "f1_vs_mean_decision_tte.png", dpi=180)
    plt.close()

    stage_plot = stage_exit_df[(stage_exit_df["split"] == "test") & (stage_exit_df["method"].isin(["adaptive_eps_0.00", "adaptive_eps_0.01", "adaptive_eps_0.02"]))].copy()
    pivot = stage_plot.pivot(index="method", columns="exit_stage", values="pct").fillna(0)[["30_60", "15_30_corrected", "5_15"]]
    pivot.plot(kind="bar", stacked=True, figsize=(7, 4))
    plt.ylabel("Fraction of tracks")
    plt.title("Fraction finalized by horizon")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fraction_finalized_by_horizon.png", dpi=180)
    plt.close()

    pr_df = test_sel[["method", "precision", "recall", "mean_decision_tte"]].copy()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(pr_df["mean_decision_tte"], pr_df["precision"], label="precision")
    ax.scatter(pr_df["mean_decision_tte"], pr_df["recall"], label="recall")
    for _, row in pr_df.iterrows():
        ax.annotate(row["method"], (row["mean_decision_tte"], row["precision"]), fontsize=7)
    ax.set_xlabel("Mean decision TTE")
    ax.set_ylabel("Score")
    ax.set_title("Precision/recall vs earliness")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "precision_recall_vs_earliness.png", dpi=180)
    plt.close()

    err = stage_exit_df[(stage_exit_df["split"] == "test") & (stage_exit_df["method"] == "adaptive_eps_0.01")].copy()
    plt.figure(figsize=(7, 4))
    width = 0.35
    x = np.arange(len(err))
    plt.bar(x - width / 2, err["fp_count"], width, label="FP")
    plt.bar(x + width / 2, err["fn_count"], width, label="FN")
    plt.xticks(x, err["exit_stage"])
    plt.ylabel("Count")
    plt.title("Adaptive stage-wise error composition")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "stagewise_error_composition.png", dpi=180)
    plt.close()


if __name__ == "__main__":
    main()
