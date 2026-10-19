from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None


os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

REPO = Path("/home/abdelrahman/Hadwa/BIU-PID-Distillation-clean")
ARCHIVE = Path("/home/abdelrahman/Hadwa/BIU-PID-Distillation")
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from biu_v2_extractor import extract_biu_v2  # noqa: E402


OUT_DIR = REPO / "experiments/025_jaad_commitment_transition_specialist"
FIG_DIR = OUT_DIR / "figures"
SEED = 12345
RNG = np.random.default_rng(SEED)

ADAPTIVE_SELECTED = REPO / "experiments/024_jaad_adaptive_early_decision/selected_operating_points.csv"
ADAPTIVE_METRICS = REPO / "experiments/024_jaad_adaptive_early_decision/metrics_summary.csv"

SPLIT_CONFIGS = {
    "30_60": {
        "split_dir": ARCHIVE / "data/jaad_splits_event_16_30_60_overlap08_default",
        "bbox_dir": ARCHIVE / "data/jaad_features_event_16_30_60_overlap08_default",
        "keypoints_jsonl": ARCHIVE / "data/jaad_keypoints_vitpose_s_extended_short_horizon.jsonl",
    },
    "15_30_corrected": {
        "split_dir": REPO / "experiments/021_jaad_corrected_15_30_rebuild/data/tte_15_30_corrected",
        "bbox_dir": REPO / "experiments/021_jaad_corrected_15_30_rebuild/data/features_tte_15_30_corrected",
        "keypoints_jsonl": ARCHIVE / "data/jaad_keypoints_vitpose_s_extended_short_horizon.jsonl",
    },
    "5_15": {
        "split_dir": ARCHIVE / "data/jaad_splits_stage_horizon_audit/tte_5_15",
        "bbox_dir": ARCHIVE / "data/jaad_features_stage_horizon_audit/tte_5_15",
        "keypoints_jsonl": ARCHIVE / "data/jaad_keypoints_vitpose_s_extended_short_horizon.jsonl",
    },
}

META_COLS = {
    "sequence_id",
    "split",
    "label",
    "set_id",
    "video_id",
    "ped_id",
    "track_id",
    "start_frame",
    "end_frame",
    "event_frame",
    "last_observed_frame",
    "tte_frames",
    "observation_length",
    "frame_step",
    "sample_stride_between_windows",
    "frame_sampling",
    "onset_source",
    "negative_alignment_source",
    "track_length",
    "crossing_attr",
    "crossing_point",
    "decision_point",
    "has_observed_crossing",
    "observed_frames_json",
    "observed_crosses_json",
    "bbox_sequence_json",
    "source_annotation_path",
    "source_attributes_path",
    "parser_mode",
    "group",
}


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def classify_group(row: pd.Series) -> str:
    if int(row["label"]) == 1:
        return "actual_positive"
    crossing_attr = int(row["crossing_attr"]) if not pd.isna(row.get("crossing_attr")) else -1
    crossing_point = float(row["crossing_point"]) if not pd.isna(row.get("crossing_point")) else -1
    decision_point = float(row["decision_point"]) if not pd.isna(row.get("decision_point")) else -1
    ambiguity = (crossing_attr == 0) or (crossing_point > 0) or (decision_point > 0)
    return "ambiguous_negative" if ambiguity else "clean_negative"


def load_split_rows(horizon: str) -> pd.DataFrame:
    split_dir = SPLIT_CONFIGS[horizon]["split_dir"]
    dfs = []
    for split in ["train", "val", "test"]:
        df = pd.read_csv(split_dir / f"{split}.csv")
        df["split"] = split
        dfs.append(df)
    out = pd.concat(dfs, ignore_index=True)
    out["group"] = out.apply(classify_group, axis=1)
    out["horizon_label"] = horizon
    return out


def canonical_rows(df: pd.DataFrame) -> pd.DataFrame:
    ordered = df.sort_values(["split", "track_id", "tte_frames", "start_frame"], ascending=[True, True, True, False]).copy()
    out = ordered.drop_duplicates(subset=["split", "track_id"], keep="first").copy()
    return out


def load_bbox_canonical(horizon: str, canonical: pd.DataFrame) -> pd.DataFrame:
    bbox_dir = SPLIT_CONFIGS[horizon]["bbox_dir"]
    dfs = []
    for split in ["train", "val", "test"]:
        feat = pd.read_csv(bbox_dir / f"{split}_bbox.csv")
        keep = canonical[canonical["split"] == split][
            ["sequence_id", "split", "track_id", "group", "label", "video_id", "ped_id", "tte_frames", "start_frame"]
        ]
        merged = keep.merge(feat, on=["sequence_id", "track_id", "label", "split", "video_id", "ped_id", "start_frame"], how="left")
        dfs.append(merged)
    out = pd.concat(dfs, ignore_index=True)
    return out


def json_list(val: Any) -> list[Any]:
    if isinstance(val, list):
        return val
    if pd.isna(val):
        return []
    return json.loads(val)


def needed_keys(rows: pd.DataFrame) -> set[tuple[str, str, int]]:
    keys: set[tuple[str, str, int]] = set()
    for _, row in rows.iterrows():
        for frame in json_list(row["observed_frames_json"]):
            keys.add((str(row["video_id"]), str(row["ped_id"]), int(frame)))
    return keys


def load_keypoint_index(jsonl_path: Path, needed: set[tuple[str, str, int]]) -> dict[tuple[str, str, int], dict[str, Any]]:
    out = {}
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            rec = json.loads(line)
            key = (str(rec["video"]), str(rec["pid"]), int(rec["frame_idx"]))
            if key in needed:
                out[key] = rec
    return out


def extract_biu_v2_rows(rows: pd.DataFrame, keypoint_index: dict[tuple[str, str, int], dict[str, Any]]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    extracted = []
    missing = []
    for _, row in rows.iterrows():
        frames = json_list(row["observed_frames_json"])
        kp_rows = []
        missing_frames = []
        for frame in frames:
            item = keypoint_index.get((str(row["video_id"]), str(row["ped_id"]), int(frame)))
            if item is None:
                missing_frames.append(int(frame))
            else:
                kp_rows.append(item)
        if missing_frames:
            missing.append(
                {
                    "horizon_label": row["horizon_label"],
                    "split": row["split"],
                    "track_id": row["track_id"],
                    "sequence_id": row["sequence_id"],
                    "missing_frames": json.dumps(missing_frames),
                }
            )
            continue
        coords = np.asarray([item["keypoints"] for item in kp_rows], dtype=float)
        conf = np.asarray([item["kp_conf"] for item in kp_rows], dtype=float)
        bbox = np.asarray(json_list(row["bbox_sequence_json"]), dtype=float)
        feats = extract_biu_v2(coords, conf, bbox_sequence=bbox)
        feats = {k: v for k, v in feats.items() if isinstance(v, (int, float, np.integer, np.floating))}
        feats.update(
            {
                "sequence_id": row["sequence_id"],
                "split": row["split"],
                "track_id": row["track_id"],
                "label": int(row["label"]),
                "group": row["group"],
                "video_id": row["video_id"],
                "ped_id": row["ped_id"],
                "tte_frames": int(row["tte_frames"]),
                "start_frame": int(row["start_frame"]),
            }
        )
        extracted.append(feats)
    return pd.DataFrame(extracted), missing


def build_model(model_name: str):
    if model_name == "logreg":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)),
            ]
        )
    if model_name == "rf":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", RandomForestClassifier(n_estimators=400, min_samples_leaf=2, class_weight="balanced", random_state=SEED, n_jobs=-1)),
            ]
        )
    if model_name == "xgb":
        if XGBClassifier is None:
            raise RuntimeError("xgboost unavailable")
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=250,
                        max_depth=4,
                        learning_rate=0.05,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        reg_lambda=1.0,
                        eval_metric="logloss",
                        random_state=SEED,
                    ),
                ),
            ]
        )
    raise ValueError(model_name)


def best_f1_threshold(y: np.ndarray, prob: np.ndarray) -> tuple[float, float]:
    candidates = np.unique(np.concatenate(([0.0], prob, [1.0])))
    best_t = 0.5
    best_f1 = -1.0
    for t in candidates:
        pred = (prob >= t).astype(int)
        f1 = f1_score(y, pred, zero_division=0)
        if f1 > best_f1 or (np.isclose(f1, best_f1) and t < best_t):
            best_f1 = f1
            best_t = float(t)
    return best_t, best_f1


def threshold_for_recall(y: np.ndarray, prob: np.ndarray, target_recall: float) -> float:
    candidates = np.unique(np.concatenate(([0.0], prob, [1.0])))
    best = None
    best_gap = float("inf")
    for t in candidates:
        pred = (prob >= t).astype(int)
        rec = recall_score(y, pred, zero_division=0)
        gap = abs(rec - target_recall)
        if gap < best_gap or (np.isclose(gap, best_gap) and (best is None or t < best)):
            best_gap = gap
            best = float(t)
    return 0.5 if best is None else best


def metric_row(y: np.ndarray, pred: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    tp = int(((y == 1) & (pred == 1)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "accuracy": float((pred == y).mean()),
        "balanced_accuracy": float((recall + spec) / 2.0),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc_score(y, prob)) if len(np.unique(y)) > 1 else float("nan"),
        "pr_auc": float(average_precision_score(y, prob)) if len(np.unique(y)) > 1 else float("nan"),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "fpr": float(fp / (fp + tn)) if (fp + tn) else float("nan"),
    }


def fit_eval_feature_family(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    horizon: str,
    population: str,
    family: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    preds = []
    for model_name in ["logreg", "rf", "xgb"]:
        if model_name == "xgb" and XGBClassifier is None:
            continue
        model = build_model(model_name)
        X_train = train_df[feature_cols].to_numpy(dtype=float)
        y_train = train_df["label"].to_numpy(dtype=int)
        X_val = val_df[feature_cols].to_numpy(dtype=float)
        y_val = val_df["label"].to_numpy(dtype=int)
        X_test = test_df[feature_cols].to_numpy(dtype=float)
        y_test = test_df["label"].to_numpy(dtype=int)
        model.fit(X_train, y_train)
        val_prob = model.predict_proba(X_val)[:, 1]
        test_prob = model.predict_proba(X_test)[:, 1]
        t_f1, val_f1 = best_f1_threshold(y_val, val_prob)
        val_pred = (val_prob >= t_f1).astype(int)
        test_pred = (test_prob >= t_f1).astype(int)
        val_metrics = metric_row(y_val, val_pred, val_prob)
        test_metrics = metric_row(y_test, test_pred, test_prob)
        rows.append(
            {
                "later_horizon": horizon,
                "population": population,
                "feature_family": family,
                "model_name": model_name,
                "n_features": len(feature_cols),
                "threshold_rule": "val_f1",
                "selected_threshold": t_f1,
                **{f"val_{k}": v for k, v in val_metrics.items()},
                **{f"test_{k}": v for k, v in test_metrics.items()},
            }
        )
        for split_name, frame, prob, pred in [("val", val_df, val_prob, val_pred), ("test", test_df, test_prob, test_pred)]:
            tmp = frame[["track_id", "split", "label", "group", "video_id", "ped_id"]].copy()
            tmp["later_horizon"] = horizon
            tmp["population"] = population
            tmp["feature_family"] = family
            tmp["model_name"] = model_name
            tmp["threshold_rule"] = "val_f1"
            tmp["selected_threshold"] = t_f1
            tmp["prob"] = prob
            tmp["pred"] = pred
            preds.append(tmp)
    return pd.DataFrame(rows), pd.concat(preds, ignore_index=True)


def build_feature_families(df: pd.DataFrame) -> dict[str, list[str]]:
    bbox_early = sorted([c for c in df.columns if c.startswith("e_bbox_")])
    bbox_late = sorted([c for c in df.columns if c.startswith("l_bbox_")])
    bbox_delta = sorted([c for c in df.columns if c.startswith("d_bbox_")])
    v2_delta = sorted([c for c in df.columns if c.startswith("d_v2_") or c.startswith("a_v2_")])
    return {
        "later_bbox_only": bbox_late,
        "early_bbox_plus_later_bbox": bbox_early + bbox_late,
        "bbox_transition": bbox_delta,
        "biu_v2_transition": v2_delta,
        "combined_bbox_biu_v2_transition": bbox_delta + v2_delta,
        "combined_transitions_plus_stage1_prob": bbox_delta + v2_delta + ["stage1_prob"],
    }


def bootstrap_compare(a: pd.DataFrame, b: pd.DataFrame, n_boot: int = 10_000) -> pd.DataFrame:
    a = a.sort_values("track_id").reset_index(drop=True)
    b = b.sort_values("track_id").reset_index(drop=True)
    assert list(a["track_id"]) == list(b["track_id"])
    y_a = a["label"].to_numpy(dtype=int)
    p_a = a["pred"].to_numpy(dtype=int)
    g_a = a["group"].to_numpy(dtype=object)
    y_b = b["label"].to_numpy(dtype=int)
    p_b = b["pred"].to_numpy(dtype=int)
    g_b = b["group"].to_numpy(dtype=object)
    idx_samples = RNG.integers(0, len(a), size=(n_boot, len(a)))
    rows = []
    for metric in ["f1", "precision", "recall", "ambiguous_negative_fpr", "clean_negative_fpr"]:
        deltas = np.empty(n_boot, dtype=float)
        for i, sample in enumerate(idx_samples):
            deltas[i] = subgroup_metric(y_a[sample], p_a[sample], g_a[sample], metric) - subgroup_metric(y_b[sample], p_b[sample], g_b[sample], metric)
        observed = subgroup_metric(y_a, p_a, g_a, metric) - subgroup_metric(y_b, p_b, g_b, metric)
        rows.append(
            {
                "metric": metric,
                "delta_observed": observed,
                "ci_low": float(np.quantile(deltas, 0.025)),
                "ci_high": float(np.quantile(deltas, 0.975)),
                "p_value": float(min(1.0, 2 * min((deltas <= 0).mean(), (deltas >= 0).mean()))),
                "bootstrap_seed": SEED,
                "n_bootstrap": n_boot,
            }
        )
    return pd.DataFrame(rows)


def subgroup_metric(y: np.ndarray, pred: np.ndarray, groups: np.ndarray, metric: str) -> float:
    tp = int(((y == 1) & (pred == 1)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    if metric == "f1":
        denom = 2 * tp + fp + fn
        return float((2 * tp) / denom) if denom else 0.0
    if metric == "precision":
        denom = tp + fp
        return float(tp / denom) if denom else 0.0
    if metric == "recall":
        denom = tp + fn
        return float(tp / denom) if denom else 0.0
    if metric == "ambiguous_negative_fpr":
        mask = groups == "ambiguous_negative"
        return float(((pred[mask] == 1) & (y[mask] == 0)).mean()) if mask.any() else float("nan")
    if metric == "clean_negative_fpr":
        mask = groups == "clean_negative"
        return float(((pred[mask] == 1) & (y[mask] == 0)).mean()) if mask.any() else float("nan")
    raise ValueError(metric)


def build_population_tables(later_horizon: str, l1: float, u1: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    early_rows = canonical_rows(load_split_rows("30_60"))
    later_rows = canonical_rows(load_split_rows(later_horizon))

    early_bbox = load_bbox_canonical("30_60", early_rows)
    later_bbox = load_bbox_canonical(later_horizon, later_rows)

    common_keys = sorted(set(early_bbox["track_id"]).intersection(set(later_bbox["track_id"])))
    early_rows = early_rows[early_rows["track_id"].isin(common_keys)].copy()
    later_rows = later_rows[later_rows["track_id"].isin(common_keys)].copy()
    early_bbox = early_bbox[early_bbox["track_id"].isin(common_keys)].copy()
    later_bbox = later_bbox[later_bbox["track_id"].isin(common_keys)].copy()

    needed = needed_keys(pd.concat([early_rows, later_rows], ignore_index=True))
    kp_index = load_keypoint_index(SPLIT_CONFIGS[later_horizon]["keypoints_jsonl"], needed)
    early_v2, early_missing = extract_biu_v2_rows(early_rows, kp_index)
    later_v2, later_missing = extract_biu_v2_rows(later_rows, kp_index)
    missing_df = pd.DataFrame(early_missing + later_missing)

    early_bbox_cols = [c for c in early_bbox.columns if c not in META_COLS and pd.api.types.is_numeric_dtype(early_bbox[c])]
    later_bbox_cols = [c for c in later_bbox.columns if c not in META_COLS and pd.api.types.is_numeric_dtype(later_bbox[c])]
    v2_cols = sorted(set(c for c in early_v2.columns if c not in {"sequence_id", "split", "track_id", "label", "group", "video_id", "ped_id", "tte_frames", "start_frame"} and pd.api.types.is_numeric_dtype(early_v2[c])) & set(c for c in later_v2.columns if c not in {"sequence_id", "split", "track_id", "label", "group", "video_id", "ped_id", "tte_frames", "start_frame"} and pd.api.types.is_numeric_dtype(later_v2[c])))

    early_stage = early_bbox[["track_id", "split", "label", "group", "video_id", "ped_id"] + early_bbox_cols].copy()
    later_stage = later_bbox[["track_id", "split", "label", "group", "video_id", "ped_id"] + later_bbox_cols].copy()
    e_v2 = early_v2[["track_id", "split"] + v2_cols].copy()
    l_v2 = later_v2[["track_id", "split"] + v2_cols].copy()

    early_stage = early_stage.rename(columns={c: f"e_bbox_{c}" for c in early_bbox_cols})
    later_stage = later_stage.rename(columns={c: f"l_bbox_{c}" for c in later_bbox_cols})
    e_v2 = e_v2.rename(columns={c: f"e_v2_{c}" for c in v2_cols})
    l_v2 = l_v2.rename(columns={c: f"l_v2_{c}" for c in v2_cols})

    merged = early_stage.merge(later_stage, on=["track_id", "split", "label", "group", "video_id", "ped_id"], how="inner")
    merged = merged.merge(e_v2, on=["track_id", "split"], how="left")
    merged = merged.merge(l_v2, on=["track_id", "split"], how="left")

    common_bbox = sorted(set(early_bbox_cols).intersection(later_bbox_cols))
    for col in common_bbox:
        merged[f"d_bbox_{col}"] = merged[f"l_bbox_{col}"] - merged[f"e_bbox_{col}"]
    for col in v2_cols:
        merged[f"d_v2_{col}"] = merged[f"l_v2_{col}"] - merged[f"e_v2_{col}"]
        merged[f"a_v2_{col}"] = (merged[f"d_v2_{col}"]).abs()

    # Stage-1 refit probabilities for train population definition; val/test later compared on exact deferred set rule.
    stage1_features = [c for c in early_stage.columns if c.startswith("e_bbox_")]
    train_mask = merged["split"] == "train"
    stage1_model = build_model("logreg")
    stage1_model.fit(merged.loc[train_mask, stage1_features], merged.loc[train_mask, "label"])
    merged["stage1_prob"] = stage1_model.predict_proba(merged[stage1_features])[:, 1]
    merged["stage1_population"] = np.where(
        merged["stage1_prob"] <= l1,
        "stage1_negative",
        np.where(merged["stage1_prob"] >= u1, "stage1_positive", "stage1_deferred"),
    )

    # Motion and confidence subgroup helpers.
    motion_col = "e_bbox_center_x_displacement" if "e_bbox_center_x_displacement" in merged.columns else stage1_features[0]
    train_motion = merged.loc[train_mask, motion_col]
    motion_thresh = float(train_motion.median())
    merged["motion_band"] = np.where(merged[motion_col] >= motion_thresh, "high_motion", "low_motion")
    if "e_v2_skeleton_conf_mean_mean" in merged.columns:
        conf_col = "e_v2_skeleton_conf_mean_mean"
    else:
        candidates = [c for c in merged.columns if c.startswith("e_v2_") and "conf" in c]
        conf_col = candidates[0] if candidates else None
    if conf_col is not None:
        conf_thresh = float(merged.loc[train_mask, conf_col].median())
        merged["confidence_band"] = np.where(merged[conf_col] >= conf_thresh, "high_confidence", "low_confidence")
    else:
        merged["confidence_band"] = "confidence_unavailable"

    merged["later_horizon"] = later_horizon
    return merged, missing_df


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    selected = pd.read_csv(ADAPTIVE_SELECTED)
    primary = selected[selected["epsilon"] == 0.01].iloc[0]
    l1, u1 = float(primary["l1"]), float(primary["u1"])

    benchmark_rows = []
    pred_rows = []
    population_rows = []
    dict_rows = []
    subgroup_rows = []
    matched_rows = []
    bootstrap_rows = []
    feature_summary_rows = []
    all_missing = []
    merged_by_horizon: dict[str, pd.DataFrame] = {}

    for later_horizon in ["15_30_corrected", "5_15"]:
        merged, missing_df = build_population_tables(later_horizon, l1, u1)
        merged_by_horizon[later_horizon] = merged.copy()
        if not missing_df.empty:
            missing_df["later_horizon"] = later_horizon
            all_missing.append(missing_df)

        feature_families = build_feature_families(merged)
        for family, cols in feature_families.items():
            for col in cols[:10]:
                dict_rows.append(
                    {
                        "later_horizon": later_horizon,
                        "feature_family": family,
                        "feature_name": col,
                        "definition": "auto-generated transition feature",
                        "source_columns": col.replace("d_", "").replace("a_", ""),
                        "temporal_direction": f"30_60_to_{later_horizon}",
                        "units_or_normalization": "inherited_from_source_feature",
                        "missing_value_handling": "train_median_imputation",
                        "leakage_assessment": "observation_only",
                    }
                )
            feature_summary_rows.append({"later_horizon": later_horizon, "feature_family": family, "n_features": len(cols)})

        for population_name in ["stage1_deferred"]:
            pop = merged[merged["stage1_population"] == population_name].copy()
            train_df = pop[pop["split"] == "train"].copy()
            val_df = pop[pop["split"] == "val"].copy()
            test_df = pop[pop["split"] == "test"].copy()
            population_rows.append(
                {
                    "later_horizon": later_horizon,
                    "population": population_name,
                    "train_rows": len(train_df),
                    "val_rows": len(val_df),
                    "test_rows": len(test_df),
                    "train_pos": int(train_df["label"].sum()),
                    "val_pos": int(val_df["label"].sum()),
                    "test_pos": int(test_df["label"].sum()),
                }
            )
            for family, cols in feature_families.items():
                rows, preds = fit_eval_feature_family(train_df, val_df, test_df, cols, later_horizon, population_name, family)
                benchmark_rows.append(rows)
                pred_rows.append(preds)

        # Sensitivity populations for only the primary comparison pair.
        for population_name in ["stage1_positive", "all_common"]:
            pop = merged.copy() if population_name == "all_common" else merged[merged["stage1_population"] == population_name].copy()
            train_df = pop[pop["split"] == "train"].copy()
            val_df = pop[pop["split"] == "val"].copy()
            test_df = pop[pop["split"] == "test"].copy()
            population_rows.append(
                {
                    "later_horizon": later_horizon,
                    "population": population_name,
                    "train_rows": len(train_df),
                    "val_rows": len(val_df),
                    "test_rows": len(test_df),
                    "train_pos": int(train_df["label"].sum()),
                    "val_pos": int(val_df["label"].sum()),
                    "test_pos": int(test_df["label"].sum()),
                }
            )
            for family in ["later_bbox_only", "combined_transitions_plus_stage1_prob"]:
                rows, preds = fit_eval_feature_family(train_df, val_df, test_df, feature_families[family], later_horizon, population_name, family)
                benchmark_rows.append(rows)
                pred_rows.append(preds)

    benchmark_df = pd.concat(benchmark_rows, ignore_index=True)
    pred_df = pd.concat(pred_rows, ignore_index=True)
    benchmark_df.to_csv(OUT_DIR / "benchmark_results.csv", index=False)
    benchmark_df.groupby(["later_horizon", "population", "feature_family"]).agg(
        mean_test_f1=("test_f1", "mean"),
        std_test_f1=("test_f1", "std"),
        mean_test_pr_auc=("test_pr_auc", "mean"),
        std_test_pr_auc=("test_pr_auc", "std"),
    ).reset_index().to_csv(OUT_DIR / "benchmark_results_mean_std.csv", index=False)
    pd.DataFrame(dict_rows).drop_duplicates().to_csv(OUT_DIR / "transition_feature_dictionary.csv", index=False)
    pd.DataFrame(population_rows).to_csv(OUT_DIR / "population_definition.csv", index=False)
    pred_df.to_csv(OUT_DIR / "track_level_predictions.csv", index=False)
    pd.DataFrame(feature_summary_rows).to_csv(OUT_DIR / "feature_family_summary.csv", index=False)
    if all_missing:
        pd.concat(all_missing, ignore_index=True).to_csv(OUT_DIR / "missing_biu_v2_rows.csv", index=False)

    # Primary population selected-model summary and matched-recall comparison.
    selected_models = []
    for later_horizon in ["15_30_corrected", "5_15"]:
        primary_df = benchmark_df[(benchmark_df["later_horizon"] == later_horizon) & (benchmark_df["population"] == "stage1_deferred")]
        for family in [
            "later_bbox_only",
            "early_bbox_plus_later_bbox",
            "bbox_transition",
            "biu_v2_transition",
            "combined_bbox_biu_v2_transition",
            "combined_transitions_plus_stage1_prob",
        ]:
            sub = primary_df[primary_df["feature_family"] == family].sort_values(["val_f1", "test_f1"], ascending=False)
            selected_models.append(sub.iloc[0].to_dict())
    selected_df = pd.DataFrame(selected_models)
    selected_df.to_csv(OUT_DIR / "model_selection.csv", index=False)

    for later_horizon in ["15_30_corrected", "5_15"]:
        bbox_row = selected_df[(selected_df["later_horizon"] == later_horizon) & (selected_df["feature_family"] == "later_bbox_only")].iloc[0]
        combo_row = selected_df[(selected_df["later_horizon"] == later_horizon) & (selected_df["feature_family"] == "combined_transitions_plus_stage1_prob")].iloc[0]

        bbox_pred = pred_df[
            (pred_df["later_horizon"] == later_horizon)
            & (pred_df["population"] == "stage1_deferred")
            & (pred_df["feature_family"] == "later_bbox_only")
            & (pred_df["model_name"] == bbox_row["model_name"])
        ].copy()
        combo_pred = pred_df[
            (pred_df["later_horizon"] == later_horizon)
            & (pred_df["population"] == "stage1_deferred")
            & (pred_df["feature_family"] == "combined_transitions_plus_stage1_prob")
            & (pred_df["model_name"] == combo_row["model_name"])
        ].copy()

        val_bbox = bbox_pred[bbox_pred["split"] == "val"].copy()
        val_combo = combo_pred[combo_pred["split"] == "val"].copy()
        target_recall = recall_score(val_bbox["label"], val_bbox["pred"], zero_division=0)
        combo_t = threshold_for_recall(val_combo["label"].to_numpy(dtype=int), val_combo["prob"].to_numpy(dtype=float), target_recall)
        test_bbox = bbox_pred[bbox_pred["split"] == "test"].copy()
        test_combo = combo_pred[combo_pred["split"] == "test"].copy()
        test_combo["matched_pred"] = (test_combo["prob"] >= combo_t).astype(int)
        matched_rows.append(
            {
                "later_horizon": later_horizon,
                "baseline_model": bbox_row["model_name"],
                "specialist_model": combo_row["model_name"],
                "baseline_test_f1": bbox_row["test_f1"],
                "specialist_test_f1_matched_recall": f1_score(test_combo["label"], test_combo["matched_pred"], zero_division=0),
                "baseline_test_recall": bbox_row["test_recall"],
                "specialist_test_recall_matched": recall_score(test_combo["label"], test_combo["matched_pred"], zero_division=0),
                "specialist_threshold_for_matched_recall": combo_t,
            }
        )
        boot = bootstrap_compare(
            test_combo.assign(pred=test_combo["matched_pred"])[["track_id", "label", "group", "pred"]],
            test_bbox[["track_id", "label", "group", "pred"]],
        )
        boot.insert(0, "later_horizon", later_horizon)
        bootstrap_rows.append(boot)

        for subgroup_col in ["group", "motion_band", "confidence_band"]:
            merged = merged_by_horizon[later_horizon]
            test_sub = merged[(merged["split"] == "test") & (merged["stage1_population"] == "stage1_deferred")][["track_id", subgroup_col]].drop_duplicates()
            test_bbox_sub = test_bbox.copy()
            test_combo_sub = test_combo.copy()
            if subgroup_col not in test_bbox_sub.columns:
                test_bbox_sub = test_bbox_sub.merge(test_sub, on="track_id", how="left")
            if subgroup_col not in test_combo_sub.columns:
                test_combo_sub = test_combo_sub.merge(test_sub, on="track_id", how="left")
            for subgroup_value, group_rows in test_bbox_sub.groupby(subgroup_col):
                combo_group = test_combo_sub[test_combo_sub[subgroup_col] == subgroup_value]
                subgroup_rows.append(
                    {
                        "later_horizon": later_horizon,
                        "population": "stage1_deferred",
                        "subgroup_column": subgroup_col,
                        "subgroup_value": subgroup_value,
                        "baseline_f1": f1_score(group_rows["label"], group_rows["pred"], zero_division=0),
                        "specialist_f1": f1_score(combo_group["label"], combo_group["matched_pred"], zero_division=0),
                        "baseline_precision": precision_score(group_rows["label"], group_rows["pred"], zero_division=0),
                        "specialist_precision": precision_score(combo_group["label"], combo_group["matched_pred"], zero_division=0),
                        "baseline_recall": recall_score(group_rows["label"], group_rows["pred"], zero_division=0),
                        "specialist_recall": recall_score(combo_group["label"], combo_group["matched_pred"], zero_division=0),
                        "baseline_fpr": metric_row(group_rows["label"].to_numpy(dtype=int), group_rows["pred"].to_numpy(dtype=int), group_rows["prob"].to_numpy(dtype=float))["fpr"],
                        "specialist_fpr": metric_row(combo_group["label"].to_numpy(dtype=int), combo_group["matched_pred"].to_numpy(dtype=int), combo_group["prob"].to_numpy(dtype=float))["fpr"],
                        "n_tracks": len(group_rows),
                    }
                )

    pd.DataFrame(matched_rows).to_csv(OUT_DIR / "matched_recall_results.csv", index=False)
    pd.concat(bootstrap_rows, ignore_index=True).to_csv(OUT_DIR / "bootstrap_comparisons.csv", index=False)
    pd.DataFrame(subgroup_rows).to_csv(OUT_DIR / "subgroup_results.csv", index=False)

    # Simple figures.
    sel_plot = selected_df[selected_df["population"] == "stage1_deferred"].copy()
    plt.figure(figsize=(8, 4))
    for horizon, sub in sel_plot.groupby("later_horizon"):
        plt.plot(sub["feature_family"], sub["test_f1"], marker="o", label=horizon)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Test F1")
    plt.title("Later BBox vs transition specialist families")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "later_bbox_vs_transition_specialist.png", dpi=180)
    plt.close()

    subgroup_df = pd.DataFrame(subgroup_rows)
    amb = subgroup_df[subgroup_df["subgroup_column"] == "group"].copy()
    if not amb.empty:
        plt.figure(figsize=(8, 4))
        x = np.arange(len(amb))
        plt.bar(x - 0.15, amb["baseline_fpr"], 0.3, label="later_bbox")
        plt.bar(x + 0.15, amb["specialist_fpr"], 0.3, label="specialist")
        plt.xticks(x, amb["later_horizon"] + ":" + amb["subgroup_value"], rotation=45, ha="right")
        plt.ylabel("FPR")
        plt.title("Hard-subgroup FPR comparison")
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIG_DIR / "hard_subgroup_fpr_comparison.png", dpi=180)
        plt.close()

    feature_sources = [
        str(ADAPTIVE_SELECTED),
        *(str(SPLIT_CONFIGS[h]["split_dir"]) for h in SPLIT_CONFIGS),
        *(str(SPLIT_CONFIGS[h]["bbox_dir"]) for h in SPLIT_CONFIGS),
        *(str(SPLIT_CONFIGS[h]["keypoints_jsonl"]) for h in SPLIT_CONFIGS),
    ]
    write_text(OUT_DIR / "files_used.txt", "\n".join(feature_sources) + "\n")
    write_text(OUT_DIR / "commands_run.txt", "python3 scripts/experiments/jaad_commitment_transition_specialist.py\n")

    best_5 = selected_df[(selected_df["later_horizon"] == "5_15") & (selected_df["population"] == "stage1_deferred")].sort_values("test_f1", ascending=False).iloc[0]
    best_15 = selected_df[(selected_df["later_horizon"] == "15_30_corrected") & (selected_df["population"] == "stage1_deferred")].sort_values("test_f1", ascending=False).iloc[0]
    verdict = (
        "# Verdict\n\n"
        f"- Deferred-population best at `15_30_corrected`: `{best_15['feature_family']}` / `{best_15['model_name']}` with test F1 `{best_15['test_f1']:.4f}`.\n"
        f"- Deferred-population best at `5_15`: `{best_5['feature_family']}` / `{best_5['model_name']}` with test F1 `{best_5['test_f1']:.4f}`.\n"
        "- The main retained comparison is `combined_transitions_plus_stage1_prob` versus `later_bbox_only` at the same later horizon.\n"
        "- Read `matched_recall_results.csv` and `bootstrap_comparisons.csv` before claiming any transition gain as a thesis contribution.\n"
    )
    write_text(OUT_DIR / "verdict.md", verdict)

    report = (
        "# Experiment 025: JAAD commitment-transition specialist\n\n"
        "## Purpose\n\n"
        "Test whether explicit early-to-later transition features improve on the later-horizon BBox model for tracks deferred by the adaptive Stage-1 rule from experiment 024.\n\n"
        "## Population\n\n"
        f"- Primary Stage-1 defer rule from exp024: `L1={l1:.2f}`, `U1={u1:.2f}`.\n"
        "- Sensitivity populations: Stage-1 positives and all common tracks.\n\n"
        "## Later horizons\n\n"
        "- `15_30_corrected`\n"
        "- `5_15`\n\n"
        "## Outputs\n\n"
        "- `transition_feature_dictionary.csv`\n"
        "- `population_definition.csv`\n"
        "- `model_selection.csv`\n"
        "- `benchmark_results.csv`\n"
        "- `benchmark_results_mean_std.csv`\n"
        "- `matched_recall_results.csv`\n"
        "- `bootstrap_comparisons.csv`\n"
        "- `subgroup_results.csv`\n"
        "- `track_level_predictions.csv`\n"
        "- `verdict.md`\n"
    )
    write_text(OUT_DIR / "README.md", report)


if __name__ == "__main__":
    main()
