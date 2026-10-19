from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None

CLEAN_ROOT = Path("/home/abdelrahman/Hadwa/BIU-PID-Distillation-clean")
ARCHIVE_ROOT = Path("/home/abdelrahman/Hadwa/BIU-PID-Distillation")
SRC_ROOT = CLEAN_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from biu_v2_extractor import extract_biu_v2

OUT_ROOT = CLEAN_ROOT / "experiments/009_jaad_biu_v2_benchmark_comparison"
METRICS_DIR = OUT_ROOT / "metrics"
FIGURES_DIR = OUT_ROOT / "figures"
DICT_PATH = CLEAN_ROOT / "experiments/008_biu_v2_design_spec/metrics/biu_v2_feature_dictionary.csv"
REPORT_PATH = OUT_ROOT / "report.md"
README_PATH = OUT_ROOT / "README.md"

HORIZON_CONFIGS = {
    "30_60": {
        "split_dir": ARCHIVE_ROOT / "data/jaad_splits_event_16_30_60_overlap08_default",
        "feature_dir": ARCHIVE_ROOT / "data/jaad_features_event_16_30_60_overlap08_default",
        "keypoints_jsonl": ARCHIVE_ROOT / "data/jaad_keypoints_vitpose_s_extended_short_horizon.jsonl",
        "tte_min": 30,
        "tte_max": 60,
    },
    "15_30": {
        "split_dir": ARCHIVE_ROOT / "data/jaad_splits_stage_horizon_audit/tte_15_30",
        "feature_dir": ARCHIVE_ROOT / "data/jaad_features_stage_horizon_audit/tte_15_30",
        "keypoints_jsonl": ARCHIVE_ROOT / "data/jaad_keypoints_vitpose_s_extended_short_horizon.jsonl",
        "tte_min": 15,
        "tte_max": 30,
    },
    "5_15": {
        "split_dir": ARCHIVE_ROOT / "data/jaad_splits_stage_horizon_audit/tte_5_15",
        "feature_dir": ARCHIVE_ROOT / "data/jaad_features_stage_horizon_audit/tte_5_15",
        "keypoints_jsonl": ARCHIVE_ROOT / "data/jaad_keypoints_vitpose_s_extended_short_horizon.jsonl",
        "tte_min": 5,
        "tte_max": 15,
    },
}

SPLITS = ["train", "val", "test"]
SEEDS = [42, 43, 44]
MODEL_ORDER = ["logreg", "rf", "xgb"]
THRESHOLD_RULES = [
    "fixed_0_5",
    "val_f1",
    "val_balanced_accuracy",
    "val_fpr_le_0p1",
    "val_precision_ge_0p8",
]
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
    "event_frame_or_anchor",
    "horizon_label",
    "tte_min",
    "tte_max",
    "group",
    "group_notes",
    "canonical_selection_rule",
    "direction_sign",
    "direction_source",
    "was_flipped",
    "direction_confidence",
}

FEATURE_GROUP_MAP = {
    "original_anatomical_biu": "original_anatomical",
    "lower_body_commitment_biu": "lower_body",
    "torso_center_commitment_biu": "torso/body_center",
    "head_attention_proxy_biu": "head/attention",
    "scale_viewpoint_metadata": "scale/viewpoint",
    "confidence_metadata": "confidence",
}


@dataclass
class ThresholdSelection:
    rule: str
    threshold: float | None
    status: str
    val_metrics: dict[str, float]


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b else float("nan")


def _parse_json_list(s: Any) -> list[Any]:
    if isinstance(s, list):
        return s
    if pd.isna(s):
        return []
    return json.loads(s)


def _classify_group(row: pd.Series) -> str:
    if int(row["label"]) == 1:
        return "actual_positive"
    crossing_attr = int(row.get("crossing_attr", -1)) if not pd.isna(row.get("crossing_attr", -1)) else -1
    crossing_point = float(row.get("crossing_point", -1)) if not pd.isna(row.get("crossing_point", -1)) else -1
    decision_point = float(row.get("decision_point", -1)) if not pd.isna(row.get("decision_point", -1)) else -1
    ambiguity = (crossing_attr == 0) or (crossing_point > 0) or (decision_point > 0)
    return "negative_ambiguous" if ambiguity else "clean_negative"


def _load_split_rows(horizon_label: str) -> pd.DataFrame:
    cfg = HORIZON_CONFIGS[horizon_label]
    dfs = []
    for split in SPLITS:
        df = pd.read_csv(cfg["split_dir"] / f"{split}.csv")
        df["split"] = split
        dfs.append(df)
    out = pd.concat(dfs, ignore_index=True)
    out["horizon_label"] = horizon_label
    out["tte_min"] = cfg["tte_min"]
    out["tte_max"] = cfg["tte_max"]
    out["group"] = out.apply(_classify_group, axis=1)
    out["event_frame_or_anchor"] = out["event_frame"]
    return out


def _needed_frame_keys(split_rows: dict[str, pd.DataFrame]) -> set[tuple[str, str, int]]:
    needed: set[tuple[str, str, int]] = set()
    for df in split_rows.values():
        for _, row in df.iterrows():
            frames = _parse_json_list(row["observed_frames_json"])
            for frame_idx in frames:
                needed.add((str(row["video_id"]), str(row["ped_id"]), int(frame_idx)))
    return needed


def _load_keypoint_index(jsonl_path: Path, needed: set[tuple[str, str, int]]) -> dict[tuple[str, str, int], dict[str, Any]]:
    out: dict[tuple[str, str, int], dict[str, Any]] = {}
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            rec = json.loads(line)
            key = (str(rec["video"]), str(rec["pid"]), int(rec["frame_idx"]))
            if key in needed:
                out[key] = rec
    return out


def _extract_v2_for_rows(df: pd.DataFrame, kp_index: dict[tuple[str, str, int], dict[str, Any]]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    rows = []
    missing = []
    for _, rec in df.iterrows():
        frames = _parse_json_list(rec["observed_frames_json"])
        bbox_seq = np.asarray(_parse_json_list(rec["bbox_sequence_json"]), dtype=float)
        kp_rows = []
        missing_frames = []
        for frame in frames:
            item = kp_index.get((str(rec["video_id"]), str(rec["ped_id"]), int(frame)))
            if item is None:
                missing_frames.append(int(frame))
            else:
                kp_rows.append(item)
        if missing_frames:
            missing.append(
                {
                    "sequence_id": rec["sequence_id"],
                    "video_id": rec["video_id"],
                    "ped_id": rec["ped_id"],
                    "horizon_label": rec["horizon_label"],
                    "split": rec["split"],
                    "missing_frames": json.dumps(missing_frames),
                }
            )
            continue
        coords = np.asarray([r["keypoints"] for r in kp_rows], dtype=float)
        conf = np.asarray([r["kp_conf"] for r in kp_rows], dtype=float)
        feats = extract_biu_v2(coords, conf, bbox_sequence=bbox_seq)
        base = {
            "sequence_id": rec["sequence_id"],
            "split": rec["split"],
            "label": int(rec["label"]),
            "video_id": rec["video_id"],
            "ped_id": rec["ped_id"],
            "track_id": rec["track_id"],
            "start_frame": int(rec["start_frame"]),
            "end_frame": int(rec["end_frame"]),
            "group": rec["group"],
            "horizon_label": rec["horizon_label"],
        }
        base.update(feats)
        rows.append(base)
    return pd.DataFrame(rows), missing


def _numeric_feature_cols(df: pd.DataFrame) -> list[str]:
    cols = []
    for col in df.columns:
        if col in META_COLS:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols


def _base_feature_prefixes(feature_dict: pd.DataFrame, groups: list[str]) -> list[str]:
    return feature_dict.loc[feature_dict["group"].isin(groups), "feature_name"].tolist()


def _select_prefixed_columns(df: pd.DataFrame, prefixes: list[str]) -> list[str]:
    selected = []
    for col in df.columns:
        if col in META_COLS:
            continue
        for prefix in prefixes:
            if col == prefix or col.startswith(prefix + "_"):
                selected.append(col)
                break
    return sorted(selected)


def _load_existing_feature_frame(horizon_label: str, name: str) -> pd.DataFrame:
    cfg = HORIZON_CONFIGS[horizon_label]
    dfs = []
    for split in SPLITS:
        path = cfg["feature_dir"] / f"{split}_{name}.csv"
        df = pd.read_csv(path)
        df["split"] = split
        dfs.append(df)
    out = pd.concat(dfs, ignore_index=True)
    return out


def _merge_base_and_features(base: pd.DataFrame, feat_df: pd.DataFrame) -> pd.DataFrame:
    keep = ["sequence_id"] + [c for c in feat_df.columns if c != "sequence_id" and c not in base.columns]
    merged = base.merge(feat_df[keep], on="sequence_id", how="inner")
    return merged


def _build_feature_sets(split_rows: dict[str, pd.DataFrame], v2_rows: dict[str, pd.DataFrame], feature_dict: pd.DataFrame) -> tuple[dict[str, dict[str, pd.DataFrame]], list[dict[str, Any]], list[dict[str, Any]]]:
    data_by_horizon: dict[str, dict[str, pd.DataFrame]] = {}
    definitions: list[dict[str, Any]] = []
    coverage: list[dict[str, Any]] = []

    anatomical_prefixes = _base_feature_prefixes(feature_dict, ["original_anatomical_biu"])
    lower_prefixes = _base_feature_prefixes(feature_dict, ["lower_body_commitment_biu"])
    torso_prefixes = _base_feature_prefixes(feature_dict, ["torso_center_commitment_biu"])
    scale_prefixes = _base_feature_prefixes(feature_dict, ["scale_viewpoint_metadata"])
    conf_prefixes = _base_feature_prefixes(feature_dict, ["confidence_metadata"])

    for horizon_label, base in split_rows.items():
        bbox_df = _merge_base_and_features(base, _load_existing_feature_frame(horizon_label, "bbox"))
        biu_v1_df = _merge_base_and_features(base, _load_existing_feature_frame(horizon_label, "biu"))
        v2_df = _merge_base_and_features(base, v2_rows[horizon_label])

        bbox_cols = _numeric_feature_cols(bbox_df)
        v1_cols = _numeric_feature_cols(biu_v1_df)
        v2_anatomical_cols = _select_prefixed_columns(v2_df, anatomical_prefixes)
        v2_lower_cols = _select_prefixed_columns(v2_df, lower_prefixes)
        v2_torso_cols = _select_prefixed_columns(v2_df, torso_prefixes)
        v2_scale_cols = _select_prefixed_columns(v2_df, scale_prefixes)
        v2_conf_cols = _select_prefixed_columns(v2_df, conf_prefixes)
        v2_core_cols = sorted(set(v2_anatomical_cols + v2_lower_cols + v2_torso_cols))
        v2_core_no_scale_cols = sorted(set(v2_core_cols) - set(v2_scale_cols))
        v2_core_with_conf_cols = sorted(set(v2_core_cols + v2_conf_cols))

        feature_sets = {
            "bbox_only": bbox_df,
            "biu_v1_only": biu_v1_df,
            "biu_v2_original_anatomical_only": v2_df,
            "biu_v2_lower_body_only": v2_df,
            "biu_v2_torso_body_center_only": v2_df,
            "biu_v2_core": v2_df,
        }
        feature_sets["biu_v1_plus_bbox"] = biu_v1_df[[c for c in biu_v1_df.columns if c not in bbox_cols]].merge(
            bbox_df[["sequence_id"] + bbox_cols], on="sequence_id", how="inner"
        )
        feature_sets["biu_v2_core_plus_bbox"] = v2_df[[c for c in v2_df.columns if c not in bbox_cols]].merge(
            bbox_df[["sequence_id"] + bbox_cols], on="sequence_id", how="inner"
        )
        column_map = {
            "bbox_only": bbox_cols,
            "biu_v1_only": v1_cols,
            "biu_v2_original_anatomical_only": v2_anatomical_cols,
            "biu_v2_lower_body_only": v2_lower_cols,
            "biu_v2_torso_body_center_only": v2_torso_cols,
            "biu_v2_core": v2_core_cols,
            "biu_v1_plus_bbox": sorted(set(v1_cols + bbox_cols)),
            "biu_v2_core_plus_bbox": sorted(set(v2_core_cols + bbox_cols)),
        }
        notes_map = {
            "bbox_only": "Existing JAAD enriched bbox features from archived benchmark tables.",
            "biu_v1_only": "Original BIU v1 features as used in prior JAAD experiments.",
            "biu_v2_original_anatomical_only": "Direction-aware BIU v2 continuity subset of original anatomical BIUs.",
            "biu_v2_lower_body_only": "Direction-aware lower-body commitment BIUs only.",
            "biu_v2_torso_body_center_only": "Direction-aware torso/body-center commitment BIUs only.",
            "biu_v2_core": "Lower-body + torso/body-center + retained compact anatomical BIUs, no confidence metadata.",
            "biu_v1_plus_bbox": "Original BIU v1 concatenated with bbox features.",
            "biu_v2_core_plus_bbox": "BIU v2 core concatenated with bbox features.",
        }
        for feature_set, cols in column_map.items():
            definitions.append(
                {
                    "feature_set": feature_set,
                    "horizon_label": horizon_label,
                    "n_features": len(cols),
                    "notes": notes_map[feature_set],
                    "status": "ready",
                }
            )
            df = feature_sets[feature_set]
            coverage.append(
                {
                    "horizon_label": horizon_label,
                    "feature_set": feature_set,
                    "source_rows": len(df),
                    "source_unique_sequence_ids": int(df["sequence_id"].nunique()),
                    "n_features": len(cols),
                    "missing_feature_rows": int(len(base) - df["sequence_id"].nunique()),
                }
            )
        definitions.append(
            {
                "feature_set": "biu_v1_direction_canonicalized",
                "horizon_label": horizon_label,
                "n_features": 0,
                "notes": "Skipped in this benchmark run to avoid ambiguous duplication with BIU v2 continuity/anatomical features.",
                "status": "skipped",
            }
        )
        for skipped_feature_set, skipped_notes in [
            ("biu_v2_core_no_scale_metadata", "Optional control skipped to keep runtime bounded; BIU v2 core already excludes confidence metadata and remains the primary benchmark target."),
            ("biu_v2_core_with_confidence_metadata", "Optional confidence control skipped in this benchmark run; confidence-aware BIU was explicitly deferred."),
            ("bbox_plus_biu_v2_lower_body_only", "Optional lower-body-plus-bbox control skipped to keep the mandatory BIU v1 vs BIU v2 comparison tractable."),
        ]:
            definitions.append({"feature_set": skipped_feature_set, "horizon_label": horizon_label, "n_features": 0, "notes": skipped_notes, "status": "skipped"})
        data_by_horizon[horizon_label] = {k: feature_sets[k] for k in feature_sets}
        data_by_horizon[horizon_label]["_column_map"] = column_map
    return data_by_horizon, definitions, coverage


def _candidate_thresholds(probs: np.ndarray) -> np.ndarray:
    uniq = np.unique(np.asarray(probs, dtype=float))
    if len(uniq) > 1000:
        qs = np.linspace(0.0, 1.0, 1001)
        uniq = np.unique(np.quantile(uniq, qs))
    return np.unique(np.concatenate(([0.0, 0.5, 1.0], uniq)))


def _compute_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> dict[str, float]:
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    out = {
        "accuracy": float(accuracy_score(y_true, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, preds)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probs)) if len(np.unique(y_true)) == 2 else float("nan"),
        "pr_auc": float(average_precision_score(y_true, probs)) if len(np.unique(y_true)) == 2 else float("nan"),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "fpr": _safe_div(fp, fp + tn),
        "fnr": _safe_div(fn, fn + tp),
    }
    return out


def _select_thresholds(y_val: np.ndarray, p_val: np.ndarray) -> list[ThresholdSelection]:
    thresholds = _candidate_thresholds(p_val)
    evals = []
    for thr in thresholds:
        metrics = _compute_metrics(y_val, p_val, float(thr))
        evals.append((float(thr), metrics))

    out: list[ThresholdSelection] = []
    out.append(ThresholdSelection("fixed_0_5", 0.5, "available", _compute_metrics(y_val, p_val, 0.5)))

    best_f1 = max(evals, key=lambda x: (x[1]["f1"], x[1]["precision"], x[1]["recall"]))
    out.append(ThresholdSelection("val_f1", best_f1[0], "available", best_f1[1]))

    best_bacc = max(evals, key=lambda x: (x[1]["balanced_accuracy"], x[1]["recall"], -abs(x[0] - 0.5)))
    out.append(ThresholdSelection("val_balanced_accuracy", best_bacc[0], "available", best_bacc[1]))

    fpr_ok = [(thr, m) for thr, m in evals if not math.isnan(m["fpr"]) and m["fpr"] <= 0.1]
    if fpr_ok:
        best_fpr = max(fpr_ok, key=lambda x: (x[1]["recall"], x[1]["precision"], x[1]["f1"]))
        out.append(ThresholdSelection("val_fpr_le_0p1", best_fpr[0], "available", best_fpr[1]))
    else:
        out.append(ThresholdSelection("val_fpr_le_0p1", None, "unavailable", {}))

    prec_ok = [(thr, m) for thr, m in evals if not math.isnan(m["precision"]) and m["precision"] >= 0.8]
    if prec_ok:
        best_prec = max(prec_ok, key=lambda x: (x[1]["recall"], x[1]["f1"], -x[0]))
        out.append(ThresholdSelection("val_precision_ge_0p8", best_prec[0], "available", best_prec[1]))
    else:
        out.append(ThresholdSelection("val_precision_ge_0p8", None, "unavailable", {}))
    return out


def _build_model(model_name: str, seed: int, y_train: np.ndarray):
    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    scale_pos_weight = float(neg / max(pos, 1))
    if model_name == "logreg":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        )
    if model_name == "rf":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        min_samples_leaf=2,
                        class_weight="balanced_subsample",
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
    if model_name == "xgb" and XGBClassifier is not None:
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=250,
                        max_depth=4,
                        learning_rate=0.05,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        reg_lambda=1.0,
                        objective="binary:logistic",
                        eval_metric="logloss",
                        random_state=seed,
                        n_jobs=4,
                        scale_pos_weight=scale_pos_weight,
                    ),
                ),
            ]
        )
    if model_name == "lgbm" and LGBMClassifier is not None:
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    LGBMClassifier(
                        n_estimators=250,
                        learning_rate=0.05,
                        max_depth=-1,
                        num_leaves=31,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=seed,
                        n_jobs=4,
                        class_weight="balanced",
                    ),
                ),
            ]
        )
    return None


def _extract_importance(pipe: Pipeline, model_name: str, feature_names: list[str]) -> pd.DataFrame:
    model = pipe.named_steps["model"]
    rows = []
    if model_name == "logreg":
        values = np.abs(model.coef_[0])
    elif hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
    else:
        return pd.DataFrame()
    for feat, val in zip(feature_names, values):
        rows.append({"feature_name": feat, "importance": float(val)})
    return pd.DataFrame(rows)


def _feature_group_from_name(feature_name: str, feature_dict: pd.DataFrame) -> str:
    if feature_name.startswith("bbox_") or feature_name.startswith("center_"):
        return "bbox"
    match = feature_dict.loc[feature_dict["feature_name"].apply(lambda x: feature_name == x or feature_name.startswith(x + "_"))]
    if match.empty:
        return "other"
    return FEATURE_GROUP_MAP.get(match.iloc[0]["group"], match.iloc[0]["group"])


def _run_benchmark(data_by_horizon: dict[str, dict[str, pd.DataFrame]], feature_dict: pd.DataFrame):
    results = []
    thresholds_rows = []
    group_rows = []
    cm_rows = []
    importance_rows = []
    matrix_rows = []

    for horizon_label, feature_map in data_by_horizon.items():
        column_map = feature_map["_column_map"]
        for feature_set, cols in column_map.items():
            if not cols:
                continue
            df = feature_map[feature_set].copy()
            split_counts = df.groupby("split")["label"].agg(["count", "sum"]).reset_index()
            for _, rec in split_counts.iterrows():
                matrix_rows.append(
                    {
                        "horizon_label": horizon_label,
                        "feature_set": feature_set,
                        "split": rec["split"],
                        "n_rows": int(rec["count"]),
                        "n_positive": int(rec["sum"]),
                        "n_negative": int(rec["count"] - rec["sum"]),
                        "n_features": len(cols),
                    }
                )
            train_df = df[df["split"] == "train"].copy()
            val_df = df[df["split"] == "val"].copy()
            test_df = df[df["split"] == "test"].copy()
            X_train = train_df[cols].astype(float)
            y_train = train_df["label"].astype(int).to_numpy()
            X_val = val_df[cols].astype(float)
            y_val = val_df["label"].astype(int).to_numpy()
            X_test = test_df[cols].astype(float)
            y_test = test_df["label"].astype(int).to_numpy()

            if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2 or len(np.unique(y_test)) < 2:
                for model_name in MODEL_ORDER:
                    results.append({
                        "horizon_label": horizon_label,
                        "feature_set": feature_set,
                        "model_name": model_name,
                        "seed": None,
                        "threshold_rule": "skipped",
                        "threshold_status": f"single_class_split train={sorted(np.unique(y_train).tolist())} val={sorted(np.unique(y_val).tolist())} test={sorted(np.unique(y_test).tolist())}",
                        "selected_threshold": None,
                        "n_features": len(cols),
                        "train_rows": len(train_df),
                        "val_rows": len(val_df),
                        "test_rows": len(test_df),
                    })
                continue

            for seed in SEEDS:
                for model_name in MODEL_ORDER:
                    pipe = _build_model(model_name, seed, y_train)
                    if pipe is None:
                        continue
                    pipe.fit(X_train, y_train)
                    p_val = pipe.predict_proba(X_val)[:, 1]
                    p_test = pipe.predict_proba(X_test)[:, 1]

                    selections = _select_thresholds(y_val, p_val)
                    for sel in selections:
                        row = {
                            "horizon_label": horizon_label,
                            "feature_set": feature_set,
                            "model_name": model_name,
                            "seed": seed,
                            "threshold_rule": sel.rule,
                            "threshold_status": sel.status,
                            "selected_threshold": sel.threshold,
                            "n_features": len(cols),
                            "train_rows": len(train_df),
                            "val_rows": len(val_df),
                            "test_rows": len(test_df),
                        }
                        if sel.status == "available" and sel.threshold is not None:
                            test_metrics = _compute_metrics(y_test, p_test, float(sel.threshold))
                            for k, v in sel.val_metrics.items():
                                row[f"val_{k}"] = v
                            row.update({f"test_{k}": v for k, v in test_metrics.items()})
                            preds = (p_test >= float(sel.threshold)).astype(int)
                            for group_name in ["clean_negative", "negative_ambiguous", "actual_positive"]:
                                mask = test_df["group"].to_numpy() == group_name
                                if not mask.any():
                                    continue
                                g_y = y_test[mask]
                                g_p = p_test[mask]
                                g_pred = preds[mask]
                                g_tn, g_fp, g_fn, g_tp = confusion_matrix(g_y, g_pred, labels=[0, 1]).ravel()
                                group_rows.append(
                                    {
                                        "horizon_label": horizon_label,
                                        "feature_set": feature_set,
                                        "model_name": model_name,
                                        "seed": seed,
                                        "threshold_rule": sel.rule,
                                        "group": group_name,
                                        "n": int(mask.sum()),
                                        "score_mean": float(np.mean(g_p)),
                                        "score_median": float(np.median(g_p)),
                                        "predicted_positive_rate": float(np.mean(g_pred)),
                                        "fpr": _safe_div(g_fp, g_fp + g_tn),
                                        "recall": _safe_div(g_tp, g_tp + g_fn),
                                        "fp_count": int(g_fp),
                                        "tn_count": int(g_tn),
                                        "fn_count": int(g_fn),
                                        "tp_count": int(g_tp),
                                    }
                                )
                            cm_rows.append(
                                {
                                    "horizon_label": horizon_label,
                                    "feature_set": feature_set,
                                    "model_name": model_name,
                                    "seed": seed,
                                    "threshold_rule": sel.rule,
                                    "threshold": sel.threshold,
                                    "tn": int(test_metrics["tn"]),
                                    "fp": int(test_metrics["fp"]),
                                    "fn": int(test_metrics["fn"]),
                                    "tp": int(test_metrics["tp"]),
                                }
                            )
                        results.append(row)
                        thresholds_rows.append(row.copy())

                    if seed == 42 and feature_set in {"biu_v2_core", "biu_v2_core_plus_bbox"}:
                        imp = _extract_importance(pipe, model_name, cols)
                        if not imp.empty:
                            imp["horizon_label"] = horizon_label
                            imp["feature_set"] = feature_set
                            imp["model_name"] = model_name
                            imp["seed"] = seed
                            imp["feature_group"] = imp["feature_name"].apply(lambda x: _feature_group_from_name(x, feature_dict))
                            importance_rows.extend(imp.to_dict(orient="records"))
    return (
        pd.DataFrame(results),
        pd.DataFrame(thresholds_rows),
        pd.DataFrame(group_rows),
        pd.DataFrame(cm_rows),
        pd.DataFrame(importance_rows),
        pd.DataFrame(matrix_rows),
    )


def _aggregate_results(results_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [c for c in results_df.columns if c.startswith("test_") or c.startswith("val_")]
    agg = results_df.groupby(["horizon_label", "feature_set", "model_name", "threshold_rule", "threshold_status"], dropna=False)[metric_cols].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg = agg.reset_index()
    return agg


def _make_figures(results_df: pd.DataFrame, group_df: pd.DataFrame, importance_df: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sel = results_df[(results_df["threshold_rule"] == "val_f1") & (results_df["threshold_status"] == "available")].copy()
    if not sel.empty:
        agg = sel.groupby(["horizon_label", "feature_set"])[["test_f1", "test_pr_auc"]].mean().reset_index()
        pivot = agg.pivot(index="horizon_label", columns="feature_set", values="test_f1")
        plt.figure(figsize=(12, 5))
        pivot.plot(kind="bar")
        plt.ylabel("Mean test F1")
        plt.title("JAAD BIU v1 vs BIU v2 mean test F1 by horizon (val_f1 threshold)")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "biu_v1_vs_biu_v2_f1_by_horizon.png", dpi=160)
        plt.close()

    if not group_df.empty:
        amb = group_df[(group_df["group"] == "negative_ambiguous") & (group_df["threshold_rule"] == "fixed_0_5")]
        if not amb.empty:
            amb_agg = amb.groupby(["horizon_label", "feature_set"])[["fpr"]].mean().reset_index()
            pivot = amb_agg.pivot(index="horizon_label", columns="feature_set", values="fpr")
            plt.figure(figsize=(12, 5))
            pivot.plot(kind="bar")
            plt.ylabel("Ambiguous-negative FPR")
            plt.title("Ambiguous-negative FPR at fixed 0.5")
            plt.tight_layout()
            plt.savefig(FIGURES_DIR / "ambiguous_negative_fpr_by_feature_set.png", dpi=160)
            plt.close()

        pos = group_df[(group_df["group"] == "actual_positive") & (group_df["threshold_rule"] == "fixed_0_5")]
        if not pos.empty:
            pos_agg = pos.groupby(["horizon_label", "feature_set"])[["recall"]].mean().reset_index()
            pivot = pos_agg.pivot(index="horizon_label", columns="feature_set", values="recall")
            plt.figure(figsize=(12, 5))
            pivot.plot(kind="bar")
            plt.ylabel("Actual-positive recall")
            plt.title("Actual-positive recall at fixed 0.5")
            plt.tight_layout()
            plt.savefig(FIGURES_DIR / "actual_positive_recall_by_feature_set.png", dpi=160)
            plt.close()

    if not importance_df.empty:
        top = importance_df.groupby(["feature_group"]) ["importance"].mean().sort_values(ascending=False).head(10)
        plt.figure(figsize=(8, 4))
        top.sort_values().plot(kind="barh")
        plt.xlabel("Mean model importance")
        plt.title("Mean BIU v2 feature-group importance")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "biu_v2_feature_group_importance.png", dpi=160)
        plt.close()


def _build_report(results_df: pd.DataFrame, group_df: pd.DataFrame, importance_df: pd.DataFrame) -> tuple[str, str]:
    available = results_df[(results_df["threshold_rule"] == "val_f1") & (results_df["threshold_status"] == "available")].copy()
    summary_lines = []
    if available.empty:
        summary_lines.append("No available benchmark rows were produced.")
    else:
        mean_scores = available.groupby(["horizon_label", "feature_set"]) [["test_f1", "test_pr_auc", "test_balanced_accuracy"]].mean().reset_index()
        def get_row(h, f):
            sub = mean_scores[(mean_scores["horizon_label"] == h) & (mean_scores["feature_set"] == f)]
            return sub.iloc[0] if not sub.empty else None
        comparisons = []
        for horizon in ["30_60", "15_30", "5_15"]:
            v1 = get_row(horizon, "biu_v1_only")
            v2 = get_row(horizon, "biu_v2_core")
            bbox = get_row(horizon, "bbox_only")
            v1b = get_row(horizon, "biu_v1_plus_bbox")
            v2b = get_row(horizon, "biu_v2_core_plus_bbox")
            if v1 is not None and v2 is not None:
                comparisons.append((horizon, float(v2["test_f1"] - v1["test_f1"]), float(v2["test_pr_auc"] - v1["test_pr_auc"])))
            if v2 is not None and bbox is not None:
                summary_lines.append(f"- {horizon}: `biu_v2_core` F1={v2['test_f1']:.3f}, PR-AUC={v2['test_pr_auc']:.3f}; `bbox_only` F1={bbox['test_f1']:.3f}, PR-AUC={bbox['test_pr_auc']:.3f}.")
            if v1b is not None and v2b is not None:
                summary_lines.append(f"- {horizon}: `biu_v2_core_plus_bbox` vs `biu_v1_plus_bbox` delta F1={v2b['test_f1'] - v1b['test_f1']:.3f}, delta PR-AUC={v2b['test_pr_auc'] - v1b['test_pr_auc']:.3f}.")
        if comparisons:
            best_horizon = max(comparisons, key=lambda x: x[1])
            summary_lines.insert(0, f"- Largest BIU v2 over BIU v1 F1 gain appears at `{best_horizon[0]}`: delta F1={best_horizon[1]:.3f}, delta PR-AUC={best_horizon[2]:.3f}.")

    group_lines = []
    if not group_df.empty:
        grp = group_df.groupby(["horizon_label", "feature_set", "threshold_rule", "group"]) [["fpr", "recall", "fp_count"]].mean().reset_index()
        for horizon in ["30_60", "15_30", "5_15"]:
            amb_v1 = grp[(grp["horizon_label"] == horizon) & (grp["feature_set"] == "biu_v1_only") & (grp["threshold_rule"] == "fixed_0_5") & (grp["group"] == "negative_ambiguous")]
            amb_v2 = grp[(grp["horizon_label"] == horizon) & (grp["feature_set"] == "biu_v2_core") & (grp["threshold_rule"] == "fixed_0_5") & (grp["group"] == "negative_ambiguous")]
            pos_v1 = grp[(grp["horizon_label"] == horizon) & (grp["feature_set"] == "biu_v1_only") & (grp["threshold_rule"] == "fixed_0_5") & (grp["group"] == "actual_positive")]
            pos_v2 = grp[(grp["horizon_label"] == horizon) & (grp["feature_set"] == "biu_v2_core") & (grp["threshold_rule"] == "fixed_0_5") & (grp["group"] == "actual_positive")]
            if not amb_v1.empty and not amb_v2.empty:
                group_lines.append(f"- {horizon}: ambiguous-negative FPR at fixed 0.5 changed from {amb_v1.iloc[0]['fpr']:.3f} (BIU v1) to {amb_v2.iloc[0]['fpr']:.3f} (BIU v2 core).")
            if not pos_v1.empty and not pos_v2.empty:
                group_lines.append(f"- {horizon}: actual-positive recall at fixed 0.5 changed from {pos_v1.iloc[0]['recall']:.3f} (BIU v1) to {pos_v2.iloc[0]['recall']:.3f} (BIU v2 core).")

    importance_lines = []
    if not importance_df.empty:
        top_groups = importance_df.groupby("feature_group")["importance"].mean().sort_values(ascending=False)
        top_text = ", ".join([f"{idx}={val:.3f}" for idx, val in top_groups.head(5).items()])
        importance_lines.append(f"- Mean model importance is concentrated in: {top_text}.")

    report = "# JAAD BIU v2 benchmark comparison\n\n"
    report += "## Purpose\n"
    report += "Compare BIU v1 against direction-aware BIU v2 on existing JAAD tabular benchmark splits for `30_60`, `15_30`, and `5_15`, using validation-only threshold selection and explicit clean-vs-ambiguous negative breakdowns.\n\n"
    report += "## Inputs\n"
    report += "- `docs/BIU_V2_SPEC.md`\n"
    report += "- `src/biu_v2_extractor.py`\n"
    report += "- Archived JAAD split/feature tables under `/home/abdelrahman/Hadwa/BIU-PID-Distillation/data/...`\n"
    report += "- Archived extended JAAD keypoints JSONL under `/home/abdelrahman/Hadwa/BIU-PID-Distillation/data/jaad_keypoints_vitpose_s_extended_short_horizon.jsonl`\n\n"
    report += "## Main findings\n"
    report += "\n".join(summary_lines + group_lines + importance_lines) + "\n\n"
    report += "## Interpretation\n"
    report += "- The primary comparison is `biu_v1_only` vs `biu_v2_core`, and `biu_v1_plus_bbox` vs `biu_v2_core_plus_bbox`.\n"
    report += "- `biu_v2_core` is built from direction-aware lower-body commitment, torso/body-center commitment, and retained compact anatomical cues.\n"
    report += "- Negative subgroup breakdown uses the same crossing-relevance logic as the prior JAAD temporal audit: `clean_negative`, `negative_ambiguous`, and `actual_positive`.\n"
    report += "- If `bbox_only` still dominates, that should be stated directly; BIU v2 is only useful for the paper if it adds either independent recall or lower ambiguous-negative FPR.\n\n"
    report += "## Next recommendation\n"
    report += "- If BIU v2 improves `5_15` and especially reduces ambiguous-negative false positives, the next step should be a staged `BBox/context -> BIU v2 confirmation` model.\n"
    report += "- If BIU v2 helps little once bbox is present, the next step should be threshold/selective-prediction auditing rather than another BIU redesign cycle.\n"

    readme = "# Experiment 009: JAAD BIU v2 benchmark comparison\n\n"
    readme += "## Purpose\n"
    readme += "Benchmark BIU v2 against BIU v1 on JAAD `30_60`, `15_30`, and `5_15` using the existing train/val/test splits and validation-only threshold selection.\n\n"
    readme += "## Inputs used\n"
    readme += "- `docs/BIU_V2_SPEC.md`\n"
    readme += "- `src/biu_v2_extractor.py`\n"
    readme += "- `experiments/008_biu_v2_design_spec/metrics/biu_v2_feature_dictionary.csv`\n"
    readme += "- Archived JAAD split/feature tables and extended JAAD keypoints JSONL from the read-only archive repo\n\n"
    readme += "## How to rerun\n"
    readme += "```bash\n"
    readme += "cd /home/abdelrahman/Hadwa/BIU-PID-Distillation-clean\n"
    readme += "python3 scripts/experiments/jaad_biu_v2_benchmark_comparison.py\n"
    readme += "```\n\n"
    readme += "## Outputs\n"
    readme += "- `metrics/jaad_biu_v2_benchmark_results.csv`\n"
    readme += "- `metrics/jaad_biu_v2_benchmark_results_mean_std.csv`\n"
    readme += "- `metrics/jaad_biu_v2_thresholds.csv`\n"
    readme += "- `metrics/jaad_biu_v2_group_breakdown.csv`\n"
    readme += "- `metrics/jaad_biu_v2_confusion_matrices.csv`\n"
    readme += "- `metrics/jaad_biu_v2_feature_importance.csv`\n"
    readme += "- `report.md`\n\n"
    readme += "## Final conclusion\n"
    readme += "See `report.md` for the horizon-by-horizon BIU v1 vs BIU v2 comparison and the ambiguous-negative error analysis.\n"
    return report, readme


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    feature_dict = pd.read_csv(DICT_PATH)
    split_rows = {h: _load_split_rows(h) for h in HORIZON_CONFIGS}

    artifact_rows = []
    for h, cfg in HORIZON_CONFIGS.items():
        artifact_rows.append({"horizon_label": h, "artifact_type": "split_dir", "path": str(cfg["split_dir"]), "exists": cfg["split_dir"].exists()})
        artifact_rows.append({"horizon_label": h, "artifact_type": "feature_dir", "path": str(cfg["feature_dir"]), "exists": cfg["feature_dir"].exists()})
        artifact_rows.append({"horizon_label": h, "artifact_type": "keypoints_jsonl", "path": str(cfg["keypoints_jsonl"]), "exists": cfg["keypoints_jsonl"].exists()})
    pd.DataFrame(artifact_rows).to_csv(METRICS_DIR / "jaad_biu_v2_artifact_coverage.csv", index=False)

    needed = _needed_frame_keys(split_rows)
    keypoints_path = next(iter(HORIZON_CONFIGS.values()))["keypoints_jsonl"]
    kp_index = _load_keypoint_index(keypoints_path, needed)

    v2_rows = {}
    missing_rows = []
    for horizon_label, df in split_rows.items():
        v2_df, missing = _extract_v2_for_rows(df, kp_index)
        v2_rows[horizon_label] = v2_df
        missing_rows.extend(missing)
    if missing_rows:
        pd.DataFrame(missing_rows).to_csv(METRICS_DIR / "jaad_biu_v2_missing_keypoint_rows.csv", index=False)

    data_by_horizon, definitions, coverage = _build_feature_sets(split_rows, v2_rows, feature_dict)
    pd.DataFrame(definitions).to_csv(METRICS_DIR / "jaad_biu_v2_feature_set_definitions.csv", index=False)
    pd.DataFrame(coverage).to_csv(METRICS_DIR / "jaad_biu_v2_feature_matrix_summary.csv", index=False)

    results_df, thresholds_df, group_df, cm_df, importance_df, matrix_df = _run_benchmark(data_by_horizon, feature_dict)
    mean_std_df = _aggregate_results(results_df)

    results_df.to_csv(METRICS_DIR / "jaad_biu_v2_benchmark_results.csv", index=False)
    mean_std_df.to_csv(METRICS_DIR / "jaad_biu_v2_benchmark_results_mean_std.csv", index=False)
    thresholds_df.to_csv(METRICS_DIR / "jaad_biu_v2_thresholds.csv", index=False)
    group_df.to_csv(METRICS_DIR / "jaad_biu_v2_group_breakdown.csv", index=False)
    cm_df.to_csv(METRICS_DIR / "jaad_biu_v2_confusion_matrices.csv", index=False)
    importance_df.to_csv(METRICS_DIR / "jaad_biu_v2_feature_importance.csv", index=False)
    matrix_df.to_csv(METRICS_DIR / "jaad_biu_v2_feature_matrix_split_summary.csv", index=False)

    _make_figures(results_df, group_df, importance_df)
    report, readme = _build_report(results_df, group_df, importance_df)
    REPORT_PATH.write_text(report, encoding="utf-8")
    README_PATH.write_text(readme, encoding="utf-8")

    print(json.dumps({
        "results_rows": int(len(results_df)),
        "group_rows": int(len(group_df)),
        "importance_rows": int(len(importance_df)),
        "missing_keypoint_rows": int(len(missing_rows)),
    }, indent=2))


if __name__ == "__main__":
    main()
