"""Experiment 016: JAAD scene-context tabular features for the 30_60 benchmark.

Builds a lightweight CONTEXT feature family from JAAD's existing scene
annotations (annotations_attributes + annotations_traffic XML, no image
processing) and benchmarks context_only / bbox_plus_context /
biu_bbox_plus_context against the existing bbox_only / biu_bbox references
on the archived JAAD 30_60 event benchmark splits.

Conventions follow scripts/experiments/jaad_biu_v2_benchmark_comparison.py:
- models: logreg (balanced, median-impute + standardize), rf (300 trees,
  min_samples_leaf=2, balanced_subsample), xgb (250 est, depth 4, lr 0.05,
  scale_pos_weight)
- seeds 42/43/44
- thresholds: fixed 0.5 and validation-F1 (validation-only selection)
- test metrics reported as mean/std over seeds.

Leakage policy: the JAAD per-pedestrian attributes `crossing`,
`crossing_point`, `decision_point` (and intention-like fields) are label- or
future-derived and are EXPLICITLY excluded. Per-frame traffic flags are
aggregated over the OBSERVED window frames only (no future frames).
"""
from __future__ import annotations

import json
import os
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

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
except Exception:  # pragma: no cover
    XGBClassifier = None

CLEAN_ROOT = Path("/home/abdelrahman/Hadwa/BIU-PID-Distillation-clean")
ARCHIVE_ROOT = Path("/home/abdelrahman/Hadwa/BIU-PID-Distillation")
JAAD_ROOT = Path("/home/abdelrahman/Hadwa/datasets/JAAD")

SPLIT_DIR = ARCHIVE_ROOT / "data/jaad_splits_event_16_30_60_overlap08_default"
FEATURE_DIR = ARCHIVE_ROOT / "data/jaad_features_event_16_30_60_overlap08_default"
ATTR_DIR = JAAD_ROOT / "annotations_attributes"
TRAFFIC_DIR = JAAD_ROOT / "annotations_traffic"

OUT_ROOT = CLEAN_ROOT / "experiments/016_jaad_context_features"
METRICS_DIR = OUT_ROOT / "metrics"
PRED_DIR = OUT_ROOT / "predictions"

SPLITS = ["train", "val", "test"]
SEEDS = [42, 43, 44]
MODEL_ORDER = ["logreg", "rf", "xgb"]

# Attribute fields that are label-derived or future-derived and MUST NOT be
# used as features (leakage). `crossing` is the crossing label attribute,
# `crossing_point`/`decision_point` are future event frame indices.
FORBIDDEN_ATTR_FIELDS = {"crossing", "crossing_point", "decision_point", "intention"}
# Identifier fields, never features.
ID_ATTR_FIELDS = {"id", "old_id"}


@dataclass
class ThresholdSelection:
    rule: str
    threshold: float
    val_metrics: dict[str, float]


# ---------------------------------------------------------------------------
# Context feature construction
# ---------------------------------------------------------------------------

def load_attribute_index() -> tuple[dict[tuple[str, str], dict[str, str]], set[str]]:
    """Map (video_id, ped_id) -> raw attribute dict. Also return the set of
    raw attribute field names seen (for documentation/leakage audit)."""
    index: dict[tuple[str, str], dict[str, str]] = {}
    fields_seen: set[str] = set()
    for xml_path in sorted(ATTR_DIR.glob("video_*_attributes.xml")):
        video_id = xml_path.name.replace("_attributes.xml", "")
        root = ET.parse(xml_path).getroot()
        for ped in root.findall("pedestrian"):
            fields_seen.update(ped.attrib.keys())
            index[(video_id, ped.attrib["id"])] = dict(ped.attrib)
    return index, fields_seen


def load_traffic_index() -> dict[str, dict[str, Any]]:
    """Map video_id -> {road_type: str, frames: {frame_idx: attrib dict}}."""
    index: dict[str, dict[str, Any]] = {}
    for xml_path in sorted(TRAFFIC_DIR.glob("video_*_traffic.xml")):
        video_id = xml_path.name.replace("_traffic.xml", "")
        root = ET.parse(xml_path).getroot()
        road_type_el = root.find("road_type")
        frames = {int(fr.attrib["id"]): fr.attrib for fr in root.findall("frame")}
        index[video_id] = {
            "road_type": road_type_el.text if road_type_el is not None else "n/a",
            "frames": frames,
        }
    return index


AGE_ORDINAL = {"child": 0.0, "young": 1.0, "adult": 2.0, "senior": 3.0}

# (column_name, source_annotation_file, source_field, encoding_description)
CONTEXT_FEATURE_DEFINITIONS = [
    ("attr_age_ordinal", "annotations_attributes", "age", "ordinal child=0, young=1, adult=2, senior=3; NaN if unknown"),
    ("attr_gender_female", "annotations_attributes", "gender", "1 if female else 0 (n/a -> 0)"),
    ("attr_gender_male", "annotations_attributes", "gender", "1 if male else 0 (n/a -> 0)"),
    ("attr_group_size", "annotations_attributes", "group_size", "integer group size as float"),
    ("attr_designated_crossing", "annotations_attributes", "designated", "1 if D (designated crossing point) else 0 (ND)"),
    ("attr_intersection", "annotations_attributes", "intersection", "1 if yes else 0"),
    ("attr_num_lanes", "annotations_attributes", "num_lanes", "integer number of lanes as float"),
    ("attr_signalized_s", "annotations_attributes", "signalized", "1 if S (signalized) else 0 (NS or n/a -> 0)"),
    ("attr_signalized_ns", "annotations_attributes", "signalized", "1 if NS (not signalized) else 0 (S or n/a -> 0)"),
    ("attr_traffic_oneway", "annotations_attributes", "traffic_direction", "1 if OW (one-way) else 0 (TW two-way)"),
    ("attr_motion_lateral", "annotations_attributes", "motion_direction", "1 if LAT else 0 (LONG or n/a -> 0); track-level annotation, see caveat"),
    ("attr_motion_longitudinal", "annotations_attributes", "motion_direction", "1 if LONG else 0 (LAT or n/a -> 0); track-level annotation, see caveat"),
    ("traffic_road_type_street", "annotations_traffic", "road_type", "1 if road_type == street else 0 (video-level)"),
    ("traffic_road_type_parking_lot", "annotations_traffic", "road_type", "1 if road_type == parking_lot else 0 (video-level; garage is the implicit baseline)"),
    ("traffic_ped_crossing_frac", "annotations_traffic", "frame/ped_crossing", "fraction of OBSERVED window frames with a marked pedestrian crosswalk in scene"),
    ("traffic_ped_sign_frac", "annotations_traffic", "frame/ped_sign", "fraction of OBSERVED window frames with a pedestrian sign in scene"),
    ("traffic_stop_sign_frac", "annotations_traffic", "frame/stop_sign", "fraction of OBSERVED window frames with a stop sign in scene"),
    ("traffic_light_red_frac", "annotations_traffic", "frame/traffic_light", "fraction of OBSERVED window frames with traffic_light == red (n/a -> 0)"),
    ("traffic_light_green_frac", "annotations_traffic", "frame/traffic_light", "fraction of OBSERVED window frames with traffic_light == green (n/a -> 0)"),
    ("traffic_light_present_frac", "annotations_traffic", "frame/traffic_light", "fraction of OBSERVED window frames with any traffic light state annotated (red or green)"),
]
CONTEXT_COLS = [d[0] for d in CONTEXT_FEATURE_DEFINITIONS]

# motion_direction is annotated by the JAAD annotators over the WHOLE track
# (including frames after the observation window). Empirically it is a
# near-label proxy at 30_60 (all train positives have motion LAT; train AUC
# 0.957), so it is treated as future-informed. The *_no_motion feature sets
# exclude it and are the honest context variants.
MOTION_COLS = ["attr_motion_lateral", "attr_motion_longitudinal"]
CONTEXT_NO_MOTION_COLS = [c for c in CONTEXT_COLS if c not in MOTION_COLS]


def build_context_row(
    rec: pd.Series,
    attr_index: dict[tuple[str, str], dict[str, str]],
    traffic_index: dict[str, dict[str, Any]],
) -> dict[str, float]:
    video_id = str(rec["video_id"])
    ped_id = str(rec["ped_id"])
    attrs = attr_index.get((video_id, ped_id))
    if attrs is None:
        raise KeyError(f"No attribute annotation for ({video_id}, {ped_id})")
    traffic = traffic_index.get(video_id)
    if traffic is None:
        raise KeyError(f"No traffic annotation for {video_id}")

    out: dict[str, float] = {}
    out["attr_age_ordinal"] = AGE_ORDINAL.get(attrs.get("age", "n/a"), float("nan"))
    gender = attrs.get("gender", "n/a")
    out["attr_gender_female"] = 1.0 if gender == "female" else 0.0
    out["attr_gender_male"] = 1.0 if gender == "male" else 0.0
    try:
        out["attr_group_size"] = float(attrs.get("group_size", "nan"))
    except ValueError:
        out["attr_group_size"] = float("nan")
    out["attr_designated_crossing"] = 1.0 if attrs.get("designated") == "D" else 0.0
    out["attr_intersection"] = 1.0 if attrs.get("intersection") == "yes" else 0.0
    try:
        out["attr_num_lanes"] = float(attrs.get("num_lanes", "nan"))
    except ValueError:
        out["attr_num_lanes"] = float("nan")
    signalized = attrs.get("signalized", "n/a")
    out["attr_signalized_s"] = 1.0 if signalized == "S" else 0.0
    out["attr_signalized_ns"] = 1.0 if signalized == "NS" else 0.0
    out["attr_traffic_oneway"] = 1.0 if attrs.get("traffic_direction") == "OW" else 0.0
    motion = attrs.get("motion_direction", "n/a")
    out["attr_motion_lateral"] = 1.0 if motion == "LAT" else 0.0
    out["attr_motion_longitudinal"] = 1.0 if motion == "LONG" else 0.0

    road_type = traffic["road_type"]
    out["traffic_road_type_street"] = 1.0 if road_type == "street" else 0.0
    out["traffic_road_type_parking_lot"] = 1.0 if road_type == "parking_lot" else 0.0

    observed_frames = json.loads(rec["observed_frames_json"])
    frame_map = traffic["frames"]
    flags = {"ped_crossing": [], "ped_sign": [], "stop_sign": []}
    light_red, light_green, light_present = [], [], []
    for fr in observed_frames:
        item = frame_map.get(int(fr))
        if item is None:
            continue
        for k in flags:
            flags[k].append(1.0 if item.get(k) == "1" else 0.0)
        tl = item.get("traffic_light", "n/a")
        light_red.append(1.0 if tl == "red" else 0.0)
        light_green.append(1.0 if tl == "green" else 0.0)
        light_present.append(1.0 if tl in ("red", "green") else 0.0)
    def frac(vals: list[float]) -> float:
        return float(np.mean(vals)) if vals else float("nan")
    out["traffic_ped_crossing_frac"] = frac(flags["ped_crossing"])
    out["traffic_ped_sign_frac"] = frac(flags["ped_sign"])
    out["traffic_stop_sign_frac"] = frac(flags["stop_sign"])
    out["traffic_light_red_frac"] = frac(light_red)
    out["traffic_light_green_frac"] = frac(light_green)
    out["traffic_light_present_frac"] = frac(light_present)
    return out


def leakage_audit(attr_fields_seen: set[str]) -> pd.DataFrame:
    """Assert no context column is derived from a forbidden (label/future)
    field, and document the disposition of every raw attribute field."""
    used_source_fields = {d[2].split("/")[-1] for d in CONTEXT_FEATURE_DEFINITIONS if d[1] == "annotations_attributes"}
    overlap = used_source_fields & FORBIDDEN_ATTR_FIELDS
    assert not overlap, f"Leakage: forbidden attribute fields used as features: {overlap}"
    # Columns whose NAME contains "crossing" but whose SOURCE is scene
    # infrastructure (designated crossing point / crosswalk in scene), not the
    # crossing label. Any other context column may not contain a forbidden
    # field name.
    scene_allowlist = {"attr_designated_crossing", "traffic_ped_crossing_frac"}
    for col in CONTEXT_COLS:
        if col in scene_allowlist:
            continue
        for bad in FORBIDDEN_ATTR_FIELDS:
            assert bad not in col.lower(), f"Suspicious context column name {col}"
    rows = []
    for field in sorted(attr_fields_seen):
        if field in FORBIDDEN_ATTR_FIELDS:
            disposition = "EXCLUDED (label- or future-derived: leakage risk)"
        elif field in ID_ATTR_FIELDS:
            disposition = "EXCLUDED (identifier)"
        elif field in used_source_fields:
            disposition = "used as context feature"
        else:
            disposition = "not used"
        rows.append({"annotation_source": "annotations_attributes", "raw_field": field, "disposition": disposition})
    for field, disp in [
        ("road_type", "used as context feature (video-level)"),
        ("frame/ped_crossing", "used, aggregated over observed window frames only"),
        ("frame/ped_sign", "used, aggregated over observed window frames only"),
        ("frame/stop_sign", "used, aggregated over observed window frames only"),
        ("frame/traffic_light", "used, aggregated over observed window frames only"),
    ]:
        rows.append({"annotation_source": "annotations_traffic", "raw_field": field, "disposition": disp})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_splits() -> pd.DataFrame:
    dfs = []
    for split in SPLITS:
        df = pd.read_csv(SPLIT_DIR / f"{split}.csv")
        df["split"] = split
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


META_COLS = {
    "sequence_id", "split", "label", "set_id", "video_id", "ped_id", "track_id",
    "start_frame", "end_frame",
}


def load_feature_table(name: str) -> tuple[pd.DataFrame, list[str]]:
    dfs = []
    for split in SPLITS:
        df = pd.read_csv(FEATURE_DIR / f"{split}_{name}.csv")
        df["split"] = split
        dfs.append(df)
    out = pd.concat(dfs, ignore_index=True)
    cols = [c for c in out.columns if c not in META_COLS and pd.api.types.is_numeric_dtype(out[c])]
    return out, cols


# ---------------------------------------------------------------------------
# Benchmark machinery (same conventions as experiment 009)
# ---------------------------------------------------------------------------

def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b else float("nan")


def compute_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> dict[str, float]:
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, preds)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probs)),
        "pr_auc": float(average_precision_score(y_true, probs)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "fpr": _safe_div(fp, fp + tn),
        "fnr": _safe_div(fn, fn + tp),
    }


def select_thresholds(y_val: np.ndarray, p_val: np.ndarray) -> list[ThresholdSelection]:
    uniq = np.unique(np.asarray(p_val, dtype=float))
    if len(uniq) > 1000:
        uniq = np.unique(np.quantile(uniq, np.linspace(0.0, 1.0, 1001)))
    thresholds = np.unique(np.concatenate(([0.0, 0.5, 1.0], uniq)))
    evals = [(float(t), compute_metrics(y_val, p_val, float(t))) for t in thresholds]
    out = [ThresholdSelection("fixed_0_5", 0.5, compute_metrics(y_val, p_val, 0.5))]
    best_f1 = max(evals, key=lambda x: (x[1]["f1"], x[1]["precision"], x[1]["recall"]))
    out.append(ThresholdSelection("val_f1", best_f1[0], best_f1[1]))
    return out


def build_model(model_name: str, seed: int, y_train: np.ndarray):
    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    scale_pos_weight = float(neg / max(pos, 1))
    if model_name == "logreg":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)),
        ])
    if model_name == "rf":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=300, min_samples_leaf=2, class_weight="balanced_subsample",
                random_state=seed, n_jobs=-1)),
        ])
    if model_name == "xgb" and XGBClassifier is not None:
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", XGBClassifier(
                n_estimators=250, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                objective="binary:logistic", eval_metric="logloss",
                random_state=seed, n_jobs=4, scale_pos_weight=scale_pos_weight)),
        ])
    return None


def run_benchmark(feature_sets: dict[str, tuple[pd.DataFrame, list[str]]]):
    results = []
    for feature_set, (df, cols) in feature_sets.items():
        train_df = df[df["split"] == "train"]
        val_df = df[df["split"] == "val"]
        test_df = df[df["split"] == "test"]
        X_train = train_df[cols].astype(float)
        y_train = train_df["label"].astype(int).to_numpy()
        X_val = val_df[cols].astype(float)
        y_val = val_df["label"].astype(int).to_numpy()
        X_test = test_df[cols].astype(float)
        y_test = test_df["label"].astype(int).to_numpy()
        for seed in SEEDS:
            for model_name in MODEL_ORDER:
                pipe = build_model(model_name, seed, y_train)
                if pipe is None:
                    continue
                pipe.fit(X_train, y_train)
                p_val = pipe.predict_proba(X_val)[:, 1]
                p_test = pipe.predict_proba(X_test)[:, 1]

                pred_rows = pd.concat([
                    pd.DataFrame({
                        "sequence_id": val_df["sequence_id"].to_numpy(),
                        "split": "val", "label": y_val, "probability": p_val,
                    }),
                    pd.DataFrame({
                        "sequence_id": test_df["sequence_id"].to_numpy(),
                        "split": "test", "label": y_test, "probability": p_test,
                    }),
                ], ignore_index=True)
                pred_rows.insert(0, "feature_set", feature_set)
                pred_rows.insert(1, "model_name", model_name)
                pred_rows.insert(2, "seed", seed)
                pred_rows.to_csv(PRED_DIR / f"{feature_set}__{model_name}__seed{seed}.csv", index=False)

                for sel in select_thresholds(y_val, p_val):
                    row = {
                        "feature_set": feature_set,
                        "model_name": model_name,
                        "seed": seed,
                        "threshold_rule": sel.rule,
                        "selected_threshold": sel.threshold,
                        "n_features": len(cols),
                        "train_rows": len(train_df),
                        "val_rows": len(val_df),
                        "test_rows": len(test_df),
                    }
                    row.update({f"val_{k}": v for k, v in sel.val_metrics.items()})
                    row.update({f"test_{k}": v for k, v in compute_metrics(y_test, p_test, sel.threshold).items()})
                    results.append(row)
    return pd.DataFrame(results)


def aggregate(results_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [c for c in results_df.columns if c.startswith("test_") or c.startswith("val_") or c == "selected_threshold"]
    agg = results_df.groupby(["feature_set", "model_name", "threshold_rule"])[metric_cols].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    return agg.reset_index()


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)

    base = load_splits()
    attr_index, attr_fields_seen = load_attribute_index()
    traffic_index = load_traffic_index()

    audit_df = leakage_audit(attr_fields_seen)
    audit_df.to_csv(METRICS_DIR / "leakage_audit_raw_field_disposition.csv", index=False)

    context_rows = []
    for _, rec in base.iterrows():
        row = {
            "sequence_id": rec["sequence_id"],
            "split": rec["split"],
            "label": int(rec["label"]),
            "video_id": rec["video_id"],
            "ped_id": rec["ped_id"],
        }
        row.update(build_context_row(rec, attr_index, traffic_index))
        context_rows.append(row)
    context_df = pd.DataFrame(context_rows)
    assert len(context_df) == len(base), "Context coverage is not 100%"
    assert set(CONTEXT_COLS).issubset(context_df.columns)
    context_df.to_csv(METRICS_DIR / "jaad_context_feature_table.csv", index=False)

    pd.DataFrame(
        CONTEXT_FEATURE_DEFINITIONS,
        columns=["column", "source_annotation_dir", "source_field", "encoding"],
    ).to_csv(METRICS_DIR / "context_feature_definitions.csv", index=False)

    bbox_df, bbox_cols = load_feature_table("bbox")
    biu_bbox_df, biu_bbox_cols = load_feature_table("biu_bbox")

    def merge_context(feat_df: pd.DataFrame) -> pd.DataFrame:
        return feat_df.merge(context_df[["sequence_id"] + CONTEXT_COLS], on="sequence_id", how="inner")

    bbox_ctx_df = merge_context(bbox_df)
    biu_bbox_ctx_df = merge_context(biu_bbox_df)
    assert len(bbox_ctx_df) == len(bbox_df)
    assert len(biu_bbox_ctx_df) == len(biu_bbox_df)

    feature_sets = {
        "bbox_only": (bbox_df, bbox_cols),
        "biu_bbox": (biu_bbox_df, biu_bbox_cols),
        "context_only": (context_df, CONTEXT_COLS),
        "bbox_plus_context": (bbox_ctx_df, sorted(set(bbox_cols + CONTEXT_COLS))),
        "biu_bbox_plus_context": (biu_bbox_ctx_df, sorted(set(biu_bbox_cols + CONTEXT_COLS))),
        # Honest variants excluding the future-informed motion_direction cols.
        "context_no_motion_only": (context_df, CONTEXT_NO_MOTION_COLS),
        "bbox_plus_context_no_motion": (bbox_ctx_df, sorted(set(bbox_cols + CONTEXT_NO_MOTION_COLS))),
        "biu_bbox_plus_context_no_motion": (biu_bbox_ctx_df, sorted(set(biu_bbox_cols + CONTEXT_NO_MOTION_COLS))),
    }

    # Global leakage assertion on every feature column actually used.
    for name, (_, cols) in feature_sets.items():
        for col in cols:
            low = col.lower()
            assert "crossing_point" not in low and "decision_point" not in low and "intention" not in low, (
                f"Leaky column {col} in {name}"
            )
            assert low not in ("crossing", "crossing_attr", "label"), f"Leaky column {col} in {name}"

    pd.DataFrame([
        {"feature_set": k, "n_features": len(cols), "n_rows": len(df)}
        for k, (df, cols) in feature_sets.items()
    ]).to_csv(METRICS_DIR / "feature_set_summary.csv", index=False)

    results_df = run_benchmark(feature_sets)
    results_df.to_csv(METRICS_DIR / "jaad_context_benchmark_results_per_seed.csv", index=False)
    agg_df = aggregate(results_df)
    agg_df.to_csv(METRICS_DIR / "jaad_context_benchmark_results_mean_std.csv", index=False)

    headline = agg_df[agg_df["threshold_rule"] == "val_f1"][
        ["feature_set", "model_name", "test_f1_mean", "test_f1_std", "test_pr_auc_mean", "test_roc_auc_mean", "test_balanced_accuracy_mean"]
    ].sort_values(["feature_set", "model_name"])
    print(headline.to_string(index=False))
    print(json.dumps({"result_rows": len(results_df), "context_rows": len(context_df)}))


if __name__ == "__main__":
    main()
