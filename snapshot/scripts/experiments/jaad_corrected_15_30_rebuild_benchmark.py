"""Experiment 021 step 4a: window-level benchmark on the corrected JAAD 15_30 table.

Reruns the quarantined exp-009 comparison on the corrected 15_30 windows:
feature sets {bbox_only, biu_v1_only, biu_v1_plus_bbox, biu_v2_core,
biu_v2_core_plus_bbox} x models {logreg, rf, xgb} (exp-009 `_build_model`)
x seeds {42, 43, 44}; thresholds fixed_0_5 + val_f1 (validation-only
selection). Saves per-sample val+test probabilities in the exp-011 prediction
format under experiments/021_jaad_corrected_15_30_rebuild/predictions/corrected_15_30/.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import numpy as np
import pandas as pd

CLEAN_ROOT = Path("/home/abdelrahman/Hadwa/BIU-PID-Distillation-clean")
SCRIPTS_DIR = CLEAN_ROOT / "scripts/experiments"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import jaad_biu_v2_benchmark_comparison as base  # noqa: E402  (exp-009 module)

OUT_ROOT = CLEAN_ROOT / "experiments/021_jaad_corrected_15_30_rebuild"
DATA_DIR = OUT_ROOT / "data/tte_15_30_corrected"
FEAT_DIR = OUT_ROOT / "data/features_tte_15_30_corrected"
METRICS_DIR = OUT_ROOT / "metrics"
SPLITS = ["train", "val", "test"]

PRED_DIR = OUT_ROOT / "predictions/corrected_15_30"
DICT_PATH = CLEAN_ROOT / "experiments/008_biu_v2_design_spec/metrics/biu_v2_feature_dictionary.csv"

HORIZON_LABEL = "15_30_corrected"
FEATURE_SETS = ["bbox_only", "biu_v1_only", "biu_v1_plus_bbox", "biu_v2_core", "biu_v2_core_plus_bbox"]
MODELS = ["logreg", "rf", "xgb"]
SEEDS = [42, 43, 44]
REPORTED_RULES = ["fixed_0_5", "val_f1"]


def load_base_rows() -> pd.DataFrame:
    dfs = []
    for split in SPLITS:
        df = pd.read_csv(DATA_DIR / f"{split}.csv")
        df["split"] = split
        dfs.append(df)
    out = pd.concat(dfs, ignore_index=True)
    out["horizon_label"] = HORIZON_LABEL
    out["group"] = out.apply(base._classify_group, axis=1)
    out["event_frame_or_anchor"] = out["event_frame"]
    return out


def load_feature_frame(name: str) -> pd.DataFrame:
    dfs = []
    for split in SPLITS:
        df = pd.read_csv(FEAT_DIR / f"{split}_{name}.csv")
        df["split"] = split
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def build_feature_sets(base_rows: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], dict[str, list[str]]]:
    feature_dict = pd.read_csv(DICT_PATH)
    bbox_df = base._merge_base_and_features(base_rows, load_feature_frame("bbox"))
    biu_v1_df = base._merge_base_and_features(base_rows, load_feature_frame("biu"))
    v2_df = base._merge_base_and_features(base_rows, load_feature_frame("biu_v2"))

    bbox_cols = base._numeric_feature_cols(bbox_df)
    v1_cols = base._numeric_feature_cols(biu_v1_df)
    anatomical = base._base_feature_prefixes(feature_dict, ["original_anatomical_biu"])
    lower = base._base_feature_prefixes(feature_dict, ["lower_body_commitment_biu"])
    torso = base._base_feature_prefixes(feature_dict, ["torso_center_commitment_biu"])
    v2_core_cols = sorted(
        set(
            base._select_prefixed_columns(v2_df, anatomical)
            + base._select_prefixed_columns(v2_df, lower)
            + base._select_prefixed_columns(v2_df, torso)
        )
    )

    frames = {
        "bbox_only": bbox_df,
        "biu_v1_only": biu_v1_df,
        "biu_v2_core": v2_df,
        "biu_v1_plus_bbox": biu_v1_df[[c for c in biu_v1_df.columns if c not in bbox_cols]].merge(
            bbox_df[["sequence_id"] + bbox_cols], on="sequence_id", how="inner"
        ),
        "biu_v2_core_plus_bbox": v2_df[[c for c in v2_df.columns if c not in bbox_cols]].merge(
            bbox_df[["sequence_id"] + bbox_cols], on="sequence_id", how="inner"
        ),
    }
    columns = {
        "bbox_only": bbox_cols,
        "biu_v1_only": v1_cols,
        "biu_v1_plus_bbox": sorted(set(v1_cols + bbox_cols)),
        "biu_v2_core": v2_core_cols,
        "biu_v2_core_plus_bbox": sorted(set(v2_core_cols + bbox_cols)),
    }
    return frames, columns


def main() -> None:
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    base_rows = load_base_rows()
    frames, columns = build_feature_sets(base_rows)

    results = []
    for feature_set in FEATURE_SETS:
        df = frames[feature_set]
        cols = columns[feature_set]
        assert not df["sequence_id"].duplicated().any(), feature_set
        train_df = df[df["split"] == "train"]
        val_df = df[df["split"] == "val"]
        test_df = df[df["split"] == "test"]
        X_train, y_train = train_df[cols].astype(float), train_df["label"].astype(int).to_numpy()
        X_val, y_val = val_df[cols].astype(float), val_df["label"].astype(int).to_numpy()
        X_test, y_test = test_df[cols].astype(float), test_df["label"].astype(int).to_numpy()

        for seed in SEEDS:
            for model_name in MODELS:
                pipe = base._build_model(model_name, seed, y_train)
                if pipe is None:
                    continue
                pipe.fit(X_train, y_train)
                p_val = pipe.predict_proba(X_val)[:, 1]
                p_test = pipe.predict_proba(X_test)[:, 1]

                pred_rows = []
                for split_df, probs, split in [(val_df, p_val, "val"), (test_df, p_test, "test")]:
                    out = split_df[["sequence_id", "split", "label", "group", "video_id", "ped_id", "track_id"]].copy()
                    out["prob"] = probs
                    out["horizon_label"] = HORIZON_LABEL
                    out["feature_set"] = feature_set
                    out["model_name"] = model_name
                    out["seed"] = seed
                    pred_rows.append(out)
                pd.concat(pred_rows, ignore_index=True).to_csv(
                    PRED_DIR / f"{feature_set}__{model_name}__seed{seed}.csv", index=False
                )

                for sel in base._select_thresholds(y_val, p_val):
                    if sel.rule not in REPORTED_RULES or sel.status != "available" or sel.threshold is None:
                        continue
                    test_metrics = base._compute_metrics(y_test, p_test, float(sel.threshold))
                    row = {
                        "horizon_label": HORIZON_LABEL,
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
                    row.update({f"test_{k}": v for k, v in test_metrics.items()})
                    results.append(row)
        print(f"done feature_set={feature_set}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(METRICS_DIR / "corrected_window_benchmark_results.csv", index=False)

    metric_cols = [c for c in results_df.columns if c.startswith("test_") or c.startswith("val_")]
    agg = results_df.groupby(["horizon_label", "feature_set", "model_name", "threshold_rule"], dropna=False)[metric_cols].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg = agg.reset_index()
    agg.to_csv(METRICS_DIR / "corrected_window_benchmark_results_mean_std.csv", index=False)
    print(json.dumps({"result_rows": len(results_df), "prediction_files": len(list(PRED_DIR.glob('*.csv')))}, indent=2))


if __name__ == "__main__":
    main()
