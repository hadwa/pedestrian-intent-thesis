"""Experiment 011 phase 1: rescore the exp-009 JAAD BIU v2 benchmark saving
per-sample probabilities (the artifact exp 009 did not save).

Reuses exp 009's loaders/model builders verbatim (imported from
scripts/experiments/jaad_biu_v2_benchmark_comparison.py) so probabilities are
consistent with the published 009 aggregate numbers. Adds a parity check
against 009's mean/std CSV.

Feature sets: bbox_only, biu_v1_only, biu_v1_plus_bbox, biu_v2_core,
biu_v2_core_plus_bbox. Models: logreg, rf, xgb. Seeds: 42/43/44.
Horizons: 30_60, 15_30, 5_15. Selection: validation only. Test never selects.

Outputs (experiments/011_jaad_staged_bbox_biu_v2/):
- predictions/{horizon}/{feature_set}__{model}__seed{seed}.csv
  (val+test rows: sequence_id, split, label, group, track ids, prob)
- metrics/rescore_thresholds.csv       (validation-selected thresholds)
- metrics/rescore_parity_check.csv     (fixed_0_5 + val_f1 test F1 vs exp 009)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CLEAN_ROOT = Path("/home/abdelrahman/Hadwa/BIU-PID-Distillation-clean")
SCRIPTS_DIR = CLEAN_ROOT / "scripts/experiments"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import jaad_biu_v2_benchmark_comparison as base  # noqa: E402

OUT_ROOT = CLEAN_ROOT / "experiments/011_jaad_staged_bbox_biu_v2"
PRED_DIR = OUT_ROOT / "predictions"
METRICS_DIR = OUT_ROOT / "metrics"

FEATURE_SETS = [
    "bbox_only",
    "biu_v1_only",
    "biu_v1_plus_bbox",
    "biu_v2_core",
    "biu_v2_core_plus_bbox",
]
MODELS = ["logreg", "rf", "xgb"]
ID_COLS = ["sequence_id", "split", "label", "group", "video_id", "ped_id", "track_id"]


def main() -> None:
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    feature_dict = pd.read_csv(base.DICT_PATH)
    split_rows = {h: base._load_split_rows(h) for h in base.HORIZON_CONFIGS}
    needed = base._needed_frame_keys(split_rows)
    keypoints_path = next(iter(base.HORIZON_CONFIGS.values()))["keypoints_jsonl"]
    kp_index = base._load_keypoint_index(keypoints_path, needed)

    v2_rows = {}
    n_missing = 0
    for horizon_label, df in split_rows.items():
        v2_df, missing = base._extract_v2_for_rows(df, kp_index)
        v2_rows[horizon_label] = v2_df
        n_missing += len(missing)

    data_by_horizon, _defs, _cov = base._build_feature_sets(split_rows, v2_rows, feature_dict)

    threshold_rows = []
    parity_rows = []
    for horizon_label, feature_map in data_by_horizon.items():
        column_map = feature_map["_column_map"]
        hdir = PRED_DIR / horizon_label
        hdir.mkdir(parents=True, exist_ok=True)
        for feature_set in FEATURE_SETS:
            cols = column_map[feature_set]
            df = feature_map[feature_set].copy()
            train_df = df[df["split"] == "train"]
            val_df = df[df["split"] == "val"]
            test_df = df[df["split"] == "test"]
            X_train = train_df[cols].astype(float)
            y_train = train_df["label"].astype(int).to_numpy()
            X_val = val_df[cols].astype(float)
            y_val = val_df["label"].astype(int).to_numpy()
            X_test = test_df[cols].astype(float)
            y_test = test_df["label"].astype(int).to_numpy()

            for seed in base.SEEDS:
                for model_name in MODELS:
                    pipe = base._build_model(model_name, seed, y_train)
                    if pipe is None:
                        continue
                    pipe.fit(X_train, y_train)
                    p_val = pipe.predict_proba(X_val)[:, 1]
                    p_test = pipe.predict_proba(X_test)[:, 1]

                    out = pd.concat(
                        [
                            val_df[ID_COLS].assign(prob=p_val),
                            test_df[ID_COLS].assign(prob=p_test),
                        ],
                        ignore_index=True,
                    )
                    out["horizon_label"] = horizon_label
                    out["feature_set"] = feature_set
                    out["model_name"] = model_name
                    out["seed"] = seed
                    out.to_csv(hdir / f"{feature_set}__{model_name}__seed{seed}.csv", index=False)

                    for sel in base._select_thresholds(y_val, p_val):
                        row = {
                            "horizon_label": horizon_label,
                            "feature_set": feature_set,
                            "model_name": model_name,
                            "seed": seed,
                            "threshold_rule": sel.rule,
                            "threshold_status": sel.status,
                            "selected_threshold": sel.threshold,
                        }
                        if sel.status == "available" and sel.threshold is not None:
                            tm = base._compute_metrics(y_test, p_test, float(sel.threshold))
                            row.update({f"test_{k}": v for k, v in tm.items()})
                            row.update({f"val_{k}": v for k, v in sel.val_metrics.items()})
                        threshold_rows.append(row)
                        if sel.rule in ("fixed_0_5", "val_f1") and sel.status == "available":
                            parity_rows.append(
                                {
                                    "horizon_label": horizon_label,
                                    "feature_set": feature_set,
                                    "model_name": model_name,
                                    "seed": seed,
                                    "threshold_rule": sel.rule,
                                    "test_f1_rescore": row.get("test_f1"),
                                    "test_pr_auc_rescore": row.get("test_pr_auc"),
                                }
                            )
        print(f"done horizon {horizon_label}")

    thr_df = pd.DataFrame(threshold_rows)
    thr_df.to_csv(METRICS_DIR / "rescore_thresholds.csv", index=False)

    # parity check vs the published 009 aggregates
    parity = pd.DataFrame(parity_rows)
    parity_mean = (
        parity.groupby(["horizon_label", "feature_set", "model_name", "threshold_rule"])
        [["test_f1_rescore", "test_pr_auc_rescore"]]
        .mean()
        .reset_index()
    )
    ref = pd.read_csv(base.METRICS_DIR / "jaad_biu_v2_benchmark_results_mean_std.csv")
    ref = ref[ref["threshold_rule"].isin(["fixed_0_5", "val_f1"])][
        ["horizon_label", "feature_set", "model_name", "threshold_rule", "test_f1_mean", "test_pr_auc_mean"]
    ]
    merged = parity_mean.merge(ref, on=["horizon_label", "feature_set", "model_name", "threshold_rule"], how="left")
    merged["f1_abs_diff"] = (merged["test_f1_rescore"] - merged["test_f1_mean"]).abs()
    merged["pr_auc_abs_diff"] = (merged["test_pr_auc_rescore"] - merged["test_pr_auc_mean"]).abs()
    merged.to_csv(METRICS_DIR / "rescore_parity_check.csv", index=False)

    print(
        json.dumps(
            {
                "prediction_files": sum(1 for _ in PRED_DIR.rglob("*.csv")),
                "missing_keypoint_rows": n_missing,
                "parity_max_f1_abs_diff": float(merged["f1_abs_diff"].max()),
                "parity_max_pr_auc_abs_diff": float(merged["pr_auc_abs_diff"].max()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
