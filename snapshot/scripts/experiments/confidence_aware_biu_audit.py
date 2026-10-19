"""Experiment 014: confidence-aware BIU audit on JAAD.

Question: do low-confidence skeletons explain the remaining JAAD BIU false
positives/negatives, and does adding confidence metadata help?

Analyses
A. Test-set FP/FN stratification by window skeleton-confidence quartile
   (quartile edges from train+val only), per horizon and feature set.
B. Same stratification by lower-body keypoint confidence (COCO 11-16).
C. Selective prediction: abstain on the lowest-decile / lowest-quartile
   skeleton-confidence test windows (cutoffs from train+val), coverage vs
   F1/FPR at fixed threshold 0.5.
D. Confidence-gated feature variant: retrain biu_v2_core PLUS the BIU v2
   confidence_metadata columns (the control skipped in experiment 009) with
   the exact 009/011 model builders, seeds 42/43/44, and compare against a
   same-run biu_v2_core retrain using validation-only model selection.

Reuses saved per-sample predictions from experiments/011_jaad_staged_bbox_biu_v2
for A-C (no retraining) and the machinery of
scripts/experiments/jaad_biu_v2_benchmark_comparison.py for D.

Threshold policy: fixed 0.5 everywhere (project default). Model choice per
feature set uses VALIDATION F1 at 0.5 on the seed-ensemble mean probability.
Test data is never used for any selection.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLEAN_ROOT = Path("/home/abdelrahman/Hadwa/BIU-PID-Distillation-clean")
ARCHIVE_ROOT = Path("/home/abdelrahman/Hadwa/BIU-PID-Distillation")

SCRIPTS_DIR = CLEAN_ROOT / "scripts/experiments"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import jaad_biu_v2_benchmark_comparison as bench  # noqa: E402

OUT_ROOT = CLEAN_ROOT / "experiments/014_confidence_aware_biu_audit"
METRICS_DIR = OUT_ROOT / "metrics"
FIGURES_DIR = OUT_ROOT / "figures"

PRED_ROOT = CLEAN_ROOT / "experiments/011_jaad_staged_bbox_biu_v2/predictions"
KEYPOINTS_JSONL = ARCHIVE_ROOT / "data/jaad_keypoints_vitpose_s_extended_short_horizon.jsonl"
DICT_PATH = CLEAN_ROOT / "experiments/008_biu_v2_design_spec/metrics/biu_v2_feature_dictionary.csv"

HORIZONS = ["30_60", "15_30", "5_15"]
FEATURE_SETS_EVAL = ["bbox_only", "biu_v1_only", "biu_v2_core", "biu_v2_core_plus_bbox"]
MODELS = ["logreg", "rf", "xgb"]
SEEDS = [42, 43, 44]
THRESHOLD = 0.5

# COCO indices.
LOWER_BODY_IDX = [11, 12, 13, 14, 15, 16]  # hips, knees, ankles
TORSO_IDX = [5, 6, 11, 12]
HEAD_IDX = [0, 1, 2, 3, 4]
# Required-keypoint set of the archive BIU v1 low-confidence gate
# (src/features/biu_features.py: LOW_CONFIDENCE_THRESHOLD = 0.2).
V1_GATE_IDX = [0, 3, 4, 5, 6, 9, 10, 11, 12, 15, 16]
V1_GATE_THRESHOLD = 0.2

CONF_MEASURES = {
    "skeleton_conf_mean": "window mean of per-frame skeleton_conf",
    "skeleton_conf_min": "window min of per-frame skeleton_conf",
    "lower_body_kp_conf_mean": "window mean of per-frame mean kp_conf over COCO 11-16",
}
ABSTAIN_FRACTIONS = [0.0, 0.05, 0.10, 0.25, 0.50]


# ---------------------------------------------------------------------------
# Window-level confidence aggregates
# ---------------------------------------------------------------------------

def compute_window_confidence(split_rows: dict[str, pd.DataFrame], kp_index: dict) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for horizon, df in split_rows.items():
        rows = []
        for _, rec in df.iterrows():
            frames = bench._parse_json_list(rec["observed_frames_json"])
            skel = []
            kp = []
            for frame in frames:
                item = kp_index.get((str(rec["video_id"]), str(rec["ped_id"]), int(frame)))
                if item is None:
                    continue
                skel.append(float(item["skeleton_conf"]))
                kp.append(np.asarray(item["kp_conf"], dtype=float))
            row: dict[str, Any] = {
                "sequence_id": rec["sequence_id"],
                "split": rec["split"],
                "label": int(rec["label"]),
                "group": rec["group"],
                "video_id": rec["video_id"],
                "ped_id": rec["ped_id"],
                "horizon_label": horizon,
                "n_frames_expected": len(frames),
                "n_frames_with_keypoints": len(skel),
            }
            if skel:
                kp_arr = np.stack(kp)  # [T, 17]
                lower_per_frame = np.nanmean(kp_arr[:, LOWER_BODY_IDX], axis=1)
                gate_min_per_frame = np.nanmin(kp_arr[:, V1_GATE_IDX], axis=1)
                row.update(
                    {
                        "skeleton_conf_mean": float(np.nanmean(skel)),
                        "skeleton_conf_min": float(np.nanmin(skel)),
                        "lower_body_kp_conf_mean": float(np.nanmean(lower_per_frame)),
                        "lower_body_kp_conf_min": float(np.nanmin(lower_per_frame)),
                        "v1_gate_low_conf_frame_ratio": float(np.mean(gate_min_per_frame < V1_GATE_THRESHOLD)),
                    }
                )
            else:
                row.update(
                    {
                        "skeleton_conf_mean": np.nan,
                        "skeleton_conf_min": np.nan,
                        "lower_body_kp_conf_mean": np.nan,
                        "lower_body_kp_conf_min": np.nan,
                        "v1_gate_low_conf_frame_ratio": np.nan,
                    }
                )
            rows.append(row)
        out[horizon] = pd.DataFrame(rows)
    return out


# ---------------------------------------------------------------------------
# Seed-ensemble predictions from experiment 011 (read-only reuse)
# ---------------------------------------------------------------------------

def load_ensemble_predictions(horizon: str, feature_set: str, model: str) -> pd.DataFrame:
    dfs = []
    for seed in SEEDS:
        path = PRED_ROOT / horizon / f"{feature_set}__{model}__seed{seed}.csv"
        df = pd.read_csv(path)[["sequence_id", "split", "label", "group", "prob"]]
        df = df.rename(columns={"prob": f"prob_seed{seed}"})
        dfs.append(df)
    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.merge(df, on=["sequence_id", "split", "label", "group"], how="inner")
    prob_cols = [f"prob_seed{s}" for s in SEEDS]
    merged["prob"] = merged[prob_cols].mean(axis=1)
    return merged.drop(columns=prob_cols)


def f1_at_threshold(y: np.ndarray, p: np.ndarray, thr: float = THRESHOLD) -> float:
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    denom = 2 * tp + fp + fn
    return float(2 * tp / denom) if denom else 0.0


def select_models_by_val_f1() -> tuple[pd.DataFrame, dict[tuple[str, str], pd.DataFrame]]:
    """Pick one model per (horizon, feature_set) using VALIDATION F1 at 0.5
    on the seed-ensemble mean probability. Returns the selection table and
    the ensemble prediction frame of the selected model."""
    rows = []
    selected: dict[tuple[str, str], pd.DataFrame] = {}
    for horizon in HORIZONS:
        for feature_set in FEATURE_SETS_EVAL:
            best = None
            for model in MODELS:
                ens = load_ensemble_predictions(horizon, feature_set, model)
                val = ens[ens["split"] == "val"]
                val_f1 = f1_at_threshold(val["label"].to_numpy(int), val["prob"].to_numpy(float))
                rows.append(
                    {
                        "horizon_label": horizon,
                        "feature_set": feature_set,
                        "model_name": model,
                        "val_f1_at_0p5_ensemble": val_f1,
                        "n_val": int(len(val)),
                        "n_test": int((ens["split"] == "test").sum()),
                    }
                )
                if best is None or val_f1 > best[0]:
                    best = (val_f1, model, ens)
            selected[(horizon, feature_set)] = best[2]
            for r in rows:
                if r["horizon_label"] == horizon and r["feature_set"] == feature_set:
                    r["selected"] = int(r["model_name"] == best[1])
    return pd.DataFrame(rows), selected


# ---------------------------------------------------------------------------
# A/B. Quartile stratification of test errors
# ---------------------------------------------------------------------------

def quartile_edges(conf_df: pd.DataFrame, measure: str) -> np.ndarray:
    trainval = conf_df[conf_df["split"].isin(["train", "val"])][measure].dropna().to_numpy(float)
    return np.quantile(trainval, [0.25, 0.50, 0.75])


def stratify_errors(
    selected: dict[tuple[str, str], pd.DataFrame],
    conf_by_horizon: dict[str, pd.DataFrame],
    selection_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for horizon in HORIZONS:
        conf_df = conf_by_horizon[horizon]
        for measure in CONF_MEASURES:
            edges = quartile_edges(conf_df, measure)
            bins = [-np.inf, *edges.tolist(), np.inf]
            for feature_set in FEATURE_SETS_EVAL:
                ens = selected[(horizon, feature_set)]
                model = selection_df[
                    (selection_df["horizon_label"] == horizon)
                    & (selection_df["feature_set"] == feature_set)
                    & (selection_df["selected"] == 1)
                ]["model_name"].iloc[0]
                test = ens[ens["split"] == "test"].merge(
                    conf_df[["sequence_id", measure]], on="sequence_id", how="inner"
                )
                test = test.dropna(subset=[measure]).copy()
                test["quartile"] = pd.cut(test[measure], bins=bins, labels=["Q1", "Q2", "Q3", "Q4"])
                test["pred"] = (test["prob"] >= THRESHOLD).astype(int)
                for q in ["Q1", "Q2", "Q3", "Q4"]:
                    sub = test[test["quartile"] == q]
                    neg = sub[sub["label"] == 0]
                    pos = sub[sub["label"] == 1]
                    fp = int((neg["pred"] == 1).sum())
                    fn = int((pos["pred"] == 0).sum())
                    rows.append(
                        {
                            "horizon_label": horizon,
                            "conf_measure": measure,
                            "feature_set": feature_set,
                            "model_name": model,
                            "quartile": q,
                            "edge_q25": float(edges[0]),
                            "edge_q50": float(edges[1]),
                            "edge_q75": float(edges[2]),
                            "n_windows": int(len(sub)),
                            "n_negative": int(len(neg)),
                            "n_positive": int(len(pos)),
                            "fp_count": fp,
                            "fn_count": fn,
                            "fp_rate_among_negatives": float(fp / len(neg)) if len(neg) else np.nan,
                            "fn_rate_among_positives": float(fn / len(pos)) if len(pos) else np.nan,
                        }
                    )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# C. Selective prediction (abstain on low skeleton confidence)
# ---------------------------------------------------------------------------

def selective_prediction(
    selected: dict[tuple[str, str], pd.DataFrame],
    conf_by_horizon: dict[str, pd.DataFrame],
    selection_df: pd.DataFrame,
) -> pd.DataFrame:
    measure = "skeleton_conf_mean"
    rows = []
    for horizon in HORIZONS:
        conf_df = conf_by_horizon[horizon]
        trainval = conf_df[conf_df["split"].isin(["train", "val"])][measure].dropna().to_numpy(float)
        for feature_set in FEATURE_SETS_EVAL:
            ens = selected[(horizon, feature_set)]
            model = selection_df[
                (selection_df["horizon_label"] == horizon)
                & (selection_df["feature_set"] == feature_set)
                & (selection_df["selected"] == 1)
            ]["model_name"].iloc[0]
            test = ens[ens["split"] == "test"].merge(
                conf_df[["sequence_id", measure]], on="sequence_id", how="inner"
            )
            test = test.dropna(subset=[measure])
            n_total = len(test)
            for frac in ABSTAIN_FRACTIONS:
                cutoff = float(np.quantile(trainval, frac)) if frac > 0 else -np.inf
                kept = test[test[measure] >= cutoff]
                y = kept["label"].to_numpy(int)
                p = kept["prob"].to_numpy(float)
                m = bench._compute_metrics(y, p, THRESHOLD) if len(np.unique(y)) == 2 else None
                rows.append(
                    {
                        "horizon_label": horizon,
                        "feature_set": feature_set,
                        "model_name": model,
                        "abstain_fraction_trainval": frac,
                        "conf_cutoff": cutoff if np.isfinite(cutoff) else np.nan,
                        "n_test_total": n_total,
                        "n_test_kept": int(len(kept)),
                        "coverage": float(len(kept) / n_total) if n_total else np.nan,
                        "f1": m["f1"] if m else np.nan,
                        "precision": m["precision"] if m else np.nan,
                        "recall": m["recall"] if m else np.nan,
                        "fpr": m["fpr"] if m else np.nan,
                        "fnr": m["fnr"] if m else np.nan,
                        "balanced_accuracy": m["balanced_accuracy"] if m else np.nan,
                    }
                )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# D. Confidence-gated feature variant (retrain)
# ---------------------------------------------------------------------------

def confidence_metadata_for_rows(df: pd.DataFrame, kp_index: dict) -> pd.DataFrame:
    """Replicates the confidence_metadata block of src/biu_v2_extractor.py
    (extract_biu_v2 with include_optional=True) for windows with full
    keypoint coverage: per-frame nanmean over keypoint subsets, then
    mean/min summaries (SUMMARY_POLICY['confidence_metadata'] = 'mean;min')."""
    rows = []
    for _, rec in df.iterrows():
        frames = bench._parse_json_list(rec["observed_frames_json"])
        kp = []
        complete = True
        for frame in frames:
            item = kp_index.get((str(rec["video_id"]), str(rec["ped_id"]), int(frame)))
            if item is None:
                complete = False
                break
            kp.append(np.asarray(item["kp_conf"], dtype=float))
        if not complete or not kp:
            continue
        conf = np.stack(kp)
        series = {
            "skeleton_conf": np.nanmean(conf, axis=1),
            "lower_body_conf": np.nanmean(conf[:, LOWER_BODY_IDX], axis=1),
            "torso_conf": np.nanmean(conf[:, TORSO_IDX], axis=1),
            "head_conf": np.nanmean(conf[:, HEAD_IDX], axis=1),
        }
        row = {"sequence_id": rec["sequence_id"]}
        for name, vals in series.items():
            row[f"{name}_mean"] = float(np.nanmean(vals))
            row[f"{name}_min"] = float(np.nanmin(vals))
        rows.append(row)
    return pd.DataFrame(rows)


def run_gated_variant(split_rows: dict[str, pd.DataFrame], kp_index: dict, feature_dict: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    anatomical = bench._base_feature_prefixes(feature_dict, ["original_anatomical_biu"])
    lower = bench._base_feature_prefixes(feature_dict, ["lower_body_commitment_biu"])
    torso = bench._base_feature_prefixes(feature_dict, ["torso_center_commitment_biu"])

    per_seed_rows = []
    selected_rows = []
    parity_rows = []

    for horizon, base in split_rows.items():
        v2_df, missing = bench._extract_v2_for_rows(base, kp_index)
        merged = bench._merge_base_and_features(base, v2_df)
        core_cols = sorted(
            set(
                bench._select_prefixed_columns(merged, anatomical)
                + bench._select_prefixed_columns(merged, lower)
                + bench._select_prefixed_columns(merged, torso)
            )
        )
        conf_meta = confidence_metadata_for_rows(base, kp_index)
        conf_cols = sorted(c for c in conf_meta.columns if c != "sequence_id")
        merged = merged.merge(conf_meta, on="sequence_id", how="inner")

        variants = {
            "biu_v2_core_retrained": core_cols,
            "biu_v2_core_with_confidence_metadata": sorted(set(core_cols + conf_cols)),
        }

        train_df = merged[merged["split"] == "train"]
        val_df = merged[merged["split"] == "val"]
        test_df = merged[merged["split"] == "test"]
        y_train = train_df["label"].astype(int).to_numpy()
        y_val = val_df["label"].astype(int).to_numpy()
        y_test = test_df["label"].astype(int).to_numpy()

        for variant, cols in variants.items():
            ens_val = {m: [] for m in MODELS}
            ens_test = {m: [] for m in MODELS}
            for model in MODELS:
                for seed in SEEDS:
                    pipe = bench._build_model(model, seed, y_train)
                    if pipe is None:
                        continue
                    pipe.fit(train_df[cols].astype(float), y_train)
                    p_val = pipe.predict_proba(val_df[cols].astype(float))[:, 1]
                    p_test = pipe.predict_proba(test_df[cols].astype(float))[:, 1]
                    ens_val[model].append(p_val)
                    ens_test[model].append(p_test)
                    vm = bench._compute_metrics(y_val, p_val, THRESHOLD)
                    tm = bench._compute_metrics(y_test, p_test, THRESHOLD)
                    row = {
                        "horizon_label": horizon,
                        "variant": variant,
                        "model_name": model,
                        "seed": seed,
                        "n_features": len(cols),
                        "threshold_rule": "fixed_0_5",
                    }
                    row.update({f"val_{k}": v for k, v in vm.items()})
                    row.update({f"test_{k}": v for k, v in tm.items()})
                    per_seed_rows.append(row)

            # Validation-only model selection on the seed-ensemble mean prob.
            best = None
            for model in MODELS:
                if not ens_val[model]:
                    continue
                p_val = np.mean(np.stack(ens_val[model]), axis=0)
                vf1 = f1_at_threshold(y_val, p_val)
                if best is None or vf1 > best[0]:
                    best = (vf1, model)
            model = best[1]
            p_val = np.mean(np.stack(ens_val[model]), axis=0)
            p_test = np.mean(np.stack(ens_test[model]), axis=0)
            vm = bench._compute_metrics(y_val, p_val, THRESHOLD)
            tm = bench._compute_metrics(y_test, p_test, THRESHOLD)
            sel = {
                "horizon_label": horizon,
                "variant": variant,
                "selected_model": model,
                "selection_rule": "val_f1_at_fixed_0p5_on_seed_ensemble",
                "n_features": len(variants[variant]),
                "n_missing_keypoint_rows": len(missing),
            }
            sel.update({f"val_{k}": v for k, v in vm.items()})
            sel.update({f"test_{k}": v for k, v in tm.items()})
            selected_rows.append(sel)

            # Parity check of the core retrain against 011 saved predictions.
            if variant == "biu_v2_core_retrained":
                for m in MODELS:
                    if not ens_test[m]:
                        continue
                    saved = load_ensemble_predictions(horizon, "biu_v2_core", m)
                    saved_test = saved[saved["split"] == "test"][["sequence_id", "prob"]]
                    new = pd.DataFrame(
                        {
                            "sequence_id": test_df["sequence_id"].to_numpy(),
                            "prob_new": np.mean(np.stack(ens_test[m]), axis=0),
                        }
                    )
                    joined = saved_test.merge(new, on="sequence_id", how="inner")
                    parity_rows.append(
                        {
                            "horizon_label": horizon,
                            "model_name": m,
                            "n_joined": int(len(joined)),
                            "max_abs_prob_diff": float((joined["prob"] - joined["prob_new"]).abs().max()),
                            "pearson_r": float(joined["prob"].corr(joined["prob_new"])),
                        }
                    )

    return pd.DataFrame(per_seed_rows), pd.DataFrame(selected_rows), pd.DataFrame(parity_rows)


def aggregate_gated(per_seed: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [c for c in per_seed.columns if c.startswith("val_") or c.startswith("test_")]
    agg = per_seed.groupby(["horizon_label", "variant", "model_name"])[metric_cols].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    return agg.reset_index()


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def make_figures(strat_df: pd.DataFrame, sel_df: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    colors = {
        "bbox_only": "#4477AA",
        "biu_v1_only": "#EE6677",
        "biu_v2_core": "#228833",
        "biu_v2_core_plus_bbox": "#CCBB44",
    }
    fig, axes = plt.subplots(len(HORIZONS), 2, figsize=(11, 11), sharex=True)
    sub_all = strat_df[strat_df["conf_measure"] == "skeleton_conf_mean"]
    for i, horizon in enumerate(HORIZONS):
        for j, (col, title) in enumerate(
            [("fp_rate_among_negatives", "FP rate among negatives"), ("fn_rate_among_positives", "FN rate among positives")]
        ):
            ax = axes[i, j]
            for fs in FEATURE_SETS_EVAL:
                sub = sub_all[(sub_all["horizon_label"] == horizon) & (sub_all["feature_set"] == fs)]
                sub = sub.set_index("quartile").reindex(["Q1", "Q2", "Q3", "Q4"])
                ax.plot(["Q1", "Q2", "Q3", "Q4"], sub[col], marker="o", label=fs, color=colors[fs])
            ax.set_title(f"{horizon}: {title} (fixed 0.5)")
            ax.set_ylim(-0.02, 1.02)
            ax.grid(alpha=0.3)
            if i == len(HORIZONS) - 1:
                ax.set_xlabel("Skeleton-confidence quartile (Q1 = lowest, train+val edges)")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("JAAD test errors by window skeleton-confidence quartile")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(FIGURES_DIR / "error_rates_by_skeleton_conf_quartile.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(HORIZONS), figsize=(14, 4.2), sharey=True)
    for i, horizon in enumerate(HORIZONS):
        ax = axes[i]
        for fs in FEATURE_SETS_EVAL:
            sub = sel_df[(sel_df["horizon_label"] == horizon) & (sel_df["feature_set"] == fs)].sort_values("coverage")
            ax.plot(sub["coverage"], sub["f1"], marker="o", label=fs, color=colors[fs])
        ax.set_title(f"{horizon}")
        ax.set_xlabel("Coverage (test)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Test F1 at fixed 0.5")
    axes[0].legend(fontsize=8)
    fig.suptitle("Selective prediction: abstain on lowest skeleton_conf_mean windows (train+val cutoffs)")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(FIGURES_DIR / "selective_prediction_coverage_vs_f1.png", dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    feature_dict = pd.read_csv(DICT_PATH)
    split_rows = {h: bench._load_split_rows(h) for h in HORIZONS}
    needed = bench._needed_frame_keys(split_rows)
    kp_index = bench._load_keypoint_index(KEYPOINTS_JSONL, needed)

    conf_by_horizon = compute_window_confidence(split_rows, kp_index)
    for horizon, df in conf_by_horizon.items():
        df.to_csv(METRICS_DIR / f"window_confidence_{horizon}.csv", index=False)

    selection_df, selected = select_models_by_val_f1()
    selection_df.to_csv(METRICS_DIR / "model_selection_val_f1_fixed_0p5.csv", index=False)

    strat_df = stratify_errors(selected, conf_by_horizon, selection_df)
    strat_df.to_csv(METRICS_DIR / "error_stratification_by_confidence_quartile.csv", index=False)

    sel_df = selective_prediction(selected, conf_by_horizon, selection_df)
    sel_df.to_csv(METRICS_DIR / "selective_prediction_skeleton_conf.csv", index=False)

    per_seed, gated_selected, parity = run_gated_variant(split_rows, kp_index, feature_dict)
    per_seed.to_csv(METRICS_DIR / "gated_variant_per_seed_results.csv", index=False)
    aggregate_gated(per_seed).to_csv(METRICS_DIR / "gated_variant_results_mean_std.csv", index=False)
    gated_selected.to_csv(METRICS_DIR / "gated_variant_selected_model_test_results.csv", index=False)
    parity.to_csv(METRICS_DIR / "gated_variant_core_retrain_parity_vs_011.csv", index=False)

    make_figures(strat_df, sel_df)

    print(
        json.dumps(
            {
                "windows_with_confidence": {h: int(df["skeleton_conf_mean"].notna().sum()) for h, df in conf_by_horizon.items()},
                "stratification_rows": int(len(strat_df)),
                "selective_rows": int(len(sel_df)),
                "gated_per_seed_rows": int(len(per_seed)),
                "parity_max_abs_prob_diff": float(parity["max_abs_prob_diff"].max()) if not parity.empty else None,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
