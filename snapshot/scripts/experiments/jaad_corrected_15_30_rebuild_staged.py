"""Experiment 021 step 4b: track-level staged rerun on the corrected JAAD 15_30 table.

Reuses exp-011 machinery (scripts/experiments/jaad_staged_bbox_biu_v2.py):
canonical rule (min tte_frames, then latest start_frame), exp-004 group
assignments, threshold_for_recall / threshold_val_f1, staged_predict,
bootstrap_delta (10k resamples, seed 12345).

Stage 1 = bbox_only@30_60 seed-ensemble probabilities from
experiments/011_jaad_staged_bbox_biu_v2/predictions/30_60/, with the model
exp-011 selected on validation (read from its staged_model_selection.csv).
Stage 2 = biu_v2_core / biu_v2_core_plus_bbox on the corrected 15_30 table
(predictions from experiments/021_.../predictions/corrected_15_30/).

Also: single-stage track-level rows at the corrected 15_30 for all five
feature sets, bootstrap of single-stage biu_v2_core vs bbox_only at the
corrected 15_30, old-vs-corrected comparison CSVs, and the report figure.
"""

from __future__ import annotations

import json
import os
import sys
from itertools import product
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CLEAN_ROOT = Path("/home/abdelrahman/Hadwa/BIU-PID-Distillation-clean")
SCRIPTS_DIR = CLEAN_ROOT / "scripts/experiments"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import jaad_biu_v2_benchmark_comparison as base  # noqa: E402  (exp-009 module)
import jaad_staged_bbox_biu_v2 as s11  # noqa: E402  (exp-011 module)
from jaad_corrected_15_30_rebuild import DATA_DIR, METRICS_DIR, OUT_ROOT, SPLITS  # noqa: E402

PRED_DIR_CORRECTED = OUT_ROOT / "predictions/corrected_15_30"
FIGURES_DIR = OUT_ROOT / "figures"

OLD_011_METRICS = CLEAN_ROOT / "experiments/011_jaad_staged_bbox_biu_v2/metrics"
OLD_009_MEAN_STD = CLEAN_ROOT / "experiments/009_jaad_biu_v2_benchmark_comparison/metrics/jaad_biu_v2_benchmark_results_mean_std.csv"

CORRECTED_HORIZON = "15_30_corrected"
FEATURE_SETS = ["bbox_only", "biu_v1_only", "biu_v1_plus_bbox", "biu_v2_core", "biu_v2_core_plus_bbox"]
MODELS = ["logreg", "rf", "xgb"]
SEEDS = [42, 43, 44]
RECALL_TARGETS = [0.90, 0.95]


def load_canonical_meta_corrected() -> pd.DataFrame:
    """Corrected-table analogue of s11.load_canonical_meta (same rules)."""
    assign = pd.read_csv(s11.GROUP_ASSIGNMENTS)[["track_key", "group"]].rename(columns={"group": "group_004"})
    dfs = []
    for split in SPLITS:
        df = pd.read_csv(DATA_DIR / f"{split}.csv")
        df["split"] = split
        dfs.append(df)
    rows = pd.concat(dfs, ignore_index=True)
    rows["group"] = rows.apply(base._classify_group, axis=1)
    rows["track_key"] = rows["video_id"].astype(str) + ":" + rows["ped_id"].astype(str)
    rows = rows.sort_values(["track_key", "tte_frames", "start_frame"], ascending=[True, True, False])
    canon = rows.groupby(["track_key", "split"], as_index=False).first()
    canon["horizon_label"] = CORRECTED_HORIZON
    canon = canon.merge(assign, on="track_key", how="left")
    n_missing = int(canon["group_004"].isna().sum())
    if n_missing:
        print(f"warning: {n_missing} corrected-15_30 tracks lack an exp-004 group assignment; keeping exp-009 rule for them")
    canon["group"] = canon["group_004"].fillna(canon["group"].replace({"negative_ambiguous": "ambiguous_negative"}))
    canon = canon.drop(columns=["group_004"])
    return canon[["sequence_id", "horizon_label", "track_key", "split", "label", "group", "tte_frames", "start_frame"]]


def load_canonical_meta_30_60() -> pd.DataFrame:
    meta = s11.load_canonical_meta()
    return meta[meta["horizon_label"] == "30_60"].copy()


def load_ensemble_probs_corrected(feature_set: str, model: str) -> pd.DataFrame:
    dfs = []
    for seed in SEEDS:
        path = PRED_DIR_CORRECTED / f"{feature_set}__{model}__seed{seed}.csv"
        dfs.append(pd.read_csv(path)[["sequence_id", "split", "label", "prob"]])
    cat = pd.concat(dfs, ignore_index=True)
    return cat.groupby(["sequence_id", "split", "label"], as_index=False)["prob"].mean()


def build_track_table() -> pd.DataFrame:
    out = []
    meta_30_60 = load_canonical_meta_30_60()
    for fs, m in product(["bbox_only"], MODELS):
        ens = s11.load_ensemble_probs("30_60", fs, m)
        merged = meta_30_60.merge(ens[["sequence_id", "prob"]], on="sequence_id", how="inner")
        merged["feature_set"] = fs
        merged["model_name"] = m
        out.append(merged)
    meta_cor = load_canonical_meta_corrected()
    for fs, m in product(FEATURE_SETS, MODELS):
        ens = load_ensemble_probs_corrected(fs, m)
        merged = meta_cor.merge(ens[["sequence_id", "prob"]], on="sequence_id", how="inner")
        merged["feature_set"] = fs
        merged["model_name"] = m
        out.append(merged)
    return pd.concat(out, ignore_index=True)


def make_figure() -> None:
    """Test F1 by feature set: broken vs corrected 15_30 (window level, both threshold rules)."""
    old = pd.read_csv(OLD_009_MEAN_STD)
    old = old[(old["horizon_label"] == "15_30") & (old["feature_set"].isin(FEATURE_SETS))]
    new = pd.read_csv(METRICS_DIR / "corrected_window_benchmark_results_mean_std.csv")

    surface, text_primary, text_secondary = "#fcfcfb", "#0b0b0b", "#52514e"
    color_broken, color_corrected = "#eda100", "#2a78d6"

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), dpi=150, facecolor=surface)
    for ax, rule, rule_label in zip(axes, ["fixed_0_5", "val_f1"], ["fixed threshold 0.5", "validation-F1 threshold"]):
        rows = []
        for fs in FEATURE_SETS:
            o = old[(old["feature_set"] == fs) & (old["threshold_rule"] == rule)]
            n = new[(new["feature_set"] == fs) & (new["threshold_rule"] == rule)]
            # validation-only model choice: highest val_f1_mean within the feature set
            o_best = o.loc[o["val_f1_mean"].idxmax()]
            n_best = n.loc[n["val_f1_mean"].idxmax()]
            rows.append((fs, o_best["test_f1_mean"], o_best["test_f1_std"], o_best["model_name"],
                         n_best["test_f1_mean"], n_best["test_f1_std"], n_best["model_name"]))
        x = np.arange(len(rows))
        width = 0.36
        ax.set_facecolor(surface)
        ax.bar(x - width / 2, [r[1] for r in rows], width, yerr=[r[2] for r in rows],
               label="broken 15_30 (exp 009)", color=color_broken, error_kw={"ecolor": text_secondary, "lw": 1}, zorder=3)
        ax.bar(x + width / 2, [r[4] for r in rows], width, yerr=[r[5] for r in rows],
               label="corrected 15_30 (exp 021)", color=color_corrected, error_kw={"ecolor": text_secondary, "lw": 1}, zorder=3)
        for xi, r in zip(x, rows):
            ax.text(xi - width / 2, r[1] + r[2] + 0.015, f"{r[1]:.2f}", ha="center", fontsize=7.5, color=text_secondary)
            ax.text(xi + width / 2, r[4] + r[5] + 0.015, f"{r[4]:.2f}", ha="center", fontsize=7.5, color=text_secondary)
        ax.set_xticks(x)
        ax.set_xticklabels([r[0].replace("_", "\n") for r in rows], fontsize=8, color=text_primary)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("test F1 (mean over seeds 42/43/44)", fontsize=9, color=text_primary)
        ax.set_title(f"JAAD 15_30 window-level test F1 — {rule_label}\n(val-F1-selected model per feature set)", fontsize=9.5, color=text_primary)
        ax.grid(axis="y", color="#e8e7e3", lw=0.8, zorder=0)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_color("#d8d7d3")
        ax.tick_params(colors=text_secondary, labelsize=8)
    axes[0].legend(fontsize=8, frameon=False, loc="upper left")
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "f1_by_feature_set_broken_vs_corrected_15_30.png", bbox_inches="tight", facecolor=surface)
    plt.close(fig)


def main() -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    track_table = build_track_table()
    track_table.to_csv(METRICS_DIR / "corrected_staged_track_table.csv", index=False)

    # Stage 1 model exactly as exp-011 selected it
    sel_011 = pd.read_csv(OLD_011_METRICS / "staged_model_selection.csv")
    s1_m = sel_011.loc[(sel_011["horizon_label"] == "30_60") & (sel_011["feature_set"] == "bbox_only"), "chosen_model"].item()

    # validation-based model choice per corrected-15_30 feature set (exp-011 rule)
    selection_rows = [{"horizon_label": "30_60", "feature_set": "bbox_only", "chosen_model": s1_m,
                       "source": "reused from exp-011 staged_model_selection.csv"}]
    chosen: dict[str, str] = {}
    for fs in FEATURE_SETS:
        m = s11.pick_model(track_table, CORRECTED_HORIZON, fs)
        chosen[fs] = m
        selection_rows.append({"horizon_label": CORRECTED_HORIZON, "feature_set": fs, "chosen_model": m,
                               "source": "validation F1 at val-F1 threshold (exp-011 pick_model)"})
    pd.DataFrame(selection_rows).to_csv(METRICS_DIR / "corrected_staged_model_selection.csv", index=False)

    results, op_rows, boot_rows = [], [], []

    # ---- single-stage baselines
    baseline_preds: dict[tuple[str, str, str], pd.DataFrame] = {}
    single_configs = [("30_60", "bbox_only", s1_m)] + [(CORRECTED_HORIZON, fs, chosen[fs]) for fs in FEATURE_SETS]
    for h, fs, m in single_configs:
        val = s11.get_split(track_table, h, fs, m, "val")
        test = s11.get_split(track_table, h, fs, m, "test")
        yv, pv = val["label"].to_numpy(), val["prob"].to_numpy()
        yt, pt = test["label"].to_numpy(), test["prob"].to_numpy()
        gt = test["group"].to_numpy()

        t = s11.threshold_val_f1(yv, pv)
        pred = (pt >= t).astype(int)
        results.append({"config": f"single|{fs}|{h}", "model": m, "rule": "val_f1", "t1": t, "t2": None,
                        **s11.track_metrics(yt, pred, gt)})
        baseline_preds[(h, fs, "val_f1")] = test.assign(pred=pred)

        for target in RECALL_TARGETS:
            tr = s11.threshold_for_recall(yv, pv, target)
            predr = (pt >= tr).astype(int)
            op_rows.append({"config": f"single|{fs}|{h}", "model": m, "recall_target": target, "t1": tr, "t2": None,
                            **s11.track_metrics(yt, predr, gt)})
            baseline_preds[(h, fs, f"recall_{target}")] = test.assign(pred=predr)

    # bootstrap: single-stage biu_v2_core vs bbox_only at corrected 15_30 (val_f1 rule)
    for challenger in ["biu_v2_core", "biu_v2_core_plus_bbox"]:
        a = baseline_preds[(CORRECTED_HORIZON, "bbox_only", "val_f1")]
        b = baseline_preds[(CORRECTED_HORIZON, challenger, "val_f1")]
        common = a.index.intersection(b.index)
        aa, bb = a.loc[common], b.loc[common]
        res = s11.bootstrap_delta(aa["label"].to_numpy(), aa["group"].to_numpy(),
                                  aa["pred"].to_numpy(), bb["pred"].to_numpy())
        boot_rows.append({"config": f"single|{challenger}|{CORRECTED_HORIZON}", "recall_target": None,
                          "baseline": f"single|bbox_only|{CORRECTED_HORIZON}", "n_common_tracks": len(common), **res})

    # ---- staged configurations (Stage 1 = exp-011 bbox_only@30_60 ensemble)
    s1_val = s11.get_split(track_table, "30_60", "bbox_only", s1_m, "val")
    s1_test = s11.get_split(track_table, "30_60", "bbox_only", s1_m, "test")

    for s2_fs in ["biu_v2_core", "biu_v2_core_plus_bbox"]:
        s2_m = chosen[s2_fs]
        s2_val = s11.get_split(track_table, CORRECTED_HORIZON, s2_fs, s2_m, "val")
        s2_test = s11.get_split(track_table, CORRECTED_HORIZON, s2_fs, s2_m, "test")
        cfg = f"staged|bbox30_60->{s2_fs}@{CORRECTED_HORIZON}"

        for target in RECALL_TARGETS:
            t1 = s11.threshold_for_recall(s1_val["label"].to_numpy(), s1_val["prob"].to_numpy(), target)
            passed = s1_val[(s1_val["prob"] >= t1)]
            s2_val_passed = s2_val.loc[s2_val.index.intersection(passed.index)]
            if s2_val_passed.empty or s2_val_passed["label"].nunique() < 2:
                t2 = 0.5
            else:
                t2 = s11.threshold_for_recall(s2_val_passed["label"].to_numpy(), s2_val_passed["prob"].to_numpy(), target)

            joined = s11.staged_predict(s1_test, s2_test, t1, t2)
            y, g, pred = joined["label"].to_numpy(), joined["group"].to_numpy(), joined["pred"].to_numpy()
            op_rows.append({"config": cfg, "model": f"{s1_m}+{s2_m}", "recall_target": target, "t1": t1,
                            "t2": t2, **s11.track_metrics(y, pred, g)})

            for base_key, base_name in [
                (("30_60", "bbox_only", f"recall_{target}"), "bbox30_60"),
                ((CORRECTED_HORIZON, "bbox_only", f"recall_{target}"), f"bbox@{CORRECTED_HORIZON}"),
                ((CORRECTED_HORIZON, "biu_v2_core_plus_bbox", f"recall_{target}"), f"v2bbox@{CORRECTED_HORIZON}"),
            ]:
                bdf = baseline_preds[base_key]
                common = joined.index.intersection(bdf.index)
                jb, bb = joined.loc[common], bdf.loc[common]
                res = s11.bootstrap_delta(jb["label"].to_numpy(), jb["group"].to_numpy(),
                                          bb["pred"].to_numpy(), jb["pred"].to_numpy())
                boot_rows.append({"config": cfg, "recall_target": target, "baseline": base_name,
                                  "n_common_tracks": len(common), **res})

        t1 = s11.threshold_val_f1(s1_val["label"].to_numpy(), s1_val["prob"].to_numpy())
        t2 = s11.threshold_val_f1(s2_val["label"].to_numpy(), s2_val["prob"].to_numpy())
        joined = s11.staged_predict(s1_test, s2_test, t1, t2)
        results.append({"config": cfg, "model": f"{s1_m}+{s2_m}", "rule": "val_f1", "t1": t1, "t2": t2,
                        **s11.track_metrics(joined["label"].to_numpy(), joined["pred"].to_numpy(),
                                            joined["group"].to_numpy())})

    pd.DataFrame(results).to_csv(METRICS_DIR / "corrected_staged_results.csv", index=False)
    pd.DataFrame(op_rows).to_csv(METRICS_DIR / "corrected_staged_operating_points.csv", index=False)
    pd.DataFrame(boot_rows).to_csv(METRICS_DIR / "corrected_staged_bootstrap.csv", index=False)

    # parity check: recomputed single|bbox_only|30_60 rows vs exp-011 archived rows
    old_op = pd.read_csv(OLD_011_METRICS / "staged_operating_points.csv")
    parity = []
    for target in RECALL_TARGETS:
        old_row = old_op[(old_op["config"] == "single|bbox_only|30_60") & (old_op["recall_target"] == target)].iloc[0]
        new_row = [r for r in op_rows if r["config"] == "single|bbox_only|30_60" and r["recall_target"] == target][0]
        parity.append({"recall_target": target,
                       "old_f1": old_row["f1"], "new_f1": new_row["f1"],
                       "old_amb_fpr": old_row["amb_neg_fpr"], "new_amb_fpr": new_row["amb_neg_fpr"],
                       "identical": bool(np.isclose(old_row["f1"], new_row["f1"]) and np.isclose(old_row["amb_neg_fpr"], new_row["amb_neg_fpr"]))})
    pd.DataFrame(parity).to_csv(METRICS_DIR / "stage1_parity_check_vs_011.csv", index=False)

    # ---- old-vs-corrected comparison CSVs
    old_ms = pd.read_csv(OLD_009_MEAN_STD)
    old_ms = old_ms[(old_ms["horizon_label"] == "15_30") & (old_ms["feature_set"].isin(FEATURE_SETS))
                    & (old_ms["threshold_rule"].isin(["fixed_0_5", "val_f1"]))]
    new_ms = pd.read_csv(METRICS_DIR / "corrected_window_benchmark_results_mean_std.csv")
    keep = ["feature_set", "model_name", "threshold_rule",
            "test_f1_mean", "test_f1_std", "test_precision_mean", "test_recall_mean",
            "test_pr_auc_mean", "test_roc_auc_mean"]
    cmp = old_ms[keep].merge(new_ms[keep], on=["feature_set", "model_name", "threshold_rule"],
                             suffixes=("_broken", "_corrected"))
    for col in ["test_f1_mean"]:
        cmp[f"{col}_delta"] = cmp[f"{col}_corrected"] - cmp[f"{col}_broken"]
    cmp.to_csv(METRICS_DIR / "old_vs_corrected_window_results.csv", index=False)

    old_staged = old_op[old_op["config"].str.contains(r"15_30|single\|bbox_only\|30_60", regex=True)].copy()
    old_staged["table"] = "broken_15_30_exp011"
    new_staged = pd.DataFrame(op_rows).copy()
    new_staged["table"] = "corrected_15_30_exp021"
    pd.concat([old_staged, new_staged], ignore_index=True).to_csv(METRICS_DIR / "old_vs_corrected_staged_operating_points.csv", index=False)

    make_figure()
    print(json.dumps({"results": len(results), "operating_points": len(op_rows), "bootstrap_rows": len(boot_rows)}, indent=2))


if __name__ == "__main__":
    main()
