"""Experiment 011 phase 2: staged BBox -> BIU v2 confirmation model on JAAD.

Design (all selection on validation only; test evaluated once per configuration):
- Unit of analysis: TRACK (video_id:ped_id). Per track and horizon we take one
  canonical window (minimum tte_frames, then latest start_frame — the exp-004
  rule) and use the seed-ensemble probability (mean over seeds 42/43/44).
- Stage 1 (relevance): bbox_only at 30_60, threshold chosen on validation for a
  recall target (its job is to not miss).
- Stage 2 (commitment confirmation): BIU v2 (biu_v2_core, and
  biu_v2_core_plus_bbox as variant) at 15_30 or 5_15, re-scoring only tracks
  flagged by Stage 1. Final positive = Stage1 AND Stage2.
- Baselines: single-stage bbox_only / biu_v2_core / biu_v2_core_plus_bbox at
  each horizon under the same track-level canonicalization.
- Matched-recall veto analysis: for target recalls {0.90, 0.95} select both
  stage thresholds on validation, report test ambiguous-negative FPR,
  clean-negative FPR, precision, recall, F1.
- Uncertainty: bootstrap over tracks (10k resamples, seed 12345) for the
  staged-vs-baseline deltas.

Inputs: experiments/011_jaad_staged_bbox_biu_v2/predictions/ (phase 1) +
archived split CSVs for tte_frames/start_frame canonicalization.
Outputs (experiments/011_jaad_staged_bbox_biu_v2/metrics/):
- staged_track_table.csv        (per-track canonical probs per horizon/set)
- staged_model_selection.csv    (validation-based model choice per stage)
- staged_results.csv            (staged + baselines at val-selected thresholds)
- staged_operating_points.csv   (matched-recall sweep)
- staged_bootstrap.csv          (track bootstrap deltas)
plus figures/ and report.md.
"""

from __future__ import annotations

import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

CLEAN_ROOT = Path("/home/abdelrahman/Hadwa/BIU-PID-Distillation-clean")
SCRIPTS_DIR = CLEAN_ROOT / "scripts/experiments"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import jaad_biu_v2_benchmark_comparison as base  # noqa: E402

OUT_ROOT = CLEAN_ROOT / "experiments/011_jaad_staged_bbox_biu_v2"
PRED_DIR = OUT_ROOT / "predictions"
METRICS_DIR = OUT_ROOT / "metrics"
FIGURES_DIR = OUT_ROOT / "figures"
REPORT_PATH = OUT_ROOT / "report.md"

HORIZONS = ["30_60", "15_30", "5_15"]
FEATURE_SETS = ["bbox_only", "biu_v1_only", "biu_v1_plus_bbox", "biu_v2_core", "biu_v2_core_plus_bbox"]
MODELS = ["logreg", "rf", "xgb"]
SEEDS = [42, 43, 44]
RECALL_TARGETS = [0.90, 0.95]
N_BOOT = 10000
BOOT_SEED = 12345


GROUP_ASSIGNMENTS = CLEAN_ROOT / "results/jaad_temporal_biu_transition_audit/jaad_three_group_assignments.csv"


def load_canonical_meta() -> pd.DataFrame:
    """sequence_id -> (horizon, track, split, label, group, canonical flag).

    Group labels use the exp-004 authoritative track-level three-group assignment
    (clean_negative = crossing_attr -1, ambiguous_negative = crossing_attr 0);
    the broader exp-009 window rule collapses every JAAD test negative into
    'ambiguous', which is too coarse for the veto analysis.
    """
    assign = pd.read_csv(GROUP_ASSIGNMENTS)[["track_key", "group"]].rename(columns={"group": "group_004"})
    frames = []
    for h in HORIZONS:
        rows = base._load_split_rows(h)
        rows["track_key"] = rows["video_id"].astype(str) + ":" + rows["ped_id"].astype(str)
        rows = rows.sort_values(["track_key", "tte_frames", "start_frame"], ascending=[True, True, False])
        canon = rows.groupby(["track_key", "split"], as_index=False).first()
        canon["horizon_label"] = h
        canon = canon.merge(assign, on="track_key", how="left")
        n_missing = int(canon["group_004"].isna().sum())
        if n_missing:
            print(f"warning: {n_missing} tracks in {h} lack an exp-004 group assignment; keeping exp-009 rule for them")
        canon["group"] = canon["group_004"].fillna(canon["group"].replace({"negative_ambiguous": "ambiguous_negative"}))
        canon = canon.drop(columns=["group_004"])
        frames.append(canon[["sequence_id", "horizon_label", "track_key", "split", "label", "group", "tte_frames", "start_frame"]])
    return pd.concat(frames, ignore_index=True)


def load_ensemble_probs(horizon: str, feature_set: str, model: str) -> pd.DataFrame:
    dfs = []
    for seed in SEEDS:
        path = PRED_DIR / horizon / f"{feature_set}__{model}__seed{seed}.csv"
        dfs.append(pd.read_csv(path)[["sequence_id", "split", "label", "group", "prob"]])
    cat = pd.concat(dfs, ignore_index=True)
    ens = cat.groupby(["sequence_id", "split", "label", "group"], as_index=False)["prob"].mean()
    return ens


def build_track_table() -> pd.DataFrame:
    """One row per (track, horizon, feature_set, model) with the canonical-window ensemble prob."""
    meta = load_canonical_meta()
    out = []
    for h, fs, m in product(HORIZONS, FEATURE_SETS, MODELS):
        ens = load_ensemble_probs(h, fs, m)
        merged = meta[meta["horizon_label"] == h].merge(ens[["sequence_id", "prob"]], on="sequence_id", how="inner")
        merged["feature_set"] = fs
        merged["model_name"] = m
        out.append(merged)
    return pd.concat(out, ignore_index=True)


def track_metrics(y: np.ndarray, pred: np.ndarray, groups: np.ndarray) -> dict[str, float]:
    amb = groups == "ambiguous_negative"
    cln = groups == "clean_negative"
    return {
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "amb_neg_fpr": float(pred[amb].mean()) if amb.any() else float("nan"),
        "clean_neg_fpr": float(pred[cln].mean()) if cln.any() else float("nan"),
        "n_tracks": int(len(y)),
        "n_amb": int(amb.sum()),
        "n_clean": int(cln.sum()),
        "n_pos": int((y == 1).sum()),
    }


def threshold_for_recall(y_val: np.ndarray, p_val: np.ndarray, target: float) -> float:
    """Highest threshold whose validation recall >= target (fallback: lowest prob)."""
    cands = np.unique(np.concatenate(([0.5], p_val)))[::-1]
    for t in cands:
        if recall_score(y_val, (p_val >= t).astype(int), zero_division=0) >= target:
            return float(t)
    return float(cands[-1])


def threshold_val_f1(y_val: np.ndarray, p_val: np.ndarray) -> float:
    cands = np.unique(np.concatenate(([0.5], p_val)))
    best_t, best = 0.5, -1.0
    for t in cands:
        s = f1_score(y_val, (p_val >= t).astype(int), zero_division=0)
        if s > best:
            best, best_t = s, float(t)
    return best_t


def pick_model(track_table: pd.DataFrame, horizon: str, feature_set: str) -> str:
    """Choose the classifier per (horizon, feature set) by validation F1 at the val-F1 threshold."""
    best_model, best_score = None, -1.0
    for m in MODELS:
        sub = track_table[
            (track_table["horizon_label"] == horizon)
            & (track_table["feature_set"] == feature_set)
            & (track_table["model_name"] == m)
            & (track_table["split"] == "val")
        ]
        if sub.empty:
            continue
        y, p = sub["label"].to_numpy(), sub["prob"].to_numpy()
        t = threshold_val_f1(y, p)
        s = f1_score(y, (p >= t).astype(int), zero_division=0)
        if s > best_score:
            best_score, best_model = s, m
    return best_model


def get_split(track_table: pd.DataFrame, horizon: str, feature_set: str, model: str, split: str) -> pd.DataFrame:
    sub = track_table[
        (track_table["horizon_label"] == horizon)
        & (track_table["feature_set"] == feature_set)
        & (track_table["model_name"] == model)
        & (track_table["split"] == split)
    ]
    return sub.set_index("track_key")


def staged_predict(s1: pd.DataFrame, s2: pd.DataFrame, t1: float, t2: float) -> pd.DataFrame:
    """Join stage tables on track. Tracks missing from stage 2 keep the stage-1 decision."""
    joined = s1[["label", "group", "prob"]].rename(columns={"prob": "p1"}).join(
        s2[["prob"]].rename(columns={"prob": "p2"}), how="left"
    )
    stage1 = (joined["p1"] >= t1).astype(int)
    stage2 = np.where(joined["p2"].notna(), (joined["p2"] >= t2).astype(int), 1)
    joined["pred"] = (stage1 & stage2).astype(int)
    joined["stage1_pred"] = stage1
    return joined


def bootstrap_delta(y: np.ndarray, groups: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict[str, float]:
    rng = np.random.default_rng(BOOT_SEED)
    n = len(y)
    d_f1, d_amb = [], []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        yy, gg = y[idx], groups[idx]
        if yy.min() == yy.max():
            continue
        ma = track_metrics(yy, pred_a[idx], gg)
        mb = track_metrics(yy, pred_b[idx], gg)
        d_f1.append(mb["f1"] - ma["f1"])
        if not (np.isnan(ma["amb_neg_fpr"]) or np.isnan(mb["amb_neg_fpr"])):
            d_amb.append(mb["amb_neg_fpr"] - ma["amb_neg_fpr"])

    def s(d, tag):
        d = np.array(d)
        if len(d) == 0:
            return {f"{tag}_mean": float("nan")}
        return {
            f"{tag}_mean": float(d.mean()),
            f"{tag}_ci_low": float(np.percentile(d, 2.5)),
            f"{tag}_ci_high": float(np.percentile(d, 97.5)),
            f"{tag}_p": float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))),
        }

    return {**s(d_f1, "delta_f1"), **s(d_amb, "delta_amb_fpr")}


def main() -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    track_table = build_track_table()
    track_table.to_csv(METRICS_DIR / "staged_track_table.csv", index=False)

    # validation-based model choice per (horizon, feature_set)
    selection_rows = []
    chosen: dict[tuple[str, str], str] = {}
    for h, fs in product(HORIZONS, FEATURE_SETS):
        m = pick_model(track_table, h, fs)
        chosen[(h, fs)] = m
        selection_rows.append({"horizon_label": h, "feature_set": fs, "chosen_model": m})
    pd.DataFrame(selection_rows).to_csv(METRICS_DIR / "staged_model_selection.csv", index=False)

    results = []
    op_rows = []
    boot_rows = []

    # -------- single-stage baselines (val-F1 threshold + matched-recall points)
    baseline_preds: dict[tuple[str, str, str], pd.DataFrame] = {}
    for h, fs in product(HORIZONS, FEATURE_SETS):
        m = chosen[(h, fs)]
        val = get_split(track_table, h, fs, m, "val")
        test = get_split(track_table, h, fs, m, "test")
        yv, pv = val["label"].to_numpy(), val["prob"].to_numpy()
        yt, pt = test["label"].to_numpy(), test["prob"].to_numpy()
        gt = test["group"].to_numpy()

        t = threshold_val_f1(yv, pv)
        pred = (pt >= t).astype(int)
        results.append({"config": f"single|{fs}|{h}", "model": m, "rule": "val_f1", "t1": t, "t2": None,
                        **track_metrics(yt, pred, gt)})
        baseline_preds[(h, fs, "val_f1")] = test.assign(pred=pred)

        for target in RECALL_TARGETS:
            tr = threshold_for_recall(yv, pv, target)
            predr = (pt >= tr).astype(int)
            op_rows.append({"config": f"single|{fs}|{h}", "model": m, "recall_target": target, "t1": tr, "t2": None,
                            **track_metrics(yt, predr, gt)})
            baseline_preds[(h, fs, f"recall_{target}")] = test.assign(pred=predr)

    # -------- staged configurations
    s1_fs = "bbox_only"
    s1_m = chosen[("30_60", s1_fs)]
    s1_val = get_split(track_table, "30_60", s1_fs, s1_m, "val")
    s1_test = get_split(track_table, "30_60", s1_fs, s1_m, "test")

    for s2_h, s2_fs in product(["15_30", "5_15"], ["biu_v2_core", "biu_v2_core_plus_bbox"]):
        s2_m = chosen[(s2_h, s2_fs)]
        s2_val = get_split(track_table, s2_h, s2_fs, s2_m, "val")
        s2_test = get_split(track_table, s2_h, s2_fs, s2_m, "test")
        cfg = f"staged|bbox30_60->{s2_fs}@{s2_h}"

        for target in RECALL_TARGETS:
            # stage 1 threshold: recall target on val; stage 2 threshold: recall target on the
            # val tracks that pass stage 1 (conditional selection, validation only)
            t1 = threshold_for_recall(s1_val["label"].to_numpy(), s1_val["prob"].to_numpy(), target)
            passed = s1_val[(s1_val["prob"] >= t1)]
            s2_val_passed = s2_val.loc[s2_val.index.intersection(passed.index)]
            if s2_val_passed.empty or s2_val_passed["label"].nunique() < 2:
                t2 = 0.5
            else:
                t2 = threshold_for_recall(s2_val_passed["label"].to_numpy(), s2_val_passed["prob"].to_numpy(), target)

            joined = staged_predict(s1_test, s2_test, t1, t2)
            y, g, pred = joined["label"].to_numpy(), joined["group"].to_numpy(), joined["pred"].to_numpy()
            op_rows.append({"config": cfg, "model": f"{s1_m}+{s2_m}", "recall_target": target, "t1": t1,
                            "t2": t2, **track_metrics(y, pred, g)})

            # bootstrap vs the two strongest single-stage baselines at the same recall target
            for base_key, base_name in [
                (("30_60", "bbox_only", f"recall_{target}"), "bbox30_60"),
                ((s2_h, "bbox_only", f"recall_{target}"), f"bbox{s2_h}"),
                ((s2_h, "biu_v2_core_plus_bbox", f"recall_{target}"), f"v2bbox{s2_h}"),
            ]:
                bdf = baseline_preds[base_key]
                common = joined.index.intersection(bdf.index)
                jb, bb = joined.loc[common], bdf.loc[common]
                res = bootstrap_delta(jb["label"].to_numpy(), jb["group"].to_numpy(),
                                      bb["pred"].to_numpy(), jb["pred"].to_numpy())
                boot_rows.append({"config": cfg, "recall_target": target, "baseline": base_name,
                                  "n_common_tracks": len(common), **res})

        # also a plain val-F1-threshold staged variant
        t1 = threshold_val_f1(s1_val["label"].to_numpy(), s1_val["prob"].to_numpy())
        t2 = threshold_val_f1(s2_val["label"].to_numpy(), s2_val["prob"].to_numpy())
        joined = staged_predict(s1_test, s2_test, t1, t2)
        results.append({"config": cfg, "model": f"{s1_m}+{s2_m}", "rule": "val_f1", "t1": t1, "t2": t2,
                        **track_metrics(joined["label"].to_numpy(), joined["pred"].to_numpy(),
                                        joined["group"].to_numpy())})

    pd.DataFrame(results).to_csv(METRICS_DIR / "staged_results.csv", index=False)
    pd.DataFrame(op_rows).to_csv(METRICS_DIR / "staged_operating_points.csv", index=False)
    pd.DataFrame(boot_rows).to_csv(METRICS_DIR / "staged_bootstrap.csv", index=False)

    print(json.dumps({"results": len(results), "operating_points": len(op_rows), "bootstrap_rows": len(boot_rows)}, indent=2))


if __name__ == "__main__":
    main()
