"""Experiment 019: JAAD commitment verifier (Stage 2 trained on actual_positive vs ambiguous_negative).

Question: exp-011's Stage-2 models were generic crossing classifiers (trained on
positives vs ALL negatives). Should Stage 2 instead be trained on the discrimination
that actually matters inside the cascade — actual_positive vs ambiguous_negative —
using cross-horizon DELTA features (short-horizon value minus 30_60 value)?

Design (exploratory; train population is tiny: ~102 actual_positive vs ~27
ambiguous_negative tracks):
- Unit of analysis: TRACK, one canonical window per track per horizon
  (min tte_frames, then latest start_frame — exp-004 rule, identical to exp 011).
- Verifier features, per short horizon hs (separate variants): cross-horizon
  deltas d(f) = f(hs) - f(30_60) of bbox features and BIU v2 core features, plus
  (in one variant) the short-horizon static values.
  Variants: bbox_deltas_only, biu_v2_deltas_only, bbox_plus_biu_v2_deltas,
  deltas_plus_short_static.
- IMPORTANT DATA CAVEAT (checked in metrics/canonical_window_overlap.csv): the
  archived tte_15_30 split contains exactly ONE window per track, all at
  tte_frames = 30, and that window is IDENTICAL (same start_frame, same tte) to
  the canonical 30_60 window under the min-tte rule. Cross-horizon deltas
  30_60 -> 15_30 are therefore structurally zero (exp-004's own
  delta_30_60_to_15_30 columns are all zero as well), so the planned 15_30 delta
  variant is impossible with these archived splits and only 5_15 is run.
- Feature matrices are REBUILT from the exp-009 machinery (archived bbox feature
  tables + BIU v2 extraction from archived keypoints) because the exp-004
  transition table only contains BIU v1 features.
- Dimensionality control: top-K (K=20) features by |single-feature ROC-AUC - 0.5|
  for actual-vs-ambiguous computed on TRAIN ONLY.
- Models: logreg (L2, class_weight=balanced) and shallow RF (200 trees,
  min_samples_leaf=3), seeds 42/43/44, seed-ensemble mean probability.
- Standalone eval (test, once per variant/model): actual-vs-ambiguous ROC/PR-AUC.
- Staged eval: Stage 1 = exp-011 bbox_only@30_60 logreg seed ensemble
  (predictions reloaded from experiments/011_jaad_staged_bbox_biu_v2/predictions/),
  t1 = val recall-target threshold; Stage 2 = verifier probability, t2 = val
  recall-target threshold on val stage-1-passing tracks (011's conditional rule).
  Tracks missing verifier features keep the Stage-1 decision (counted).
  Baselines at the same recall target: (a) exp-011 staged bbox->biu_v2_core@5_15
  (and @15_30 for the 15_30 verifier), recomputed via the imported exp-011 module;
  (b) single-stage bbox@30_60. Track bootstrap 10k, seed 12345 (same resample
  stream as exp-011's bootstrap_delta, reimplemented vectorized for speed and
  verified against stg.bootstrap_delta on one config).
- All selection (features, model, thresholds) on train/validation only. Test is
  evaluated once per reported configuration.

Outputs: experiments/019_jaad_commitment_verifier/{metrics,figures}.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

CLEAN_ROOT = Path("/home/abdelrahman/Hadwa/BIU-PID-Distillation-clean")
SCRIPTS_DIR = CLEAN_ROOT / "scripts/experiments"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import jaad_biu_v2_benchmark_comparison as base  # noqa: E402
import jaad_staged_bbox_biu_v2 as stg  # noqa: E402

OUT_ROOT = CLEAN_ROOT / "experiments/019_jaad_commitment_verifier"
METRICS_DIR = OUT_ROOT / "metrics"
FIGURES_DIR = OUT_ROOT / "figures"

BASE_HORIZON = "30_60"
# 15_30 is excluded: its canonical window is identical to the canonical 30_60
# window in the archived splits, so 30_60->15_30 deltas are all exactly zero
# (see canonical_window_overlap.csv and the module docstring).
SHORT_HORIZONS = ["5_15"]
OVERLAP_CHECK_HORIZONS = ["15_30", "5_15"]
VARIANTS = ["bbox_deltas_only", "biu_v2_deltas_only", "bbox_plus_biu_v2_deltas", "deltas_plus_short_static"]
MODELS = ["logreg", "rf"]
SEEDS = [42, 43, 44]
TOP_K = 20
RECALL_TARGETS = [0.90, 0.95]
N_BOOT = 10000
BOOT_SEED = 12345
STAGE1_FS, STAGE1_MODEL = "bbox_only", "logreg"  # from exp-011 metrics/staged_model_selection.csv
BASELINE_S2_MODEL = {"15_30": "logreg", "5_15": "rf"}  # biu_v2_core chosen models in exp 011


# --------------------------------------------------------------------------- features
def build_family_frames() -> tuple[dict, dict, pd.DataFrame, list[dict]]:
    """Per horizon, track-indexed bbox and BIU-v2-core feature frames on canonical windows."""
    meta = stg.load_canonical_meta()
    track_info = meta.sort_values("horizon_label").groupby("track_key", as_index=False).first()[
        ["track_key", "split", "label", "group"]
    ]

    feature_dict = pd.read_csv(base.DICT_PATH)
    v2_prefixes = base._base_feature_prefixes(
        feature_dict,
        ["original_anatomical_biu", "lower_body_commitment_biu", "torso_center_commitment_biu"],
    )

    horizons = [BASE_HORIZON] + SHORT_HORIZONS
    canon_rows: dict[str, pd.DataFrame] = {}
    for h in horizons:
        rows = base._load_split_rows(h)
        ids = set(meta.loc[meta["horizon_label"] == h, "sequence_id"])
        canon_rows[h] = rows[rows["sequence_id"].isin(ids)].copy()

    # keypoint index for canonical windows only
    needed = base._needed_frame_keys(canon_rows)
    kp_index = base._load_keypoint_index(base.HORIZON_CONFIGS[BASE_HORIZON]["keypoints_jsonl"], needed)

    bbox_frames: dict[str, pd.DataFrame] = {}
    v2_frames: dict[str, pd.DataFrame] = {}
    bbox_cols_by_h: dict[str, set] = {}
    v2_cols_by_h: dict[str, set] = {}
    coverage_rows: list[dict] = []

    for h in horizons:
        id2track = meta.loc[meta["horizon_label"] == h].set_index("sequence_id")["track_key"]
        feat = base._load_existing_feature_frame(h, "bbox")
        # Drop meta columns from the archived feature table BEFORE merging: a
        # collision on 'label' would otherwise produce label_x/label_y columns
        # that escape the META_COLS filter and leak the target into the
        # short-static feature family.
        feat = feat.drop(columns=[c for c in feat.columns if c != "sequence_id" and c in base.META_COLS],
                         errors="ignore")
        merged = canon_rows[h][["sequence_id", "split", "label", "group"]].merge(feat, on="sequence_id", how="inner")
        cols = [c for c in base._numeric_feature_cols(merged) if "label" not in c]
        merged["track_key"] = merged["sequence_id"].map(id2track)
        bbox_frames[h] = merged.set_index("track_key")[cols]
        bbox_cols_by_h[h] = set(cols)

        v2_df, missing = base._extract_v2_for_rows(canon_rows[h], kp_index)
        v2_cols = base._select_prefixed_columns(v2_df, v2_prefixes)
        v2_df["track_key"] = v2_df["sequence_id"].map(id2track)
        v2_frames[h] = v2_df.set_index("track_key")[v2_cols]
        v2_cols_by_h[h] = set(v2_cols)

        for split in ["train", "val", "test"]:
            n_canon = int((canon_rows[h]["split"] == split).sum())
            coverage_rows.append({
                "horizon": h, "split": split, "family": "canonical_tracks", "n_tracks": n_canon,
            })
            for fam, frame in [("bbox", bbox_frames[h]), ("biu_v2_core", v2_frames[h])]:
                idx = frame.index.intersection(track_info.loc[track_info["split"] == split, "track_key"])
                coverage_rows.append({"horizon": h, "split": split, "family": fam, "n_tracks": int(len(idx))})

    bbox_common = sorted(set.intersection(*bbox_cols_by_h.values()))
    v2_common = sorted(set.intersection(*v2_cols_by_h.values()))
    for h in horizons:
        bbox_frames[h] = bbox_frames[h][bbox_common]
        v2_frames[h] = v2_frames[h][v2_common]
    return ({"bbox": bbox_frames, "v2": v2_frames}, {"bbox": bbox_common, "v2": v2_common},
            track_info, coverage_rows)


def build_variant_matrix(frames: dict, hs: str, variant: str) -> pd.DataFrame:
    """Track-indexed feature matrix for one (short horizon, variant)."""

    def delta(fam: str) -> pd.DataFrame:
        short, long = frames[fam][hs], frames[fam][BASE_HORIZON]
        common = short.index.intersection(long.index)
        d = short.loc[common] - long.loc[common]
        return d.add_prefix("dlt_")

    def static(fam: str) -> pd.DataFrame:
        return frames[fam][hs].add_prefix("sh_")

    parts = {
        "bbox_deltas_only": [delta("bbox")],
        "biu_v2_deltas_only": [delta("v2")],
        "bbox_plus_biu_v2_deltas": [delta("bbox"), delta("v2")],
        "deltas_plus_short_static": [delta("bbox"), delta("v2"), static("bbox"), static("v2")],
    }[variant]
    out = parts[0]
    for p in parts[1:]:
        out = out.join(p, how="inner")
    return out


# --------------------------------------------------------------------------- selection / training
def select_features(X: pd.DataFrame, y: np.ndarray, k: int) -> pd.DataFrame:
    rows = []
    for col in X.columns:
        x = X[col].to_numpy(dtype=float)
        valid = ~np.isnan(x)
        if valid.sum() < 10 or len(np.unique(y[valid])) < 2 or len(np.unique(x[valid])) < 2:
            continue
        auc = float(roc_auc_score(y[valid], x[valid]))
        rows.append({"feature": col, "train_auc": auc, "abs_auc_minus_half": abs(auc - 0.5)})
    if not rows:
        raise RuntimeError("no scorable features (all constant or too sparse) — degenerate variant")
    rank = pd.DataFrame(rows).sort_values("abs_auc_minus_half", ascending=False).reset_index(drop=True)
    rank["rank"] = np.arange(1, len(rank) + 1)
    rank["selected"] = rank["rank"] <= k
    return rank


def build_model(name: str, seed: int) -> Pipeline:
    if name == "logreg":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, penalty="l2", class_weight="balanced", random_state=seed)),
        ])
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(n_estimators=200, min_samples_leaf=3,
                                         class_weight="balanced_subsample", random_state=seed, n_jobs=-1)),
    ])


def canonical_window_overlap(meta: pd.DataFrame) -> pd.DataFrame:
    """How often the canonical short-horizon window is the exact same window as the canonical 30_60 one."""
    b = meta[meta["horizon_label"] == BASE_HORIZON].set_index("track_key")
    rows = []
    for hs in OVERLAP_CHECK_HORIZONS:
        s = meta[meta["horizon_label"] == hs].set_index("track_key")
        common = s.index.intersection(b.index)
        same = (s.loc[common, "start_frame"].to_numpy() == b.loc[common, "start_frame"].to_numpy()) & (
            s.loc[common, "tte_frames"].to_numpy() == b.loc[common, "tte_frames"].to_numpy()
        )
        rows.append({"short_horizon": hs, "n_tracks_both_horizons": int(len(common)),
                     "n_identical_canonical_window": int(same.sum()),
                     "frac_identical": float(same.mean()) if len(common) else float("nan")})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- fast bootstrap
def _fast_metrics(y: np.ndarray, pred: np.ndarray, amb: np.ndarray) -> tuple[float, float]:
    tp = int(np.sum((y == 1) & (pred == 1)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    amb_fpr = float(pred[amb].mean()) if amb.any() else float("nan")
    return f1, amb_fpr


def bootstrap_delta_fast(y: np.ndarray, groups: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict:
    """Vectorized reimplementation of stg.bootstrap_delta (same seed, same resample stream)."""
    rng = np.random.default_rng(BOOT_SEED)
    n = len(y)
    amb_all = groups == "ambiguous_negative"
    d_f1, d_amb = [], []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        aa = amb_all[idx]
        f1_a, amb_a = _fast_metrics(yy, pred_a[idx], aa)
        f1_b, amb_b = _fast_metrics(yy, pred_b[idx], aa)
        d_f1.append(f1_b - f1_a)
        if not (np.isnan(amb_a) or np.isnan(amb_b)):
            d_amb.append(amb_b - amb_a)

    def s(d: list, tag: str) -> dict:
        d = np.asarray(d)
        if len(d) == 0:
            return {f"{tag}_mean": float("nan")}
        return {
            f"{tag}_mean": float(d.mean()),
            f"{tag}_ci_low": float(np.percentile(d, 2.5)),
            f"{tag}_ci_high": float(np.percentile(d, 97.5)),
            f"{tag}_p": float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))),
        }

    return {**s(d_f1, "delta_f1"), **s(d_amb, "delta_amb_fpr")}


# --------------------------------------------------------------------------- main
def main() -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    frames, common_cols, track_info, coverage_rows = build_family_frames()
    tinfo = track_info.set_index("track_key")

    # ---------------- verifier matrices, feature selection, training, scoring
    selection_frames = []
    score_frames = []
    coverage_variant_rows = []
    train_pop_rows = []

    for hs in SHORT_HORIZONS:
        for variant in VARIANTS:
            X = build_variant_matrix(frames, hs, variant)
            info = tinfo.loc[X.index]
            for split in ["train", "val", "test"]:
                sub = info[info["split"] == split]
                row = {"short_horizon": hs, "variant": variant, "split": split,
                       "n_covered_tracks": int(len(sub))}
                for g in ["actual_positive", "ambiguous_negative", "clean_negative"]:
                    row[f"n_{g}"] = int((sub["group"] == g).sum())
                coverage_variant_rows.append(row)

            tr_mask = (info["split"] == "train") & info["group"].isin(["actual_positive", "ambiguous_negative"])
            X_tr = X[tr_mask.to_numpy()]
            y_tr = (info.loc[tr_mask.to_numpy(), "group"] == "actual_positive").astype(int).to_numpy()
            train_pop_rows.append({"short_horizon": hs, "variant": variant,
                                   "n_train_actual_positive": int(y_tr.sum()),
                                   "n_train_ambiguous_negative": int((1 - y_tr).sum()),
                                   "n_features_before_selection": int(X.shape[1])})

            rank = select_features(X_tr, y_tr, TOP_K)
            rank.insert(0, "variant", variant)
            rank.insert(0, "short_horizon", hs)
            selection_frames.append(rank)
            sel_cols = rank.loc[rank["selected"], "feature"].tolist()

            eval_mask = info["split"].isin(["val", "test"]).to_numpy()
            X_eval = X.loc[eval_mask, sel_cols]
            for model_name in MODELS:
                probs = []
                for seed in SEEDS:
                    pipe = build_model(model_name, seed)
                    pipe.fit(X_tr[sel_cols], y_tr)
                    probs.append(pipe.predict_proba(X_eval)[:, 1])
                ens = np.mean(probs, axis=0)
                sf = info.loc[eval_mask, ["split", "label", "group"]].copy()
                sf["prob"] = ens
                sf["short_horizon"] = hs
                sf["variant"] = variant
                sf["model_name"] = model_name
                score_frames.append(sf.reset_index())

    pd.concat(selection_frames, ignore_index=True).to_csv(METRICS_DIR / "verifier_feature_selection.csv", index=False)
    scores = pd.concat(score_frames, ignore_index=True)
    scores.to_csv(METRICS_DIR / "verifier_scores.csv", index=False)
    pd.DataFrame(coverage_rows).to_csv(METRICS_DIR / "coverage_family_counts.csv", index=False)
    pd.DataFrame(coverage_variant_rows).to_csv(METRICS_DIR / "coverage_variant_counts.csv", index=False)
    pd.DataFrame(train_pop_rows).to_csv(METRICS_DIR / "train_population_counts.csv", index=False)

    # ---------------- standalone verifier evaluation (actual vs ambiguous) + group score stats
    standalone_rows = []
    group_stat_rows = []
    for (hs, variant, model_name), sub in scores.groupby(["short_horizon", "variant", "model_name"]):
        row = {"short_horizon": hs, "variant": variant, "model_name": model_name}
        for split in ["val", "test"]:
            pa = sub[(sub["split"] == split) & sub["group"].isin(["actual_positive", "ambiguous_negative"])]
            y = (pa["group"] == "actual_positive").astype(int).to_numpy()
            p = pa["prob"].to_numpy()
            if len(np.unique(y)) == 2:
                row[f"{split}_roc_auc_actual_vs_amb"] = float(roc_auc_score(y, p))
                row[f"{split}_pr_auc_actual_vs_amb"] = float(average_precision_score(y, p))
            row[f"{split}_n_actual"] = int(y.sum())
            row[f"{split}_n_ambiguous"] = int((1 - y).sum())
        standalone_rows.append(row)
        for g, gsub in sub[sub["split"] == "test"].groupby("group"):
            group_stat_rows.append({"short_horizon": hs, "variant": variant, "model_name": model_name,
                                    "group": g, "n": int(len(gsub)),
                                    "score_mean": float(gsub["prob"].mean()),
                                    "score_median": float(gsub["prob"].median())})
    standalone = pd.DataFrame(standalone_rows)
    standalone.to_csv(METRICS_DIR / "verifier_standalone_auc.csv", index=False)
    pd.DataFrame(group_stat_rows).to_csv(METRICS_DIR / "verifier_test_group_score_stats.csv", index=False)

    # ---------------- model selection per (hs, variant): validation actual-vs-ambiguous ROC-AUC
    model_choice = {}
    choice_rows = []
    for (hs, variant), sub in standalone.groupby(["short_horizon", "variant"]):
        sub = sub.sort_values(["val_roc_auc_actual_vs_amb", "model_name"], ascending=[False, True])
        best = sub.iloc[0]
        model_choice[(hs, variant)] = best["model_name"]
        choice_rows.append({"short_horizon": hs, "variant": variant, "chosen_model": best["model_name"],
                            "val_roc_auc_actual_vs_amb": float(best["val_roc_auc_actual_vs_amb"]),
                            "note": "val has only 18 actual / 5 ambiguous tracks; selection is noisy"})
    pd.DataFrame(choice_rows).to_csv(METRICS_DIR / "verifier_model_selection.csv", index=False)

    # ---------------- stage-1 and exp-011 baseline tables (from exp-011 predictions)
    meta = stg.load_canonical_meta()
    canonical_window_overlap(meta).to_csv(METRICS_DIR / "canonical_window_overlap.csv", index=False)

    def track_frame(h: str, fs: str, m: str, split: str) -> pd.DataFrame:
        ens = stg.load_ensemble_probs(h, fs, m)
        merged = meta[(meta["horizon_label"] == h) & (meta["split"] == split)].merge(
            ens[["sequence_id", "prob"]], on="sequence_id", how="inner")
        return merged.set_index("track_key")

    s1_val = track_frame(BASE_HORIZON, STAGE1_FS, STAGE1_MODEL, "val")
    s1_test = track_frame(BASE_HORIZON, STAGE1_FS, STAGE1_MODEL, "test")

    staged_rows = []
    boot_rows = []
    verifier_scores_idx = scores.set_index(["short_horizon", "variant", "model_name"])

    for target in RECALL_TARGETS:
        t1 = stg.threshold_for_recall(s1_val["label"].to_numpy(), s1_val["prob"].to_numpy(), target)
        passed_val_idx = s1_val.index[s1_val["prob"] >= t1]

        # baseline (b): single-stage bbox@30_60 at same recall target
        pred_bbox = (s1_test["prob"] >= t1).astype(int).to_numpy()
        y_test = s1_test["label"].to_numpy()
        g_test = s1_test["group"].to_numpy()
        staged_rows.append({"config": "single|bbox_only|30_60", "recall_target": target,
                            "t1": t1, "t2": None, "n_stage2_missing_kept": None,
                            **stg.track_metrics(y_test, pred_bbox, g_test)})

        # baseline (a): exp-011 staged bbox -> biu_v2_core @ hs (recomputed via imported module)
        baseline_preds = {"bbox30_60": pd.Series(pred_bbox, index=s1_test.index)}
        for hs in SHORT_HORIZONS:
            s2_m = BASELINE_S2_MODEL[hs]
            s2_val = track_frame(hs, "biu_v2_core", s2_m, "val")
            s2_test = track_frame(hs, "biu_v2_core", s2_m, "test")
            s2_val_passed = s2_val.loc[s2_val.index.intersection(passed_val_idx)]
            if s2_val_passed.empty or s2_val_passed["label"].nunique() < 2:
                t2b = 0.5
            else:
                t2b = stg.threshold_for_recall(s2_val_passed["label"].to_numpy(),
                                               s2_val_passed["prob"].to_numpy(), target)
            joined = stg.staged_predict(s1_test, s2_test, t1, t2b)
            staged_rows.append({"config": f"staged011|bbox30_60->biu_v2_core@{hs}", "recall_target": target,
                                "t1": t1, "t2": t2b, "n_stage2_missing_kept":
                                    int(((joined["p1"] >= t1) & joined["p2"].isna()).sum()),
                                **stg.track_metrics(joined["label"].to_numpy(), joined["pred"].to_numpy(),
                                                    joined["group"].to_numpy())})
            baseline_preds[f"staged011_v2core_{hs}"] = joined["pred"]

        # verifier-staged configs
        for hs in SHORT_HORIZONS:
            for variant in VARIANTS:
                m = model_choice[(hs, variant)]
                v = verifier_scores_idx.loc[(hs, variant, m)].set_index("track_key")
                v_val = v[v["split"] == "val"]
                v_test = v[v["split"] == "test"]

                sel_idx = v_val.index.intersection(passed_val_idx)
                v_sel = v_val.loc[sel_idx]
                if v_sel.empty or v_sel["label"].nunique() < 2:
                    t2 = 0.5
                else:
                    t2 = stg.threshold_for_recall(v_sel["label"].to_numpy(), v_sel["prob"].to_numpy(), target)

                joined = s1_test[["label", "group", "prob"]].rename(columns={"prob": "p1"}).join(
                    v_test[["prob"]].rename(columns={"prob": "p2"}), how="left")
                stage1 = (joined["p1"] >= t1).astype(int)
                stage2 = np.where(joined["p2"].notna(), (joined["p2"] >= t2).astype(int), 1)
                joined["pred"] = (stage1.to_numpy() & stage2).astype(int)
                n_missing_kept = int(((stage1 == 1) & joined["p2"].isna()).sum())

                cfg = f"verifier|bbox30_60->{variant}@{hs}|{m}"
                staged_rows.append({"config": cfg, "recall_target": target, "t1": t1, "t2": t2,
                                    "n_stage2_missing_kept": n_missing_kept,
                                    **stg.track_metrics(joined["label"].to_numpy(), joined["pred"].to_numpy(),
                                                        joined["group"].to_numpy())})

                base_names = ["bbox30_60", f"staged011_v2core_{hs}"]
                if hs != "5_15":
                    base_names.append("staged011_v2core_5_15")
                for bname in base_names:
                    bpred = baseline_preds[bname]
                    common = joined.index.intersection(bpred.index)
                    res = bootstrap_delta_fast(joined.loc[common, "label"].to_numpy(),
                                               joined.loc[common, "group"].to_numpy(),
                                               bpred.loc[common].to_numpy(),
                                               joined.loc[common, "pred"].to_numpy())
                    boot_rows.append({"config": cfg, "recall_target": target, "baseline": bname,
                                      "n_common_tracks": int(len(common)), **res})

    pd.DataFrame(staged_rows).to_csv(METRICS_DIR / "staged_verifier_results.csv", index=False)
    pd.DataFrame(boot_rows).to_csv(METRICS_DIR / "staged_verifier_bootstrap.csv", index=False)

    # parity check: fast bootstrap vs stg.bootstrap_delta on one config
    y = s1_test["label"].to_numpy()
    g = s1_test["group"].to_numpy()
    pa = baseline_preds["bbox30_60"].to_numpy()
    pb = baseline_preds["staged011_v2core_5_15"].loc[s1_test.index].to_numpy()
    ref = stg.bootstrap_delta(y, g, pa, pb)
    fast = bootstrap_delta_fast(y, g, pa, pb)
    parity = {k: (ref[k], fast[k]) for k in ref}
    pd.DataFrame([{"key": k, "stg_value": a, "fast_value": b, "abs_diff": abs(a - b)}
                  for k, (a, b) in parity.items()]).to_csv(METRICS_DIR / "bootstrap_parity_check.csv", index=False)

    # ---------------- figure: verifier score distributions per test group (val-selected primary config)
    primary = max(choice_rows, key=lambda r: r["val_roc_auc_actual_vs_amb"])
    p_hs, p_var, p_m = primary["short_horizon"], primary["variant"], primary["chosen_model"]
    psc = scores[(scores["short_horizon"] == p_hs) & (scores["variant"] == p_var)
                 & (scores["model_name"] == p_m) & (scores["split"] == "test")]
    plt.figure(figsize=(8, 4.5))
    bins = np.linspace(0, 1, 21)
    colors = {"actual_positive": "#2166ac", "ambiguous_negative": "#d6604d", "clean_negative": "#4dac26"}
    for grp in ["actual_positive", "ambiguous_negative", "clean_negative"]:
        vals = psc.loc[psc["group"] == grp, "prob"]
        plt.hist(vals, bins=bins, alpha=0.55, label=f"{grp} (n={len(vals)})",
                 color=colors[grp], density=True)
    plt.xlabel("Verifier probability (actual-positive vs ambiguous-negative)")
    plt.ylabel("Density")
    plt.title(f"Test verifier scores by group — {p_var}@{p_hs} ({p_m}, val-selected)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "verifier_score_distributions.png", dpi=160)
    plt.close()

    summary = {
        "primary_config_val_selected": primary,
        "n_bootstrap_rows": len(boot_rows),
        "n_staged_rows": len(staged_rows),
        "bootstrap_parity_max_abs_diff": max(abs(a - b) for a, b in parity.values()),
    }
    (METRICS_DIR / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
