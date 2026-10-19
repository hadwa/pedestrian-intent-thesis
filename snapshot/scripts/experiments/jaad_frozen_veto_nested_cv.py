"""Experiment 027: frozen exp-023 veto pipeline under nested video-disjoint CV.

Exp-023 C1b found an L1-logreg actual-vs-ambiguous "veto" stage 2 behind
exp-011's bbox@30_60 stage 1 (test F1 0.882 / amb-FPR 1/32 at recall 0.90 vs
generic staged 0.805 / 0.094). That single-split result is contaminated by
adaptive test reuse (the test split informed feature-family discovery across
experiments) and by saturated val selection. This experiment FREEZES the entire
exp-023 procedure and re-evaluates it with nested video-disjoint grouped
cross-validation: all training and selection happen inside each outer training
partition; outer folds are untouched. This validates procedure STABILITY, not
discovery.

Pre-registered endpoints (fixed before the nested run; see README):
- Primary: pooled-outer-fold delta-F1 / delta-amb-FPR of frozen veto vs frozen
  generic staged baseline (bbox@30_60 -> biu_v2_core@5_15) at recall 0.90.
- Secondary: same at recall 0.95.
- Inference: 10k-resample paired track bootstrap, seed 12345, pooled outer
  predictions (exp-011 bootstrap_delta).

Parity gate: before the nested evaluation, the reused machinery must reproduce
exp-023's saved single-split C1b primary numbers (train on original train,
select on original val, evaluate on original test) and the generic staged
baseline operating points. If parity fails the script STOPS.

Frozen procedure (exp-023, no innovation):
- canonical window: min tte_frames then latest start_frame per (track, split)
- stage 1: bbox_only@30_60, seeds 42/43/44 ensemble, model in {logreg,rf,xgb}
  by exp-011 pick_model rule on inner-val; t1 = threshold_for_recall(inner-val)
- stage 2 veto: L1 logreg (liblinear, balanced, impute-median + standardize) on
  inner-train actual_positive vs ambiguous_negative tracks; features EXACTLY
  exp-023 build_rule_features (context no-motion + bbox deltas 30_60->5_15 +
  BIU-v2 core statics @5_15 + skeleton confidence); C in {0.05,0.1,0.5,1.0} by
  inner-val AUC, lexicographic tie-break; veto threshold = threshold_for_recall
  on true labels of stage-1-passed inner-val tracks (fallback 0.5); missing
  veto features -> keep stage-1 decision.
- baseline through the IDENTICAL nested scheme: bbox@30_60 -> biu_v2_core@5_15
  with stage-2 model retrained per outer partition, thresholds on inner-val.
- forbidden features guarded by exp-023 assert_clean_features on every matrix.

Nested scheme: outer = 5 video-disjoint StratifiedGroupKFold folds (seed 42,
stratified on the exp-004 three-group membership) over all 333 pooled tracks;
inner = first valid ~80/20 video-disjoint stratified split (seed 42) inside
each outer-train partition, serving ALL selections.

Rerun:
cd /home/abdelrahman/Hadwa/BIU-PID-Distillation-clean && \
  MPLCONFIGDIR=/tmp/mplconfig python3 scripts/experiments/jaad_frozen_veto_nested_cv.py

Outputs: experiments/027_jaad_frozen_veto_nested_cv/{metrics,logs,cache}
"""

from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Inconsistent values: penalty=l1")

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

CLEAN_ROOT = Path("/home/abdelrahman/Hadwa/BIU-PID-Distillation-clean")
SCRIPTS_DIR = CLEAN_ROOT / "scripts/experiments"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import jaad_biu_v2_benchmark_comparison as base  # noqa: E402  (exp-009 machinery)
import jaad_staged_bbox_biu_v2 as s11  # noqa: E402  (exp-011 machinery)
import jaad_ambiguous_negative_pattern_mining as s23  # noqa: E402  (exp-023 machinery, caches read-only)

OUT_ROOT = CLEAN_ROOT / "experiments/027_jaad_frozen_veto_nested_cv"
METRICS_DIR = OUT_ROOT / "metrics"
LOGS_DIR = OUT_ROOT / "logs"
CACHE_DIR = OUT_ROOT / "cache"
LOG_PATH = LOGS_DIR / "run.log"

CR23_PATH = s23.METRICS_DIR / "combo_rule_results.csv"
OPS23_PATH = s23.METRICS_DIR / "staged_baseline_operating_points.csv"

MODELS = ["logreg", "rf", "xgb"]
SEEDS = [42, 43, 44]
RECALL_TARGETS = [0.90, 0.95]
C_GRID = [0.05, 0.1, 0.5, 1.0]
OUTER_K = 5
FOLD_SEED = 42
STAGE_CONFIGS = [("30_60", "bbox_only", "s1"), ("5_15", "biu_v2_core", "s2")]
BASELINE_NAME = "generic_staged_bbox30_60_to_biu_v2_core_5_15"
VETO_NAME = "frozen_veto_bbox30_60_to_l1_rule"


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


# =========================================================================== frozen building blocks
def l1_pipeline(C: float, seed: int) -> Pipeline:
    """Exactly exp-023's L1 rule pipeline."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(penalty="l1", solver="liblinear", C=C,
                                     class_weight="balanced", max_iter=2000,
                                     random_state=seed)),
    ])


def fit_l1_ensemble(X_tr: pd.DataFrame, y_tr: np.ndarray, X_all: pd.DataFrame,
                    index_all: np.ndarray, C: float) -> pd.Series:
    """Seed-42/43/44 ensemble veto probabilities for every row of X_all."""
    probs = []
    for seed in SEEDS:
        pipe = l1_pipeline(C, seed)
        pipe.fit(X_tr, y_tr)
        probs.append(pipe.predict_proba(X_all)[:, 1])
    return pd.Series(np.mean(probs, axis=0), index=index_all)


def select_l1_C(cands: dict[str, pd.Series], val_keys: np.ndarray,
                y_val: np.ndarray) -> tuple[str, pd.DataFrame]:
    """Exp-023 rule_model_selection rule: AUC desc, lexicographic name asc."""
    rows = []
    for name, s in cands.items():
        pv = s.loc[val_keys].to_numpy()
        auc = float(roc_auc_score(y_val, pv)) if len(np.unique(y_val)) == 2 else float("nan")
        rows.append({"rule_model": name, "val_roc_auc_actual_vs_amb": auc})
    sel = pd.DataFrame(rows).sort_values(["val_roc_auc_actual_vs_amb", "rule_model"],
                                         ascending=[False, True]).reset_index(drop=True)
    return sel.iloc[0]["rule_model"], sel


def veto_threshold(s1_val: pd.DataFrame, v_val: pd.DataFrame, t1: float, target: float) -> float:
    """Exp-023 phase_c1b veto threshold: matched recall on the true crossing
    labels of stage-1-passed validation tracks (fallback 0.5)."""
    passed_idx = s1_val.index[s1_val["prob"] >= t1]
    v_sel = v_val.loc[v_val.index.intersection(passed_idx)]
    lbl = s1_val.loc[v_sel.index, "label"].to_numpy()
    if v_sel.empty or len(np.unique(lbl)) < 2:
        return 0.5
    return s11.threshold_for_recall(lbl, v_sel["prob"].to_numpy(), target)


def baseline_t2(s1_val: pd.DataFrame, s2_val: pd.DataFrame, t1: float, target: float) -> float:
    """Exp-023 recompute_staged_baselines conditional stage-2 threshold."""
    passed = s1_val[s1_val["prob"] >= t1]
    s2_val_passed = s2_val.loc[s2_val.index.intersection(passed.index)]
    if s2_val_passed.empty or s2_val_passed["label"].nunique() < 2:
        return 0.5
    return s11.threshold_for_recall(s2_val_passed["label"].to_numpy(),
                                    s2_val_passed["prob"].to_numpy(), target)


def veto_predict(s1_eval: pd.DataFrame, v_eval: pd.DataFrame, t1: float, t2: float) -> pd.DataFrame:
    """Exp-023 phase_c1b staged join: stage-2-missing tracks keep stage 1."""
    joined = s1_eval[["label", "group", "prob"]].rename(columns={"prob": "p1"}).join(
        v_eval[["prob"]].rename(columns={"prob": "p2"}), how="left")
    stage1 = (joined["p1"] >= t1).astype(int)
    stage2 = np.where(joined["p2"].notna(), (joined["p2"] >= t2).astype(int), 1)
    joined["pred"] = (stage1.to_numpy() & stage2).astype(int)
    joined["stage1_pred"] = stage1
    return joined


def pick_model_from_val(val_tables: dict[str, pd.DataFrame]) -> str:
    """Exp-011 pick_model rule on an inner-val canonical track table:
    best val-F1 at the val-F1 threshold; ties keep the earlier model in MODELS."""
    best_model, best_score = None, -1.0
    for m in MODELS:
        sub = val_tables[m]
        if sub.empty:
            continue
        y, p = sub["label"].to_numpy(), sub["prob"].to_numpy()
        t = s11.threshold_val_f1(y, p)
        score = f1_score(y, (p >= t).astype(int), zero_division=0)
        if score > best_score:
            best_score, best_model = score, m
    return best_model


# =========================================================================== data assembly
def load_shared():
    assign = s23.load_assignments()
    assign = assign.copy()
    assign["video_id"] = assign["track_key"].str.split(":").str[0]
    meta = s11.load_canonical_meta()
    meta = meta[meta["horizon_label"].isin(["30_60", "5_15"])].copy()

    prof, _attr_cols = s23.build_profile_frame(meta, assign)  # cached (exp-023 cache, read-only)
    X, xcols = s23.build_rule_features(meta, prof)            # frozen leak-audited feature set
    info = assign.set_index("track_key")[["split", "group"]]
    X = X.merge(info, left_on="track_key", right_index=True, how="inner")
    s23.assert_clean_features(xcols, "exp027_veto_features")

    window_data = {}
    for h, fs, tag in STAGE_CONFIGS:
        frames, columns, _rows = s23.build_window_feature_sets(h)
        df = frames[fs].copy()
        cols = columns[fs]
        s23.assert_clean_features(cols, f"exp027_windows:{h}|{fs}")
        df["track_key"] = df["video_id"].astype(str) + ":" + df["ped_id"].astype(str)
        df["video_str"] = df["video_id"].astype(str)
        canon = meta[meta["horizon_label"] == h][["sequence_id", "track_key", "label", "group"]]
        window_data[tag] = {"df": df.reset_index(drop=True), "cols": cols,
                            "canon": canon, "horizon": h, "feature_set": fs}
    return assign, meta, X, xcols, window_data


# =========================================================================== parity gate
def parity_gate(X: pd.DataFrame, xcols: list[str]) -> bool:
    """Reproduce exp-023's single-split C1b primary + generic staged baseline
    with the reused machinery; compare against exp-023's saved CSVs."""
    tt = s23.load_track_table()
    chosen = s23.chosen_models()
    s1_m = chosen[("30_60", "bbox_only")]
    s1_val = s11.get_split(tt, "30_60", "bbox_only", s1_m, "val")
    s1_test = s11.get_split(tt, "30_60", "bbox_only", s1_m, "test")

    # frozen veto on the original split
    tr_mask = (X["split"] == "train") & X["group"].isin(["actual_positive", "ambiguous_negative"])
    X_tr = X.loc[tr_mask, xcols]
    y_tr = (X.loc[tr_mask, "group"] == "actual_positive").astype(int).to_numpy()
    cands = {f"l1_logreg_C{C}": fit_l1_ensemble(X_tr, y_tr, X[xcols], X["track_key"].to_numpy(), C)
             for C in C_GRID}
    val_mask = (X["split"] == "val") & X["group"].isin(["actual_positive", "ambiguous_negative"])
    y_val = (X.loc[val_mask, "group"] == "actual_positive").astype(int).to_numpy()
    primary, sel = select_l1_C(cands, X.loc[val_mask, "track_key"].to_numpy(), y_val)
    log(f"parity: selected rule model {primary} "
        f"(AUCs: {sel.set_index('rule_model')['val_roc_auc_actual_vs_amb'].round(4).to_dict()})")

    xinfo = X.set_index("track_key")
    v = pd.DataFrame({"prob": cands[primary]})
    v["split"] = xinfo["split"]
    v_val, v_test = v[v["split"] == "val"], v[v["split"] == "test"]

    cr = pd.read_csv(CR23_PATH)
    saved_veto = cr[(cr["attempt"] == "C1b_interpretable_rule") & (cr["is_primary_val_selected"])]
    ops = pd.read_csv(OPS23_PATH)
    saved_base = ops[ops["config"] == "staged|bbox30_60->biu_v2_core@5_15"]

    s2_m = chosen[("5_15", "biu_v2_core")]
    s2_val = s11.get_split(tt, "5_15", "biu_v2_core", s2_m, "val")
    s2_test = s11.get_split(tt, "5_15", "biu_v2_core", s2_m, "test")

    rows = []
    for target in RECALL_TARGETS:
        t1 = s11.threshold_for_recall(s1_val["label"].to_numpy(), s1_val["prob"].to_numpy(), target)
        # --- veto pipeline
        t2v = veto_threshold(s1_val, v_val, t1, target)
        vj = veto_predict(s1_test, v_test, t1, t2v)
        met_v = s11.track_metrics(vj["label"].to_numpy(), vj["pred"].to_numpy(), vj["group"].to_numpy())
        sv = saved_veto[saved_veto["recall_target"] == target].iloc[0]
        ok_v = bool(np.isclose(met_v["f1"], sv["f1"]) and np.isclose(met_v["amb_neg_fpr"], sv["amb_neg_fpr"])
                    and np.isclose(met_v["recall"], sv["recall"]) and np.isclose(t2v, sv["t2"])
                    and str(sv["config"]).endswith(f"rule[{primary}]"))
        rows.append({"pipeline": "C1b_veto_primary", "recall_target": target,
                     "saved_config": sv["config"], "reproduced_rule_model": primary,
                     "saved_f1": sv["f1"], "new_f1": met_v["f1"],
                     "saved_amb_fpr": sv["amb_neg_fpr"], "new_amb_fpr": met_v["amb_neg_fpr"],
                     "saved_recall": sv["recall"], "new_recall": met_v["recall"],
                     "saved_t2": sv["t2"], "new_t2": t2v, "status": "OK" if ok_v else "MISMATCH"})
        # --- generic staged baseline
        t2b = baseline_t2(s1_val, s2_val, t1, target)
        bj = s11.staged_predict(s1_test, s2_test, t1, t2b)
        met_b = s11.track_metrics(bj["label"].to_numpy(), bj["pred"].to_numpy(), bj["group"].to_numpy())
        sb = saved_base[saved_base["recall_target"] == target].iloc[0]
        ok_b = bool(np.isclose(met_b["f1"], sb["f1"]) and np.isclose(met_b["amb_neg_fpr"], sb["amb_neg_fpr"])
                    and np.isclose(t2b, sb["t2"]))
        rows.append({"pipeline": "generic_staged_baseline", "recall_target": target,
                     "saved_config": sb["config"], "reproduced_rule_model": "",
                     "saved_f1": sb["f1"], "new_f1": met_b["f1"],
                     "saved_amb_fpr": sb["amb_neg_fpr"], "new_amb_fpr": met_b["amb_neg_fpr"],
                     "saved_recall": sb["recall"], "new_recall": met_b["recall"],
                     "saved_t2": sb["t2"], "new_t2": t2b, "status": "OK" if ok_b else "MISMATCH"})
    parity = pd.DataFrame(rows)
    parity.to_csv(METRICS_DIR / "parity_check.csv", index=False)
    n_bad = int((parity["status"] != "OK").sum())
    log(f"parity gate: {len(parity) - n_bad}/{len(parity)} rows OK")
    return n_bad == 0


# =========================================================================== fold scheme
def build_outer_scheme(assign: pd.DataFrame) -> pd.DataFrame:
    y = assign["group"].to_numpy()
    g = assign["video_id"].to_numpy()
    sgkf = StratifiedGroupKFold(n_splits=OUTER_K, shuffle=True, random_state=FOLD_SEED)
    fold = np.full(len(assign), -1, dtype=int)
    for k, (_tr, te) in enumerate(sgkf.split(np.zeros(len(assign)), y, groups=g)):
        fold[te] = k
    scheme = assign[["track_key", "video_id", "split", "group"]].copy()
    scheme["label"] = (scheme["group"] == "actual_positive").astype(int)
    scheme["fold"] = fold
    assert (scheme["fold"] >= 0).all()
    assert scheme.groupby("video_id")["fold"].nunique().max() == 1, "video split across folds"
    for k, sub in scheme.groupby("fold"):
        n_amb = int((sub["group"] == "ambiguous_negative").sum())
        n_pos = int(sub["label"].sum())
        n_neg = int((1 - sub["label"]).sum())
        log(f"outer fold {k}: n={len(sub)} videos={sub['video_id'].nunique()} "
            f"pos={n_pos} amb={n_amb} clean={int((sub['group'] == 'clean_negative').sum())}")
        assert n_amb >= 2 and n_pos >= 1 and n_neg >= 1, f"fold {k} violates constraints"
    scheme.to_csv(METRICS_DIR / "outer_fold_scheme.csv", index=False)
    return scheme


def inner_split_videos(outer_train: pd.DataFrame) -> tuple[set, set, int, str]:
    """First valid ~80/20 video-disjoint stratified candidate split (seed 42)."""
    y = outer_train["group"].to_numpy()
    g = outer_train["video_id"].to_numpy()
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=FOLD_SEED)
    splits = list(sgkf.split(np.zeros(len(outer_train)), y, groups=g))
    for rule, (min_amb, min_pos) in [("strict>=2", (2, 2)), ("relaxed>=1", (1, 1))]:
        for i, (tr_idx, va_idx) in enumerate(splits):
            va = outer_train.iloc[va_idx]
            tr = outer_train.iloc[tr_idx]
            if ((va["group"] == "ambiguous_negative").sum() >= min_amb
                    and va["label"].sum() >= min_pos and (1 - va["label"]).sum() >= 1
                    and (tr["group"] == "ambiguous_negative").sum() >= 1 and tr["label"].sum() >= 1):
                return set(tr["video_id"]), set(va["video_id"]), i, rule
    raise RuntimeError("no valid inner split found")


# =========================================================================== nested evaluation
def fold_stage_track_probs(tag: str, wd: dict, inner_train_videos: set,
                           universe_videos: set, fold_k: int) -> pd.DataFrame:
    """Canonical track-level seed-ensemble probs (all tracks of the horizon) for
    the three candidate models, fit on inner-train windows only. Cached."""
    cache = CACHE_DIR / f"fold{fold_k}_{tag}_track_probs.csv"
    if cache.exists():
        return pd.read_csv(cache)
    df, cols, canon = wd["df"], wd["cols"], wd["canon"]
    trainable = df["video_str"].isin(inner_train_videos) | ~df["video_str"].isin(universe_videos)
    tr = df[trainable]
    y_tr = tr["label"].astype(int).to_numpy()
    out = canon.copy()
    for m in MODELS:
        seed_probs = []
        for seed in SEEDS:
            pipe = base._build_model(m, seed, y_tr)
            pipe.fit(tr[cols].astype(float), y_tr)
            seed_probs.append(pipe.predict_proba(df[cols].astype(float))[:, 1])
        ens = pd.Series(np.mean(seed_probs, axis=0), index=df["sequence_id"].to_numpy())
        out[f"prob_{m}"] = out["sequence_id"].map(ens)
    out.to_csv(cache, index=False)
    return out


def run_nested(assign: pd.DataFrame, scheme: pd.DataFrame, X: pd.DataFrame,
               xcols: list[str], window_data: dict):
    universe_videos = set(assign["video_id"])
    fold_of_track = scheme.set_index("track_key")["fold"]
    video_fold = scheme.drop_duplicates("video_id").set_index("video_id")["fold"]

    pooled_rows, fold_result_rows, sel_rows, coef_rows = [], [], [], []

    for k in range(OUTER_K):
        t0 = time.time()
        outer_train = scheme[scheme["fold"] != k]
        fold_tracks = scheme.loc[scheme["fold"] == k, "track_key"]
        in_tr_videos, in_val_videos, cand_i, cand_rule = inner_split_videos(outer_train)
        log(f"fold {k}: inner split candidate {cand_i} ({cand_rule}); "
            f"{len(in_tr_videos)} train videos / {len(in_val_videos)} val videos")

        # ---- stage models (window-level, inner-train windows only)
        stage = {}
        for h, fs, tag in STAGE_CONFIGS:
            probs = fold_stage_track_probs(tag, window_data[tag], in_tr_videos, universe_videos, k)
            probs["video_str"] = probs["track_key"].str.split(":").str[0]
            val_tables = {m: probs[probs["video_str"].isin(in_val_videos)]
                          .rename(columns={f"prob_{m}": "prob"})[["track_key", "label", "group", "prob"]]
                          .dropna(subset=["prob"]) for m in MODELS}
            m_star = pick_model_from_val(val_tables)
            chosen_tab = probs.rename(columns={f"prob_{m_star}": "prob"})
            chosen_tab = chosen_tab[["track_key", "video_str", "label", "group", "prob"]].dropna(subset=["prob"])
            stage[tag] = {"model": m_star,
                          "val": chosen_tab[chosen_tab["video_str"].isin(in_val_videos)].set_index("track_key"),
                          "fold": chosen_tab[chosen_tab["track_key"].isin(set(fold_tracks))].set_index("track_key")}
            log(f"fold {k}: {tag} ({h}|{fs}) chosen model = {m_star}")

        s1_val, s1_fold = stage["s1"]["val"], stage["s1"]["fold"]
        s2_val, s2_fold = stage["s2"]["val"], stage["s2"]["fold"]

        # ---- veto rule (track-level, frozen features)
        Xf = X.merge(scheme[["track_key", "fold", "video_id"]], on="track_key", how="inner")
        vt_tr = Xf["video_id"].isin(in_tr_videos) & Xf["group"].isin(["actual_positive", "ambiguous_negative"])
        X_tr = Xf.loc[vt_tr, xcols]
        y_tr = (Xf.loc[vt_tr, "group"] == "actual_positive").astype(int).to_numpy()
        cands = {f"l1_logreg_C{C}": fit_l1_ensemble(X_tr, y_tr, Xf[xcols], Xf["track_key"].to_numpy(), C)
                 for C in C_GRID}
        vv_mask = Xf["video_id"].isin(in_val_videos) & Xf["group"].isin(["actual_positive", "ambiguous_negative"])
        y_vval = (Xf.loc[vv_mask, "group"] == "actual_positive").astype(int).to_numpy()
        primary, sel = select_l1_C(cands, Xf.loc[vv_mask, "track_key"].to_numpy(), y_vval)
        chosen_C = float(primary.split("C")[-1])
        log(f"fold {k}: veto chosen {primary} "
            f"(AUCs {sel.set_index('rule_model')['val_roc_auc_actual_vs_amb'].round(4).to_dict()})")

        veto_probs = cands[primary]
        v_all = pd.DataFrame({"prob": veto_probs})
        v_all["video_str"] = pd.Series(v_all.index, index=v_all.index).str.split(":").str[0]
        v_val = v_all[v_all["video_str"].isin(in_val_videos)]
        v_fold = v_all[v_all.index.isin(set(fold_tracks))]

        # ---- L1 coefficients (seed-42 refit; chosen C + C=0.1 for 023 comparison)
        for c_tag, c_val in {f"chosen_C{chosen_C:g}": chosen_C, "C0.1": 0.1}.items():
            pipe = l1_pipeline(c_val, 42)
            pipe.fit(X_tr, y_tr)
            coefs = pipe.named_steps["model"].coef_[0]
            for feat, cf in zip(xcols, coefs):
                if cf != 0.0:
                    coef_rows.append({"fold": k, "C_setting": c_tag, "C": c_val,
                                      "feature": feat, "l1_coef_seed42": float(cf)})

        # ---- thresholds + outer-fold predictions per target
        sel_row = {"fold": k, "inner_candidate_index": cand_i, "inner_rule": cand_rule,
                   "n_inner_val_tracks": int((outer_train["video_id"].isin(in_val_videos)).sum()),
                   "n_inner_val_pos": int(outer_train.loc[outer_train["video_id"].isin(in_val_videos), "label"].sum()),
                   "n_inner_val_amb": int((outer_train.loc[outer_train["video_id"].isin(in_val_videos), "group"]
                                           == "ambiguous_negative").sum()),
                   "s1_model": stage["s1"]["model"], "s2_baseline_model": stage["s2"]["model"],
                   "veto_rule_model": primary}
        for name, auc in sel.set_index("rule_model")["val_roc_auc_actual_vs_amb"].items():
            sel_row[f"val_auc_{name}"] = auc

        for target in RECALL_TARGETS:
            t1 = s11.threshold_for_recall(s1_val["label"].to_numpy(), s1_val["prob"].to_numpy(), target)
            t2b = baseline_t2(s1_val, s2_val, t1, target)
            t2v = veto_threshold(s1_val, v_val, t1, target)
            bj = s11.staged_predict(s1_fold, s2_fold, t1, t2b)
            vj = veto_predict(s1_fold, v_fold, t1, t2v)
            assert bj.index.equals(vj.index)
            tk = f"{target:.2f}".replace(".", "")
            sel_row[f"t1_r{tk}"] = t1
            sel_row[f"baseline_t2_r{tk}"] = t2b
            sel_row[f"veto_t2_r{tk}"] = t2v

            met_b = s11.track_metrics(bj["label"].to_numpy(), bj["pred"].to_numpy(), bj["group"].to_numpy())
            met_v = s11.track_metrics(vj["label"].to_numpy(), vj["pred"].to_numpy(), vj["group"].to_numpy())
            fold_result_rows.append({"fold": k, "recall_target": target, "pipeline": BASELINE_NAME,
                                     "n_stage2_missing_kept": int((bj["stage1_pred"] == 1)[bj["p2"].isna()].sum()),
                                     **met_b})
            fold_result_rows.append({"fold": k, "recall_target": target, "pipeline": VETO_NAME,
                                     "n_stage2_missing_kept": int((vj["stage1_pred"] == 1)[vj["p2"].isna()].sum()),
                                     **met_v})

            merged = pd.DataFrame({
                "track_key": bj.index, "fold": k, "recall_target": target,
                "orig_split": bj.index.map(assign.set_index("track_key")["split"]),
                "label": bj["label"].to_numpy(), "group": bj["group"].to_numpy(),
                "p1": bj["p1"].to_numpy(), "t1": t1, "stage1_pred": bj["stage1_pred"].to_numpy(),
                "baseline_p2": bj["p2"].to_numpy(), "baseline_t2": t2b,
                "baseline_pred": bj["pred"].to_numpy(),
                "veto_p2": vj["p2"].to_numpy(), "veto_t2": t2v, "veto_pred": vj["pred"].to_numpy(),
            })
            pooled_rows.append(merged)
        sel_rows.append(sel_row)
        log(f"fold {k}: done in {time.time() - t0:.1f}s "
            f"({len(fold_tracks)} outer tracks evaluated)")

    pooled = pd.concat(pooled_rows, ignore_index=True)
    return pooled, pd.DataFrame(fold_result_rows), pd.DataFrame(sel_rows), pd.DataFrame(coef_rows)


# =========================================================================== endpoints
def compute_endpoints(pooled: pd.DataFrame):
    ep_rows, boot_rows = [], []
    for target in RECALL_TARGETS:
        sub = pooled[pooled["recall_target"] == target]
        assert sub["track_key"].is_unique
        y = sub["label"].to_numpy()
        g = sub["group"].to_numpy()
        pb = sub["baseline_pred"].to_numpy()
        pv = sub["veto_pred"].to_numpy()
        met_b = s11.track_metrics(y, pb, g)
        met_v = s11.track_metrics(y, pv, g)
        ep_rows.append({"recall_target": target, "pipeline": BASELINE_NAME, **met_b})
        ep_rows.append({"recall_target": target, "pipeline": VETO_NAME, **met_v})
        boot = s11.bootstrap_delta(y, g, pb, pv)  # delta = veto - baseline
        boot_rows.append({"recall_target": target,
                          "endpoint": "primary" if target == 0.90 else "secondary",
                          "n_tracks": len(sub),
                          "baseline_f1": met_b["f1"], "veto_f1": met_v["f1"],
                          "point_delta_f1": met_v["f1"] - met_b["f1"],
                          "baseline_amb_fpr": met_b["amb_neg_fpr"], "veto_amb_fpr": met_v["amb_neg_fpr"],
                          "point_delta_amb_fpr": met_v["amb_neg_fpr"] - met_b["amb_neg_fpr"],
                          "n_boot": s11.N_BOOT, "boot_seed": s11.BOOT_SEED, **boot})
    return pd.DataFrame(ep_rows), pd.DataFrame(boot_rows)


# =========================================================================== main
def main() -> None:
    for d in (METRICS_DIR, LOGS_DIR, CACHE_DIR):
        d.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text("", encoding="utf-8")
    t_start = time.time()
    log("=== exp 027: frozen veto pipeline under nested video-disjoint CV ===")

    assign, meta, X, xcols, window_data = load_shared()
    log(f"pool: {len(assign)} tracks / {assign['video_id'].nunique()} videos; "
        f"veto feature matrix {X.shape[0]} tracks x {len(xcols)} features")

    log("=== parity gate (frozen machinery vs exp-023 saved numbers) ===")
    if not parity_gate(X, xcols):
        log("PARITY FAILED - stopping before the nested evaluation (see metrics/parity_check.csv)")
        sys.exit(1)
    log("parity gate PASSED")

    log("=== outer fold scheme ===")
    scheme = build_outer_scheme(assign)

    log("=== nested evaluation ===")
    pooled, per_fold, sel_log, coefs = run_nested(assign, scheme, X, xcols, window_data)
    pooled.to_csv(METRICS_DIR / "pooled_track_predictions.csv", index=False)
    per_fold.to_csv(METRICS_DIR / "per_fold_results.csv", index=False)
    sel_log.to_csv(METRICS_DIR / "inner_selection_log.csv", index=False)
    coefs.to_csv(METRICS_DIR / "l1_coefficients_per_fold.csv", index=False)

    log("=== pooled endpoints + bootstrap ===")
    endpoints, boots = compute_endpoints(pooled)
    endpoints.to_csv(METRICS_DIR / "pooled_endpoints.csv", index=False)
    boots.to_csv(METRICS_DIR / "bootstrap_results.csv", index=False)
    for _, r in boots.iterrows():
        log(f"[{r['endpoint']}] recall target {r['recall_target']}: "
            f"delta F1 {r['point_delta_f1']:+.4f} "
            f"[{r['delta_f1_ci_low']:+.4f}, {r['delta_f1_ci_high']:+.4f}] p={r['delta_f1_p']:.4f}; "
            f"delta amb-FPR {r['point_delta_amb_fpr']:+.4f} "
            f"[{r['delta_amb_fpr_ci_low']:+.4f}, {r['delta_amb_fpr_ci_high']:+.4f}] p={r['delta_amb_fpr_p']:.4f}")

    summary = {
        "parity_status": "PASSED",
        "n_tracks_pool": int(len(assign)),
        "n_videos_pool": int(assign["video_id"].nunique()),
        "outer_k": OUTER_K,
        "fold_seed": FOLD_SEED,
        "seeds": SEEDS,
        "boot": {"n": s11.N_BOOT, "seed": s11.BOOT_SEED},
        "per_fold_choices": sel_log[["fold", "s1_model", "s2_baseline_model", "veto_rule_model"]]
        .to_dict(orient="records"),
        "endpoints": boots.to_dict(orient="records"),
        "runtime_seconds": round(time.time() - t_start, 1),
    }
    (METRICS_DIR / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"done in {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
