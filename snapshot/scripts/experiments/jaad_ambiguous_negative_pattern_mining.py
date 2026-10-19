"""Experiment 023: JAAD ambiguous-negative pattern mining.

Research question (thesis author): "there must be some pattern or combination of
patterns that separates ambiguous negatives - find it, or prove how much is
irreducible." Ambiguous negatives (crossing_attr 0, exp-004 authoritative
track-level assignment: 77/32/42 pos/amb/clean on test) are the dominant JAAD
false-positive source.

Analyses:
A. Persistence map. Track-level FP matrix: ambiguous-negative track x
   configuration (5 feature sets x 3 models x 3 horizons; 30_60 + 5_15 from the
   exp-011 saved seed-ensemble track table, 15_30 from the exp-021 CORRECTED
   table - the exp-011 15_30 folder is a known artifact and is NOT used).
   Thresholds: validation-F1 per configuration (validation-only selection).
   Test/val tracks use the 45 saved configurations. Train tracks have no saved
   predictions, so honest train probabilities are regenerated OUT-OF-FOLD
   (StratifiedGroupKFold by track, 5 folds, split seed 42; models re-fit with
   the exp-009 machinery, seeds 42/43/44, seed-ensemble mean) for the 15
   validation-selected (horizon, feature_set) configurations only.
   Each ambiguous track is classified: NEVER_FP / ALWAYS_FP / MOSTLY_FP (>=80%)
   / RESCUED_BY_POSE (+WAITING variants) / MIXED.
B. Group profiling on interpretable per-track attributes (context no-motion,
   bbox geometry, BIU v2 cues, skeleton confidence, observation-window JAAD
   behavior annotations, track length / TTE), with rank-biserial + Cohen's d +
   Mann-Whitney p (exploratory; BH-FDR q reported).
C. Combo search (train -> val selection -> ONE test evaluation):
   C1a stage-2 retrain with context: biu_v2_core_plus_bbox + context_no_motion
       at 5_15 and corrected 15_30, staged behind exp-011's bbox@30_60 stage 1,
       matched-recall thresholds (0.90/0.95), track bootstrap (10k, seed 12345)
       vs the generic staged baseline.
   C1b interpretable rule stage 2: depth-<=3 decision tree and L1-logreg on
       actual_positive-vs-ambiguous_negative track discrimination using ONLY
       leak-audited features (context no-motion + cross-horizon bbox deltas +
       5_15 BIU v2 statics + skeleton confidence).
   C2 oracle bounds: ceiling amb-FPR / F1 if a candidate pattern perfectly
       identified its ambiguous subgroup.
D. Dossier: one row per ambiguous-negative track (all splits) for hand-tagging.

Hard rules honoured:
- motion_direction / attr_motion_lateral and any label/future-derived column
  are NEVER used as features (assert_clean_features guards every matrix,
  including after every merge).
- No test-set selection. Test is evaluated once per reported configuration.
- Track bootstrap: 10k resamples, seed 12345 (exp-011 bootstrap_delta).
- JAAD per-frame behavior annotations are aggregated over OBSERVED window
  frames only and used for PROFILING/dossier only, never as model features.

Rerun:
cd /home/abdelrahman/Hadwa/BIU-PID-Distillation-clean && \
  MPLCONFIGDIR=/tmp/mplconfig python3 scripts/experiments/jaad_ambiguous_negative_pattern_mining.py

Outputs: experiments/023_jaad_ambiguous_negative_pattern_mining/{metrics,predictions,cache}
"""

from __future__ import annotations

import json
import os
import sys
import xml.etree.ElementTree as ET
from itertools import product
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

CLEAN_ROOT = Path("/home/abdelrahman/Hadwa/BIU-PID-Distillation-clean")
SCRIPTS_DIR = CLEAN_ROOT / "scripts/experiments"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import jaad_biu_v2_benchmark_comparison as base  # noqa: E402  (exp-009 machinery)
import jaad_staged_bbox_biu_v2 as s11  # noqa: E402  (exp-011 machinery)
import jaad_corrected_15_30_rebuild_benchmark as s21b  # noqa: E402  (exp-021 corrected tables)
import jaad_corrected_15_30_rebuild_staged as s21  # noqa: E402  (exp-021 canonical meta)
import jaad_context_features as ctx  # noqa: E402  (exp-016 context machinery)

OUT_ROOT = CLEAN_ROOT / "experiments/023_jaad_ambiguous_negative_pattern_mining"
METRICS_DIR = OUT_ROOT / "metrics"
PRED_DIR = OUT_ROOT / "predictions"
CACHE_DIR = OUT_ROOT / "cache"

TT11_PATH = CLEAN_ROOT / "experiments/011_jaad_staged_bbox_biu_v2/metrics/staged_track_table.csv"
TT21_PATH = CLEAN_ROOT / "experiments/021_jaad_corrected_15_30_rebuild/metrics/corrected_staged_track_table.csv"
SEL11_PATH = CLEAN_ROOT / "experiments/011_jaad_staged_bbox_biu_v2/metrics/staged_model_selection.csv"
SEL21_PATH = CLEAN_ROOT / "experiments/021_jaad_corrected_15_30_rebuild/metrics/corrected_staged_model_selection.csv"
OP11_PATH = CLEAN_ROOT / "experiments/011_jaad_staged_bbox_biu_v2/metrics/staged_operating_points.csv"
OP21_PATH = CLEAN_ROOT / "experiments/021_jaad_corrected_15_30_rebuild/metrics/corrected_staged_operating_points.csv"
CTX_TABLE_PATH = CLEAN_ROOT / "experiments/016_jaad_context_features/metrics/jaad_context_feature_table.csv"
ASSIGN_PATH = s11.GROUP_ASSIGNMENTS
JAAD_ANN_DIR = Path("/home/abdelrahman/Hadwa/datasets/JAAD/annotations")

HORIZONS_011 = ["30_60", "5_15"]
H_COR = "15_30_corrected"
ALL_HORIZONS = HORIZONS_011 + [H_COR]
FEATURE_SETS = ["bbox_only", "biu_v1_only", "biu_v1_plus_bbox", "biu_v2_core", "biu_v2_core_plus_bbox"]
POSE_SETS = {"biu_v2_core", "biu_v2_core_plus_bbox"}
MODELS = ["logreg", "rf", "xgb"]
SEEDS = [42, 43, 44]
RECALL_TARGETS = [0.90, 0.95]
OOF_SPLIT_SEED = 42
N_OOF_FOLDS = 5

# Scene-infrastructure columns whose NAME contains "cross" but which are NOT
# label-derived (exp-016 leakage audit).
SCENE_ALLOWLIST = {"attr_designated_crossing", "traffic_ped_crossing_frac"}
FORBIDDEN_EXACT = {"label", "cross", "crossing", "crossing_attr", "crossing_point",
                   "decision_point", "intention", "group", "split"}
FORBIDDEN_TOKENS = ["motion_direction", "attr_motion", "crossing_point", "decision_point",
                    "intention", "event_frame"]


def assert_clean_features(cols, where: str) -> None:
    """Hard leak guard: no label-derived / future-derived / meta column may be a feature."""
    for c in cols:
        lc = str(c).lower()
        assert lc not in FORBIDDEN_EXACT, f"[{where}] forbidden feature column: {c}"
        for tok in FORBIDDEN_TOKENS:
            assert tok not in lc, f"[{where}] forbidden token '{tok}' in feature column: {c}"
        if "cross" in lc:
            assert c in SCENE_ALLOWLIST, f"[{where}] suspicious 'cross' column not allowlisted: {c}"
        assert not lc.startswith("beh_"), f"[{where}] behavior annotation used as model feature: {c}"
        assert "label" not in lc, f"[{where}] label-like feature column: {c}"


# =========================================================================== shared loading
def load_track_table() -> pd.DataFrame:
    """Saved seed-ensemble canonical track tables: 30_60 + 5_15 (exp-011) and
    corrected 15_30 (exp-021). The exp-011 15_30 rows (artifact table) are dropped."""
    tt11 = pd.read_csv(TT11_PATH)
    tt11 = tt11[tt11["horizon_label"].isin(HORIZONS_011)].copy()
    tt21 = pd.read_csv(TT21_PATH)
    tt21 = tt21[tt21["horizon_label"] == H_COR].copy()
    tt = pd.concat([tt11, tt21], ignore_index=True)
    tt["config_id"] = tt["horizon_label"] + "|" + tt["feature_set"] + "|" + tt["model_name"]
    return tt


def load_assignments() -> pd.DataFrame:
    return pd.read_csv(ASSIGN_PATH)


def chosen_models() -> dict[tuple[str, str], str]:
    out = {}
    sel11 = pd.read_csv(SEL11_PATH)
    for _, r in sel11.iterrows():
        if r["horizon_label"] in HORIZONS_011:
            out[(r["horizon_label"], r["feature_set"])] = r["chosen_model"]
    sel21 = pd.read_csv(SEL21_PATH)
    for _, r in sel21.iterrows():
        if r["horizon_label"] == H_COR:
            out[(H_COR, r["feature_set"])] = r["chosen_model"]
    return out


def canonical_meta_all_splits() -> pd.DataFrame:
    """Canonical windows (min tte, then latest start; exp-004 rule) for the three
    honest horizons, ALL splits (train included)."""
    meta11 = s11.load_canonical_meta()
    meta11 = meta11[meta11["horizon_label"].isin(HORIZONS_011)]
    metac = s21.load_canonical_meta_corrected()
    return pd.concat([meta11, metac], ignore_index=True)


# =========================================================================== phase A1: val/test FP matrix
def phase_a1(tt: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-config val-F1 thresholds + val/test track predictions for all 45 configs."""
    pred_rows = []
    thr_rows = []
    for cfg_id, sub in tt.groupby("config_id"):
        val = sub[sub["split"] == "val"]
        t = s11.threshold_val_f1(val["label"].to_numpy(), val["prob"].to_numpy())
        thr_rows.append({"config_id": cfg_id, "threshold_rule": "val_f1", "threshold": t,
                         "n_val_tracks": len(val)})
        ev = sub[sub["split"].isin(["val", "test"])].copy()
        ev["pred"] = (ev["prob"] >= t).astype(int)
        ev["threshold"] = t
        pred_rows.append(ev[["track_key", "split", "label", "group", "config_id",
                             "horizon_label", "feature_set", "model_name", "prob", "threshold", "pred"]])
    preds = pd.concat(pred_rows, ignore_index=True)
    thrs = pd.DataFrame(thr_rows)
    return preds, thrs


# =========================================================================== staged baselines (parity-checked)
def recompute_staged_baselines(tt: pd.DataFrame, chosen: dict) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Recompute the exp-011/exp-021 staged predictions per track (needed for
    bootstrap/oracle) and parity-check the resulting operating points against the
    archived CSVs (rescore-parity convention)."""
    s1_m = chosen[("30_60", "bbox_only")]
    s1_val = s11.get_split(tt, "30_60", "bbox_only", s1_m, "val")
    s1_test = s11.get_split(tt, "30_60", "bbox_only", s1_m, "test")

    staged_preds: dict[tuple[str, str, float], pd.DataFrame] = {}
    op_rows = []
    for s2_h in ["5_15", H_COR]:
        for s2_fs in ["biu_v2_core", "biu_v2_core_plus_bbox"]:
            s2_m = chosen[(s2_h, s2_fs)]
            s2_val = s11.get_split(tt, s2_h, s2_fs, s2_m, "val")
            s2_test = s11.get_split(tt, s2_h, s2_fs, s2_m, "test")
            for target in RECALL_TARGETS:
                t1 = s11.threshold_for_recall(s1_val["label"].to_numpy(), s1_val["prob"].to_numpy(), target)
                passed = s1_val[s1_val["prob"] >= t1]
                s2_val_passed = s2_val.loc[s2_val.index.intersection(passed.index)]
                if s2_val_passed.empty or s2_val_passed["label"].nunique() < 2:
                    t2 = 0.5
                else:
                    t2 = s11.threshold_for_recall(s2_val_passed["label"].to_numpy(),
                                                  s2_val_passed["prob"].to_numpy(), target)
                joined = s11.staged_predict(s1_test, s2_test, t1, t2)
                staged_preds[(s2_h, s2_fs, target)] = joined
                op_rows.append({"config": f"staged|bbox30_60->{s2_fs}@{s2_h}", "recall_target": target,
                                "t1": t1, "t2": t2,
                                **s11.track_metrics(joined["label"].to_numpy(), joined["pred"].to_numpy(),
                                                    joined["group"].to_numpy())})
    ops = pd.DataFrame(op_rows)

    # parity check vs archived operating points
    parity_rows = []
    old11 = pd.read_csv(OP11_PATH)
    old21 = pd.read_csv(OP21_PATH)
    for _, r in ops.iterrows():
        cfg = r["config"]
        # exp-011 named the 5_15 configs "...@5_15", exp-021 "...@15_30_corrected"
        old = old21 if cfg.endswith(H_COR) else old11
        m = old[(old["config"] == cfg) & (old["recall_target"] == r["recall_target"])]
        if m.empty:
            parity_rows.append({"config": cfg, "recall_target": r["recall_target"],
                                "status": "MISSING_IN_ARCHIVE"})
            continue
        o = m.iloc[0]
        ok = bool(np.isclose(o["f1"], r["f1"]) and np.isclose(o["amb_neg_fpr"], r["amb_neg_fpr"])
                  and np.isclose(o["recall"], r["recall"]))
        parity_rows.append({"config": cfg, "recall_target": r["recall_target"],
                            "old_f1": o["f1"], "new_f1": r["f1"],
                            "old_amb_fpr": o["amb_neg_fpr"], "new_amb_fpr": r["amb_neg_fpr"],
                            "status": "OK" if ok else "MISMATCH"})
    parity = pd.DataFrame(parity_rows)
    n_bad = int((parity["status"] != "OK").sum())
    if n_bad:
        print(f"WARNING: {n_bad} staged parity rows are not OK - inspect staged_baseline_parity_check.csv")
    return staged_preds, ops, parity


# =========================================================================== window-level feature data
_KP_INDEX_CACHE: dict | None = None


def get_split_rows(h: str) -> pd.DataFrame:
    if h == H_COR:
        return s21b.load_base_rows()
    return base._load_split_rows(h)


def get_kp_index() -> dict:
    """Keypoint index covering ALL windows of both archived horizons (30_60 + 5_15),
    built once. (Loading per-horizon would silently miss the other horizon's frames.)"""
    global _KP_INDEX_CACHE
    if _KP_INDEX_CACHE is None:
        rows_by_h = {h: base._load_split_rows(h) for h in HORIZONS_011}
        needed = base._needed_frame_keys(rows_by_h)
        _KP_INDEX_CACHE = base._load_keypoint_index(
            base.HORIZON_CONFIGS["30_60"]["keypoints_jsonl"], needed)
    return _KP_INDEX_CACHE


def get_v2_windows(h: str, split_rows: pd.DataFrame) -> pd.DataFrame:
    """BIU v2 features (all groups incl. confidence) for ALL windows of a horizon; cached."""
    cache = CACHE_DIR / f"v2_windows_{h}.csv"
    if cache.exists():
        return pd.read_csv(cache)
    if h == H_COR:
        v2 = s21b.load_feature_frame("biu_v2")
        v2.to_csv(cache, index=False)
        return v2
    kp = get_kp_index()
    v2, missing = base._extract_v2_for_rows(split_rows, kp)
    if missing:
        pd.DataFrame(missing).to_csv(METRICS_DIR / f"v2_missing_keypoint_rows_{h}.csv", index=False)
    v2.to_csv(cache, index=False)
    return v2


def build_window_feature_sets(h: str) -> tuple[dict[str, pd.DataFrame], dict[str, list[str]], pd.DataFrame]:
    """Window-level frames + column lists for the 5 benchmark feature sets, one horizon."""
    rows = get_split_rows(h)
    if h == H_COR:
        frames, columns = s21b.build_feature_sets(rows)
        return frames, columns, rows
    feature_dict = pd.read_csv(base.DICT_PATH)
    v2 = get_v2_windows(h, rows)
    data_by_h, _defs, _cov = base._build_feature_sets({h: rows}, {h: v2}, feature_dict)
    fmap = data_by_h[h]
    columns = {fs: fmap["_column_map"][fs] for fs in FEATURE_SETS}
    frames = {fs: fmap[fs] for fs in FEATURE_SETS}
    return frames, columns, rows


# =========================================================================== phase A2: train OOF
def phase_a2_train_oof(meta: pd.DataFrame, thrs: pd.DataFrame, chosen: dict) -> pd.DataFrame:
    """Honest train-split track predictions: StratifiedGroupKFold(5) OOF window
    probabilities (grouped by track to avoid overlapping-window leakage), seeds
    42/43/44 seed-ensemble, canonical-window selection, config's val-F1 threshold."""
    cache = CACHE_DIR / "train_oof_track_preds.csv"
    if cache.exists():
        return pd.read_csv(cache)

    out_rows = []
    for h in ALL_HORIZONS:
        frames, columns, rows = build_window_feature_sets(h)
        canon = meta[(meta["horizon_label"] == h) & (meta["split"] == "train")]
        canon_ids = canon.set_index("sequence_id")["track_key"]
        for fs in FEATURE_SETS:
            m = chosen[(h, fs)]
            cfg_id = f"{h}|{fs}|{m}"
            t = thrs.loc[thrs["config_id"] == cfg_id, "threshold"].item()
            df = frames[fs]
            cols = columns[fs]
            assert_clean_features(cols, f"train_oof:{cfg_id}")
            tr = df[df["split"] == "train"].reset_index(drop=True)
            X = tr[cols].astype(float).to_numpy()
            y = tr["label"].astype(int).to_numpy()
            grp = (tr["video_id"].astype(str) + ":" + tr["ped_id"].astype(str)).to_numpy()
            sgkf = StratifiedGroupKFold(n_splits=N_OOF_FOLDS, shuffle=True, random_state=OOF_SPLIT_SEED)
            probs = np.zeros((len(SEEDS), len(tr)), dtype=float)
            probs[:] = np.nan
            for tr_idx, te_idx in sgkf.split(X, y, groups=grp):
                if len(np.unique(y[tr_idx])) < 2:
                    continue
                for si, seed in enumerate(SEEDS):
                    pipe = base._build_model(m, seed, y[tr_idx])
                    pipe.fit(pd.DataFrame(X[tr_idx], columns=cols), y[tr_idx])
                    probs[si, te_idx] = pipe.predict_proba(pd.DataFrame(X[te_idx], columns=cols))[:, 1]
            ens = np.nanmean(probs, axis=0)
            tr = tr.assign(oof_prob=ens)
            tr["track_key"] = tr["sequence_id"].map(canon_ids)
            canon_tr = tr[tr["track_key"].notna()].copy()  # canonical windows only
            canon_tr["config_id"] = cfg_id
            canon_tr["horizon_label"] = h
            canon_tr["feature_set"] = fs
            canon_tr["model_name"] = m
            canon_tr["threshold"] = t
            canon_tr["pred"] = (canon_tr["oof_prob"] >= t).astype(int)
            grp_map = canon.set_index("track_key")["group"]
            canon_tr["group"] = canon_tr["track_key"].map(grp_map)
            out_rows.append(canon_tr[["track_key", "label", "group", "config_id", "horizon_label",
                                      "feature_set", "model_name", "oof_prob", "threshold", "pred"]])
            print(f"  OOF done: {cfg_id} ({len(canon_tr)} canonical train tracks)")
    oof = pd.concat(out_rows, ignore_index=True)
    oof["split"] = "train"
    oof = oof.rename(columns={"oof_prob": "prob"})
    oof.to_csv(cache, index=False)
    return oof


# =========================================================================== phase A3: classification
def classify_tracks(fp_long: pd.DataFrame, basis: str) -> pd.DataFrame:
    """fp_long: one row per (amb track, config) with pred (1 = FP). Returns per-track rates + category."""
    rows = []
    for tk, sub in fp_long.groupby("track_key"):
        n = len(sub)
        fp_rate = sub["pred"].mean()

        def rate(mask) -> float:
            s = sub[mask]
            return float(s["pred"].mean()) if len(s) else float("nan")

        r_bbox = rate(sub["feature_set"] == "bbox_only")
        r_pose = rate(sub["feature_set"].isin(POSE_SETS))
        r_3060 = rate(sub["horizon_label"] == "30_60")
        r_1530 = rate(sub["horizon_label"] == H_COR)
        r_515 = rate(sub["horizon_label"] == "5_15")
        r_bbox3060 = rate((sub["horizon_label"] == "30_60") & (sub["feature_set"] == "bbox_only"))

        fooled_bbox = (not np.isnan(r_bbox)) and r_bbox >= 0.5
        pose_ok = (not np.isnan(r_pose)) and r_pose < 0.5
        fooled_3060 = (not np.isnan(r_3060)) and r_3060 >= 0.5
        wait_ok = (not np.isnan(r_515)) and r_515 < 0.5

        if fp_rate == 0.0:
            cat = "NEVER_FP"
        elif fp_rate == 1.0:
            cat = "ALWAYS_FP"
        elif fp_rate >= 0.8:
            cat = "MOSTLY_FP"
        elif fooled_bbox and pose_ok and fooled_3060 and wait_ok:
            cat = "RESCUED_BY_POSE_AND_WAITING"
        elif fooled_bbox and pose_ok:
            cat = "RESCUED_BY_POSE"
        elif fooled_3060 and wait_ok:
            cat = "RESCUED_BY_WAITING"
        else:
            cat = "MIXED"
        rows.append({"track_key": tk, "split": sub["split"].iloc[0], "basis": basis,
                     "n_configs": n, "n_fp": int(sub["pred"].sum()), "fp_rate": float(fp_rate),
                     "fp_rate_bbox_family": r_bbox, "fp_rate_pose_family": r_pose,
                     "fp_rate_30_60": r_3060, "fp_rate_15_30_corrected": r_1530, "fp_rate_5_15": r_515,
                     "fp_rate_bbox_30_60": r_bbox3060, "category": cat})
    return pd.DataFrame(rows)


def phase_a3(preds: pd.DataFrame, oof: pd.DataFrame, assign: pd.DataFrame,
             chosen: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    amb = assign[assign["group"] == "ambiguous_negative"][["track_key", "split"]]
    amb_keys = set(amb["track_key"])

    test_val = preds[preds["track_key"].isin(amb_keys) & (preds["split"] == preds["track_key"].map(
        amb.set_index("track_key")["split"]))].copy()

    # primary classification: test/val on all 45 configs
    cls_full = classify_tracks(test_val, basis="45_saved_configs_val_f1_thresholds")

    # secondary: the 15 val-selected configs for every split (train uses honest OOF)
    sel_cfg_ids = {f"{h}|{fs}|{chosen[(h, fs)]}" for h in ALL_HORIZONS for fs in FEATURE_SETS}
    tv15 = test_val[test_val["config_id"].isin(sel_cfg_ids)]
    oof_amb = oof[oof["track_key"].isin(amb_keys)]
    cls15 = pd.concat([
        classify_tracks(tv15, basis="15_selected_configs_val_f1_thresholds"),
        classify_tracks(oof_amb, basis="15_selected_configs_train_oof"),
    ], ignore_index=True)

    cls = pd.concat([cls_full, cls15], ignore_index=True)

    # wide persistence matrix (rows: amb tracks all splits; cols: configs)
    mat_long = pd.concat([
        test_val[["track_key", "split", "config_id", "pred"]],
        oof_amb[["track_key", "split", "config_id", "pred"]],
    ], ignore_index=True)
    mat = mat_long.pivot_table(index=["track_key", "split"], columns="config_id",
                               values="pred", aggfunc="first")
    mat = mat.reset_index()
    return cls, mat


# =========================================================================== phase B: profiling
BEH_ATTRS = ["look", "action", "hand_gesture", "nod", "reaction", "occlusion"]
_ANN_CACHE: dict[str, dict] = {}


def _load_video_behaviors(video_id: str) -> dict[tuple[str, int], dict[str, str]]:
    """(ped_id, frame) -> behavior attr dict, parsed once per video."""
    if video_id in _ANN_CACHE:
        return _ANN_CACHE[video_id]
    out: dict[tuple[str, int], dict[str, str]] = {}
    path = JAAD_ANN_DIR / f"{video_id}.xml"
    if path.exists():
        root = ET.parse(path).getroot()
        for track in root.findall("track"):
            for box in track.findall("box"):
                attrs = {a.attrib["name"]: (a.text or "") for a in box.findall("attribute")}
                pid = attrs.get("id")
                if pid is None:
                    continue
                out[(pid, int(float(box.attrib["frame"])))] = attrs
    _ANN_CACHE[video_id] = out
    return out


def behavior_features(rec: pd.Series) -> dict[str, float]:
    """Behavior fractions over the OBSERVED window frames only (never future frames).
    The per-frame `cross` tag is label-defining and is NOT extracted."""
    frames = json.loads(rec["observed_frames_json"])
    beh = _load_video_behaviors(str(rec["video_id"]))
    pid = str(rec["ped_id"])
    vals = {k: [] for k in ["look", "walk", "stand", "gesture", "nod", "react", "occl_any", "occl_full"]}
    n_found = 0
    for fr in frames:
        a = beh.get((pid, int(fr)))
        if a is None:
            continue
        n_found += 1
        vals["look"].append(1.0 if a.get("look") == "looking" else 0.0)
        vals["walk"].append(1.0 if a.get("action") == "walking" else 0.0)
        vals["stand"].append(1.0 if a.get("action") == "standing" else 0.0)
        vals["gesture"].append(0.0 if a.get("hand_gesture") in ("__undefined__", "", None) else 1.0)
        vals["nod"].append(0.0 if a.get("nod") in ("__undefined__", "", None) else 1.0)
        vals["react"].append(0.0 if a.get("reaction") in ("__undefined__", "", None) else 1.0)
        occ = a.get("occlusion", "none")
        vals["occl_any"].append(1.0 if occ in ("part", "full") else 0.0)
        vals["occl_full"].append(1.0 if occ == "full" else 0.0)

    def frac(v):
        return float(np.mean(v)) if v else float("nan")

    return {
        "beh_look_frac": frac(vals["look"]),
        "beh_walking_frac": frac(vals["walk"]),
        "beh_standing_frac": frac(vals["stand"]),
        "beh_gesture_frac": frac(vals["gesture"]),
        "beh_nod_frac": frac(vals["nod"]),
        "beh_reaction_frac": frac(vals["react"]),
        "beh_occlusion_frac": frac(vals["occl_any"]),
        "beh_occlusion_full_frac": frac(vals["occl_full"]),
        "beh_frames_annotated": float(n_found),
    }


# COCO-17 keypoint indices (matches src/biu_v2_extractor.py KPT_INDEX)
_CONF_IDX = {
    "lower_body": [11, 12, 13, 14, 15, 16],  # hips, knees, ankles
    "torso": [5, 6, 11, 12],                 # shoulders, hips
    "head": [0, 1, 2, 3, 4],                 # nose, eyes, ears
}


def confidence_features(rec: pd.Series, kp_index: dict) -> dict[str, float]:
    """Skeleton-confidence aggregates over the OBSERVED window frames. Computed
    directly from the archived ViTPose keypoints because the exp-009 extraction
    path uses include_optional=False (confidence_metadata never emitted)."""
    frames = json.loads(rec["observed_frames_json"])
    confs = []
    for fr in frames:
        item = kp_index.get((str(rec["video_id"]), str(rec["ped_id"]), int(fr)))
        if item is not None:
            confs.append(np.asarray(item["kp_conf"], dtype=float))
    if not confs:
        return {k: float("nan") for k in
                ["conf_skeleton_mean", "conf_skeleton_min_frame", "conf_lower_body_mean",
                 "conf_torso_mean", "conf_head_mean"]}
    C = np.stack(confs)  # (T, 17)
    return {
        "conf_skeleton_mean": float(np.nanmean(C)),
        "conf_skeleton_min_frame": float(np.nanmin(np.nanmean(C, axis=1))),
        "conf_lower_body_mean": float(np.nanmean(C[:, _CONF_IDX["lower_body"]])),
        "conf_torso_mean": float(np.nanmean(C[:, _CONF_IDX["torso"]])),
        "conf_head_mean": float(np.nanmean(C[:, _CONF_IDX["head"]])),
    }


def geometry_features(rec: pd.Series) -> dict[str, float]:
    """Interpretable bbox geometry over the observed window (JAAD pixel coords)."""
    seq = np.asarray(json.loads(rec["bbox_sequence_json"]), dtype=float)  # (T, 4) x1 y1 x2 y2
    w = seq[:, 2] - seq[:, 0]
    h = seq[:, 3] - seq[:, 1]
    cx = (seq[:, 0] + seq[:, 2]) / 2.0
    cy = (seq[:, 1] + seq[:, 3]) / 2.0
    mh = float(np.mean(h)) if np.mean(h) > 0 else float("nan")
    dx = np.diff(cx)
    dy = np.diff(cy)
    path = float(np.sum(np.hypot(dx, dy)))
    net = float(np.hypot(cx[-1] - cx[0], cy[-1] - cy[0]))
    return {
        "geo_bbox_height_mean": float(np.mean(h)),
        "geo_bbox_width_mean": float(np.mean(w)),
        "geo_bbox_aspect_mean": float(np.mean(w / np.where(h > 0, h, np.nan))),
        "geo_bbox_area_ratio_last_first": float((w[-1] * h[-1]) / max(w[0] * h[0], 1e-6)),
        "geo_center_net_dx_per_h": float(abs(cx[-1] - cx[0]) / mh) if mh == mh else float("nan"),
        "geo_center_net_dy_per_h": float(abs(cy[-1] - cy[0]) / mh) if mh == mh else float("nan"),
        "geo_center_net_disp_per_h": float(net / mh) if mh == mh else float("nan"),
        "geo_center_path_len_per_h": float(path / mh) if mh == mh else float("nan"),
        "geo_path_straightness": float(net / path) if path > 1e-6 else float("nan"),
    }


def build_profile_frame(meta: pd.DataFrame, assign: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """One row per exp-004 track: interpretable attributes at the canonical 30_60 window."""
    cache = CACHE_DIR / "track_profile.csv"
    cols_cache = CACHE_DIR / "track_profile_columns.json"
    if cache.exists() and cols_cache.exists():
        return pd.read_csv(cache), json.loads(cols_cache.read_text())

    rows3060 = base._load_split_rows("30_60")
    canon3060 = meta[meta["horizon_label"] == "30_60"]
    canon_rows = rows3060[rows3060["sequence_id"].isin(set(canon3060["sequence_id"]))].copy()
    canon_rows["track_key"] = canon_rows["video_id"].astype(str) + ":" + canon_rows["ped_id"].astype(str)

    # context (exp-016 table, no-motion honest set), keyed by 30_60 sequence_id
    ctx_tab = pd.read_csv(CTX_TABLE_PATH)
    ctx_cols = list(ctx.CONTEXT_NO_MOTION_COLS)

    # BIU v2 (all groups incl confidence) at canonical 30_60 windows
    v2 = get_v2_windows("30_60", rows3060)
    v2 = v2[v2["sequence_id"].isin(set(canon3060["sequence_id"]))]
    v2_cols = [c for c in base._numeric_feature_cols(v2)]

    kp_index = get_kp_index()
    prof_rows = []
    for _, rec in canon_rows.iterrows():
        r = {"track_key": rec["track_key"], "sequence_id_30_60": rec["sequence_id"],
             "meta_track_length": float(rec.get("track_length", np.nan)),
             "meta_tte_frames_30_60": float(rec["tte_frames"]),
             "meta_start_frame_30_60": float(rec["start_frame"])}
        r.update(geometry_features(rec))
        r.update(behavior_features(rec))
        r.update(confidence_features(rec, kp_index))
        prof_rows.append(r)
    prof = pd.DataFrame(prof_rows)
    prof = prof.merge(ctx_tab[["sequence_id"] + ctx_cols].rename(columns={"sequence_id": "sequence_id_30_60"}),
                      on="sequence_id_30_60", how="left")
    v2m = v2[["sequence_id"] + v2_cols].rename(columns={"sequence_id": "sequence_id_30_60"})
    prof = prof.merge(v2m, on="sequence_id_30_60", how="left")

    prof = assign[["track_key", "split", "group"]].merge(prof, on="track_key", how="left")

    attr_cols = (ctx_cols + v2_cols
                 + [c for c in prof.columns if c.startswith(("geo_", "beh_", "meta_", "conf_"))])
    attr_cols = [c for c in dict.fromkeys(attr_cols) if c != "beh_frames_annotated"]
    prof.to_csv(cache, index=False)
    cols_cache.write_text(json.dumps(attr_cols))
    return prof, attr_cols


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    sp = np.sqrt(((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / max(na + nb - 2, 1))
    if sp == 0:
        return float("nan")
    return float((np.mean(a) - np.mean(b)) / sp)


def effect_table(prof: pd.DataFrame, attr_cols: list[str],
                 mask_a: pd.Series, mask_b: pd.Series,
                 name_a: str, name_b: str, comparison: str) -> pd.DataFrame:
    rows = []
    for col in attr_cols:
        a = prof.loc[mask_a, col].dropna().to_numpy(dtype=float)
        b = prof.loc[mask_b, col].dropna().to_numpy(dtype=float)
        if len(a) < 3 or len(b) < 3 or (np.std(a) == 0 and np.std(b) == 0):
            continue
        try:
            u, p = mannwhitneyu(a, b, alternative="two-sided")
        except ValueError:
            continue
        rb = float(2.0 * u / (len(a) * len(b)) - 1.0)  # rank-biserial (positive => a > b)
        rows.append({"comparison": comparison, "attribute": col,
                     "group_a": name_a, "group_b": name_b,
                     "n_a": len(a), "n_b": len(b),
                     "mean_a": float(np.mean(a)), "mean_b": float(np.mean(b)),
                     "median_a": float(np.median(a)), "median_b": float(np.median(b)),
                     "cohen_d": cohen_d(a, b), "rank_biserial": rb, "mw_p": float(p)})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # Benjamini-Hochberg within the comparison (exploratory honesty)
    df = df.sort_values("mw_p").reset_index(drop=True)
    m = len(df)
    q = df["mw_p"].to_numpy() * m / (np.arange(m) + 1)
    df["bh_q"] = np.minimum.accumulate(q[::-1])[::-1]
    return df.sort_values("rank_biserial", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)


def phase_b(prof: pd.DataFrame, attr_cols: list[str], cls: pd.DataFrame) -> pd.DataFrame:
    cls15 = cls[cls["basis"].str.startswith("15_selected")].set_index("track_key")["category"]
    prof = prof.copy()
    prof["category_15cfg"] = prof["track_key"].map(cls15)

    is_amb = prof["group"] == "ambiguous_negative"
    is_pos = prof["group"] == "actual_positive"
    irr = is_amb & prof["category_15cfg"].isin(["ALWAYS_FP", "MOSTLY_FP"])
    resc = is_amb & ~prof["category_15cfg"].isin(["ALWAYS_FP", "MOSTLY_FP"]) & prof["category_15cfg"].notna()

    tables = [
        effect_table(prof, attr_cols, irr, resc,
                     "amb_irreducible", "amb_other", "A_irreducible_vs_other_ambiguous"),
        effect_table(prof, attr_cols, is_amb, is_pos,
                     "ambiguous_negative", "actual_positive", "B_ambiguous_vs_actual_positive"),
        effect_table(prof, attr_cols, irr, is_pos,
                     "amb_irreducible", "actual_positive", "C_irreducible_amb_vs_actual_positive"),
    ]
    eff = pd.concat([t for t in tables if not t.empty], ignore_index=True)
    return eff


# =========================================================================== phase C1a: stage-2 with context
def build_context_for_rows(rows: pd.DataFrame) -> pd.DataFrame:
    attr_index, _ = ctx.load_attribute_index()
    traffic_index = ctx.load_traffic_index()
    out = []
    for _, rec in rows.iterrows():
        r = {"sequence_id": rec["sequence_id"]}
        r.update(ctx.build_context_row(rec, attr_index, traffic_index))
        out.append(r)
    df = pd.DataFrame(out)
    return df[["sequence_id"] + list(ctx.CONTEXT_NO_MOTION_COLS)]


def phase_c1a(tt: pd.DataFrame, meta: pd.DataFrame, chosen: dict,
              staged_preds: dict, staged_ops: pd.DataFrame) -> tuple[list[dict], list[dict], dict]:
    """Retrain stage 2 = biu_v2_core_plus_bbox + context_no_motion at 5_15 and
    corrected 15_30; staged behind exp-011 stage 1 at matched recall targets."""
    fs_name = "v2bbox_ctx_no_motion"
    result_rows, boot_rows = [], []
    test_preds: dict[tuple[str, float], pd.DataFrame] = {}

    s1_m = chosen[("30_60", "bbox_only")]
    s1_val = s11.get_split(tt, "30_60", "bbox_only", s1_m, "val")
    s1_test = s11.get_split(tt, "30_60", "bbox_only", s1_m, "test")

    for h in ["5_15", H_COR]:
        frames, columns, rows = build_window_feature_sets(h)
        vb = frames["biu_v2_core_plus_bbox"]
        vb_cols = columns["biu_v2_core_plus_bbox"]
        ctx_cache = CACHE_DIR / f"context_windows_{h}.csv"
        if ctx_cache.exists():
            ctx_df = pd.read_csv(ctx_cache)
        else:
            ctx_df = build_context_for_rows(rows)
            ctx_df.to_csv(ctx_cache, index=False)
        df = vb.merge(ctx_df, on="sequence_id", how="inner")
        assert len(df) == len(vb), f"context merge changed row count at {h}"
        cols = sorted(set(vb_cols) | set(ctx.CONTEXT_NO_MOTION_COLS))
        assert_clean_features(cols, f"c1a:{h}")

        # train seeds x models, save val/test predictions in the exp-011 format
        pred_h_dir = PRED_DIR / h
        pred_h_dir.mkdir(parents=True, exist_ok=True)
        tr = df[df["split"] == "train"]
        va = df[df["split"] == "val"]
        te = df[df["split"] == "test"]
        y_tr = tr["label"].astype(int).to_numpy()
        track_rows = []
        for m in MODELS:
            probs_val, probs_test = [], []
            for seed in SEEDS:
                pth = pred_h_dir / f"{fs_name}__{m}__seed{seed}.csv"
                if pth.exists():
                    saved = pd.read_csv(pth)
                    pv = saved[saved["split"] == "val"].set_index("sequence_id")["prob"]
                    pt = saved[saved["split"] == "test"].set_index("sequence_id")["prob"]
                    probs_val.append(pv.loc[va["sequence_id"]].to_numpy())
                    probs_test.append(pt.loc[te["sequence_id"]].to_numpy())
                    continue
                pipe = base._build_model(m, seed, y_tr)
                pipe.fit(tr[cols].astype(float), y_tr)
                pv = pipe.predict_proba(va[cols].astype(float))[:, 1]
                pt = pipe.predict_proba(te[cols].astype(float))[:, 1]
                probs_val.append(pv)
                probs_test.append(pt)
                keep = [c for c in ["sequence_id", "split", "label", "group", "video_id", "ped_id", "track_id"]
                        if c in df.columns]
                out = pd.concat([
                    va[keep].assign(prob=pv),
                    te[keep].assign(prob=pt),
                ], ignore_index=True)
                out["horizon_label"] = h
                out["feature_set"] = fs_name
                out["model_name"] = m
                out["seed"] = seed
                out.to_csv(pth, index=False)
            ens_val = np.mean(probs_val, axis=0)
            ens_test = np.mean(probs_test, axis=0)
            mm = meta[meta["horizon_label"] == h][["sequence_id", "track_key", "split", "label", "group"]]
            sub = pd.concat([
                va[["sequence_id"]].assign(prob=ens_val),
                te[["sequence_id"]].assign(prob=ens_test),
            ], ignore_index=True).merge(mm, on="sequence_id", how="inner")  # canonical windows only
            sub["horizon_label"] = h
            sub["feature_set"] = fs_name
            sub["model_name"] = m
            track_rows.append(sub)
        c1a_tt = pd.concat(track_rows, ignore_index=True)

        m_star = s11.pick_model(c1a_tt, h, fs_name)
        s2_val = s11.get_split(c1a_tt, h, fs_name, m_star, "val")
        s2_test = s11.get_split(c1a_tt, h, fs_name, m_star, "test")

        for target in RECALL_TARGETS:
            t1 = s11.threshold_for_recall(s1_val["label"].to_numpy(), s1_val["prob"].to_numpy(), target)
            passed = s1_val[s1_val["prob"] >= t1]
            s2_val_passed = s2_val.loc[s2_val.index.intersection(passed.index)]
            if s2_val_passed.empty or s2_val_passed["label"].nunique() < 2:
                t2 = 0.5
            else:
                t2 = s11.threshold_for_recall(s2_val_passed["label"].to_numpy(),
                                              s2_val_passed["prob"].to_numpy(), target)
            joined = s11.staged_predict(s1_test, s2_test, t1, t2)
            test_preds[(h, target)] = joined
            met = s11.track_metrics(joined["label"].to_numpy(), joined["pred"].to_numpy(),
                                    joined["group"].to_numpy())
            result_rows.append({"attempt": "C1a_stage2_context_retrain",
                                "config": f"staged|bbox30_60->{fs_name}@{h}", "model": f"{s1_m}+{m_star}",
                                "recall_target": target, "t1": t1, "t2": t2, **met})

            # bootstrap vs the generic staged baselines at the same target
            for base_fs in ["biu_v2_core", "biu_v2_core_plus_bbox"]:
                bjoined = staged_preds[(h, base_fs, target)]
                common = joined.index.intersection(bjoined.index)
                res = s11.bootstrap_delta(joined.loc[common, "label"].to_numpy(),
                                          joined.loc[common, "group"].to_numpy(),
                                          bjoined.loc[common, "pred"].to_numpy(),
                                          joined.loc[common, "pred"].to_numpy())
                brow = staged_ops[(staged_ops["config"] == f"staged|bbox30_60->{base_fs}@{h}")
                                  & (staged_ops["recall_target"] == target)].iloc[0]
                boot_rows.append({"attempt": "C1a_stage2_context_retrain",
                                  "config": f"staged|bbox30_60->{fs_name}@{h}",
                                  "baseline": f"staged|bbox30_60->{base_fs}@{h}",
                                  "recall_target": target, "n_common_tracks": len(common),
                                  "baseline_f1": brow["f1"], "baseline_amb_fpr": brow["amb_neg_fpr"],
                                  **res})
    return result_rows, boot_rows, test_preds


# =========================================================================== phase C1b: interpretable rule stage 2
def build_rule_features(meta: pd.DataFrame, prof: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Track-level leak-audited features: context no-motion + bbox deltas
    (5_15 - 30_60) + BIU v2 core statics @5_15 + skeleton confidence @30_60."""
    ctx_cols = list(ctx.CONTEXT_NO_MOTION_COLS)
    conf_cols = [c for c in prof.columns if c.startswith("conf_")]

    feature_dict = pd.read_csv(base.DICT_PATH)
    v2_core_prefixes = base._base_feature_prefixes(
        feature_dict, ["original_anatomical_biu", "lower_body_commitment_biu", "torso_center_commitment_biu"])

    fam = {}
    for h in ["30_60", "5_15"]:
        rows = get_split_rows(h)
        canon_ids = meta.loc[meta["horizon_label"] == h].set_index("sequence_id")["track_key"]
        bbox = base._load_existing_feature_frame(h, "bbox")
        bbox = bbox.drop(columns=[c for c in bbox.columns if c != "sequence_id" and c in base.META_COLS],
                         errors="ignore")
        bbox = bbox[bbox["sequence_id"].isin(canon_ids.index)].copy()
        bbox["track_key"] = bbox["sequence_id"].map(canon_ids)
        bcols = [c for c in bbox.columns if c not in ("sequence_id", "track_key")
                 and pd.api.types.is_numeric_dtype(bbox[c]) and "label" not in c.lower()]
        v2 = get_v2_windows(h, rows)
        v2 = v2[v2["sequence_id"].isin(canon_ids.index)].copy()
        v2["track_key"] = v2["sequence_id"].map(canon_ids)
        vcols = base._select_prefixed_columns(v2, v2_core_prefixes)
        fam[h] = {"bbox": bbox.set_index("track_key")[bcols], "v2": v2.set_index("track_key")[vcols]}

    bcommon = sorted(set(fam["30_60"]["bbox"].columns) & set(fam["5_15"]["bbox"].columns))
    common_idx = fam["5_15"]["bbox"].index.intersection(fam["30_60"]["bbox"].index)
    dlt = (fam["5_15"]["bbox"].loc[common_idx, bcommon]
           - fam["30_60"]["bbox"].loc[common_idx, bcommon]).add_prefix("dlt_bbox_")
    sh_v2 = fam["5_15"]["v2"].add_prefix("sh515_")

    pctx = prof.set_index("track_key")[ctx_cols + conf_cols]
    X = dlt.join(sh_v2, how="inner").join(pctx, how="left")
    cols = list(X.columns)
    assert_clean_features(cols, "c1b_rule_features")
    X = X.reset_index().rename(columns={"index": "track_key"})
    return X, cols


def phase_c1b(tt: pd.DataFrame, meta: pd.DataFrame, prof: pd.DataFrame, assign: pd.DataFrame,
              chosen: dict, staged_preds: dict, staged_ops: pd.DataFrame
              ) -> tuple[list[dict], list[dict], dict, dict]:
    X, cols = build_rule_features(meta, prof)
    info = assign.set_index("track_key")[["split", "group"]]
    X = X.merge(info, left_on="track_key", right_index=True, how="inner")

    tr_mask = (X["split"] == "train") & X["group"].isin(["actual_positive", "ambiguous_negative"])
    X_tr = X.loc[tr_mask, cols]
    y_tr = (X.loc[tr_mask, "group"] == "actual_positive").astype(int).to_numpy()

    def fit_ensemble(kind: str, C: float | None = None) -> pd.Series:
        probs = []
        for seed in SEEDS:
            if kind == "tree":
                pipe = Pipeline([("imputer", SimpleImputer(strategy="median")),
                                 ("model", DecisionTreeClassifier(max_depth=3, min_samples_leaf=5,
                                                                  class_weight="balanced", random_state=seed))])
            else:
                pipe = Pipeline([("imputer", SimpleImputer(strategy="median")),
                                 ("scaler", StandardScaler()),
                                 ("model", LogisticRegression(penalty="l1", solver="liblinear", C=C,
                                                              class_weight="balanced", max_iter=2000,
                                                              random_state=seed))])
            pipe.fit(X_tr, y_tr)
            probs.append(pipe.predict_proba(X[cols])[:, 1])
        return pd.Series(np.mean(probs, axis=0), index=X["track_key"].to_numpy())

    # validation-only selection: which rule model (and which C for L1) by val actual-vs-amb ROC-AUC
    val_mask = (X["split"] == "val") & X["group"].isin(["actual_positive", "ambiguous_negative"])
    y_val = (X.loc[val_mask, "group"] == "actual_positive").astype(int).to_numpy()
    candidates: dict[str, pd.Series] = {"tree_depth3": fit_ensemble("tree")}
    for C in [0.05, 0.1, 0.5, 1.0]:
        candidates[f"l1_logreg_C{C}"] = fit_ensemble("l1", C)
    sel_rows = []
    for name, s in candidates.items():
        pv = s.loc[X.loc[val_mask, "track_key"]].to_numpy()
        auc = float(roc_auc_score(y_val, pv)) if len(np.unique(y_val)) == 2 else float("nan")
        sel_rows.append({"rule_model": name, "val_roc_auc_actual_vs_amb": auc,
                         "note": "val has only 18 actual / 5 ambiguous tracks; selection is noisy"})
    sel_df = pd.DataFrame(sel_rows).sort_values(["val_roc_auc_actual_vs_amb", "rule_model"],
                                                ascending=[False, True]).reset_index(drop=True)
    sel_df.to_csv(METRICS_DIR / "rule_model_selection.csv", index=False)
    primary = sel_df.iloc[0]["rule_model"]

    # export interpretability artifacts (seed 42 refit)
    tree = Pipeline([("imputer", SimpleImputer(strategy="median")),
                     ("model", DecisionTreeClassifier(max_depth=3, min_samples_leaf=5,
                                                      class_weight="balanced", random_state=42))])
    tree.fit(X_tr, y_tr)
    (METRICS_DIR / "rule_model_tree.txt").write_text(
        "Depth-3 decision tree, actual_positive(1) vs ambiguous_negative(0), train tracks, seed 42\n\n"
        + export_text(tree.named_steps["model"], feature_names=cols), encoding="utf-8")
    l1 = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()),
                   ("model", LogisticRegression(penalty="l1", solver="liblinear", C=0.1,
                                                class_weight="balanced", max_iter=2000, random_state=42))])
    l1.fit(X_tr, y_tr)
    coefs = pd.DataFrame({"feature": cols, "l1_coef_C0.1_seed42": l1.named_steps["model"].coef_[0]})
    coefs[coefs["l1_coef_C0.1_seed42"] != 0].sort_values(
        "l1_coef_C0.1_seed42", key=lambda s: s.abs(), ascending=False
    ).to_csv(METRICS_DIR / "rule_model_l1_coefficients.csv", index=False)

    # staged evaluation for the two rule kinds (primary flagged), matched-recall thresholds
    s1_m = chosen[("30_60", "bbox_only")]
    s1_val = s11.get_split(tt, "30_60", "bbox_only", s1_m, "val")
    s1_test = s11.get_split(tt, "30_60", "bbox_only", s1_m, "test")
    result_rows, boot_rows = [], []
    test_preds: dict[tuple[str, float], pd.DataFrame] = {}
    xinfo = X.set_index("track_key")

    eval_names = ["tree_depth3", primary] if primary != "tree_depth3" else ["tree_depth3", sel_df.iloc[1]["rule_model"]]
    for name in dict.fromkeys(eval_names):
        s = candidates[name]
        v = pd.DataFrame({"prob": s})
        v["split"] = xinfo["split"]
        v["label"] = (xinfo["group"] == "actual_positive").astype(int)  # rule prob semantics
        v_val = v[v["split"] == "val"]
        v_test = v[v["split"] == "test"]
        for target in RECALL_TARGETS:
            t1 = s11.threshold_for_recall(s1_val["label"].to_numpy(), s1_val["prob"].to_numpy(), target)
            passed_idx = s1_val.index[s1_val["prob"] >= t1]
            v_sel = v_val.loc[v_val.index.intersection(passed_idx)]
            # threshold on the true crossing label of stage-1-passed val tracks
            lbl = s1_val.loc[v_sel.index, "label"].to_numpy()
            if v_sel.empty or len(np.unique(lbl)) < 2:
                t2 = 0.5
            else:
                t2 = s11.threshold_for_recall(lbl, v_sel["prob"].to_numpy(), target)
            joined = s1_test[["label", "group", "prob"]].rename(columns={"prob": "p1"}).join(
                v_test[["prob"]].rename(columns={"prob": "p2"}), how="left")
            stage1 = (joined["p1"] >= t1).astype(int)
            stage2 = np.where(joined["p2"].notna(), (joined["p2"] >= t2).astype(int), 1)
            joined["pred"] = (stage1.to_numpy() & stage2).astype(int)
            test_preds[(name, target)] = joined
            met = s11.track_metrics(joined["label"].to_numpy(), joined["pred"].to_numpy(),
                                    joined["group"].to_numpy())
            result_rows.append({"attempt": "C1b_interpretable_rule",
                                "config": f"staged|bbox30_60->rule[{name}]",
                                "model": f"{s1_m}+{name}", "recall_target": target,
                                "t1": t1, "t2": t2,
                                "is_primary_val_selected": name == primary,
                                "n_stage2_missing_kept": int(((stage1 == 1) & joined["p2"].isna()).sum()),
                                **met})
            for (bh, bfs) in [("5_15", "biu_v2_core"), (H_COR, "biu_v2_core_plus_bbox")]:
                bjoined = staged_preds[(bh, bfs, target)]
                common = joined.index.intersection(bjoined.index)
                res = s11.bootstrap_delta(joined.loc[common, "label"].to_numpy(),
                                          joined.loc[common, "group"].to_numpy(),
                                          bjoined.loc[common, "pred"].to_numpy(),
                                          joined.loc[common, "pred"].to_numpy())
                brow = staged_ops[(staged_ops["config"] == f"staged|bbox30_60->{bfs}@{bh}")
                                  & (staged_ops["recall_target"] == target)].iloc[0]
                boot_rows.append({"attempt": "C1b_interpretable_rule",
                                  "config": f"staged|bbox30_60->rule[{name}]",
                                  "baseline": f"staged|bbox30_60->{bfs}@{bh}",
                                  "recall_target": target, "n_common_tracks": len(common),
                                  "baseline_f1": brow["f1"], "baseline_amb_fpr": brow["amb_neg_fpr"],
                                  **res})
    return result_rows, boot_rows, {"primary_rule_model": primary}, test_preds


# =========================================================================== phase C2: oracle bounds
def pattern_direction(prof: pd.DataFrame, col: str, mask_target: pd.Series, mask_ref: pd.Series,
                      trainval_only: bool = True) -> tuple[float, str] | None:
    """Cutpoint + direction chosen on train+val only: the side of the cut where the
    TARGET group (e.g. ambiguous) is enriched relative to the REFERENCE group."""
    sel = prof["split"].isin(["train", "val"]) if trainval_only else pd.Series(True, index=prof.index)
    a = prof.loc[mask_target & sel, col].dropna()
    b = prof.loc[mask_ref & sel, col].dropna()
    if len(a) < 3 or len(b) < 3:
        return None
    vals = pd.concat([a, b])
    uniq = set(np.unique(vals))
    cut = 0.5 if uniq <= {0.0, 1.0} else float(np.median(vals))
    frac_a_hi = float((a >= cut).mean())
    frac_b_hi = float((b >= cut).mean())
    direction = ">=" if frac_a_hi >= frac_b_hi else "<"
    return cut, direction


def apply_pattern(prof_idx: pd.DataFrame, col: str, cut: float, direction: str) -> pd.Series:
    v = prof_idx[col]
    return (v >= cut) if direction == ">=" else (v < cut)


def phase_c2(prof: pd.DataFrame, eff: pd.DataFrame, cls: pd.DataFrame,
             staged_preds: dict, staged_ops: pd.DataFrame) -> pd.DataFrame:
    """Oracle ceilings: if a pattern perfectly identified its ambiguous subgroup and
    vetoed only those tracks, what would test amb-FPR / F1 become?"""
    prof_idx = prof.set_index("track_key")
    is_amb = prof["group"] == "ambiguous_negative"
    is_pos = prof["group"] == "actual_positive"
    cls15 = cls[cls["basis"].str.startswith("15_selected")].set_index("track_key")["category"]
    irr_keys = set(cls15[cls15.isin(["ALWAYS_FP", "MOSTLY_FP"])].index)
    irr_mask = prof["track_key"].isin(irr_keys) & is_amb
    other_amb_mask = is_amb & ~prof["track_key"].isin(irr_keys)

    # candidate patterns: top-5 from B (amb vs pos) and top-5 from A (irreducible vs other amb)
    cands: list[tuple[str, str]] = []
    for comp, m_t, m_r in [("B_ambiguous_vs_actual_positive", is_amb, is_pos),
                           ("A_irreducible_vs_other_ambiguous", irr_mask, other_amb_mask)]:
        sub = eff[eff["comparison"] == comp].head(5)
        for _, r in sub.iterrows():
            cands.append((r["attribute"], comp))
    seen = set()
    patterns = []
    for col, comp in cands:
        if col in seen:
            continue
        seen.add(col)
        m_t, m_r = (is_amb, is_pos) if comp.startswith("B") else (irr_mask, other_amb_mask)
        pd_res = pattern_direction(prof, col, m_t, m_r)
        if pd_res is None:
            continue
        cut, direction = pd_res
        patterns.append({"pattern": f"{col} {direction} {cut:g}", "column": col,
                         "cut": cut, "direction": direction, "source_comparison": comp})

    rows = []
    ref_configs = [("5_15", "biu_v2_core"), (H_COR, "biu_v2_core_plus_bbox")]
    for (h, fs), target in product(ref_configs, RECALL_TARGETS):
        joined = staged_preds[(h, fs, target)]
        y = joined["label"].to_numpy()
        g = joined["group"].to_numpy()
        pred0 = joined["pred"].to_numpy()
        base_met = s11.track_metrics(y, pred0, g)
        cfg = f"staged|bbox30_60->{fs}@{h}"

        def oracle_row(name: str, suppress_keys: set, extra: dict | None = None) -> dict:
            pred = pred0.copy()
            amb_here = joined.index.isin(suppress_keys) & (g == "ambiguous_negative")
            n_flipped = int((pred[amb_here] == 1).sum())
            pred[amb_here] = 0
            met = s11.track_metrics(y, pred, g)
            return {"reference_config": cfg, "recall_target": target, "oracle": name,
                    "base_f1": base_met["f1"], "base_amb_fpr": base_met["amb_neg_fpr"],
                    "oracle_f1": met["f1"], "oracle_amb_fpr": met["amb_neg_fpr"],
                    "delta_f1": met["f1"] - base_met["f1"],
                    "delta_amb_fpr": met["amb_neg_fpr"] - base_met["amb_neg_fpr"],
                    "n_amb_fp_suppressed": n_flipped,
                    "n_amb_tracks": base_met["n_amb"], **(extra or {})}

        # 1. full oracle: every ambiguous FP suppressed
        rows.append(oracle_row("FULL_all_ambiguous_fps_suppressed", set(joined.index)))
        # 2. rescuable-only oracle: irreducible-core FPs cannot be fixed
        rows.append(oracle_row("RESCUABLE_only_non_irreducible_suppressed",
                               set(joined.index) - irr_keys,
                               {"note": "irreducible = ALWAYS_FP/MOSTLY_FP (15-config basis)"}))
        # 3. per-pattern oracles
        for p in patterns:
            match = apply_pattern(prof_idx, p["column"], p["cut"], p["direction"])
            keys = set(match[match].index)
            rows.append(oracle_row(f"PATTERN[{p['pattern']}]", keys,
                                   {"note": p["source_comparison"]}))
    return pd.DataFrame(rows)


# =========================================================================== phase D: dossier
def phase_d(assign: pd.DataFrame, cls: pd.DataFrame, preds: pd.DataFrame, oof: pd.DataFrame,
            prof: pd.DataFrame, eff: pd.DataFrame, meta: pd.DataFrame,
            staged_preds: dict, c1a_preds: dict, c1b_preds: dict) -> pd.DataFrame:
    amb = assign[assign["group"] == "ambiguous_negative"].copy()
    cls45 = cls[cls["basis"] == "45_saved_configs_val_f1_thresholds"].set_index("track_key")
    cls15 = cls[cls["basis"].str.startswith("15_selected")].set_index("track_key")

    top_b = eff[eff["comparison"] == "B_ambiguous_vs_actual_positive"].head(5)["attribute"].tolist()
    prof_idx = prof.drop_duplicates("track_key").set_index("track_key")

    fp_all = pd.concat([preds[["track_key", "config_id", "pred"]],
                        oof[["track_key", "config_id", "pred"]]], ignore_index=True)

    def window_str(tk: str, h: str) -> str:
        m = meta[(meta["horizon_label"] == h) & (meta["track_key"] == tk)]
        if m.empty:
            return ""
        sid = m.iloc[0]["sequence_id"]
        parts = sid.split(":")
        return f"obs {parts[2]}-{parts[3]}, anchor {parts[4]}" if len(parts) >= 5 else sid

    rows = []
    for _, r in amb.iterrows():
        tk = r["track_key"]
        c45 = cls45.loc[tk] if tk in cls45.index else None
        c15 = cls15.loc[tk] if tk in cls15.index else None
        sub = fp_all[fp_all["track_key"] == tk]
        fp_configs = sorted(sub.loc[sub["pred"] == 1, "config_id"].unique())
        row = {
            "video": r["video"], "pid": r["pid"], "track_key": tk, "split": r["split"],
            "category_primary": (c45["category"] if c45 is not None
                                 else (c15["category"] if c15 is not None else "NO_COVERAGE")),
            "category_basis": (c45["basis"] if c45 is not None
                               else (c15["basis"] if c15 is not None else "")),
            "fp_rate": (c45["fp_rate"] if c45 is not None else (c15["fp_rate"] if c15 is not None else np.nan)),
            "n_configs_covered": (c45["n_configs"] if c45 is not None
                                  else (c15["n_configs"] if c15 is not None else 0)),
            "fp_configs": ";".join(fp_configs),
            "canonical_window_30_60": window_str(tk, "30_60"),
            "canonical_window_15_30_corrected": window_str(tk, H_COR),
            "canonical_window_5_15": window_str(tk, "5_15"),
            "crossing_point_meta": r["crossing_point"], "decision_point_meta": r["decision_point"],
            "exp004_notes": r["notes"],
        }
        for (h, fs), target in product([("5_15", "biu_v2_core"), (H_COR, "biu_v2_core_plus_bbox")],
                                       RECALL_TARGETS):
            j = staged_preds[(h, fs, target)]
            key = f"staged_fp_{fs}@{h}_r{target}"
            row[key] = int(j.loc[tk, "pred"]) if tk in j.index else np.nan
        for (h, target), j in c1a_preds.items():
            row[f"c1a_ctx_stage2_fp@{h}_r{target}"] = int(j.loc[tk, "pred"]) if tk in j.index else np.nan
        for (name, target), j in c1b_preds.items():
            row[f"c1b_rule_{name}_fp_r{target}"] = int(j.loc[tk, "pred"]) if tk in j.index else np.nan
        for col in top_b:
            row[f"attr_top_{col}"] = prof_idx.loc[tk, col] if tk in prof_idx.index else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


# =========================================================================== main
def main() -> None:
    for d in (METRICS_DIR, PRED_DIR, CACHE_DIR):
        d.mkdir(parents=True, exist_ok=True)

    assign = load_assignments()
    tt = load_track_table()
    chosen = chosen_models()
    meta = canonical_meta_all_splits()

    print("=== A1: val/test FP matrix (45 saved configs) ===")
    preds, thrs = phase_a1(tt)
    thrs.to_csv(METRICS_DIR / "config_val_f1_thresholds.csv", index=False)

    print("=== staged baselines (parity-checked recompute) ===")
    staged_preds, staged_ops, parity = recompute_staged_baselines(tt, chosen)
    staged_ops.to_csv(METRICS_DIR / "staged_baseline_operating_points.csv", index=False)
    parity.to_csv(METRICS_DIR / "staged_baseline_parity_check.csv", index=False)

    print("=== A2: train OOF track predictions (15 selected configs) ===")
    oof = phase_a2_train_oof(meta, thrs, chosen)

    print("=== A3: persistence classification ===")
    cls, mat = phase_a3(preds, oof, assign, chosen)
    cls.to_csv(METRICS_DIR / "track_classification.csv", index=False)
    mat.to_csv(METRICS_DIR / "fp_persistence_matrix.csv", index=False)

    print("=== B: profiling ===")
    prof, attr_cols = build_profile_frame(meta, assign)
    eff = phase_b(prof, attr_cols, cls)
    eff.to_csv(METRICS_DIR / "group_profile_effects.csv", index=False)

    print("=== C1a: stage-2 context retrain ===")
    c1a_rows, c1a_boot, c1a_preds = phase_c1a(tt, meta, chosen, staged_preds, staged_ops)

    print("=== C1b: interpretable rule stage 2 ===")
    c1b_rows, c1b_boot, c1b_info, c1b_preds = phase_c1b(tt, meta, prof, assign, chosen,
                                                        staged_preds, staged_ops)

    combo = pd.DataFrame(c1a_rows + c1b_rows)
    combo.to_csv(METRICS_DIR / "combo_rule_results.csv", index=False)
    pd.DataFrame(c1a_boot + c1b_boot).to_csv(METRICS_DIR / "combo_rule_bootstrap.csv", index=False)

    print("=== C2: oracle bounds ===")
    oracle = phase_c2(prof, eff, cls, staged_preds, staged_ops)
    oracle.to_csv(METRICS_DIR / "oracle_bounds.csv", index=False)

    print("=== D: dossier ===")
    dossier = phase_d(assign, cls, preds, oof, prof, eff, meta, staged_preds, c1a_preds, c1b_preds)
    dossier.to_csv(METRICS_DIR / "ambiguous_negative_dossier.csv", index=False)

    test_counts = (cls[(cls["basis"] == "45_saved_configs_val_f1_thresholds") & (cls["split"] == "test")]
                   .groupby("category").size().to_dict())
    summary = {
        "test_amb_category_counts_45cfg": test_counts,
        "n_dossier_rows": int(len(dossier)),
        "primary_rule_model": c1b_info["primary_rule_model"],
        "parity_status": parity["status"].value_counts().to_dict(),
    }
    (METRICS_DIR / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
