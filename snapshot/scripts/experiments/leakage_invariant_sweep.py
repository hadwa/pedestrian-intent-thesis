"""Experiment 022: leakage & invariant sweep (verification-only audit, no model training).

After experiments 011-021 surfaced two bugs -- (1) exp-019's pandas merge
collision that let the label enter the verifier features as `sh_label_x`, and
(2) the archived JAAD tte_15_30 / tte_60_90 stage tables being silent
artifacts of a 30_60-only keypoint filter -- this script systematically
verifies the whole July experiment set with three sweeps:

Sweep A -- merge-collision / label-leak audit of every feature matrix actually
  fed to models in experiments 009/011/012/016/019/021 (rebuilt via each
  experiment's OWN loading code, imported as modules):
  - JAAD 30_60 / 5_15 (+ broken archived 15_30, as consumed by exp 011) via
    scripts/experiments/jaad_biu_v2_benchmark_comparison.py, 5 feature sets.
  - Corrected 15_30 matrices (exp 021) via jaad_corrected_15_30_rebuild_benchmark.py.
  - Exp-016 context feature sets via jaad_context_features.py (all 8 sets).
  - Exp-019 verifier variant matrices (post-fix) via jaad_commitment_verifier.py.
  - PIE 30_60 matrices via pie_biu_v2_30_60_benchmark.py.
  Checks per final model-input matrix: forbidden label/meta columns (meta-STEM
  matching with sh_/dlt_ prefix and _x/_y merge-suffix stripping, so bbox *_x
  coordinate columns are NOT false positives), per-column standalone
  train ROC-AUC (rank-based; NaNs dropped; flag oriented AUC >= 0.95; top-10
  recorded per matrix), and perfect label correlation (|Spearman| == 1).
  Test-set per-column AUC is ALSO computed but ONLY as an audit statistic to
  report leakage risk -- it is never used for any selection.
  Plus a STATIC audit of every pandas merge/join/concat in the July scripts
  with a curated one-line verdict per site.

Sweep B -- proxy screen: the 20 kept exp-016 context features and all
  corrected-15_30 feature columns, each with standalone train AUC and a note.

Sweep C -- dataset invariant certificate for the tables the thesis quotes
  (JAAD 30_60, JAAD 5_15, JAAD corrected 15_30, PIE 30_60): TTE bounds,
  window-before-event, split disjointness (track + video / PIE set-based),
  cross-horizon canonical-window overlap (exp-004 rule: min tte_frames then
  latest start_frame), label consistency + class ratios. The archived broken
  15_30 and 60_90 tables are certified too as KNOWN-BROKEN reference rows.

Rules honoured: verification only (no retraining, no test-set selection; the
only models this script touches are never fit); on FAIL nothing is "fixed" --
the failure is recorded precisely and the sweep continues.

Outputs: experiments/022_leakage_and_invariant_sweep/metrics/
  feature_matrix_audit.csv, feature_matrix_audit_columns.csv,
  standalone_auc_top10.csv, merge_static_audit.csv,
  context_proxy_screen.csv, invariant_certificate.csv

Rerun:
  cd /home/abdelrahman/Hadwa/BIU-PID-Distillation-clean && \
  MPLCONFIGDIR=/tmp/mplconfig python3 scripts/experiments/leakage_invariant_sweep.py
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

CLEAN_ROOT = Path("/home/abdelrahman/Hadwa/BIU-PID-Distillation-clean")
ARCHIVE_ROOT = Path("/home/abdelrahman/Hadwa/BIU-PID-Distillation")
SCRIPTS_DIR = CLEAN_ROOT / "scripts/experiments"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import jaad_biu_v2_benchmark_comparison as base  # noqa: E402  (exp-009 loaders)
import jaad_context_features as ctx  # noqa: E402  (exp-016 loaders)
import jaad_commitment_verifier as v19  # noqa: E402  (exp-019 loaders, post-fix)
import jaad_corrected_15_30_rebuild as c21r  # noqa: E402  (exp-021 rebuild: canonical rule, table paths)
import jaad_corrected_15_30_rebuild_benchmark as c21  # noqa: E402  (exp-021 matrix loaders)
import pie_biu_v2_30_60_benchmark as pie  # noqa: E402  (exp-012 loaders)

OUT_ROOT = CLEAN_ROOT / "experiments/022_leakage_and_invariant_sweep"
METRICS_DIR = OUT_ROOT / "metrics"

AUC_FLAG_THRESHOLD = 0.95
SPEARMAN_PERFECT_TOL = 1e-9
TOP_N = 10

# ---------------------------------------------------------------------------
# Forbidden-column logic (Sweep A check 1)
# ---------------------------------------------------------------------------
# Exact meta/label stems. A feature column is forbidden when, after stripping
# known family prefixes (sh_, dlt_ from exp-019) and pandas merge suffixes
# (_x/_y, up to twice), the remainder equals one of these stems -- so
# `sh_label_x` -> `label` (forbidden) while `bbox_center_x` -> `bbox_center`
# (legitimate). Matching is on the STEM, never on the _x/_y suffix alone.
META_STEMS = {
    "label", "crossing", "crossing_attr", "crossing_point", "decision_point",
    "motion_direction", "intention", "group", "split",
    "tte", "tte_frames", "tte_min", "tte_max", "time_to_event_frames",
    "frame_idx", "frame_index", "start_frame", "end_frame", "event_frame",
    "last_observed_frame", "event_frame_or_anchor",
    "track_id", "video_id", "ped_id", "sequence_id", "set_id", "track_key",
    "has_observed_crossing", "horizon_label",
}
FAMILY_PREFIXES = ("sh_", "dlt_")
MERGE_SUFFIX_RE = re.compile(r"_(x|y)$")
# Unambiguous label/future substrings: no legitimate feature name contains these.
FORBIDDEN_SUBSTRINGS = ("crossing_point", "decision_point", "motion_direction", "intention")


def forbidden_reason(col: str) -> str | None:
    c = col.lower()
    for bad in FORBIDDEN_SUBSTRINGS:
        if bad in c:
            return f"contains forbidden substring '{bad}'"
    if "label" in c.split("_"):
        return "contains token 'label'"
    candidates = {c}
    for pref in FAMILY_PREFIXES:
        if c.startswith(pref):
            candidates.add(c[len(pref):])
    expanded = set()
    for v in candidates:
        expanded.add(v)
        v1 = MERGE_SUFFIX_RE.sub("", v)
        expanded.add(v1)
        expanded.add(MERGE_SUFFIX_RE.sub("", v1))
    for v in sorted(expanded):
        if v in META_STEMS:
            return f"meta stem '{v}' after prefix/merge-suffix stripping"
    return None


# Known notes for columns that are expected to be flagged (flag, don't judge).
KNOWN_COLUMN_NOTES = {
    "attr_motion_lateral": (
        "encodes JAAD motion_direction==LAT (track-level, future-informed annotation); "
        "present BY DESIGN only in exp-016 'with motion' diagnostic sets; the honest "
        "*_no_motion sets exclude it (exp-016 documented caveat)"
    ),
    "attr_motion_longitudinal": (
        "encodes JAAD motion_direction==LONG (track-level, future-informed annotation); "
        "same disposition as attr_motion_lateral"
    ),
    "attr_signalized_s": "scene infrastructure effect -- may be legitimately strong; flag, don't judge",
    "attr_signalized_ns": "scene infrastructure effect -- may be legitimately strong; flag, don't judge",
    "attr_designated_crossing": "scene infrastructure (designated crossing point), exp-016 allowlisted; not the crossing label",
    "traffic_ped_crossing_frac": "scene infrastructure (crosswalk visible in observed frames), exp-016 allowlisted; not the crossing label",
}


# ---------------------------------------------------------------------------
# Per-column statistics
# ---------------------------------------------------------------------------

def _column_stats(x: pd.Series, y: np.ndarray) -> dict[str, Any]:
    xv = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(xv)
    out: dict[str, Any] = {
        "n_valid": int(valid.sum()),
        "auc": float("nan"),
        "auc_oriented": float("nan"),
        "spearman": float("nan"),
        "identical_to_label": False,
        "stat_note": "",
    }
    if valid.sum() < 10:
        out["stat_note"] = "insufficient non-NaN values (<10)"
        return out
    yv = np.asarray(y, dtype=int)[valid]
    xvv = xv[valid]
    if len(np.unique(yv)) < 2:
        out["stat_note"] = "single-class label among valid rows"
        return out
    if len(np.unique(xvv)) < 2:
        out["stat_note"] = "constant column"
        return out
    auc = float(roc_auc_score(yv, xvv))
    out["auc"] = auc
    out["auc_oriented"] = float(max(auc, 1.0 - auc))
    out["spearman"] = float(pd.Series(xvv).corr(pd.Series(yv.astype(float)), method="spearman"))
    out["identical_to_label"] = bool(np.array_equal(xvv, yv.astype(float)))
    return out


def audit_matrix(
    name: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame | None,
    y_test: np.ndarray | None,
    label_definition: str,
    summary_rows: list[dict],
    column_rows: list[dict],
    top_rows: list[dict],
    matrix_note: str = "",
) -> None:
    cols = list(X_train.columns)
    per_col = []
    for col in cols:
        reason = forbidden_reason(col)
        stats = _column_stats(X_train[col], y_train)
        test_auc = float("nan")
        if X_test is not None and y_test is not None and len(X_test):
            tstats = _column_stats(X_test[col], y_test)
            test_auc = tstats["auc_oriented"]
        perfect = bool(
            stats["identical_to_label"]
            or (np.isfinite(stats["spearman"]) and abs(abs(stats["spearman"]) - 1.0) < SPEARMAN_PERFECT_TOL)
        )
        flagged = bool(np.isfinite(stats["auc_oriented"]) and stats["auc_oriented"] >= AUC_FLAG_THRESHOLD)
        rec = {
            "matrix": name,
            "column": col,
            "forbidden": reason is not None,
            "forbidden_reason": reason or "",
            "n_train_valid": stats["n_valid"],
            "train_auc": stats["auc"],
            "train_auc_oriented": stats["auc_oriented"],
            "train_spearman_vs_label": stats["spearman"],
            "perfect_label_correlation": perfect,
            "flagged_auc_ge_0p95": flagged,
            "test_auc_oriented_AUDIT_ONLY": test_auc,
            "stat_note": stats["stat_note"],
            "known_note": KNOWN_COLUMN_NOTES.get(col, ""),
        }
        per_col.append(rec)
        column_rows.append(rec)

    ranked = sorted(
        [r for r in per_col if np.isfinite(r["train_auc_oriented"])],
        key=lambda r: r["train_auc_oriented"],
        reverse=True,
    )
    for rank, r in enumerate(ranked[:TOP_N], start=1):
        top_rows.append(
            {
                "matrix": name,
                "rank": rank,
                "column": r["column"],
                "train_auc": r["train_auc"],
                "train_auc_oriented": r["train_auc_oriented"],
                "n_train_valid": r["n_train_valid"],
                "test_auc_oriented_AUDIT_ONLY": r["test_auc_oriented_AUDIT_ONLY"],
                "flagged_auc_ge_0p95": r["flagged_auc_ge_0p95"],
                "forbidden": r["forbidden"],
                "known_note": r["known_note"],
            }
        )

    forbidden_cols = [r["column"] for r in per_col if r["forbidden"]]
    perfect_cols = [r["column"] for r in per_col if r["perfect_label_correlation"]]
    flagged_cols = [r["column"] for r in per_col if r["flagged_auc_ge_0p95"]]
    if forbidden_cols or perfect_cols:
        verdict = "FAIL"
    elif flagged_cols:
        verdict = "FLAGGED"
    else:
        verdict = "PASS"
    top = ranked[0] if ranked else None
    summary_rows.append(
        {
            "matrix": name,
            "n_train_rows": int(len(X_train)),
            "n_cols": len(cols),
            "label_definition": label_definition,
            "forbidden_cols_found": ";".join(forbidden_cols),
            "n_forbidden": len(forbidden_cols),
            "n_perfect_label_correlation": len(perfect_cols),
            "perfect_correlation_cols": ";".join(perfect_cols),
            "n_flagged_auc": len(flagged_cols),
            "flagged_cols": ";".join(flagged_cols),
            "top_feature": top["column"] if top else "",
            "top_auc": top["train_auc_oriented"] if top else float("nan"),
            "verdict": verdict,
            "note": matrix_note,
        }
    )
    print(f"[A] {name}: n_cols={len(cols)} verdict={verdict} "
          f"forbidden={forbidden_cols} flagged={flagged_cols} "
          f"top={top['column'] if top else ''}={top['train_auc_oriented']:.3f}" if top else f"[A] {name}: empty")


def split_entry(df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    tr = df[df["split"] == "train"]
    te = df[df["split"] == "test"]
    return tr[cols], tr["label"].astype(int).to_numpy(), te[cols], te["label"].astype(int).to_numpy()


# ---------------------------------------------------------------------------
# Sweep A matrix collections
# ---------------------------------------------------------------------------
JAAD_FS = ["bbox_only", "biu_v1_only", "biu_v1_plus_bbox", "biu_v2_core", "biu_v2_core_plus_bbox"]


def collect_jaad_base(summary, columns, tops) -> None:
    """JAAD 30_60 / 5_15 (+ broken archived 15_30 as consumed by exp 011) via exp-009 loaders."""
    t0 = time.time()
    feature_dict = pd.read_csv(base.DICT_PATH)
    horizons = ["30_60", "15_30", "5_15"]
    split_rows = {h: base._load_split_rows(h) for h in horizons}
    needed = base._needed_frame_keys(split_rows)
    kp_index = base._load_keypoint_index(base.HORIZON_CONFIGS["30_60"]["keypoints_jsonl"], needed)
    v2_rows = {}
    for h in horizons:
        v2_rows[h], _missing = base._extract_v2_for_rows(split_rows[h], kp_index)
    data_by_horizon, _defs, _cov = base._build_feature_sets(split_rows, v2_rows, feature_dict)
    print(f"[A] JAAD base matrices rebuilt in {time.time() - t0:.1f}s")
    name_map = {"30_60": "jaad_30_60", "15_30": "jaad_15_30_broken_archived", "5_15": "jaad_5_15"}
    for h in horizons:
        fmap = data_by_horizon[h]
        cmap = fmap["_column_map"]
        for fs in JAAD_FS:
            df, cols = fmap[fs], cmap[fs]
            Xtr, ytr, Xte, yte = split_entry(df, cols)
            note = "" if h != "15_30" else "archived BROKEN 15_30 table (silent 30_60 duplicate; superseded by exp 021) -- audited because exp 011 consumed it"
            audit_matrix(f"{name_map[h]}:{fs}", Xtr, ytr, Xte, yte,
                         "window label (crossing within TTE horizon)", summary, columns, tops, note)


def collect_corrected_15_30(summary, columns, tops) -> tuple[dict, dict, pd.DataFrame]:
    """Corrected 15_30 matrices via exp-021 benchmark loaders. Returns frames for Sweep B."""
    base_rows = c21.load_base_rows()
    frames, colmap = c21.build_feature_sets(base_rows)
    for fs in JAAD_FS:
        df, cols = frames[fs], colmap[fs]
        Xtr, ytr, Xte, yte = split_entry(df, cols)
        audit_matrix(f"jaad_15_30_corrected:{fs}", Xtr, ytr, Xte, yte,
                     "window label (crossing within TTE 15-30)", summary, columns, tops)
    return frames, colmap, base_rows


def collect_context_016(summary, columns, tops) -> pd.DataFrame:
    """Exp-016 context feature sets, rebuilt with the exp-016 module's own functions."""
    base_rows = ctx.load_splits()
    attr_index, _fields = ctx.load_attribute_index()
    traffic_index = ctx.load_traffic_index()
    context_records = []
    for _, rec in base_rows.iterrows():
        row = {
            "sequence_id": rec["sequence_id"],
            "split": rec["split"],
            "label": int(rec["label"]),
            "video_id": rec["video_id"],
            "ped_id": rec["ped_id"],
        }
        row.update(ctx.build_context_row(rec, attr_index, traffic_index))
        context_records.append(row)
    context_df = pd.DataFrame(context_records)
    assert len(context_df) == len(base_rows)

    bbox_df, bbox_cols = ctx.load_feature_table("bbox")
    biu_bbox_df, biu_bbox_cols = ctx.load_feature_table("biu_bbox")
    bbox_ctx_df = bbox_df.merge(context_df[["sequence_id"] + ctx.CONTEXT_COLS], on="sequence_id", how="inner")
    biu_bbox_ctx_df = biu_bbox_df.merge(context_df[["sequence_id"] + ctx.CONTEXT_COLS], on="sequence_id", how="inner")
    assert len(bbox_ctx_df) == len(bbox_df) and len(biu_bbox_ctx_df) == len(biu_bbox_df)

    feature_sets = {
        "bbox_only": (bbox_df, bbox_cols),
        "biu_bbox": (biu_bbox_df, biu_bbox_cols),
        "context_only": (context_df, ctx.CONTEXT_COLS),
        "bbox_plus_context": (bbox_ctx_df, sorted(set(bbox_cols + ctx.CONTEXT_COLS))),
        "biu_bbox_plus_context": (biu_bbox_ctx_df, sorted(set(biu_bbox_cols + ctx.CONTEXT_COLS))),
        "context_no_motion_only": (context_df, ctx.CONTEXT_NO_MOTION_COLS),
        "bbox_plus_context_no_motion": (bbox_ctx_df, sorted(set(bbox_cols + ctx.CONTEXT_NO_MOTION_COLS))),
        "biu_bbox_plus_context_no_motion": (biu_bbox_ctx_df, sorted(set(biu_bbox_cols + ctx.CONTEXT_NO_MOTION_COLS))),
    }
    for fs, (df, cols) in feature_sets.items():
        Xtr, ytr, Xte, yte = split_entry(df, cols)
        note = ""
        if fs in ("context_only", "bbox_plus_context", "biu_bbox_plus_context"):
            note = ("includes attr_motion_* (motion_direction encodings) BY DESIGN as exp-016 "
                    "documented future-informed diagnostic; honest variants are *_no_motion")
        audit_matrix(f"jaad_30_60_ctx016:{fs}", Xtr, ytr, Xte, yte,
                     "window label (crossing within TTE 30-60)", summary, columns, tops, note)
    return context_df


def collect_verifier_019(summary, columns, tops) -> None:
    """Exp-019 verifier variant matrices (post-fix), via the exp-019 module itself."""
    t0 = time.time()
    frames, _common_cols, track_info, _cov = v19.build_family_frames()
    tinfo = track_info.set_index("track_key")
    print(f"[A] exp-019 family frames rebuilt in {time.time() - t0:.1f}s")
    for variant in v19.VARIANTS:
        X = v19.build_variant_matrix(frames, "5_15", variant)
        info = tinfo.loc[X.index]
        pop = info["group"].isin(["actual_positive", "ambiguous_negative"])
        tr = ((info["split"] == "train") & pop).to_numpy()
        te = ((info["split"] == "test") & pop).to_numpy()
        y_tr = (info.loc[tr, "group"] == "actual_positive").astype(int).to_numpy()
        y_te = (info.loc[te, "group"] == "actual_positive").astype(int).to_numpy()
        audit_matrix(
            f"jaad_verifier019_5_15:{variant}", X[tr], y_tr, X[te], y_te,
            "verifier training label: actual_positive (1) vs ambiguous_negative (0), track level",
            summary, columns, tops,
            "post-fix exp-019 matrices; the historical sh_label_x leak entered here pre-fix",
        )


def collect_pie(summary, columns, tops) -> None:
    """PIE 30_60 matrices via exp-012 loaders (bbox / BIU v1 tables + BIU v2 extraction)."""
    t0 = time.time()
    feature_dict = pd.read_csv(pie.DICT_PATH)
    split_df = pie._load_split_rows()
    needed = pie._needed_frame_keys(split_df)
    kp_index = pie._load_keypoint_index(needed)
    print(f"[A] PIE keypoint index: {len(kp_index)} keys ({time.time() - t0:.1f}s)")
    t0 = time.time()
    bbox_index = pie._load_bbox_index(needed)
    print(f"[A] PIE bbox index: {len(bbox_index)} keys ({time.time() - t0:.1f}s)")
    v2_df_raw, _missing, _stats = pie._extract_v2_for_rows(split_df, kp_index, bbox_index)
    feature_sets, column_map, _defs = pie._build_feature_sets(split_df, v2_df_raw, feature_dict)
    for fs in pie.FEATURE_SET_ORDER:
        df, cols = feature_sets[fs], column_map[fs]
        Xtr, ytr, Xte, yte = split_entry(df, cols)
        audit_matrix(f"pie_30_60:{fs}", Xtr, ytr, Xte, yte,
                     "window label (crossing within TTE 30-60)", summary, columns, tops)


# ---------------------------------------------------------------------------
# Sweep A static merge audit
# ---------------------------------------------------------------------------
STATIC_FILES = [
    "jaad_biu_v2_benchmark_comparison.py",  # exp-009 base module: its loaders feed 011/019/021 reuse
    "jaad_staged_predictions_rescore.py",
    "jaad_staged_bbox_biu_v2.py",
    "jaad_context_features.py",
    "jaad_commitment_verifier.py",
    "jaad_confidence_gated_veto.py",
    "jaad_corrected_15_30_rebuild.py",
    "jaad_corrected_15_30_rebuild_benchmark.py",
    "jaad_corrected_15_30_rebuild_staged.py",
    "pie_biu_v2_30_60_benchmark.py",
    "calibration_analysis.py",
    "confidence_aware_biu_audit.py",
    "mslstm_retrain_benchmark.py",
    "pie_ambiguous_negative_intention_audit.py",
    "psi_keypoint_fidelity_study.py",
]

# Curated one-line verdicts, matched by (file, substring of the code line).
MERGE_VERDICTS: list[tuple[str, str, str]] = [
    ("jaad_biu_v2_benchmark_comparison.py", "pd.concat(dfs, ignore_index=True)",
     "safe: vertical concat of per-split CSVs -- adds rows, not columns; no collision possible"),
    ("jaad_biu_v2_benchmark_comparison.py", "base.merge(feat_df[keep]",
     "safe: right side keeps only sequence_id + columns ABSENT from base, so label/meta cannot collide (no _x/_y suffix can be created)"),
    ("jaad_biu_v2_benchmark_comparison.py", 'feature_sets["biu_v1_plus_bbox"]',
     "safe: right frame is sequence_id + numeric bbox feature cols only; overlapping cols dropped from left before merge"),
    ("jaad_biu_v2_benchmark_comparison.py", 'feature_sets["biu_v2_core_plus_bbox"]',
     "safe: right frame is sequence_id + numeric bbox feature cols only; overlapping cols dropped from left before merge"),
    ("jaad_staged_predictions_rescore.py", "out = pd.concat(",
     "safe: stacks val/test prediction rows (ID cols + prob); prediction output, not a feature frame"),
    ("jaad_staged_predictions_rescore.py", "parity_mean.merge(ref",
     "safe: metrics-vs-metrics parity join; no feature frame involved"),
    ("jaad_staged_bbox_biu_v2.py", 'canon.merge(assign, on="track_key"',
     "safe: adds exp-004 group column (evaluation subgrouping meta); this script fits no model and 'group' never enters a feature matrix"),
    ("jaad_staged_bbox_biu_v2.py", "pd.concat(frames, ignore_index=True)",
     "safe: vertical concat of per-horizon canonical meta rows"),
    ("jaad_staged_bbox_biu_v2.py", "cat = pd.concat(dfs, ignore_index=True)",
     "safe: stacks per-seed prediction CSVs before ensembling; predictions only"),
    ("jaad_staged_bbox_biu_v2.py", '.merge(ens[["sequence_id", "prob"]]',
     "safe: right side is sequence_id + prob only"),
    ("jaad_staged_bbox_biu_v2.py", "pd.concat(out, ignore_index=True)",
     "safe: vertical concat of track-table blocks; predictions only"),
    ("jaad_staged_bbox_biu_v2.py", 'joined = s1[["label", "group", "prob"]].rename',
     "safe: joins stage-1/stage-2 probabilities (renamed p1/p2); prediction fusion, not features"),
    ("jaad_context_features.py", "return pd.concat(dfs, ignore_index=True)",
     "safe: vertical concat of per-split CSVs"),
    ("jaad_context_features.py", "out = pd.concat(dfs, ignore_index=True)",
     "safe: vertical concat of per-split feature CSVs"),
    ("jaad_context_features.py", 'feat_df.merge(context_df[["sequence_id"] + CONTEXT_COLS]',
     "safe: right side restricted to sequence_id + the 20 context feature cols; label/split/ids excluded from right, so no collision"),
    ("jaad_context_features.py", "pred_rows = pd.concat([",
     "safe: stacks val/test prediction rows; output artifact"),
    ("jaad_commitment_verifier.py", '.merge(feat, on="sequence_id", how="inner")',
     "FIXED SITE (exp-019 label leak): meta columns incl. 'label' are now dropped from the feature table BEFORE this merge and a 'label'-substring filter guards cols; pre-fix this produced sh_label_x in the features"),
    ("jaad_commitment_verifier.py", 'out = out.join(p, how="inner")',
     "safe: joins prefix-disjoint feature families (dlt_*/sh_*); both sides pure feature frames"),
    ("jaad_commitment_verifier.py", 'meta[(meta["horizon_label"] == h) & (meta["split"] == split)].merge(',
     "safe: right side is sequence_id + prob only (continuation line)"),
    ("jaad_commitment_verifier.py", 'joined = s1_test[["label", "group", "prob"]].rename',
     "safe: joins stage probabilities p1/p2; prediction fusion, not features"),
    ("jaad_commitment_verifier.py", "pd.concat(selection_frames",
     "safe: stacks feature-selection ranking outputs"),
    ("jaad_commitment_verifier.py", "scores = pd.concat(score_frames",
     "safe: stacks verifier score outputs"),
    ("jaad_confidence_gated_veto.py", 'joined = s1[["label", "group", "prob"]].rename',
     "safe: joins stage probabilities p1/p2; prediction fusion, not features"),
    ("jaad_corrected_15_30_rebuild.py", "return pd.concat([pd.read_csv(split_dir",
     "safe: vertical concat of per-split CSVs"),
    ("jaad_corrected_15_30_rebuild.py", "prefilter_feat = pd.concat(",
     "safe: diagnosis-only concat of id/meta columns from archived pre-filter feature tables; never a feature frame"),
    ("jaad_corrected_15_30_rebuild.py", 'canon.merge(canon_30_60, on="track_id", suffixes=("_h", "_b"))',
     "creates _h/_b duplicates of meta cols BY DESIGN, but the result feeds only the canonical-overlap diagnostic CSV; never a model input"),
    ("jaad_corrected_15_30_rebuild_benchmark.py", "out = pd.concat(dfs, ignore_index=True)",
     "safe: vertical concat of per-split CSVs"),
    ("jaad_corrected_15_30_rebuild_benchmark.py", "return pd.concat(dfs, ignore_index=True)",
     "safe: vertical concat of per-split feature CSVs"),
    ("jaad_corrected_15_30_rebuild_benchmark.py", '"biu_v1_plus_bbox": biu_v1_df[',
     "safe: right frame is sequence_id + numeric bbox feature cols only; overlapping cols dropped from left first"),
    ("jaad_corrected_15_30_rebuild_benchmark.py", '"biu_v2_core_plus_bbox": v2_df[',
     "safe: right frame is sequence_id + numeric bbox feature cols only; overlapping cols dropped from left first"),
    ("jaad_corrected_15_30_rebuild_benchmark.py", "pd.concat(pred_rows, ignore_index=True)",
     "safe: stacks prediction rows; output artifact"),
    ("jaad_corrected_15_30_rebuild_staged.py", 'canon = canon.merge(assign, on="track_key"',
     "safe: adds exp-004 group column (evaluation meta); no model fitted on it"),
    ("jaad_corrected_15_30_rebuild_staged.py", "rows = pd.concat(dfs, ignore_index=True)",
     "safe: vertical concat of per-split CSVs"),
    ("jaad_corrected_15_30_rebuild_staged.py", "cat = pd.concat(dfs, ignore_index=True)",
     "safe: stacks per-seed prediction CSVs; predictions only"),
    ("jaad_corrected_15_30_rebuild_staged.py", 'meta_30_60.merge(ens[["sequence_id", "prob"]]',
     "safe: right side is sequence_id + prob only"),
    ("jaad_corrected_15_30_rebuild_staged.py", 'meta_cor.merge(ens[["sequence_id", "prob"]]',
     "safe: right side is sequence_id + prob only"),
    ("jaad_corrected_15_30_rebuild_staged.py", "return pd.concat(out, ignore_index=True)",
     "safe: vertical concat of track-table blocks; predictions only"),
    ("jaad_corrected_15_30_rebuild_staged.py", "cmp = old_ms[keep].merge(new_ms[keep]",
     "safe: metrics comparison with explicit _broken/_corrected suffixes; not a feature frame"),
    ("jaad_corrected_15_30_rebuild_staged.py", "pd.concat([old_staged, new_staged]",
     "safe: stacks metrics tables for the old-vs-corrected comparison CSV"),
    ("pie_biu_v2_30_60_benchmark.py", "out = pd.concat(dfs, ignore_index=True)",
     "safe: vertical concat of per-split CSVs"),
    ("pie_biu_v2_30_60_benchmark.py", "return pd.concat(dfs, ignore_index=True)",
     "safe: vertical concat of per-split feature CSVs"),
    ("pie_biu_v2_30_60_benchmark.py", "return base.merge(feat_df[keep]",
     "safe: right side keeps only sequence_id + columns ABSENT from base (same guard as the JAAD base module)"),
    ("pie_biu_v2_30_60_benchmark.py", "v1_plus_bbox_df = biu_v1_df[",
     "safe: right frame is sequence_id + numeric bbox feature cols only; overlapping cols dropped from left first"),
    ("pie_biu_v2_30_60_benchmark.py", "v2_plus_bbox_df = v2_df[",
     "safe: right frame is sequence_id + numeric bbox feature cols only; overlapping cols dropped from left first"),
    ("pie_biu_v2_30_60_benchmark.py", "pred_rows = pd.concat(",
     "safe: stacks val/test prediction rows; output artifact"),
    ("confidence_aware_biu_audit.py", 'merged.merge(df, on=["sequence_id", "split", "label", "group"]',
     "safe: per-seed prediction frames merged on identical id+label keys; only prob_seedN columns added; predictions only"),
    ("confidence_aware_biu_audit.py", 'test = ens[ens["split"] == "test"].merge(',
     "safe: adds one confidence-measure column (sequence_id + measure on the right, continuation line) to SAVED test predictions for quartile/abstention evaluation; not a training feature frame"),
    ("confidence_aware_biu_audit.py", "merged = merged.merge(conf_meta",
     "safe: conf_meta is sequence_id + 8 keypoint-confidence summary columns only (confidence_metadata_for_rows); no label/meta on the right; these ARE the intended features of the confidence variant"),
    ("confidence_aware_biu_audit.py", "joined = saved_test.merge(new",
     "safe: prob-vs-prob parity check between saved and recomputed predictions"),
    ("mslstm_retrain_benchmark.py", "out = pd.concat(dfs, ignore_index=True)",
     "safe: vertical concat of per-split CSVs"),
    ("mslstm_retrain_benchmark.py", "pred_rows = pd.concat([",
     "safe: stacks prediction rows; output artifact"),
    ("pie_ambiguous_negative_intention_audit.py", "merged = df.merge(",
     "attention but safe: joins 'intention_prob' and 'crossing' attributes onto SAVED test predictions for subgroup FPR evaluation only; script fits no model, so these label-adjacent columns never enter a feature matrix"),
]


def static_merge_audit() -> pd.DataFrame:
    site_re = re.compile(r"(pd\.merge\(|\.merge\(|pd\.concat\(|\.join\()")
    rows = []
    for fname in STATIC_FILES:
        path = SCRIPTS_DIR / fname
        n_sites = 0
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if "os.path.join" in line or ".str.join(" in line:
                continue
            for m in site_re.finditer(line):
                tok = m.group(1)
                if tok == ".join(" and m.start() > 0 and line[m.start() - 1] in "\"'":
                    continue  # string .join()
                kind = {"pd.merge(": "merge", ".merge(": "merge", "pd.concat(": "concat", ".join(": "join"}[tok]
                verdict = "UNREVIEWED -- needs manual review"
                for vf, sub, v in MERGE_VERDICTS:
                    if vf == fname and sub in line:
                        verdict = v
                        break
                rows.append({"file": f"scripts/experiments/{fname}", "line": lineno, "kind": kind,
                             "code": line.strip()[:160], "verdict": verdict})
                n_sites += 1
        if n_sites == 0:
            rows.append({"file": f"scripts/experiments/{fname}", "line": -1, "kind": "none",
                         "code": "", "verdict": "no pandas merge/join/concat sites in this script"})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Sweep B: proxy screen
# ---------------------------------------------------------------------------

def proxy_screen(context_df: pd.DataFrame, corrected_frames: dict, corrected_cols: dict) -> pd.DataFrame:
    rows = []
    ctx_notes = {d[0]: f"{d[1]}/{d[2]}: {d[3]}" for d in ctx.CONTEXT_FEATURE_DEFINITIONS}

    def screen(family: str, df: pd.DataFrame, cols: list[str], note_map: dict[str, str]) -> None:
        tr = df[df["split"] == "train"]
        y = tr["label"].astype(int).to_numpy()
        for col in cols:
            stats = _column_stats(tr[col], y)
            flagged = bool(np.isfinite(stats["auc_oriented"]) and stats["auc_oriented"] >= AUC_FLAG_THRESHOLD)
            note = KNOWN_COLUMN_NOTES.get(col, "") or note_map.get(col, "")
            if flagged and "flag" not in note.lower() and "future-informed" not in note.lower():
                note = (note + " | " if note else "") + "FLAG: train AUC >= 0.95 -- inspect before thesis use"
            rows.append({
                "family": family,
                "column": col,
                "n_train_valid": stats["n_valid"],
                "train_auc": stats["auc"],
                "train_auc_oriented": stats["auc_oriented"],
                "train_spearman_vs_label": stats["spearman"],
                "flagged_auc_ge_0p95": flagged,
                "stat_note": stats["stat_note"],
                "note": note,
            })

    screen("context_016_jaad_30_60", context_df, ctx.CONTEXT_COLS, ctx_notes)
    screen("corrected_15_30_bbox", corrected_frames["bbox_only"], corrected_cols["bbox_only"], {})
    screen("corrected_15_30_biu_v1", corrected_frames["biu_v1_only"], corrected_cols["biu_v1_only"], {})
    screen("corrected_15_30_biu_v2_core", corrected_frames["biu_v2_core"], corrected_cols["biu_v2_core"], {})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Sweep C: dataset invariant certificate
# ---------------------------------------------------------------------------

def _cert(rows: list[dict], table: str, check: str, status: str, measured: str, rule: str) -> None:
    rows.append({"table": table, "check": check, "status": status, "measured_value": measured, "rule": rule})
    print(f"[C] {table} | {check} | {status} | {measured}")


def certify_table(
    rows: list[dict],
    table: str,
    df: pd.DataFrame,
    tte_col: str,
    tte_lo: int,
    tte_hi: int,
    video_key_cols: list[str],
    check_crossing_attr: bool,
) -> None:
    # 1. TTE bounds (inclusive, matching each table's declared convention)
    tte = df[tte_col].astype(int)
    n_out = int((~tte.between(tte_lo, tte_hi)).sum())
    _cert(rows, table, f"tte_bounds_[{tte_lo},{tte_hi}]",
          "PASS" if n_out == 0 else "FAIL",
          f"min={int(tte.min())}, max={int(tte.max())}, out_of_range={n_out}, n={len(df)}",
          f"every {tte_col} in inclusive [{tte_lo},{tte_hi}]")

    # 2. Window strictly before event anchor.
    n_bad_last = int((df["last_observed_frame"] != df["end_frame"]).sum())
    n_bad_tte = int(((df["event_frame"] - df["end_frame"]) != tte).sum())
    n_bad_order = int((df["end_frame"] >= df["event_frame"]).sum())
    ok = n_bad_last == 0 and n_bad_tte == 0 and n_bad_order == 0
    _cert(rows, table, "window_before_event",
          "PASS" if ok else "FAIL",
          f"last_observed!=end_frame: {n_bad_last}; event-end!={tte_col}: {n_bad_tte}; end>=event: {n_bad_order}",
          "last_observed_frame == end_frame AND event_frame - end_frame == tte AND end_frame < event_frame (strict)")

    # 3. Split disjointness.
    tmulti = df.groupby("track_id")["split"].nunique()
    n_track_bad = int((tmulti > 1).sum())
    _cert(rows, table, "track_split_disjoint",
          "PASS" if n_track_bad == 0 else "FAIL",
          f"tracks_in_multiple_splits={n_track_bad} of {df['track_id'].nunique()}",
          "no track_id appears in more than one of train/val/test")
    vkey = df[video_key_cols].astype(str).agg(":".join, axis=1)
    vmulti = pd.DataFrame({"v": vkey, "split": df["split"]}).groupby("v")["split"].nunique()
    n_video_bad = int((vmulti > 1).sum())
    _cert(rows, table, "video_split_disjoint",
          "PASS" if n_video_bad == 0 else "FAIL",
          f"videos_in_multiple_splits={n_video_bad} of {vkey.nunique()} (key={'+'.join(video_key_cols)})",
          "no video appears in more than one split")

    # 5. Label consistency + class ratios.
    lmulti = df.groupby("track_id")["label"].nunique()
    n_label_bad = int((lmulti > 1).sum())
    _cert(rows, table, "label_constant_per_track",
          "PASS" if n_label_bad == 0 else "FAIL",
          f"tracks_with_mixed_labels={n_label_bad}",
          "all windows of a track carry the track's single label")
    if check_crossing_attr and "crossing_attr" in df.columns:
        known = df[df["crossing_attr"].notna()]
        mism = int(((known["label"] == 1) != (known["crossing_attr"] > 0)).sum())
        _cert(rows, table, "label_agrees_with_crossing_attr",
              "PASS" if mism == 0 else "FAIL",
              f"mismatches={mism} of {len(known)} rows with crossing_attr",
              "label==1 <=> crossing_attr>0 (JAAD builder semantics: positives from crossing tracks only)")
    ratios = df.groupby("split")["label"].agg(["count", "mean"])
    _cert(rows, table, "positive_ratio_per_split", "INFO",
          "; ".join(f"{s}: n={int(r['count'])}, pos_rate={r['mean']:.3f}" for s, r in ratios.iterrows()),
          "recorded, not judged")


def canonical_overlap_check(rows: list[dict], name: str, short_df: pd.DataFrame, base_df: pd.DataFrame,
                            expect_zero: bool, note: str) -> None:
    canon_s = c21r.canonical_windows(short_df)
    canon_b = c21r.canonical_windows(base_df)
    merged = canon_s.merge(canon_b, on="track_id", suffixes=("_s", "_b"))
    identical = int(((merged["start_frame_s"] == merged["start_frame_b"])
                     & (merged["end_frame_s"] == merged["end_frame_b"])).sum())
    frac = identical / len(merged) if len(merged) else float("nan")
    status = ("PASS" if identical == 0 else "FAIL") if expect_zero else "INFO"
    _cert(rows, name, "canonical_window_overlap_vs_30_60", status,
          f"identical={identical}/{len(merged)} tracks ({frac:.1%}){' -- ' + note if note else ''}",
          "exp-004 canonical rule (min tte_frames, then latest start_frame) per track; identical = same start_frame AND end_frame")


def broken_artifact_signature(rows: list[dict], name: str, df: pd.DataFrame, base_df: pd.DataFrame,
                              boundary_tte: int, note: str, expected_clean: bool = False) -> None:
    """Detector for the exp-021 'silent artifact' mechanism: the 30_60-only keypoint
    filter collapses a stage table to one window per track at the shared TTE boundary,
    every window byte-identical to a window of the 30_60 table."""
    all_boundary = bool((df["tte_frames"] == boundary_tte).all())
    key_cols = ["track_id", "start_frame", "end_frame"]
    base_keys = set(map(tuple, base_df[key_cols].itertuples(index=False)))
    contained = int(sum(tuple(r) in base_keys for r in df[key_cols].itertuples(index=False)))
    frac = contained / len(df) if len(df) else float("nan")
    max_win = int(df.groupby("track_id").size().max())
    is_artifact = all_boundary and frac == 1.0 and max_win == 1
    _cert(rows, name, "silent_artifact_signature",
          "FAIL" if is_artifact else ("PASS" if expected_clean else "INFO"),
          f"all_tte=={boundary_tte}: {all_boundary}; windows contained in 30_60 table: {contained}/{len(df)} "
          f"({frac:.1%}); max windows/track: {max_win}{' -- ' + note if note else ''}",
          "artifact iff every window sits at the shared TTE boundary, duplicates a 30_60-table window "
          "(same track_id/start_frame/end_frame) and each track has exactly one window")


def sweep_c(corrected_all: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    jaad_30_60 = base._load_split_rows("30_60")
    jaad_5_15 = base._load_split_rows("5_15")
    pie_df = pie._load_split_rows()
    broken_15_30 = c21r._read_split_table(c21r.BROKEN_SPLIT_DIR)
    broken_60_90 = c21r._read_split_table(c21r.T60_90_SPLIT_DIR)

    certify_table(rows, "jaad_30_60", jaad_30_60, "tte_frames", 30, 60, ["video_id"], True)
    certify_table(rows, "jaad_5_15", jaad_5_15, "tte_frames", 5, 15, ["video_id"], True)
    certify_table(rows, "jaad_15_30_corrected", corrected_all, "tte_frames", 15, 30, ["video_id"], True)
    certify_table(rows, "pie_30_60", pie_df, "time_to_event_frames", 30, 60, ["set_id", "video_id"], False)

    # PIE set-based split definition.
    expected_sets = {"train": {"set01", "set02", "set06"}, "val": {"set04", "set05"}, "test": {"set03"}}
    bad = []
    for split, sets in expected_sets.items():
        actual = set(pie_df.loc[pie_df["split"] == split, "set_id"].astype(str).unique())
        if actual != sets:
            bad.append(f"{split}: {sorted(actual)} != {sorted(sets)}")
    _cert(rows, "pie_30_60", "set_based_split_definition",
          "PASS" if not bad else "FAIL",
          "; ".join(bad) if bad else "train={set01,set02,set06}, val={set04,set05}, test={set03}",
          "PIE splits are set-based: set03 test, set01/02/06 train, set04/05 val")

    # 4. Cross-horizon canonical-window overlap vs the 30_60 table.
    canonical_overlap_check(rows, "jaad_15_30_corrected", corrected_all, jaad_30_60, True,
                            "021 verified 0% -- reconfirmed independently here")
    canonical_overlap_check(rows, "jaad_5_15", jaad_5_15, jaad_30_60, True, "")
    canonical_overlap_check(rows, "jaad_15_30_broken_archived(reference)", broken_15_30, jaad_30_60, True,
                            "KNOWN-BROKEN archived artifact, superseded by exp 021; expected to FAIL")
    canonical_overlap_check(rows, "jaad_60_90_broken_archived(reference)", broken_60_90, jaad_30_60, False,
                            "canonical-overlap is NOT the right detector for 60_90 (its duplicate windows sit at "
                            "tte 60, while the canonical 30_60 window is min-tte 30); see silent_artifact_signature")
    broken_artifact_signature(rows, "jaad_15_30_broken_archived(reference)", broken_15_30, jaad_30_60, 30,
                              "KNOWN-BROKEN archived artifact, superseded by exp 021; expected to FAIL")
    broken_artifact_signature(rows, "jaad_60_90_broken_archived(reference)", broken_60_90, jaad_30_60, 60,
                              "KNOWN-BROKEN archived artifact (same keypoint-filter mechanism); expected to FAIL")
    # The certified thesis tables must NOT carry the artifact signature.
    broken_artifact_signature(rows, "jaad_15_30_corrected", corrected_all, jaad_30_60, 30,
                              "corrected table must NOT be an artifact", expected_clean=True)
    broken_artifact_signature(rows, "jaad_5_15", jaad_5_15, jaad_30_60, 15,
                              "5_15 table must NOT be an artifact", expected_clean=True)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    summary: list[dict] = []
    columns: list[dict] = []
    tops: list[dict] = []

    def flush() -> None:
        pd.DataFrame(summary).to_csv(METRICS_DIR / "feature_matrix_audit.csv", index=False)
        pd.DataFrame(columns).to_csv(METRICS_DIR / "feature_matrix_audit_columns.csv", index=False)
        pd.DataFrame(tops).to_csv(METRICS_DIR / "standalone_auc_top10.csv", index=False)

    # ---- static merge audit (instant)
    static_df = static_merge_audit()
    static_df.to_csv(METRICS_DIR / "merge_static_audit.csv", index=False)
    n_unrev = int((static_df["verdict"].str.startswith("UNREVIEWED")).sum())
    print(f"[static] {len(static_df)} rows, unreviewed={n_unrev}")

    # ---- Sweep A + B inputs, cheapest first
    corrected_frames, corrected_cols, _corrected_base_rows = collect_corrected_15_30(summary, columns, tops)
    flush()
    context_df = collect_context_016(summary, columns, tops)
    flush()

    # ---- Sweep B
    proxy_df = proxy_screen(context_df, corrected_frames, corrected_cols)
    proxy_df.to_csv(METRICS_DIR / "context_proxy_screen.csv", index=False)
    print(f"[B] context_proxy_screen rows: {len(proxy_df)}, "
          f"flagged: {int(proxy_df['flagged_auc_ge_0p95'].sum())}")

    # ---- Sweep C (needs only split tables)
    corrected_all = pd.concat(
        [pd.read_csv(c21r.DATA_DIR / f"{s}.csv").assign(split=s) for s in c21r.SPLITS], ignore_index=True
    )
    cert_df = sweep_c(corrected_all)
    cert_df.to_csv(METRICS_DIR / "invariant_certificate.csv", index=False)

    # ---- Sweep A heavy collections
    collect_jaad_base(summary, columns, tops)
    flush()
    collect_verifier_019(summary, columns, tops)
    flush()
    collect_pie(summary, columns, tops)
    flush()

    sdf = pd.DataFrame(summary)
    print(json.dumps({
        "matrices_audited": int(len(sdf)),
        "sweep_a_verdicts": sdf["verdict"].value_counts().to_dict(),
        "sweep_a_fail_matrices": sdf.loc[sdf["verdict"] == "FAIL", "matrix"].tolist(),
        "sweep_a_flagged_matrices": sdf.loc[sdf["verdict"] == "FLAGGED", "matrix"].tolist(),
        "static_sites": int(len(static_df)),
        "static_unreviewed": n_unrev,
        "proxy_rows": int(len(proxy_df)),
        "certificate_status_counts": cert_df["status"].value_counts().to_dict(),
        "certificate_failures": cert_df.loc[cert_df["status"] == "FAIL", ["table", "check"]].to_dict(orient="records"),
    }, indent=2))


if __name__ == "__main__":
    main()
