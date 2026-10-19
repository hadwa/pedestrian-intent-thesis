"""Experiment 021 step 1-3: diagnose, rebuild and re-featurize the JAAD 15_30 stage table.

Background: the archived stage-horizon split table
`<archive>/data/jaad_splits_stage_horizon_audit/tte_15_30/` is broken -- all 275
rows have tte_frames == 30 (one window per track), byte-identical to the 30_60
benchmark's canonical windows (experiments/019 verified 275/275 identical).

This script:
1. DIAGNOSES: reproduces the archive builder's candidate-window generation
   (copied verbatim from
   `<archive>/src/analysis/staged_prediction_hypothesis_audit.py`:
   `_jaad_window_stride`, `_jaad_window_end_frames`, `_jaad_build_sample` and
   the positive/negative loop of `build_jaad_horizon_splits`;
   `load_jaad_tracks` is imported from
   `<archive>/src/benchmark/build_jaad_event_aligned_candidates.py`), then
   shows empirically that exact-keypoint alignment against the ORIGINAL
   truncated JSONL (`jaad_keypoints_vitpose_s.jsonl`, extracted only for the
   30_60 benchmark windows) collapses the correct multi-window 15_30 candidate
   set to exactly the archived 275 tte==30 rows.
2. REBUILDS: the corrected tte_15_30 table with the same protocol as the
   working tte_5_15 table (16 consecutive observed frames, overlap 0.8 /
   window stride 3, inclusive TTE range [15, 30] mirroring the inclusive
   [5, 15] convention, no observed cross==1 frames, negatives anchored at the
   last visible frame) filtered against the EXTENDED JSONL
   (`jaad_keypoints_vitpose_s_extended_short_horizon.jsonl`).
   Split membership comes from the same JAAD `split_ids/default` video split
   used by every existing table (no re-splitting).
3. VALIDATES: canonical-window overlap vs 30_60 (019-style CSV), split
   disjointness, split-membership consistency vs the 30_60/5_15 tables,
   counts vs the broken table, keypoint coverage.
4. FEATURES: bbox (53-feature) and BIU v1 (72-feature) tables via the archive
   builders (`src/features/build_jaad_bbox_feature_tables.build_tables`,
   `src/features/build_jaad_kp_biu_feature_tables.build_all`), and BIU v2
   features via the exp-009 machinery
   (`scripts/experiments/jaad_biu_v2_benchmark_comparison`).

Everything is written inside experiments/021_jaad_corrected_15_30_rebuild/;
the archive repository is never written to.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import numpy as np
import pandas as pd

CLEAN_ROOT = Path("/home/abdelrahman/Hadwa/BIU-PID-Distillation-clean")
ARCHIVE_ROOT = Path("/home/abdelrahman/Hadwa/BIU-PID-Distillation")
JAAD_DATASET_ROOT = Path("/home/abdelrahman/Hadwa/datasets/JAAD")

if str(ARCHIVE_ROOT) not in sys.path:
    sys.path.insert(0, str(ARCHIVE_ROOT))
SCRIPTS_DIR = CLEAN_ROOT / "scripts/experiments"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from src.benchmark.build_jaad_event_aligned_candidates import TrackRecord, load_jaad_tracks  # noqa: E402
from src.features.build_jaad_bbox_feature_tables import build_tables as build_jaad_bbox_tables  # noqa: E402
from src.features.build_jaad_kp_biu_feature_tables import build_all as build_jaad_kp_biu_tables  # noqa: E402

import jaad_biu_v2_benchmark_comparison as base  # noqa: E402  (exp-009 module)

OUT_ROOT = CLEAN_ROOT / "experiments/021_jaad_corrected_15_30_rebuild"
DATA_DIR = OUT_ROOT / "data/tte_15_30_corrected"
FEAT_DIR = OUT_ROOT / "data/features_tte_15_30_corrected"
METRICS_DIR = OUT_ROOT / "metrics"
LOGS_DIR = OUT_ROOT / "logs"

OLD_JSONL = ARCHIVE_ROOT / "data/jaad_keypoints_vitpose_s.jsonl"
EXT_JSONL = ARCHIVE_ROOT / "data/jaad_keypoints_vitpose_s_extended_short_horizon.jsonl"

BROKEN_SPLIT_DIR = ARCHIVE_ROOT / "data/jaad_splits_stage_horizon_audit/tte_15_30"
BROKEN_FEATURE_DIR = ARCHIVE_ROOT / "data/jaad_features_stage_horizon_audit/tte_15_30"
T5_15_SPLIT_DIR = ARCHIVE_ROOT / "data/jaad_splits_stage_horizon_audit/tte_5_15"
T60_90_SPLIT_DIR = ARCHIVE_ROOT / "data/jaad_splits_stage_horizon_audit/tte_60_90"
MAIN_30_60_SPLIT_DIR = ARCHIVE_ROOT / "data/jaad_splits_event_16_30_60_overlap08_default"
REF_5_15_FEATURE_DIR = ARCHIVE_ROOT / "data/jaad_features_stage_horizon_audit/tte_5_15"

SPLITS = ["train", "val", "test"]
TTE_MIN, TTE_MAX = 15, 30  # inclusive, mirrors the 5_15 table ([5, 15] inclusive)
OVERLAP = 0.8


# ---------------------------------------------------------------------------
# Builder logic copied verbatim from
# <archive>/src/analysis/staged_prediction_hypothesis_audit.py
# (_jaad_window_stride, _jaad_window_end_frames, _jaad_build_sample,
#  candidate loop of build_jaad_horizon_splits, _load_jaad_keypoint_frame_index)
# ---------------------------------------------------------------------------

def _jaad_window_stride(overlap: float) -> int:
    return max(1, int(round(16 * (1.0 - overlap))))


def _jaad_window_end_frames(track: TrackRecord, event_frame: int, tte_min: int, tte_max: int, window_stride: int) -> list[int]:
    earliest_end = max(track.frames[0] + 16 - 1, event_frame - tte_max)
    latest_end = event_frame - tte_min
    if earliest_end > latest_end:
        return []
    return list(range(earliest_end, latest_end + 1, int(window_stride)))


def _jaad_build_sample(track: TrackRecord, event_frame: int, end_frame: int, label: int, onset_source: str, negative_alignment_source: str, tte_min: int, tte_max: int, window_stride: int) -> dict[str, Any] | None:
    start_frame = end_frame - 16 + 1
    observed_frames = list(range(start_frame, end_frame + 1))
    if any(frame not in track.frame_to_cross for frame in observed_frames):
        return None
    observed_crosses = [track.frame_to_cross[frame] for frame in observed_frames]
    if any(cross == 1 for cross in observed_crosses):
        return None
    bbox_sequence = [track.frame_to_bbox[frame] for frame in observed_frames]
    tte_frames = event_frame - end_frame
    if not (tte_min <= tte_frames <= tte_max):
        return None
    sequence_id = f"{track.track_id}:{start_frame}:{end_frame}:{event_frame}:{label}"
    return {
        "sequence_id": sequence_id,
        "split": track.split,
        "video_id": track.video_id,
        "ped_id": track.ped_id,
        "track_id": track.track_id,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "event_frame": event_frame,
        "last_observed_frame": end_frame,
        "tte_frames": tte_frames,
        "label": label,
        "observation_length": 16,
        "frame_step": 1,
        "sample_stride_between_windows": int(window_stride),
        "frame_sampling": "consecutive",
        "onset_source": onset_source,
        "negative_alignment_source": negative_alignment_source,
        "track_length": track.track_length,
        "crossing_attr": track.crossing_attr,
        "crossing_point": track.crossing_point,
        "decision_point": track.decision_point,
        "has_observed_crossing": 0,
        "observed_frames_json": json.dumps(observed_frames),
        "observed_crosses_json": json.dumps(observed_crosses),
        "bbox_sequence_json": json.dumps(bbox_sequence),
        "source_annotation_path": track.source_annotation_path,
        "source_attributes_path": track.source_attributes_path,
        "parser_mode": "jaad_raw_annotations",
    }


def build_candidates(tte_min: int, tte_max: int, overlap: float) -> tuple[pd.DataFrame, dict[str, set[str]]]:
    tracks, split_videos = load_jaad_tracks(JAAD_DATASET_ROOT, "default")
    window_stride = _jaad_window_stride(overlap)
    samples: list[dict[str, Any]] = []
    for track in tracks:
        if track.crossing_attr is None:
            continue
        if track.crossing_attr > 0:
            if track.starts_already_crossing or len(track.transition_frames) > 1 or not track.is_continuous or track.track_length < 16:
                continue
            onset_frame: int | None = None
            onset_source = ""
            if len(track.transition_frames) == 1:
                onset_frame = track.transition_frames[0]
                onset_source = "frame_cross_transition"
            else:
                cp = track.crossing_point
                if cp is None or cp < 0 or cp < track.frames[0] or cp > track.frames[-1]:
                    continue
                onset_frame = cp
                onset_source = "crossing_point_fallback"
            for end_frame in _jaad_window_end_frames(track, onset_frame, tte_min, tte_max, window_stride):
                sample = _jaad_build_sample(track, onset_frame, end_frame, 1, onset_source, "", tte_min, tte_max, window_stride)
                if sample is not None:
                    samples.append(sample)
        else:
            if not track.is_continuous or track.track_length < 16 or track.starts_already_crossing or track.transition_frames or any(cross == 1 for cross in track.crosses):
                continue
            event_frame = track.frames[-1]
            for end_frame in _jaad_window_end_frames(track, event_frame, tte_min, tte_max, window_stride):
                sample = _jaad_build_sample(track, event_frame, end_frame, 0, "", "last_visible_frame", tte_min, tte_max, window_stride)
                if sample is not None:
                    samples.append(sample)
    df = pd.DataFrame(samples).sort_values(["split", "video_id", "track_id", "start_frame"]).reset_index(drop=True)
    return df, split_videos


def _load_keypoint_frame_index(jsonl_path: Path) -> tuple[set[tuple[str, str]], dict[tuple[str, str], set[int]]]:
    track_ids: set[tuple[str, str]] = set()
    track_frames: dict[tuple[str, str], set[int]] = {}
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            video_id = str(record.get("video", ""))
            ped_id = str(record.get("pid", ""))
            frame_idx = record.get("frame_idx")
            if video_id and ped_id and frame_idx is not None:
                key = (video_id, ped_id)
                track_ids.add(key)
                track_frames.setdefault(key, set()).add(int(frame_idx))
    return track_ids, track_frames


def apply_keypoint_filter(df: pd.DataFrame, jsonl_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Exact-frame keypoint alignment filter, same rule as the archive builder."""
    kp_tracks, kp_frames = _load_keypoint_frame_index(jsonl_path)
    keep, reasons = [], []
    for record in df.to_dict(orient="records"):
        key = (str(record["video_id"]), str(record["ped_id"]))
        if key not in kp_tracks:
            keep.append(False)
            reasons.append("missing_track_in_keypoints")
            continue
        observed = json.loads(record["observed_frames_json"])
        available = kp_frames.get(key, set())
        if any(int(f) not in available for f in observed):
            keep.append(False)
            reasons.append("missing_exact_observed_frames")
            continue
        keep.append(True)
        reasons.append("")
    mask = pd.Series(keep, index=df.index)
    dropped = df.loc[~mask].copy()
    if not dropped.empty:
        dropped["drop_reason"] = [r for k, r in zip(keep, reasons) if not k]
    kept = df.loc[mask].copy().sort_values(["split", "video_id", "track_id", "start_frame"]).reset_index(drop=True)
    return kept, dropped


# ---------------------------------------------------------------------------
# Diagnosis helpers
# ---------------------------------------------------------------------------

def _read_split_table(split_dir: Path) -> pd.DataFrame:
    return pd.concat([pd.read_csv(split_dir / f"{s}.csv").assign(split=s) for s in SPLITS], ignore_index=True)


def _table_stats(name: str, df: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for split in SPLITS + ["all"]:
        sub = df if split == "all" else df[df["split"] == split]
        if sub.empty:
            continue
        rows.append(
            {
                "table": name,
                "split": split,
                "n_rows": int(len(sub)),
                "n_tracks": int(sub["track_id"].nunique()),
                "n_videos": int(sub["video_id"].nunique()),
                "n_positive": int((sub["label"] == 1).sum()),
                "positive_rate": round(float(sub["label"].mean()), 4),
                "tte_min": int(sub["tte_frames"].min()),
                "tte_max": int(sub["tte_frames"].max()),
                "max_windows_per_track": int(sub.groupby("track_id").size().max()),
                "tte_value_counts": json.dumps({int(k): int(v) for k, v in sub["tte_frames"].value_counts().sort_index().items()}),
            }
        )
    return rows


def canonical_windows(df: pd.DataFrame) -> pd.DataFrame:
    """exp-004/011 canonical rule: per track minimum tte_frames, then latest start_frame."""
    ordered = df.sort_values(["track_id", "tte_frames", "start_frame"], ascending=[True, True, False])
    return ordered.groupby("track_id", as_index=False).first()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    for d in [DATA_DIR, FEAT_DIR, METRICS_DIR, LOGS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # ---------- STEP 1: diagnosis ----------
    broken = _read_split_table(BROKEN_SPLIT_DIR)
    t5_15 = _read_split_table(T5_15_SPLIT_DIR)
    t60_90 = _read_split_table(T60_90_SPLIT_DIR)
    main_30_60 = _read_split_table(MAIN_30_60_SPLIT_DIR)

    stats_rows: list[dict[str, Any]] = []
    for name, df in [
        ("archived_tte_15_30_broken", broken),
        ("archived_tte_5_15", t5_15),
        ("archived_tte_60_90", t60_90),
        ("archived_main_30_60", main_30_60),
    ]:
        stats_rows.extend(_table_stats(name, df))

    # archived pre-filter feature tables for 15_30 (built 2026-06-25, before the
    # split CSVs were overwritten on 2026-06-26) retain the full window set
    prefilter_feat = pd.concat(
        [pd.read_csv(BROKEN_FEATURE_DIR / f"{s}_bbox.csv")[["sequence_id", "split", "label", "video_id", "track_id", "start_frame", "end_frame"]] for s in SPLITS],
        ignore_index=True,
    )
    parts = prefilter_feat["sequence_id"].str.split(":")
    prefilter_feat["tte_frames"] = parts.str[-2].astype(int) - parts.str[-3].astype(int)
    stats_rows.extend(_table_stats("archived_tte_15_30_prefilter_feature_tables", prefilter_feat))

    # regenerate candidates with the archive builder logic
    candidates, split_videos = build_candidates(TTE_MIN, TTE_MAX, OVERLAP)
    stats_rows.extend(_table_stats("regenerated_candidates_tte_15_30", candidates))
    pd.DataFrame(stats_rows).to_csv(METRICS_DIR / "diagnosis_tte_distribution.csv", index=False)

    # mechanism verification
    cand_ids = set(candidates["sequence_id"])
    prefilter_ids = set(prefilter_feat["sequence_id"])
    broken_ids = set(broken["sequence_id"])

    kept_old, dropped_old = apply_keypoint_filter(candidates, OLD_JSONL)
    kept_old_ids = set(kept_old["sequence_id"])

    verification = [
        {"check": "regenerated_candidates_equal_archived_prefilter_feature_rows",
         "n_left": len(cand_ids), "n_right": len(prefilter_ids),
         "n_common": len(cand_ids & prefilter_ids),
         "identical": cand_ids == prefilter_ids},
        {"check": "candidates_filtered_by_ORIGINAL_truncated_jsonl_equal_archived_broken_split",
         "n_left": len(kept_old_ids), "n_right": len(broken_ids),
         "n_common": len(kept_old_ids & broken_ids),
         "identical": kept_old_ids == broken_ids},
        {"check": "old_jsonl_filter_drop_count",
         "n_left": int(len(dropped_old)), "n_right": len(cand_ids) - len(broken_ids),
         "n_common": -1,
         "identical": len(dropped_old) == len(cand_ids) - len(broken_ids)},
        {"check": "surviving_windows_all_tte_30_under_old_jsonl",
         "n_left": int((kept_old["tte_frames"] == 30).sum()), "n_right": int(len(kept_old)),
         "n_common": -1,
         "identical": bool((kept_old["tte_frames"] == 30).all())},
    ]
    pd.DataFrame(verification).to_csv(METRICS_DIR / "diagnosis_rebuild_verification.csv", index=False)
    print("verification:", json.dumps(verification, indent=2, default=str))

    # ---------- STEP 2: corrected table ----------
    corrected, dropped_ext = apply_keypoint_filter(candidates, EXT_JSONL)
    corrected.to_csv(DATA_DIR / "all_samples.csv", index=False)
    for split in SPLITS:
        corrected.loc[corrected["split"] == split].to_csv(DATA_DIR / f"{split}.csv", index=False)

    metadata = {
        "tte_min": TTE_MIN,
        "tte_max": TTE_MAX,
        "tte_range_convention": "inclusive [15, 30], mirroring the archived tte_5_15 table (inclusive [5, 15]; contains tte=5 and tte=15 rows)",
        "overlap": OVERLAP,
        "window_stride": _jaad_window_stride(OVERLAP),
        "keypoint_jsonl": str(EXT_JSONL),
        "builder_source": "copied from <archive>/src/analysis/staged_prediction_hypothesis_audit.py::build_jaad_horizon_splits (verbatim window/candidate logic); load_jaad_tracks imported from <archive>/src/benchmark/build_jaad_event_aligned_candidates.py",
        "split_source": "JAAD split_ids/default video split via load_jaad_tracks -- identical membership rule to the 30_60 benchmark tables; no re-splitting",
        "num_rows": int(len(corrected)),
        "num_positive": int((corrected["label"] == 1).sum()),
        "num_negative": int((corrected["label"] == 0).sum()),
        "track_counts": {s: int(corrected.loc[corrected["split"] == s, "track_id"].nunique()) for s in SPLITS},
        "split_video_counts": {s: int(len(split_videos[s])) for s in SPLITS},
        "keypoint_alignment": {
            "dropped_rows": int(len(dropped_ext)),
            "dropped_tracks": int(dropped_ext["track_id"].nunique()) if not dropped_ext.empty else 0,
            "dropped_rows_by_reason": dropped_ext["drop_reason"].value_counts().to_dict() if not dropped_ext.empty else {},
        },
    }
    (DATA_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # ---------- STEP 2 validation ----------
    canon_30_60 = canonical_windows(main_30_60)
    canon_corrected = canonical_windows(corrected)
    canon_broken = canonical_windows(broken)

    overlap_rows = []
    for name, canon in [("corrected_15_30", canon_corrected), ("broken_15_30", canon_broken)]:
        merged = canon.merge(canon_30_60, on="track_id", suffixes=("_h", "_b"))
        identical = ((merged["start_frame_h"] == merged["start_frame_b"]) & (merged["end_frame_h"] == merged["end_frame_b"])).sum()
        overlap_rows.append(
            {
                "short_horizon": name,
                "n_tracks_both_horizons": int(len(merged)),
                "n_identical_canonical_window": int(identical),
                "frac_identical": round(float(identical / len(merged)), 4) if len(merged) else float("nan"),
                "canonical_tte_min": int(canon["tte_frames"].min()),
                "canonical_tte_max": int(canon["tte_frames"].max()),
                "canonical_tte_value_counts": json.dumps({int(k): int(v) for k, v in canon["tte_frames"].value_counts().sort_index().items()}),
            }
        )
    pd.DataFrame(overlap_rows).to_csv(METRICS_DIR / "canonical_window_overlap_corrected.csv", index=False)

    checks: list[dict[str, Any]] = []

    def add_check(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    track_splits = corrected.groupby("track_id")["split"].nunique()
    add_check("track_split_disjoint", bool((track_splits == 1).all()), f"tracks in >1 split: {int((track_splits > 1).sum())}")
    video_splits = corrected.groupby("video_id")["split"].nunique()
    add_check("video_split_disjoint", bool((video_splits == 1).all()), f"videos in >1 split: {int((video_splits > 1).sum())}")

    for ref_name, ref_df in [("main_30_60", main_30_60), ("tte_5_15", t5_15), ("broken_15_30", broken)]:
        ref_map = ref_df.drop_duplicates("track_id").set_index("track_id")["split"]
        cor_map = corrected.drop_duplicates("track_id").set_index("track_id")["split"]
        common = ref_map.index.intersection(cor_map.index)
        mismatches = int((ref_map.loc[common] != cor_map.loc[common]).sum())
        add_check(f"split_membership_consistent_vs_{ref_name}", mismatches == 0, f"common tracks: {len(common)}, split mismatches: {mismatches}")

    add_check("corrected_canonical_tte_below_30", bool((canon_corrected["tte_frames"] < 30).all()),
              f"canonical tte range: [{int(canon_corrected['tte_frames'].min())}, {int(canon_corrected['tte_frames'].max())}]")
    n_boundary = int((corrected["tte_frames"] == 30).sum())
    shared_with_30_60 = len(set(corrected["sequence_id"]) & set(main_30_60["sequence_id"]))
    add_check("boundary_tte30_windows_documented", True,
              f"windows at tte==30 (shared boundary with inclusive 30_60 range): {n_boundary}; "
              f"sequence_ids also present in the 30_60 table: {shared_with_30_60} "
              f"(protocol-faithful consequence of mirroring the inclusive-range convention; canonical windows are all tte<30)")
    add_check("keypoint_coverage_extended_jsonl", len(dropped_ext) == 0,
              f"windows dropped for missing keypoint frames under extended JSONL: {len(dropped_ext)} of {len(candidates)}")
    add_check("broken_table_windows_recovered", broken_ids <= set(corrected["sequence_id"]),
              f"broken-table windows contained in corrected table: {len(broken_ids & set(corrected['sequence_id']))}/{len(broken_ids)}")
    pd.DataFrame(checks).to_csv(METRICS_DIR / "validation_checks.csv", index=False)
    print(pd.DataFrame(checks).to_string())

    # counts comparison broken vs corrected
    count_rows = _table_stats("broken_15_30", broken) + _table_stats("corrected_15_30", corrected)
    pd.DataFrame(count_rows).to_csv(METRICS_DIR / "validation_counts_broken_vs_corrected.csv", index=False)

    # ---------- STEP 3: features ----------
    build_jaad_bbox_tables(splits_dir=DATA_DIR, output_dir=FEAT_DIR, report_path=LOGS_DIR / "bbox_feature_report.md")
    build_jaad_kp_biu_tables(
        jsonl_path=EXT_JSONL,
        splits_dir=DATA_DIR,
        output_dir=FEAT_DIR,
        audit_report_path=LOGS_DIR / "kp_biu_feature_report.md",
        bbox_reference_summary_path=None,
    )

    # BIU v2 via exp-009 machinery
    split_rows = {}
    for split in SPLITS:
        df = pd.read_csv(DATA_DIR / f"{split}.csv")
        df["split"] = split
        df["horizon_label"] = "15_30_corrected"
        df["group"] = df.apply(base._classify_group, axis=1)
        split_rows[split] = df
    needed = base._needed_frame_keys(split_rows)
    kp_index = base._load_keypoint_index(EXT_JSONL, needed)
    v2_missing_all = []
    for split, df in split_rows.items():
        v2_df, missing = base._extract_v2_for_rows(df, kp_index)
        v2_df.to_csv(FEAT_DIR / f"{split}_biu_v2.csv", index=False)
        v2_missing_all.extend(missing)
    pd.DataFrame(v2_missing_all).to_csv(METRICS_DIR / "biu_v2_missing_keypoint_rows.csv", index=False)

    # feature-column parity vs archived working tables (tte_5_15)
    col_rows = []
    for name in ["bbox", "biu"]:
        ref_cols = list(pd.read_csv(REF_5_15_FEATURE_DIR / f"train_{name}.csv", nrows=1).columns)
        new_cols = list(pd.read_csv(FEAT_DIR / f"train_{name}.csv", nrows=1).columns)
        col_rows.append(
            {
                "table": name,
                "n_ref_cols": len(ref_cols),
                "n_new_cols": len(new_cols),
                "columns_identical": ref_cols == new_cols,
                "missing_vs_ref": json.dumps(sorted(set(ref_cols) - set(new_cols))),
                "extra_vs_ref": json.dumps(sorted(set(new_cols) - set(ref_cols))),
            }
        )
    for split in SPLITS:
        n_split = len(split_rows[split])
        for name in ["bbox", "biu", "biu_v2"]:
            n_feat = len(pd.read_csv(FEAT_DIR / f"{split}_{name}.csv"))
            col_rows.append({"table": f"{split}_{name}_row_parity", "n_ref_cols": n_split, "n_new_cols": n_feat,
                             "columns_identical": n_split == n_feat, "missing_vs_ref": "", "extra_vs_ref": ""})
    pd.DataFrame(col_rows).to_csv(METRICS_DIR / "feature_table_parity_checks.csv", index=False)
    print(pd.DataFrame(col_rows).to_string())

    print(json.dumps({"corrected_rows": len(corrected), "dropped_ext": len(dropped_ext), "v2_missing_rows": len(v2_missing_all)}, indent=2))


if __name__ == "__main__":
    main()
