"""Audit why JAAD W=16, TTE=5-15 yields zero validated samples."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.benchmark.build_jaad_event_aligned_candidates import TrackRecord, load_jaad_tracks
from src.features.biu_features import compute_biu_features_for_sequence, frame_has_invalid_required_keypoints, frame_has_low_confidence, load_keypoint_index


JAAD_ROOT = Path("/home/abdelrahman/Hadwa/datasets/JAAD")
KEYPOINT_JSONL = Path("data/jaad_keypoints_vitpose_s.jsonl")
CURRENT_JAAD_30_60_SPLITS = Path("data/jaad_splits_event_16_30_60_overlap08_default")
REPORT_PATH = Path("reports/JAAD_5_15_ZERO_SAMPLE_ROOT_CAUSE.md")
STAGE_CSV_PATH = Path("results/jaad_stage_horizon_audit/jaad_5_15_drop_stage_audit.csv")
MISSING_KP_PATH = Path("results/jaad_stage_horizon_audit/jaad_5_15_missing_keypoint_rows.csv")

OBS_LEN = 16
TTE_MIN = 5
TTE_MAX = 15
OVERLAP = 0.8
SPLITS = ["train", "val", "test"]


@dataclass
class CandidateWindow:
    split: str
    video_id: str
    pid: str
    track_id: str
    label: int
    event_frame: int
    start_frame: int
    end_frame: int
    observed_frames: list[int]
    onset_source: str
    negative_alignment_source: str
    tte_frames: int


def _window_stride() -> int:
    return max(1, int(round(OBS_LEN * (1.0 - OVERLAP))))


def _window_end_frames(track: TrackRecord, event_frame: int) -> list[int]:
    earliest_end = max(track.frames[0] + OBS_LEN - 1, event_frame - TTE_MAX)
    latest_end = event_frame - TTE_MIN
    if earliest_end > latest_end:
        return []
    return list(range(earliest_end, latest_end + 1, _window_stride()))


def _example_record(
    *,
    track: TrackRecord | None = None,
    window: CandidateWindow | None = None,
    event_frame: int | None = None,
    missing_frames: list[int] | None = None,
) -> dict[str, Any]:
    if window is not None:
        return {
            "example_video": window.video_id,
            "example_pid": window.pid,
            "example_start_frame": int(window.start_frame),
            "example_end_frame": int(window.end_frame),
            "example_event_frame": int(window.event_frame),
            "example_missing_frames": json.dumps(missing_frames or []),
        }
    if track is not None:
        return {
            "example_video": track.video_id,
            "example_pid": track.ped_id,
            "example_start_frame": int(track.frames[0]) if track.frames else -1,
            "example_end_frame": int(track.frames[-1]) if track.frames else -1,
            "example_event_frame": int(event_frame) if event_frame is not None else -1,
            "example_missing_frames": json.dumps(missing_frames or []),
        }
    return {
        "example_video": "",
        "example_pid": "",
        "example_start_frame": -1,
        "example_end_frame": -1,
        "example_event_frame": -1,
        "example_missing_frames": json.dumps(missing_frames or []),
    }


def _stage_row(stage: str, input_count: int, output_count: int, drop_reason: str, example: dict[str, Any] | None = None) -> dict[str, Any]:
    row = {
        "stage": stage,
        "input_count": int(input_count),
        "output_count": int(output_count),
        "dropped_count": int(input_count - output_count),
        "drop_reason": drop_reason,
        "example_video": "",
        "example_pid": "",
        "example_start_frame": -1,
        "example_end_frame": -1,
        "example_event_frame": -1,
        "example_missing_frames": "[]",
    }
    if example:
        row.update(example)
    return row


def _build_window(track: TrackRecord, event_frame: int, end_frame: int, label: int, onset_source: str, negative_alignment_source: str) -> tuple[CandidateWindow | None, str]:
    start_frame = end_frame - OBS_LEN + 1
    observed_frames = list(range(start_frame, end_frame + 1))
    if not track.is_continuous:
        return None, "non_consecutive_annotation_track"
    missing_frames = [frame for frame in observed_frames if frame not in track.frame_to_cross]
    if missing_frames:
        return None, "annotation_frames_missing"
    observed_crosses = [track.frame_to_cross[frame] for frame in observed_frames]
    if any(cross == 1 for cross in observed_crosses):
        return None, "observed_frame_already_crossing"
    return (
        CandidateWindow(
            split=track.split,
            video_id=track.video_id,
            pid=track.ped_id,
            track_id=track.track_id,
            label=int(label),
            event_frame=int(event_frame),
            start_frame=int(start_frame),
            end_frame=int(end_frame),
            observed_frames=observed_frames,
            onset_source=onset_source,
            negative_alignment_source=negative_alignment_source,
            tte_frames=int(event_frame - end_frame),
        ),
        "valid",
    )


def _positive_tracks(tracks: list[TrackRecord]) -> tuple[list[tuple[TrackRecord, int, str]], dict[str, int], dict[str, dict[str, Any]]]:
    usable: list[tuple[TrackRecord, int, str]] = []
    exclusion_counts: Counter[str] = Counter()
    examples: dict[str, dict[str, Any]] = {}
    for track in tracks:
        if track.crossing_attr is None or track.crossing_attr <= 0:
            continue
        if track.starts_already_crossing:
            exclusion_counts["already_crossing_at_first_visible_frame"] += 1
            examples.setdefault("already_crossing_at_first_visible_frame", _example_record(track=track))
            continue
        if len(track.transition_frames) > 1:
            exclusion_counts["multiple_crossing_transitions"] += 1
            examples.setdefault("multiple_crossing_transitions", _example_record(track=track))
            continue
        if not track.is_continuous:
            exclusion_counts["non_consecutive_annotation_track"] += 1
            examples.setdefault("non_consecutive_annotation_track", _example_record(track=track))
            continue
        if track.track_length < OBS_LEN:
            exclusion_counts["track_too_short_for_observation_window"] += 1
            examples.setdefault("track_too_short_for_observation_window", _example_record(track=track))
            continue
        if len(track.transition_frames) == 1:
            usable.append((track, int(track.transition_frames[0]), "frame_cross_transition"))
            continue
        cp = track.crossing_point
        if cp is not None and cp >= 0 and cp >= track.frames[0] and cp <= track.frames[-1]:
            usable.append((track, int(cp), "crossing_point_fallback"))
        else:
            exclusion_counts["no_valid_crossing_onset"] += 1
            examples.setdefault("no_valid_crossing_onset", _example_record(track=track))
    return usable, dict(exclusion_counts), examples


def _positive_candidate_windows(usable: list[tuple[TrackRecord, int, str]]) -> tuple[list[CandidateWindow], Counter[str], dict[str, dict[str, Any]], dict[str, Any]]:
    valid: list[CandidateWindow] = []
    counts: Counter[str] = Counter()
    examples: dict[str, dict[str, Any]] = {}
    attempted = 0
    valid_example: dict[str, Any] | None = None
    for track, onset_frame, onset_source in usable:
        end_frames = _window_end_frames(track, onset_frame)
        if not end_frames:
            counts["not_enough_frames_before_onset"] += 1
            examples.setdefault("not_enough_frames_before_onset", _example_record(track=track, event_frame=onset_frame))
            continue
        for end_frame in end_frames:
            attempted += 1
            window, status = _build_window(track, onset_frame, end_frame, 1, onset_source, "")
            if window is None:
                counts[status] += 1
                missing = []
                if status == "annotation_frames_missing":
                    missing = [frame for frame in range(end_frame - OBS_LEN + 1, end_frame + 1) if frame not in track.frame_to_cross]
                examples.setdefault(status, _example_record(track=track, event_frame=onset_frame, missing_frames=missing))
                continue
            valid.append(window)
            if valid_example is None:
                valid_example = _example_record(window=window)
    counts["valid"] = len(valid)
    summary = {
        "attempted": attempted,
        "valid_example": valid_example or _example_record(),
    }
    return valid, counts, examples, summary


def _negative_candidate_windows(tracks: list[TrackRecord]) -> list[CandidateWindow]:
    valid: list[CandidateWindow] = []
    for track in tracks:
        if track.crossing_attr is None or track.crossing_attr > 0:
            continue
        if not track.is_continuous or track.track_length < OBS_LEN or track.starts_already_crossing or track.transition_frames or any(cross == 1 for cross in track.crosses):
            continue
        event_frame = int(track.frames[-1])
        for end_frame in _window_end_frames(track, event_frame):
            window, status = _build_window(track, event_frame, end_frame, 0, "", "last_visible_frame")
            if window is not None and status == "valid":
                valid.append(window)
    return valid


def _bbox_alignment(candidates: list[CandidateWindow], track_map: dict[str, TrackRecord]) -> tuple[list[CandidateWindow], Counter[str], dict[str, int], dict[str, Any]]:
    valid: list[CandidateWindow] = []
    reasons: Counter[str] = Counter()
    missing_keys: Counter[tuple[str, str, int]] = Counter()
    example: dict[str, Any] | None = None
    for window in candidates:
        track = track_map[window.track_id]
        missing = [frame for frame in window.observed_frames if frame not in track.frame_to_bbox]
        if missing:
            reasons["missing_bbox_frame"] += 1
            for frame in missing:
                missing_keys[(window.video_id, window.pid, int(frame))] += 1
            if example is None:
                example = _example_record(window=window, missing_frames=missing)
            continue
        valid.append(window)
    return valid, reasons, {f"{video}|{pid}|{frame}": count for (video, pid, frame), count in missing_keys.most_common(20)}, example or _example_record()


def _keypoint_alignment(
    candidates: list[CandidateWindow],
    keypoint_jsonl: Path,
) -> tuple[list[CandidateWindow], Counter[str], pd.DataFrame, dict[str, Any], dict[str, Any]]:
    keypoint_index = load_keypoint_index(keypoint_jsonl)
    valid: list[CandidateWindow] = []
    reasons: Counter[str] = Counter()
    missing_keys: Counter[tuple[str, str, int]] = Counter()
    missing_rows: list[dict[str, Any]] = []
    example: dict[str, Any] | None = None
    for window in candidates:
        missing = []
        for frame in window.observed_frames:
            key = ("jaad", window.video_id, window.pid, int(frame))
            if key not in keypoint_index:
                missing.append(int(frame))
                missing_keys[(window.video_id, window.pid, int(frame))] += 1
                missing_rows.append(
                    {
                        "video_id": window.video_id,
                        "pid": window.pid,
                        "frame_idx": int(frame),
                        "event_frame": int(window.event_frame),
                        "start_frame": int(window.start_frame),
                        "end_frame": int(window.end_frame),
                        "tte_to_onset": int(window.event_frame - frame),
                        "label": int(window.label),
                        "split": window.split,
                    }
                )
        if missing:
            reasons["missing_exact_keypoint_frame"] += 1
            if example is None:
                example = _example_record(window=window, missing_frames=missing)
            continue
        valid.append(window)
    if missing_rows:
        missing_df = pd.DataFrame(missing_rows).drop_duplicates(subset=["video_id", "pid", "frame_idx"]).sort_values(
            ["video_id", "pid", "frame_idx"]
        )
    else:
        missing_df = pd.DataFrame(columns=["video_id", "pid", "frame_idx", "event_frame", "start_frame", "end_frame", "tte_to_onset", "label", "split"])
    top_missing = {
        f"{video}|{pid}|{frame}": count for (video, pid, frame), count in missing_keys.most_common(50)
    }
    near_onset = {}
    if not missing_df.empty:
        near_onset = {
            "missing_unique_rows": int(len(missing_df)),
            "pct_tte_leq_15": float((missing_df["tte_to_onset"] <= 15).mean()),
            "pct_tte_leq_20": float((missing_df["tte_to_onset"] <= 20).mean()),
            "median_tte_to_onset": float(missing_df["tte_to_onset"].median()),
            "min_tte_to_onset": int(missing_df["tte_to_onset"].min()),
            "max_tte_to_onset": int(missing_df["tte_to_onset"].max()),
        }
    return valid, reasons, missing_df, top_missing, {
        "example": example or _example_record(),
        "near_onset": near_onset,
    }


def _biu_stage(candidates: list[CandidateWindow], keypoint_jsonl: Path) -> tuple[list[CandidateWindow], Counter[str], dict[str, Any]]:
    keypoint_index = load_keypoint_index(keypoint_jsonl)
    valid: list[CandidateWindow] = []
    reasons: Counter[str] = Counter()
    example: dict[str, Any] | None = None
    for window in candidates:
        frames = []
        for frame_idx in window.observed_frames:
            frames.append(keypoint_index[("jaad", window.video_id, window.pid, int(frame_idx))])
        invalid_required = sum(frame_has_invalid_required_keypoints(item["keypoints"]) for item in frames)
        low_conf = sum(frame_has_low_confidence(item.get("kp_conf")) for item in frames)
        try:
            features = compute_biu_features_for_sequence(frames)
        except Exception:
            reasons["biu_compute_exception"] += 1
            if example is None:
                example = _example_record(window=window)
            continue
        values = np.asarray(list(features.values()), dtype=float)
        if np.isnan(values).all():
            reasons["all_biu_values_nan"] += 1
            if example is None:
                example = _example_record(window=window)
            continue
        if invalid_required > 0:
            reasons["missing_required_keypoints_in_biu_frames"] += 1
            if example is None:
                example = _example_record(window=window)
            continue
        if low_conf > 0:
            reasons["low_confidence_required_keypoints_in_biu_frames"] += 1
            if example is None:
                example = _example_record(window=window)
            continue
        valid.append(window)
    return valid, reasons, example or _example_record()


def _split_assignment(candidates: list[CandidateWindow]) -> pd.DataFrame:
    rows = []
    for window in candidates:
        rows.append(
            {
                "split": window.split,
                "video_id": window.video_id,
                "track_id": window.track_id,
                "label": window.label,
            }
        )
    return pd.DataFrame(rows)


def _label_values_by_split(split_df: pd.DataFrame) -> dict[str, list[int]]:
    if split_df.empty:
        return {split: [] for split in SPLITS}
    return {
        split: sorted(split_df.loc[split_df["split"] == split, "label"].astype(int).unique().tolist())
        for split in SPLITS
    }


def _coverage_diagnostic(raw_candidates: list[CandidateWindow], tracks: list[TrackRecord], keypoint_jsonl: Path) -> dict[str, Any]:
    current_30_60 = {}
    for split in SPLITS:
        path = CURRENT_JAAD_30_60_SPLITS / f"{split}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            current_30_60.setdefault((str(row["video_id"]), str(row["ped_id"])), []).extend(json.loads(row["observed_frames_json"]))

    keypoint_index = load_keypoint_index(keypoint_jsonl)
    kp_track_frames: defaultdict[tuple[str, str], set[int]] = defaultdict(set)
    for set_id, video_id, pid, frame_idx in keypoint_index:
        if str(set_id) == "jaad":
            kp_track_frames[(str(video_id), str(pid))].add(int(frame_idx))

    track_map = {track.track_id: track for track in tracks}
    stats = Counter()
    for window in raw_candidates:
        track_key = (window.video_id, window.pid)
        available = kp_track_frames.get(track_key, set())
        if not available:
            stats["tracks_with_no_keypoints"] += 1
            continue
        max_kp = max(available)
        max_5_15_needed = max(window.observed_frames)
        current_frames = current_30_60.get(track_key, [])
        max_30_60_needed = max(current_frames) if current_frames else None
        if track_map[window.track_id].frames and min(available) == track_map[window.track_id].frames[0]:
            stats["prefix_coverage_from_track_start"] += 1
        if max_kp < max_5_15_needed:
            stats["max_kp_before_5_15_requirement"] += 1
        if max_30_60_needed is not None and max_kp >= max_30_60_needed and max_kp < max_5_15_needed:
            stats["covers_current_30_60_but_not_5_15"] += 1
    return dict(stats)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keypoint-jsonl", type=Path, default=KEYPOINT_JSONL)
    parser.add_argument("--report-path", type=Path, default=REPORT_PATH)
    parser.add_argument("--stage-csv-path", type=Path, default=STAGE_CSV_PATH)
    parser.add_argument("--missing-kp-path", type=Path, default=MISSING_KP_PATH)
    return parser.parse_args()


def run(
    *,
    keypoint_jsonl: Path,
    report_path: Path,
    stage_csv_path: Path,
    missing_kp_path: Path,
) -> None:
    tracks, split_videos = load_jaad_tracks(JAAD_ROOT, "default")
    track_map = {track.track_id: track for track in tracks}
    rows: list[dict[str, Any]] = []

    total_tracks = len(tracks)
    tracks_with_cross_any = sum(any(cross == 1 for cross in track.crosses) for track in tracks)
    tracks_with_transition = sum(bool(track.transition_frames) for track in tracks)
    positive_tracks = [track for track in tracks if track.crossing_attr is not None and track.crossing_attr > 0]
    positive_starts_crossing = sum(track.starts_already_crossing for track in positive_tracks)
    positive_multiple_transitions = sum(len(track.transition_frames) > 1 for track in positive_tracks)
    usable_positive, positive_exclusions, positive_examples = _positive_tracks(tracks)
    usable_crossing_onsets = len(usable_positive)

    rows.extend(
        [
            _stage_row("stage1_raw_event_tracks", total_tracks, total_tracks, "all_pedestrian_tracks"),
            _stage_row("stage1_raw_event_tracks", total_tracks, tracks_with_cross_any, "tracks_with_cross_1_anywhere"),
            _stage_row("stage1_raw_event_tracks", total_tracks, tracks_with_transition, "tracks_with_valid_0_to_1_transition"),
            _stage_row(
                "stage1_raw_event_tracks",
                len(positive_tracks),
                len(positive_tracks) - positive_starts_crossing,
                "excluded_already_crossing_at_first_visible_frame",
                positive_examples.get("already_crossing_at_first_visible_frame"),
            ),
            _stage_row(
                "stage1_raw_event_tracks",
                len(positive_tracks),
                len(positive_tracks) - positive_multiple_transitions,
                "excluded_multiple_crossing_transitions",
                positive_examples.get("multiple_crossing_transitions"),
            ),
            _stage_row(
                "stage1_raw_event_tracks",
                len(positive_tracks),
                usable_crossing_onsets,
                "usable_crossing_onsets",
                _example_record(track=usable_positive[0][0], event_frame=usable_positive[0][1]) if usable_positive else None,
            ),
        ]
    )
    for reason, count in sorted(positive_exclusions.items()):
        rows.append(
            _stage_row(
                "stage1_raw_event_tracks",
                len(positive_tracks),
                len(positive_tracks) - count,
                f"excluded_{reason}",
                positive_examples.get(reason),
            )
        )

    positive_windows, positive_window_counts, positive_window_examples, positive_summary = _positive_candidate_windows(usable_positive)
    rows.append(
        _stage_row(
            "stage2_positive_candidate_windows",
            positive_summary["attempted"],
            positive_window_counts.get("valid", 0),
            "valid_positive_candidate_windows",
            positive_summary["valid_example"],
        )
    )
    for reason in [
        "not_enough_frames_before_onset",
        "non_consecutive_annotation_track",
        "observed_frame_already_crossing",
        "annotation_frames_missing",
    ]:
        count = positive_window_counts.get(reason, 0)
        input_count = len(usable_positive) if reason == "not_enough_frames_before_onset" else positive_summary["attempted"]
        output_count = input_count - count
        rows.append(
            _stage_row(
                "stage2_positive_candidate_windows",
                input_count,
                output_count,
                reason,
                positive_window_examples.get(reason),
            )
        )

    positive_bbox, bbox_reasons, bbox_missing_top20, bbox_example = _bbox_alignment(positive_windows, track_map)
    rows.append(_stage_row("stage3_bbox_alignment", len(positive_windows), len(positive_bbox), "bbox_alignment", bbox_example))
    if bbox_reasons:
        for reason, count in bbox_reasons.items():
            rows.append(_stage_row("stage3_bbox_alignment", len(positive_windows), len(positive_windows) - count, reason, bbox_example))
    else:
        rows.append(_stage_row("stage3_bbox_alignment", len(positive_windows), len(positive_bbox), "no_bbox_drops", bbox_example))

    positive_kp, kp_reasons, missing_kp_df, missing_top50, kp_meta = _keypoint_alignment(positive_bbox, keypoint_jsonl)
    rows.append(
        _stage_row(
            "stage4_keypoint_alignment",
            len(positive_bbox),
            len(positive_kp),
            "keypoint_exact_frame_alignment",
            kp_meta["example"],
        )
    )
    for reason, count in kp_reasons.items():
        rows.append(_stage_row("stage4_keypoint_alignment", len(positive_bbox), len(positive_bbox) - count, reason, kp_meta["example"]))

    positive_biu, biu_reasons, biu_example = _biu_stage(positive_kp, keypoint_jsonl)
    rows.append(_stage_row("stage5_biu_feature_construction", len(positive_kp), len(positive_biu), "biu_feature_construction", biu_example))
    for reason, count in biu_reasons.items():
        rows.append(_stage_row("stage5_biu_feature_construction", len(positive_kp), len(positive_kp) - count, reason, biu_example))

    negative_windows = _negative_candidate_windows(tracks)
    negative_bbox, negative_bbox_reasons, _, _ = _bbox_alignment(negative_windows, track_map)
    negative_kp, negative_kp_reasons, negative_missing_kp_df, _, _ = _keypoint_alignment(negative_bbox, keypoint_jsonl)
    all_raw_candidates = positive_windows + negative_windows
    all_bbox, _, _, _ = _bbox_alignment(all_raw_candidates, track_map)
    all_kp, _, _, _, _ = _keypoint_alignment(all_bbox, keypoint_jsonl)
    all_biu, _, _ = _biu_stage(all_kp, keypoint_jsonl)
    split_df = _split_assignment(all_biu)
    if split_df.empty:
        split_counts = {split: 0 for split in SPLITS}
        positive_ratio = 0.0
        unique_videos = 0
        unique_tracks = 0
    else:
        split_counts = {split: int((split_df["split"] == split).sum()) for split in SPLITS}
        positive_ratio = float(split_df["label"].mean())
        unique_videos = int(split_df["video_id"].nunique())
        unique_tracks = int(split_df["track_id"].nunique())
    label_values_by_split = _label_values_by_split(split_df)
    rows.append(_stage_row("stage6_split_assignment", len(all_biu), sum(split_counts.values()), "final_validated_samples"))

    stage_csv_path.parent.mkdir(parents=True, exist_ok=True)
    audit_df = pd.DataFrame(rows)
    audit_df.to_csv(stage_csv_path, index=False)
    missing_kp_df.to_csv(missing_kp_path, index=False)

    coverage = _coverage_diagnostic(positive_windows, tracks, keypoint_jsonl)
    if len(positive_windows) == 0:
        conclusion = "1. No raw 5_15 candidates exist."
    elif len(positive_bbox) == 0:
        conclusion = "3. Candidates exist but bbox alignment fails."
    elif len(positive_kp) == 0:
        conclusion = "4. Candidates exist but keypoint alignment fails."
    elif len(positive_biu) == 0:
        conclusion = "5. Candidates pass keypoints but BIU feature construction fails."
    elif sum(split_counts.values()) == 0:
        conclusion = "6. Candidates pass all stages but split assignment removes them."
    else:
        conclusion = "Validated 5_15 samples now exist after extending JAAD keypoint coverage."

    lines = [
        "# JAAD 5-15 Zero-Sample Root Cause",
        "",
        "## Conclusion",
        "",
        conclusion,
        "",
        f"The raw JAAD annotations do produce valid `W=16`, `TTE=5-15` positive windows before any feature alignment. Those windows also retain valid bbox coverage because the boxes come from the same visible annotation frames used to form the windows. The collapse happens at exact keypoint alignment against `{keypoint_jsonl}`: zero positive windows, and therefore zero final validated benchmark rows, retain all 16 required observed keypoint frames." if len(positive_kp) == 0 else f"The raw JAAD annotations do produce valid `W=16`, `TTE=5-15` positive windows before feature alignment, and exact keypoint alignment against `{keypoint_jsonl}` now retains `{len(positive_kp)}` positive windows.",
        "",
        "## Stage 1 - Raw Event Tracks",
        "",
        f"- Pedestrian tracks: `{total_tracks}`.",
        f"- Tracks with frame-level `cross=1` anywhere: `{tracks_with_cross_any}`.",
        f"- Tracks with a valid `0->1` transition: `{tracks_with_transition}`.",
        f"- Positive crossing tracks excluded because already crossing at first visible frame: `{positive_starts_crossing}`.",
        f"- Positive crossing tracks excluded because multiple transitions: `{positive_multiple_transitions}`.",
        f"- Usable crossing onsets after current builder-style exclusions: `{usable_crossing_onsets}`.",
        "",
        "## Stage 2 - 5-15 Candidate Windows Before Feature Alignment",
        "",
        f"- Total candidate windows attempted: `{positive_summary['attempted']}`.",
        f"- Total valid positive candidate windows: `{len(positive_windows)}`.",
        f"- Dropped because not enough frames before onset: `{positive_window_counts.get('not_enough_frames_before_onset', 0)}` onset tracks.",
        f"- Dropped because observation frames are not consecutive: `{positive_window_counts.get('non_consecutive_annotation_track', 0)}`.",
        f"- Dropped because an observed frame already has `cross=1`: `{positive_window_counts.get('observed_frame_already_crossing', 0)}`.",
        f"- Dropped because annotation frames are missing: `{positive_window_counts.get('annotation_frames_missing', 0)}`.",
        f"- Example valid sample: `{positive_summary['valid_example']}`.",
        "",
        "## Stage 3 - BBox Alignment",
        "",
        f"- Candidates before bbox alignment: `{len(positive_windows)}`.",
        f"- Candidates after bbox alignment: `{len(positive_bbox)}`.",
        f"- Candidates dropped due to missing bbox frames: `{sum(bbox_reasons.values())}`.",
        f"- Top missing bbox keys: `{bbox_missing_top20}`.",
        "",
        "## Stage 4 - Keypoint Alignment",
        "",
        f"- Candidates before keypoint alignment: `{len(positive_bbox)}`.",
        f"- Candidates after exact keypoint alignment: `{len(positive_kp)}`.",
        f"- Candidates dropped due to missing keypoint frame(s): `{sum(kp_reasons.values())}`.",
        f"- Top 50 missing keypoint keys: `{missing_top50}`.",
        "",
    ]
    if kp_meta["near_onset"]:
        lines.extend(
            [
                "### Missing-Keypoint Timing",
                "",
                f"- Unique missing `(video, pid, frame_idx)` rows: `{kp_meta['near_onset']['missing_unique_rows']}`.",
                f"- Share of missing rows with `tte_to_onset <= 15`: `{kp_meta['near_onset']['pct_tte_leq_15']:.4f}`.",
                f"- Share of missing rows with `tte_to_onset <= 20`: `{kp_meta['near_onset']['pct_tte_leq_20']:.4f}`.",
                f"- Median `tte_to_onset` of missing rows: `{kp_meta['near_onset']['median_tte_to_onset']:.1f}`.",
                f"- Min/max `tte_to_onset` of missing rows: `{kp_meta['near_onset']['min_tte_to_onset']}` / `{kp_meta['near_onset']['max_tte_to_onset']}`.",
                "",
            ]
        )

    lines.extend(
        [
            "### JSONL Coverage Diagnostic",
            "",
            f"- Coverage diagnostic counts: `{coverage}`.",
            "- Interpretation: the keypoint JSONL does not behave like full-track coverage for the candidate `5-15` windows. It behaves like a truncated coverage source that is sufficient for the existing `30-60` benchmark windows but often stops before the later near-onset frames required by `5-15`.",
            "",
            "### Required ViTPose Extraction",
            "",
            f"- Exact missing keypoint rows that must be extracted are saved in `{missing_kp_path}`.",
            f"- Total unique missing rows requiring ViTPose extraction: `{int(len(missing_kp_df))}`.",
            "",
            "## Stage 5 - BIU Feature Construction",
            "",
            f"- Candidates before BIU build: `{len(positive_kp)}`.",
            f"- Candidates after BIU build: `{len(positive_biu)}`.",
            f"- Dropped due to low-confidence/missing required keypoints: `{biu_reasons.get('missing_required_keypoints_in_biu_frames', 0) + biu_reasons.get('low_confidence_required_keypoints_in_biu_frames', 0)}`.",
            f"- Dropped due to NaN/invalid BIU values: `{biu_reasons.get('all_biu_values_nan', 0) + biu_reasons.get('biu_compute_exception', 0)}`.",
            "",
            "## Stage 6 - Final Split Assignment",
            "",
            f"- Final validated train rows: `{split_counts['train']}`.",
            f"- Final validated val rows: `{split_counts['val']}`.",
            f"- Final validated test rows: `{split_counts['test']}`.",
            f"- Final unique videos: `{unique_videos}`.",
            f"- Final unique tracks: `{unique_tracks}`.",
            f"- Final positive ratio: `{positive_ratio:.4f}`.",
            "",
        "## Full-Benchmark Note",
        "",
        f"- Raw positive windows before feature alignment: `{len(positive_windows)}`.",
        f"- Raw negative windows before feature alignment: `{len(negative_windows)}`.",
        f"- Positive windows after exact keypoint alignment: `{len(positive_kp)}`.",
        f"- Negative windows after exact keypoint alignment: `{len(negative_kp)}`.",
        f"- Raw total windows before feature alignment: `{len(all_raw_candidates)}`.",
        f"- Total windows after exact keypoint alignment: `{len(all_kp)}`.",
        f"- Total windows after BIU-stage validation: `{len(all_biu)}`.",
        f"- Negative windows dropped by bbox alignment: `{sum(negative_bbox_reasons.values())}`.",
        f"- Negative windows dropped by exact keypoint alignment: `{sum(negative_kp_reasons.values())}`.",
        f"- Remaining negative missing-keypoint rows after this audit: `{int(len(negative_missing_kp_df))}`.",
        "",
        "## Final Split Labels",
        "",
        f"- Train label values: `{label_values_by_split['train']}`.",
        f"- Val label values: `{label_values_by_split['val']}`.",
        f"- Test label values: `{label_values_by_split['test']}`.",
        "",
        "## Artifacts",
            "",
            f"- Stage audit CSV: `{stage_csv_path}`.",
            f"- Missing keypoint requirement CSV: `{missing_kp_path}`.",
        ]
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    args = _parse_args()
    run(
        keypoint_jsonl=args.keypoint_jsonl,
        report_path=args.report_path,
        stage_csv_path=args.stage_csv_path,
        missing_kp_path=args.missing_kp_path,
    )
