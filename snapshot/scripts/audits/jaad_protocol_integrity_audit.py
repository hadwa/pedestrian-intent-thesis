"""Audit JAAD protocol integrity against raw frame-level crossing annotations."""

from __future__ import annotations

import json
import textwrap
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


CLEAN_ROOT = Path("/home/abdelrahman/Hadwa/BIU-PID-Distillation-clean")
ARCHIVE_ROOT = Path("/home/abdelrahman/Hadwa/BIU-PID-Distillation")
JAAD_ROOT = Path("/home/abdelrahman/Hadwa/datasets/JAAD")

EXPERIMENT_ROOT = CLEAN_ROOT / "experiments/005_jaad_protocol_integrity_audit"
METRICS_DIR = EXPERIMENT_ROOT / "metrics"
REPORT_PATH = EXPERIMENT_ROOT / "report.md"
README_PATH = EXPERIMENT_ROOT / "README.md"

CP_MINUS1_TRACKS_PATH = METRICS_DIR / "jaad_crossing_point_minus1_tracks.csv"
ANNOTATION_SUMMARY_PATH = METRICS_DIR / "jaad_annotation_transition_summary.csv"
WINDOW_SUMMARY_PATH = METRICS_DIR / "jaad_window_integrity_summary.csv"
CONTAMINATED_WINDOWS_PATH = METRICS_DIR / "jaad_contaminated_windows.csv"
ARTIFACT_COVERAGE_PATH = METRICS_DIR / "jaad_artifact_coverage.csv"
CP_MINUS1_WINDOW_IMPACT_PATH = METRICS_DIR / "jaad_crossing_point_minus1_window_impact.csv"

SEVERE_ISSUES = {
    "missing_observation_frames_in_track",
    "observed_crossing_frame_present",
    "positive_missing_future_0_to_1_onset",
    "positive_crossing_point_fallback_with_negative_crossing_point",
    "positive_crossing_point_fallback_without_frame_transition",
    "positive_window_during_crossing_segment",
    "positive_window_after_crossing_segment",
    "negative_track_contains_crossing_frames",
    "negative_window_during_crossing_segment",
    "negative_window_after_crossing_segment",
}


@dataclass
class TrackRecord:
    split: str
    video: str
    ped_id: str
    track_id: str
    frames: list[int]
    crosses: list[int]
    crossing_attr: int | None
    crossing_point: int | None
    decision_point: int | None

    @property
    def starts_crossing(self) -> bool:
        return bool(self.crosses and self.crosses[0] == 1)

    @property
    def has_any_crossing_frame(self) -> bool:
        return any(c == 1 for c in self.crosses)

    @property
    def transition_01_frames(self) -> list[int]:
        return [
            self.frames[idx]
            for idx in range(1, len(self.frames))
            if self.crosses[idx - 1] == 0 and self.crosses[idx] == 1
        ]

    @property
    def transition_10_frames(self) -> list[int]:
        return [
            self.frames[idx]
            for idx in range(1, len(self.frames))
            if self.crosses[idx - 1] == 1 and self.crosses[idx] == 0
        ]

    @property
    def first_cross_frame(self) -> int | None:
        for frame, cross in zip(self.frames, self.crosses, strict=True):
            if cross == 1:
                return frame
        return None

    @property
    def last_cross_frame(self) -> int | None:
        for frame, cross in zip(reversed(self.frames), reversed(self.crosses), strict=True):
            if cross == 1:
                return frame
        return None


def _parse_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _table_text(df: pd.DataFrame) -> str:
    if df.empty:
        return "No rows."
    try:
        return df.to_markdown(index=False)
    except Exception:
        return "```\n" + df.to_string(index=False) + "\n```"


def _read_split_ids() -> dict[str, str]:
    split_dir = JAAD_ROOT / "split_ids/default"
    mapping: dict[str, str] = {}
    for split in ["train", "val", "test"]:
        for line in (split_dir / f"{split}.txt").read_text(encoding="utf-8").splitlines():
            video = line.strip()
            if video:
                mapping[video] = split
    return mapping


def _parse_attributes(path: Path) -> dict[str, dict[str, str]]:
    root = ET.parse(path).getroot()
    return {ped.get("id"): dict(ped.attrib) for ped in root.findall("./pedestrian") if ped.get("id")}


def _find_box_attr(box: ET.Element, name: str) -> str | None:
    attr = box.find(f'./attribute[@name="{name}"]')
    if attr is None or attr.text is None:
        return None
    return attr.text.strip()


def load_tracks() -> dict[str, TrackRecord]:
    split_map = _read_split_ids()
    tracks: dict[str, TrackRecord] = {}
    for ann_path in sorted((JAAD_ROOT / "annotations").glob("*.xml")):
        video = ann_path.stem
        split = split_map.get(video, "")
        attrs = _parse_attributes(JAAD_ROOT / "annotations_attributes" / f"{video}_attributes.xml")
        root = ET.parse(ann_path).getroot()
        for track_el in root.findall("./track"):
            if track_el.get("label") != "pedestrian":
                continue
            boxes = [box for box in track_el.findall("./box") if int(box.get("outside", "0")) == 0]
            if not boxes:
                continue
            ped_id = _find_box_attr(boxes[0], "id")
            if not ped_id:
                continue
            frames: list[int] = []
            crosses: list[int] = []
            for box in boxes:
                cross_raw = _find_box_attr(box, "cross")
                if cross_raw not in {"crossing", "not-crossing"}:
                    continue
                frames.append(int(box.get("frame")))
                crosses.append(1 if cross_raw == "crossing" else 0)
            if not frames:
                continue
            ped_attrs = attrs.get(ped_id, {})
            track_id = f"{video}:{ped_id}"
            tracks[track_id] = TrackRecord(
                split=split,
                video=video,
                ped_id=ped_id,
                track_id=track_id,
                frames=frames,
                crosses=crosses,
                crossing_attr=_parse_int(ped_attrs.get("crossing")),
                crossing_point=_parse_int(ped_attrs.get("crossing_point")),
                decision_point=_parse_int(ped_attrs.get("decision_point")),
            )
    return tracks


def _track_annotation_row(track: TrackRecord) -> dict[str, Any]:
    first_cross = track.first_cross_frame
    last_cross = track.last_cross_frame
    wrong_if_whole_track_positive = sum(1 for c in track.crosses if c == 0)
    pre_cross_non_cross = sum(1 for f, c in zip(track.frames, track.crosses, strict=True) if c == 0 and first_cross is not None and f < first_cross)
    post_cross_non_cross = sum(1 for f, c in zip(track.frames, track.crosses, strict=True) if c == 0 and last_cross is not None and f > last_cross)
    if not track.has_any_crossing_frame:
        classification = "no_crossing_frames"
    elif track.starts_crossing:
        classification = "crossing_from_start"
    elif track.transition_01_frames:
        classification = "has_0_to_1_onset"
    else:
        classification = "crossing_without_clean_0_to_1"
    return {
        "split": track.split,
        "video": track.video,
        "pedestrian_id": track.ped_id,
        "track_id": track.track_id,
        "track_length": len(track.frames),
        "crossing_attr": track.crossing_attr,
        "crossing_point": track.crossing_point,
        "decision_point": track.decision_point,
        "has_frame_crossing_anywhere": int(track.has_any_crossing_frame),
        "starts_crossing_from_first_visible_frame": int(track.starts_crossing),
        "has_0_to_1_transition": int(bool(track.transition_01_frames)),
        "num_0_to_1_transitions": len(track.transition_01_frames),
        "has_1_to_0_transition": int(bool(track.transition_10_frames)),
        "num_1_to_0_transitions": len(track.transition_10_frames),
        "has_post_cross_non_cross_tail": int(post_cross_non_cross > 0),
        "first_cross_frame": first_cross,
        "last_cross_frame": last_cross,
        "transition_01_frames_json": json.dumps(track.transition_01_frames),
        "transition_10_frames_json": json.dumps(track.transition_10_frames),
        "wrong_if_whole_track_positive_frame_count": wrong_if_whole_track_positive,
        "pre_cross_non_cross_frame_count": pre_cross_non_cross,
        "post_cross_non_cross_frame_count": post_cross_non_cross,
        "classification": classification,
    }


def build_annotation_tables(tracks: dict[str, TrackRecord]) -> tuple[pd.DataFrame, pd.DataFrame]:
    annotation_df = pd.DataFrame([_track_annotation_row(track) for track in tracks.values()]).sort_values(["split", "video", "pedestrian_id"])
    annotation_df.to_csv(ANNOTATION_SUMMARY_PATH, index=False)
    cp_minus1_df = annotation_df.loc[(annotation_df["crossing_attr"] == 1) & (annotation_df["crossing_point"] == -1)].copy()
    cp_minus1_df.to_csv(CP_MINUS1_TRACKS_PATH, index=False)
    return annotation_df, cp_minus1_df


def _discover_artifacts() -> list[dict[str, Any]]:
    return [
        {
            "artifact_name": "jaad_30_60_overlap08",
            "horizon": "30_60",
            "type": "split_dir",
            "path": ARCHIVE_ROOT / "data/jaad_splits_event_16_30_60_overlap08_default",
            "source_repo": "archive",
        },
        {
            "artifact_name": "jaad_15_30_stage",
            "horizon": "15_30",
            "type": "split_dir",
            "path": ARCHIVE_ROOT / "data/jaad_splits_stage_horizon_audit/tte_15_30",
            "source_repo": "archive",
        },
        {
            "artifact_name": "jaad_5_15_stage",
            "horizon": "5_15",
            "type": "split_dir",
            "path": ARCHIVE_ROOT / "data/jaad_splits_stage_horizon_audit/tte_5_15",
            "source_repo": "archive",
        },
        {
            "artifact_name": "jaad_temporal_transition_rows",
            "horizon": "mixed",
            "type": "csv",
            "path": CLEAN_ROOT / "experiments/004_jaad_temporal_biu_transition/metrics/horizon_feature_rows.csv",
            "source_repo": "clean",
        },
    ]


def _load_artifact_rows(artifact: dict[str, Any]) -> pd.DataFrame:
    path = Path(artifact["path"])
    if artifact["type"] == "split_dir":
        frames = []
        for split in ["train", "val", "test"]:
            split_path = path / f"{split}.csv"
            if split_path.exists():
                df = pd.read_csv(split_path)
                df["split"] = split
                frames.append(df)
        if not frames:
            return pd.DataFrame()
        df = pd.concat(frames, ignore_index=True)
    else:
        if not path.exists():
            return pd.DataFrame()
        df = pd.read_csv(path)
    df["artifact_name"] = artifact["artifact_name"]
    df["artifact_path"] = str(path)
    if "horizon_label" not in df.columns:
        df["horizon_label"] = artifact["horizon"]
    return df


def _normalize_rows(df: pd.DataFrame) -> pd.DataFrame:
    if "video_id" not in df.columns and "video" in df.columns:
        df["video_id"] = df["video"]
    if "ped_id" not in df.columns and "pid" in df.columns:
        df["ped_id"] = df["pid"]
    df["track_id"] = df["video_id"].astype(str) + ":" + df["ped_id"].astype(str)
    df["label"] = df["label"].astype(int)
    df["frame_start"] = df["start_frame"].astype(int)
    df["frame_end"] = df["end_frame"].astype(int)
    if "event_frame_or_anchor" not in df.columns:
        df["event_frame_or_anchor"] = df["event_frame"]
    return df


def _window_position(track: TrackRecord, start_frame: int, end_frame: int) -> str:
    if not track.has_any_crossing_frame:
        return "no_per_frame_crossing_available"
    frame_to_cross = dict(zip(track.frames, track.crosses, strict=True))
    observed = [frame_to_cross.get(frame) for frame in range(start_frame, end_frame + 1)]
    if any(c == 1 for c in observed if c is not None):
        return "during_crossing"
    first_cross = track.first_cross_frame
    last_cross = track.last_cross_frame
    if first_cross is not None and end_frame < first_cross:
        return "pre_crossing_clean"
    if last_cross is not None and start_frame > last_cross:
        return "post_crossing"
    return "unknown"


def _issue_row(record: pd.Series, track: TrackRecord, issue_type: str, explanation: str, position: str) -> dict[str, Any]:
    return {
        "artifact_name": record["artifact_name"],
        "artifact_path": record["artifact_path"],
        "split": record.get("split", ""),
        "horizon": record.get("horizon_label", ""),
        "video": track.video,
        "pedestrian_id": track.ped_id,
        "track_id": track.track_id,
        "frame_start": int(record["frame_start"]),
        "frame_end": int(record["frame_end"]),
        "event_frame_or_anchor": record.get("event_frame_or_anchor"),
        "label": int(record["label"]),
        "issue_type": issue_type,
        "explanation": explanation,
        "position_class": position,
        "crossing_attr": track.crossing_attr,
        "crossing_point": track.crossing_point,
        "onset_source": str(record.get("onset_source", "") or ""),
        "sequence_id": record.get("sequence_id", ""),
    }


def audit_artifacts(tracks: dict[str, TrackRecord]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    coverage_rows: list[dict[str, Any]] = []
    contaminated_rows: list[dict[str, Any]] = []
    cp_minus1_window_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    issue_columns = [
        "artifact_name",
        "artifact_path",
        "split",
        "horizon",
        "video",
        "pedestrian_id",
        "track_id",
        "frame_start",
        "frame_end",
        "event_frame_or_anchor",
        "label",
        "issue_type",
        "explanation",
        "position_class",
        "crossing_attr",
        "crossing_point",
        "onset_source",
        "sequence_id",
    ]

    for artifact in _discover_artifacts():
        df = _load_artifact_rows(artifact)
        coverage_rows.append(
            {
                "artifact_name": artifact["artifact_name"],
                "artifact_path": str(artifact["path"]),
                "source_repo": artifact["source_repo"],
                "artifact_type": artifact["type"],
                "exists": int(not df.empty),
                "row_count": int(len(df)),
                "columns": "|".join(df.columns) if not df.empty else "",
            }
        )
        if df.empty:
            continue

        df = _normalize_rows(df.copy())
        if artifact["artifact_name"] == "jaad_temporal_transition_rows":
            df = df.drop_duplicates(subset=["sequence_id", "horizon_label"], keep="first")

        for record in df.to_dict(orient="records"):
            track = tracks.get(str(record["track_id"]))
            if track is None:
                continue
            frame_to_cross = dict(zip(track.frames, track.crosses, strict=True))
            observed_crosses = [frame_to_cross.get(frame) for frame in range(int(record["frame_start"]), int(record["frame_end"]) + 1)]
            position = _window_position(track, int(record["frame_start"]), int(record["frame_end"]))
            label = int(record["label"])
            onset_source = str(record.get("onset_source", "") or "")

            if any(c is None for c in observed_crosses):
                contaminated_rows.append(_issue_row(pd.Series(record), track, "missing_observation_frames_in_track", "Observed frame missing from raw track.", position))
            if any(c == 1 for c in observed_crosses if c is not None):
                contaminated_rows.append(_issue_row(pd.Series(record), track, "observed_crossing_frame_present", "Observation window contains frame-level cross=1.", position))

            if label == 1:
                future_onsets = [frame for frame in track.transition_01_frames if frame > int(record["frame_end"])]
                if not future_onsets:
                    contaminated_rows.append(_issue_row(pd.Series(record), track, "positive_missing_future_0_to_1_onset", "Positive window has no future frame-level 0->1 onset after observation.", position))
                if onset_source == "crossing_point_fallback":
                    contaminated_rows.append(_issue_row(pd.Series(record), track, "positive_crossing_point_fallback_used", "Positive label used crossing_point fallback.", position))
                    if track.crossing_point is None or track.crossing_point < 0:
                        contaminated_rows.append(_issue_row(pd.Series(record), track, "positive_crossing_point_fallback_with_negative_crossing_point", "Fallback would rely on crossing_point < 0.", position))
                    if not track.transition_01_frames:
                        contaminated_rows.append(_issue_row(pd.Series(record), track, "positive_crossing_point_fallback_without_frame_transition", "Fallback used without any frame-level 0->1 transition.", position))
                if position == "during_crossing":
                    contaminated_rows.append(_issue_row(pd.Series(record), track, "positive_window_during_crossing_segment", "Positive observation overlaps crossing segment.", position))
                if position == "post_crossing":
                    contaminated_rows.append(_issue_row(pd.Series(record), track, "positive_window_after_crossing_segment", "Positive observation occurs after crossing segment.", position))
            else:
                if track.has_any_crossing_frame:
                    contaminated_rows.append(_issue_row(pd.Series(record), track, "negative_track_contains_crossing_frames", "Negative sample comes from track with frame-level crossing elsewhere.", position))
                if position == "during_crossing":
                    contaminated_rows.append(_issue_row(pd.Series(record), track, "negative_window_during_crossing_segment", "Negative observation overlaps crossing segment.", position))
                if position == "post_crossing":
                    contaminated_rows.append(_issue_row(pd.Series(record), track, "negative_window_after_crossing_segment", "Negative observation occurs after crossing segment.", position))

            if track.crossing_attr == 1 and track.crossing_point == -1:
                cp_minus1_window_rows.append(
                    _issue_row(
                        pd.Series(record),
                        track,
                        "window_on_crossing_attr_positive_crossing_point_minus1_track",
                        f"Track has crossing=1 and crossing_point=-1; position={position}.",
                        position,
                    )
                )

        cont_df = pd.DataFrame(contaminated_rows, columns=issue_columns)
        cp_rows_df = pd.DataFrame(cp_minus1_window_rows, columns=issue_columns)
        for (split, horizon, label), group_df in df.groupby(["split", "horizon_label", "label"], dropna=False):
            local = cont_df.loc[
                (cont_df["artifact_name"] == artifact["artifact_name"])
                & (cont_df["split"].astype(str) == str(split))
                & (cont_df["horizon"].astype(str) == str(horizon))
                & (cont_df["label"].astype(str) == str(label))
            ]
            cp_minus1_tracks = {
                track_id
                for track_id in group_df["track_id"].tolist()
                if track_id in tracks and tracks[track_id].crossing_attr == 1 and tracks[track_id].crossing_point == -1
            }
            summary_rows.append(
                {
                    "artifact_name": artifact["artifact_name"],
                    "artifact_path": str(artifact["path"]),
                    "split": split,
                    "horizon": horizon,
                    "label": int(label),
                    "window_count": int(len(group_df)),
                    "track_count": int(group_df["track_id"].nunique()),
                    "contaminated_window_count": int(local["sequence_id"].nunique()) if not local.empty else 0,
                    "issue_types": "|".join(sorted(local["issue_type"].unique().tolist())) if not local.empty else "",
                    "crossing_point_minus1_positive_track_count": len(cp_minus1_tracks),
                    "crossing_point_minus1_window_count": int(sum(group_df["track_id"].isin(cp_minus1_tracks))),
                    "crossing_point_minus1_window_positions": "|".join(
                        sorted(
                            cp_rows_df.loc[
                                (cp_rows_df["artifact_name"] == artifact["artifact_name"])
                                & (cp_rows_df["split"].astype(str) == str(split))
                                & (cp_rows_df["horizon"].astype(str) == str(horizon))
                                & (cp_rows_df["label"].astype(str) == str(label)),
                                "position_class",
                            ].dropna().unique().tolist()
                        )
                    )
                    if not cp_rows_df.empty
                    else "",
                    "positive_fallback_count": int((group_df["onset_source"] == "crossing_point_fallback").sum()) if "onset_source" in group_df.columns else 0,
                }
            )

    coverage_df = pd.DataFrame(coverage_rows)
    coverage_df.to_csv(ARTIFACT_COVERAGE_PATH, index=False)
    contaminated_df = pd.DataFrame(contaminated_rows)
    if contaminated_df.empty:
        contaminated_df = pd.DataFrame(columns=issue_columns)
    contaminated_df = contaminated_df.sort_values(["artifact_name", "split", "horizon", "video", "pedestrian_id", "frame_start", "issue_type"])
    contaminated_df.to_csv(CONTAMINATED_WINDOWS_PATH, index=False)
    cp_minus1_window_df = pd.DataFrame(cp_minus1_window_rows)
    if cp_minus1_window_df.empty:
        cp_minus1_window_df = pd.DataFrame(columns=issue_columns)
    cp_minus1_window_df = cp_minus1_window_df.sort_values(["artifact_name", "split", "horizon", "video", "pedestrian_id", "frame_start", "issue_type"])
    cp_minus1_window_df.to_csv(CP_MINUS1_WINDOW_IMPACT_PATH, index=False)
    summary_df = pd.DataFrame(summary_rows).sort_values(["artifact_name", "split", "horizon", "label"])
    summary_df.to_csv(WINDOW_SUMMARY_PATH, index=False)
    return coverage_df, contaminated_df, cp_minus1_window_df, summary_df


def _code_summary() -> list[str]:
    summary: list[str] = []
    archive_builder = ARCHIVE_ROOT / "src/benchmark/build_jaad_event_aligned_candidates.py"
    clean_stage_builder = CLEAN_ROOT / "src/analysis/staged_prediction_hypothesis_audit.py"
    if archive_builder.exists():
        text = archive_builder.read_text(encoding="utf-8")
        if "if any(cross == 1 for cross in observed_crosses):" in text:
            summary.append("- Archive overlap-0.8 builder rejects any observation window containing frame-level `cross=1`.")
        if 'if cp is None or cp < 0:' in text and 'onset_source = "crossing_point_fallback"' in text:
            summary.append("- Archive overlap-0.8 builder only uses `crossing_point_fallback` when `crossing_point` is non-negative.")
    if clean_stage_builder.exists():
        text = clean_stage_builder.read_text(encoding="utf-8")
        if "if cp is None or cp < 0 or cp < track.frames[0] or cp > track.frames[-1]:" in text:
            summary.append("- Clean staged-horizon builder also rejects negative `crossing_point` values before fallback.")
        if "track.starts_already_crossing or track.transition_frames or any(cross == 1 for cross in track.crosses)" in text:
            summary.append("- Clean staged-horizon negatives are rejected if any frame-level crossing exists anywhere in the track.")
    if not summary:
        summary.append("- Exact builder path could not be proven from the available source files.")
    return summary


def write_report(annotation_df: pd.DataFrame, cp_minus1_df: pd.DataFrame, contaminated_df: pd.DataFrame, cp_minus1_window_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    positives_with_negative_cp_fallback = contaminated_df.loc[contaminated_df["issue_type"] == "positive_crossing_point_fallback_with_negative_crossing_point"]
    observed_crossing_windows = contaminated_df.loc[contaminated_df["issue_type"] == "observed_crossing_frame_present"]
    post_cross_windows = contaminated_df.loc[contaminated_df["issue_type"].isin(["positive_window_after_crossing_segment", "negative_window_after_crossing_segment"])]
    positive_fallback_windows = contaminated_df.loc[contaminated_df["issue_type"] == "positive_crossing_point_fallback_used"]
    affected_by_artifact = contaminated_df.groupby(["artifact_name", "issue_type"]).size().reset_index(name="count") if not contaminated_df.empty else pd.DataFrame()
    cp_minus1_by_artifact = cp_minus1_window_df.groupby(["artifact_name", "position_class"]).size().reset_index(name="count") if not cp_minus1_window_df.empty else pd.DataFrame()

    lines = [
        "# JAAD Protocol Integrity Audit",
        "",
        "## Raw annotation findings",
        "",
        f"- Total tracks parsed: **{len(annotation_df)}**.",
        f"- Tracks with `attributes['crossing']=1`: **{int((annotation_df['crossing_attr'] == 1).sum())}**.",
        f"- Tracks with `crossing_point=-1`: **{int((annotation_df['crossing_point'] == -1).sum())}**.",
        f"- Tracks with `attributes['crossing']=1` and `crossing_point=-1`: **{len(cp_minus1_df)}**.",
        f"- Wrongly positive frames if those `crossing=1, crossing_point=-1` tracks were treated as crossing for the whole track: **{int(cp_minus1_df['wrong_if_whole_track_positive_frame_count'].sum()) if not cp_minus1_df.empty else 0}**.",
        "",
        "The bachelor-thesis concern is plausible in raw JAAD because these tracks do exist and can contain non-crossing visible frames before or after the actual crossing segment.",
        "",
        "## Current artifact findings",
        "",
        f"- Observation windows containing any frame-level `cross=1`: **{observed_crossing_windows['sequence_id'].nunique() if not observed_crossing_windows.empty else 0}**.",
        f"- Positive windows using `crossing_point_fallback`: **{positive_fallback_windows['sequence_id'].nunique() if not positive_fallback_windows.empty else 0}**.",
        f"- Positive windows that would require fallback despite `crossing_point < 0`: **{positives_with_negative_cp_fallback['sequence_id'].nunique() if not positives_with_negative_cp_fallback.empty else 0}**.",
        f"- Windows occurring in post-crossing regions: **{post_cross_windows['sequence_id'].nunique() if not post_cross_windows.empty else 0}**.",
        f"- Windows on `crossing=1, crossing_point=-1` tracks: **{cp_minus1_window_df['sequence_id'].nunique() if not cp_minus1_window_df.empty else 0}**.",
        "",
        "### Contamination by artifact",
        "",
    ]
    if affected_by_artifact.empty:
        lines.append("No contaminated windows were found in the audited artifacts.")
    else:
        lines.append(_table_text(affected_by_artifact))
    lines.extend(
        [
            "",
            "### `crossing_point=-1` window impact",
            "",
            _table_text(cp_minus1_by_artifact) if not cp_minus1_by_artifact.empty else "No windows from `crossing=1, crossing_point=-1` tracks appeared in the audited artifacts.",
            "",
        ]
    )
    lines.extend(
        [
            "",
            "### Window integrity summary",
            "",
            _table_text(summary_df) if not summary_df.empty else "No window summary rows were produced.",
            "",
            "## Code path summary",
            "",
            *_code_summary(),
            "",
            "## Conclusions",
            "",
            f"- Raw JAAD API concern real/plausible: **yes**, based on **{len(cp_minus1_df)}** `crossing=1, crossing_point=-1` tracks.",
            f"- Current `30_60` / `15_30` / `5_15` artifacts affected by actual contamination: **{'yes' if not contaminated_df.empty else 'no'}**.",
            f"- Positive labels based on `crossing_point=-1` fallback: **{positives_with_negative_cp_fallback['sequence_id'].nunique() if not positives_with_negative_cp_fallback.empty else 0}**.",
            f"- Observation windows containing already-crossing frames: **{observed_crossing_windows['sequence_id'].nunique() if not observed_crossing_windows.empty else 0}**.",
            f"- Samples from post-crossing regions: **{post_cross_windows['sequence_id'].nunique() if not post_cross_windows.empty else 0}**.",
            f"- Windows on `crossing=1, crossing_point=-1` tracks are present, but they are **informational rather than contaminating** when they remain `pre_crossing_clean` and use `frame_cross_transition`.",
            "",
            "If the contaminated-window count is zero, the current JAAD experiments are protocol-clean on this issue and safe to continue from.",
            "If any contaminated windows appear, those specific artifact families should be regenerated before future experiments.",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def write_readme() -> None:
    README_PATH.write_text(
        textwrap.dedent(
            """\
            # JAAD Protocol Integrity Audit

            ## Purpose
            Audit whether current JAAD sample windows are contaminated by track-level crossing metadata such as `crossing=1` with `crossing_point=-1`, post-crossing windows, or already-crossing observation frames.

            ## Inputs inspected
            - Raw JAAD annotations in `/home/abdelrahman/Hadwa/datasets/JAAD`
            - Archive JAAD split/sample artifacts for `30_60`, `15_30`, and `5_15`
            - Clean temporal BIU transition audit rows in `experiments/004_jaad_temporal_biu_transition/metrics/horizon_feature_rows.csv`
            - Clean and archive JAAD builder code paths

            ## How to rerun
            ```bash
            cd /home/abdelrahman/Hadwa/BIU-PID-Distillation-clean
            python3 scripts/audits/jaad_protocol_integrity_audit.py
            ```

            ## Main outputs
            - `experiments/005_jaad_protocol_integrity_audit/report.md`
            - `experiments/005_jaad_protocol_integrity_audit/metrics/jaad_crossing_point_minus1_tracks.csv`
            - `experiments/005_jaad_protocol_integrity_audit/metrics/jaad_annotation_transition_summary.csv`
            - `experiments/005_jaad_protocol_integrity_audit/metrics/jaad_window_integrity_summary.csv`
            - `experiments/005_jaad_protocol_integrity_audit/metrics/jaad_contaminated_windows.csv`
            - `experiments/005_jaad_protocol_integrity_audit/metrics/jaad_artifact_coverage.csv`
            - `experiments/005_jaad_protocol_integrity_audit/metrics/jaad_crossing_point_minus1_window_impact.csv`

            ## Final conclusion
            See `report.md` for the exact contamination counts and whether the current JAAD artifacts are safe to continue from.
            """
        ),
        encoding="utf-8",
    )


def main() -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    tracks = load_tracks()
    annotation_df, cp_minus1_df = build_annotation_tables(tracks)
    _, contaminated_df, cp_minus1_window_df, summary_df = audit_artifacts(tracks)
    write_report(annotation_df, cp_minus1_df, contaminated_df, cp_minus1_window_df, summary_df)
    write_readme()
    print(
        json.dumps(
            {
                "tracks_total": len(annotation_df),
                "crossing_attr_positive_and_crossing_point_minus1_tracks": len(cp_minus1_df),
                "contaminated_windows": int(contaminated_df["sequence_id"].nunique()) if not contaminated_df.empty else 0,
                "report": str(REPORT_PATH),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
