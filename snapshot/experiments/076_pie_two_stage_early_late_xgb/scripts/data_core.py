"""Exact-frame source loading and view materialization for Exp076/077."""

from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import two_stage_core  # noqa: E402


Key = tuple[str, str, str, int]


def validate_window_cohort(frame: pd.DataFrame, *, tte_column: str) -> None:
    required = {
        "sequence_id", "track_id", "label", "start_frame", "end_frame",
        "observation_length", "frame_step", "frame_sampling", tte_column,
    }
    if required.difference(frame.columns) or frame.empty or frame["sequence_id"].duplicated().any():
        raise ValueError("16-frame cohort schema or identity is invalid")
    exact = (
        frame["observation_length"].astype(int).eq(16)
        & frame["frame_step"].astype(int).eq(1)
        & frame["frame_sampling"].astype(str).eq("consecutive")
        & (frame["end_frame"].astype(int) - frame["start_frame"].astype(int)).eq(15)
        & frame[tte_column].astype(int).between(30, 60, inclusive="both")
    )
    if not exact.all():
        raise ValueError("cohort is not the exact 16-frame/TTE 30--60 benchmark")
    if not set(frame["label"].astype(int).unique()).issubset({0, 1}):
        raise ValueError("cohort labels are not binary")


def observed_frames(row: Any) -> tuple[int, ...]:
    raw = getattr(row, "observed_frames_json", None)
    if raw is not None and not pd.isna(raw):
        frames = tuple(map(int, json.loads(str(raw))))
    else:
        frames = tuple(range(int(row.start_frame), int(row.end_frame) + 1))
    if len(frames) != 16 or frames != tuple(range(frames[0], frames[0] + 16)):
        raise ValueError(f"sequence {row.sequence_id} does not contain 16 exact consecutive frames")
    return frames


def needed_keys(frame: pd.DataFrame) -> set[Key]:
    output: set[Key] = set()
    for row in frame.itertuples(index=False):
        for frame_index in observed_frames(row):
            output.add((str(row.set_id), str(row.video_id), str(row.ped_id), int(frame_index)))
    return output


def _same_pose(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return np.array_equal(np.asarray(left["keypoints"], dtype=float), np.asarray(right["keypoints"], dtype=float)) and np.array_equal(
        np.asarray(left.get("kp_conf"), dtype=float),
        np.asarray(right.get("kp_conf"), dtype=float),
    )


def load_keypoint_index(paths: Sequence[Path], needed: set[Key]) -> tuple[dict[Key, dict[str, Any]], dict[str, Any]]:
    output: dict[Key, dict[str, Any]] = {}
    source_new_frames: dict[str, int] = {}
    duplicate_identical = 0
    for source in map(Path, paths):
        resolved = source.resolve(strict=True)
        added = 0
        with resolved.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                item = json.loads(line)
                key = (str(item["set"]), str(item["video"]), str(item["pid"]), int(item["frame_idx"]))
                if key not in needed:
                    continue
                record = {
                    "keypoints": np.asarray(item["keypoints"], dtype=float),
                    "kp_conf": np.asarray(item["kp_conf"], dtype=float),
                    "source": str(resolved),
                    "line": int(line_number),
                }
                if record["keypoints"].shape != (17, 2) or record["kp_conf"].shape != (17,):
                    raise ValueError(f"invalid COCO-17 pose row at {resolved}:{line_number}")
                if key in output:
                    if not _same_pose(output[key], record):
                        raise ValueError(f"conflicting keypoint record for {key}")
                    duplicate_identical += 1
                    continue
                output[key] = record
                added += 1
        source_new_frames[str(resolved)] = added
    missing = needed.difference(output)
    if missing:
        raise ValueError(f"missing exact keypoint frame: {sorted(missing)[0]}")
    return output, {
        "required_frames": int(len(needed)),
        "loaded_frames": int(len(output)),
        "missing_frames": int(len(missing)),
        "duplicate_identical_frames": int(duplicate_identical),
        "source_new_frames": source_new_frames,
    }


def load_pie_bboxes(frame: pd.DataFrame, annotations_dir: Path) -> dict[Key, np.ndarray]:
    needed = needed_keys(frame)
    output: dict[Key, np.ndarray] = {}
    videos = sorted(set(zip(frame["set_id"].astype(str), frame["video_id"].astype(str))))
    for set_id, video_id in videos:
        path = Path(annotations_dir) / set_id / f"{video_id}_annt.xml"
        for track in ET.parse(path).getroot().findall("track"):
            if track.attrib.get("label") != "pedestrian":
                continue
            for box in track.findall("box"):
                ped_id = next((str(item.text) for item in box.findall("attribute") if item.attrib.get("name") == "id"), None)
                if ped_id is None:
                    continue
                key = (set_id, video_id, ped_id, int(box.attrib["frame"]))
                if key not in needed:
                    continue
                if key in output:
                    raise ValueError(f"duplicate exact PIE BBox frame: {key}")
                output[key] = np.asarray(
                    [box.attrib["xtl"], box.attrib["ytl"], box.attrib["xbr"], box.attrib["ybr"]],
                    dtype=float,
                )
    missing = needed.difference(output)
    if missing:
        raise ValueError(f"missing exact PIE BBox frame: {sorted(missing)[0]}")
    return output


def load_jaad_bboxes(frame: pd.DataFrame) -> dict[Key, np.ndarray]:
    output: dict[Key, np.ndarray] = {}
    for row in frame.itertuples(index=False):
        frames = observed_frames(row)
        boxes = np.asarray(json.loads(str(row.bbox_sequence_json)), dtype=float)
        if boxes.shape != (16, 4):
            raise ValueError(f"JAAD BBox identity changed for {row.sequence_id}")
        for frame_index, box in zip(frames, boxes):
            key = (str(row.set_id), str(row.video_id), str(row.ped_id), int(frame_index))
            if key in output and not np.array_equal(output[key], box):
                raise ValueError(f"conflicting JAAD BBox record for {key}")
            output[key] = box
    missing = needed_keys(frame).difference(output)
    if missing:
        raise ValueError(f"missing exact JAAD BBox frame: {sorted(missing)[0]}")
    return output


def load_pie_speed(frame: pd.DataFrame, obd_dir: Path) -> dict[tuple[str, str, int], float]:
    output: dict[tuple[str, str, int], float] = {}
    videos = sorted(set(zip(frame["set_id"].astype(str), frame["video_id"].astype(str))))
    for set_id, video_id in videos:
        path = Path(obd_dir) / set_id / f"{video_id}_obd.xml"
        for item in ET.parse(path).getroot().findall("frame"):
            key = (set_id, video_id, int(item.attrib["id"]))
            if key in output:
                raise ValueError(f"duplicate PIE OBD frame: {key}")
            output[key] = float(item.attrib["OBD_speed"])
    required = {
        (str(row.set_id), str(row.video_id), frame_index)
        for row in frame.itertuples(index=False)
        for frame_index in observed_frames(row)
    }
    missing = required.difference(output)
    if missing:
        raise ValueError(f"missing exact PIE OBD frame: {sorted(missing)[0]}")
    return output


def materialize_views(
    frame: pd.DataFrame,
    keypoints: Mapping[Key, Mapping[str, Any]],
    bboxes: Mapping[Key, np.ndarray],
    *,
    contract: Mapping[str, tuple[str, ...]],
    speed: Mapping[tuple[str, str, int], float] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = []
    missing_windows = 0
    meta_columns = [
        name for name in (
            "sequence_id", "split", "set_id", "video_id", "ped_id", "track_id", "start_frame",
            "end_frame", "label", "observation_length", "frame_step", "frame_sampling",
            "time_to_event_frames", "tte_frames", "group",
        ) if name in frame.columns
    ]
    for row in frame.itertuples(index=False):
        frames = observed_frames(row)
        keys = [(str(row.set_id), str(row.video_id), str(row.ped_id), value) for value in frames]
        if any(key not in keypoints or key not in bboxes for key in keys):
            missing_windows += 1
            raise ValueError(f"missing exact source frame for {row.sequence_id}")
        coords = np.asarray([keypoints[key]["keypoints"] for key in keys], dtype=float)
        confidence = np.asarray([keypoints[key]["kp_conf"] for key in keys], dtype=float)
        boxes = np.asarray([bboxes[key] for key in keys], dtype=float)
        views = two_stage_core.extract_window_views(coords, confidence, boxes, contract=contract)
        result = {name: getattr(row, name) for name in meta_columns}
        result.update({f"early__{name}": value for name, value in views["early"].items()})
        result.update({f"full__{name}": value for name, value in views["full"].items()})
        result.update({f"portable__{name}": value for name, value in views["portable"].items()})
        if speed is not None:
            early_speed = np.asarray(
                [speed[(str(row.set_id), str(row.video_id), value)] for value in frames[:8]],
                dtype=float,
            )
            result.update(
                {f"early_ego__{name}": value for name, value in two_stage_core.speed_features(early_speed).items()}
            )
        rows.append(result)
    output = pd.DataFrame(rows)
    if len(output) != len(frame) or output["sequence_id"].duplicated().any():
        raise RuntimeError("materialized view cohort changed")
    two_stage_core.assert_cohort_parity(frame, output)
    return output, {
        "rows": int(len(output)),
        "missing_windows": int(missing_windows),
        "early_feature_count": 161 + (5 if speed is not None else 0),
        "portable_feature_count": 182,
    }
