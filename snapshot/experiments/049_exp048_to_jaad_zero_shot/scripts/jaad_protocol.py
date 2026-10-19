"""Label-sealed JAAD next-raw-frame cohort construction."""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from contracts import OBSERVATION_LENGTH, PREDICTION_OFFSET


JAAD_ROOT = Path("/home/abdelrahman/Hadwa/datasets/JAAD")
JAAD_PARSER_PATH = JAAD_ROOT / "jaad_data.py"
JAAD_SPLIT_ROOT = JAAD_ROOT / "split_ids" / "default"

MANIFEST_COLUMNS = (
    "sequence_id",
    "split",
    "video_id",
    "ped_id",
    "start_frame",
    "end_frame",
    "target_frame",
    "observation_frames",
    "frame_width",
    "frame_height",
)


def read_official_split(split: str) -> tuple[str, ...]:
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported JAAD split: {split!r}.")
    path = JAAD_SPLIT_ROOT / f"{split}.txt"
    video_ids = tuple(line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    if not video_ids or len(set(video_ids)) != len(video_ids):
        raise ValueError(f"Malformed official JAAD split file: {path}")
    return video_ids


def _is_behavior_track(ped_id: str, record: Mapping[str, object]) -> bool:
    behavior = record.get("behavior", {})
    return "b" in str(ped_id) and isinstance(behavior, Mapping)


def build_observation_manifest(
    db: Mapping[str, Mapping[str, object]],
    video_ids: Sequence[str],
    split: str,
) -> pd.DataFrame:
    """Build exact 30-observation/next-frame identities without target labels."""
    requested = tuple(str(video_id) for video_id in video_ids)
    missing = sorted(set(requested).difference(db))
    if missing:
        raise ValueError(f"Video outside declared split/database: {missing[:5]}")

    rows: list[dict[str, object]] = []
    required_region = OBSERVATION_LENGTH + PREDICTION_OFFSET
    for video_id in requested:
        video = db[video_id]
        width = int(video["width"])
        height = int(video["height"])
        pedestrians = video.get("ped_annotations", {})
        if not isinstance(pedestrians, Mapping):
            raise ValueError(f"Malformed pedestrian annotations for {video_id}.")
        for ped_id in sorted(pedestrians):
            pedestrian = pedestrians[ped_id]
            if not isinstance(pedestrian, Mapping) or not _is_behavior_track(str(ped_id), pedestrian):
                continue
            frames = tuple(int(frame) for frame in pedestrian.get("frames", ()))
            if len(frames) < required_region:
                continue
            if len(set(frames)) != len(frames):
                raise ValueError(f"Duplicate raw frames for {video_id}:{ped_id}.")
            for offset in range(len(frames) - required_region + 1):
                region = frames[offset : offset + required_region]
                if any(right != left + 1 for left, right in zip(region, region[1:])):
                    continue
                observed = tuple(region[:OBSERVATION_LENGTH])
                target = int(region[-1])
                start = int(observed[0])
                end = int(observed[-1])
                rows.append(
                    {
                        "sequence_id": f"jaad:{split}:{video_id}:{ped_id}:{start}:{end}:{target}",
                        "split": str(split),
                        "video_id": video_id,
                        "ped_id": str(ped_id),
                        "start_frame": start,
                        "end_frame": end,
                        "target_frame": target,
                        "observation_frames": observed,
                        "frame_width": width,
                        "frame_height": height,
                    }
                )
    manifest = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    if not manifest.empty:
        manifest = manifest.sort_values(
            ["video_id", "ped_id", "start_frame", "target_frame"], kind="stable"
        ).reset_index(drop=True)
        if manifest["sequence_id"].duplicated().any():
            raise ValueError("JAAD manifest contains duplicate sequence IDs.")
    return manifest


def attach_target_labels(
    manifest: pd.DataFrame,
    db: Mapping[str, Mapping[str, object]],
) -> pd.DataFrame:
    """Attach only the target frame's binary behavior.cross annotation."""
    if "label" in manifest.columns:
        raise ValueError("Target labels are already attached.")
    output = manifest.copy()
    labels: list[object] = []
    reasons: list[str] = []
    for row in output.itertuples(index=False):
        pedestrian = db[str(row.video_id)]["ped_annotations"][str(row.ped_id)]
        frames = [int(frame) for frame in pedestrian.get("frames", ())]
        crosses = pedestrian.get("behavior", {}).get("cross", ())
        try:
            index = frames.index(int(row.target_frame))
        except ValueError:
            labels.append(pd.NA)
            reasons.append("missing_target_behavior")
            continue
        if index >= len(crosses):
            labels.append(pd.NA)
            reasons.append("missing_target_behavior")
            continue
        try:
            value = int(crosses[index])
        except (TypeError, ValueError):
            labels.append(pd.NA)
            reasons.append("nonbinary_target_behavior")
            continue
        if value not in (0, 1):
            labels.append(pd.NA)
            reasons.append("nonbinary_target_behavior")
            continue
        labels.append(value)
        reasons.append("")
    output["label"] = pd.array(labels, dtype="Int64")
    output["exclusion_reason"] = reasons
    return output


def audit_manifest(
    manifest: pd.DataFrame,
    expected_videos: Sequence[str],
) -> dict[str, object]:
    required = set(MANIFEST_COLUMNS)
    if not required.issubset(manifest.columns):
        raise ValueError(f"Manifest is missing columns: {sorted(required.difference(manifest.columns))}")
    observed_videos = set(manifest["video_id"].astype(str))
    unexpected = sorted(observed_videos.difference(str(value) for value in expected_videos))
    if unexpected:
        raise ValueError(f"Manifest contains videos outside official split: {unexpected[:5]}")
    continuity_failures = 0
    temporal_failures = 0
    for row in manifest.itertuples(index=False):
        frames = tuple(int(frame) for frame in row.observation_frames)
        continuity_failures += int(any(b != a + 1 for a, b in zip(frames, frames[1:])))
        temporal_failures += int(
            len(frames) != OBSERVATION_LENGTH
            or int(row.target_frame) != frames[-1] + PREDICTION_OFFSET
        )
    result: dict[str, object] = {
        "rows": int(len(manifest)),
        "official_split_videos": int(len(expected_videos)),
        "videos_with_windows": int(len(observed_videos)),
        "behavior_tracks_with_windows": int(
            manifest[["video_id", "ped_id"]].drop_duplicates().shape[0]
        ),
        "continuity_failures": int(continuity_failures),
        "temporal_contract_failures": int(temporal_failures),
        "duplicate_sequence_ids": int(manifest["sequence_id"].duplicated().sum()),
        "target_labels_attached": bool("label" in manifest.columns),
    }
    if "label" in manifest.columns:
        valid = manifest[manifest["label"].notna()]
        result.update(
            {
                "valid_target_labels": int(len(valid)),
                "positive_target_labels": int((valid["label"] == 1).sum()),
                "negative_target_labels": int((valid["label"] == 0).sum()),
                "excluded_target_labels": int(manifest["label"].isna().sum()),
                "exclusion_reasons": {
                    str(key): int(value)
                    for key, value in manifest.loc[
                        manifest["label"].isna(), "exclusion_reason"
                    ].value_counts().sort_index().items()
                },
            }
        )
    return result


def manifest_for_csv(manifest: pd.DataFrame) -> pd.DataFrame:
    output = manifest.copy()
    output["observation_frames"] = output["observation_frames"].map(
        lambda frames: json.dumps(list(frames), separators=(",", ":"))
    )
    return output


def load_selected_jaad_database(
    video_ids: Sequence[str],
    *,
    include_cross: bool = True,
) -> dict[str, dict[str, object]]:
    """Parse only explicitly requested videos, avoiding JAAD's global cache writer."""
    module_name = "exp049_jaad_parser"
    if module_name in sys.modules:
        module = sys.modules[module_name]
    else:
        if importlib.util.find_spec("cv2") is None and "cv2" not in sys.modules:
            # Annotation parsing does not use OpenCV, but the upstream module imports it globally.
            sys.modules["cv2"] = types.ModuleType("cv2")
        spec = importlib.util.spec_from_file_location(module_name, JAAD_PARSER_PATH)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot import JAAD parser: {JAAD_PARSER_PATH}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    parser = module.JAAD(data_path=str(JAAD_ROOT))
    database: dict[str, dict[str, object]] = {}
    for video_id in video_ids:
        video = parser._get_annotations(str(video_id))
        attributes = parser._get_ped_attributes(str(video_id))
        for ped_id, pedestrian in video["ped_annotations"].items():
            pedestrian["attributes"] = attributes.get(ped_id, {})
            if not include_cross:
                pedestrian.get("behavior", {}).pop("cross", None)
        video["vehicle_annotations"] = parser._get_vehicle_attributes(str(video_id))
        video["traffic_annotations"] = parser._get_traffic_attributes(str(video_id))
        database[str(video_id)] = video
    return database
