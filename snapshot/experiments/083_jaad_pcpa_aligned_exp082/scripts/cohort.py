"""Official PCPA-style JAAD_beh and JAAD_all cohort construction."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd


OBS_LEN = 16
TTE_MIN = 30
TTE_MAX = 60
OVERLAP = 0.8
WINDOW_STRIDE = 3
MIN_TRACK_SIZE = OBS_LEN + TTE_MAX
SPLITS = ("train", "val", "test")
COHORT_TO_SAMPLE_TYPE = {"jaad_beh": "beh", "jaad_all": "all"}
JAAD_ROOT = Path("/home/abdelrahman/Hadwa/datasets/JAAD")
CACHE_ROOT = Path("/tmp/exp083_jaad_api_cache")


def _json(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"))


def _flatten(value: Any) -> str:
    if isinstance(value, (list, tuple, np.ndarray)):
        return _flatten(value[0]) if len(value) else ""
    return str(value)


def _frame_id(path: str) -> int:
    return int(Path(path).stem)


def _video_id(path: str) -> str:
    return Path(path).parent.name


def load_jaad_class(api_path: Path):
    spec = importlib.util.spec_from_file_location("exp083_official_jaad", api_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load official JAAD API from {api_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    module.JAAD.cache_path = property(lambda self: str(CACHE_ROOT))
    return module.JAAD


def load_database_and_api(api_path: Path) -> tuple[Any, dict[str, Any]]:
    imdb = load_jaad_class(api_path)(data_path=str(JAAD_ROOT))
    return imdb, imdb.generate_database()


def _track_record(database: dict[str, Any], video_id: str, ped_id: str) -> dict[str, Any]:
    candidates = (ped_id, f"{ped_id}b") if not ped_id.endswith("b") else (ped_id,)
    records = database[video_id]["ped_annotations"]
    for candidate in candidates:
        if candidate in records:
            return records[candidate]
    raise KeyError(f"Missing JAAD annotation for {video_id}:{ped_id}")


def _crossing_metadata(track: dict[str, Any], observed_frames: list[int]) -> dict[str, Any]:
    frames = [int(value) for value in track.get("frames", [])]
    behavior = track.get("behavior", {}) or {}
    crosses = [int(value) for value in behavior.get("cross", [])]
    frame_to_cross = dict(zip(frames, crosses)) if len(frames) == len(crosses) else {}
    observed_crosses = [frame_to_cross.get(frame) for frame in observed_frames]
    transition_frames = [
        frames[index]
        for index in range(1, min(len(frames), len(crosses)))
        if crosses[index - 1] == 0 and crosses[index] == 1
    ]
    verified_onset = transition_frames[0] if len(transition_frames) == 1 else None
    attrs = track.get("attributes", {}) or {}
    return {
        "crossing_attr": attrs.get("crossing"),
        "crossing_point": attrs.get("crossing_point"),
        "decision_point": attrs.get("decision_point"),
        "observed_crosses_json": _json(observed_crosses),
        "has_observed_crossing": any(value == 1 for value in observed_crosses),
        "verified_onset_frame": verified_onset,
    }


def build_windows(
    raw: dict[str, Any],
    database: dict[str, Any],
    *,
    cohort: Literal["jaad_beh", "jaad_all"],
    split: Literal["train", "val", "test"],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for index, image_sequence in enumerate(raw["image"]):
        sequence_length = len(image_sequence)
        start_index = sequence_length - OBS_LEN - TTE_MAX
        end_index = sequence_length - OBS_LEN - TTE_MIN
        if start_index < 0:
            continue
        ped_id = _flatten(raw["pid"][index][0])
        is_behavior = ped_id.endswith("b")
        if cohort == "jaad_beh" and not is_behavior:
            raise RuntimeError("Official JAAD behavior cohort returned a non-behavior track")
        for window_start in range(start_index, end_index + 1, WINDOW_STRIDE):
            images = image_sequence[window_start : window_start + OBS_LEN]
            observed_frames = [_frame_id(path) for path in images]
            raw_frame_deltas = np.diff(observed_frames)
            raw_frame_consecutive = bool(np.all(raw_frame_deltas == 1))
            max_raw_frame_gap = int(raw_frame_deltas.max())
            video_id = _video_id(images[0])
            label = int(raw["activities"][index][window_start + OBS_LEN - 1][0])
            if not is_behavior and label != 0:
                raise RuntimeError("Official JAAD non-behavior track received a positive action label")
            event_frame = _frame_id(image_sequence[-1])
            tte = sequence_length - (window_start + OBS_LEN)
            start_frame = observed_frames[0]
            end_frame = observed_frames[-1]
            track = _track_record(database, video_id, ped_id)
            crossing = _crossing_metadata(track, observed_frames)
            sequence_id = (
                f"{cohort}:{split}:{video_id}:{ped_id}:"
                f"{start_frame}:{end_frame}:{tte}:{label}"
            )
            rows.append(
                {
                    "sequence_id": sequence_id,
                    "cohort": cohort,
                    "split": split,
                    "label": label,
                    "video_id": video_id,
                    "ped_id": ped_id,
                    "track_id": f"{video_id}:{ped_id}",
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "event_frame": event_frame,
                    "tte_frames": tte,
                    "observation_length": OBS_LEN,
                    "frame_step": 1,
                    "frame_sampling": "consecutive_track_observations",
                    "sample_stride_between_windows": WINDOW_STRIDE,
                    "raw_frame_consecutive": raw_frame_consecutive,
                    "max_raw_frame_gap": max_raw_frame_gap,
                    "observed_frames_json": _json(observed_frames),
                    "bbox_sequence_json": _json(
                        raw["bbox"][index][window_start : window_start + OBS_LEN]
                    ),
                    "is_behavior_track": is_behavior,
                    "label_source": (
                        "behavior_crossing_action"
                        if is_behavior
                        else "official_non_behavior_negative"
                    ),
                    "api_track_length": sequence_length,
                    "api_window_start_index": window_start,
                    "api_window_end_index": window_start + OBS_LEN - 1,
                    "source_annotation_path": str(
                        JAAD_ROOT / "annotations" / f"{video_id}.xml"
                    ),
                    "source_attributes_path": str(
                        JAAD_ROOT
                        / "annotations_attributes"
                        / f"{video_id}_attributes.xml"
                    ),
                    **crossing,
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError(f"Official JAAD builder produced no rows for {cohort}/{split}")
    frame = frame.sort_values(
        ["video_id", "ped_id", "start_frame"], kind="stable"
    ).reset_index(drop=True)
    validate_manifest(frame, cohort=cohort, split=split)
    return frame


def build_cohort(
    sample_type: Literal["beh", "all"],
    split: Literal["train", "val", "test"],
    *,
    api_path: Path,
    imdb: Any | None = None,
    database: dict[str, Any] | None = None,
) -> pd.DataFrame:
    if sample_type not in {"beh", "all"}:
        raise ValueError("sample_type must be beh or all")
    if split not in SPLITS:
        raise ValueError("split must be train, val, or test")
    if imdb is None or database is None:
        imdb, database = load_database_and_api(api_path)
    raw = imdb.generate_data_trajectory_sequence(
        split,
        fstride=1,
        sample_type=sample_type,
        subset="default",
        data_split_type="default",
        seq_type="crossing",
        min_track_size=MIN_TRACK_SIZE,
    )
    cohort_name = "jaad_beh" if sample_type == "beh" else "jaad_all"
    return build_windows(raw, database, cohort=cohort_name, split=split)


def validate_manifest(frame: pd.DataFrame, *, cohort: str, split: str) -> None:
    required = {
        "sequence_id",
        "cohort",
        "split",
        "label",
        "video_id",
        "ped_id",
        "track_id",
        "start_frame",
        "end_frame",
        "event_frame",
        "tte_frames",
        "observation_length",
        "frame_step",
        "frame_sampling",
        "sample_stride_between_windows",
        "raw_frame_consecutive",
        "max_raw_frame_gap",
        "observed_frames_json",
        "bbox_sequence_json",
        "is_behavior_track",
    }
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"Manifest missing columns: {sorted(missing)}")
    if frame["sequence_id"].duplicated().any():
        raise RuntimeError("Manifest contains duplicate sequence IDs")
    if not frame["cohort"].astype(str).eq(cohort).all():
        raise RuntimeError("Manifest cohort identity changed")
    if not frame["split"].astype(str).eq(split).all():
        raise RuntimeError("Manifest split identity changed")
    if not frame["label"].astype(int).isin([0, 1]).all():
        raise RuntimeError("Manifest labels are not binary")
    if not frame["observation_length"].astype(int).eq(OBS_LEN).all():
        raise RuntimeError("Manifest observation length changed")
    if not frame["frame_step"].astype(int).eq(1).all():
        raise RuntimeError("Manifest frame step changed")
    if not frame["frame_sampling"].astype(str).eq("consecutive_track_observations").all():
        raise RuntimeError("Manifest sampling changed")
    for row in frame.itertuples(index=False):
        observed = [int(value) for value in json.loads(row.observed_frames_json)]
        if len(observed) != OBS_LEN or observed[0] != int(row.start_frame) or observed[-1] != int(row.end_frame):
            raise RuntimeError("Manifest observed window changed")
        deltas = np.diff(observed)
        if not np.all(deltas > 0):
            raise RuntimeError("Manifest raw frame IDs are not strictly increasing")
        actual_consecutive = bool(np.all(deltas == 1))
        if actual_consecutive != bool(row.raw_frame_consecutive):
            raise RuntimeError("Manifest raw-frame continuity flag changed")
        if int(deltas.max()) != int(row.max_raw_frame_gap):
            raise RuntimeError("Manifest maximum raw-frame gap changed")
    if not frame["tte_frames"].astype(int).between(TTE_MIN, TTE_MAX).all():
        raise RuntimeError("Manifest TTE is outside 30--60")
    if not frame["sample_stride_between_windows"].astype(int).eq(WINDOW_STRIDE).all():
        raise RuntimeError("Manifest overlap stride changed")
    if cohort == "jaad_beh" and not frame["is_behavior_track"].astype(bool).all():
        raise RuntimeError("JAAD behavior cohort contains a non-behavior track")
    non_behavior = ~frame["is_behavior_track"].astype(bool)
    if cohort == "jaad_all" and (frame.loc[non_behavior, "label"].astype(int) != 0).any():
        raise RuntimeError("JAAD all non-behavior track has a positive label")


def summarize_manifests(manifests: dict[tuple[str, str], pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for (cohort, split), frame in sorted(manifests.items()):
        rows.append(
            {
                "cohort": cohort,
                "split": split,
                "windows": len(frame),
                "tracks": frame["track_id"].nunique(),
                "positive_windows": int(frame["label"].sum()),
                "negative_windows": int((frame["label"] == 0).sum()),
                "positive_rate": float(frame["label"].mean()),
                "behavior_tracks": frame.loc[
                    frame["is_behavior_track"], "track_id"
                ].nunique(),
                "non_behavior_tracks": frame.loc[
                    ~frame["is_behavior_track"], "track_id"
                ].nunique(),
                "missing_crossing_point_windows": int(
                    pd.to_numeric(frame["crossing_point"], errors="coerce").fillna(-1).eq(-1).sum()
                ),
                "observed_crossing_windows": int(frame["has_observed_crossing"].sum()),
                "unverified_endpoint_windows": int(
                    pd.to_numeric(frame["verified_onset_frame"], errors="coerce").fillna(-1)
                    .ne(frame["event_frame"].astype(int))
                    .sum()
                ),
                "nonconsecutive_raw_frame_windows": int(
                    (~frame["raw_frame_consecutive"].astype(bool)).sum()
                ),
                "maximum_raw_frame_gap": int(frame["max_raw_frame_gap"].max()),
            }
        )
    return pd.DataFrame(rows)


def assert_split_disjoint(manifests: dict[tuple[str, str], pd.DataFrame]) -> None:
    for cohort in COHORT_TO_SAMPLE_TYPE:
        tracks = {
            split: set(manifests[(cohort, split)]["track_id"].astype(str))
            for split in SPLITS
        }
        for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
            overlap = tracks[left] & tracks[right]
            if overlap:
                raise RuntimeError(
                    f"{cohort} cross-split track overlap for {left}/{right}: {sorted(overlap)[:3]}"
                )


def assert_approved_raw_frame_gaps(
    manifests: dict[tuple[str, str], pd.DataFrame]
) -> None:
    observed = []
    for (cohort, split), frame in manifests.items():
        gaps = frame.loc[~frame["raw_frame_consecutive"].astype(bool)]
        for row in gaps.itertuples(index=False):
            observed.append((cohort, split, str(row.track_id)))
    expected = [("jaad_all", "train", "video_0098:0_98_544")] * 2
    if sorted(observed) != sorted(expected):
        raise RuntimeError(
            f"Official raw-frame gap population changed: observed={sorted(observed)}"
        )
