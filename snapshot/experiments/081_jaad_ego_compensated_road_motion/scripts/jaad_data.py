"""Exact, test-sealed JAAD cohort and frame access for Exp081."""

from __future__ import annotations

import importlib.util
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping
from xml.etree import ElementTree

import cv2
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = SCRIPT_DIR.parent
ACTIVE_ROOT = EXPERIMENT_ROOT.parents[1]
JAAD_ROOT = Path("/home/abdelrahman/Hadwa/datasets/JAAD")
VIDEO_ROOT = JAAD_ROOT / "JAAD_clips"
ANNOTATION_ROOT = JAAD_ROOT / "annotations"
EXP077_ROOT = ACTIVE_ROOT / "experiments/077_jaad_two_stage_early_late_xgb_transfer"


def _load_contract():
    path = SCRIPT_DIR / "contract.py"
    spec = importlib.util.spec_from_file_location("exp081_contract_for_data", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load Exp081 contract from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CONTRACT = _load_contract()


def _development_anchor_path(split: str) -> Path:
    if split == "train":
        return CONTRACT.resolve_source("exp077_prepared_train")
    if split in {"validation", "val"}:
        return CONTRACT.resolve_source("exp077_prepared_validation")
    raise ValueError(f"Unsupported development split: {split}")


def _authorize_test_access(allow_test: bool) -> None:
    if not allow_test:
        raise RuntimeError("Exp081 test access is sealed before validation advance")
    try:
        marker = CONTRACT.authenticate_phase(EXPERIMENT_ROOT, "validate")
    except RuntimeError as exc:
        raise RuntimeError("Exp081 test access is sealed before validation advance") from exc
    if marker.get("gate_status") != "advance" or marker.get("test_accessed") is not False:
        raise RuntimeError("Exp081 test access is sealed before validation advance")


def _validate_anchor(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    required = {
        "sequence_id",
        "video_id",
        "track_id",
        "start_frame",
        "end_frame",
        "label",
        "observation_length",
        "frame_step",
        "frame_sampling",
        "tte_frames",
    }
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"Exp081 {split} anchor is missing columns: {sorted(missing)}")
    if frame["sequence_id"].duplicated().any():
        raise RuntimeError(f"Exp081 {split} anchor contains duplicate sequence IDs")
    if not frame["observation_length"].astype(int).eq(16).all():
        raise RuntimeError(f"Exp081 {split} observation length changed")
    if not frame["frame_step"].astype(int).eq(1).all():
        raise RuntimeError(f"Exp081 {split} frame step changed")
    if not frame["frame_sampling"].astype(str).eq("consecutive").all():
        raise RuntimeError(f"Exp081 {split} sampling changed")
    if not (
        frame["end_frame"].astype(int) - frame["start_frame"].astype(int)
    ).eq(15).all():
        raise RuntimeError(f"Exp081 {split} frame window changed")
    if not frame["tte_frames"].between(30, 60).all():
        raise RuntimeError(f"Exp081 {split} TTE changed")
    if not frame["label"].astype(int).isin([0, 1]).all():
        raise RuntimeError(f"Exp081 {split} labels are not binary")
    return frame


def load_anchor(split: str, allow_test: bool = False) -> pd.DataFrame:
    """Load an exact Exp077 anchor while keeping test sealed by default."""
    normalized = "validation" if split == "val" else split
    if normalized == "test":
        _authorize_test_access(allow_test)
        path = EXP077_ROOT / "metrics/prepared_jaad_test.csv"
    elif normalized in {"train", "validation"}:
        path = _development_anchor_path(normalized)
    else:
        raise ValueError("Exp081 split must be train, validation, or test")
    frame = pd.read_csv(path)
    expected_rows = {"train": 1552, "validation": 274, "test": 1501}[normalized]
    if len(frame) != expected_rows:
        raise RuntimeError(
            f"Exp081 {normalized} row count changed: {len(frame)} != {expected_rows}"
        )
    return _validate_anchor(frame, normalized)


def observed_frames(row: Mapping[str, object] | pd.Series) -> tuple[int, ...]:
    start = int(row["start_frame"])
    end = int(row["end_frame"])
    step = int(row.get("frame_step", 1))
    frames = tuple(range(start, end + 1, step))
    if len(frames) != 16 or step != 1:
        raise RuntimeError("Exp081 observed-frame identity changed")
    return frames


def required_video_frames(rows: pd.DataFrame) -> dict[str, tuple[int, ...]]:
    """Return sorted unique observed frame IDs for each development video."""
    required: dict[str, set[int]] = defaultdict(set)
    for _, row in rows.iterrows():
        required[str(row["video_id"])].update(observed_frames(row))
    return {video: tuple(sorted(indices)) for video, indices in sorted(required.items())}


def decode_frames(video_id: str, indices: Iterable[int]) -> dict[int, np.ndarray]:
    """Decode only the requested zero-based frame IDs and fail on seek drift."""
    requested = tuple(sorted({int(index) for index in indices}))
    if any(index < 0 for index in requested):
        raise ValueError("JAAD frame indices must be non-negative")
    path = VIDEO_ROOT / f"{video_id}.mp4"
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open JAAD video: {path}")
    decoded: dict[int, np.ndarray] = {}
    try:
        for index in requested:
            if not capture.set(cv2.CAP_PROP_POS_FRAMES, float(index)):
                raise RuntimeError(f"Cannot seek {video_id} to frame {index}")
            position = int(round(capture.get(cv2.CAP_PROP_POS_FRAMES)))
            if position != index:
                raise RuntimeError(
                    f"JAAD frame seek drift for {video_id}: requested {index}, got {position}"
                )
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError(f"Cannot decode {video_id} frame {index}")
            after = int(round(capture.get(cv2.CAP_PROP_POS_FRAMES)))
            if after != index + 1:
                raise RuntimeError(
                    f"JAAD frame decode drift for {video_id}: requested {index}, ended {after}"
                )
            decoded[index] = frame
    finally:
        capture.release()
    return decoded


def _original_size(root: ElementTree.Element) -> tuple[float, float] | None:
    size = root.find("./meta/task/original_size")
    if size is None:
        return None
    width = size.findtext("width")
    height = size.findtext("height")
    if width is None or height is None:
        return None
    return float(width), float(height)


def parse_all_pedestrian_boxes(
    xml_path: Path | str, indices: Iterable[int]
) -> dict[int, np.ndarray]:
    """Parse every visible pedestrian-like JAAD box at requested frames."""
    requested = {int(index) for index in indices}
    root = ElementTree.parse(xml_path).getroot()
    original_size = _original_size(root)
    boxes: dict[int, list[list[float]]] = {index: [] for index in requested}
    for track in root.findall("track"):
        if track.get("label") not in {"pedestrian", "ped", "people"}:
            continue
        for box in track.findall("box"):
            frame_id = int(box.get("frame", "-1"))
            if frame_id not in requested or box.get("outside", "0") != "0":
                continue
            coordinates = np.asarray(
                [
                    float(box.get("xtl", "nan")),
                    float(box.get("ytl", "nan")),
                    float(box.get("xbr", "nan")),
                    float(box.get("ybr", "nan")),
                ],
                dtype=np.float64,
            )
            if not np.isfinite(coordinates).all():
                raise RuntimeError(f"Invalid JAAD box at frame {frame_id} in {xml_path}")
            if original_size is not None:
                width, height = original_size
                coordinates[[0, 2]] = np.clip(coordinates[[0, 2]], 0.0, width)
                coordinates[[1, 3]] = np.clip(coordinates[[1, 3]], 0.0, height)
            if coordinates[2] <= coordinates[0] or coordinates[3] <= coordinates[1]:
                continue
            boxes[frame_id].append(coordinates.tolist())
    return {
        index: np.asarray(values, dtype=np.float64).reshape((-1, 4))
        for index, values in boxes.items()
    }


def load_all_pedestrian_boxes(
    video_id: str, indices: Iterable[int]
) -> dict[int, np.ndarray]:
    path = ANNOTATION_ROOT / f"{video_id}.xml"
    if not path.is_file():
        raise FileNotFoundError(path)
    return parse_all_pedestrian_boxes(path, indices)

