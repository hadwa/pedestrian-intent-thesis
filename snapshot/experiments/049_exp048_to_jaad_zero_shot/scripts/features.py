"""Frozen non-YOLOP JAAD adapters for Exp049."""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from contracts import EXP048_ROOT, EXPERIMENT_ROOT, OBSERVATION_LENGTH, REPO_ROOT, sha256_file


JAAD_POSE_PATH = EXPERIMENT_ROOT / "cache" / "jaad_keypoints_vitpose_s_exp049_extended.jsonl"
JAAD_POSE_SHA256 = "9eae258cd7fed3b1278aa86f6ab5c508fb759a24c7912d5c57677506ae86e4db"
COCO17_ORDER = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)


def _load_module(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import frozen lineage module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _exp047_core():
    return _load_module(
        "exp049_immutable_exp047_core",
        REPO_ROOT
        / "experiments"
        / "047_mslstm_vitpose_biu_pose_replacement"
        / "scripts"
        / "exp047_core.py",
    )


def _exp048_core():
    return _load_module(
        "exp049_immutable_exp048_features_core",
        EXP048_ROOT / "scripts" / "exp048_core.py",
    )


def _exp048_biu_v1_core():
    return _load_module(
        "exp049_immutable_exp048_biu_v1_features_core",
        EXP048_ROOT / "scripts" / "biu_v1_amendment_core.py",
    )


@dataclass(frozen=True)
class FeatureBundle:
    bbox: np.ndarray
    ped_action: np.ndarray
    ped_look: np.ndarray
    ego_speed: np.ndarray
    ego_acc: np.ndarray
    static_context: np.ndarray
    pose: np.ndarray
    biu_v2: np.ndarray
    biu_v1: np.ndarray
    pose_valid: bool
    valid_joint_fraction: float
    biu_v2_direction_source: str
    action_missing_frames: int
    look_missing_frames: int


class PoseStore:
    """Authenticated exact-key reader for the historical JAAD pose cache."""

    def __init__(self, path: Path = JAAD_POSE_PATH, expected_sha256: str = JAAD_POSE_SHA256):
        self.path = Path(path)
        actual_sha = sha256_file(self.path)
        if actual_sha != expected_sha256:
            raise ValueError(f"JAAD pose cache hash mismatch: {self.path}")
        records: dict[tuple[str, str, int], tuple[np.ndarray, np.ndarray]] = {}
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                raw = json.loads(line)
                key = (str(raw["video"]), str(raw["pid"]), int(raw["frame_idx"]))
                if key in records:
                    raise ValueError(f"Duplicate JAAD pose identity at line {line_number}: {key}")
                coords = np.asarray(raw["keypoints"], dtype=np.float32)
                confidence = np.asarray(raw["kp_conf"], dtype=np.float32)
                if coords.shape != (17, 2) or confidence.shape != (17,):
                    raise ValueError(f"Malformed COCO-17 pose at line {line_number}.")
                records[key] = (coords, confidence)
        self._records = records

    def get_window(
        self,
        video_id: str,
        ped_id: str,
        frames: Sequence[int],
    ) -> tuple[np.ndarray, np.ndarray]:
        identities = [(str(video_id), str(ped_id), int(frame)) for frame in frames]
        missing = [identity for identity in identities if identity not in self._records]
        if missing:
            raise ValueError(f"Missing exact JAAD pose frames: {missing[:3]}")
        coords = np.stack([self._records[key][0] for key in identities]).astype(np.float32)
        confidence = np.stack([self._records[key][1] for key in identities]).astype(np.float32)
        return coords, confidence

    def contains_window(self, video_id: str, ped_id: str, frames: Sequence[int]) -> bool:
        return all((str(video_id), str(ped_id), int(frame)) in self._records for frame in frames)


def _row_values(row: object) -> dict[str, object]:
    if hasattr(row, "_asdict"):
        return dict(row._asdict())
    if hasattr(row, "to_dict"):
        return dict(row.to_dict())
    return dict(row)


def adapt_bbox(row: object, db: Mapping[str, Mapping[str, object]]) -> tuple[np.ndarray, np.ndarray]:
    values = _row_values(row)
    frames = tuple(int(frame) for frame in values["observation_frames"])
    if len(frames) != OBSERVATION_LENGTH:
        raise ValueError("BBox adapter requires exactly 30 observation frames.")
    pedestrian = db[str(values["video_id"])]["ped_annotations"][str(values["ped_id"])]
    source_frames = [int(frame) for frame in pedestrian.get("frames", ())]
    source_boxes = pedestrian.get("bbox", ())
    frame_to_index = {frame: index for index, frame in enumerate(source_frames)}
    pixels = []
    for frame in frames:
        index = frame_to_index.get(frame)
        if index is None or index >= len(source_boxes):
            raise ValueError(f"Missing exact JAAD BBox for frame {frame}.")
        box = np.asarray(source_boxes[index], dtype=np.float32)
        if box.shape != (4,) or not np.isfinite(box).all():
            raise ValueError(f"Malformed JAAD BBox for frame {frame}.")
        pixels.append(box)
    pixel_array = np.stack(pixels).astype(np.float32)
    width = float(values["frame_width"])
    height = float(values["frame_height"])
    x1, y1, x2, y2 = pixel_array.T
    normalized = np.column_stack(
        (((x1 + x2) / 2.0) / width, ((y1 + y2) / 2.0) / height, (x2 - x1) / width, (y2 - y1) / height)
    ).astype(np.float32)
    if (normalized[:, 2:] <= 0).any() or not np.isfinite(normalized).all():
        raise ValueError("JAAD BBox normalization produced an invalid box.")
    return normalized, pixel_array


def adapt_action_look(
    row: object,
    db: Mapping[str, Mapping[str, object]],
) -> tuple[np.ndarray, np.ndarray, int, int]:
    values = _row_values(row)
    frames = tuple(int(frame) for frame in values["observation_frames"])
    pedestrian = db[str(values["video_id"])]["ped_annotations"][str(values["ped_id"])]
    source_frames = [int(frame) for frame in pedestrian.get("frames", ())]
    frame_to_index = {frame: index for index, frame in enumerate(source_frames)}
    behavior = pedestrian.get("behavior", {})
    outputs = []
    missing_counts = []
    for name in ("action", "look"):
        source = behavior.get(name, ())
        output = np.zeros((OBSERVATION_LENGTH, 1), dtype=np.float32)
        missing = 0
        for out_index, frame in enumerate(frames):
            source_index = frame_to_index.get(frame)
            if source_index is None or source_index >= len(source):
                missing += 1
                continue
            value = int(source[source_index])
            if value not in (0, 1):
                raise ValueError(f"Nonbinary JAAD {name} value for frame {frame}: {value}")
            output[out_index, 0] = float(value)
        outputs.append(output)
        missing_counts.append(missing)
    return outputs[0], outputs[1], missing_counts[0], missing_counts[1]


def adapt_static_context(attributes: Mapping[str, object]) -> np.ndarray:
    return np.asarray(_exp047_core().build_static_context(attributes), dtype=np.float32)


def adapt_ego(row: object, db: Mapping[str, Mapping[str, object]]) -> tuple[np.ndarray, np.ndarray]:
    values = _row_values(row)
    vehicle = db[str(values["video_id"])].get("vehicle_annotations", {})
    speed = np.zeros((OBSERVATION_LENGTH, 1), dtype=np.float32)
    for index, frame in enumerate(values["observation_frames"]):
        raw = vehicle.get(int(frame), 0)
        speed[index, 0] = np.clip(float(raw) / 4.0, 0.0, 1.0)
    acceleration = np.zeros((OBSERVATION_LENGTH, 2), dtype=np.float32)
    return speed, acceleration


def _valid_joint_fraction(coords: np.ndarray) -> float:
    available = np.isfinite(coords).all(axis=2) & np.any(coords != 0.0, axis=2)
    return float(available.mean())


def build_non_yolop_features(
    row: object,
    db: Mapping[str, Mapping[str, object]],
    pose_store: PoseStore,
) -> FeatureBundle:
    values = _row_values(row)
    frames = tuple(int(frame) for frame in values["observation_frames"])
    bbox, bbox_pixels = adapt_bbox(values, db)
    action, look, action_missing, look_missing = adapt_action_look(values, db)
    ego_speed, ego_acc = adapt_ego(values, db)
    pedestrian = db[str(values["video_id"])]["ped_annotations"][str(values["ped_id"])]
    static_context = adapt_static_context(pedestrian.get("attributes", {}))
    coords, confidence = pose_store.get_window(str(values["video_id"]), str(values["ped_id"]), frames)
    if coords.shape != (OBSERVATION_LENGTH, 17, 2) or confidence.shape != (OBSERVATION_LENGTH, 17):
        raise ValueError("Pose store returned an invalid 30-frame COCO-17 window.")
    width = int(values["frame_width"])
    height = int(values["frame_height"])
    normalized = coords.copy()
    normalized[:, :, 0] /= float(width)
    normalized[:, :, 1] /= float(height)
    pose = normalized.reshape(OBSERVATION_LENGTH, 34).astype(np.float32)
    biu_v2, _, direction_source = _exp048_core().biu_with_metadata_from_normalized_measurements(
        pose, bbox, width=width, height=height
    )
    biu_v1, _ = _exp048_biu_v1_core().extract_biu_v1_window(coords, confidence)
    joint_fraction = _valid_joint_fraction(coords)
    return FeatureBundle(
        bbox=bbox,
        ped_action=action,
        ped_look=look,
        ego_speed=ego_speed,
        ego_acc=ego_acc,
        static_context=static_context,
        pose=pose,
        biu_v2=np.asarray(biu_v2, dtype=np.float32),
        biu_v1=np.asarray(biu_v1, dtype=np.float32),
        pose_valid=bool(joint_fraction > 0.0),
        valid_joint_fraction=joint_fraction,
        biu_v2_direction_source=str(direction_source),
        action_missing_frames=int(action_missing),
        look_missing_frames=int(look_missing),
    )
