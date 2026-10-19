"""Sealed standardized-JAAD data adapter for Exp089."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import NamedTuple, Sequence

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
EXP_DIR = SCRIPT_DIR.parent
REPO_ROOT = EXP_DIR.parents[1]
ARCHIVE_ROOT = Path("/home/abdelrahman/Hadwa/BIU-PID-Distillation")
SPLIT_ROOT = ARCHIVE_ROOT / "data/jaad_splits_event_16_30_60_overlap08_default"
PRIMARY_POSE = REPO_ROOT / "data/jaad_keypoints_vitpose_s_extended_short_horizon.jsonl"
SUPPLEMENT_POSE = (
    REPO_ROOT
    / "experiments/030_final_confidence_benchmark/jaad30_repair/jaad30_missing_supplemental_keypoints.jsonl"
)
PRIMARY_SHA256 = "4cf675a62f158dcedf79b8037ab39b4fdc94cefe1dd8783eb6e4523d792b2b64"
SUPPLEMENT_SHA256 = "660eb16b1d758f971ff34b8e41f2453ba1c43c5d1410753e77c23481a660c57e"
CACHE_ROOT = EXP_DIR / "cache"
OBS_LENGTH = 16

EXP049_SCRIPTS = REPO_ROOT / "experiments/049_exp048_to_jaad_zero_shot/scripts"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(EXP049_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(EXP049_SCRIPTS))

from jaad_protocol import load_selected_jaad_database  # noqa: E402
from src.biu_v2_extractor import extract_biu_v2  # noqa: E402
from src.features.biu_features import (  # noqa: E402
    AGGREGATIONS,
    BIU_BASE_FEATURES,
    aggregate_biu_sequence,
    compute_frame_biu_features,
    frame_has_invalid_required_keypoints,
    frame_has_low_confidence,
)


class Population(NamedTuple):
    rows: pd.DataFrame
    bbox: np.ndarray
    ego: np.ndarray
    pose: np.ndarray
    biu_v1: np.ndarray
    biu_v2: np.ndarray
    biu_v1_names: tuple[str, ...]
    biu_v2_names: tuple[str, ...]
    source_dir: Path


class StandardizationStats(NamedTuple):
    mean: np.ndarray
    std: np.ndarray


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_pose_hashes() -> None:
    observed = {PRIMARY_POSE: sha256(PRIMARY_POSE), SUPPLEMENT_POSE: sha256(SUPPLEMENT_POSE)}
    expected = {PRIMARY_POSE: PRIMARY_SHA256, SUPPLEMENT_POSE: SUPPLEMENT_SHA256}
    wrong = {str(path): value for path, value in observed.items() if value != expected[path]}
    if wrong:
        raise ValueError(f"Exp089 pose artifact SHA-256 mismatch: {wrong}")


def _read_split(split: str) -> pd.DataFrame:
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unknown split: {split}")
    rows = pd.read_csv(SPLIT_ROOT / f"{split}.csv")
    if rows["sequence_id"].duplicated().any() or not rows["split"].eq(split).all():
        raise ValueError(f"Malformed standardized JAAD {split} manifest")
    if not rows["observation_length"].eq(OBS_LENGTH).all():
        raise ValueError("Exp089 requires exactly 16 observed frames")
    if not rows["tte_frames"].between(30, 60).all():
        raise ValueError("Exp089 requires TTE30--60")
    return rows


def audit_manifests(*, include_test: bool) -> dict[str, object]:
    names = ("train", "val", "test") if include_test else ("train", "val")
    frames = {name: _read_split(name) for name in names}
    result: dict[str, object] = {
        "rows": {name: int(len(frame)) for name, frame in frames.items()},
        "positives": {name: int(frame["label"].sum()) for name, frame in frames.items()},
    }
    if include_test:
        pairs = (("train", "val"), ("train", "test"), ("val", "test"))
        result["video_overlap"] = {
            f"{left}_{right}": int(len(set(frames[left]["video_id"]) & set(frames[right]["video_id"])))
            for left, right in pairs
        }
        result["track_overlap"] = {
            f"{left}_{right}": int(len(set(frames[left]["track_id"]) & set(frames[right]["track_id"])))
            for left, right in pairs
        }
    return result


def _required_keys(rows: pd.DataFrame) -> set[tuple[str, str, int]]:
    required: set[tuple[str, str, int]] = set()
    for row in rows.itertuples(index=False):
        frames = json.loads(row.observed_frames_json)
        if len(frames) != OBS_LENGTH or frames != list(range(int(row.start_frame), int(row.end_frame) + 1)):
            raise ValueError(f"Nonconsecutive observation window: {row.sequence_id}")
        required.update((str(row.video_id), str(row.ped_id), int(frame)) for frame in frames)
    return required


def _load_pose_index(
    required: set[tuple[str, str, int]],
) -> tuple[dict[tuple[str, str, int], tuple[np.ndarray, np.ndarray]], set[tuple[str, str, int]]]:
    _verify_pose_hashes()
    records: dict[tuple[str, str, int], tuple[np.ndarray, np.ndarray]] = {}
    supplement_keys: set[tuple[str, str, int]] = set()
    for path, is_supplement in ((PRIMARY_POSE, False), (SUPPLEMENT_POSE, True)):
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                raw = json.loads(line)
                key = (str(raw["video"]), str(raw["pid"]), int(raw["frame_idx"]))
                if key not in required:
                    continue
                coords = np.asarray(raw["keypoints"], dtype=np.float32)
                confidence = np.asarray(raw["kp_conf"], dtype=np.float32)
                if coords.shape != (17, 2) or confidence.shape != (17,):
                    raise ValueError(f"Malformed pose record {path}:{line_number}")
                if key in records:
                    old_coords, old_confidence = records[key]
                    if not (np.array_equal(coords, old_coords) and np.array_equal(confidence, old_confidence)):
                        raise ValueError(f"Conflicting pose record for {key}")
                    continue
                records[key] = (coords, confidence)
                if is_supplement:
                    supplement_keys.add(key)
    missing = required.difference(records)
    if missing:
        raise ValueError(f"Complete pose pair misses {len(missing)} required frames; first={next(iter(missing))}")
    return records, supplement_keys


def audit_pose_sources(*, include_test: bool) -> dict[str, object]:
    names = ("train", "val", "test") if include_test else ("train", "val")
    rows = pd.concat([_read_split(name) for name in names], ignore_index=True)
    required = _required_keys(rows)
    records, supplement = _load_pose_index(required)
    valid = []
    for coords, _ in records.values():
        valid.extend((np.isfinite(coords).all(axis=1) & np.any(coords != 0.0, axis=1)).tolist())
    return {
        "primary_path": str(PRIMARY_POSE),
        "supplement_path": str(SUPPLEMENT_POSE),
        "primary_sha256": PRIMARY_SHA256,
        "supplement_sha256": SUPPLEMENT_SHA256,
        "required_unique_frames": int(len(required)),
        "primary_covered_frames": int(len(required) - len(supplement)),
        "supplement_covered_frames": int(len(supplement)),
        "missing_frames": int(len(required.difference(records))),
        "valid_joint_fraction": float(np.mean(valid)),
    }


def _extract_biu_v1(coords: np.ndarray, confidence: np.ndarray) -> tuple[np.ndarray, tuple[str, ...]]:
    names = tuple(
        f"{feature}_{aggregation}" for feature in BIU_BASE_FEATURES for aggregation in AGGREGATIONS
    ) + ("biu_valid_frame_count", "biu_low_confidence_frame_count")
    aggregated = aggregate_biu_sequence([compute_frame_biu_features(frame) for frame in coords])
    values = [aggregated[name] for name in names[:-2]]
    values.append(sum(not frame_has_invalid_required_keypoints(frame) for frame in coords))
    values.append(sum(frame_has_low_confidence(frame) for frame in confidence))
    result = np.nan_to_num(np.asarray(values, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    if result.shape != (72,):
        raise RuntimeError(f"Frozen BIU-v1 must remain 72-D, got {result.shape}")
    return result.astype(np.float32), names


def _extract_biu_v2(
    coords: np.ndarray, boxes: np.ndarray, image_size: tuple[int, int]
) -> tuple[np.ndarray, tuple[str, ...], str]:
    features = extract_biu_v2(
        coords,
        kp_conf=None,
        bbox_sequence=boxes,
        image_size=image_size,
        include_optional=False,
    )
    names = tuple(
        name for name, value in features.items()
        if isinstance(value, (int, float, np.integer, np.floating))
    )
    result = np.nan_to_num(
        np.asarray([features[name] for name in names], dtype=np.float64),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    if result.shape != (63,):
        raise RuntimeError(f"Frozen BIU-v2 must remain 63-D, got {result.shape}")
    return result.astype(np.float32), names, str(features["direction_source"])


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_npy(path: Path, value: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, np.asarray(value, dtype=np.float32))
    os.replace(temporary, path)


def _build_source_cache(names: Sequence[str], source: Path) -> None:
    rows = pd.concat([_read_split(name) for name in names], ignore_index=True)
    required = _required_keys(rows)
    pose_index, supplement_keys = _load_pose_index(required)
    videos = sorted(rows["video_id"].astype(str).unique())
    database = load_selected_jaad_database(videos, include_cross=True)
    count = len(rows)
    bbox = np.empty((count, OBS_LENGTH, 4), dtype=np.float32)
    ego = np.empty((count, OBS_LENGTH, 1), dtype=np.float32)
    pose = np.empty((count, OBS_LENGTH, 34), dtype=np.float32)
    biu_v1 = np.empty((count, 72), dtype=np.float32)
    biu_v2 = np.empty((count, 63), dtype=np.float32)
    repaired_counts: list[int] = []
    event_boxes: list[str] = []
    widths: list[int] = []
    heights: list[int] = []
    fallback_count = 0
    v1_names: tuple[str, ...] | None = None
    v2_names: tuple[str, ...] | None = None

    for position, row in enumerate(rows.itertuples(index=False)):
        video = database[str(row.video_id)]
        width, height = int(video["width"]), int(video["height"])
        frames = [int(frame) for frame in json.loads(row.observed_frames_json)]
        boxes = np.asarray(json.loads(row.bbox_sequence_json), dtype=np.float32)
        if boxes.shape != (OBS_LENGTH, 4):
            raise ValueError(f"Malformed BBox window: {row.sequence_id}")
        keys = [(str(row.video_id), str(row.ped_id), frame) for frame in frames]
        values = [pose_index[key] for key in keys]
        coords = np.stack([value[0] for value in values])
        confidence = np.stack([value[1] for value in values])
        bbox[position] = boxes / np.asarray([width, height, width, height], dtype=np.float32)
        pose[position] = (coords / np.asarray([width, height], dtype=np.float32)).reshape(OBS_LENGTH, 34)
        vehicle = video.get("vehicle_annotations", {})
        if any(frame not in vehicle for frame in frames):
            raise ValueError(f"Missing vehicle action in {row.sequence_id}")
        ego[position, :, 0] = np.clip(
            np.asarray([float(vehicle[frame]) for frame in frames], dtype=np.float32) / 4.0,
            0.0,
            1.0,
        )
        v1, names1 = _extract_biu_v1(coords, confidence)
        v2, names2, direction = _extract_biu_v2(coords, boxes, (width, height))
        if v1_names is not None and names1 != v1_names:
            raise RuntimeError("BIU-v1 schema drift")
        if v2_names is not None and names2 != v2_names:
            raise RuntimeError("BIU-v2 schema drift")
        v1_names, v2_names = names1, names2
        biu_v1[position], biu_v2[position] = v1, v2
        fallback_count += int(direction == "bbox_center_x")
        repaired_counts.append(sum(key in supplement_keys for key in keys))

        pedestrian = video["ped_annotations"][str(row.ped_id)]
        frame_to_index = {int(frame): index for index, frame in enumerate(pedestrian["frames"])}
        event_index = frame_to_index.get(int(row.event_frame))
        if event_index is None or event_index >= len(pedestrian["bbox"]):
            raise ValueError(f"Exact event BBox missing for {row.sequence_id}")
        event_boxes.append(json.dumps([float(value) for value in pedestrian["bbox"][event_index]]))
        widths.append(width)
        heights.append(height)

    arrays = {"bbox": bbox, "ego_speed": ego, "pose": pose, "biu_v1": biu_v1, "biu_v2": biu_v2}
    if any(not np.isfinite(values).all() for values in arrays.values()):
        raise ValueError("Exp089 source arrays contain non-finite values")
    rows = rows.copy()
    rows["set_id"] = "jaad"
    rows["frame_width"] = widths
    rows["frame_height"] = heights
    rows["event_bbox_json"] = event_boxes
    rows["pose_repaired_frame_count"] = repaired_counts
    rows["time_to_event_frames"] = rows["tte_frames"]

    source.mkdir(parents=True, exist_ok=True)
    temporary_rows = source / "rows.csv.tmp"
    rows.to_csv(temporary_rows, index=False)
    os.replace(temporary_rows, source / "rows.csv")
    for name, values in arrays.items():
        _atomic_npy(source / f"{name}.npy", values)
    metadata = {
        "splits": list(names),
        "rows": int(count),
        "biu_v1_feature_names": list(v1_names or ()),
        "biu_v2_feature_names": list(v2_names or ()),
        "bbox_direction_fallback_count": int(fallback_count),
        "bbox_direction_fallback_rate": float(fallback_count / count),
        "repaired_frame_assignments": int(sum(repaired_counts)),
        "repaired_windows": int(sum(value > 0 for value in repaired_counts)),
        "primary_pose_sha256": PRIMARY_SHA256,
        "supplement_pose_sha256": SUPPLEMENT_SHA256,
    }
    _atomic_json(source / "metadata.json", metadata)


def load_population(population: str, *, allow_test: bool) -> Population:
    if population == "trainval":
        names, source = ("train", "val"), CACHE_ROOT / "trainval_source"
    elif population == "test":
        if not allow_test:
            raise PermissionError("final JAAD test access requires allow_test=True after the checkpoint gate")
        names, source = ("test",), CACHE_ROOT / "test_source"
    else:
        raise ValueError(f"Unknown population: {population}")
    if not (source / "metadata.json").is_file():
        _build_source_cache(names, source)
    rows = pd.read_csv(source / "rows.csv")
    metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    arrays = {
        name: np.asarray(np.load(source / f"{name}.npy"), dtype=np.float32)
        for name in ("bbox", "ego_speed", "pose", "biu_v1", "biu_v2")
    }
    if any(len(values) != len(rows) or not np.isfinite(values).all() for values in arrays.values()):
        raise ValueError("Cached Exp089 population is incomplete or non-finite")
    if list(metadata["splits"]) != list(names):
        raise ValueError("Cached Exp089 population has wrong split membership")
    return Population(
        rows,
        arrays["bbox"],
        arrays["ego_speed"],
        arrays["pose"],
        arrays["biu_v1"],
        arrays["biu_v2"],
        tuple(metadata["biu_v1_feature_names"]),
        tuple(metadata["biu_v2_feature_names"]),
        source,
    )


def build_union(
    biu_v1: np.ndarray,
    biu_v2: np.ndarray,
    v1_names: Sequence[str],
    v2_names: Sequence[str],
) -> tuple[np.ndarray, tuple[str, ...]]:
    names = tuple([f"v1::{name}" for name in v1_names] + [f"v2::{name}" for name in v2_names])
    if len(names) != 135 or len(names) != len(set(names)):
        raise ValueError("BIU union must contain 135 unique source-namespaced features")
    values = np.concatenate([np.asarray(biu_v1), np.asarray(biu_v2)], axis=1).astype(np.float32)
    return values, names


def standardize_from_training(
    values: np.ndarray, train_mask: np.ndarray
) -> tuple[np.ndarray, StandardizationStats]:
    matrix = np.asarray(values, dtype=np.float32)
    mask = np.asarray(train_mask, dtype=bool)
    if matrix.ndim != 2 or mask.shape != (len(matrix),) or not mask.any():
        raise ValueError("Standardization requires aligned 2-D values and training mask")
    mean = matrix[mask].mean(axis=0, dtype=np.float64).astype(np.float32)
    std = matrix[mask].std(axis=0, dtype=np.float64).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    transformed = ((matrix - mean) / std).astype(np.float32)
    if not np.isfinite(transformed).all():
        raise ValueError("Training-only standardization produced non-finite values")
    return transformed, StandardizationStats(mean, std)


def build_event_bbox_targets(rows: pd.DataFrame) -> np.ndarray:
    targets = np.empty((len(rows), 4), dtype=np.float32)
    for position, row in enumerate(rows.itertuples(index=False)):
        box = np.asarray(json.loads(row.event_bbox_json), dtype=np.float32)
        if box.shape != (4,) or int(row.event_frame) <= int(row.last_observed_frame):
            raise ValueError(f"Invalid future endpoint target for {row.sequence_id}")
        targets[position] = box / np.asarray(
            [row.frame_width, row.frame_height, row.frame_width, row.frame_height], dtype=np.float32
        )
    if not np.isfinite(targets).all():
        raise ValueError("Endpoint targets contain non-finite values")
    return targets
