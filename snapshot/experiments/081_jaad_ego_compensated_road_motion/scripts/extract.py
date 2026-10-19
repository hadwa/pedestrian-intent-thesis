"""Label-sealed compact feature extraction and caching for Exp081."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = SCRIPT_DIR.parent

IDENTITY_COLUMNS = (
    "sequence_id",
    "split",
    "video_id",
    "ped_id",
    "track_id",
    "start_frame",
    "end_frame",
    "observation_length",
    "frame_step",
    "frame_sampling",
    "bbox_sequence_json",
)
TARGET_COLUMNS = frozenset(
    {
        "label",
        "event_frame",
        "last_observed_frame",
        "tte_frames",
        "crossing_attr",
        "crossing_point",
        "decision_point",
        "has_observed_crossing",
        "observed_crosses_json",
    }
)
PART_ARRAYS = {
    "proxy_15d": 15,
    "road_raw_60d": 60,
    "road_compensated_60d": 60,
}


def _load_sibling(filename: str, name: str):
    path = SCRIPT_DIR / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load Exp081 module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CONTRACT = _load_sibling("contract.py", "exp081_contract_for_extract")


def _feature_modules():
    return (
        _load_sibling("motion.py", "exp081_motion_for_extract"),
        _load_sibling("road_context.py", "exp081_road_for_extract"),
    )


def validate_identity_rows(rows: pd.DataFrame) -> pd.DataFrame:
    forbidden = sorted(TARGET_COLUMNS.intersection(rows.columns))
    if forbidden:
        raise ValueError(f"Exp081 perception rows contain targets: {forbidden}")
    missing = sorted(set(IDENTITY_COLUMNS).difference(rows.columns))
    if missing:
        raise ValueError(f"Exp081 perception rows are missing identity fields: {missing}")
    checked = rows.loc[:, IDENTITY_COLUMNS].reset_index(drop=True).copy()
    if checked.empty or checked["sequence_id"].astype(str).duplicated().any():
        raise ValueError("Exp081 perception sequence IDs must be nonempty and unique")
    starts = checked["start_frame"].to_numpy(np.int64)
    ends = checked["end_frame"].to_numpy(np.int64)
    exact = checked["observation_length"].to_numpy(np.int64) == 16
    exact = (
        np.asarray(exact)
        & (checked["frame_step"].to_numpy(np.int64) == 1)
        & (checked["frame_sampling"].astype(str).to_numpy() == "consecutive")
        & ((ends - starts) == 15)
    )
    if not exact.all():
        raise ValueError("Exp081 perception rows must be exact consecutive 16-frame windows")
    return checked


def required_frames(rows: pd.DataFrame) -> dict[str, tuple[int, ...]]:
    checked = validate_identity_rows(rows)
    required: dict[str, set[int]] = defaultdict(set)
    for row in checked.itertuples(index=False):
        required[str(row.video_id)].update(
            range(int(row.start_frame), int(row.end_frame) + 1)
        )
    return {
        video_id: tuple(sorted(frame_ids))
        for video_id, frame_ids in sorted(required.items())
    }


def load_identity_rows(split: str) -> pd.DataFrame:
    sources = {
        "train": "jaad_train_split",
        "validation": "jaad_validation_split",
    }
    if split not in sources:
        raise ValueError("Development extraction supports only train and validation")
    rows = pd.read_csv(CONTRACT.resolve_source(sources[split]), usecols=IDENTITY_COLUMNS)
    expected_split = "val" if split == "validation" else split
    if not rows["split"].astype(str).eq(expected_split).all():
        raise RuntimeError(f"Exp081 {split} identity membership changed")
    expected_rows = {"train": 1552, "validation": 274}[split]
    if len(rows) != expected_rows:
        raise RuntimeError(f"Exp081 {split} identity row count changed")
    return validate_identity_rows(rows)


def load_test_identity_rows(
    path: Path | str = Path(
        "/home/abdelrahman/Hadwa/BIU-PID-Distillation/data/jaad_splits_event_16_30_60_overlap08_default/test.csv"
    ),
) -> pd.DataFrame:
    """Load test identities only after the runner has authenticated advancement."""
    rows = pd.read_csv(path, usecols=IDENTITY_COLUMNS)
    if len(rows) != 1501 or not rows["split"].astype(str).eq("test").all():
        raise RuntimeError("Exp081 test identity membership changed")
    return validate_identity_rows(rows)


def _validate_part_values(values: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    expected_keys = {"sequence_id", *PART_ARRAYS}
    if set(values) != expected_keys:
        raise ValueError(f"Exp081 compact part keys changed: {sorted(values)}")
    sequence_ids = np.asarray(values["sequence_id"]).astype(str)
    if sequence_ids.ndim != 1 or not len(sequence_ids) or len(set(sequence_ids)) != len(sequence_ids):
        raise ValueError("Exp081 compact part IDs must be unique and one-dimensional")
    checked: dict[str, np.ndarray] = {"sequence_id": sequence_ids}
    for name, dimension in PART_ARRAYS.items():
        array = np.asarray(values[name], dtype=np.float32)
        if array.shape != (len(sequence_ids), dimension) or not np.isfinite(array).all():
            raise ValueError(f"Exp081 compact part {name} has an invalid shape or value")
        checked[name] = array
    return checked


def write_part(path: Path | str, values: Mapping[str, np.ndarray]) -> dict[str, Any]:
    checked = _validate_part_values(values)
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.stem}.", suffix=".npz", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(temporary, **checked)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    record = CONTRACT.artifact_record(destination)
    record.update(
        {
            "rows": len(checked["sequence_id"]),
            "first_sequence_id": str(checked["sequence_id"][0]),
            "last_sequence_id": str(checked["sequence_id"][-1]),
            "feature_shapes": {
                name: list(checked[name].shape) for name in PART_ARRAYS
            },
        }
    )
    return record


def load_part(path: Path | str, record: Mapping[str, Any]) -> dict[str, np.ndarray]:
    actual = CONTRACT.artifact_record(path)
    if (
        actual["sha256"] != record.get("sha256")
        or actual["size_bytes"] != record.get("size_bytes")
    ):
        raise RuntimeError(f"Exp081 compact part authentication failed: {path}")
    with np.load(path, allow_pickle=False) as bundle:
        values = {name: np.asarray(bundle[name]) for name in ("sequence_id", *PART_ARRAYS)}
    checked = _validate_part_values(values)
    if len(checked["sequence_id"]) != int(record.get("rows", -1)):
        raise RuntimeError(f"Exp081 compact part row count changed: {path}")
    return checked


def load_context(
    root: Path | str,
    split: str,
    expected_ids: np.ndarray,
    *,
    manifest_name: str = "extraction_manifest.json",
) -> pd.DataFrame:
    """Authenticate compact parts and expose the exact model context ABI."""
    root = Path(root)
    try:
        manifest = json.loads(
            (root / "metrics" / manifest_name).read_text(encoding="utf-8")
        )
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise RuntimeError("Exp081 extraction manifest is missing or malformed") from exc
    if (
        manifest.get("target_labels_passed_to_perception") is not False
        or manifest.get("full_resolution_masks_persisted") is not False
    ):
        raise RuntimeError("Exp081 extraction manifest violates the frozen contract")
    expected_test_access = split == "test"
    if manifest.get("test_accessed") is not expected_test_access:
        raise RuntimeError("Exp081 extraction manifest has the wrong test-access state")
    records = [
        record for record in manifest.get("parts", []) if record.get("split") == split
    ]
    if not records:
        raise RuntimeError(f"Exp081 extraction manifest has no {split} parts")
    bundles = [load_part(record["path"], record) for record in records]
    ids = np.concatenate([bundle["sequence_id"] for bundle in bundles]).astype(str)
    expected = np.asarray(expected_ids).astype(str)
    if (
        len(set(ids)) != len(ids)
        or len(set(expected)) != len(expected)
        or set(ids) != set(expected)
    ):
        raise RuntimeError("Exp081 compact context IDs differ from the anchor IDs")
    positions = {sequence_id: index for index, sequence_id in enumerate(ids)}
    order = np.asarray([positions[sequence_id] for sequence_id in expected], dtype=np.int64)
    model_module = _load_sibling("model.py", "exp081_model_for_context")
    ids = ids[order]
    proxy = np.concatenate([bundle["proxy_15d"] for bundle in bundles])[order]
    raw = np.concatenate([bundle["road_raw_60d"] for bundle in bundles])[order]
    compensated = np.concatenate(
        [bundle["road_compensated_60d"] for bundle in bundles]
    )[order]
    context = pd.DataFrame({"sequence_id": ids})
    context = pd.concat(
        (
            context,
            pd.DataFrame(proxy, columns=model_module.proxy_columns()),
            pd.DataFrame(raw, columns=model_module.raw_road_columns()),
            pd.DataFrame(
                compensated, columns=model_module.compensated_road_columns()
            ),
        ),
        axis=1,
    )
    if context.shape != (len(ids), 136) or context.isna().any().any():
        raise RuntimeError("Exp081 compact context model ABI failed")
    return context


def _target_boxes(row: Any) -> np.ndarray:
    try:
        values = np.asarray(json.loads(str(row.bbox_sequence_json)), dtype=np.float64)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid Exp081 BBox JSON for {row.sequence_id}") from exc
    if values.shape != (16, 4) or not np.isfinite(values).all():
        raise ValueError(f"Invalid Exp081 BBox shape for {row.sequence_id}")
    if np.any(values[:, 2:] <= values[:, :2]):
        raise ValueError(f"Non-positive Exp081 BBox for {row.sequence_id}")
    return values


def _infer_unique_masks(
    frame_ids: tuple[int, ...],
    frames: Mapping[int, np.ndarray],
    extractor: Any,
    batch_size: int,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    if batch_size <= 0:
        raise ValueError("Exp081 TwinLite batch size must be positive")
    roads: dict[int, np.ndarray] = {}
    lanes: dict[int, np.ndarray] = {}
    for start in range(0, len(frame_ids), batch_size):
        batch_ids = frame_ids[start : start + batch_size]
        images = [np.asarray(frames[frame_id]) for frame_id in batch_ids]
        road_batch, lane_batch = extractor.predict(images)
        if len(road_batch) != len(batch_ids) or len(lane_batch) != len(batch_ids):
            raise RuntimeError("Exp081 TwinLite batch output count changed")
        for frame_id, image, road, lane in zip(
            batch_ids, images, road_batch, lane_batch
        ):
            road_values = np.asarray(road, dtype=bool)
            lane_values = np.asarray(lane, dtype=bool)
            if (
                road_values.shape != image.shape[:2]
                or lane_values.shape != image.shape[:2]
            ):
                raise RuntimeError("Exp081 TwinLite masks changed source geometry")
            roads[frame_id] = road_values
            lanes[frame_id] = lane_values
    return roads, lanes


def extract_video_rows(
    rows: pd.DataFrame,
    frames: Mapping[int, np.ndarray],
    pedestrian_boxes: Mapping[int, np.ndarray],
    extractor: Any,
    motion_settings: Mapping[str, Any],
    *,
    batch_size: int,
) -> tuple[dict[str, np.ndarray], pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Extract compact features for one video's label-free window rows."""
    checked = validate_identity_rows(rows)
    video_ids = tuple(checked["video_id"].astype(str).unique())
    if len(video_ids) != 1:
        raise ValueError("Exp081 video extraction requires exactly one video")
    needed = required_frames(checked)[video_ids[0]]
    if set(frames) != set(needed) or set(pedestrian_boxes) != set(needed):
        raise RuntimeError("Exp081 exact frame or pedestrian-mask identity changed")
    roads, lanes = _infer_unique_masks(needed, frames, extractor, batch_size)
    motion_module, road_module = _feature_modules()
    sequence_ids: list[str] = []
    proxy_values: list[np.ndarray] = []
    raw_values: list[np.ndarray] = []
    compensated_values: list[np.ndarray] = []
    motion_rows: list[dict[str, Any]] = []
    road_rows: list[dict[str, Any]] = []
    for row in checked.itertuples(index=False):
        frame_ids = tuple(range(int(row.start_frame), int(row.end_frame) + 1))
        frame_values = [frames[frame_id] for frame_id in frame_ids]
        all_boxes = [
            np.asarray(pedestrian_boxes[frame_id], dtype=np.float64).reshape((-1, 4))
            for frame_id in frame_ids
        ]
        target_boxes = _target_boxes(row)
        movement = motion_module.estimate_window_motion(
            frame_values, all_boxes, motion_settings
        )
        raw_sequence = road_module.uncompensated_sequence(
            [roads[frame_id] for frame_id in frame_ids],
            [lanes[frame_id] for frame_id in frame_ids],
            target_boxes,
        )
        compensated_sequence = road_module.compensated_sequence(
            [roads[frame_id] for frame_id in frame_ids],
            [lanes[frame_id] for frame_id in frame_ids],
            target_boxes,
            movement.to_first,
        )
        sequence_ids.append(str(row.sequence_id))
        proxy_values.append(movement.proxy.astype(np.float32))
        raw_values.append(road_module.summarize(raw_sequence))
        compensated_values.append(road_module.summarize(compensated_sequence))
        valid = movement.valid
        raw_displacement = movement.background_displacement[valid]
        residual_displacement = movement.residual_displacement[valid]
        raw_median = float(np.median(raw_displacement)) if len(raw_displacement) else 0.0
        residual_median = (
            float(np.median(residual_displacement)) if len(residual_displacement) else 0.0
        )
        reduction = 1.0 - residual_median / raw_median if raw_median > 1e-12 else 0.0
        motion_rows.append(
            {
                "sequence_id": str(row.sequence_id),
                "split": str(row.split),
                "valid_transition_count": int(valid.sum()),
                "valid_transition_fraction": float(valid.mean()),
                "mean_tracked_point_count": float(movement.tracked_point_count.mean()),
                "mean_valid_inlier_fraction": float(
                    movement.inlier_fraction[valid].mean() if valid.any() else 0.0
                ),
                "mean_forward_backward_error": float(
                    movement.forward_backward_error[valid].mean()
                    if valid.any()
                    else 0.0
                ),
                "median_background_displacement_px": raw_median,
                "median_residual_displacement_px": residual_median,
                "median_displacement_reduction_fraction": reduction,
            }
        )
        road_rows.append(
            {
                "sequence_id": str(row.sequence_id),
                "split": str(row.split),
                "road_geometry_valid_fraction": float(raw_sequence[:, 6].mean()),
                "lane_geometry_valid_fraction": float(raw_sequence[:, 12].mean()),
                "completely_invalid": bool(
                    not raw_sequence[:, 6].any() and not raw_sequence[:, 12].any()
                ),
            }
        )
    values = _validate_part_values(
        {
            "sequence_id": np.asarray(sequence_ids),
            "proxy_15d": np.stack(proxy_values),
            "road_raw_60d": np.stack(raw_values),
            "road_compensated_60d": np.stack(compensated_values),
        }
    )
    motion_quality = pd.DataFrame(motion_rows)
    road_quality = pd.DataFrame(road_rows)
    report = {
        "video_id": video_ids[0],
        "unique_frames": len(needed),
        "windows": len(checked),
        "target_columns_seen": False,
        "full_resolution_masks_persisted": False,
        "valid_transition_fraction": float(
            motion_quality["valid_transition_count"].sum() / (15 * len(checked))
        ),
    }
    return values, motion_quality, road_quality, report


def evaluate_quality_gate(motion_quality: pd.DataFrame) -> dict[str, Any]:
    """Evaluate the frozen label-free motion feasibility gate."""
    required = {
        "sequence_id",
        "split",
        "valid_transition_count",
        "median_displacement_reduction_fraction",
    }
    missing = required - set(motion_quality.columns)
    if missing:
        raise ValueError(f"Exp081 motion quality is missing: {sorted(missing)}")
    settings = CONTRACT.load_registration()["extraction"]
    sample_count = int(settings["audit_sample_windows_per_split"])
    minimum_valid = float(settings["minimum_valid_transition_fraction"])
    minimum_reduction = float(
        settings["minimum_median_displacement_reduction_fraction"]
    )
    results: dict[str, Any] = {}
    for source_name, report_name in (("train", "train"), ("val", "validation")):
        subset = motion_quality.loc[
            motion_quality["split"].astype(str).eq(source_name)
        ].sort_values("sequence_id", kind="stable")
        if len(subset) < sample_count:
            raise RuntimeError(f"Exp081 {report_name} quality sample is incomplete")
        valid_fraction = float(subset["valid_transition_count"].sum() / (15 * len(subset)))
        audit = subset.iloc[:sample_count]
        median_reduction = float(
            audit["median_displacement_reduction_fraction"].median()
        )
        results[report_name] = {
            "windows": len(subset),
            "valid_transition_fraction": valid_fraction,
            "valid_transition_pass": valid_fraction >= minimum_valid,
            "audit_sample_windows": sample_count,
            "median_displacement_reduction_fraction": median_reduction,
            "displacement_reduction_pass": median_reduction >= minimum_reduction,
        }
    passed = all(
        record["valid_transition_pass"] and record["displacement_reduction_pass"]
        for record in results.values()
    )
    return {"status": "pass" if passed else "fail", "splits": results}


def _load_twinlite_adapter():
    """Load the authenticated Exp075 adapter with its own contract binding."""
    adapter_path = CONTRACT.resolve_source("exp075_twinlite_adapter")
    CONTRACT.resolve_source("twinlite_checkpoint")
    contract_path = adapter_path.parent / "contract.py"
    old_contract = sys.modules.get("contract")
    try:
        contract_spec = importlib.util.spec_from_file_location("contract", contract_path)
        if contract_spec is None or contract_spec.loader is None:
            raise RuntimeError("Cannot load authenticated Exp075 contract")
        exp075_contract = importlib.util.module_from_spec(contract_spec)
        sys.modules["contract"] = exp075_contract
        contract_spec.loader.exec_module(exp075_contract)
        adapter_spec = importlib.util.spec_from_file_location(
            "exp081_authenticated_twinlite_adapter", adapter_path
        )
        if adapter_spec is None or adapter_spec.loader is None:
            raise RuntimeError("Cannot load authenticated Exp075 TwinLite adapter")
        adapter = importlib.util.module_from_spec(adapter_spec)
        sys.modules[adapter_spec.name] = adapter
        adapter_spec.loader.exec_module(adapter)
    finally:
        if old_contract is None:
            sys.modules.pop("contract", None)
        else:
            sys.modules["contract"] = old_contract
    return adapter


def create_twinlite_extractor(*, device: str = "cuda", half: bool = True):
    adapter = _load_twinlite_adapter()
    return adapter.TwinLiteExtractor(device=device, half=half)


def _decode_exact_batch(
    capture: cv2.VideoCapture, video_id: str, frame_ids: tuple[int, ...]
) -> list[np.ndarray]:
    images: list[np.ndarray] = []
    for frame_id in frame_ids:
        if not capture.set(cv2.CAP_PROP_POS_FRAMES, float(frame_id)):
            raise RuntimeError(f"Cannot seek {video_id} to frame {frame_id}")
        before = int(round(capture.get(cv2.CAP_PROP_POS_FRAMES)))
        if before != frame_id:
            raise RuntimeError(
                f"Exp081 frame seek drift for {video_id}: {frame_id} != {before}"
            )
        ok, image = capture.read()
        if not ok or image is None:
            raise RuntimeError(f"Cannot decode {video_id} frame {frame_id}")
        after = int(round(capture.get(cv2.CAP_PROP_POS_FRAMES)))
        if after != frame_id + 1:
            raise RuntimeError(
                f"Exp081 frame decode drift for {video_id}: {frame_id + 1} != {after}"
            )
        images.append(image)
    return images


def _window_quality(
    sequence_id: str, split: str, movement: Any, raw_sequence: np.ndarray
) -> tuple[dict[str, Any], dict[str, Any]]:
    valid = movement.valid
    raw_displacement = movement.background_displacement[valid]
    residual_displacement = movement.residual_displacement[valid]
    raw_median = float(np.median(raw_displacement)) if len(raw_displacement) else 0.0
    residual_median = (
        float(np.median(residual_displacement)) if len(residual_displacement) else 0.0
    )
    reduction = 1.0 - residual_median / raw_median if raw_median > 1e-12 else 0.0
    motion_record = {
        "sequence_id": sequence_id,
        "split": split,
        "valid_transition_count": int(valid.sum()),
        "valid_transition_fraction": float(valid.mean()),
        "mean_tracked_point_count": float(movement.tracked_point_count.mean()),
        "mean_valid_inlier_fraction": float(
            movement.inlier_fraction[valid].mean() if valid.any() else 0.0
        ),
        "mean_forward_backward_error": float(
            movement.forward_backward_error[valid].mean() if valid.any() else 0.0
        ),
        "median_background_displacement_px": raw_median,
        "median_residual_displacement_px": residual_median,
        "median_displacement_reduction_fraction": reduction,
    }
    road_record = {
        "sequence_id": sequence_id,
        "split": split,
        "road_geometry_valid_fraction": float(raw_sequence[:, 6].mean()),
        "lane_geometry_valid_fraction": float(raw_sequence[:, 12].mean()),
        "completely_invalid": bool(
            not raw_sequence[:, 6].any() and not raw_sequence[:, 12].any()
        ),
    }
    return motion_record, road_record


def extract_video_streaming(
    rows: pd.DataFrame,
    extractor: Any,
    motion_settings: Mapping[str, Any],
    *,
    batch_size: int,
) -> tuple[dict[str, np.ndarray], pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Stream one real JAAD video and persist no full-resolution masks."""
    checked = validate_identity_rows(rows)
    video_ids = tuple(checked["video_id"].astype(str).unique())
    if len(video_ids) != 1:
        raise ValueError("Exp081 streaming extraction requires one video")
    video_id = video_ids[0]
    needed = required_frames(checked)[video_id]
    jaad_data = _load_sibling("jaad_data.py", "exp081_jaad_for_streaming")
    motion_module, road_module = _feature_modules()
    all_boxes = jaad_data.load_all_pedestrian_boxes(video_id, needed)
    parsed_boxes = [_target_boxes(row) for row in checked.itertuples(index=False)]
    static = np.zeros((len(checked), 16, 13), dtype=np.float32)
    users: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for row_index, row in enumerate(checked.itertuples(index=False)):
        for relative, frame_id in enumerate(
            range(int(row.start_frame), int(row.end_frame) + 1)
        ):
            users[frame_id].append((row_index, relative))

    pair_motion: dict[tuple[int, int], Any] = {}
    video_path = jaad_data.VIDEO_ROOT / f"{video_id}.mp4"
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(video_path)
    previous_id: int | None = None
    previous_image: np.ndarray | None = None
    frame_shape: tuple[int, int] | None = None
    drivable_zero = 0
    lane_zero = 0
    try:
        for start in range(0, len(needed), batch_size):
            batch_ids = needed[start : start + batch_size]
            images = _decode_exact_batch(capture, video_id, batch_ids)
            road_masks, lane_masks = extractor.predict(images)
            if len(road_masks) != len(images) or len(lane_masks) != len(images):
                raise RuntimeError("Exp081 TwinLite streaming output count changed")
            for frame_id, image, road_mask, lane_mask in zip(
                batch_ids, images, road_masks, lane_masks
            ):
                road = np.asarray(road_mask, dtype=bool)
                lane = np.asarray(lane_mask, dtype=bool)
                if road.shape != image.shape[:2] or lane.shape != image.shape[:2]:
                    raise RuntimeError("Exp081 TwinLite streaming geometry changed")
                frame_shape = image.shape[:2]
                drivable_zero += int(not road.any())
                lane_zero += int(not lane.any())
                for row_index, relative in users[frame_id]:
                    static[row_index, relative] = road_module.static_features(
                        road, lane, parsed_boxes[row_index][relative]
                    )
                if previous_id is not None and frame_id == previous_id + 1:
                    pair_motion[(previous_id, frame_id)] = motion_module.estimate_pair_motion(
                        previous_image,
                        image,
                        all_boxes[previous_id],
                        all_boxes[frame_id],
                        motion_settings,
                    )
                previous_id = frame_id
                previous_image = image
    finally:
        capture.release()
    if frame_shape is None:
        raise RuntimeError(f"Exp081 decoded no frames for {video_id}")

    sequence_ids: list[str] = []
    proxies: list[np.ndarray] = []
    raw_summaries: list[np.ndarray] = []
    compensated_summaries: list[np.ndarray] = []
    motion_records: list[dict[str, Any]] = []
    road_records: list[dict[str, Any]] = []
    for row_index, row in enumerate(checked.itertuples(index=False)):
        frame_ids = tuple(range(int(row.start_frame), int(row.end_frame) + 1))
        pairs = [pair_motion[(frame_ids[index], frame_ids[index + 1])] for index in range(15)]
        movement = motion_module.assemble_window_motion(
            pairs, width=frame_shape[1], height=frame_shape[0]
        )
        raw = road_module.assemble_uncompensated(static[row_index], parsed_boxes[row_index])
        compensated = road_module.assemble_compensated(
            static[row_index], parsed_boxes[row_index], movement.to_first, raw=raw
        )
        sequence_id = str(row.sequence_id)
        sequence_ids.append(sequence_id)
        proxies.append(movement.proxy)
        raw_summaries.append(road_module.summarize(raw))
        compensated_summaries.append(road_module.summarize(compensated))
        motion_record, road_record = _window_quality(
            sequence_id, str(row.split), movement, raw
        )
        motion_records.append(motion_record)
        road_records.append(road_record)
    values = _validate_part_values(
        {
            "sequence_id": np.asarray(sequence_ids),
            "proxy_15d": np.stack(proxies),
            "road_raw_60d": np.stack(raw_summaries),
            "road_compensated_60d": np.stack(compensated_summaries),
        }
    )
    annotation_path = jaad_data.ANNOTATION_ROOT / f"{video_id}.xml"
    report = {
        "video_id": video_id,
        "unique_frames": len(needed),
        "windows": len(checked),
        "target_columns_seen": False,
        "full_resolution_masks_persisted": False,
        "drivable_all_zero_frames": drivable_zero,
        "lane_all_zero_frames": lane_zero,
        "source_video": CONTRACT.artifact_record(video_path),
        "source_annotation": CONTRACT.artifact_record(annotation_path),
    }
    return values, pd.DataFrame(motion_records), pd.DataFrame(road_records), report


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    CONTRACT.atomic_write_text(path, frame.to_csv(index=False, lineterminator="\n"))


def run_development_extraction(
    root: Path | str,
    extractor: Any,
    *,
    batch_size: int | None = None,
) -> tuple[list[Path], dict[str, Any]]:
    """Extract authenticated train/validation compact parts and quality reports."""
    root = Path(root)
    registration = CONTRACT.load_registration()
    selected_batch = int(batch_size or registration["extraction"]["batch_size"])
    part_records: list[dict[str, Any]] = []
    video_reports: list[dict[str, Any]] = []
    all_motion: list[pd.DataFrame] = []
    all_road: list[pd.DataFrame] = []
    outputs: list[Path] = []
    for split in ("train", "validation"):
        rows = load_identity_rows(split)
        for video_id, video_rows in rows.groupby("video_id", sort=True):
            values, motion_quality, road_quality, report = extract_video_streaming(
                video_rows.reset_index(drop=True),
                extractor,
                registration["motion_estimator"],
                batch_size=selected_batch,
            )
            part_path = root / "cache/development_parts" / split / f"{video_id}.npz"
            record = write_part(part_path, values)
            record.update({"split": split, "video_id": str(video_id)})
            part_records.append(record)
            report.update({"split": split, "part": record})
            video_reports.append(report)
            all_motion.append(motion_quality)
            all_road.append(road_quality)
            outputs.append(part_path)
            print(
                f"Exp081 extract-development split={split} video={video_id} "
                f"windows={len(video_rows)} frames={report['unique_frames']}",
                flush=True,
            )
    motion_quality = pd.concat(all_motion, ignore_index=True)
    road_quality = pd.concat(all_road, ignore_index=True)
    gate = evaluate_quality_gate(motion_quality)
    motion_path = root / "metrics/motion_quality.csv"
    road_path = root / "metrics/road_geometry_quality.csv"
    gate_path = root / "metrics/extraction_quality_gate.json"
    manifest_path = root / "metrics/extraction_manifest.json"
    _atomic_csv(motion_path, motion_quality)
    _atomic_csv(road_path, road_quality)
    CONTRACT.atomic_write_json(gate_path, gate)
    manifest = {
        "experiment": registration["experiment"],
        "splits": ["train", "validation"],
        "target_labels_passed_to_perception": False,
        "test_accessed": False,
        "full_resolution_masks_persisted": False,
        "batch_size": selected_batch,
        "checkpoint_sha256": registration["twinlite"]["checkpoint_sha256"],
        "parts": part_records,
        "videos": video_reports,
    }
    CONTRACT.atomic_write_json(manifest_path, manifest)
    outputs.extend([motion_path, road_path, gate_path, manifest_path])
    if gate["status"] != "pass":
        raise RuntimeError("Exp081 development extraction quality gate failed")
    return outputs, gate


def run_test_extraction(
    root: Path | str,
    extractor: Any,
    rows: pd.DataFrame,
    *,
    batch_size: int | None = None,
) -> list[Path]:
    """Extract the already-authorized test cohort exactly once."""
    root = Path(root)
    checked = validate_identity_rows(rows)
    if len(checked) != 1501 or not checked["split"].astype(str).eq("test").all():
        raise RuntimeError("Exp081 authorized test cohort changed")
    registration = CONTRACT.load_registration()
    selected_batch = int(batch_size or registration["extraction"]["batch_size"])
    part_records: list[dict[str, Any]] = []
    video_reports: list[dict[str, Any]] = []
    all_motion: list[pd.DataFrame] = []
    all_road: list[pd.DataFrame] = []
    outputs: list[Path] = []
    for video_id, video_rows in checked.groupby("video_id", sort=True):
        values, motion_quality, road_quality, report = extract_video_streaming(
            video_rows.reset_index(drop=True),
            extractor,
            registration["motion_estimator"],
            batch_size=selected_batch,
        )
        part_path = root / "cache/test_parts" / f"{video_id}.npz"
        record = write_part(part_path, values)
        record.update({"split": "test", "video_id": str(video_id)})
        part_records.append(record)
        report.update({"split": "test", "part": record})
        video_reports.append(report)
        all_motion.append(motion_quality)
        all_road.append(road_quality)
        outputs.append(part_path)
        print(
            f"Exp081 extract-test video={video_id} windows={len(video_rows)} "
            f"frames={report['unique_frames']}",
            flush=True,
        )
    motion_path = root / "metrics/test_motion_quality.csv"
    road_path = root / "metrics/test_road_geometry_quality.csv"
    manifest_path = root / "metrics/test_extraction_manifest.json"
    _atomic_csv(motion_path, pd.concat(all_motion, ignore_index=True))
    _atomic_csv(road_path, pd.concat(all_road, ignore_index=True))
    test_split_path = Path(
        "/home/abdelrahman/Hadwa/BIU-PID-Distillation/data/jaad_splits_event_16_30_60_overlap08_default/test.csv"
    )
    CONTRACT.atomic_write_json(
        manifest_path,
        {
            "experiment": registration["experiment"],
            "splits": ["test"],
            "target_labels_passed_to_perception": False,
            "test_accessed": True,
            "full_resolution_masks_persisted": False,
            "batch_size": selected_batch,
            "checkpoint_sha256": registration["twinlite"]["checkpoint_sha256"],
            "test_split": CONTRACT.artifact_record(test_split_path),
            "parts": part_records,
            "videos": video_reports,
        },
    )
    outputs.extend([motion_path, road_path, manifest_path])
    return outputs
