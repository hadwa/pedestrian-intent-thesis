"""Label-sealed exact-frame extraction core for Exp075."""

from __future__ import annotations

import os
import xml.etree.ElementTree as element_tree
import hashlib
import json
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import cv2
import torch

import context
import contract


IDENTITY_COLUMNS = (
    "sequence_id",
    "split",
    "set_id",
    "video_id",
    "ped_id",
    "start_frame",
    "end_frame",
    "observation_length",
    "frame_step",
    "frame_sampling",
)
TARGET_COLUMNS = frozenset(
    {
        "label",
        "event_frame",
        "time_to_event_frames",
        "has_observed_crossing",
        "first_crossing_onset_frame",
        "official_crossing_point",
    }
)
PIE_ANNOTATION_ROOT = Path(
    "/home/abdelrahman/Hadwa/MSLSTM-PID/PIE/annotations/annotations"
)
PIE_VIDEO_ROOT = Path("/home/abdelrahman/Hadwa/PIE_Dataset")
EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class FrameRecord:
    """Compact perception state for one exact pedestrian/frame identity."""

    static: np.ndarray
    bbox: np.ndarray


def load_identity_rows(split: str) -> pd.DataFrame:
    """Read only perception-safe identity columns from an authenticated split."""

    source_names = {
        "train": "exp072_prepared_train",
        "validation": "exp072_prepared_validation",
        "test": "exp072_prepared_test",
    }
    try:
        path = contract.resolve_source(source_names[str(split)])
    except KeyError as error:
        raise ValueError(f"unsupported split: {split}") from error
    rows = pd.read_csv(path, usecols=list(IDENTITY_COLUMNS))
    row_split = "val" if split == "validation" else str(split)
    return validate_rows(rows, allowed_splits=(row_split,), strict_counts=True)


def validate_rows(
    rows: pd.DataFrame,
    *,
    allowed_splits: Sequence[str],
    strict_counts: bool = False,
) -> pd.DataFrame:
    forbidden = sorted(TARGET_COLUMNS.intersection(rows.columns))
    if forbidden:
        raise ValueError(f"perception rows contain target columns: {forbidden}")
    missing = sorted(set(IDENTITY_COLUMNS).difference(rows.columns))
    if missing:
        raise ValueError(f"perception rows are missing identity columns: {missing}")
    checked = rows.loc[:, IDENTITY_COLUMNS].reset_index(drop=True).copy()
    if checked.empty or checked["sequence_id"].astype(str).duplicated().any():
        raise ValueError("sequence IDs must be nonempty and unique")
    allowed = tuple(map(str, allowed_splits))
    if not allowed or not set(checked["split"].astype(str)).issubset(allowed):
        raise ValueError("perception rows contain a disallowed split")
    registration = contract.load_registration()
    cohort_names = {"val": "validation"}
    allowed_sets = {
        set_id
        for split in allowed
        for set_id in registration["cohorts"][cohort_names.get(split, split)]["sets"]
    }
    if not set(checked["set_id"].astype(str)).issubset(allowed_sets):
        raise ValueError("perception rows contain a disallowed cohort")
    starts = checked["start_frame"].to_numpy(dtype=np.int64)
    ends = checked["end_frame"].to_numpy(dtype=np.int64)
    exact = (
        (checked["observation_length"].to_numpy(dtype=np.int64) == 16)
        & (checked["frame_step"].to_numpy(dtype=np.int64) == 1)
        & (checked["frame_sampling"].astype(str).to_numpy() == "consecutive")
        & ((ends - starts) == 15)
    )
    if not exact.all():
        raise ValueError("perception rows must be exact 16-frame consecutive windows")
    if strict_counts:
        for split in allowed:
            actual = int(checked["split"].astype(str).eq(split).sum())
            expected = int(
                registration["cohorts"][cohort_names.get(split, split)]["rows"]
            )
            if actual != expected:
                raise ValueError(f"authenticated {split} row count drifted")
    return checked


def required_frames(
    rows: pd.DataFrame, *, allowed_splits: Sequence[str]
) -> dict[tuple[str, str], dict[int, tuple[str, ...]]]:
    checked = validate_rows(rows, allowed_splits=allowed_splits)
    required: dict[tuple[str, str], dict[int, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    for row in checked.itertuples(index=False):
        key = (str(row.set_id), str(row.video_id))
        for frame in range(int(row.start_frame), int(row.end_frame) + 1):
            required[key][frame].add(str(row.ped_id))
    return {
        key: {
            frame: tuple(sorted(pedestrians))
            for frame, pedestrians in sorted(frames.items())
        }
        for key, frames in sorted(required.items())
    }


def load_required_bboxes(
    set_id: str,
    video_id: str,
    required: Mapping[int, tuple[str, ...]],
    *,
    annotation_root: Path = PIE_ANNOTATION_ROOT,
) -> dict[tuple[str, int], np.ndarray]:
    """Load exact original PIE XYXY boxes for requested frame/pedestrian keys."""

    allowed_sets = {
        set_name
        for cohort in contract.load_registration()["cohorts"].values()
        for set_name in cohort["sets"]
    }
    if str(set_id) not in allowed_sets:
        raise ValueError(f"disallowed PIE cohort: {set_id}")
    requested = {
        (str(pedestrian), int(frame))
        for frame, pedestrians in required.items()
        for pedestrian in pedestrians
    }
    if not requested:
        raise ValueError("at least one BBox identity is required")
    path = Path(annotation_root) / str(set_id) / f"{video_id}_annt.xml"
    if not path.is_file():
        raise FileNotFoundError(f"missing PIE annotation: {path}")
    result: dict[tuple[str, int], np.ndarray] = {}
    root = element_tree.parse(path).getroot()
    for track in root.findall(".//track"):
        if track.get("label", "").lower() != "pedestrian":
            continue
        for box in track.findall("./box"):
            if box.get("outside") == "1":
                continue
            identity = box.find('./attribute[@name="id"]')
            if identity is None or identity.text is None:
                continue
            key = (str(identity.text), int(box.get("frame", -1)))
            if key not in requested:
                continue
            values = np.asarray(
                [float(box.get(name)) for name in ("xtl", "ytl", "xbr", "ybr")],
                dtype=np.float32,
            )
            if (
                key in result
                or not np.isfinite(values).all()
                or values[2] <= values[0]
                or values[3] <= values[1]
            ):
                raise ValueError(f"malformed or duplicate PIE BBox: {key}")
            result[key] = values
    missing = sorted(requested.difference(result))
    if missing:
        raise KeyError(f"missing exact PIE BBoxes: {missing[:5]}")
    return result


def select_parity_frames(
    frame_ids: Sequence[tuple[str, str, int]], *, count: int = 32
) -> tuple[tuple[str, str, int], ...]:
    """Select frozen evenly spaced identities from lexicographically sorted frames."""

    identities = sorted(
        {(str(set_id), str(video_id), int(frame)) for set_id, video_id, frame in frame_ids}
    )
    if count <= 0 or len(identities) < count:
        raise ValueError("insufficient unique train frames for parity selection")
    indices = np.linspace(0, len(identities) - 1, count, dtype=int)
    if len(set(indices.tolist())) != count:
        raise RuntimeError("parity sampling did not produce unique positions")
    return tuple(identities[int(index)] for index in indices)


def _mask_digest(heads: tuple[list[np.ndarray], list[np.ndarray]]) -> str:
    digest = hashlib.sha256()
    for head in heads:
        for mask in head:
            values = np.asarray(mask, dtype=bool)
            digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
            digest.update(np.packbits(values).tobytes())
    return digest.hexdigest()


def assert_batch_parity(extractor, images: Sequence[np.ndarray]) -> dict[str, object]:
    """Choose batch 8 only when it reproduces batch-1 masks byte for byte."""

    if len(images) != 32:
        raise ValueError("batch parity requires exactly 32 frozen train frames")
    single_drivable: list[np.ndarray] = []
    single_lane: list[np.ndarray] = []
    for image in images:
        drivable, lane = extractor.predict([image])
        single_drivable.extend(drivable)
        single_lane.extend(lane)
    batch_drivable: list[np.ndarray] = []
    batch_lane: list[np.ndarray] = []
    for start in range(0, len(images), 8):
        drivable, lane = extractor.predict(images[start : start + 8])
        batch_drivable.extend(drivable)
        batch_lane.extend(lane)
    identical = all(
        np.array_equal(left, right)
        for left, right in zip(
            single_drivable + single_lane, batch_drivable + batch_lane
        )
    )
    single = (single_drivable, single_lane)
    batched = (batch_drivable, batch_lane)
    return {
        "frame_count": len(images),
        "candidate_batch_size": 8,
        "selected_batch_size": 8 if identical else 1,
        "byte_identical": bool(identical),
        "batch1_mask_sha256": _mask_digest(single),
        "batch8_mask_sha256": _mask_digest(batched),
    }


def read_mask_batches(
    video_path: Path,
    frames: Sequence[int],
    extractor,
    *,
    batch_size: int,
    capture_factory=cv2.VideoCapture,
):
    """Yield exact one-based frame IDs with batched source-coordinate masks."""

    requested = tuple(map(int, frames))
    if requested != tuple(sorted(set(requested))) or not requested or requested[0] < 1:
        raise ValueError("frame IDs must be unique sorted positive one-based values")
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    capture = capture_factory(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"cannot open PIE video: {video_path}")
    try:
        frame_batch: list[int] = []
        image_batch: list[np.ndarray] = []
        for frame in requested:
            target = frame - 1
            current = int(round(capture.get(cv2.CAP_PROP_POS_FRAMES)))
            if current != target:
                capture.set(cv2.CAP_PROP_POS_FRAMES, target)
            ok, image = capture.read()
            if not ok or image is None:
                raise RuntimeError(f"cannot decode exact PIE frame: {video_path}:{frame}")
            frame_batch.append(frame)
            image_batch.append(image)
            if len(image_batch) == batch_size:
                drivable, lane = extractor.predict(image_batch)
                yield tuple(zip(frame_batch, drivable, lane))
                frame_batch, image_batch = [], []
        if image_batch:
            drivable, lane = extractor.predict(image_batch)
            yield tuple(zip(frame_batch, drivable, lane))
    finally:
        capture.release()


def extract_video(
    set_id: str,
    video_id: str,
    video_rows: pd.DataFrame,
    required: Mapping[int, tuple[str, ...]],
    extractor,
    *,
    batch_size: int,
    video_root: Path = PIE_VIDEO_ROOT,
    annotation_root: Path = PIE_ANNOTATION_ROOT,
    capture_factory=cv2.VideoCapture,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Extract one video's exact masks and assemble authoritative windows."""

    row_split = tuple(sorted(set(video_rows["split"].astype(str))))
    if len(row_split) != 1:
        raise ValueError("one video extraction requires one split")
    bboxes = load_required_bboxes(
        set_id,
        video_id,
        required,
        annotation_root=annotation_root,
    )
    video_path = Path(video_root) / str(set_id) / f"{video_id}.mp4"
    records: dict[tuple[str, int], FrameRecord] = {}
    drivable_zero = 0
    lane_zero = 0
    drivable_fractions: list[float] = []
    lane_fractions: list[float] = []
    for batch in read_mask_batches(
        video_path,
        tuple(sorted(required)),
        extractor,
        batch_size=batch_size,
        capture_factory=capture_factory,
    ):
        for frame, drivable, lane in batch:
            drivable_values = np.asarray(drivable, dtype=bool)
            lane_values = np.asarray(lane, dtype=bool)
            if drivable_values.shape != lane_values.shape or drivable_values.ndim != 2:
                raise RuntimeError("TwinLite masks must share one 2-D source shape")
            drivable_zero += int(not drivable_values.any())
            lane_zero += int(not lane_values.any())
            drivable_fractions.append(float(drivable_values.mean()))
            lane_fractions.append(float(lane_values.mean()))
            for pedestrian in required[frame]:
                key = (str(pedestrian), int(frame))
                records[key] = FrameRecord(
                    static=context.road_relative_static(
                        drivable_values, lane_values, bboxes[key]
                    ),
                    bbox=bboxes[key],
                )
    ids, road = assemble_windows(
        video_rows,
        records,
        allowed_splits=row_split,
    )
    annotation_path = Path(annotation_root) / str(set_id) / f"{video_id}_annt.xml"
    report: dict[str, object] = {
        "set_id": str(set_id),
        "video_id": str(video_id),
        "source_video": str(video_path),
        "source_annotation": str(annotation_path),
        "unique_frames": len(required),
        "pedestrian_frame_records": len(records),
        "windows": len(ids),
        "drivable_all_zero_frames": drivable_zero,
        "lane_all_zero_frames": lane_zero,
        "drivable_foreground_pixel_fraction": float(np.mean(drivable_fractions)),
        "lane_foreground_pixel_fraction": float(np.mean(lane_fractions)),
        "road_geometry_valid_window_frames": int(road[:, :, 6].sum()),
        "lane_geometry_valid_window_frames": int(road[:, :, 12].sum()),
        "window_frames": int(road.shape[0] * road.shape[1]),
        "completely_invalid_windows": int(
            (
                (road[:, :, 6].sum(axis=1) == 0)
                & (road[:, :, 12].sum(axis=1) == 0)
            ).sum()
        ),
    }
    if video_path.is_file():
        report["source_video_sha256"] = contract.sha256_file(video_path)
    if annotation_path.is_file():
        report["source_annotation_sha256"] = contract.sha256_file(annotation_path)
    return ids, road, report


def aggregate_coverage(
    split: str,
    reports: Sequence[Mapping[str, object]],
    summaries: np.ndarray,
) -> pd.DataFrame:
    """Aggregate required source, geometry, and 60-D coverage diagnostics."""

    if not reports:
        raise ValueError("coverage requires at least one video report")
    values = np.asarray(summaries, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 60 or values.shape[0] <= 0:
        raise ValueError("coverage summaries must have shape Nx60")
    frames = sum(int(report["unique_frames"]) for report in reports)
    windows = sum(int(report["windows"]) for report in reports)
    window_frames = sum(int(report["window_frames"]) for report in reports)
    if frames <= 0 or windows != len(values) or window_frames <= 0:
        raise RuntimeError("coverage population does not match extracted reports")

    def total(name: str) -> int:
        return sum(int(report.get(name, 0)) for report in reports)

    def weighted(name: str) -> float:
        return sum(
            float(report[name]) * int(report["unique_frames"]) for report in reports
        ) / frames

    metrics: list[tuple[str, float]] = [
        ("unique_source_frames", float(frames)),
        ("windows", float(windows)),
        ("drivable_all_zero_frame_fraction", total("drivable_all_zero_frames") / frames),
        ("lane_all_zero_frame_fraction", total("lane_all_zero_frames") / frames),
        ("drivable_foreground_pixel_fraction", weighted("drivable_foreground_pixel_fraction")),
        ("lane_foreground_pixel_fraction", weighted("lane_foreground_pixel_fraction")),
        ("road_geometry_valid_frame_fraction", total("road_geometry_valid_window_frames") / window_frames),
        ("lane_geometry_valid_frame_fraction", total("lane_geometry_valid_window_frames") / window_frames),
        ("completely_invalid_windows", float(total("completely_invalid_windows"))),
    ]
    for index in range(60):
        column = values[:, index]
        metrics.extend(
            [
                (f"summary_{index:02d}_finite_fraction", float(np.isfinite(column).mean())),
                (f"summary_{index:02d}_nonzero_fraction", float(np.count_nonzero(column) / len(column))),
            ]
        )
    return pd.DataFrame(
        [{"split": str(split), "metric": name, "value": value} for name, value in metrics]
    )


def decode_exact_frame(
    set_id: str,
    video_id: str,
    frame: int,
    *,
    video_root: Path = PIE_VIDEO_ROOT,
) -> np.ndarray:
    """Decode one exact one-based PIE frame for the frozen parity sample."""

    video_path = Path(video_root) / str(set_id) / f"{video_id}.mp4"
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"cannot open PIE video: {video_path}")
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame) - 1)
        ok, image = capture.read()
        if not ok or image is None:
            raise RuntimeError(f"cannot decode exact PIE frame: {video_path}:{frame}")
        return image
    finally:
        capture.release()


def _load_report(path: Path, part: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise RuntimeError(f"missing resumable extraction report: {path}") from error
    if value.get("part_sha256") != contract.sha256_file(part):
        raise RuntimeError(f"resumable extraction report authentication failed: {path}")
    return value


def extract_development(
    extractor,
    *,
    output_root: Path = EXPERIMENT_ROOT,
) -> dict[str, object]:
    """Extract exact train/validation masks and road context with safe resume."""

    root = Path(output_root)
    train = load_identity_rows("train")
    validation = load_identity_rows("validation")
    split_rows = {"train": train, "validation": validation}
    train_required = required_frames(train, allowed_splits=("train",))
    train_frame_ids = [
        (set_id, video_id, frame)
        for (set_id, video_id), frames in train_required.items()
        for frame in frames
    ]
    parity_ids = select_parity_frames(train_frame_ids, count=32)
    parity_images = [decode_exact_frame(*identity) for identity in parity_ids]
    parity = assert_batch_parity(extractor, parity_images)
    parity["frame_ids"] = [list(identity) for identity in parity_ids]
    parity["checkpoint_sha256"] = contract.EXPECTED_REGISTRATION["twinlite"][
        "checkpoint_sha256"
    ]
    batch_size = int(parity["selected_batch_size"])
    parity_path = root / "metrics/batch_parity.json"
    contract.atomic_json(parity_path, parity)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    reports_by_split: dict[str, list[dict[str, object]]] = {}
    summaries_by_split: dict[str, np.ndarray] = {}
    part_records: list[dict[str, object]] = []
    for split, rows in split_rows.items():
        row_split = "train" if split == "train" else "val"
        required = required_frames(rows, allowed_splits=(row_split,))
        split_reports: list[dict[str, object]] = []
        split_road: list[np.ndarray] = []
        for set_id, video_id in required:
            video_rows = rows.loc[
                rows["set_id"].astype(str).eq(set_id)
                & rows["video_id"].astype(str).eq(video_id)
            ].reset_index(drop=True)
            expected_ids = video_rows["sequence_id"].astype(str).to_numpy()
            part = root / "cache/development_parts" / f"{set_id}_{video_id}.npz"
            report_path = (
                root / "cache/development_reports" / f"{set_id}_{video_id}.json"
            )
            if part.exists():
                validate_part(part, expected_ids)
                report = _load_report(report_path, part)
                report["resumed"] = True
            else:
                ids, road, report = extract_video(
                    set_id,
                    video_id,
                    video_rows,
                    required[(set_id, video_id)],
                    extractor,
                    batch_size=batch_size,
                )
                write_part(part, ids, road)
                report.update(
                    {
                        "part_path": str(part),
                        "part_sha256": contract.sha256_file(part),
                        "rows": len(ids),
                        "split": split,
                        "resumed": False,
                    }
                )
                contract.atomic_json(report_path, report)
            with np.load(part, allow_pickle=False) as bundle:
                split_road.append(np.asarray(bundle["road_relative"], dtype=np.float32))
            split_reports.append(report)
            part_records.append(
                {
                    "split": split,
                    "set_id": set_id,
                    "video_id": video_id,
                    "path": str(part),
                    **contract.artifact_record(part),
                    "rows": len(expected_ids),
                }
            )
            print(
                f"exp075_extract split={split} video={set_id}/{video_id} "
                f"windows={len(expected_ids)} resumed={int(bool(report['resumed']))}",
                flush=True,
            )
        road_values = np.concatenate(split_road)
        summaries = np.stack([context.summarize_road(value) for value in road_values])
        if len(summaries) != len(rows):
            raise RuntimeError(f"{split} extracted window count drifted")
        reports_by_split[split] = split_reports
        summaries_by_split[split] = summaries

    elapsed = time.perf_counter() - started
    coverage = pd.concat(
        [
            aggregate_coverage(split, reports_by_split[split], summaries_by_split[split])
            for split in ("train", "validation")
        ],
        ignore_index=True,
    )
    coverage_path = root / "metrics/context_coverage.csv"
    contract.atomic_csv(coverage_path, coverage)
    unique_frames = sum(
        int(report["unique_frames"])
        for reports in reports_by_split.values()
        for report in reports
    )
    efficiency = {
        "device": str(extractor.device),
        "precision": "float16" if extractor.half else "float32",
        "batch_size": batch_size,
        "model_parameters": int(extractor.model_parameters),
        "checkpoint_size_bytes": int(
            contract.EXPECTED_REGISTRATION["twinlite"]["checkpoint_size_bytes"]
        ),
        "unique_frames": unique_frames,
        "elapsed_seconds": elapsed,
        "throughput_unique_frames_per_second": unique_frames / elapsed,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated())
        if torch.cuda.is_available()
        else None,
    }
    efficiency_path = root / "metrics/perception_efficiency.json"
    contract.atomic_json(efficiency_path, efficiency)
    manifest = {
        "phase": "label_sealed_twinlitenet_development_extraction",
        "target_labels_passed_to_perception": False,
        "set03_accessed": False,
        "checkpoint_sha256": contract.EXPECTED_REGISTRATION["twinlite"][
            "checkpoint_sha256"
        ],
        "source_commit": contract.EXPECTED_REGISTRATION["twinlite"]["commit"],
        "batch_parity": parity,
        "windows": {split: len(rows) for split, rows in split_rows.items()},
        "parts": part_records,
        "reports": reports_by_split,
    }
    manifest_path = root / "metrics/extraction_manifest.json"
    contract.atomic_json(manifest_path, manifest)
    return {
        "manifest": manifest,
        "outputs": [parity_path, coverage_path, efficiency_path, manifest_path],
    }


def extract_test(
    extractor,
    *,
    output_root: Path = EXPERIMENT_ROOT,
) -> dict[str, object]:
    """Extract set03 only after the caller authenticates validation advance."""

    root = Path(output_root)
    rows = load_identity_rows("test")
    required = required_frames(rows, allowed_splits=("test",))
    parity_path = root / "metrics/batch_parity.json"
    parity = json.loads(parity_path.read_text(encoding="utf-8"))
    if parity.get("checkpoint_sha256") != contract.EXPECTED_REGISTRATION["twinlite"][
        "checkpoint_sha256"
    ]:
        raise RuntimeError("development batch-parity artifact drifted")
    batch_size = int(parity["selected_batch_size"])
    reports: list[dict[str, object]] = []
    roads: list[np.ndarray] = []
    parts: list[dict[str, object]] = []
    for set_id, video_id in required:
        video_rows = rows.loc[
            rows["set_id"].astype(str).eq(set_id)
            & rows["video_id"].astype(str).eq(video_id)
        ].reset_index(drop=True)
        expected_ids = video_rows["sequence_id"].astype(str).to_numpy()
        part = root / "cache/test_parts" / f"{set_id}_{video_id}.npz"
        report_path = root / "cache/test_reports" / f"{set_id}_{video_id}.json"
        if part.exists():
            validate_part(part, expected_ids)
            report = _load_report(report_path, part)
            report["resumed"] = True
        else:
            ids, road, report = extract_video(
                set_id,
                video_id,
                video_rows,
                required[(set_id, video_id)],
                extractor,
                batch_size=batch_size,
            )
            write_part(part, ids, road)
            report.update(
                {
                    "part_path": str(part),
                    "part_sha256": contract.sha256_file(part),
                    "rows": len(ids),
                    "split": "test",
                    "resumed": False,
                }
            )
            contract.atomic_json(report_path, report)
        with np.load(part, allow_pickle=False) as bundle:
            roads.append(np.asarray(bundle["road_relative"], dtype=np.float32))
        reports.append(report)
        parts.append(
            {
                "split": "test",
                "set_id": set_id,
                "video_id": video_id,
                "path": str(part),
                **contract.artifact_record(part),
                "rows": len(expected_ids),
            }
        )
    road_values = np.concatenate(roads)
    summaries = np.stack([context.summarize_road(value) for value in road_values])
    if len(summaries) != len(rows):
        raise RuntimeError("test extracted window count drifted")
    coverage = aggregate_coverage("test", reports, summaries)
    coverage_path = root / "metrics/test_context_coverage.csv"
    contract.atomic_csv(coverage_path, coverage)
    manifest = {
        "phase": "label_sealed_twinlitenet_test_extraction",
        "target_labels_passed_to_perception": False,
        "set03_accessed": True,
        "checkpoint_sha256": contract.EXPECTED_REGISTRATION["twinlite"][
            "checkpoint_sha256"
        ],
        "batch_size": batch_size,
        "windows": len(rows),
        "parts": parts,
        "reports": reports,
    }
    manifest_path = root / "metrics/test_extraction_manifest.json"
    contract.atomic_json(manifest_path, manifest)
    return {"manifest": manifest, "outputs": [coverage_path, manifest_path]}


def assemble_windows(
    rows: pd.DataFrame,
    records: Mapping[tuple[str, int], FrameRecord],
    *,
    allowed_splits: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    checked = validate_rows(rows, allowed_splits=allowed_splits)
    sequence_ids: list[str] = []
    sequences: list[np.ndarray] = []
    for row in checked.itertuples(index=False):
        frames: list[FrameRecord] = []
        for frame in range(int(row.start_frame), int(row.end_frame) + 1):
            key = (str(row.ped_id), frame)
            if key not in records:
                raise KeyError(f"missing perception record: {key}")
            frames.append(records[key])
        static = np.stack([value.static for value in frames])
        bboxes = np.stack([value.bbox for value in frames])
        sequences.append(context.assemble_road_relative(static, bboxes))
        sequence_ids.append(str(row.sequence_id))
    return np.asarray(sequence_ids), np.stack(sequences).astype(np.float32, copy=False)


def write_part(
    destination: Path,
    sequence_ids: np.ndarray,
    road_relative: np.ndarray,
) -> None:
    path = Path(destination)
    ids = np.asarray(sequence_ids).astype(str)
    values = np.asarray(road_relative, dtype=np.float32)
    if values.shape != (len(ids), 16, 20) or not np.isfinite(values).all():
        raise ValueError("part must contain finite Nx16x20 road-relative values")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, sequence_id=ids, road_relative=values)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def validate_part(destination: Path, expected_ids: np.ndarray) -> dict[str, object]:
    path = Path(destination)
    with np.load(path, allow_pickle=False) as bundle:
        ids = np.asarray(bundle["sequence_id"]).astype(str)
        values = np.asarray(bundle["road_relative"], dtype=np.float32)
    expected = np.asarray(expected_ids).astype(str)
    if not np.array_equal(ids, expected):
        raise ValueError(f"part sequence IDs do not match: {path}")
    if values.shape != (len(ids), 16, 20) or not np.isfinite(values).all():
        raise ValueError(f"part contents are malformed: {path}")
    return {"rows": len(ids), "sha256": contract.sha256_file(path)}
