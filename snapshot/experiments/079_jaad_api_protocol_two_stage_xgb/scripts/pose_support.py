"""Exact-frame pose coverage and timing helpers for Exp079."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


PoseKey = tuple[str, str, str, int]


def _record_key(record: dict[str, Any]) -> PoseKey:
    return (
        str(record.get("set", "jaad")),
        str(record.get("video", record.get("video_id"))),
        str(record.get("pid", record.get("ped_id"))),
        int(record.get("frame_idx", record.get("frame"))),
    )


def _same_payload(left: dict[str, Any], right: dict[str, Any]) -> bool:
    fields = ("keypoints", "kp_conf")
    if any(field not in left or field not in right for field in fields):
        return True
    return all(
        np.array_equal(np.asarray(left[field], dtype=float), np.asarray(right[field], dtype=float))
        for field in fields
    )


def load_pose_cache_keys(
    paths: Sequence[Path], *, verify_payload: bool = False
) -> tuple[set[PoseKey], dict[str, int]]:
    """Load exact identities and reject contradictory duplicate pose rows."""
    records: dict[PoseKey, dict[str, Any]] = {}
    source_new_keys: dict[str, int] = {}
    for source in map(Path, paths):
        resolved = source.resolve(strict=True)
        added = 0
        with resolved.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                item = json.loads(line)
                key = _record_key(item)
                if key in records:
                    if verify_payload and not _same_payload(records[key], item):
                        raise ValueError(f"conflicting pose records for {key} at {resolved}:{line_number}")
                    continue
                records[key] = item if verify_payload else {}
                added += 1
        source_new_keys[str(resolved)] = added
    return set(records), source_new_keys


def _window_keys(row: Any) -> tuple[PoseKey, ...]:
    frames = tuple(int(value) for value in json.loads(str(row.observed_frames_json)))
    return tuple(("jaad", str(row.video_id), str(row.ped_id), frame) for frame in frames)


def _coverage_for_windows(windows: Iterable[tuple[PoseKey, ...]], cached: set[PoseKey]) -> dict[str, int | float]:
    materialized = list(windows)
    required = {key for window in materialized for key in window}
    cached_required = required.intersection(cached)
    fully_covered = sum(all(key in cached for key in window) for window in materialized)
    return {
        "windows": len(materialized),
        "fully_covered_windows": fully_covered,
        "fully_covered_window_fraction": fully_covered / len(materialized) if materialized else 0.0,
        "required_unique_frames": len(required),
        "cached_unique_frames": len(cached_required),
        "missing_unique_frames": len(required - cached),
    }


def build_missing_manifest(
    window_csv: Path, cache_paths: Sequence[Path]
) -> tuple[pd.DataFrame, dict[str, dict[str, int | float]]]:
    """Build a deterministic exact-ID manifest without pedestrian-ID aliases."""
    cohort = pd.read_csv(window_csv)
    required_columns = {
        "sequence_id", "split", "video_id", "ped_id", "observed_frames_json", "bbox_sequence_json"
    }
    missing_columns = sorted(required_columns.difference(cohort.columns))
    if missing_columns:
        raise ValueError(f"window cohort missing columns: {missing_columns}")
    if cohort.empty or cohort["sequence_id"].duplicated().any():
        raise ValueError("window cohort is empty or has duplicate sequence IDs")

    cached, _ = load_pose_cache_keys(cache_paths, verify_payload=True)
    windows_by_split: dict[str, list[tuple[PoseKey, ...]]] = defaultdict(list)
    frame_splits: dict[PoseKey, set[str]] = defaultdict(set)
    frame_window_counts: dict[PoseKey, int] = defaultdict(int)
    frame_bboxes: dict[PoseKey, np.ndarray] = {}
    for row in cohort.itertuples(index=False):
        keys = _window_keys(row)
        boxes = np.asarray(json.loads(str(row.bbox_sequence_json)), dtype=float)
        if boxes.shape != (len(keys), 4) or not np.isfinite(boxes).all():
            raise ValueError(f"invalid frozen BBox sequence for {row.sequence_id}")
        split = str(row.split)
        windows_by_split[split].append(keys)
        for key, box in zip(keys, boxes, strict=True):
            if key in frame_bboxes and not np.array_equal(frame_bboxes[key], box):
                raise ValueError(f"conflicting frozen BBoxes for {key}")
            frame_bboxes[key] = box
            frame_splits[key].add(split)
            frame_window_counts[key] += 1

    all_windows = [window for windows in windows_by_split.values() for window in windows]
    coverage = {
        split: _coverage_for_windows(windows_by_split.get(split, []), cached)
        for split in ("train", "val", "test")
    }
    coverage["all"] = _coverage_for_windows(all_windows, cached)

    missing = sorted(set(frame_splits).difference(cached), key=lambda key: (key[1], key[2], key[3]))
    manifest = pd.DataFrame(
        [
            {
                "video_id": key[1],
                "pid": key[2],
                "frame_idx": key[3],
                "bbox_xyxy_json": json.dumps(frame_bboxes[key].tolist(), separators=(",", ":")),
                "split_membership": ";".join(sorted(frame_splits[key])),
                "required_window_count": frame_window_counts[key],
            }
            for key in missing
        ]
    )
    return manifest, coverage


def load_extraction_requests(manifest_csv: Path, dataset_root: Path) -> list[dict[str, Any]]:
    """Resolve exact source images while retaining manifest-provided BBoxes."""
    manifest = pd.read_csv(manifest_csv)
    required = {"video_id", "pid", "frame_idx", "bbox_xyxy_json"}
    missing = sorted(required.difference(manifest.columns))
    if missing or manifest.empty:
        raise ValueError(f"pose manifest missing required data: {missing}")
    requests = []
    for row in manifest.itertuples(index=False):
        frame_idx = int(row.frame_idx)
        image_dir = Path(dataset_root) / "images" / str(row.video_id)
        image_path = image_dir / f"{frame_idx:05d}.png"
        if not image_path.exists():
            image_path = image_dir / f"{frame_idx:05d}.jpg"
        if not image_path.exists():
            raise FileNotFoundError(f"missing exact JAAD image: {image_path}")
        bbox = np.asarray(json.loads(str(row.bbox_xyxy_json)), dtype=float)
        if bbox.shape != (4,) or not np.isfinite(bbox).all():
            raise ValueError(f"invalid manifest BBox for {(row.video_id, row.pid, frame_idx)}")
        requests.append(
            {
                "set": "jaad",
                "video": str(row.video_id),
                "pid": str(row.pid),
                "frame_idx": frame_idx,
                "bbox_xyxy": bbox.tolist(),
                "image_path": str(image_path),
            }
        )
    return requests


def validate_pose_output(output_jsonl: Path, manifest_csv: Path) -> dict[str, Any]:
    """Fail closed unless output is a finite, exact-BBox completion of the manifest."""
    manifest = pd.read_csv(manifest_csv)
    expected = {
        ("jaad", str(row.video_id), str(row.pid), int(row.frame_idx)): np.asarray(
            json.loads(str(row.bbox_xyxy_json)), dtype=float
        )
        for row in manifest.itertuples(index=False)
    }
    seen: set[PoseKey] = set()
    with Path(output_jsonl).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            key = _record_key(item)
            if key in seen:
                raise ValueError(f"duplicate pose output identity {key}")
            if key not in expected:
                raise ValueError(f"unexpected pose output identity {key}")
            keypoints = np.asarray(item.get("keypoints"), dtype=float)
            confidence = np.asarray(item.get("kp_conf"), dtype=float)
            bbox = np.asarray(item.get("bbox_xyxy"), dtype=float)
            if keypoints.shape != (17, 2) or confidence.shape != (17,):
                raise ValueError(f"invalid COCO-17 pose output at line {line_number}")
            if not np.isfinite(keypoints).all() or not np.isfinite(confidence).all():
                raise ValueError(f"nonfinite pose output at line {line_number}")
            if not np.array_equal(bbox, expected[key]):
                raise ValueError(f"BBox mismatch for {key}")
            seen.add(key)
    missing = set(expected).difference(seen)
    if missing:
        raise ValueError(f"missing pose output identity {sorted(missing)[0]}")
    return {
        "records": len(seen),
        "expected_records": len(expected),
        "exact_manifest_coverage": seen == set(expected),
        "coco17_shape_valid": True,
        "finite": True,
        "exact_bbox": True,
    }


def _distribution(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def summarize_timing_events(
    events: pd.DataFrame, *, total_elapsed_seconds: float, output_records: int
) -> dict[str, Any]:
    required = {"warmup", "inference_ms", "decode_ms", "write_ms", "pedestrian_count"}
    missing = sorted(required.difference(events.columns))
    if missing or events.empty:
        raise ValueError(f"timing events missing required data: {missing}")
    steady = events.loc[~events["warmup"].astype(bool)].copy()
    if steady.empty:
        raise ValueError("no steady-state timing events remain after warmup exclusion")
    inference = steady["inference_ms"].to_numpy(float)
    return {
        "all_source_frames": int(len(events)),
        "steady_state_source_frames": int(len(steady)),
        "steady_state_pedestrian_records": int(steady["pedestrian_count"].sum()),
        "output_records": int(output_records),
        "total_elapsed_seconds": float(total_elapsed_seconds),
        "end_to_end_records_per_second": float(output_records / total_elapsed_seconds),
        "inference_ms": _distribution(inference),
        "decode_ms": _distribution(steady["decode_ms"].to_numpy(float)),
        "write_ms": _distribution(steady["write_ms"].to_numpy(float)),
        "estimated_sequential_16_frame_inference_ms_from_median": float(np.median(inference) * 16.0),
        "estimated_sequential_16_frame_inference_ms_from_mean": float(np.mean(inference) * 16.0),
        "timing_definition": (
            "CUDA-synchronized inference_topdown wall time per unique source image; "
            "16-frame values are sequential estimates, not a batched-video benchmark"
        ),
    }
