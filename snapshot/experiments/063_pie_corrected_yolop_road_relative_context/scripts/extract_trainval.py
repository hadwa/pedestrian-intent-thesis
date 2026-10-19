"""Label-sealed, resumable corrected YOLOP extraction for PIE train/validation."""

from __future__ import annotations

import argparse
import json
import os
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from .contracts import (
    ExperimentConfig,
    assert_development_cohort,
    assert_development_path,
    sha256_file,
)
from .representations import (
    assemble_road_relative_temporal,
    corrected_grid_features,
    road_relative_sequence,
)
from .yolop_inference import CorrectedYolopExtractor


REPO_ROOT = Path(__file__).resolve().parents[3]
EXP_ROOT = Path(__file__).resolve().parents[1]
EXP062_ROOT = REPO_ROOT / "experiments/062_hybrid_tree_anchor_context_correction_skl152"
EXP062_ROWS = EXP062_ROOT / "staged_sources/rows.csv"
PIE_VIDEO_ROOT = Path("/home/abdelrahman/Hadwa/PIE_Dataset")
PIE_ANNOTATION_ROOT = Path(
    "/home/abdelrahman/Hadwa/MSLSTM-PID/PIE/annotations/annotations"
)
FROZEN_INFERENCE_BATCH_SIZE = 8


@dataclass(frozen=True)
class TargetFrameRecord:
    """Label-free feature state for one exact pedestrian/frame identity."""

    corrected_grid: np.ndarray
    road_static: np.ndarray
    bbox: np.ndarray


def validate_development_rows(
    rows: pd.DataFrame, *, strict_counts: bool = True
) -> pd.DataFrame:
    """Validate exact observation identity before any frame/model operation."""
    required = {
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
    }
    missing = sorted(required.difference(rows.columns))
    if missing:
        raise ValueError(f"Development rows are missing required columns: {missing}")
    checked = rows.reset_index(drop=True).copy()
    if checked.empty or checked["sequence_id"].astype(str).duplicated().any():
        raise ValueError("Development sequence IDs must be nonempty and unique")
    for set_id in checked["set_id"].astype(str).unique():
        assert_development_cohort(set_id)
    if not set(checked["split"].astype(str)).issubset({"train", "val"}):
        raise ValueError("Development rows contain a non-train/validation split")
    start = checked["start_frame"].to_numpy(dtype=np.int64)
    end = checked["end_frame"].to_numpy(dtype=np.int64)
    exact_windows = (
        (checked["observation_length"].to_numpy(dtype=np.int64) == 16)
        & (checked["frame_step"].to_numpy(dtype=np.int64) == 1)
        & (checked["frame_sampling"].astype(str).to_numpy() == "consecutive")
        & ((end - start) == 15)
    )
    if not exact_windows.all():
        raise ValueError("Development rows must be exact 16-frame consecutive windows")
    if strict_counts:
        config = ExperimentConfig()
        if len(checked) != config.train_rows + config.val_rows:
            raise ValueError("Authenticated Exp062 train/validation row count drifted")
        split = checked["split"].astype(str)
        if int(split.eq("train").sum()) != config.train_rows or int(
            split.eq("val").sum()
        ) != config.val_rows:
            raise ValueError("Authenticated Exp062 split counts drifted")
        for split_name, cohorts in (
            ("train", config.train_cohorts),
            ("val", config.val_cohorts),
        ):
            actual = tuple(
                sorted(checked.loc[split.eq(split_name), "set_id"].astype(str).unique())
            )
            if actual != tuple(sorted(cohorts)):
                raise ValueError(f"Authenticated Exp062 {split_name} cohorts drifted")
    return checked


def load_authenticated_rows() -> pd.DataFrame:
    """Load only the frozen Exp062 train/validation row manifest."""
    config = ExperimentConfig()
    assert_development_path(EXP062_ROWS)
    if not EXP062_ROWS.is_file() or sha256_file(EXP062_ROWS) != config.exp062_rows_sha256:
        raise RuntimeError("Exp062 train/validation row authentication failed")
    return validate_development_rows(pd.read_csv(EXP062_ROWS), strict_counts=True)


def required_bbox_records(
    rows: pd.DataFrame,
) -> dict[tuple[str, str], dict[int, tuple[str, ...]]]:
    """Deduplicate label-free set/video/frame/pedestrian extraction identities."""
    checked = validate_development_rows(rows, strict_counts=False)
    required: dict[tuple[str, str], dict[int, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    for row in checked.itertuples(index=False):
        video_key = (str(row.set_id), str(row.video_id))
        for frame in range(int(row.start_frame), int(row.end_frame) + 1):
            required[video_key][frame].add(str(row.ped_id))
    return {
        key: {frame: tuple(sorted(pedestrians)) for frame, pedestrians in sorted(frames.items())}
        for key, frames in sorted(required.items())
    }


def _load_required_bboxes(
    set_id: str,
    video_id: str,
    required: Mapping[int, tuple[str, ...]],
) -> dict[tuple[str, int], np.ndarray]:
    """Read exact CVAT boxes for only the required pedestrian/frame keys."""
    assert_development_cohort(set_id)
    annotation_path = assert_development_path(
        PIE_ANNOTATION_ROOT / set_id / f"{video_id}_annt.xml"
    )
    if not annotation_path.is_file():
        raise FileNotFoundError(f"Missing PIE annotation: {annotation_path}")
    requested = {
        (ped_id, int(frame))
        for frame, pedestrian_ids in required.items()
        for ped_id in pedestrian_ids
    }
    result: dict[tuple[str, int], np.ndarray] = {}
    root = ET.parse(annotation_path).getroot()
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
            bbox = np.asarray(
                [float(box.get(name)) for name in ("xtl", "ytl", "xbr", "ybr")],
                dtype=np.float32,
            )
            if key in result or not np.isfinite(bbox).all() or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                raise ValueError(f"Malformed or duplicate PIE BBox: {set_id}:{video_id}:{key}")
            result[key] = bbox
    missing = sorted(requested.difference(result))
    if missing:
        raise KeyError(f"Missing exact PIE BBoxes for {set_id}:{video_id}: {missing[:5]}")
    return result


def assemble_video_windows(
    rows: pd.DataFrame,
    records: Mapping[tuple[str, int], TargetFrameRecord],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assemble exact 16-frame arrays, resetting temporal channels per window."""
    checked = validate_development_rows(rows, strict_counts=False)
    sequence_ids: list[str] = []
    corrected_sequences: list[np.ndarray] = []
    road_sequences: list[np.ndarray] = []
    for row in checked.itertuples(index=False):
        frames = range(int(row.start_frame), int(row.end_frame) + 1)
        window_records: list[TargetFrameRecord] = []
        for frame in frames:
            key = (str(row.ped_id), frame)
            if key not in records:
                raise KeyError(f"Missing extracted target/frame record: {key}")
            window_records.append(records[key])
        corrected = np.stack([record.corrected_grid for record in window_records])
        road_static = np.stack([record.road_static for record in window_records])
        bboxes = np.stack([record.bbox for record in window_records])
        road = assemble_road_relative_temporal(road_static, bboxes)
        if corrected.shape != (16, 20) or road.shape != (16, 20):
            raise RuntimeError("Assembled scene window violated the frozen 16x20 ABI")
        sequence_ids.append(str(row.sequence_id))
        corrected_sequences.append(corrected.astype(np.float32, copy=False))
        road_sequences.append(road.astype(np.float32, copy=False))
    return (
        np.asarray(sequence_ids),
        np.stack(corrected_sequences),
        np.stack(road_sequences),
    )


def write_video_part(
    destination: Path,
    sequence_ids: np.ndarray,
    corrected_grid: np.ndarray,
    road_relative: np.ndarray,
) -> None:
    """Atomically persist one complete video-level extraction part."""
    destination = assert_development_path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if corrected_grid.shape != road_relative.shape or corrected_grid.shape != (
        len(sequence_ids),
        16,
        20,
    ):
        raise ValueError("Video part arrays do not share the frozen Nx16x20 shape")
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            sequence_id=np.asarray(sequence_ids),
            corrected_grid=np.asarray(corrected_grid, dtype=np.float32),
            road_relative=np.asarray(road_relative, dtype=np.float32),
        )
    os.replace(temporary, destination)


def validate_existing_part(destination: Path, expected_ids: np.ndarray) -> dict[str, object]:
    """Authenticate a resumable part against exact expected row order."""
    destination = assert_development_path(destination)
    with np.load(destination, allow_pickle=False) as bundle:
        ids = bundle["sequence_id"].astype(str)
        corrected = bundle["corrected_grid"]
        road = bundle["road_relative"]
    if not np.array_equal(ids, np.asarray(expected_ids).astype(str)):
        raise ValueError(f"Existing part sequence IDs do not match: {destination}")
    if corrected.shape != (len(ids), 16, 20) or road.shape != corrected.shape:
        raise ValueError(f"Existing part shape is malformed: {destination}")
    if not np.isfinite(corrected).all() or not np.isfinite(road).all():
        raise ValueError(f"Existing part contains non-finite values: {destination}")
    return {
        "rows": len(ids),
        "sha256": sha256_file(destination),
        "resumed": True,
    }


def _read_inference_batches(
    video_path: Path,
    frames: tuple[int, ...],
    extractor: CorrectedYolopExtractor,
    batch_size: int,
):
    """Yield exact one-based frame numbers with batched corrected inference."""
    import cv2

    video_path = assert_development_path(video_path)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Cannot open PIE video: {video_path}")
    try:
        frame_batch: list[int] = []
        image_batch: list[np.ndarray] = []
        for frame in frames:
            target_position = int(frame) - 1
            current_position = int(round(capture.get(cv2.CAP_PROP_POS_FRAMES)))
            if current_position != target_position:
                capture.set(cv2.CAP_PROP_POS_FRAMES, target_position)
            ok, image = capture.read()
            if not ok or image is None:
                raise RuntimeError(f"Cannot decode exact PIE frame: {video_path}:{frame}")
            frame_batch.append(int(frame))
            image_batch.append(image)
            if len(image_batch) == batch_size:
                perceptions = extractor.infer_batch(image_batch)
                yield tuple(zip(frame_batch, perceptions))
                frame_batch, image_batch = [], []
        if image_batch:
            perceptions = extractor.infer_batch(image_batch)
            yield tuple(zip(frame_batch, perceptions))
    finally:
        capture.release()


def _extract_video(
    set_id: str,
    video_id: str,
    video_rows: pd.DataFrame,
    required: Mapping[int, tuple[str, ...]],
    extractor: CorrectedYolopExtractor,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    bboxes = _load_required_bboxes(set_id, video_id, required)
    video_path = assert_development_path(PIE_VIDEO_ROOT / set_id / f"{video_id}.mp4")
    records: dict[tuple[str, int], TargetFrameRecord] = {}
    for batch in _read_inference_batches(
        video_path, tuple(sorted(required)), extractor, batch_size
    ):
        for frame, perception in batch:
            for pedestrian_id in required[frame]:
                bbox = bboxes[(pedestrian_id, frame)]
                grid = corrected_grid_features(
                    perception.drivable_mask,
                    perception.lane_mask,
                    perception.detections,
                    bbox,
                )
                static = road_relative_sequence(
                    [perception.drivable_mask],
                    [perception.lane_mask],
                    np.asarray([bbox]),
                )[0, :13]
                records[(pedestrian_id, frame)] = TargetFrameRecord(grid, static, bbox)
    sequence_ids, corrected, road = assemble_video_windows(video_rows, records)
    report = {
        "set_id": set_id,
        "video_id": video_id,
        "source_video": str(video_path),
        "source_video_sha256": sha256_file(video_path),
        "source_annotation": str(
            PIE_ANNOTATION_ROOT / set_id / f"{video_id}_annt.xml"
        ),
        "source_annotation_sha256": sha256_file(
            PIE_ANNOTATION_ROOT / set_id / f"{video_id}_annt.xml"
        ),
        "unique_frames": len(required),
        "pedestrian_frame_records": len(records),
        "windows": len(sequence_ids),
    }
    return sequence_ids, corrected, road, report


def extract_trainval(
    *,
    device: str = "cuda:0",
    batch_size: int = FROZEN_INFERENCE_BATCH_SIZE,
    output_root: Path | None = None,
) -> dict[str, object]:
    """Extract every train/validation video with resumable, exact-key parts."""
    if batch_size != FROZEN_INFERENCE_BATCH_SIZE:
        raise ValueError(f"Exp063 inference batch size is frozen at {FROZEN_INFERENCE_BATCH_SIZE}")
    rows = load_authenticated_rows()
    required = required_bbox_records(rows)
    output_root = assert_development_path(output_root or (EXP_ROOT / "cache/trainval_parts"))
    output_root.mkdir(parents=True, exist_ok=True)
    log_path = assert_development_path(EXP_ROOT / "logs/extraction.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(
            f"phase=extract start_or_resume device={device} batch_size={batch_size}\n"
        )
    extractor = CorrectedYolopExtractor(device)
    started = time.time()
    video_reports: list[dict[str, object]] = []
    total_frames = sum(len(frames) for frames in required.values())
    total_records = sum(
        len(pedestrians) for frames in required.values() for pedestrians in frames.values()
    )
    completed_frames = 0
    for video_index, ((set_id, video_id), frame_map) in enumerate(required.items(), start=1):
        video_rows = rows.loc[
            rows["set_id"].astype(str).eq(set_id)
            & rows["video_id"].astype(str).eq(video_id)
        ].reset_index(drop=True)
        expected_ids = video_rows["sequence_id"].astype(str).to_numpy()
        destination = output_root / f"{set_id}_{video_id}.npz"
        if destination.exists():
            resumed = validate_existing_part(destination, expected_ids)
            report = {
                "set_id": set_id,
                "video_id": video_id,
                "unique_frames": len(frame_map),
                "pedestrian_frame_records": sum(len(values) for values in frame_map.values()),
                "windows": len(expected_ids),
                **resumed,
            }
        else:
            ids, corrected, road, report = _extract_video(
                set_id,
                video_id,
                video_rows,
                frame_map,
                extractor,
                batch_size,
            )
            write_video_part(destination, ids, corrected, road)
            report.update(
                {"sha256": sha256_file(destination), "rows": len(ids), "resumed": False}
            )
        video_reports.append(report)
        completed_frames += len(frame_map)
        progress_line = (
            f"exp063_yolop_progress={video_index}/{len(required)} "
            f"frames={completed_frames}/{total_frames} resumed={int(bool(report['resumed']))}"
        )
        print(progress_line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(progress_line + "\n")
    manifest = {
        "phase": "label_sealed_trainval_corrected_yolop_extraction",
        "target_labels_passed_to_perception": False,
        "set03_accessed": False,
        "development_cohorts": list(ExperimentConfig().development_cohorts),
        "videos": len(required),
        "unique_frames": total_frames,
        "pedestrian_frame_records": total_records,
        "windows": len(rows),
        "inference_batch_size": batch_size,
        "device": device,
        "model_parameters": extractor.parameter_count,
        "elapsed_seconds": time.time() - started,
        "video_parts": video_reports,
    }
    manifest_path = EXP_ROOT / "metrics/extraction_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, manifest_path)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(
            f"phase=extract complete videos={len(required)} frames={total_frames} "
            f"records={total_records} elapsed_seconds={manifest['elapsed_seconds']}\n"
        )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=FROZEN_INFERENCE_BATCH_SIZE)
    arguments = parser.parse_args()
    report = extract_trainval(device=arguments.device, batch_size=arguments.batch_size)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "TargetFrameRecord",
    "assemble_video_windows",
    "extract_trainval",
    "load_authenticated_rows",
    "required_bbox_records",
    "validate_development_rows",
    "validate_existing_part",
    "write_video_part",
]
