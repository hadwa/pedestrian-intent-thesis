"""Label-sealed Exp081 context extraction adapter for Exp083 manifests."""

from __future__ import annotations

import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import cv2
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
PARENT_EXTRACT_PATH = (
    REPO_ROOT
    / "experiments/081_jaad_ego_compensated_road_motion/scripts/extract.py"
)
PARENT_DEVELOPMENT_CACHE = (
    REPO_ROOT
    / "experiments/081_jaad_ego_compensated_road_motion/cache/development_parts"
)
PARENT_MOTION_QUALITY = (
    REPO_ROOT
    / "experiments/081_jaad_ego_compensated_road_motion/metrics/motion_quality.csv"
)
PARENT_ROAD_QUALITY = (
    REPO_ROOT
    / "experiments/081_jaad_ego_compensated_road_motion/metrics/road_geometry_quality.csv"
)
PARENT_TEST_CACHE = (
    REPO_ROOT
    / "experiments/082_jaad_compensated_road_followup_evaluation/cache/test_parts"
)
PARENT_TEST_MOTION_QUALITY = (
    REPO_ROOT
    / "experiments/082_jaad_compensated_road_followup_evaluation/metrics/"
    "test_motion_quality.csv"
)
PARENT_TEST_ROAD_QUALITY = (
    REPO_ROOT
    / "experiments/082_jaad_compensated_road_followup_evaluation/metrics/"
    "test_road_geometry_quality.csv"
)

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
    "observed_frames_json",
    "bbox_sequence_json",
)
TARGET_COLUMNS = frozenset(
    {
        "label",
        "event_frame",
        "tte_frames",
        "crossing_attr",
        "crossing_point",
        "decision_point",
        "has_observed_crossing",
        "observed_crosses_json",
        "verified_onset_frame",
    }
)


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_contract():
    return _load_module(SCRIPT_DIR / "contract.py", "exp083_contract_for_context")


CONTRACT = _load_contract()


def source_sequence_ids(manifest: pd.DataFrame) -> pd.Series:
    """Create the exact cohort-independent identity used by Exp081/082 caches."""
    return (
        manifest["video_id"].astype(str)
        + ":"
        + manifest["ped_id"].astype(str)
        + ":"
        + manifest["start_frame"].astype(int).astype(str)
        + ":"
        + manifest["end_frame"].astype(int).astype(str)
        + ":"
        + manifest["event_frame"].astype(int).astype(str)
        + ":"
        + manifest["label"].astype(int).astype(str)
    )


def perception_rows_from_manifest(
    manifest: pd.DataFrame,
    *,
    allowed_splits: Iterable[str] = ("train", "val"),
) -> pd.DataFrame:
    """Build label-free rows after deriving their immutable source identity."""
    rows = manifest.copy()
    rows["sequence_id"] = source_sequence_ids(rows)
    return validate_perception_rows(
        rows.loc[:, IDENTITY_COLUMNS], allowed_splits=allowed_splits
    )


def _observed_frames(value: object) -> tuple[int, ...]:
    try:
        frames = tuple(int(frame_id) for frame_id in json.loads(str(value)))
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValueError("invalid observed_frames_json") from exc
    if len(frames) != 16 or any(right <= left for left, right in zip(frames, frames[1:])):
        raise ValueError("observed frames must be 16 strictly increasing identities")
    return frames


def validate_perception_rows(
    rows: pd.DataFrame,
    *,
    allowed_splits: Iterable[str],
) -> pd.DataFrame:
    forbidden = sorted(TARGET_COLUMNS.intersection(rows.columns))
    if forbidden:
        raise ValueError(f"Exp083 perception rows contain targets: {forbidden}")
    missing = sorted(set(IDENTITY_COLUMNS).difference(rows.columns))
    if missing:
        raise ValueError(f"Exp083 perception rows are missing identity fields: {missing}")
    checked = rows.loc[:, IDENTITY_COLUMNS].reset_index(drop=True).copy()
    if checked.empty or checked["sequence_id"].astype(str).duplicated().any():
        raise ValueError("Exp083 perception sequence IDs must be nonempty and unique")
    allowed = {str(split) for split in allowed_splits}
    observed_splits = set(checked["split"].astype(str))
    if not observed_splits.issubset(allowed):
        raise ValueError(
            f"Exp083 perception split is not authorized: {sorted(observed_splits - allowed)}"
        )
    if not checked["observation_length"].astype(int).eq(16).all():
        raise ValueError("Exp083 perception requires 16 observations")
    if not checked["frame_step"].astype(int).eq(1).all():
        raise ValueError("Exp083 perception frame step changed")
    if not checked["frame_sampling"].astype(str).eq(
        "consecutive_track_observations"
    ).all():
        raise ValueError("Exp083 perception sampling changed")
    parsed = checked["observed_frames_json"].map(_observed_frames)
    if not all(
        frames[0] == int(start) and frames[-1] == int(end)
        for frames, start, end in zip(
            parsed, checked["start_frame"], checked["end_frame"]
        )
    ):
        raise ValueError("Exp083 perception frame endpoints changed")
    return checked


def required_frames(rows: pd.DataFrame) -> dict[str, tuple[int, ...]]:
    required: dict[str, set[int]] = defaultdict(set)
    for row in rows.itertuples(index=False):
        required[str(row.video_id)].update(_observed_frames(row.observed_frames_json))
    return {
        video_id: tuple(sorted(frame_ids))
        for video_id, frame_ids in sorted(required.items())
    }


def required_pairs(rows: pd.DataFrame) -> dict[str, tuple[tuple[int, int], ...]]:
    required: dict[str, set[tuple[int, int]]] = defaultdict(set)
    for row in rows.itertuples(index=False):
        frames = _observed_frames(row.observed_frames_json)
        required[str(row.video_id)].update(zip(frames, frames[1:]))
    return {
        video_id: tuple(sorted(pairs))
        for video_id, pairs in sorted(required.items())
    }


def select_uncached_perception_rows(
    manifest: pd.DataFrame,
    *,
    available_source_ids: set[str],
    allowed_splits: Iterable[str] = ("train", "val"),
) -> pd.DataFrame:
    rows = perception_rows_from_manifest(manifest, allowed_splits=allowed_splits)
    return rows.loc[
        ~rows["sequence_id"].astype(str).isin(available_source_ids)
    ].reset_index(drop=True)


def require_cuda(cuda: Any) -> None:
    if not bool(cuda.is_available()):
        raise RuntimeError("Exp083 context extraction requires CUDA; CPU fallback refused")


def _parent_extract():
    authenticated = CONTRACT.resolve_source("exp081_extract")
    if authenticated != PARENT_EXTRACT_PATH.resolve():
        raise RuntimeError("Exp083 authenticated Exp081 extractor path changed")
    return _load_module(authenticated, "exp083_authenticated_exp081_extract")


def existing_development_context_ids() -> set[str]:
    identities: set[str] = set()
    for path in sorted(PARENT_DEVELOPMENT_CACHE.rglob("*.npz")):
        with np.load(path, allow_pickle=False) as archive:
            values = archive["sequence_id"].astype(str).tolist()
        overlap = identities.intersection(values)
        if overlap:
            raise RuntimeError(f"duplicate parent context identities: {sorted(overlap)[:3]}")
        identities.update(values)
    return identities


def existing_test_context_ids() -> set[str]:
    identities: set[str] = set()
    for path in sorted(PARENT_TEST_CACHE.rglob("*.npz")):
        with np.load(path, allow_pickle=False) as archive:
            values = archive["sequence_id"].astype(str).tolist()
        overlap = identities.intersection(values)
        if overlap:
            raise RuntimeError(f"duplicate parent test context identities: {sorted(overlap)[:3]}")
        identities.update(values)
    return identities


def _target_boxes(parent: Any, row: Any) -> np.ndarray:
    return parent._target_boxes(row)


def extract_video_rows(
    rows: pd.DataFrame,
    extractor: Any,
    motion_settings: Mapping[str, Any],
    *,
    batch_size: int,
    allowed_splits: Iterable[str] = ("train", "val"),
) -> tuple[dict[str, np.ndarray], pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Run unchanged Exp081 equations over explicit Exp083 observations."""
    checked = validate_perception_rows(rows, allowed_splits=allowed_splits)
    video_ids = tuple(checked["video_id"].astype(str).unique())
    if len(video_ids) != 1:
        raise ValueError("Exp083 context extraction requires one video")
    if batch_size <= 0:
        raise ValueError("Exp083 TwinLite batch size must be positive")

    parent = _parent_extract()
    motion_module, road_module = parent._feature_modules()
    jaad_data = parent._load_sibling(
        "jaad_data.py", "exp083_authenticated_exp081_jaad_data"
    )
    video_id = video_ids[0]
    needed = required_frames(checked)[video_id]
    needed_pairs = set(required_pairs(checked)[video_id])
    all_boxes = jaad_data.load_all_pedestrian_boxes(video_id, needed)
    parsed_boxes = [_target_boxes(parent, row) for row in checked.itertuples(index=False)]

    static = np.zeros((len(checked), 16, 13), dtype=np.float32)
    users: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for row_index, row in enumerate(checked.itertuples(index=False)):
        for relative, frame_id in enumerate(_observed_frames(row.observed_frames_json)):
            users[frame_id].append((row_index, relative))

    predecessors: dict[int, list[int]] = defaultdict(list)
    retained_predecessors = set()
    for left, right in needed_pairs:
        predecessors[right].append(left)
        if right != left + 1:
            retained_predecessors.add(left)

    pair_motion: dict[tuple[int, int], Any] = {}
    retained_images: dict[int, np.ndarray] = {}
    previous_id: int | None = None
    previous_image: np.ndarray | None = None
    frame_shape: tuple[int, int] | None = None
    drivable_zero = 0
    lane_zero = 0
    video_path = jaad_data.VIDEO_ROOT / f"{video_id}.mp4"
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(video_path)
    try:
        for start in range(0, len(needed), batch_size):
            batch_ids = needed[start : start + batch_size]
            images = parent._decode_exact_batch(capture, video_id, batch_ids)
            road_masks, lane_masks = extractor.predict(images)
            if len(road_masks) != len(images) or len(lane_masks) != len(images):
                raise RuntimeError("Exp083 TwinLite batch output count changed")
            for frame_id, image, road_mask, lane_mask in zip(
                batch_ids, images, road_masks, lane_masks
            ):
                road = np.asarray(road_mask, dtype=bool)
                lane = np.asarray(lane_mask, dtype=bool)
                if road.shape != image.shape[:2] or lane.shape != image.shape[:2]:
                    raise RuntimeError("Exp083 TwinLite source geometry changed")
                frame_shape = image.shape[:2]
                drivable_zero += int(not road.any())
                lane_zero += int(not lane.any())
                for row_index, relative in users[frame_id]:
                    static[row_index, relative] = road_module.static_features(
                        road, lane, parsed_boxes[row_index][relative]
                    )
                for predecessor in predecessors.get(frame_id, []):
                    if predecessor == previous_id:
                        first_image = previous_image
                    else:
                        first_image = retained_images.get(predecessor)
                    if first_image is None:
                        raise RuntimeError(
                            f"missing exact transition image {video_id}:{predecessor}->{frame_id}"
                        )
                    pair_motion[(predecessor, frame_id)] = (
                        motion_module.estimate_pair_motion(
                            first_image,
                            image,
                            all_boxes[predecessor],
                            all_boxes[frame_id],
                            motion_settings,
                        )
                    )
                if frame_id in retained_predecessors:
                    retained_images[frame_id] = image.copy()
                previous_id = frame_id
                previous_image = image
    finally:
        capture.release()
    if frame_shape is None or set(pair_motion) != needed_pairs:
        raise RuntimeError(f"Exp083 did not extract every exact transition for {video_id}")

    sequence_ids = []
    proxies = []
    raw_summaries = []
    compensated_summaries = []
    motion_records = []
    road_records = []
    nonconsecutive_pairs = 0
    for row_index, row in enumerate(checked.itertuples(index=False)):
        frame_ids = _observed_frames(row.observed_frames_json)
        pairs = [
            pair_motion[(left, right)] for left, right in zip(frame_ids, frame_ids[1:])
        ]
        nonconsecutive_pairs += sum(
            right != left + 1 for left, right in zip(frame_ids, frame_ids[1:])
        )
        movement = motion_module.assemble_window_motion(
            pairs, width=frame_shape[1], height=frame_shape[0]
        )
        raw = road_module.assemble_uncompensated(
            static[row_index], parsed_boxes[row_index]
        )
        compensated = road_module.assemble_compensated(
            static[row_index], parsed_boxes[row_index], movement.to_first, raw=raw
        )
        sequence_id = str(row.sequence_id)
        sequence_ids.append(sequence_id)
        proxies.append(movement.proxy)
        raw_summaries.append(road_module.summarize(raw))
        compensated_summaries.append(road_module.summarize(compensated))
        motion_record, road_record = parent._window_quality(
            sequence_id, str(row.split), movement, raw
        )
        motion_records.append(motion_record)
        road_records.append(road_record)

    values = parent._validate_part_values(
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
        "unique_transition_pairs": len(needed_pairs),
        "nonconsecutive_transition_uses": nonconsecutive_pairs,
        "windows": len(checked),
        "target_columns_seen": False,
        "full_resolution_masks_persisted": False,
        "drivable_all_zero_frames": drivable_zero,
        "lane_all_zero_frames": lane_zero,
        "source_video": parent.CONTRACT.artifact_record(video_path),
        "source_annotation": parent.CONTRACT.artifact_record(annotation_path),
    }
    return (
        values,
        pd.DataFrame(motion_records),
        pd.DataFrame(road_records),
        report,
    )


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    CONTRACT.atomic_write_text(path, frame.to_csv(index=False, lineterminator="\n"))


def _video_paths(root: Path, split: str, video_id: str) -> dict[str, Path]:
    if split == "test":
        base = root / "cache/context_parts/test"
    else:
        base = root / "cache/context_parts/development" / split
    return {
        "part": base / f"{video_id}.npz",
        "motion": base / f"{video_id}.motion.csv",
        "road": base / f"{video_id}.road.csv",
        "report": base / f"{video_id}.report.json",
        "marker": base / f".{video_id}.complete.json",
    }


def _write_video_result(
    root: Path,
    split: str,
    video_id: str,
    expected_ids: list[str],
    values: Mapping[str, np.ndarray],
    motion: pd.DataFrame,
    road: pd.DataFrame,
    report: Mapping[str, Any],
) -> dict[str, Any]:
    parent = _parent_extract()
    paths = _video_paths(root, split, video_id)
    if paths["marker"].exists():
        raise RuntimeError(f"refusing to overwrite completed context part: {video_id}")
    part_record = parent.write_part(paths["part"], values)
    _atomic_csv(paths["motion"], motion)
    _atomic_csv(paths["road"], road)
    CONTRACT.atomic_write_json(paths["report"], dict(report))
    outputs = [paths["part"], paths["motion"], paths["road"], paths["report"]]
    marker = {
        "status": "complete",
        "split": split,
        "video_id": video_id,
        "sequence_ids": expected_ids,
        "part": part_record,
        "outputs": [CONTRACT.artifact_record(path) for path in outputs],
    }
    CONTRACT.atomic_write_json(paths["marker"], marker)
    return marker


def _authenticate_video_result(
    root: Path,
    split: str,
    video_id: str,
    expected_ids: list[str],
) -> dict[str, Any] | None:
    paths = _video_paths(root, split, video_id)
    if not paths["marker"].is_file():
        partial = [path for name, path in paths.items() if name != "marker" and path.exists()]
        if partial:
            raise RuntimeError(f"partial unauthenticated context result: {partial[0]}")
        return None
    try:
        marker = json.loads(paths["marker"].read_text(encoding="utf-8"))
        if (
            marker["status"] != "complete"
            or marker["split"] != split
            or marker["video_id"] != video_id
            or marker["sequence_ids"] != expected_ids
        ):
            raise RuntimeError("video marker identity changed")
        for expected in marker["outputs"]:
            actual = CONTRACT.artifact_record(expected["path"])
            if (
                actual["sha256"] != expected["sha256"]
                or actual["size_bytes"] != expected["size_bytes"]
            ):
                raise RuntimeError("video artifact authentication failed")
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid context completion marker: {video_id}") from exc
    return marker


def _load_new_quality(
    root: Path,
    missing_rows: Mapping[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]], list[Path]]:
    motion_parts = []
    road_parts = []
    markers = []
    outputs = []
    for split, rows in missing_rows.items():
        for video_id, video_rows in rows.groupby("video_id", sort=True):
            expected_ids = video_rows["sequence_id"].astype(str).tolist()
            marker = _authenticate_video_result(
                root, split, str(video_id), expected_ids
            )
            if marker is None:
                raise RuntimeError(f"missing completed context part: {split}/{video_id}")
            paths = _video_paths(root, split, str(video_id))
            motion_parts.append(pd.read_csv(paths["motion"]))
            road_parts.append(pd.read_csv(paths["road"]))
            markers.append(marker)
            outputs.extend(paths.values())
    return (
        pd.concat(motion_parts, ignore_index=True),
        pd.concat(road_parts, ignore_index=True),
        markers,
        outputs,
    )


def _combine_quality(
    manifests: Mapping[str, pd.DataFrame],
    missing_rows: Mapping[str, pd.DataFrame],
    new_motion: pd.DataFrame,
    new_road: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    current_ids = {
        split: set(source_sequence_ids(manifest).astype(str))
        for split, manifest in manifests.items()
    }
    missing_ids = {
        split: set(rows["sequence_id"].astype(str))
        for split, rows in missing_rows.items()
    }
    reused_ids = {
        split: current_ids[split].difference(missing_ids[split])
        for split in manifests
    }
    parent_motion = pd.read_csv(PARENT_MOTION_QUALITY)
    parent_road = pd.read_csv(PARENT_ROAD_QUALITY)
    reused_all = set().union(*reused_ids.values())
    parent_motion = parent_motion.loc[
        parent_motion["sequence_id"].astype(str).isin(reused_all)
    ]
    parent_road = parent_road.loc[
        parent_road["sequence_id"].astype(str).isin(reused_all)
    ]
    motion = pd.concat([parent_motion, new_motion], ignore_index=True)
    road = pd.concat([parent_road, new_road], ignore_index=True)
    expected = set().union(*current_ids.values())
    for name, frame in (("motion", motion), ("road", road)):
        actual = set(frame["sequence_id"].astype(str))
        if actual != expected or frame["sequence_id"].astype(str).duplicated().any():
            raise RuntimeError(f"Exp083 full development {name} coverage changed")
    return motion, road, {split: len(values) for split, values in reused_ids.items()}


def evaluate_complete_split_quality_gate(
    motion_quality: pd.DataFrame,
) -> dict[str, Any]:
    """Apply the frozen Exp081 thresholds to each complete development split."""
    required = {
        "sequence_id",
        "split",
        "valid_transition_count",
        "median_displacement_reduction_fraction",
    }
    missing = required.difference(motion_quality.columns)
    if missing:
        raise ValueError(f"Exp083 motion quality is missing: {sorted(missing)}")
    settings = _parent_extract().CONTRACT.load_registration()["extraction"]
    minimum_valid = float(settings["minimum_valid_transition_fraction"])
    minimum_reduction = float(
        settings["minimum_median_displacement_reduction_fraction"]
    )
    results: dict[str, Any] = {}
    for source_name, report_name in (("train", "train"), ("val", "validation")):
        subset = motion_quality.loc[
            motion_quality["split"].astype(str).eq(source_name)
        ]
        if subset.empty:
            raise RuntimeError(f"Exp083 {report_name} quality split is empty")
        valid_fraction = float(
            subset["valid_transition_count"].sum() / (15 * len(subset))
        )
        median_reduction = float(
            subset["median_displacement_reduction_fraction"].median()
        )
        results[report_name] = {
            "windows": len(subset),
            "valid_transition_fraction": valid_fraction,
            "valid_transition_pass": valid_fraction >= minimum_valid,
            "median_displacement_reduction_fraction": median_reduction,
            "displacement_reduction_pass": median_reduction >= minimum_reduction,
        }
    passed = all(
        record["valid_transition_pass"] and record["displacement_reduction_pass"]
        for record in results.values()
    )
    return {
        "status": "pass" if passed else "fail",
        "sampling_rule": "complete_split",
        "minimum_valid_transition_fraction": minimum_valid,
        "minimum_median_displacement_reduction_fraction": minimum_reduction,
        "splits": results,
    }


def run_development_extraction(
    root: Path | str,
    *,
    batch_size: int = 8,
) -> dict[str, Any]:
    """Extract missing JAAD_all train/val context and publish a sealed cache."""
    root = Path(root)
    coverage_marker = CONTRACT.authenticate_phase(root, "coverage-audit")
    if coverage_marker.get("gate_status") != "blocked_new_extraction_required":
        raise RuntimeError("Exp083 extraction requires the authenticated coverage gate")

    manifests = {
        split: pd.read_csv(root / "manifests" / f"jaad_all_{split}.csv")
        for split in ("train", "val")
    }
    available_ids = existing_development_context_ids()
    missing_rows = {
        split: select_uncached_perception_rows(
            manifest, available_source_ids=available_ids
        )
        for split, manifest in manifests.items()
    }
    coverage = pd.read_csv(root / "metrics/input_coverage.csv")
    for split, rows in missing_rows.items():
        expected = coverage.loc[
            coverage["cohort"].eq("jaad_all")
            & coverage["split"].eq(split)
            & coverage["input"].eq("road_context_60d"),
            "missing",
        ]
        if len(expected) != 1 or len(rows) != int(expected.iloc[0]):
            raise RuntimeError(f"Exp083 {split} missing-context workload changed")

    import torch

    require_cuda(torch.cuda)
    parent = _parent_extract()
    extractor = parent.create_twinlite_extractor(device="cuda", half=True)
    motion_settings = parent.CONTRACT.load_registration()["motion_estimator"]

    for split in ("train", "val"):
        rows = missing_rows[split]
        for video_id, video_rows in rows.groupby("video_id", sort=True):
            video_rows = video_rows.reset_index(drop=True)
            expected_ids = video_rows["sequence_id"].astype(str).tolist()
            if _authenticate_video_result(
                root, split, str(video_id), expected_ids
            ) is not None:
                print(
                    f"Exp083 context resume split={split} video={video_id} "
                    f"windows={len(video_rows)}",
                    flush=True,
                )
                continue
            values, motion, road, report = extract_video_rows(
                video_rows,
                extractor,
                motion_settings,
                batch_size=batch_size,
            )
            _write_video_result(
                root,
                split,
                str(video_id),
                expected_ids,
                values,
                motion,
                road,
                report,
            )
            print(
                f"Exp083 context extracted split={split} video={video_id} "
                f"windows={len(video_rows)} frames={report['unique_frames']}",
                flush=True,
            )

    new_motion, new_road, video_markers, outputs = _load_new_quality(
        root, missing_rows
    )
    motion, road, reused_counts = _combine_quality(
        manifests, missing_rows, new_motion, new_road
    )
    motion_path = root / "metrics/development_motion_quality.csv"
    road_path = root / "metrics/development_road_geometry_quality.csv"
    gate_path = root / "metrics/development_context_quality_gate.json"
    manifest_path = root / "metrics/development_context_manifest.json"
    legacy_prefix_gate = parent.evaluate_quality_gate(motion)
    quality_gate = evaluate_complete_split_quality_gate(motion)
    quality_gate["legacy_lexical_prefix_diagnostic"] = legacy_prefix_gate
    _atomic_csv(motion_path, motion)
    _atomic_csv(road_path, road)
    CONTRACT.atomic_write_json(gate_path, quality_gate)
    if quality_gate["status"] != "pass":
        raise RuntimeError("Exp083 full development context quality gate failed")
    manifest = {
        "experiment": "083_jaad_pcpa_aligned_exp082",
        "splits": ["train", "val"],
        "target_labels_passed_to_perception": False,
        "test_accessed": False,
        "full_resolution_masks_persisted": False,
        "cuda_required": True,
        "batch_size": batch_size,
        "checkpoint_sha256": CONTRACT.load_registration()["sources"][
            "twinlite_checkpoint"
        ]["sha256"],
        "new_windows": {split: len(rows) for split, rows in missing_rows.items()},
        "reused_windows": reused_counts,
        "video_parts": video_markers,
    }
    CONTRACT.atomic_write_json(manifest_path, manifest)
    outputs.extend([motion_path, road_path, gate_path, manifest_path])
    marker = CONTRACT.publish_phase(
        root,
        "extract-development-context",
        outputs,
        test_accessed=False,
        test_features_accessed=False,
        test_metrics_accessed=False,
        quality_status="pass",
        new_windows=manifest["new_windows"],
        reused_windows=reused_counts,
    )
    return {
        "marker": str(marker),
        "quality_status": "pass",
        "new_windows": manifest["new_windows"],
        "reused_windows": reused_counts,
    }


def run_test_extraction(
    root: Path | str,
    *,
    batch_size: int = 8,
) -> dict[str, Any]:
    """Extract test context only after all development artifacts are frozen."""
    root = Path(root)
    development = CONTRACT.authenticate_phase(root, "train-validate")
    if development.get("test_accessed") is not False:
        raise RuntimeError("Exp083 train-validate did not preserve the test seal")
    manifest = pd.read_csv(root / "manifests/jaad_all_test.csv")
    available_ids = existing_test_context_ids()
    missing = select_uncached_perception_rows(
        manifest,
        available_source_ids=available_ids,
        allowed_splits={"test"},
    )
    coverage = pd.read_csv(root / "metrics/input_coverage.csv")
    expected = coverage.loc[
        coverage["cohort"].eq("jaad_all")
        & coverage["split"].eq("test")
        & coverage["input"].eq("road_context_60d"),
        "missing",
    ]
    if len(expected) != 1 or len(missing) != int(expected.iloc[0]):
        raise RuntimeError("Exp083 test missing-context workload changed")

    import torch

    require_cuda(torch.cuda)
    parent = _parent_extract()
    extractor = parent.create_twinlite_extractor(device="cuda", half=True)
    motion_settings = parent.CONTRACT.load_registration()["motion_estimator"]
    for video_id, video_rows in missing.groupby("video_id", sort=True):
        video_rows = video_rows.reset_index(drop=True)
        expected_ids = video_rows["sequence_id"].astype(str).tolist()
        if _authenticate_video_result(
            root, "test", str(video_id), expected_ids
        ) is not None:
            print(
                f"Exp083 test context resume video={video_id} windows={len(video_rows)}",
                flush=True,
            )
            continue
        values, motion, road, report = extract_video_rows(
            video_rows,
            extractor,
            motion_settings,
            batch_size=batch_size,
            allowed_splits={"test"},
        )
        _write_video_result(
            root,
            "test",
            str(video_id),
            expected_ids,
            values,
            motion,
            road,
            report,
        )
        print(
            f"Exp083 test context extracted video={video_id} "
            f"windows={len(video_rows)} frames={report['unique_frames']}",
            flush=True,
        )

    new_motion, new_road, video_markers, outputs = _load_new_quality(
        root, {"test": missing}
    )
    current_ids = set(source_sequence_ids(manifest).astype(str))
    missing_ids = set(missing["sequence_id"].astype(str))
    reused_ids = current_ids.difference(missing_ids)
    parent_motion = pd.read_csv(PARENT_TEST_MOTION_QUALITY)
    parent_road = pd.read_csv(PARENT_TEST_ROAD_QUALITY)
    parent_motion = parent_motion.loc[
        parent_motion["sequence_id"].astype(str).isin(reused_ids)
    ]
    parent_road = parent_road.loc[
        parent_road["sequence_id"].astype(str).isin(reused_ids)
    ]
    motion = pd.concat([parent_motion, new_motion], ignore_index=True)
    road = pd.concat([parent_road, new_road], ignore_index=True)
    for name, frame in (("motion", motion), ("road", road)):
        if (
            set(frame["sequence_id"].astype(str)) != current_ids
            or frame["sequence_id"].astype(str).duplicated().any()
        ):
            raise RuntimeError(f"Exp083 full test {name} coverage changed")

    motion_path = root / "metrics/test_motion_quality.csv"
    road_path = root / "metrics/test_road_geometry_quality.csv"
    manifest_path = root / "metrics/test_context_manifest.json"
    _atomic_csv(motion_path, motion)
    _atomic_csv(road_path, road)
    context_manifest = {
        "experiment": "083_jaad_pcpa_aligned_exp082",
        "splits": ["test"],
        "target_labels_passed_to_perception": False,
        "test_accessed": True,
        "test_metrics_accessed": False,
        "full_resolution_masks_persisted": False,
        "cuda_required": True,
        "batch_size": batch_size,
        "new_windows": len(missing),
        "reused_windows": len(reused_ids),
        "video_parts": video_markers,
    }
    CONTRACT.atomic_write_json(manifest_path, context_manifest)
    outputs.extend([motion_path, road_path, manifest_path])
    phase = CONTRACT.publish_phase(
        root,
        "extract-test-context",
        outputs,
        test_accessed=True,
        test_features_accessed=False,
        test_metrics_accessed=False,
        new_windows=len(missing),
        reused_windows=len(reused_ids),
    )
    return {
        "marker": str(phase),
        "new_windows": len(missing),
        "reused_windows": len(reused_ids),
    }
