"""Prepare or run JAAD ViTPose extraction to PIE-compatible JSONL."""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import shutil
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


EXPECTED_KEYPOINT_COUNT = 17
DEFAULT_SET_ID = "jaad"
RECORD_KEY_FIELDS = ("set", "video", "pid", "frame_idx")


def _check_runtime_dependencies() -> None:
    missing = [name for name in ["cv2", "mmpose"] if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError(
            "Missing runtime dependencies for pose extraction: "
            f"{missing}. Install the missing packages before running extraction."
        )


def _find_attr_text(box: ET.Element, name: str) -> str | None:
    attr = box.find(f'./attribute[@name="{name}"]')
    if attr is None or attr.text is None:
        return None
    return attr.text.strip()


def _label_from_cross(cross_text: str | None) -> int:
    return 1 if (cross_text or "not-crossing").strip().lower() == "crossing" else 0


def collect_required_jaad_frames(
    dataset_root: Path,
    split_csv_dir: Path | None,
    all_behavior_tracks: bool,
    required_rows_csv: Path | None = None,
) -> list[dict[str, Any]]:
    return _collect_required_jaad_frames_runtime(dataset_root, split_csv_dir, all_behavior_tracks, required_rows_csv)


def _record_key(record: dict[str, Any]) -> tuple[str, str, str, int]:
    return (
        str(record["set"]),
        str(record["video"]),
        str(record["pid"]),
        int(record["frame_idx"]),
    )


def _is_complete_existing_record(record: Any) -> bool:
    if not isinstance(record, dict):
        return False
    try:
        _record_key(record)
    except (KeyError, TypeError, ValueError):
        return False
    return True


def _backup_path_for(output_jsonl: Path) -> Path:
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_jsonl.with_name(f"{output_jsonl.name}.resume_backup_{timestamp}")


def _load_existing_completed_keys(output_jsonl: Path) -> tuple[set[tuple[str, str, str, int]], int, bool, Path | None]:
    completed_keys: set[tuple[str, str, str, int]] = set()
    valid_records: list[dict[str, Any]] = []
    malformed_found = False

    with output_jsonl.open("r", encoding="utf-8") as fh:
        for line_number, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                malformed_found = True
                print(f"Malformed JSONL line ignored during resume scan: line={line_number}")
                continue
            if not _is_complete_existing_record(record):
                malformed_found = True
                print(
                    "Incomplete JSONL line ignored during resume scan: "
                    f"line={line_number}, required_fields={RECORD_KEY_FIELDS}"
                )
                continue
            valid_record = dict(record)
            completed_keys.add(_record_key(valid_record))
            valid_records.append(valid_record)

    backup_path: Path | None = None
    if malformed_found:
        backup_path = _backup_path_for(output_jsonl)
        shutil.copy2(output_jsonl, backup_path)
        with output_jsonl.open("w", encoding="utf-8") as fh:
            for record in valid_records:
                fh.write(json.dumps(record) + "\n")
        print(
            json.dumps(
                {
                    "resume_rewrite_performed": True,
                    "backup_path": str(backup_path),
                    "valid_records_rewritten": len(valid_records),
                },
                indent=2,
            )
        )

    return completed_keys, len(valid_records), malformed_found, backup_path


def _collect_required_jaad_frames_runtime(
    dataset_root: Path,
    split_csv_dir: Path | None,
    all_behavior_tracks: bool,
    required_rows_csv: Path | None,
) -> list[dict[str, Any]]:
    import pandas as pd

    annotations_root = dataset_root / "annotations"
    images_root = dataset_root / "images"

    required_keys: set[tuple[str, str, int]] | None = None
    required_metadata: dict[tuple[str, str, int], dict[str, Any]] = {}
    if required_rows_csv is not None:
        required_df = pd.read_csv(required_rows_csv)
        required_cols = {"video_id", "pid", "frame_idx"}
        missing = sorted(required_cols - set(required_df.columns))
        if missing:
            raise ValueError(f"{required_rows_csv} missing required columns: {missing}")
        required_keys = set()
        for row in required_df.to_dict(orient="records"):
            key = (str(row["video_id"]), str(row["pid"]), int(row["frame_idx"]))
            required_keys.add(key)
            required_metadata[key] = row
    elif split_csv_dir is not None and not all_behavior_tracks:
        required_keys = set()
        for split in ["train", "val", "test"]:
            path = split_csv_dir / f"{split}.csv"
            df = pd.read_csv(path)
            for row in df.itertuples(index=False):
                frames = json.loads(row.observed_frames_json)
                for frame_idx in frames:
                    required_keys.add((str(row.video_id), str(row.ped_id), int(frame_idx)))

    requests: list[dict[str, Any]] = []
    found_keys: set[tuple[str, str, int]] = set()
    for annotation_path in sorted(annotations_root.glob("*.xml")):
        video_id = annotation_path.stem
        frame_dir = images_root / video_id
        if not frame_dir.exists():
            continue
        root = ET.parse(annotation_path).getroot()
        for track in root.findall("./track"):
            if track.get("label") != "pedestrian":
                continue
            for box in track.findall("./box"):
                if int(box.get("outside", "0")) == 1:
                    continue
                ped_id = _find_attr_text(box, "id")
                if not ped_id:
                    continue
                frame_idx = int(box.get("frame"))
                if required_keys is not None and (video_id, ped_id, frame_idx) not in required_keys:
                    continue
                image_path = frame_dir / f"{frame_idx:05d}.png"
                if not image_path.exists():
                    image_path = frame_dir / f"{frame_idx:05d}.jpg"
                requests.append(
                    {
                        "set": DEFAULT_SET_ID,
                        "video": video_id,
                        "pid": ped_id,
                        "frame_idx": frame_idx,
                        "label": _label_from_cross(_find_attr_text(box, "cross")),
                        "bbox_xyxy": [
                            float(box.get("xtl")),
                            float(box.get("ytl")),
                            float(box.get("xbr")),
                            float(box.get("ybr")),
                        ],
                        "image_path": str(image_path),
                    }
                )
                found_keys.add((video_id, ped_id, frame_idx))
    requests.sort(key=lambda item: (item["video"], item["frame_idx"], item["pid"]))
    if required_rows_csv is not None:
        missing_keys = sorted(required_keys - found_keys) if required_keys is not None else []
        summary = {
            "required_rows_csv": str(required_rows_csv),
            "requested_rows": int(len(required_keys or [])),
            "matched_rows": int(len(found_keys)),
            "missing_rows": int(len(missing_keys)),
        }
        print(json.dumps(summary, indent=2))
        if missing_keys:
            sample_rows = [
                required_metadata.get(key, {"video_id": key[0], "pid": key[1], "frame_idx": key[2]})
                for key in missing_keys[:10]
            ]
            print(json.dumps({"missing_required_row_examples": sample_rows}, indent=2))
    return requests


def run_extraction(
    requests: list[dict[str, Any]],
    output_jsonl: Path,
    config_path: Path,
    checkpoint_path: Path,
    device: str,
    batch_size: int,
    output_mode: str = "w",
    flush_every_record: bool = False,
) -> None:
    _check_runtime_dependencies()
    import cv2
    import numpy as np
    import torch
    from mmpose.apis import inference_topdown, init_model

    if not config_path.exists():
        raise FileNotFoundError(f"Missing ViTPose config: {config_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing ViTPose checkpoint: {checkpoint_path}")

    model = init_model(str(config_path), str(checkpoint_path), device=device)
    model.eval()

    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for item in requests:
        grouped[(item["video"], int(item["frame_idx"]))].append(item)

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open(output_mode, encoding="utf-8") as fh:
        for (_video_id, _frame_idx), items in grouped.items():
            image_path = items[0]["image_path"]
            image = cv2.imread(image_path)
            if image is None:
                continue
            bboxes = [item["bbox_xyxy"] for item in items]
            with torch.inference_mode():
                results = inference_topdown(model, image, bboxes=bboxes, bbox_format="xyxy")
            for item, result in zip(items, results):
                pred = result.pred_instances
                if len(pred.keypoints) == 0:
                    continue
                keypoints = np.asarray(pred.keypoints[0], dtype=float)
                kp_conf = np.asarray(pred.keypoint_scores[0], dtype=float)
                if keypoints.shape != (EXPECTED_KEYPOINT_COUNT, 2):
                    continue
                if kp_conf.shape != (EXPECTED_KEYPOINT_COUNT,):
                    continue
                record = {
                    "set": item["set"],
                    "video": item["video"],
                    "pid": item["pid"],
                    "frame_idx": int(item["frame_idx"]),
                    "keypoints": keypoints.tolist(),
                    "label": int(item["label"]),
                    "kp_conf": kp_conf.tolist(),
                    "skeleton_conf": float(kp_conf.mean()),
                    "bbox_xyxy": item["bbox_xyxy"],
                }
                fh.write(json.dumps(record) + "\n")
                if flush_every_record:
                    fh.flush()


def _validate_limit(limit_records: int | None) -> None:
    if limit_records is not None and limit_records < 0:
        raise ValueError(f"--limit_records must be >= 0, got {limit_records}")


def _prepare_requests_for_run(
    requests: list[dict[str, Any]],
    limit_records: int | None,
    resume: bool,
    output_jsonl: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _validate_limit(limit_records)

    total_requested = len(requests)
    existing_valid_records = 0
    skipped_existing_records = 0
    resume_rewrite_performed = False
    resume_backup_path: str | None = None
    completed_keys: set[tuple[str, str, str, int]] = set()

    if resume and output_jsonl.exists():
        completed_keys, existing_valid_records, resume_rewrite_performed, backup_path = _load_existing_completed_keys(output_jsonl)
        skipped_existing_records = sum(1 for item in requests if _record_key(item) in completed_keys)
        requests = [item for item in requests if _record_key(item) not in completed_keys]
        if backup_path is not None:
            resume_backup_path = str(backup_path)

    remaining_before_limit = len(requests)
    if limit_records is None:
        limited_requests = requests
    else:
        limited_requests = requests[:limit_records]

    output_mode = "append" if resume else "write"
    summary = {
        "total_requested_records_before_resume_filtering": total_requested,
        "existing_valid_records_found": existing_valid_records,
        "skipped_existing_records": skipped_existing_records,
        "remaining_records_to_process_before_limit": remaining_before_limit,
        "limit_records": limit_records,
        "records_processed_after_limit": len(limited_requests),
        "output_mode": output_mode,
        "resume_rewrite_performed": resume_rewrite_performed,
        "resume_backup_path": resume_backup_path,
    }
    print(json.dumps(summary, indent=2))
    return limited_requests, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_root", type=Path, default=Path("/home/abdelrahman/Hadwa/datasets/JAAD"))
    parser.add_argument("--split_csv_dir", type=Path, default=Path("data/jaad_splits_event_16_30_60_default"))
    parser.add_argument("--output_jsonl", type=Path, default=Path("data/jaad_keypoints_vitpose_s.jsonl"))
    parser.add_argument(
        "--config_path",
        type=Path,
        default=Path("/home/abdelrahman/Hadwa/Pedestrian-keypoints-crossing-intent/classification/models/vitpose/td_hm_ViTPose_small_8xb64.py"),
    )
    parser.add_argument(
        "--checkpoint_path",
        type=Path,
        default=Path("/home/abdelrahman/Hadwa/Pedestrian-keypoints-crossing-intent/classification/models/checkpoints/td_hm_ViTPose_small_8xb64_210e_coco.pth"),
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--limit_records", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--flush_every_record", action="store_true")
    parser.add_argument("--all_behavior_tracks", action="store_true")
    parser.add_argument("--prepare_only", action="store_true")
    parser.add_argument(
        "--required_rows_csv",
        type=Path,
        default=None,
        help="Optional CSV of exact JAAD rows to extract; requires columns video_id,pid,frame_idx.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    requests = collect_required_jaad_frames(
        args.dataset_root,
        args.split_csv_dir,
        args.all_behavior_tracks,
        args.required_rows_csv,
    )
    limited_requests, limit_summary = _prepare_requests_for_run(
        requests=requests,
        limit_records=args.limit_records,
        resume=bool(args.resume),
        output_jsonl=args.output_jsonl,
    )
    summary = {
        "request_count": len(limited_requests),
        "unique_videos": len({item["video"] for item in limited_requests}),
        "unique_tracks": len({(item["video"], item["pid"]) for item in limited_requests}),
        "output_jsonl": str(args.output_jsonl),
        "config_path": str(args.config_path),
        "checkpoint_path": str(args.checkpoint_path),
        **limit_summary,
        "resume": bool(args.resume),
        "flush_every_record": bool(args.flush_every_record),
        "prepare_only": bool(args.prepare_only),
    }
    if args.prepare_only:
        print(json.dumps(summary, indent=2))
        return
    run_extraction(
        requests=limited_requests,
        output_jsonl=args.output_jsonl,
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
        device=args.device,
        batch_size=args.batch_size,
        output_mode="a" if args.resume else "w",
        flush_every_record=bool(args.flush_every_record),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
