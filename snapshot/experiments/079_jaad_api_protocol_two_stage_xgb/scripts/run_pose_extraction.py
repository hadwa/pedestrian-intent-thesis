"""Resumable CUDA-timed ViTPose-S extraction for Exp079 missing rows."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
import uuid
from collections import defaultdict
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ACTIVE_ROOT = Path(__file__).resolve().parents[3]
if str(ACTIVE_ROOT) not in sys.path:
    sys.path.insert(0, str(ACTIVE_ROOT))

from src.pose.extract_jaad_vitpose_jsonl import (  # noqa: E402
    EXPECTED_KEYPOINT_COUNT,
    _load_existing_completed_keys,
)
from pose_support import load_extraction_requests, summarize_timing_events  # noqa: E402


DEFAULT_DATASET = Path("/home/abdelrahman/Hadwa/datasets/JAAD")
DEFAULT_CONFIG = Path(
    "/home/abdelrahman/miniconda3/envs/cv_env/lib/python3.9/site-packages/mmpose/.mim/configs/"
    "body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-small_8xb64-210e_coco-256x192.py"
)
DEFAULT_CHECKPOINT = Path(
    "/home/abdelrahman/Hadwa/Pedestrian-keypoints-crossing-intent/classification/models/checkpoints/"
    "td-hm_ViTPose-small_8xb64-210e_coco-256x192.pth"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--manifest", type=Path, default=ROOT / "cache/required_missing_pose_rows.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "cache/jaad_api_protocol_missing_pose.jsonl")
    parser.add_argument("--events", type=Path, default=ROOT / "metrics/pose_timing_events.csv")
    parser.add_argument("--summary", type=Path, default=ROOT / "metrics/pose_inference_timing.json")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup-source-frames", type=int, default=20)
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument("--limit-records", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.device.startswith("cuda"):
        raise ValueError("Exp079 primary extraction requires a CUDA device")
    if not args.config.exists() or not args.checkpoint.exists():
        raise FileNotFoundError("pinned ViTPose-S configuration or checkpoint is missing")

    import cv2
    import numpy as np
    import torch
    from mmpose.apis import inference_topdown, init_model

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; refusing CPU fallback")

    requests = load_extraction_requests(args.manifest, args.dataset_root)
    completed = set()
    if args.output.exists():
        completed, _, _, _ = _load_existing_completed_keys(args.output)
    requests = [
        item
        for item in requests
        if (str(item["set"]), str(item["video"]), str(item["pid"]), int(item["frame_idx"])) not in completed
    ]
    if args.limit_records is not None:
        requests = requests[: args.limit_records]
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for item in requests:
        grouped[(str(item["video"]), int(item["frame_idx"]))].append(item)

    run_id = f"{dt.datetime.now().isoformat()}_{uuid.uuid4().hex[:8]}"
    started = time.perf_counter()
    model_started = time.perf_counter()
    model = init_model(str(args.config), str(args.checkpoint), device=args.device)
    model.eval()
    torch.cuda.synchronize()
    model_load_seconds = time.perf_counter() - model_started

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.events.parent.mkdir(parents=True, exist_ok=True)
    event_rows = []
    output_records = 0
    mode = "a" if args.output.exists() else "w"
    with args.output.open(mode, encoding="utf-8") as output:
        for index, ((video_id, frame_idx), items) in enumerate(grouped.items(), start=1):
            decode_started = time.perf_counter()
            image = cv2.imread(items[0]["image_path"])
            decode_ms = (time.perf_counter() - decode_started) * 1000.0
            if image is None:
                raise RuntimeError(f"failed to decode exact source frame {items[0]['image_path']}")

            bboxes = [item["bbox_xyxy"] for item in items]
            torch.cuda.synchronize()
            inference_started = time.perf_counter()
            with torch.inference_mode():
                results = inference_topdown(model, image, bboxes=bboxes, bbox_format="xyxy")
            torch.cuda.synchronize()
            inference_ms = (time.perf_counter() - inference_started) * 1000.0
            if len(results) != len(items):
                raise RuntimeError(f"pose result count mismatch for {(video_id, frame_idx)}")

            write_started = time.perf_counter()
            written_for_image = 0
            for item, result in zip(items, results):
                pred = result.pred_instances
                if len(pred.keypoints) == 0:
                    raise RuntimeError(f"no keypoints returned for {(video_id, item['pid'], frame_idx)}")
                keypoints = np.asarray(pred.keypoints[0], dtype=float)
                confidence = np.asarray(pred.keypoint_scores[0], dtype=float)
                if keypoints.shape != (EXPECTED_KEYPOINT_COUNT, 2) or confidence.shape != (EXPECTED_KEYPOINT_COUNT,):
                    raise RuntimeError(f"invalid COCO-17 output for {(video_id, item['pid'], frame_idx)}")
                record = {
                    "set": item["set"],
                    "video": video_id,
                    "pid": item["pid"],
                    "frame_idx": frame_idx,
                    "keypoints": keypoints.tolist(),
                    "kp_conf": confidence.tolist(),
                    "skeleton_conf": float(confidence.mean()),
                    "bbox_xyxy": item["bbox_xyxy"],
                }
                output.write(json.dumps(record) + "\n")
                written_for_image += 1
            output.flush()
            write_ms = (time.perf_counter() - write_started) * 1000.0
            output_records += written_for_image
            event_rows.append(
                {
                    "run_id": run_id,
                    "video_id": video_id,
                    "frame_idx": frame_idx,
                    "pedestrian_count": written_for_image,
                    "warmup": index <= args.warmup_source_frames,
                    "decode_ms": decode_ms,
                    "inference_ms": inference_ms,
                    "write_ms": write_ms,
                }
            )
            if index % args.progress_every == 0 or index == len(grouped):
                elapsed = time.perf_counter() - started
                rate = output_records / elapsed if elapsed else 0.0
                remaining = sum(len(value) for value in list(grouped.values())[index:])
                print(
                    json.dumps(
                        {
                            "source_frames_done": index,
                            "source_frames_total": len(grouped),
                            "records_written": output_records,
                            "records_per_second": rate,
                            "eta_seconds": remaining / rate if rate else None,
                        }
                    ),
                    flush=True,
                )

    elapsed = time.perf_counter() - started
    new_events = pd.DataFrame(event_rows)
    if args.events.exists():
        old_events = pd.read_csv(args.events)
        all_events = pd.concat([old_events, new_events], ignore_index=True)
    else:
        all_events = new_events
    all_events.to_csv(args.events, index=False)
    summary = summarize_timing_events(new_events, total_elapsed_seconds=elapsed, output_records=output_records)
    summary.update(
        {
            "run_id": run_id,
            "device": args.device,
            "model_load_seconds": model_load_seconds,
            "requests_remaining_at_start": len(requests),
            "source_frames_processed": len(grouped),
            "config_path": str(args.config.resolve()),
            "checkpoint_path": str(args.checkpoint.resolve()),
        }
    )
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
