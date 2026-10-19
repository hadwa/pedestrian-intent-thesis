"""Experiment 028 stage 2: ViTPose-S top-down keypoint extraction on tracked boxes.

Run with the pose environment:
    /home/abdelrahman/miniconda3/envs/cv_env/bin/python \
        scripts/experiments/cairo_dashcam_vitpose_extract.py --video dash_cam/<clip>.mp4

Reads the kept-track boxes CSV produced by cairo_dashcam_detect_track.py and
writes keypoints JSONL in the SAME record schema as the JAAD extraction
(/home/abdelrahman/Hadwa/BIU-PID-Distillation/data/jaad_keypoints_vitpose_s.jsonl,
produced by src/pose/extract_jaad_vitpose_jsonl.py in the archive repo), i.e.
fields: set, video, pid, frame_idx, keypoints[17][2], label, kp_conf[17],
skeleton_conf, bbox_xyxy. Differences for Cairo: set="cairo", video=clip stem,
pid=str(track_id), and label=-1 (UNLABELED — ground truth comes only from the
manual labeling_sheet.csv pass, which happens before any model scoring).

Same ViTPose-S model as the JAAD extraction (td-hm_ViTPose-small_8xb64-210e
COCO 256x192). Inference call pattern replicated from
/home/abdelrahman/Hadwa/BIU-PID-Distillation/src/pose/extract_jaad_vitpose_jsonl.py.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

EXP_ROOT = Path("/home/abdelrahman/Hadwa/BIU-PID-Distillation-clean/experiments/028_cairo_dashcam_case_study")
DEFAULT_CONFIG = Path(
    "/home/abdelrahman/miniconda3/envs/cv_env/lib/python3.9/site-packages/mmpose/.mim/configs/"
    "body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-small_8xb64-210e_coco-256x192.py"
)
DEFAULT_CHECKPOINT = Path(
    "/home/abdelrahman/Hadwa/Pedestrian-keypoints-crossing-intent/classification/models/checkpoints/"
    "td-hm_ViTPose-small_8xb64-210e_coco-256x192.pth"
)
EXPECTED_KEYPOINT_COUNT = 17
UNLABELED = -1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True, help="Input mp4 clip.")
    parser.add_argument("--tracks_csv", type=Path, default=None,
                        help="Boxes CSV from stage 1; default tracks/<clip>_tracks.csv under out_root.")
    parser.add_argument("--output_jsonl", type=Path, default=None,
                        help="Default keypoints/<clip>_keypoints_vitpose_s.jsonl under out_root.")
    parser.add_argument("--out_root", type=Path, default=EXP_ROOT)
    parser.add_argument("--config_path", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint_path", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max_frames", type=int, default=None, help="Optional frame cap for smoke tests.")
    return parser.parse_args()


def load_boxes_by_frame(tracks_csv: Path) -> dict[int, list[dict]]:
    import csv

    by_frame: dict[int, list[dict]] = defaultdict(list)
    with tracks_csv.open("r", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            by_frame[int(row["frame_idx"])].append(
                {
                    "track_id": int(row["track_id"]),
                    "bbox_xyxy": [float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])],
                }
            )
    return by_frame


def main() -> None:
    args = parse_args()
    clip_id = args.video.stem
    tracks_csv = args.tracks_csv or (args.out_root / "tracks" / f"{clip_id}_tracks.csv")
    output_jsonl = args.output_jsonl or (args.out_root / "keypoints" / f"{clip_id}_keypoints_vitpose_s.jsonl")
    if not tracks_csv.exists():
        raise FileNotFoundError(f"Missing tracks CSV (run stage 1 first): {tracks_csv}")
    if not args.config_path.exists():
        raise FileNotFoundError(f"Missing ViTPose config: {args.config_path}")
    if not args.checkpoint_path.exists():
        raise FileNotFoundError(f"Missing ViTPose checkpoint: {args.checkpoint_path}")

    import cv2
    import numpy as np
    import torch
    from mmpose.apis import inference_topdown, init_model

    model = init_model(str(args.config_path), str(args.checkpoint_path), device=args.device)
    model.eval()

    by_frame = load_boxes_by_frame(tracks_csv)
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    n_records = 0
    n_skipped = 0
    conf_sums: dict[int, float] = defaultdict(float)
    conf_counts: dict[int, int] = defaultdict(int)
    max_needed = max(by_frame) if by_frame else -1
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as fh:
        frame_idx = 0
        while frame_idx <= max_needed:
            if args.max_frames is not None and frame_idx >= args.max_frames:
                break
            ok, image = cap.read()
            if not ok:
                break
            items = by_frame.get(frame_idx, [])
            if items:
                bboxes = [item["bbox_xyxy"] for item in items]
                with torch.inference_mode():
                    results = inference_topdown(model, image, bboxes=bboxes, bbox_format="xyxy")
                for item, result in zip(items, results):
                    pred = result.pred_instances
                    if len(pred.keypoints) == 0:
                        n_skipped += 1
                        continue
                    keypoints = np.asarray(pred.keypoints[0], dtype=float)
                    kp_conf = np.asarray(pred.keypoint_scores[0], dtype=float)
                    if keypoints.shape != (EXPECTED_KEYPOINT_COUNT, 2) or kp_conf.shape != (EXPECTED_KEYPOINT_COUNT,):
                        n_skipped += 1
                        continue
                    record = {
                        "set": "cairo",
                        "video": clip_id,
                        "pid": str(item["track_id"]),
                        "frame_idx": int(frame_idx),
                        "keypoints": keypoints.tolist(),
                        "label": UNLABELED,
                        "kp_conf": kp_conf.tolist(),
                        "skeleton_conf": float(kp_conf.mean()),
                        "bbox_xyxy": item["bbox_xyxy"],
                    }
                    fh.write(json.dumps(record) + "\n")
                    n_records += 1
                    conf_sums[item["track_id"]] += float(kp_conf.mean())
                    conf_counts[item["track_id"]] += 1
            frame_idx += 1
    cap.release()

    per_track_conf = {
        str(tid): round(conf_sums[tid] / conf_counts[tid], 4) for tid in sorted(conf_counts)
    }
    all_mean = (
        round(sum(conf_sums.values()) / max(sum(conf_counts.values()), 1), 4) if conf_counts else None
    )
    summary = {
        "clip_id": clip_id,
        "tracks_csv": str(tracks_csv),
        "output_jsonl": str(output_jsonl),
        "config_path": str(args.config_path),
        "checkpoint_path": str(args.checkpoint_path),
        "device": args.device,
        "records_written": n_records,
        "records_skipped": n_skipped,
        "mean_skeleton_conf_overall": all_mean,
        "mean_skeleton_conf_per_track": per_track_conf,
        "label_field_note": "label=-1 means UNLABELED; ground truth comes from labeling_sheet.csv only.",
    }
    summary_path = output_jsonl.parent / f"{clip_id}_vitpose_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
