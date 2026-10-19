"""Experiment 028 stage 1: YOLO person detection + ByteTrack tracking on Cairo dashcam clips.

Run with the detection/tracking environment:
    /home/abdelrahman/miniconda3/envs/mslstm_pid_env/bin/python \
        scripts/experiments/cairo_dashcam_detect_track.py --video dash_cam/<clip>.mp4

Outputs (under experiments/028_cairo_dashcam_case_study/):
    tracks/<clip>_tracks.csv        per-frame per-track boxes (kept tracks only)
    tracks/<clip>_tracks_raw.csv    all raw person detections before filtering (audit)
    thumbnails/<clip>/<track>/      3 annotated JPEG thumbnails per kept track
    labeling_sheet.csv              one row per kept track (empty label columns for
                                    the manual pass; rows for the same clip are
                                    replaced on re-run, other clips preserved)

Pipeline-stage only: no model scoring happens here. Hand labeling of
labeling_sheet.csv must be completed BEFORE any model evaluation on this data.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

EXP_ROOT = Path("/home/abdelrahman/Hadwa/BIU-PID-Distillation-clean/experiments/028_cairo_dashcam_case_study")

LABEL_SHEET_COLUMNS = [
    "clip_id",
    "track_id",
    "first_frame",
    "last_frame",
    "n_frames",
    "duration_s",
    "mean_box_height_px",
    "mean_conf",
    "thumbnail_1",
    "thumbnail_2",
    "thumbnail_3",
    # Empty columns below: to be filled BY HAND before any model scoring.
    "crossing_label",
    "crossing_onset_frame",
    "hesitation_observed",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True, help="Input mp4 clip.")
    parser.add_argument("--out_root", type=Path, default=EXP_ROOT)
    parser.add_argument("--weights", type=str, default="yolov8s.pt")
    parser.add_argument("--tracker", type=str, default="bytetrack.yaml")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--det_conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=1280,
                        help="Inference size; 1280 helps small pedestrians at 2560x1440.")
    parser.add_argument("--min_box_height", type=float, default=60.0,
                        help="Drop detections shorter than this many pixels.")
    parser.add_argument("--min_track_frames", type=int, default=30,
                        help="Drop tracks observed in fewer frames than this (30 = 1 s at 30 fps).")
    parser.add_argument("--max_frames", type=int, default=None, help="Optional frame cap for smoke tests.")
    return parser.parse_args()


def run_tracking(args: argparse.Namespace) -> tuple[list[dict], float, int]:
    from ultralytics import YOLO

    model = YOLO(args.weights)
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    clip_id = args.video.stem

    rows: list[dict] = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if args.max_frames is not None and frame_idx >= args.max_frames:
            break
        results = model.track(
            frame,
            persist=True,
            tracker=args.tracker,
            classes=[0],  # person
            conf=args.det_conf,
            imgsz=args.imgsz,
            device=args.device,
            verbose=False,
        )
        boxes = results[0].boxes
        if boxes is not None and boxes.id is not None:
            xyxy = boxes.xyxy.cpu().numpy()
            ids = boxes.id.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()
            for (x1, y1, x2, y2), tid, conf in zip(xyxy, ids, confs):
                rows.append(
                    {
                        "clip_id": clip_id,
                        "track_id": int(tid),
                        "frame_idx": frame_idx,
                        "x1": float(x1),
                        "y1": float(y1),
                        "x2": float(x2),
                        "y2": float(y2),
                        "conf": float(conf),
                    }
                )
        frame_idx += 1
    cap.release()
    return rows, float(fps), frame_idx


def filter_tracks(rows: list[dict], min_box_height: float, min_track_frames: int) -> tuple[list[dict], dict]:
    tall = [r for r in rows if (r["y2"] - r["y1"]) >= min_box_height]
    by_track: dict[int, list[dict]] = defaultdict(list)
    for r in tall:
        by_track[r["track_id"]].append(r)
    kept_ids = sorted(tid for tid, items in by_track.items() if len(items) >= min_track_frames)
    kept = [r for tid in kept_ids for r in sorted(by_track[tid], key=lambda x: x["frame_idx"])]
    stats = {
        "raw_detections": len(rows),
        "raw_track_ids": len({r["track_id"] for r in rows}),
        "detections_after_height_filter": len(tall),
        "kept_tracks": len(kept_ids),
        "kept_track_ids": kept_ids,
        "kept_detections": len(kept),
    }
    return kept, stats


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in columns})


def export_thumbnails(video: Path, kept: list[dict], out_root: Path, clip_id: str) -> dict[int, list[str]]:
    """Second sequential pass; save 3 annotated frames (first/middle/last) per kept track."""
    by_track: dict[int, list[dict]] = defaultdict(list)
    for r in kept:
        by_track[r["track_id"]].append(r)

    wanted: dict[int, list[dict]] = defaultdict(list)  # frame_idx -> rows to draw+save
    for tid, items in by_track.items():
        items = sorted(items, key=lambda x: x["frame_idx"])
        picks = {0, len(items) // 2, len(items) - 1}
        for i in sorted(picks):
            wanted[items[i]["frame_idx"]].append(items[i])

    thumb_paths: dict[int, list[str]] = defaultdict(list)
    cap = cv2.VideoCapture(str(video))
    frame_idx = 0
    max_needed = max(wanted) if wanted else -1
    while frame_idx <= max_needed:
        ok, frame = cap.read()
        if not ok:
            break
        for r in wanted.get(frame_idx, []):
            tid = r["track_id"]
            x1, y1, x2, y2 = (int(r["x1"]), int(r["y1"]), int(r["x2"]), int(r["y2"]))
            canvas = frame.copy()
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 200, 255), 3)
            # Context crop around the box (2x margin), clipped to frame.
            h, w = canvas.shape[:2]
            bw, bh = x2 - x1, y2 - y1
            cx1 = max(0, x1 - bw)
            cy1 = max(0, y1 - bh // 2)
            cx2 = min(w, x2 + bw)
            cy2 = min(h, y2 + bh // 2)
            crop = canvas[cy1:cy2, cx1:cx2]
            out_dir = out_root / "thumbnails" / clip_id / f"track_{tid:04d}"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"frame_{frame_idx:05d}.jpg"
            cv2.imwrite(str(out_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
            thumb_paths[tid].append(str(out_path))
        frame_idx += 1
    cap.release()
    return thumb_paths


def update_labeling_sheet(sheet_path: Path, clip_id: str, new_rows: list[dict]) -> None:
    existing: list[dict] = []
    if sheet_path.exists():
        with sheet_path.open("r", newline="", encoding="utf-8") as fh:
            existing = [row for row in csv.DictReader(fh) if row.get("clip_id") != clip_id]
    write_csv(sheet_path, existing + new_rows, LABEL_SHEET_COLUMNS)


def main() -> None:
    args = parse_args()
    clip_id = args.video.stem
    out_root = args.out_root

    rows, fps, n_frames = run_tracking(args)
    kept, stats = filter_tracks(rows, args.min_box_height, args.min_track_frames)

    track_cols = ["clip_id", "track_id", "frame_idx", "x1", "y1", "x2", "y2", "conf"]
    write_csv(out_root / "tracks" / f"{clip_id}_tracks_raw.csv", rows, track_cols)
    write_csv(out_root / "tracks" / f"{clip_id}_tracks.csv", kept, track_cols)

    thumb_paths = export_thumbnails(args.video, kept, out_root, clip_id)

    by_track: dict[int, list[dict]] = defaultdict(list)
    for r in kept:
        by_track[r["track_id"]].append(r)
    sheet_rows = []
    for tid in stats["kept_track_ids"]:
        items = sorted(by_track[tid], key=lambda x: x["frame_idx"])
        heights = np.asarray([it["y2"] - it["y1"] for it in items], dtype=float)
        confs = np.asarray([it["conf"] for it in items], dtype=float)
        thumbs = (thumb_paths.get(tid, []) + ["", "", ""])[:3]
        sheet_rows.append(
            {
                "clip_id": clip_id,
                "track_id": tid,
                "first_frame": items[0]["frame_idx"],
                "last_frame": items[-1]["frame_idx"],
                "n_frames": len(items),
                "duration_s": round((items[-1]["frame_idx"] - items[0]["frame_idx"] + 1) / fps, 2),
                "mean_box_height_px": round(float(heights.mean()), 1),
                "mean_conf": round(float(confs.mean()), 3),
                "thumbnail_1": thumbs[0],
                "thumbnail_2": thumbs[1],
                "thumbnail_3": thumbs[2],
                "crossing_label": "",
                "crossing_onset_frame": "",
                "hesitation_observed": "",
                "notes": "",
            }
        )
    update_labeling_sheet(out_root / "labeling_sheet.csv", clip_id, sheet_rows)

    summary = {
        "clip_id": clip_id,
        "video": str(args.video),
        "frames_processed": n_frames,
        "fps": fps,
        "weights": args.weights,
        "tracker": args.tracker,
        "det_conf": args.det_conf,
        "imgsz": args.imgsz,
        "min_box_height": args.min_box_height,
        "min_track_frames": args.min_track_frames,
        **{k: v for k, v in stats.items() if k != "kept_track_ids"},
        "kept_track_ids": stats["kept_track_ids"],
        "tracks_csv": str(out_root / "tracks" / f"{clip_id}_tracks.csv"),
        "labeling_sheet": str(out_root / "labeling_sheet.csv"),
    }
    summary_path = out_root / "tracks" / f"{clip_id}_detect_track_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
