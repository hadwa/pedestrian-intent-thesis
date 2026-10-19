"""Experiment 028 stage 4: per-track review video snippets for the manual labeling pass.

Run with the detection/tracking environment (cv2 only, no GPU needed):
    /home/abdelrahman/miniconda3/envs/mslstm_pid_env/bin/python \
        scripts/experiments/cairo_dashcam_review_clips.py --video dash_cam/<clip>.mp4

For every kept track in tracks/<clip>_tracks.csv this writes
    review_clips/<clip>/track_XXXX.mp4
covering source frames first_frame-15 .. last_frame+15 (clamped), full frame
downscaled to 1280x720, 30 fps, with:
  - the track's bbox in yellow (solid where detected; thinner dashed where
    linearly interpolated through tracker gap frames),
  - track id and the ORIGINAL clip frame index stamped large in the top-left
    corner of every frame, so the user can pause and read off
    crossing_onset_frame directly (snippet-relative indices are never shown).

Codec: tries avc1 (H.264, plays in standard Windows/VLC players) and falls
back to mp4v if the OpenCV build cannot encode it. When the fallback was used
and an ffmpeg binary is available (default /snap/bin/ffmpeg), each snippet is
transcoded in place to H.264 (libx264, yuv420p, faststart) so the delivered
files still play in standard Windows/VLC players; pass --no_transcode to skip.

Pipeline-stage only: supports the hand-labeling pass that must precede any
model scoring.
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
OUT_SIZE = (1280, 720)
PAD_FRAMES = 15
BOX_COLOR = (0, 220, 255)  # yellow-ish (BGR)
TEXT_COLOR = (255, 255, 255)
TEXT_OUTLINE = (0, 0, 0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True, help="Input mp4 clip.")
    parser.add_argument("--tracks_csv", type=Path, default=None,
                        help="Kept-track boxes CSV from stage 1; default tracks/<clip>_tracks.csv.")
    parser.add_argument("--out_root", type=Path, default=EXP_ROOT)
    parser.add_argument("--pad_frames", type=int, default=PAD_FRAMES)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--ffmpeg", type=Path, default=Path("/snap/bin/ffmpeg"),
                        help="ffmpeg binary for the H.264 post-transcode when cv2 lacks avc1.")
    parser.add_argument("--no_transcode", action="store_true",
                        help="Keep raw cv2 output even if it is mp4v.")
    return parser.parse_args()


def transcode_h264(path: Path, ffmpeg: Path) -> bool:
    """Transcode a snippet to H.264 in place; returns True on success."""
    import subprocess

    tmp = path.with_suffix(".h264tmp.mp4")
    cmd = [str(ffmpeg), "-hide_banner", "-loglevel", "error", "-y", "-i", str(path),
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(tmp)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except Exception:
        tmp.unlink(missing_ok=True)
        return False
    if result.returncode != 0 or not tmp.exists() or tmp.stat().st_size == 0:
        tmp.unlink(missing_ok=True)
        return False
    tmp.replace(path)
    return True


def load_tracks(tracks_csv: Path) -> dict[int, dict[int, list[float]]]:
    """track_id -> {frame_idx: [x1,y1,x2,y2]} (detected frames only)."""
    tracks: dict[int, dict[int, list[float]]] = defaultdict(dict)
    with tracks_csv.open("r", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            tracks[int(row["track_id"])][int(row["frame_idx"])] = [
                float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"]),
            ]
    return tracks


def interpolate_track(boxes: dict[int, list[float]]) -> dict[int, tuple[list[float], bool]]:
    """frame_idx -> (bbox, is_interpolated) for every frame between first and last detection."""
    frames = sorted(boxes)
    out: dict[int, tuple[list[float], bool]] = {f: (boxes[f], False) for f in frames}
    for a, b in zip(frames, frames[1:]):
        if b - a <= 1:
            continue
        box_a = np.asarray(boxes[a], dtype=float)
        box_b = np.asarray(boxes[b], dtype=float)
        for f in range(a + 1, b):
            t = (f - a) / (b - a)
            out[f] = (((1.0 - t) * box_a + t * box_b).tolist(), True)
    return out


def draw_dashed_rect(img: np.ndarray, p1: tuple[int, int], p2: tuple[int, int],
                     color: tuple[int, int, int], thickness: int, dash: int = 8) -> None:
    x1, y1 = p1
    x2, y2 = p2
    for xa, ya, xb, yb in ((x1, y1, x2, y1), (x1, y2, x2, y2), (x1, y1, x1, y2), (x2, y1, x2, y2)):
        length = int(max(abs(xb - xa), abs(yb - ya)))
        if length == 0:
            continue
        for start in range(0, length, dash * 2):
            end = min(start + dash, length)
            t0, t1 = start / length, end / length
            q1 = (int(xa + (xb - xa) * t0), int(ya + (yb - ya) * t0))
            q2 = (int(xa + (xb - xa) * t1), int(ya + (yb - ya) * t1))
            cv2.line(img, q1, q2, color, thickness)


def stamp_text(img: np.ndarray, text: str) -> None:
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 1.1, 2
    cv2.putText(img, text, (18, 46), font, scale, TEXT_OUTLINE, thick + 3, cv2.LINE_AA)
    cv2.putText(img, text, (18, 46), font, scale, TEXT_COLOR, thick, cv2.LINE_AA)


def open_writer(path: Path, fps: float, codec_pref: list[str]) -> tuple[cv2.VideoWriter | None, str | None]:
    for codec in codec_pref:
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, OUT_SIZE)
        if writer.isOpened():
            return writer, codec
        writer.release()
    return None, None


def main() -> None:
    args = parse_args()
    clip_id = args.video.stem
    tracks_csv = args.tracks_csv or (args.out_root / "tracks" / f"{clip_id}_tracks.csv")
    if not tracks_csv.exists():
        raise FileNotFoundError(f"Missing tracks CSV (run stage 1 first): {tracks_csv}")
    out_dir = args.out_root / "review_clips" / clip_id

    tracks = load_tracks(tracks_csv)
    if not tracks:
        print(json.dumps({"clip_id": clip_id, "snippets_written": 0, "note": "no kept tracks"}))
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")
    src_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 2560
    src_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1440
    sx, sy = OUT_SIZE[0] / src_w, OUT_SIZE[1] / src_h

    spans = {}
    interp = {}
    for tid, boxes in tracks.items():
        frames = sorted(boxes)
        spans[tid] = (max(0, frames[0] - args.pad_frames), frames[-1] + args.pad_frames)
        interp[tid] = interpolate_track(boxes)

    codec_pref = ["avc1", "mp4v"]
    codec_used: str | None = None
    writers: dict[int, cv2.VideoWriter] = {}
    failed: list[int] = []
    written: list[int] = []
    max_needed = max(end for _, end in spans.values())

    frame_idx = 0
    while frame_idx <= max_needed:
        ok, frame = cap.read()
        if not ok:
            break
        small = None
        for tid, (start, end) in spans.items():
            if tid in failed or not (start <= frame_idx <= end):
                continue
            if small is None:
                small = cv2.resize(frame, OUT_SIZE, interpolation=cv2.INTER_AREA)
            if tid not in writers:
                writer, codec = open_writer(out_dir / f"track_{tid:04d}.mp4", args.fps,
                                            [codec_used] if codec_used else codec_pref)
                if writer is None:
                    failed.append(tid)
                    continue
                codec_used = codec
                writers[tid] = writer
            canvas = small.copy()
            entry = interp[tid].get(frame_idx)
            if entry is not None:
                bbox, is_interp = entry
                p1 = (int(bbox[0] * sx), int(bbox[1] * sy))
                p2 = (int(bbox[2] * sx), int(bbox[3] * sy))
                if is_interp:
                    draw_dashed_rect(canvas, p1, p2, BOX_COLOR, 1)
                else:
                    cv2.rectangle(canvas, p1, p2, BOX_COLOR, 2)
            stamp_text(canvas, f"track {tid}  frame {frame_idx}")
            writers[tid].write(canvas)
            if frame_idx == end:
                writers[tid].release()
                del writers[tid]
                written.append(tid)
        frame_idx += 1

    for tid, writer in writers.items():  # tracks whose padded end passed EOF
        writer.release()
        written.append(tid)
    cap.release()

    final_codec = codec_used
    transcode_failed: list[int] = []
    if codec_used == "mp4v" and not args.no_transcode and args.ffmpeg.exists():
        for tid in sorted(written):
            if transcode_h264(out_dir / f"track_{tid:04d}.mp4", args.ffmpeg):
                final_codec = "h264_via_ffmpeg"
            else:
                transcode_failed.append(tid)

    summary = {
        "clip_id": clip_id,
        "tracks_csv": str(tracks_csv),
        "out_dir": str(out_dir),
        "codec_used": codec_used,
        "final_codec": final_codec,
        "transcode_failed_tracks": transcode_failed,
        "snippets_written": len(written),
        "failed_tracks": sorted(failed),
        "pad_frames": args.pad_frames,
        "output_size": list(OUT_SIZE),
        "fps": args.fps,
    }
    (out_dir / f"{clip_id}_review_clips_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
