"""Experiment 028 stage 3: sliding-window BIU-v2 interpretability traces per track.

Run with default python3 (pandas/sklearn/matplotlib):
    python3 scripts/experiments/cairo_dashcam_biu_traces.py --video_stem <clip>

Loads the Cairo keypoints JSONL from stage 2 and computes, per track, 16-frame
sliding-window (stride 1) BIU-v2 features by REUSING the exp-009 feature
builders (scripts/experiments/jaad_biu_v2_benchmark_comparison.py, which
provides src/biu_v2_extractor.extract_biu_v2) and the exp-023 bbox geometry
builder (scripts/experiments/jaad_ambiguous_negative_pattern_mining.py:
geometry_features, for path straightness and bbox looming). No feature logic is
reimplemented here.

Outputs (under experiments/028_cairo_dashcam_case_study/):
    traces/<clip>/track_<id>_trace.csv    one row per window (all BIU-v2 features
                                          + geo features), indexed by window end frame
    figures/<clip>/track_<id>_traces.png  6 key traces vs frame index + 3 thumbnails

Interpretation notes (see the experiment README):
  - BIU-v2 pose features are canonicalized (direction-flipped, normalized by
    image size / first-frame-relative) — these transfer across resolutions.
  - geo_bbox_* features are RAW-PIXEL quantities at 2560x1440 and are NOT
    directly comparable to JAAD's 1920x1080 pixel scale.

Pipeline-stage only: no model scoring. Hand labeling precedes any scoring.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CLEAN_ROOT = Path("/home/abdelrahman/Hadwa/BIU-PID-Distillation-clean")
SCRIPTS_DIR = CLEAN_ROOT / "scripts/experiments"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import jaad_biu_v2_benchmark_comparison as base  # noqa: E402  exp-009 feature machinery
from jaad_ambiguous_negative_pattern_mining import geometry_features  # noqa: E402  exp-023 bbox geometry

EXP_ROOT = CLEAN_ROOT / "experiments/028_cairo_dashcam_case_study"

# Key interpretable traces plotted in the per-track figure (title, column, color).
# Colors: dataviz reference palette, light-mode categorical slots in fixed order.
PANEL_SPECS = [
    ("Ankle center x-velocity (canon.)", "ankle_center_x_velocity_mean", "#2a78d6"),
    ("Hip center x-velocity (canon.)", "hip_center_x_velocity_mean", "#1baf7a"),
    ("Shoulder center x-delta (canon.)", "shoulder_center_x_delta", "#eda100"),
    ("Bbox path straightness (0-1)", "geo_path_straightness", "#008300"),
    ("Bbox area ratio last/first (looming)", "geo_bbox_area_ratio_last_first", "#4a3aa7"),
    ("Skeleton confidence mean", "skeleton_conf_mean", "#e34948"),
]
KEY_COLUMNS = [
    "ankle_center_x_velocity_mean",
    "knee_center_x_velocity_mean",
    "hip_center_x_velocity_mean",
    "shoulder_center_x_delta",
    "geo_path_straightness",
    "geo_bbox_area_ratio_last_first",
    "geo_bbox_height_mean",
    "skeleton_conf_mean",
]
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e7e6e2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video_stem", type=str, required=True, help="Clip stem, e.g. 20260715103404_0060.")
    parser.add_argument("--keypoints_jsonl", type=Path, default=None,
                        help="Default keypoints/<stem>_keypoints_vitpose_s.jsonl under out_root.")
    parser.add_argument("--out_root", type=Path, default=EXP_ROOT)
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--img_width", type=int, default=2560)
    parser.add_argument("--img_height", type=int, default=1440)
    parser.add_argument("--max_window_gap", type=int, default=2,
                        help="Windows whose max inter-frame gap exceeds this are flagged (kept, not dropped).")
    return parser.parse_args()


def load_track_records(jsonl_path: Path) -> dict[str, list[dict]]:
    tracks: dict[str, list[dict]] = defaultdict(list)
    with jsonl_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            tracks[str(rec["pid"])].append(rec)
    for pid in tracks:
        tracks[pid].sort(key=lambda r: int(r["frame_idx"]))
    return tracks


def window_features(recs: list[dict], image_size: tuple[int, int], max_window_gap: int) -> dict:
    frames = np.asarray([int(r["frame_idx"]) for r in recs])
    coords = np.asarray([r["keypoints"] for r in recs], dtype=float)
    conf = np.asarray([r["kp_conf"] for r in recs], dtype=float)
    bbox_seq = np.asarray([r["bbox_xyxy"] for r in recs], dtype=float)

    # Reused builders: exp-009 BIU-v2 extractor + exp-023 bbox geometry.
    feats = base.extract_biu_v2(
        coords, conf, bbox_sequence=bbox_seq, image_size=image_size, include_optional=True
    )
    geo_rec = pd.Series({"bbox_sequence_json": json.dumps(bbox_seq.tolist())})
    feats.update(geometry_features(geo_rec))

    gaps = np.diff(frames)
    feats["window_start_frame"] = int(frames[0])
    feats["window_end_frame"] = int(frames[-1])
    feats["window_max_gap"] = int(gaps.max()) if len(gaps) else 1
    feats["window_gap_flag"] = int(feats["window_max_gap"] > max_window_gap)
    return feats


def find_thumbnails(out_root: Path, clip_id: str, pid: str) -> list[Path]:
    tdir = out_root / "thumbnails" / clip_id / f"track_{int(pid):04d}"
    return sorted(tdir.glob("*.jpg"))[:3] if tdir.exists() else []


def plot_track(df: pd.DataFrame, clip_id: str, pid: str, thumbs: list[Path], out_path: Path) -> None:
    fig = plt.figure(figsize=(12, 10.5), facecolor=SURFACE)
    gs = fig.add_gridspec(4, 6, height_ratios=[1, 1, 1, 1.15], hspace=0.55, wspace=0.9,
                          left=0.07, right=0.97, top=0.90, bottom=0.03)
    x = df["window_end_frame"].to_numpy()

    # 6 trace panels laid out 3 rows x 2 cols across the first three grid rows.
    for i, (title, col, color) in enumerate(PANEL_SPECS):
        row, colpos = divmod(i, 2)
        ax = fig.add_subplot(gs[row, colpos * 3:(colpos + 1) * 3])
        ax.set_facecolor(SURFACE)
        y = df[col].to_numpy() if col in df.columns else np.full_like(x, np.nan, dtype=float)
        ax.plot(x, y, color=color, linewidth=2.0)
        gap_mask = df["window_gap_flag"].to_numpy().astype(bool)
        if gap_mask.any():
            ax.plot(x[gap_mask], y[gap_mask], linestyle="none", marker="o", markersize=4,
                    color=TEXT_SECONDARY, alpha=0.6)
        ax.set_title(title, fontsize=10, color=TEXT_PRIMARY, loc="left")
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xlabel("window end frame", fontsize=8, color=TEXT_SECONDARY)

    for j, thumb in enumerate(thumbs[:3]):
        ax = fig.add_subplot(gs[3, j * 2:(j + 1) * 2])
        try:
            ax.imshow(mpimg.imread(thumb))
        except Exception:
            pass
        ax.set_title(thumb.stem, fontsize=8, color=TEXT_SECONDARY)
        ax.axis("off")

    fig.suptitle(
        f"Cairo dashcam BIU-v2 traces — clip {clip_id}, track {pid} "
        f"(frames {int(x.min())}–{int(x.max())}, {len(df)} windows)",
        fontsize=12, color=TEXT_PRIMARY, x=0.07, ha="left",
    )
    fig.text(0.07, 0.925,
             "Zero-shot pipeline output; unlabeled. Gray dots mark windows containing tracking gaps.",
             fontsize=9, color=TEXT_SECONDARY)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    clip_id = args.video_stem
    jsonl_path = args.keypoints_jsonl or (args.out_root / "keypoints" / f"{clip_id}_keypoints_vitpose_s.jsonl")
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Missing keypoints JSONL (run stage 2 first): {jsonl_path}")

    image_size = (args.img_width, args.img_height)
    tracks = load_track_records(jsonl_path)
    traces_dir = args.out_root / "traces" / clip_id
    figures_dir = args.out_root / "figures" / clip_id
    traces_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    summary_tracks = {}
    for pid, recs in sorted(tracks.items(), key=lambda kv: int(kv[0])):
        if len(recs) < args.window:
            summary_tracks[pid] = {"n_records": len(recs), "n_windows": 0, "status": "too_short_for_window"}
            continue
        rows = []
        for start in range(0, len(recs) - args.window + 1, args.stride):
            feats = window_features(recs[start:start + args.window], image_size, args.max_window_gap)
            feats["clip_id"] = clip_id
            feats["track_id"] = pid
            rows.append(feats)
        df = pd.DataFrame(rows)
        lead = ["clip_id", "track_id", "window_start_frame", "window_end_frame",
                "window_max_gap", "window_gap_flag"] + [c for c in KEY_COLUMNS if c in df.columns]
        df = df[lead + [c for c in df.columns if c not in lead]]
        trace_path = traces_dir / f"track_{int(pid):04d}_trace.csv"
        df.to_csv(trace_path, index=False)

        thumbs = find_thumbnails(args.out_root, clip_id, pid)
        fig_path = figures_dir / f"track_{int(pid):04d}_traces.png"
        plot_track(df, clip_id, pid, thumbs, fig_path)

        summary_tracks[pid] = {
            "n_records": len(recs),
            "n_windows": len(df),
            "windows_with_gap_flag": int(df["window_gap_flag"].sum()),
            "mean_skeleton_conf": round(float(df["skeleton_conf_mean"].mean()), 4)
            if "skeleton_conf_mean" in df.columns else None,
            "trace_csv": str(trace_path),
            "figure_png": str(fig_path),
        }

    summary = {
        "clip_id": clip_id,
        "keypoints_jsonl": str(jsonl_path),
        "window": args.window,
        "stride": args.stride,
        "image_size": list(image_size),
        "reused_builders": [
            "scripts/experiments/jaad_biu_v2_benchmark_comparison.py (extract_biu_v2 via src/biu_v2_extractor.py)",
            "scripts/experiments/jaad_ambiguous_negative_pattern_mining.py:geometry_features",
        ],
        "tracks": summary_tracks,
    }
    summary_path = traces_dir / f"{clip_id}_biu_traces_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
