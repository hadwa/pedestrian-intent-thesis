"""Dry-run frozen-window input extraction for the local MSLSTM-PID repo.

This script validates whether MSLSTM-PID streams can be read for the existing
16-frame frozen benchmark rows. It does not train, impute, repair, or write
model-ready datasets.
"""

from __future__ import annotations

import argparse
import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_STREAMS = [
    "bbox",
    "ego_speed",
    "ego_acc",
    "ped_action",
    "ped_look",
    "static_context",
    "pose",
    "yolop",
]

STATIC_SIGNALIZED = 4
STATIC_INTERSECTION = 5
STATIC_AGE = 4
STATIC_GENDER = 3
STATIC_TRAFFIC_DIR = 2
STATIC_LANE_CATEGORIES = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 4, 7: 4, 8: 4}
STATIC_LANE_DIM = len(set(STATIC_LANE_CATEGORIES.values()))


@dataclass
class StreamResult:
    ok: bool
    tensor: np.ndarray | None
    message: str = ""


def one_hot(index: int, size: int) -> np.ndarray:
    out = np.zeros(size, dtype=np.float32)
    out[int(np.clip(index, 0, size - 1))] = 1.0
    return out


def load_pickle(path: Path) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        with path.open("rb") as f:
            return pickle.load(f)


class FrozenMSLSTMReader:
    def __init__(self, mslstm_root: Path, streams: list[str]):
        self.root = mslstm_root
        self.streams = streams
        self.db = load_pickle(self.root / "PIE/annotations/data_cache/pie_database.pkl")
        self.scalers = load_pickle(self.root / "checkpoints/scalers.pkl")
        self.pose_cache: dict[tuple[str, str], dict[int, dict[str, np.ndarray]] | None] = {}
        self.yolop_cache: dict[tuple[str, str], dict[int, dict[str, np.ndarray]] | None] = {}

    def video_key(self, set_id: str, video_id: str) -> str:
        return f"{set_id}_{video_id}"

    def frames_for_row(self, row: pd.Series) -> list[int]:
        return list(range(int(row["start_frame"]), int(row["end_frame"]) + 1))

    def track(self, row: pd.Series) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        vdb = self.db.get(row["set_id"], {}).get(row["video_id"])
        if vdb is None:
            return None, None
        return vdb, vdb.get("ped_annotations", {}).get(row["ped_id"])

    def frame_index(self, pdb: dict[str, Any]) -> dict[int, int]:
        return {int(frame): idx for idx, frame in enumerate(pdb.get("frames", []))}

    def load_pose_video(self, set_id: str, video_id: str) -> dict[int, dict[str, np.ndarray]] | None:
        key = (set_id, video_id)
        if key not in self.pose_cache:
            path = self.root / "Extracted Poses" / set_id / f"{set_id}_{video_id}_poses.pkl"
            if path.exists():
                self.pose_cache[key] = load_pickle(path).get(self.video_key(set_id, video_id), {})
            else:
                self.pose_cache[key] = None
        return self.pose_cache[key]

    def load_yolop_video(self, set_id: str, video_id: str) -> dict[int, dict[str, np.ndarray]] | None:
        key = (set_id, video_id)
        if key not in self.yolop_cache:
            path = self.root / "Extracted YOLOP" / f"{set_id}_{video_id}_yolop_features.pkl"
            if path.exists():
                self.yolop_cache[key] = load_pickle(path).get(self.video_key(set_id, video_id), {})
            else:
                self.yolop_cache[key] = None
        return self.yolop_cache[key]

    def bbox(self, row: pd.Series) -> StreamResult:
        vdb, pdb = self.track(row)
        if vdb is None or pdb is None:
            return StreamResult(False, None, "missing PIE track")
        imap = self.frame_index(pdb)
        bboxes = pdb.get("bbox", [])
        width = float(vdb.get("width", 1920) or 1920)
        height = float(vdb.get("height", 1080) or 1080)
        values = []
        for frame in self.frames_for_row(row):
            idx = imap.get(frame)
            if idx is None or idx >= len(bboxes):
                return StreamResult(False, None, f"missing bbox at frame {frame}")
            x1, y1, x2, y2 = bboxes[idx]
            values.append([(x1 + x2) / 2 / width, (y1 + y2) / 2 / height, (x2 - x1) / width, (y2 - y1) / height])
        return StreamResult(True, np.asarray(values, dtype=np.float32))

    def ego_speed(self, row: pd.Series) -> StreamResult:
        vdb, _ = self.track(row)
        if vdb is None:
            return StreamResult(False, None, "missing PIE video")
        ego = vdb.get("vehicle_annotations", {})
        mean = self.scalers.get("ego_speed_mean", 0.0)
        std = self.scalers.get("ego_speed_std", 1.0) or 1.0
        values = []
        for frame in self.frames_for_row(row):
            item = ego.get(frame)
            if not isinstance(item, dict):
                return StreamResult(False, None, f"missing ego at frame {frame}")
            speed = item.get("OBD_speed", item.get("GPS_speed"))
            if speed is None:
                return StreamResult(False, None, f"missing speed at frame {frame}")
            values.append([(float(speed) - mean) / std])
        return StreamResult(True, np.asarray(values, dtype=np.float32))

    def ego_acc(self, row: pd.Series) -> StreamResult:
        vdb, _ = self.track(row)
        if vdb is None:
            return StreamResult(False, None, "missing PIE video")
        ego = vdb.get("vehicle_annotations", {})
        ax_mean = self.scalers.get("accX_mean", 0.0)
        ax_std = self.scalers.get("accX_std", 1.0) or 1.0
        ay_mean = self.scalers.get("accY_mean", 0.0)
        ay_std = self.scalers.get("accY_std", 1.0) or 1.0
        values = []
        for frame in self.frames_for_row(row):
            item = ego.get(frame)
            if not isinstance(item, dict):
                return StreamResult(False, None, f"missing ego at frame {frame}")
            if "accX" not in item or "accY" not in item:
                return StreamResult(False, None, f"missing acceleration at frame {frame}")
            values.append([(float(item["accX"]) - ax_mean) / ax_std, (float(item["accY"]) - ay_mean) / ay_std])
        return StreamResult(True, np.asarray(values, dtype=np.float32))

    def behavior(self, row: pd.Series, name: str) -> StreamResult:
        _, pdb = self.track(row)
        if pdb is None:
            return StreamResult(False, None, "missing PIE track")
        imap = self.frame_index(pdb)
        values_by_frame = pdb.get("behavior", {}).get(name, [])
        values = []
        for frame in self.frames_for_row(row):
            idx = imap.get(frame)
            if idx is None or idx >= len(values_by_frame):
                return StreamResult(False, None, f"missing {name} at frame {frame}")
            values.append([float(values_by_frame[idx])])
        return StreamResult(True, np.asarray(values, dtype=np.float32))

    def static_context(self, row: pd.Series) -> StreamResult:
        _, pdb = self.track(row)
        if pdb is None:
            return StreamResult(False, None, "missing PIE track")
        attr = pdb.get("attributes", {})
        missing = [key for key in ["signalized", "intersection", "age", "gender", "traffic_direction", "num_lanes"] if key not in attr]
        if missing:
            return StreamResult(False, None, f"missing static fields: {missing}")
        lane = STATIC_LANE_CATEGORIES.get(attr.get("num_lanes"), max(STATIC_LANE_CATEGORIES.values()))
        vec = np.concatenate(
            [
                one_hot(int(attr.get("signalized", 0)), STATIC_SIGNALIZED),
                one_hot(int(attr.get("intersection", 0)), STATIC_INTERSECTION),
                one_hot(int(attr.get("age", 2)), STATIC_AGE),
                one_hot(int(attr.get("gender", 0)), STATIC_GENDER),
                one_hot(int(attr.get("traffic_direction", 0)), STATIC_TRAFFIC_DIR),
                one_hot(int(lane), STATIC_LANE_DIM),
            ]
        ).astype(np.float32)
        return StreamResult(True, np.repeat(vec[None, :], len(self.frames_for_row(row)), axis=0))

    def pose(self, row: pd.Series) -> StreamResult:
        data = self.load_pose_video(row["set_id"], row["video_id"])
        if data is None:
            return StreamResult(False, None, "missing pose file")
        values = []
        for frame in self.frames_for_row(row):
            value = data.get(frame, {}).get(row["ped_id"])
            if not isinstance(value, np.ndarray) or value.shape != (34,):
                return StreamResult(False, None, f"missing pose at frame {frame}")
            values.append(value)
        return StreamResult(True, np.asarray(values, dtype=np.float32))

    def yolop(self, row: pd.Series) -> StreamResult:
        data = self.load_yolop_video(row["set_id"], row["video_id"])
        if data is None:
            return StreamResult(False, None, "missing YOLOP file")
        values = []
        for frame in self.frames_for_row(row):
            value = data.get(frame, {}).get(row["ped_id"])
            if not isinstance(value, np.ndarray) or value.shape != (20,):
                return StreamResult(False, None, f"missing YOLOP at frame {frame}")
            values.append(value)
        return StreamResult(True, np.asarray(values, dtype=np.float32))

    def extract(self, row: pd.Series) -> dict[str, StreamResult]:
        out = {}
        for stream in self.streams:
            if stream == "ped_action":
                out[stream] = self.behavior(row, "action")
            elif stream == "ped_look":
                out[stream] = self.behavior(row, "look")
            else:
                out[stream] = getattr(self, stream)(row)
        return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dry-run MSLSTM-PID frozen input extraction.")
    parser.add_argument("--mslstm_root", type=Path, default=Path("/home/abdelrahman/Hadwa/MSLSTM-PID"))
    parser.add_argument("--splits_dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--sample_per_split", type=int, default=25)
    parser.add_argument("--streams", nargs="+", default=DEFAULT_STREAMS)
    parser.add_argument("--report_path", type=Path, default=Path("MSLSTM_PID_TEACHER_INPUT_DRY_RUN_REPORT.md"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reader = FrozenMSLSTMReader(args.mslstm_root, args.streams)
    lines = [
        "# MSLSTM-PID Teacher Input Dry-Run Report",
        "",
        "This dry run only validates tensor extraction for a small sample of existing frozen 16-frame windows. It does not train, repair data, or write model-ready tensors.",
        "",
        f"- MSLSTM root: `{args.mslstm_root}`",
        f"- Streams checked: `{', '.join(args.streams)}`",
        f"- Sample per split: `{args.sample_per_split}`",
        "",
        "## Results",
        "",
        "| Split | Rows checked | Fully successful | Failed |",
        "|---|---:|---:|---:|",
    ]
    total_checked = 0
    total_failed = 0
    detail_lines: list[str] = []
    shape_lines: list[str] = []

    for split in ["train", "val", "test"]:
        df = pd.read_csv(args.splits_dir / f"{split}_sequences.csv").head(args.sample_per_split)
        ok_count = 0
        for _, row in df.iterrows():
            results = reader.extract(row)
            missing = {name: res.message for name, res in results.items() if not res.ok}
            if missing:
                total_failed += 1
                detail_lines.append(f"- `{split}` `{row['sequence_id']}` failed: {missing}")
            else:
                ok_count += 1
                if len(shape_lines) < 12:
                    shapes = {name: tuple(res.tensor.shape) for name, res in results.items() if res.tensor is not None}
                    shape_lines.append(f"- `{split}` `{row['sequence_id']}` shapes: `{shapes}`")
            total_checked += 1
        lines.append(f"| {split} | {len(df)} | {ok_count} | {len(df) - ok_count} |")

    lines.extend(["", "## Example Tensor Shapes", ""])
    lines.extend(shape_lines or ["- No fully successful sampled rows."])
    lines.extend(["", "## Failures", ""])
    lines.extend(detail_lines[:50] or ["- No failures in sampled rows."])
    args.report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"dry-run checked={total_checked} failed={total_failed} report={args.report_path}")


if __name__ == "__main__":
    main()
