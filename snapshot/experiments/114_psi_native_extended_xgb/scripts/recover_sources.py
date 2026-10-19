"""Resume-safe recovery of only amended Exp114 train/validation sources."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import re
import urllib.request
import zipfile
import zlib
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
EXP113 = REPO / "experiments/113_psi_native_biu_xgb"
URL = "https://s3.amazonaws.com/pedestrian.datatset.files/PSI_Dataset_1_2.zip"
EXPECTED_ETAG = '"3868cb96ae54e00a99ad64d9e17bc701-325"'
EXCLUDED = {"video_0032", "video_0055", "video_0093"}


class RemoteZip(io.RawIOBase):
    """HTTP range reader capped to bounded chunks for large stored videos."""

    def __init__(self) -> None:
        request = urllib.request.Request(URL, method="HEAD")
        with urllib.request.urlopen(request, timeout=60) as response:
            self.size = int(response.headers["Content-Length"])
            self.etag = response.headers["ETag"]
        if self.etag != EXPECTED_ETAG:
            raise RuntimeError("PSI archive ETag changed")
        self.position = 0
        self.transferred = 0

    def seekable(self) -> bool:
        return True

    def seek(self, offset: int, whence: int = 0) -> int:
        self.position = offset if whence == 0 else (self.position if whence == 1 else self.size) + offset
        return self.position

    def tell(self) -> int:
        return self.position

    def read(self, size: int = -1) -> bytes:
        remaining = self.size - self.position
        requested = remaining if size < 0 else min(size, remaining)
        size = min(requested, 16 * 1024 * 1024)
        if size <= 0:
            return b""
        start, end = self.position, self.position + size - 1
        request = urllib.request.Request(
            URL,
            headers={"Range": f"bytes={start}-{end}", "If-Match": self.etag},
        )
        with urllib.request.urlopen(request, timeout=180) as response:
            expected = f"bytes {start}-{end}/{self.size}"
            if response.status != 206 or response.headers["Content-Range"] != expected:
                raise RuntimeError("incorrect PSI HTTP range response")
            data = response.read()
        if len(data) != size:
            raise RuntimeError("truncated PSI HTTP range response")
        self.position += size
        self.transferred += size
        return data


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def local_crc(path: Path) -> int:
    crc = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            crc = zlib.crc32(block, crc)
    return crc & 0xFFFFFFFF


def cohort() -> tuple[dict[str, pd.DataFrame], dict[str, set[int]]]:
    train = pd.read_csv(EXP113 / "cache/train/rows.csv")
    val = pd.read_csv(EXP113 / "cache/val/rows.csv")
    train = train.loc[~train.video_id.astype(str).isin(EXCLUDED)].copy()
    assert len(train) == 26295 and len(val) == 14922
    rows = {"train": train, "val": val}
    needed: dict[str, set[int]] = {}
    for frame in rows.values():
        for row in frame.itertuples(index=False):
            frames = json.loads(row.observed_frames_json)
            assert len(frames) == 15 and frames == list(range(frames[0], frames[0] + 15))
            needed.setdefault(str(row.video_id), set()).update(map(int, frames))
    assert len(needed) == 140 and not set(needed) & EXCLUDED
    return rows, needed


def recover_speed() -> None:
    rows, needed = cohort()
    remote = RemoteZip()
    records = []
    with zipfile.ZipFile(remote) as archive:
        for index, video in enumerate(sorted(needed), start=1):
            member = f"PSI/PSI2.0_TrainVal/annotations/cv_annotation/{video}/cv_annotation.json"
            info = archive.getinfo(member)
            destination = ROOT / "cache/speed" / f"{video}.json"
            if destination.is_file():
                compact = json.loads(destination.read_text())
                speed = {int(key): float(value) for key, value in compact["speed_km_hr"].items()}
            else:
                payload = archive.read(member)
                if len(payload) != info.file_size or zlib.crc32(payload) != info.CRC:
                    raise RuntimeError(f"CV member integrity failed: {video}")
                source = json.loads(payload)
                if source.get("video_name") != video or not isinstance(source.get("frames"), dict):
                    raise RuntimeError(f"CV identity failed: {video}")
                speed = {}
                for key, value in source["frames"].items():
                    match = re.fullmatch(r"frame_(\d+)", str(key))
                    if match is None or "speed(km/hr)" not in value:
                        raise RuntimeError(f"malformed speed frame: {video}/{key}")
                    frame, number = int(match.group(1)), float(value["speed(km/hr)"])
                    if frame in speed or not (-1e6 < number < 1e6):
                        raise RuntimeError(f"invalid speed value: {video}/{key}")
                    speed[frame] = number
                compact = {
                    "video": video,
                    "unit": "km/hr",
                    "source_member": member,
                    "source_bytes": info.file_size,
                    "source_compressed_bytes": info.compress_size,
                    "source_crc32": info.CRC,
                    "source_sha256": hashlib.sha256(payload).hexdigest(),
                    "speed_km_hr": {str(key): speed[key] for key in sorted(speed)},
                }
                atomic_json(destination, compact)
            missing = sorted(needed[video] - set(speed))
            if missing:
                raise RuntimeError(f"missing retained observed speed: {video}/frame_{missing[0]}")
            observed = [speed[frame] for frame in needed[video]]
            records.append({
                "video": video,
                "required_frames": len(needed[video]),
                "minimum_km_hr": min(observed),
                "maximum_km_hr": max(observed),
                "compact_path": str(destination),
                "source_member": member,
                "source_crc32": info.CRC,
            })
            print(f"speed {index}/140 {video}: {len(needed[video])} required frames", flush=True)
    manifest = {
        "status": "complete",
        "development_only": True,
        "test_accessed": False,
        "videos": len(records),
        "train_windows": len(rows["train"]),
        "validation_windows": len(rows["val"]),
        "required_frame_assignments": sum(len(value) for value in needed.values()),
        "all_required_frames_have_numeric_speed": True,
        "unit": "km/hr",
        "archive_etag": remote.etag,
        "transferred_bytes": remote.transferred,
        "records": records,
    }
    atomic_json(ROOT / "metrics/speed_recovery.json", manifest)


def recover_videos(*, shard_index: int = 0, shard_count: int = 1) -> None:
    _, needed = cohort()
    speed_manifest = json.loads((ROOT / "metrics/speed_recovery.json").read_text())
    if speed_manifest.get("status") != "complete" or speed_manifest.get("videos") != 140:
        raise RuntimeError("video recovery requires complete exact-speed gate")
    remote = RemoteZip()
    if shard_count <= 0 or not 0 <= shard_index < shard_count:
        raise ValueError("invalid recovery shard")
    selected = sorted(needed)[shard_index::shard_count]
    records = []
    with zipfile.ZipFile(remote) as archive:
        for index, video in enumerate(selected, start=1):
            member = f"PSI/PSI2.0_videos_Train-Val/videos/{video}.mp4"
            info = archive.getinfo(member)
            destination = ROOT / "cache/videos" / f"{video}.mp4"
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not (destination.is_file() and destination.stat().st_size == info.file_size and local_crc(destination) == info.CRC):
                temporary = destination.with_suffix(".mp4.tmp")
                with archive.open(info) as source, temporary.open("wb") as output:
                    for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
                        output.write(block)
                if temporary.stat().st_size != info.file_size or local_crc(temporary) != info.CRC:
                    raise RuntimeError(f"video integrity failed: {video}")
                os.replace(temporary, destination)
            records.append({
                "video": video,
                "path": str(destination),
                "bytes": info.file_size,
                "compressed_bytes": info.compress_size,
                "crc32": info.CRC,
                "required_frames": len(needed[video]),
            })
            atomic_json(ROOT / f"cache/video_recovery_progress_{shard_index}_of_{shard_count}.json", {
                "status": "running", "completed": index, "total": len(selected),
                "test_accessed": False, "records": records,
            })
            print(f"video shard {shard_index} {index}/{len(selected)} {video}: {info.file_size} bytes verified", flush=True)
    output = ROOT / ("metrics/video_recovery.json" if shard_count == 1 else f"metrics/video_recovery_shard_{shard_index}_of_{shard_count}.json")
    atomic_json(output, {
        "status": "complete", "development_only": True, "test_accessed": False,
        "videos": len(records), "total_expected_videos": 140,
        "shard_index": shard_index, "shard_count": shard_count, "archive_etag": remote.etag,
        "transferred_bytes": remote.transferred, "records": records,
    })


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("speed", "videos"))
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    arguments = parser.parse_args()
    recover_speed() if arguments.stage == "speed" else recover_videos(
        shard_index=arguments.shard_index, shard_count=arguments.shard_count
    )
