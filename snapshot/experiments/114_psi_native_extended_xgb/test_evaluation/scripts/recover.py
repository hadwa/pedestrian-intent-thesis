"""Recover only official sources required by the frozen Exp114 test cohort."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import sys
import zipfile
import zlib
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXP114 = ROOT.parent
EXP113_TEST = EXP114.parent / "113_psi_native_biu_xgb/test_evaluation"
SPEC = importlib.util.spec_from_file_location("psi114_test_remote", EXP114 / "scripts/recover_sources.py")
REMOTE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = REMOTE
SPEC.loader.exec_module(REMOTE)


def cohort():
    frame = pd.read_csv(EXP113_TEST / "cache/rows.csv")
    assert len(frame) == 20547 and frame.sequence_id.is_unique
    needed = {}
    for row in frame.itertuples(index=False):
        frames = list(map(int, json.loads(row.observed_frames_json)))
        assert len(frames) == 15 and frames == list(range(frames[0], frames[0] + 15))
        assert int(row.target_frame) == frames[-1] + 1
        needed.setdefault(str(row.video_id), set()).update(frames)
    assert len(needed) == 49 and sum(map(len, needed.values())) == 13144
    return frame, needed


def speed() -> None:
    frame, needed = cohort()
    remote = REMOTE.RemoteZip()
    records = []
    with zipfile.ZipFile(remote) as archive:
        for number, video in enumerate(sorted(needed), start=1):
            member = f"PSI/PSI2.0_Test/annotations/cv_annotation/{video}/cv_annotation.json"
            info = archive.getinfo(member)
            destination = ROOT / f"cache/speed/{video}.json"
            if destination.is_file():
                compact = json.loads(destination.read_text())
                values = {int(key): float(value) for key, value in compact["speed_km_hr"].items()}
            else:
                payload = archive.read(member)
                if len(payload) != info.file_size or zlib.crc32(payload) != info.CRC:
                    raise RuntimeError(f"CV integrity failure: {video}")
                source = json.loads(payload)
                if source.get("video_name") != video or not isinstance(source.get("frames"), dict):
                    raise RuntimeError(f"CV identity failure: {video}")
                values = {}
                for key, item in source["frames"].items():
                    match = re.fullmatch(r"frame_(\d+)", str(key))
                    if match is None or "speed(km/hr)" not in item:
                        raise RuntimeError(f"malformed CV speed: {video}/{key}")
                    index, value = int(match.group(1)), float(item["speed(km/hr)"])
                    if index in values or not (-1e6 < value < 1e6):
                        raise RuntimeError(f"invalid CV speed: {video}/{key}")
                    values[index] = value
                compact = {
                    "video": video, "unit": "km/hr", "source_member": member,
                    "source_bytes": info.file_size, "source_compressed_bytes": info.compress_size,
                    "source_crc32": info.CRC, "source_sha256": hashlib.sha256(payload).hexdigest(),
                    "speed_km_hr": {str(key): values[key] for key in sorted(values)},
                }
                REMOTE.atomic_json(destination, compact)
            missing = needed[video] - set(values)
            if missing:
                raise RuntimeError(f"missing observed test speed: {video}/frame_{min(missing)}")
            selected = [values[index] for index in needed[video]]
            records.append({"video": video, "required_frames": len(needed[video]),
                            "minimum_km_hr": min(selected), "maximum_km_hr": max(selected),
                            "source_crc32": info.CRC, "compact_path": str(destination)})
            print(f"speed {number}/49 {video}: {len(needed[video])} frames", flush=True)
    REMOTE.atomic_json(ROOT / "metrics/speed_recovery.json", {
        "status": "complete", "test_authorized": True, "windows": len(frame), "videos": 49,
        "required_video_frames": 13144, "unit": "km/hr",
        "all_required_frames_have_numeric_speed": True, "archive_etag": remote.etag,
        "transferred_bytes": remote.transferred, "records": records,
    })


def videos(shard_index: int, shard_count: int) -> None:
    _, needed = cohort()
    gate = json.loads((ROOT / "metrics/speed_recovery.json").read_text())
    if not gate.get("all_required_frames_have_numeric_speed") or gate.get("videos") != 49:
        raise RuntimeError("test videos require complete official-speed gate")
    if shard_count <= 0 or not 0 <= shard_index < shard_count:
        raise ValueError("invalid shard")
    chosen = sorted(needed)[shard_index::shard_count]
    remote = REMOTE.RemoteZip()
    records = []
    with zipfile.ZipFile(remote) as archive:
        for number, video in enumerate(chosen, start=1):
            member = f"PSI/PSI2.0_Test/videos/{video}.mp4"
            info = archive.getinfo(member)
            destination = ROOT / f"cache/videos/{video}.mp4"
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not (destination.is_file() and destination.stat().st_size == info.file_size and REMOTE.local_crc(destination) == info.CRC):
                temporary = destination.with_suffix(".mp4.tmp")
                with archive.open(info) as source, temporary.open("wb") as output:
                    for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
                        output.write(block)
                if temporary.stat().st_size != info.file_size or REMOTE.local_crc(temporary) != info.CRC:
                    raise RuntimeError(f"video integrity failure: {video}")
                os.replace(temporary, destination)
            records.append({"video": video, "path": str(destination), "bytes": info.file_size,
                            "compressed_bytes": info.compress_size, "crc32": info.CRC,
                            "required_frames": len(needed[video])})
            print(f"video shard{shard_index} {number}/{len(chosen)} {video}", flush=True)
    name = "video_recovery.json" if shard_count == 1 else f"video_recovery_shard_{shard_index}_of_{shard_count}.json"
    REMOTE.atomic_json(ROOT / f"metrics/{name}", {
        "status": "complete", "test_authorized": True, "videos": len(records),
        "total_expected_videos": 49, "shard_index": shard_index, "shard_count": shard_count,
        "archive_etag": remote.etag, "transferred_bytes": remote.transferred, "records": records,
    })


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("speed", "videos"))
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()
    speed() if args.stage == "speed" else videos(args.shard_index, args.shard_count)
