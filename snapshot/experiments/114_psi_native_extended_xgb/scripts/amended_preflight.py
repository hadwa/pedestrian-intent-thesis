"""Freeze the user-approved Exp114 complete-input development cohort."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
EXP113 = REPO / "experiments/113_psi_native_biu_xgb"
EXCLUDED = ("video_0032", "video_0055", "video_0093")


def record(path: Path) -> dict[str, object]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest}


def main() -> None:
    amendment = json.loads((ROOT / "amended_cohort.json").read_text())
    assert tuple(amendment["train_excluded_videos"]) == EXCLUDED
    train = pd.read_csv(EXP113 / "cache/train/rows.csv")
    val = pd.read_csv(EXP113 / "cache/val/rows.csv")
    counts = {video: int(train.video_id.astype(str).eq(video).sum()) for video in EXCLUDED}
    val_counts = {video: int(val.video_id.astype(str).eq(video).sum()) for video in EXCLUDED}
    retained = train.loc[~train.video_id.astype(str).isin(EXCLUDED)].copy()
    assert counts == {"video_0032": 143, "video_0055": 141, "video_0093": 301}
    assert sum(val_counts.values()) == 0
    assert len(retained) == 26295 and len(val) == 14922
    assert retained.sequence_id.is_unique and val.sequence_id.is_unique
    assert not set(retained.sequence_id) & set(val.sequence_id)
    required = {}
    for split, rows in (("train", retained), ("val", val)):
        frames = {
            (str(row.video_id), int(frame))
            for row in rows.itertuples(index=False)
            for frame in json.loads(row.observed_frames_json)
        }
        required[split] = {
            "windows": len(rows),
            "videos": int(rows.video_id.nunique()),
            "tracks": int(rows.track_id.nunique()),
            "unique_video_frames": len(frames),
            "positives": int(rows.label.sum()),
        }
    output = {
        "status": "frozen_ready_for_source_recovery",
        "development_only": True,
        "test_accessed": False,
        "excluded_train_windows": counts,
        "excluded_validation_windows": val_counts,
        "cohort": required,
        "arms": {"S187": 187, "R247": 247},
        "planned_fits": 6,
        "video_download": {
            "members": 140,
            "bytes": 4432752457,
            "compressed_bytes": 4433470344,
        },
        "sources": [
            record(ROOT / "APPROVED_COHORT_AMENDMENT.md"),
            record(ROOT / "amended_cohort.json"),
            record(ROOT / "feature_contract.json"),
            record(ROOT / "live_zip_directory_audit.json"),
            record(EXP113 / "cache/train/rows.csv"),
            record(EXP113 / "cache/val/rows.csv"),
            record(EXP113 / "cache/train/A.npy"),
            record(EXP113 / "cache/train/B.npy"),
            record(EXP113 / "cache/train/C.npy"),
            record(EXP113 / "cache/val/A.npy"),
            record(EXP113 / "cache/val/B.npy"),
            record(EXP113 / "cache/val/C.npy"),
        ],
    }
    path = ROOT / "metrics/amended_preflight.json"
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
