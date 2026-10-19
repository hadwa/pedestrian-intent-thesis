"""Lightweight smoke test for the BIU v2 specification."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CLEAN_ROOT = Path("/home/abdelrahman/Hadwa/BIU-PID-Distillation-clean")
SRC_ROOT = CLEAN_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from biu_v2_extractor import FEATURE_DICTIONARY, feature_dictionary_df, extract_biu_v2, v1_v2_mapping_df

CANONICAL_ROWS_PATH = CLEAN_ROOT / "experiments/004_jaad_temporal_biu_transition/metrics/horizon_feature_rows.csv"
KEYPOINT_JSONL_PATH = CLEAN_ROOT / "data/jaad_keypoints_vitpose_s_extended_short_horizon.jsonl"
OUT_DIR = CLEAN_ROOT / "experiments/008_biu_v2_design_spec/metrics"


def _load_keypoint_index() -> dict[tuple[str, str, int], dict]:
    idx = {}
    with KEYPOINT_JSONL_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            rec = json.loads(line)
            idx[(str(rec["video"]), str(rec["pid"]), int(rec["frame_idx"]))] = rec
    return idx


def _required_output_names() -> set[str]:
    names = {"direction_sign", "direction_source", "was_flipped", "direction_confidence"}
    for row in FEATURE_DICTIONARY:
        if row["required_or_optional"] != "required":
            continue
        stats = [s for s in row["summary_stats"].split(";") if s]
        for stat in stats:
            names.add(f"{row['feature_name']}_{stat}")
    return names


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    feature_dictionary_df().to_csv(OUT_DIR / "biu_v2_feature_dictionary.csv", index=False)
    v1_v2_mapping_df().to_csv(OUT_DIR / "biu_v1_vs_v2_mapping.csv", index=False)

    rows = pd.read_csv(CANONICAL_ROWS_PATH)
    rows = rows[
        (rows["horizon_label"].isin(["30_60", "15_30", "5_15"]))
        & (rows["group"].isin(["actual_positive", "ambiguous_negative", "clean_negative"]))
    ].copy()
    sample_rows = (
        rows.sort_values(["horizon_label", "group", "sequence_id"])
        .groupby(["horizon_label", "group"], as_index=False)
        .head(2)
    )

    kp_index = _load_keypoint_index()
    expected = _required_output_names()
    summary_rows = []
    total_missing = 0
    total_nan = 0
    flipped = 0
    succeeded = 0

    for _, rec in sample_rows.iterrows():
        frames = json.loads(rec["observed_frames_json"])
        bbox = np.asarray(json.loads(rec["bbox_sequence_json"]), dtype=float)
        kp_rows = []
        missing_frame = None
        for frame in frames:
            item = kp_index.get((str(rec["video_id"]), str(rec["ped_id"]), int(frame)))
            if item is None:
                missing_frame = frame
                break
            kp_rows.append(item)
        if missing_frame is not None:
            summary_rows.append(
                {
                    "sequence_id": rec["sequence_id"],
                    "horizon_label": rec["horizon_label"],
                    "group": rec["group"],
                    "status": f"missing_keypoint_frame:{missing_frame}",
                }
            )
            continue

        coords = np.asarray([r["keypoints"] for r in kp_rows], dtype=float)
        conf = np.asarray([r["kp_conf"] for r in kp_rows], dtype=float)
        feats = extract_biu_v2(coords, conf, bbox_sequence=bbox)
        feat_keys = set(feats.keys())
        missing_required = sorted(expected - feat_keys)
        nan_count = int(sum(int(pd.isna(v)) for v in feats.values() if not isinstance(v, str)))
        summary_rows.append(
            {
                "sequence_id": rec["sequence_id"],
                "horizon_label": rec["horizon_label"],
                "group": rec["group"],
                "status": "ok",
                "feature_count": len(feats),
                "missing_required_count": len(missing_required),
                "nan_feature_count": nan_count,
                "direction_source": feats["direction_source"],
                "was_flipped": feats["was_flipped"],
                "direction_confidence": feats["direction_confidence"],
            }
        )
        total_missing += len(missing_required)
        total_nan += nan_count
        flipped += int(feats["was_flipped"])
        succeeded += 1

    sample_df = pd.DataFrame(summary_rows)
    sample_df.to_csv(OUT_DIR / "biu_v2_smoke_test_samples.csv", index=False)

    smoke_summary = pd.DataFrame(
        [
            {
                "sample_count_attempted": int(len(sample_rows)),
                "sample_count_succeeded": int(succeeded),
                "direction_flipped_count": int(flipped),
                "required_feature_key_count": int(len(expected)),
                "total_missing_required_keys": int(total_missing),
                "total_nan_feature_values": int(total_nan),
                "mean_nan_feature_values_per_sample": float(total_nan / max(succeeded, 1)),
            }
        ]
    )
    smoke_summary.to_csv(OUT_DIR / "biu_v2_smoke_test_summary.csv", index=False)
    print(smoke_summary.to_json(orient="records", indent=2))


if __name__ == "__main__":
    main()
