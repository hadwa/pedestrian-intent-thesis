import json
import sys
from pathlib import Path

import pandas as pd
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import pose_support  # noqa: E402


def _window(sequence_id, split, pid, frames, boxes=None):
    if boxes is None:
        boxes = [[float(frame), 1.0, float(frame + 10), 21.0] for frame in frames]
    return {
        "sequence_id": sequence_id,
        "split": split,
        "video_id": "video_0001",
        "ped_id": pid,
        "observed_frames_json": json.dumps(frames),
        "bbox_sequence_json": json.dumps(boxes),
        "label": 0,
    }


def test_missing_manifest_uses_exact_ids_and_deduplicates_frames(tmp_path):
    windows = pd.DataFrame(
        [
            _window("s1", "train", "0_1_2", [0, 1, 2]),
            _window("s2", "train", "0_1_2", [1, 2, 3]),
            _window("s3", "test", "0_1_2b", [0, 1, 2]),
        ]
    )
    window_path = tmp_path / "windows.csv"
    windows.to_csv(window_path, index=False)
    cache_path = tmp_path / "pose.jsonl"
    cache_path.write_text(
        json.dumps({"set": "jaad", "video": "video_0001", "pid": "0_1_2", "frame_idx": 1}) + "\n"
    )

    manifest, coverage = pose_support.build_missing_manifest(window_path, [cache_path])

    actual = set(map(tuple, manifest[["video_id", "pid", "frame_idx"]].itertuples(index=False, name=None)))
    assert actual == {
        ("video_0001", "0_1_2", 0),
        ("video_0001", "0_1_2", 2),
        ("video_0001", "0_1_2", 3),
        ("video_0001", "0_1_2b", 0),
        ("video_0001", "0_1_2b", 1),
        ("video_0001", "0_1_2b", 2),
    }
    assert coverage["all"]["required_unique_frames"] == 7
    assert coverage["all"]["cached_unique_frames"] == 1
    assert coverage["all"]["missing_unique_frames"] == 6
    assert coverage["train"]["fully_covered_windows"] == 0
    assert coverage["test"]["fully_covered_windows"] == 0
    row = manifest.loc[
        manifest[["video_id", "pid", "frame_idx"]].eq(["video_0001", "0_1_2", 2]).all(axis=1)
    ].iloc[0]
    assert json.loads(row["bbox_xyxy_json"]) == [2.0, 1.0, 12.0, 21.0]


def test_conflicting_window_bboxes_are_rejected(tmp_path):
    windows = pd.DataFrame(
        [
            _window("s1", "train", "0_1_2", [0], [[0.0, 0.0, 10.0, 20.0]]),
            _window("s2", "train", "0_1_2", [0], [[1.0, 0.0, 10.0, 20.0]]),
        ]
    )
    window_path = tmp_path / "windows.csv"
    windows.to_csv(window_path, index=False)
    cache_path = tmp_path / "pose.jsonl"
    cache_path.write_text("")

    with pytest.raises(ValueError, match="conflicting frozen BBoxes"):
        pose_support.build_missing_manifest(window_path, [cache_path])


def test_cache_conflict_is_rejected(tmp_path):
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    base = {
        "set": "jaad",
        "video": "video_0001",
        "pid": "0_1_2b",
        "frame_idx": 4,
        "keypoints": [[1.0, 2.0]] * 17,
        "kp_conf": [0.9] * 17,
    }
    first.write_text(json.dumps(base) + "\n")
    base["keypoints"][0] = [9.0, 9.0]
    second.write_text(json.dumps(base) + "\n")

    with pytest.raises(ValueError, match="conflicting pose records"):
        pose_support.load_pose_cache_keys([first, second], verify_payload=True)


def test_timing_summary_excludes_warmup_and_reports_16_frame_estimate():
    events = pd.DataFrame(
        {
            "warmup": [True, False, False, False],
            "inference_ms": [1000.0, 10.0, 20.0, 30.0],
            "decode_ms": [50.0, 1.0, 2.0, 3.0],
            "write_ms": [5.0, 0.1, 0.2, 0.3],
            "pedestrian_count": [1, 1, 2, 1],
        }
    )

    result = pose_support.summarize_timing_events(events, total_elapsed_seconds=2.0, output_records=4)

    assert result["steady_state_source_frames"] == 3
    assert result["inference_ms"]["mean"] == pytest.approx(20.0)
    assert result["inference_ms"]["median"] == pytest.approx(20.0)
    assert result["estimated_sequential_16_frame_inference_ms_from_median"] == pytest.approx(320.0)
    assert result["end_to_end_records_per_second"] == pytest.approx(2.0)


def test_extraction_requests_use_manifest_bbox_for_non_behavior_track(tmp_path):
    image_dir = tmp_path / "images" / "video_0001"
    image_dir.mkdir(parents=True)
    (image_dir / "00522.png").write_bytes(b"image-placeholder")
    manifest = pd.DataFrame(
        [
            {
                "video_id": "video_0001",
                "pid": "0_1_2",
                "frame_idx": 522,
                "bbox_xyxy_json": "[10.0,20.0,30.0,40.0]",
            }
        ]
    )
    manifest_path = tmp_path / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    requests = pose_support.load_extraction_requests(manifest_path, tmp_path)

    assert requests == [
        {
            "set": "jaad",
            "video": "video_0001",
            "pid": "0_1_2",
            "frame_idx": 522,
            "bbox_xyxy": [10.0, 20.0, 30.0, 40.0],
            "image_path": str(image_dir / "00522.png"),
        }
    ]


def test_pose_output_validation_requires_exact_manifest_bbox(tmp_path):
    manifest = pd.DataFrame(
        [{"video_id": "video_0001", "pid": "0_1_2", "frame_idx": 3, "bbox_xyxy_json": "[1,2,3,4]"}]
    )
    manifest_path = tmp_path / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    record = {
        "set": "jaad",
        "video": "video_0001",
        "pid": "0_1_2",
        "frame_idx": 3,
        "bbox_xyxy": [1, 2, 3, 4],
        "keypoints": [[5.0, 6.0]] * 17,
        "kp_conf": [0.8] * 17,
    }
    output_path = tmp_path / "pose.jsonl"
    output_path.write_text(json.dumps(record) + "\n")

    summary = pose_support.validate_pose_output(output_path, manifest_path)
    assert summary["records"] == 1
    assert summary["exact_manifest_coverage"] is True

    record["bbox_xyxy"] = [9, 2, 3, 4]
    output_path.write_text(json.dumps(record) + "\n")
    with pytest.raises(ValueError, match="BBox mismatch"):
        pose_support.validate_pose_output(output_path, manifest_path)
