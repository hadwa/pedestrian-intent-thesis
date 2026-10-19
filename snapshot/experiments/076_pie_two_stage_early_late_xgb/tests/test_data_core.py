from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts/data_core.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("exp076_data_core", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _record(frame: int, offset: float = 0.0) -> dict:
    return {
        "set": "jaad",
        "video": "video_0001",
        "pid": "p1",
        "frame_idx": frame,
        "keypoints": [[offset + index, offset + index + 1] for index in range(17)],
        "kp_conf": [0.9] * 17,
    }


def test_keypoint_sources_fill_only_missing_exact_frames(tmp_path: Path) -> None:
    data = _load_module()
    primary = tmp_path / "primary.jsonl"
    supplement = tmp_path / "supplement.jsonl"
    primary.write_text(json.dumps(_record(1)) + "\n", encoding="utf-8")
    supplement.write_text(json.dumps(_record(2, 10.0)) + "\n", encoding="utf-8")
    needed = {("jaad", "video_0001", "p1", 1), ("jaad", "video_0001", "p1", 2)}

    index, audit = data.load_keypoint_index((primary, supplement), needed)

    assert set(index) == needed
    assert audit["required_frames"] == 2
    assert audit["missing_frames"] == 0
    assert audit["source_new_frames"][str(primary.resolve())] == 1
    assert audit["source_new_frames"][str(supplement.resolve())] == 1


def test_keypoint_source_conflict_fails_closed(tmp_path: Path) -> None:
    data = _load_module()
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    first.write_text(json.dumps(_record(1)) + "\n", encoding="utf-8")
    second.write_text(json.dumps(_record(1, 100.0)) + "\n", encoding="utf-8")
    needed = {("jaad", "video_0001", "p1", 1)}

    with pytest.raises(ValueError, match="conflicting keypoint"):
        data.load_keypoint_index((first, second), needed)


def test_validate_window_cohort_requires_exact_16_frames_and_tte() -> None:
    data = _load_module()
    frame = pd.DataFrame(
        {
            "sequence_id": ["s1"],
            "track_id": ["t1"],
            "label": [1],
            "start_frame": [10],
            "end_frame": [25],
            "observation_length": [16],
            "frame_step": [1],
            "frame_sampling": ["consecutive"],
            "time_to_event_frames": [45],
        }
    )
    data.validate_window_cohort(frame, tte_column="time_to_event_frames")

    changed = frame.copy()
    changed.loc[0, "end_frame"] = 24
    with pytest.raises(ValueError, match="16-frame"):
        data.validate_window_cohort(changed, tte_column="time_to_event_frames")


def test_materialize_views_preserves_rows_and_builds_early_and_portable_features() -> None:
    data = _load_module()
    definitions = ROOT.parents[0] / "012_pie_biu_v2_30_60_benchmark/metrics/pie_biu_v2_feature_set_definitions.csv"
    contract = data.two_stage_core.load_joint_contract(definitions)
    row = pd.DataFrame(
        {
            "sequence_id": ["s1"],
            "set_id": ["jaad"],
            "video_id": ["video_0001"],
            "ped_id": ["p1"],
            "track_id": ["t1"],
            "label": [1],
            "start_frame": [0],
            "end_frame": [15],
            "observation_length": [16],
            "frame_step": [1],
            "frame_sampling": ["consecutive"],
            "tte_frames": [45],
        }
    )
    base = np.asarray(
        [[50, 15], [47, 13], [53, 13], [44, 14], [56, 14], [42, 30], [58, 30],
         [38, 48], [62, 48], [36, 68], [64, 68], [45, 62], [55, 62], [44, 88],
         [57, 88], [42, 116], [60, 116]],
        dtype=float,
    )
    index = {}
    boxes = {}
    for frame in range(16):
        key = ("jaad", "video_0001", "p1", frame)
        index[key] = {"keypoints": base + [frame, 0], "kp_conf": np.full(17, 0.9)}
        boxes[key] = np.asarray([20 + frame, 5, 85 + frame, 130], dtype=float)

    views, audit = data.materialize_views(row, index, boxes, contract=contract)

    assert views["sequence_id"].tolist() == ["s1"]
    assert len([name for name in views if name.startswith("early__")]) == 161
    assert len([name for name in views if name.startswith("portable__")]) == 182
    assert audit["rows"] == 1
    assert audit["missing_windows"] == 0
