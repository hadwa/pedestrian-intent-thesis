from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _module():
    return importlib.import_module(
        "experiments.063_pie_corrected_yolop_road_relative_context.scripts.extract_trainval"
    )


def _rows(set_id: str = "set01") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ("sequence_1", "sequence_2"),
            "split": ("train", "train"),
            "set_id": (set_id, set_id),
            "video_id": ("video_0001", "video_0001"),
            "ped_id": ("1_1_1", "1_1_1"),
            "start_frame": (1, 2),
            "end_frame": (16, 17),
            "observation_length": (16, 16),
            "frame_step": (1, 1),
            "frame_sampling": ("consecutive", "consecutive"),
        }
    )


def test_required_records_deduplicate_overlapping_observation_frames() -> None:
    extraction = _module()
    required = extraction.required_bbox_records(_rows())
    video = required[("set01", "video_0001")]
    assert len(video) == 17
    assert sum(len(pedestrians) for pedestrians in video.values()) == 17
    assert video[1] == ("1_1_1",)
    assert video[17] == ("1_1_1",)


def test_development_rows_reject_set03_before_frame_or_model_access() -> None:
    extraction = _module()
    with pytest.raises(ValueError, match="set03|forbids cohort"):
        extraction.validate_development_rows(_rows("set03"), strict_counts=False)


def test_assemble_video_windows_resets_temporal_features_per_window() -> None:
    extraction = _module()
    rows = _rows()
    records = {}
    for frame in range(1, 18):
        grid = np.full(20, frame / 100.0, dtype=np.float32)
        static = np.zeros(13, dtype=np.float32)
        static[0] = 0.0
        static[1] = -frame / 100.0
        static[2] = 0.5
        static[5] = 1.0
        static[6] = 1.0
        bbox = np.asarray([frame, 30, frame + 10, 50], dtype=np.float32)
        records[("1_1_1", frame)] = extraction.TargetFrameRecord(grid, static, bbox)

    sequence_ids, corrected, road = extraction.assemble_video_windows(rows, records)
    assert tuple(sequence_ids) == ("sequence_1", "sequence_2")
    assert corrected.shape == (2, 16, 20)
    assert road.shape == (2, 16, 20)
    np.testing.assert_allclose(road[:, 0, 13:20], 0.0)
    assert road[0, 1, 13] > 0
    assert road[1, 1, 13] > 0
    assert corrected[0, 0, 0] == pytest.approx(0.01)
    assert corrected[1, 0, 0] == pytest.approx(0.02)


def test_atomic_part_round_trip_and_resume_validation(tmp_path: Path) -> None:
    extraction = _module()
    destination = tmp_path / "set01_video_0001.npz"
    ids = np.asarray(("s1", "s2"))
    corrected = np.zeros((2, 16, 20), dtype=np.float32)
    road = np.ones((2, 16, 20), dtype=np.float32)
    extraction.write_video_part(destination, ids, corrected, road)
    loaded = extraction.validate_existing_part(destination, ids)
    assert loaded["rows"] == 2
    assert loaded["resumed"] is True

    with pytest.raises(ValueError, match="sequence IDs"):
        extraction.validate_existing_part(destination, np.asarray(("s2", "s1")))
