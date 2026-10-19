from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
CORE_PATH = ROOT / "scripts" / "exp050_core.py"
RUNNER_PATH = ROOT / "scripts" / "run_experiment.py"


def load_core():
    spec = importlib.util.spec_from_file_location("exp050_test_core", CORE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_runner():
    spec = importlib.util.spec_from_file_location("exp050_test_runner", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def example_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sequence_id": "train_a",
                "split": "train",
                "set_id": "set01",
                "video_id": "video_0001",
                "ped_id": "1_1_1",
                "start_frame": 10,
                "end_frame": 25,
                "last_observed_frame": 25,
                "event_frame": 60,
                "time_to_event_frames": 35,
                "observation_length": 16,
                "frame_step": 1,
                "frame_sampling": "consecutive",
                "label": 0,
            },
            {
                "sequence_id": "train_b",
                "split": "train",
                "set_id": "set02",
                "video_id": "video_0001",
                "ped_id": "2_1_1",
                "start_frame": 20,
                "end_frame": 35,
                "last_observed_frame": 35,
                "event_frame": 90,
                "time_to_event_frames": 55,
                "observation_length": 16,
                "frame_step": 1,
                "frame_sampling": "consecutive",
                "label": 1,
            },
            {
                "sequence_id": "train_c",
                "split": "train",
                "set_id": "set06",
                "video_id": "video_0001",
                "ped_id": "6_1_1",
                "start_frame": 30,
                "end_frame": 45,
                "last_observed_frame": 45,
                "event_frame": 105,
                "time_to_event_frames": 60,
                "observation_length": 16,
                "frame_step": 1,
                "frame_sampling": "consecutive",
                "label": 0,
            },
        ]
    )


def test_frozen_protocol_and_streams() -> None:
    core = load_core()
    assert core.SEQ_LEN == 16
    assert core.SEEDS == (42, 43, 44)
    assert core.VARIANTS == ("A", "B", "C", "D", "E", "F")
    assert core.VARIANT_STREAMS["A"] == (
        "bbox",
        "ped_action",
        "ped_look",
        "ego_speed",
        "ego_acc",
        "static_context",
        "yolop",
    )
    assert core.VARIANT_STREAMS["C"] == ("bbox", "ego_speed", "ego_acc", "yolop")
    assert core.VARIANT_STREAMS["D"][-1] == "pose"
    assert core.VARIANT_STREAMS["E"][-1] == "biu_v2"
    assert core.VARIANT_STREAMS["F"][-1] == "biu_v1"


def test_split_validation_is_observation_only() -> None:
    core = load_core()
    rows = core.validate_split_rows(example_rows(), allowed_splits={"train"})
    assert core.observed_frames(rows.iloc[0]) == list(range(10, 26))
    assert (rows["end_frame"] < rows["event_frame"]).all()

    invalid = example_rows()
    invalid.loc[0, "end_frame"] = 26
    with pytest.raises(ValueError, match="16 consecutive"):
        core.validate_split_rows(invalid, allowed_splits={"train"})

    leaked = example_rows()
    leaked.loc[0, "event_frame"] = 25
    with pytest.raises(ValueError, match="future event"):
        core.validate_split_rows(leaked, allowed_splits={"train"})


def test_oversampling_is_deterministic_and_balanced() -> None:
    core = load_core()
    rows = example_rows()
    first = core.build_oversampling_manifest(rows, seed=42)
    second = core.build_oversampling_manifest(rows, seed=42)
    pd.testing.assert_frame_equal(first, second)
    assert first["label"].value_counts().to_dict() == {0: 2, 1: 2}
    assert first["augmented"].sum() == 1
    assert first.loc[first["augmented"], "noise_seed"].ge(0).all()


def synthetic_measurements() -> dict[str, np.ndarray]:
    t = np.linspace(0.0, 1.0, 16, dtype=np.float32)
    pose = np.zeros((16, 34), dtype=np.float32)
    for joint in range(17):
        pose[:, 2 * joint] = 0.25 + 0.01 * joint + 0.02 * t
        pose[:, 2 * joint + 1] = 0.20 + 0.015 * joint + 0.01 * t
    bbox = np.column_stack((0.5 + 0.01 * t, 0.5 + 0.01 * t, np.full(16, 0.2), np.full(16, 0.4))).astype(np.float32)
    return {
        "bbox": bbox,
        "ego_speed": np.ones((16, 1), dtype=np.float32),
        "ego_acc": np.zeros((16, 2), dtype=np.float32),
        "yolop": np.full((16, 20), 0.5, dtype=np.float32),
        "pose": pose,
    }


def test_precomputed_augmented_biu_matches_direct_recomputation() -> None:
    core = load_core()
    original = synthetic_measurements()
    confidence = np.ones((16, 17), dtype=np.float32)
    base_v1 = core.extract_biu_v1_window(
        core.normalized_pose_to_pixels(original["pose"], width=1920, height=1080), confidence
    )[0]
    cached = core.precompute_augmented_row(
        original,
        width=1920,
        height=1080,
        noise_seed=123456,
        base_biu_v1=base_v1,
    )
    direct_measurements = core.augment_measurements(original, noise_seed=123456)
    direct_v2 = core.extract_biu_v2_window(
        direct_measurements["pose"], direct_measurements["bbox"], width=1920, height=1080
    )[0]
    direct_v1 = core.recompute_augmented_biu_v1(
        direct_measurements["pose"], width=1920, height=1080, original_biu_v1=base_v1
    )
    np.testing.assert_array_equal(cached["biu_v2"], direct_v2)
    np.testing.assert_array_equal(cached["biu_v1"], direct_v1)


def test_model_projection_dimensions_and_forward_shapes() -> None:
    core = load_core()
    for variant in core.VARIANTS:
        model = core.MslstmPracticalModel(variant)
        inputs = {}
        for stream in core.VARIANT_STREAMS[variant]:
            dim = core.STREAM_DIMS[stream]
            shape = (2, dim) if stream.startswith("biu_") else (2, 16, dim)
            inputs[stream] = torch.zeros(shape)
        assert model(inputs).shape == (2, 2)
    assert sum(p.numel() for p in core.MslstmPracticalModel("E").biu_projection.parameters()) == 74240
    assert sum(p.numel() for p in core.MslstmPracticalModel("F").biu_projection.parameters()) == 75392


def test_prepare_selection_excludes_test_rows() -> None:
    runner = load_runner()
    rows = pd.concat(
        [
            example_rows(),
            example_rows().assign(
                sequence_id=lambda frame: "val_" + frame["sequence_id"],
                split="val",
                set_id="set04",
            ),
            example_rows().assign(
                sequence_id=lambda frame: "test_" + frame["sequence_id"],
                split="test",
                set_id="set03",
            ),
        ],
        ignore_index=True,
    )
    prepared = runner.select_phase_rows(rows, phase="prepare")
    assert set(prepared["split"]) == {"train", "val"}
    assert not prepared["set_id"].eq("set03").any()
    evaluated = runner.select_phase_rows(rows, phase="evaluate")
    assert set(evaluated["split"]) == {"test"}


def test_final_evaluation_requires_all_18_training_markers(tmp_path: Path) -> None:
    runner = load_runner()
    assert len(runner.required_training_identities()) == 18
    with pytest.raises(RuntimeError, match="18 authenticated checkpoints"):
        runner.assert_training_complete(tmp_path)


def test_core_resolves_repo_imports_from_absolute_runner_context(tmp_path: Path) -> None:
    code = (
        "import importlib.util,sys; "
        f"p={str(CORE_PATH)!r}; "
        "s=importlib.util.spec_from_file_location('isolated_exp050_core',p); "
        "m=importlib.util.module_from_spec(s); sys.modules[s.name]=m; "
        "s.loader.exec_module(m); m._load_biu_dependencies()"
    )
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_report_markdown_does_not_require_optional_tabulate() -> None:
    runner = load_runner()
    frame = pd.DataFrame([{"variant": "A", "f1_mean": 0.75}])
    assert runner.frame_to_markdown(frame) == (
        "| variant | f1_mean |\n"
        "|---|---|\n"
        "| A | 0.75 |"
    )
