"""Immutable source-isolation tests for the Exp062 staging boundary."""

from __future__ import annotations

import sys
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.contracts import ExperimentConfig  # noqa: E402
from scripts.data import TrainValData  # noqa: E402
from scripts.source_staging import (  # noqa: E402
    StageBuildResult,
    load_staged_trainval,
    read_staged_validation_reference,
    stage_sources,
    verify_source_staging,
)
from scripts import runner as runner_module  # noqa: E402


def _config() -> ExperimentConfig:
    return replace(ExperimentConfig(), train_rows=2, val_rows=2)


def _data(config: ExperimentConfig) -> TrainValData:
    rows = pd.DataFrame(
        {
            "sequence_id": ["t0", "t1", "v0", "v1"],
            "split": ["train", "train", "val", "val"],
            "set_id": ["set01", "set02", "set04", "set05"],
            "track_id": ["a", "b", "c", "d"],
            "start_frame": [0, 1, 2, 3],
            "end_frame": [15, 16, 17, 18],
            "event_frame": [45, 46, 47, 48],
            "time_to_event_frames": [30, 30, 30, 30],
            "observation_length": [16] * 4,
            "frame_step": [1] * 4,
            "frame_sampling": ["consecutive"] * 4,
            "label": [0, 1, 0, 1],
            "cache_index": range(4),
        }
    )
    train_mask = rows["split"].eq("train").to_numpy()
    return TrainValData(
        rows=rows,
        summary_v1=np.zeros((4, config.rf_feature_dim)),
        summary_v2_core=np.ones((4, config.xgb_feature_dim)),
        ego=np.zeros((4, *config.ego_shape), dtype=np.float32),
        yolop=np.zeros((4, *config.yolop_shape), dtype=np.float32),
        labels=rows["label"].to_numpy(),
        train_mask=train_mask,
        val_mask=~train_mask,
    )


def _build_result(tmp_path: Path, config: ExperimentConfig) -> StageBuildResult:
    mixed = tmp_path / "mixed-source.jsonl"
    mixed.write_text("mixed historical container\n", encoding="utf-8")
    mixed_predictions = tmp_path / "mixed-predictions.csv"
    mixed_predictions.write_text("mixed historical predictions\n", encoding="utf-8")
    validation = pd.DataFrame(
        {"sequence_id": ["v0", "v1"], "split": ["val", "val"], "label": [0, 1]}
    )
    references = {
        (anchor, seed): validation.assign(probability=[0.2, 0.8])
        for anchor in config.anchor_families
        for seed in config.seeds
    }
    return StageBuildResult(
        data=_data(config),
        validation_references=references,
        original_sources={
            "mixed_keypoints": mixed,
            "mixed_predictions": mixed_predictions,
        },
        extraction_rules={
            "mixed_keypoints": "filter exact needed set01/02/04/05 keys",
            "historical_predictions": "read exactly first validation prefix; split=val",
        },
    )


def test_staging_is_immutable_and_downstream_loaders_open_only_isolated_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches any post-staging open/stat/hash of a mixed historical container."""
    config = _config()
    result = _build_result(tmp_path, config)
    manifest = stage_sources(tmp_path, config=config, build=lambda: result)
    assert manifest["schema"] == "exp062-source-staging-v1"
    assert manifest["extraction_rules"]["historical_predictions"].startswith("read exactly")

    mixed_paths = {
        os.path.abspath(os.fspath(evidence["path"]))
        for evidence in manifest["original_sources"].values()
    }
    original_open = Path.open
    original_stat = Path.stat

    def guarded_open(path: Path, *args, **kwargs):
        if os.path.abspath(os.fspath(path)) in mixed_paths:
            pytest.fail("post-staging mixed source open")
        return original_open(path, *args, **kwargs)

    def guarded_stat(path: Path, *args, **kwargs):
        if os.path.abspath(os.fspath(path)) in mixed_paths:
            pytest.fail("post-staging mixed source stat/hash")
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    monkeypatch.setattr(Path, "stat", guarded_stat)
    verified = verify_source_staging(tmp_path, config=config)
    loaded = load_staged_trainval(tmp_path, config=config)
    reference = read_staged_validation_reference(
        tmp_path, "rf_v1_bbox", 42, config=config
    )

    assert verified["derived_artifacts"] == manifest["derived_artifacts"]
    np.testing.assert_array_equal(loaded.labels, [0, 1, 0, 1])
    assert reference["sequence_id"].tolist() == ["v0", "v1"]
    with pytest.raises(FileExistsError, match="already exists"):
        stage_sources(tmp_path, config=config, build=lambda: result)


def test_staging_rejects_derived_mutation_without_reopening_original_source(
    tmp_path: Path,
) -> None:
    """Catches restaging or accepting a mutated isolated anchor matrix."""
    config = _config()
    stage_sources(tmp_path, config=config, build=lambda: _build_result(tmp_path, config))
    matrix = tmp_path / "staged_sources" / "summary_v1.npy"
    matrix.write_bytes(b"substituted")

    with pytest.raises(RuntimeError, match="derived artifact hash mismatch"):
        verify_source_staging(tmp_path, config=config)


def test_staging_requires_the_exact_isolated_artifact_set(tmp_path: Path) -> None:
    """Catches a staging marker that omits or substitutes an isolated source."""
    config = _config()
    stage_sources(tmp_path, config=config, build=lambda: _build_result(tmp_path, config))
    marker_path = tmp_path / "SOURCE_STAGING.json"
    marker = __import__("json").loads(marker_path.read_text(encoding="utf-8"))
    marker["derived_artifacts"].pop("validation_reference_rf_v1_bbox_seed42.csv")
    marker_path.write_text(__import__("json").dumps(marker), encoding="utf-8")

    with pytest.raises(RuntimeError, match="manifest is malformed"):
        verify_source_staging(tmp_path, config=config)


@pytest.mark.parametrize(
    ("phase", "collision"),
    [
        ("prepare", "metrics/context_scaler.json"),
        ("fit-anchors", "metrics/anchor_parity.csv"),
        ("train", "checkpoints/rf_v1_bbox_seed42.pt"),
        ("validate", "metrics/validation_per_seed_results.csv"),
        ("freeze", "FROZEN_FINAL_MODEL.json"),
    ],
)
def test_public_pre_freeze_commands_cannot_touch_mixed_sources_and_refuse_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
    collision: str,
) -> None:
    """Catches public phase paths bypassing staging or overwriting any output."""
    config = _config()
    build_result = _build_result(tmp_path, config)
    runner_module.stage_source_inputs(
        tmp_path, config=config, build=lambda: build_result
    )
    monkeypatch.setattr(runner_module, "ExperimentConfig", lambda: config)
    state = runner_module.PhaseState(tmp_path, config=config)
    marker_artifact = tmp_path / "marker-input.txt"
    marker_artifact.write_text("bound\n", encoding="utf-8")
    if phase in {"fit-anchors", "train", "validate", "freeze"}:
        state.write_phase_marker(
            "prepare",
            command="fixture prepare",
            artifacts={"source": marker_artifact},
        )
    if phase in {"train", "validate", "freeze"}:
        state.write_phase_marker(
            "fit-anchors",
            command="fixture fit",
            artifacts={"PREPARED.json": tmp_path / "PREPARED.json"},
        )
    if phase in {"validate", "freeze"}:
        state.write_phase_marker(
            "train",
            command="fixture train",
            artifacts={"ANCHORS_READY.json": tmp_path / "ANCHORS_READY.json"},
        )
    if phase == "freeze":
        state.write_phase_marker(
            "validate",
            command="fixture validate",
            artifacts={"TRAINING_COMPLETE.json": tmp_path / "TRAINING_COMPLETE.json"},
        )
    collision_path = tmp_path / collision
    collision_path.parent.mkdir(parents=True, exist_ok=True)
    collision_path.write_text("must not overwrite\n", encoding="utf-8")

    mixed_paths = {
        os.path.abspath(os.fspath(path))
        for path in build_result.original_sources.values()
    }
    original_open = Path.open
    original_stat = Path.stat

    def guarded_open(path: Path, *args, **kwargs):
        if os.path.abspath(os.fspath(path)) in mixed_paths:
            pytest.fail(f"{phase} opened mixed source")
        return original_open(path, *args, **kwargs)

    def guarded_stat(path: Path, *args, **kwargs):
        if os.path.abspath(os.fspath(path)) in mixed_paths:
            pytest.fail(f"{phase} stat/hashed mixed source")
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    monkeypatch.setattr(Path, "stat", guarded_stat)
    commands = {
        "prepare": lambda: runner_module.prepare(tmp_path),
        "fit-anchors": lambda: runner_module.fit_anchors(tmp_path),
        "train": lambda: runner_module.train(
            tmp_path, anchor_name="rf_v1_bbox", seed=42
        ),
        "validate": lambda: runner_module.validate(tmp_path),
        "freeze": lambda: runner_module.freeze(tmp_path),
    }
    with pytest.raises(runner_module.PhaseOrderError, match="refusing to overwrite"):
        commands[phase]()
    assert collision_path.read_text(encoding="utf-8") == "must not overwrite\n"
