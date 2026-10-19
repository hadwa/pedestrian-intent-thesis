from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts" / "run_experiment.py"


def load_runner(name: str):
    spec = importlib.util.spec_from_file_location(name, RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_candidate_registry_has_nine_anchor_candidates_and_four_context_modes() -> None:
    runner = load_runner("exp055_test_runner_registry")

    stage1 = runner.stage1_candidates()

    assert len(stage1) == 9
    assert len({candidate.candidate_id for candidate in stage1}) == 9
    assert {candidate.biu_mode for candidate in stage1} == {"v1", "v2_core", "joint_unique"}
    assert {candidate.bbox_temporal_mode for candidate in stage1} == {
        "summary_only",
        "raw4",
        "motion8",
    }
    assert {candidate.context_mode for candidate in stage1} == {"none"}
    assert runner.stage2_candidates(stage1[0]) == (
        stage1[0],
        stage1[0].with_context("ego"),
        stage1[0].with_context("yolop"),
        stage1[0].with_context("ego_yolop"),
    )


def test_aggregate_selection_uses_score_then_pr_auc_then_f1_then_size() -> None:
    runner = load_runner("exp055_test_runner_selection")
    rows = pd.DataFrame(
        [
            {"candidate_id": "a", "seed": seed, "selection_score": 0.80, "pr_auc": 0.82, "f1": 0.78, "total_parameters": 90_000}
            for seed in (42, 43, 44)
        ]
        + [
            {"candidate_id": "b", "seed": seed, "selection_score": 0.80, "pr_auc": 0.83, "f1": 0.77, "total_parameters": 80_000}
            for seed in (42, 43, 44)
        ]
    )

    selected, summary = runner.select_validation_candidate(rows)

    assert selected == "b"
    assert set(summary["candidate_id"]) == {"a", "b"}


def test_final_evaluation_gate_rejects_missing_freeze_marker(tmp_path: Path) -> None:
    runner = load_runner("exp055_test_runner_missing_freeze")

    with pytest.raises(RuntimeError, match="frozen final candidate"):
        runner.authenticate_final_candidate(tmp_path)


def test_final_evaluation_gate_rejects_incomplete_seed_grid(tmp_path: Path) -> None:
    runner = load_runner("exp055_test_runner_incomplete_grid")
    candidate = runner.stage1_candidates()[0]
    marker = {
        "candidate": candidate.to_dict(),
        "candidate_id": candidate.candidate_id,
        "seeds": [42, 43, 44],
        "checkpoint_artifacts": {},
        "validation_results_sha256": "not-a-real-hash",
    }
    (tmp_path / "FROZEN_FINAL_CANDIDATE.json").write_text(json.dumps(marker), encoding="utf-8")

    with pytest.raises(RuntimeError, match="three authenticated checkpoints"):
        runner.authenticate_final_candidate(tmp_path)


def test_preparation_contract_rejects_any_test_access(tmp_path: Path) -> None:
    runner = load_runner("exp055_test_runner_prepare_guard")
    marker = tmp_path / ".prepare.complete.json"
    marker.write_text(
        json.dumps({"test_accessed": True, "train_rows": 2366, "val_rows": 3619}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="set03 exclusion"):
        runner.validate_prepare_contract(marker)


def test_trainval_row_guard_rejects_set03_or_nonconsecutive_windows() -> None:
    runner = load_runner("exp055_test_runner_row_guard")
    valid = pd.DataFrame(
        {
            "sequence_id": ["a", "b"],
            "split": ["train", "val"],
            "set_id": ["set01", "set04"],
            "start_frame": [10, 20],
            "end_frame": [25, 35],
            "event_frame": [70, 80],
            "time_to_event_frames": [45, 45],
            "observation_length": [16, 16],
            "frame_step": [1, 1],
            "frame_sampling": ["consecutive", "consecutive"],
            "label": [0, 1],
        }
    )
    checked = runner.validate_trainval_rows(valid)
    assert len(checked) == 2

    leaked = valid.copy()
    leaked.loc[1, "set_id"] = "set03"
    with pytest.raises(RuntimeError, match="set03"):
        runner.validate_trainval_rows(leaked)

    broken = valid.copy()
    broken.loc[0, "end_frame"] = 26
    with pytest.raises(RuntimeError, match="16 consecutive"):
        runner.validate_trainval_rows(broken)

    swapped = valid.copy()
    swapped.loc[0, "set_id"] = "set04"
    with pytest.raises(RuntimeError, match="set membership"):
        runner.validate_trainval_rows(swapped)

    wrong_tte = valid.copy()
    wrong_tte.loc[0, "time_to_event_frames"] = 44
    with pytest.raises(RuntimeError, match="TTE identity"):
        runner.validate_trainval_rows(wrong_tte)


def test_candidate_training_gate_requires_authenticated_preparation(tmp_path: Path) -> None:
    runner = load_runner("exp055_test_runner_train_gate")
    with pytest.raises(RuntimeError, match="preparation"):
        runner.assert_training_allowed(tmp_path)


def test_freeze_payload_requires_same_candidate_for_all_three_seeds(tmp_path: Path) -> None:
    runner = load_runner("exp055_test_runner_freeze_payload")
    selected = runner.stage1_candidates()[0]
    rows = pd.DataFrame(
        [
            {
                "candidate_id": selected.candidate_id,
                "seed": seed,
                "selection_score": 0.8,
                "pr_auc": 0.82,
                "f1": 0.78,
                "total_parameters": 80_000,
            }
            for seed in (42, 43, 44)
        ]
    )
    checkpoints = {
        seed: tmp_path / f"{selected.candidate_id}__seed{seed}.best.pt"
        for seed in (42, 43, 44)
    }
    for path in checkpoints.values():
        path.write_bytes(b"checkpoint")

    payload = runner.build_freeze_payload(selected, rows, checkpoints, tmp_path)

    assert payload["candidate_id"] == selected.candidate_id
    assert payload["seeds"] == [42, 43, 44]
    assert len(payload["checkpoint_artifacts"]) == 3


def test_real_prepare_builds_only_aligned_trainval_features(tmp_path: Path) -> None:
    runner = load_runner("exp055_test_runner_real_prepare")

    runner.run_prepare(root=tmp_path)
    payload = runner.validate_prepare_contract(tmp_path / ".prepare.complete.json")
    rows, arrays, metadata = runner.load_prepared_cache(tmp_path / "cache/trainval")

    assert payload["test_accessed"] is False
    assert len(rows.loc[rows["split"].eq("train")]) == 2366
    assert len(rows.loc[rows["split"].eq("val")]) == 3619
    assert not rows["set_id"].astype(str).eq("set03").any()
    assert arrays["summary_joint_unique"].shape == (5985, 161)
    assert arrays["summary_v1"].shape == (5985, 125)
    assert arrays["summary_v2_core"].shape == (5985, 109)
    assert arrays["bbox_motion8"].shape == (5985, 16, 8)
    assert arrays["ego"].shape == (5985, 16, 3)
    assert arrays["yolop"].shape == (5985, 16, 20)
    assert metadata["test_accessed"] is False
    assert not (tmp_path / "cache/test").exists()


def test_two_epoch_cpu_training_smoke_writes_validation_only_artifacts(tmp_path: Path) -> None:
    runner = load_runner("exp055_test_runner_cpu_training")
    runner.run_prepare(root=tmp_path)
    candidate = runner.stage1_candidates()[0]

    marker = runner.run_training(
        candidate,
        42,
        root=tmp_path,
        device=torch.device("cpu"),
        max_epochs=2,
        early_stopping_patience=2,
    )

    assert marker["candidate_id"] == candidate.candidate_id
    assert marker["seed"] == 42
    assert marker["epochs_completed"] == 2
    prediction = pd.read_csv(
        tmp_path / f"predictions/validation__{candidate.candidate_id}__seed42.csv"
    )
    assert len(prediction) == 3619
    assert {"probability", "anchor_probability", "gate", "bounded_delta"}.issubset(
        prediction.columns
    )
    assert not prediction["set_id"].astype(str).eq("set03").any()
    assert not (tmp_path / "cache/test").exists()

    with pytest.raises(RuntimeError, match="training contract drifted"):
        runner.run_training(
            candidate,
            42,
            root=tmp_path,
            device=torch.device("cpu"),
            max_epochs=3,
            early_stopping_patience=2,
        )


def test_stage1_selection_rejects_an_incomplete_validation_grid(tmp_path: Path) -> None:
    runner = load_runner("exp055_test_runner_stage1_gate")
    runner.run_prepare(root=tmp_path)

    with pytest.raises(RuntimeError, match="27 authenticated validation runs"):
        runner.freeze_selected_anchor(root=tmp_path)


def test_final_evaluation_cannot_construct_test_cache_before_freeze(tmp_path: Path) -> None:
    runner = load_runner("exp055_test_runner_final_gate")
    runner.run_prepare(root=tmp_path)

    with pytest.raises(RuntimeError, match="frozen final candidate"):
        runner.run_final_evaluation(root=tmp_path, device=torch.device("cpu"))

    assert not (tmp_path / "cache/test").exists()


def test_test_source_alignment_requires_matching_exp050_and_exp053_sequence_hashes() -> None:
    runner = load_runner("exp055_test_runner_test_alignment")
    source = {"n_rows": 3760, "sequence_id_sha256": "same"}
    exp050 = {"test_rows": 3760, "test_sequence_id_sha256": "same"}
    exp053 = {"test_rows": 3760, "test_sequence_id_sha256": "different"}

    with pytest.raises(RuntimeError, match="test cohort alignment"):
        runner.validate_test_source_alignment(source, exp050, exp053, bbox_rows=3760)

    exp053["test_sequence_id_sha256"] = "same"
    runner.validate_test_source_alignment(source, exp050, exp053, bbox_rows=3760)


def test_augmented_rows_must_reference_matching_original_training_samples() -> None:
    runner = load_runner("exp055_test_runner_augmented_alignment")
    base = pd.DataFrame(
        {
            "sequence_id": ["train_a", "train_b", "val_c"],
            "split": ["train", "train", "val"],
            "label": [0, 1, 1],
        }
    )
    augmented = pd.DataFrame(
        {
            "source_cache_index": [0, 1, 1],
            "sequence_id": ["train_a", "train_b", "train_b"],
            "label": [0, 1, 1],
        }
    )
    runner.validate_augmented_alignment(augmented, base)

    corrupted = augmented.copy()
    corrupted.loc[2, "sequence_id"] = "wrong"
    with pytest.raises(RuntimeError, match="augmented sample alignment"):
        runner.validate_augmented_alignment(corrupted, base)


def test_final_selection_contract_rejects_a_nonwinning_candidate() -> None:
    runner = load_runner("exp055_test_runner_final_selection")
    anchor = runner.stage1_candidates()[0]
    candidates = runner.stage2_candidates(anchor)
    rows = []
    for index, candidate in enumerate(candidates):
        for seed in (42, 43, 44):
            rows.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "seed": seed,
                    "selection_score": 0.80 + index * 0.01,
                    "pr_auc": 0.82 + index * 0.01,
                    "f1": 0.78,
                    "total_parameters": 80_000 + index,
                }
            )
    results = pd.DataFrame(rows)

    with pytest.raises(RuntimeError, match="validation winner"):
        runner.validate_final_selection_contract(
            {
                "phase": "freeze_final_candidate",
                "test_accessed": False,
                "seeds": [42, 43, 44],
                "candidate_id": candidates[0].candidate_id,
            },
            results,
            anchor,
        )

    runner.validate_final_selection_contract(
        {
            "phase": "freeze_final_candidate",
            "test_accessed": False,
            "seeds": [42, 43, 44],
            "candidate_id": candidates[-1].candidate_id,
        },
        results,
        anchor,
    )


def test_grid_contract_guard_rejects_mixed_runner_or_hyperparameters() -> None:
    runner = load_runner("exp055_test_runner_grid_contract")
    base = {
        "candidate": {"biu_mode": "v1"},
        "seed": 42,
        "prepare_sha256": "prepare",
        "core_sha256": "core",
        "runner_sha256": "runner",
        "device_type": "cuda",
        "optimizer": "AdamW",
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "batch_size": 64,
        "max_epochs": 60,
        "early_stopping_patience": 10,
        "gradient_clip_norm": 5.0,
        "anchor_loss_weight": 0.3,
        "residual_penalty_weight": 0.001,
        "prepare_test_accessed": False,
    }
    compatible = dict(base, candidate={"biu_mode": "v2_core"}, seed=43)
    runner.validate_grid_training_contracts([base, compatible])

    incompatible = dict(compatible, learning_rate=0.002)
    with pytest.raises(RuntimeError, match="mixed training contracts"):
        runner.validate_grid_training_contracts([base, incompatible])
