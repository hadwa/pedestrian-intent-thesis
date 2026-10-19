from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "feature_contract.py"
RUNNER_PATH = ROOT / "scripts" / "run_experiment.py"


def load_module():
    spec = importlib.util.spec_from_file_location("exp096_feature_contract", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_runner():
    spec = importlib.util.spec_from_file_location("exp096_runner", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_matched_arms_preserve_exp072_combined_contract_and_isolate_versions():
    """Catches accidental v1/v2 leakage or drift from Exp072's 187-D candidate."""
    module = load_module()
    parent_contract = Path("experiments/037_pie_joint_biu_temporal_ego/metrics/feature_contract.json")
    arms = module.build_matched_arm_columns(
        Path("experiments/012_pie_biu_v2_30_60_benchmark/metrics/pie_biu_v2_feature_set_definitions.csv"),
        parent_contract,
    )

    assert {name: len(columns) for name, columns in arms.items()} == {
        "base": 64,
        "v1": 139,
        "v2": 135,
        "combined": 187,
    }
    assert len(arms["combined"]) == len(set(arms["combined"]))
    assert set(arms["base"]).issubset(arms["v1"])
    assert set(arms["base"]).issubset(arms["v2"])
    expected_combined = tuple(
        json.loads(parent_contract.read_text(encoding="utf-8"))["arms"]["joint_temporal_ego_xgb"]
    )
    assert arms["combined"] == expected_combined

    v1_only = set(arms["v1"]) - set(arms["base"])
    v2_only = set(arms["v2"]) - set(arms["base"])
    assert len(v1_only) == 75
    assert len(v2_only) == 71
    shared_pose_quality = {
        "temporal__pose_confidence_mean_early",
        "temporal__pose_confidence_mean_late",
        "temporal__pose_confidence_mean_late_minus_early",
    }
    assert v1_only.intersection(v2_only) == shared_pose_quality
    assert not any(name.startswith("v2__") for name in v1_only)
    assert not any("lower_body_motion" in name for name in v1_only)
    assert any(name.startswith("v2__") for name in v2_only)
    assert "temporal__lower_body_motion_magnitude_mean_early" in v2_only


def test_arm_builder_rejects_a_changed_exp072_combined_contract(tmp_path: Path):
    """Catches silently accepting a parent feature list with the wrong dimensionality."""
    module = load_module()
    definitions = Path(
        "experiments/012_pie_biu_v2_30_60_benchmark/metrics/pie_biu_v2_feature_set_definitions.csv"
    )
    changed = tmp_path / "changed_contract.json"
    changed.write_text('{"arms": {"joint_temporal_ego_xgb": ["bbox_area_mean"]}}\n', encoding="utf-8")

    try:
        module.build_matched_arm_columns(definitions, changed)
    except ValueError as error:
        assert "187" in str(error)
    else:
        raise AssertionError("changed Exp072 contract was accepted")


def test_validation_grid_requires_every_arm_and_seed():
    """Catches opening set03 after only a partial matched-control grid."""
    runner = load_runner()
    assert runner.required_runs() == [
        (arm, seed)
        for arm in ("base", "v1", "v2", "combined")
        for seed in (42, 43, 44)
    ]
    incomplete = pd.DataFrame(
        [{"arm": "base", "seed": 42, "threshold": 0.5, "f1": 0.7, "roc_auc": 0.8, "pr_auc": 0.75}]
    )
    with pytest.raises(RuntimeError, match="complete 12-run validation grid"):
        runner.validation_gate_payload(incomplete)


def test_set03_gate_rejects_missing_or_tampered_validation_completion(tmp_path: Path):
    """Catches test loading without an authenticated completed validation phase."""
    runner = load_runner()
    with pytest.raises(RuntimeError, match="authenticated validation completion"):
        runner.authenticate_test_gate(tmp_path)

    metrics = tmp_path / "metrics"
    metrics.mkdir()
    gate = metrics / "validation_gate.json"
    gate.write_text('{"status":"advance"}\n', encoding="utf-8")
    (metrics / ".validate.complete.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "outputs": {"validation_gate.json": {"sha256": "wrong", "size_bytes": 21}},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="authenticated validation completion"):
        runner.authenticate_test_gate(tmp_path)


def test_completion_manifest_authenticates_outputs_in_nested_directories(tmp_path: Path):
    """Catches flattening model/prediction paths into the metrics directory."""
    runner = load_runner()
    metric = tmp_path / "metrics" / "result.csv"
    prediction = tmp_path / "predictions" / "prediction.csv"
    model = tmp_path / "models" / "model.joblib"
    for path, value in ((metric, "metric"), (prediction, "prediction"), (model, "model")):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value, encoding="utf-8")
    runner._publish_completion(tmp_path, "nested", [metric, prediction, model])
    payload = runner._authenticate_completion(tmp_path, "nested")
    assert set(payload["outputs"]) == {
        "metrics/result.csv",
        "predictions/prediction.csv",
        "models/model.joblib",
    }


def test_markdown_renderer_does_not_require_optional_tabulate(monkeypatch):
    """Catches a report-only failure when pandas' optional tabulate is absent."""
    runner = load_runner()

    def fail_if_called(*args, **kwargs):
        raise ImportError("tabulate is intentionally unavailable")

    monkeypatch.setattr(pd.DataFrame, "to_markdown", fail_if_called)
    rendered = runner._markdown(pd.DataFrame({"arm": ["v2"], "f1": [0.81234567]}))

    assert "| arm" in rendered
    assert "| v2" in rendered
    assert "0.812346" in rendered


def test_supplemental_bootstrap_covers_every_biu_arm_against_base():
    """Catches omitting the uncertainty analysis needed for the BIU-vs-base claim."""
    runner = load_runner()
    assert runner.BASE_BOOTSTRAP_COMPARISONS == (
        ("base", "v1"),
        ("base", "v2"),
        ("base", "combined"),
    )
