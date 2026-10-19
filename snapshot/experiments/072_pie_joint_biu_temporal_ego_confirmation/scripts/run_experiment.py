"""Fail-closed phase runner for Exp072."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import tempfile
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import contract  # noqa: E402
import data_core  # noqa: E402
import model_core  # noqa: E402


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
PHASES = ("prepare", "validate", "evaluate", "report", "verify")


def _completion_path(root: Path, phase: str) -> Path:
    return Path(root) / "metrics" / f".{phase}.complete.json"


def _relative_to_metrics(root: Path, path: Path) -> str:
    return os.path.relpath(Path(path), Path(root) / "metrics")


def _publish_completion(root: Path, phase: str, outputs: list[Path], **extra: Any) -> dict[str, Any]:
    root = Path(root)
    manifest = {
        _relative_to_metrics(root, path): contract.artifact_record(path)
        for path in outputs
    }
    value = {"phase": phase, "status": "complete", "outputs": manifest, **extra}
    contract.atomic_json(_completion_path(root, phase), value)
    return value


def _authenticate_completion(root: Path, phase: str) -> dict[str, Any]:
    root = Path(root)
    marker = _completion_path(root, phase)
    try:
        value = json.loads(marker.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise RuntimeError(f"missing authenticated {phase} completion") from error
    outputs = value.get("outputs")
    if value.get("phase") != phase or value.get("status") != "complete" or not isinstance(outputs, dict) or not outputs:
        raise RuntimeError(f"invalid {phase} completion")
    for relative, expected in outputs.items():
        artifact = marker.parent / relative
        if not artifact.is_file() or contract.artifact_record(artifact) != expected:
            raise RuntimeError(f"{phase} output authentication failed: {artifact}")
    return value


def _save_model(model: Any, path: Path) -> dict[str, int | str]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        joblib.dump(model, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return contract.artifact_record(path)


def _environment() -> dict[str, str]:
    import pandas

    return {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "numpy": np.__version__,
        "pandas": pandas.__version__,
        "scikit_learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "platform": platform.platform(),
    }


def _development_provenance() -> dict[str, Any]:
    source_names = (
        "exp037_prepare_completion",
        "exp039_confirm_completion",
        "prepared_development",
        "confirmation_features",
        "feature_contract",
        "feature_definitions",
        "split_train",
        "split_validation",
    )
    sources = {}
    for name in source_names:
        record = contract.EXPECTED_REGISTRATION["sources"][name]
        path = data_core.authenticate_parent(data_core._resolve(record["path"]), record["sha256"])
        sources[name] = {"path": str(path), **contract.artifact_record(path)}
    parent_scripts = {}
    for name, expected in contract.EXPECTED_REGISTRATION["parent_script_sha256"].items():
        path = data_core.authenticate_parent(
            data_core._resolve(contract.EXPECTED_REGISTRATION["parent_scripts"][name]), expected
        )
        parent_scripts[name] = {"path": str(path), **contract.artifact_record(path)}
    anchors = {}
    for seed, record in contract.EXPECTED_REGISTRATION["anchor_predictions"].items():
        path = data_core.authenticate_parent(data_core._resolve(record["path"]), record["sha256"])
        anchors[seed] = {"path": str(path), **contract.artifact_record(path)}
    return {
        "sources": sources,
        "parent_scripts": parent_scripts,
        "anchor_predictions": anchors,
        "set03_materialized": False,
    }


def _assert_no_test_outputs(root: Path) -> None:
    forbidden = (
        Path(root) / "metrics/prepared_test.csv",
        Path(root) / "metrics/.evaluate.complete.json",
        Path(root) / "predictions/test_candidate_seed42.csv",
        Path(root) / "predictions/test_candidate_seed43.csv",
        Path(root) / "predictions/test_candidate_seed44.csv",
    )
    existing = [str(path) for path in forbidden if path.exists()]
    if existing:
        raise RuntimeError(f"set03 output exists before advancement: {existing}")


def run_prepare(output_root: Path = EXPERIMENT_ROOT) -> str:
    root = Path(output_root)
    marker = _completion_path(root, "prepare")
    if marker.exists():
        _authenticate_completion(root, "prepare")
        _assert_no_test_outputs(root)
        return "complete"
    contract.load_registration()
    _assert_no_test_outputs(root)
    train, validation, audit = data_core.assemble_development()
    metrics = root / "metrics"
    train_path = metrics / "prepared_train.csv"
    validation_path = metrics / "prepared_validation.csv"
    audit_path = metrics / "cohort_audit.json"
    provenance_path = metrics / "source_provenance.json"
    environment_path = metrics / "environment.json"
    contract.atomic_csv(train_path, train)
    contract.atomic_csv(validation_path, validation)
    contract.atomic_json(audit_path, audit)
    contract.atomic_json(provenance_path, _development_provenance())
    contract.atomic_json(environment_path, _environment())
    _publish_completion(
        root,
        "prepare",
        [train_path, validation_path, audit_path, provenance_path, environment_path],
        set03_materialized=False,
    )
    _assert_no_test_outputs(root)
    return "complete"


def _summary_results(per_seed: pd.DataFrame) -> pd.DataFrame:
    metrics = (
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
        "fpr",
        "fnr",
        "brier",
        "log_loss",
        "ece_10bin",
        "fp_ge_0p9_count",
        "fn_le_0p1_count",
        "fp_ge_0p9_rate",
        "fn_le_0p1_rate",
    )
    rows = []
    for (scope, model), group in per_seed.groupby(["scope", "model"], sort=True):
        row: dict[str, Any] = {"scope": scope, "model": model, "seed_count": int(group["seed"].nunique())}
        for name in metrics:
            row[f"{name}_mean"] = float(group[name].mean())
            row[f"{name}_std"] = float(group[name].std(ddof=1)) if len(group) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def _evaluate_gate(deltas: pd.DataFrame) -> dict[str, Any]:
    return model_core.validation_decision(deltas)


def run_validate(output_root: Path = EXPERIMENT_ROOT) -> str:
    root = Path(output_root)
    marker = _completion_path(root, "validate")
    if marker.exists():
        completion = _authenticate_completion(root, "validate")
        _assert_no_test_outputs(root)
        return str(completion["promotion_status"])
    _authenticate_completion(root, "prepare")
    _assert_no_test_outputs(root)
    metrics = root / "metrics"
    models = root / "models"
    predictions = root / "predictions"
    train = pd.read_csv(metrics / "prepared_train.csv")
    validation = pd.read_csv(metrics / "prepared_validation.csv")
    columns = data_core.feature_columns()
    if set(columns).difference(train.columns) or set(columns).difference(validation.columns):
        raise RuntimeError("prepared frames are missing frozen features")
    labels = train["label"].to_numpy(int)
    prediction_parts = []
    model_records = {}
    model_paths = []
    prediction_paths = []
    for seed in model_core.SEEDS:
        model = model_core.build_candidate(seed, labels)
        model.fit(train.loc[:, columns].astype(float), labels)
        model_path = models / f"candidate_seed{seed}.joblib"
        model_records[str(seed)] = {
            "seed": seed,
            "fit_rows": int(len(train)),
            "fit_sets": ["set01", "set02", "set06"],
            "feature_count": len(columns),
            **_save_model(model, model_path),
        }
        model_paths.append(model_path)
        probability = model.predict_proba(validation.loc[:, columns].astype(float))[:, 1]
        part = validation.loc[:, ["sequence_id", "set_id", "track_id", "video_id", "label"]].copy()
        part.insert(0, "seed", seed)
        part["probability"] = probability
        prediction_path = predictions / f"validation_candidate_seed{seed}.csv"
        contract.atomic_csv(prediction_path, part)
        prediction_paths.append(prediction_path)
        prediction_parts.append(part)

    model_manifest_path = metrics / "model_manifest_pre_gate.json"
    contract.atomic_json(
        model_manifest_path,
        {
            "fit_rule": contract.EXPECTED_REGISTRATION["fit_rule"],
            "models": model_records,
            "created_before_validation_gate": True,
        },
    )
    candidate = pd.concat(prediction_parts, ignore_index=True)
    aligned = data_core.align_predictions(candidate, data_core.load_anchor("val"))
    per_seed, deltas = model_core.paired_deltas(aligned, "validation")
    if set(deltas["seed"].astype(int)) != set(model_core.SEEDS):
        raise RuntimeError("validation paired deltas are missing seeds")
    summary = _summary_results(per_seed)
    decision = _evaluate_gate(deltas)
    decision.update(
        {
            "threshold": model_core.THRESHOLD,
            "model_manifest_pre_gate": contract.artifact_record(model_manifest_path),
            "set03_materialized": False,
        }
    )
    per_seed_path = metrics / "validation_per_seed_results.csv"
    summary_path = metrics / "validation_summary_results.csv"
    deltas_path = metrics / "validation_paired_deltas.csv"
    gate_path = metrics / "validation_gate.json"
    contract.atomic_csv(per_seed_path, per_seed)
    contract.atomic_csv(summary_path, summary)
    contract.atomic_csv(deltas_path, deltas)
    contract.atomic_json(gate_path, decision)
    output_paths = [
        *model_paths,
        *prediction_paths,
        model_manifest_path,
        per_seed_path,
        summary_path,
        deltas_path,
        gate_path,
    ]
    _publish_completion(
        root,
        "validate",
        output_paths,
        promotion_status=decision["status"],
        set03_materialized=False,
    )
    _assert_no_test_outputs(root)
    return str(decision["status"])


def _load_model_manifest(root: Path) -> dict[str, Any]:
    root = Path(root)
    gate_path = root / "metrics/validation_gate.json"
    manifest_path = root / "metrics/model_manifest_pre_gate.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    expected = gate.get("model_manifest_pre_gate")
    if expected != contract.artifact_record(manifest_path):
        raise RuntimeError("pre-gate model manifest authentication failed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("fit_rule") != contract.EXPECTED_REGISTRATION["fit_rule"]:
        raise RuntimeError("pre-gate model fit rule changed")
    if set(manifest.get("models", {})) != {"42", "43", "44"}:
        raise RuntimeError("pre-gate model manifest is missing seeds")
    return manifest


def _authenticate_models(root: Path, manifest: dict[str, Any]) -> dict[str, dict[str, int | str]]:
    records: dict[str, dict[str, int | str]] = {}
    for seed in model_core.SEEDS:
        path = Path(root) / "models" / f"candidate_seed{seed}.joblib"
        actual = contract.artifact_record(path)
        expected = manifest["models"][str(seed)]
        if actual["sha256"] != expected.get("sha256") or actual["size_bytes"] != expected.get("size_bytes"):
            raise RuntimeError(f"frozen model hash changed for seed {seed}")
        if expected.get("fit_rows") != 2366 and Path(root).resolve() == EXPERIMENT_ROOT.resolve():
            raise RuntimeError(f"frozen model fit cohort changed for seed {seed}")
        records[str(seed)] = actual
    return records


def _strict_target_assessment(per_seed: pd.DataFrame) -> dict[str, Any]:
    candidate = per_seed.loc[per_seed["model"].eq("candidate")]
    if set(candidate["seed"].astype(int)) != set(model_core.SEEDS):
        raise RuntimeError("strict targets require exactly three candidate seeds")
    means = {
        "f1": float(candidate["f1"].mean()),
        "pr_auc": float(candidate["pr_auc"].mean()),
        "roc_auc": float(candidate["roc_auc"].mean()),
    }
    targets = contract.EXPECTED_REGISTRATION["strict_test_targets"]
    checks = {name: means[name] > float(targets[name]) for name in means}
    return {
        "basis": "candidate_three_seed_metric_mean",
        "means": means,
        "targets": targets,
        "checks": checks,
        "all_targets_beaten": bool(all(checks.values())),
    }


def run_evaluate(output_root: Path = EXPERIMENT_ROOT) -> str:
    root = Path(output_root)
    marker = _completion_path(root, "evaluate")
    if marker.exists():
        _authenticate_completion(root, "evaluate")
        return "complete"
    validation_completion = _authenticate_completion(root, "validate")
    if validation_completion.get("promotion_status") != "advance":
        raise RuntimeError("validation did not advance; set03 evaluation is forbidden")
    gate_path = root / "metrics/validation_gate.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("status") != "advance" or not all(gate.get("checks", {}).values()):
        raise RuntimeError("validation did not advance; set03 evaluation is forbidden")
    model_manifest = _load_model_manifest(root)
    before_hashes = _authenticate_models(root, model_manifest)

    test, test_provenance = data_core.materialize_test_after_gate(gate_path)
    columns = data_core.feature_columns()
    if set(columns).difference(test.columns):
        raise RuntimeError("test matrix is missing frozen features")
    metrics = root / "metrics"
    predictions = root / "predictions"
    prepared_test_path = metrics / "prepared_test.csv"
    provenance_path = metrics / "test_source_provenance.json"
    contract.atomic_csv(prepared_test_path, test)
    contract.atomic_json(provenance_path, test_provenance)

    prediction_parts = []
    prediction_paths = []
    for seed in model_core.SEEDS:
        model_path = root / "models" / f"candidate_seed{seed}.joblib"
        model = joblib.load(model_path)
        probability = model.predict_proba(test.loc[:, columns].astype(float))[:, 1]
        part = test.loc[:, ["sequence_id", "set_id", "track_id", "video_id", "label"]].copy()
        part.insert(0, "seed", seed)
        part["probability"] = probability
        path = predictions / f"test_candidate_seed{seed}.csv"
        contract.atomic_csv(path, part)
        prediction_paths.append(path)
        prediction_parts.append(part)
    after_hashes = _authenticate_models(root, model_manifest)
    if before_hashes != after_hashes:
        raise RuntimeError("frozen model changed during set03 evaluation")
    model_hash_path = metrics / "model_hashes_after_evaluation.json"
    contract.atomic_json(
        model_hash_path,
        {"before_evaluation": before_hashes, "after_evaluation": after_hashes, "unchanged": True},
    )

    candidate = pd.concat(prediction_parts, ignore_index=True)
    aligned = data_core.align_predictions(candidate, data_core.load_anchor("test"))
    per_seed, deltas = model_core.paired_deltas(aligned, "test")
    summary = _summary_results(per_seed)
    transitions = model_core.error_transitions(aligned)
    bootstrap = pd.concat(
        [
            model_core.cluster_bootstrap_deltas(
                group,
                replicates=int(contract.EXPECTED_REGISTRATION["bootstrap"]["replicates"]),
                seed=int(contract.EXPECTED_REGISTRATION["bootstrap"]["seed"]),
            )
            for _, group in aligned.groupby("seed", sort=True)
        ],
        ignore_index=True,
    )
    ensemble = (
        candidate.groupby(["sequence_id", "set_id", "track_id", "video_id", "label"], sort=True)["probability"]
        .mean()
        .reset_index()
    )
    ensemble_metrics = pd.DataFrame(
        [
            {
                "scope": "test",
                "model": "secondary_seed_probability_ensemble",
                **model_core.compute_metrics(
                    ensemble["label"].to_numpy(int), ensemble["probability"].to_numpy(float)
                ),
            }
        ]
    )
    strict = _strict_target_assessment(per_seed)

    per_seed_path = metrics / "fixed_0p5_per_seed_results.csv"
    summary_path = metrics / "summary_results.csv"
    deltas_path = metrics / "paired_deltas.csv"
    bootstrap_path = metrics / "bootstrap_deltas.csv"
    transitions_path = metrics / "error_transitions_fixed_0p5.csv"
    ensemble_path = metrics / "secondary_ensemble_results.csv"
    strict_path = metrics / "strict_target_assessment.json"
    contract.atomic_csv(per_seed_path, per_seed)
    contract.atomic_csv(summary_path, summary)
    contract.atomic_csv(deltas_path, deltas)
    contract.atomic_csv(bootstrap_path, bootstrap)
    contract.atomic_csv(transitions_path, transitions)
    contract.atomic_csv(ensemble_path, ensemble_metrics)
    contract.atomic_json(strict_path, strict)
    model_paths = [root / "models" / f"candidate_seed{seed}.joblib" for seed in model_core.SEEDS]
    _publish_completion(
        root,
        "evaluate",
        [
            *model_paths,
            *prediction_paths,
            prepared_test_path,
            provenance_path,
            model_hash_path,
            per_seed_path,
            summary_path,
            deltas_path,
            bootstrap_path,
            transitions_path,
            ensemble_path,
            strict_path,
        ],
        model_refit=False,
        set03_materialized=True,
    )
    return "complete"


def _markdown(frame: pd.DataFrame) -> str:
    def cell(value: object) -> str:
        if pd.isna(value):
            rendered = ""
        elif isinstance(value, float):
            rendered = f"{value:.6f}"
        else:
            rendered = str(value)
        return rendered.replace("|", "\\|").replace("\n", " ")

    headers = [cell(column) for column in frame.columns]
    rows = [[cell(value) for value in row] for row in frame.itertuples(index=False, name=None)]
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
            *("| " + " | ".join(row) + " |" for row in rows),
        ]
    )


def run_report(output_root: Path = EXPERIMENT_ROOT) -> str:
    root = Path(output_root)
    marker = _completion_path(root, "report")
    if marker.exists():
        completion = _authenticate_completion(root, "report")
        return str(completion["result"])
    validation = _authenticate_completion(root, "validate")
    gate = json.loads((root / "metrics/validation_gate.json").read_text(encoding="utf-8"))
    validation_summary = pd.read_csv(root / "metrics/validation_summary_results.csv")
    validation_deltas = pd.read_csv(root / "metrics/validation_paired_deltas.csv")
    report_path = root / "FINAL_REPORT.md"
    lines = [
        "# Exp072 Final Report",
        "",
        "PIE 16 observed frames, TTE 30--60, overlap 0.6, fixed threshold 0.5.",
        "The candidate uses 187 frozen BBox/BIU/temporal/ego features and three seeds.",
        "",
        "## Validation",
        "",
        _markdown(validation_summary),
        "",
        "### Paired Candidate Minus Anchor Deltas",
        "",
        _markdown(validation_deltas),
        "",
        "### Frozen Gate",
        "",
        "```json",
        json.dumps(gate, indent=2, sort_keys=True),
        "```",
        "",
    ]
    outputs = [report_path]
    if validation.get("promotion_status") == "no_advance":
        lines.extend(
            [
                "## Outcome",
                "",
                "`NO ADVANCE`. Set03 was not materialized or evaluated because the frozen validation gate failed.",
                "No model, feature, threshold, or gate was changed.",
                "",
            ]
        )
        terminal_path = root / "VALIDATION_COMPLETE_NO_ADVANCE"
        contract.atomic_json(terminal_path, {"status": "no_advance", "set03_materialized": False})
        outputs.append(terminal_path)
        result = "no_advance"
    else:
        _authenticate_completion(root, "evaluate")
        per_seed = pd.read_csv(root / "metrics/fixed_0p5_per_seed_results.csv")
        summary = pd.read_csv(root / "metrics/summary_results.csv")
        deltas = pd.read_csv(root / "metrics/paired_deltas.csv")
        bootstrap = pd.read_csv(root / "metrics/bootstrap_deltas.csv")
        strict = json.loads((root / "metrics/strict_target_assessment.json").read_text(encoding="utf-8"))
        lines.extend(
            [
                "## Set03 Evaluation",
                "",
                _markdown(per_seed),
                "",
                "### Mean And Standard Deviation",
                "",
                _markdown(summary),
                "",
                "### Candidate Minus Anchor Deltas",
                "",
                _markdown(deltas),
                "",
                "### Track-Cluster Bootstrap",
                "",
                _markdown(bootstrap),
                "",
                "### Strict Target Assessment",
                "",
                "```json",
                json.dumps(strict, indent=2, sort_keys=True),
                "```",
                "",
                "## Limitations",
                "",
                "Set03 is the official Exp072 test cohort but had prior exposure in separate experiments and qualitative work.",
                "All Exp072 choices were frozen before this evaluation, and the pre-gate model hashes remained unchanged.",
                "The historical pose cache is byte-pinned but its original ViTPose-S checkpoint identity is not cryptographically verifiable.",
                "",
            ]
        )
        result = "complete"
    contract.atomic_text(report_path, "\n".join(lines))
    _publish_completion(root, "report", outputs, result=result)
    return result


def _assert_metrics_equal(actual: pd.DataFrame, expected: pd.DataFrame) -> None:
    keys = ["scope", "seed", "model"]
    actual = actual.sort_values(keys, kind="stable").reset_index(drop=True)
    expected = expected.sort_values(keys, kind="stable").reset_index(drop=True)
    if list(actual.columns) != list(expected.columns):
        raise RuntimeError("saved metric columns changed")
    try:
        pd.testing.assert_frame_equal(actual, expected, check_dtype=False, rtol=0.0, atol=1e-8)
    except AssertionError as error:
        raise RuntimeError("saved metrics do not recompute from predictions") from error


def run_verify(output_root: Path = EXPERIMENT_ROOT) -> str:
    root = Path(output_root)
    marker = _completion_path(root, "verify")
    if marker.exists():
        completion = _authenticate_completion(root, "verify")
        return str(completion["result"])
    _authenticate_completion(root, "prepare")
    validation = _authenticate_completion(root, "validate")
    _authenticate_completion(root, "report")
    checks: dict[str, str] = {
        "prepare_completion": "pass",
        "validation_completion": "pass",
        "report_completion": "pass",
    }
    verification_path = root / "metrics/verification_report.json"
    if validation.get("promotion_status") == "no_advance":
        _assert_no_test_outputs(root)
        if (root / "EXPERIMENT_COMPLETE").exists():
            raise RuntimeError("no-advance path contains EXPERIMENT_COMPLETE")
        terminal = root / "VALIDATION_COMPLETE_NO_ADVANCE"
        if not terminal.is_file():
            raise RuntimeError("no-advance terminal marker is missing")
        checks.update({"set03_absent": "pass", "no_experiment_complete": "pass"})
        contract.atomic_json(
            verification_path,
            {"status": "verified_no_advance", "checks": checks, "set03_materialized": False},
        )
        _publish_completion(
            root,
            "verify",
            [verification_path, terminal],
            result="verified_no_advance",
        )
        return "verified_no_advance"

    _authenticate_completion(root, "evaluate")
    manifest = _load_model_manifest(root)
    model_hashes = _authenticate_models(root, manifest)
    parts = [pd.read_csv(root / "predictions" / f"test_candidate_seed{seed}.csv") for seed in model_core.SEEDS]
    candidate = pd.concat(parts, ignore_index=True)
    aligned = data_core.align_predictions(candidate, data_core.load_anchor("test"))
    recomputed, _ = model_core.paired_deltas(aligned, "test")
    stored = pd.read_csv(root / "metrics/fixed_0p5_per_seed_results.csv")
    _assert_metrics_equal(recomputed, stored)
    strict = _strict_target_assessment(recomputed)
    stored_strict = json.loads((root / "metrics/strict_target_assessment.json").read_text(encoding="utf-8"))
    if strict != stored_strict:
        raise RuntimeError("strict target assessment does not recompute")
    checks.update(
        {
            "evaluation_completion": "pass",
            "frozen_model_hashes": "pass",
            "saved_metrics_recompute": "pass",
            "strict_targets_recompute": "pass",
            "evaluation_model_refit": "absent_by_source_contract_and_unchanged_model_hashes",
        }
    )
    contract.atomic_json(
        verification_path,
        {
            "status": "complete",
            "checks": checks,
            "model_hashes": model_hashes,
            "strict_target_assessment": strict,
        },
    )
    terminal = root / "EXPERIMENT_COMPLETE"
    contract.atomic_json(
        terminal,
        {
            "status": "complete",
            "verification_report": contract.artifact_record(verification_path),
            "all_targets_beaten": strict["all_targets_beaten"],
        },
    )
    _publish_completion(root, "verify", [verification_path, terminal], result="complete")
    return "complete"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=PHASES)
    arguments = parser.parse_args()
    if arguments.phase == "prepare":
        status = run_prepare()
    elif arguments.phase == "validate":
        status = run_validate()
    elif arguments.phase == "evaluate":
        status = run_evaluate()
    elif arguments.phase == "report":
        status = run_report()
    elif arguments.phase == "verify":
        status = run_verify()
    else:
        raise AssertionError(f"unreachable phase: {arguments.phase}")
    print(json.dumps({"phase": arguments.phase, "status": status}, sort_keys=True))


if __name__ == "__main__":
    main()
