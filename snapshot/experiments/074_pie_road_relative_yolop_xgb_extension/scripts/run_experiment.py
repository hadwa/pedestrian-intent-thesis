"""Fail-closed development runner for the Exp074 road-relative extension."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import context  # noqa: E402
import contract  # noqa: E402
import model  # noqa: E402


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]


def _completion_path(root: Path, phase: str) -> Path:
    return Path(root) / "metrics" / f".{phase}.complete.json"


def _publish(root: Path, phase: str, outputs: list[Path], **extra: Any) -> None:
    marker = _completion_path(root, phase)
    value = {
        "phase": phase, "status": "complete",
        "outputs": {os.path.relpath(path, marker.parent): contract.artifact_record(path) for path in outputs},
        **extra,
    }
    contract.atomic_json(marker, value)


def _authenticate(root: Path, phase: str) -> dict[str, Any]:
    marker = _completion_path(root, phase)
    try:
        value = json.loads(marker.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise RuntimeError(f"missing authenticated {phase} completion") from error
    if value.get("phase") != phase or value.get("status") != "complete" or not value.get("outputs"):
        raise RuntimeError(f"invalid {phase} completion")
    for relative, expected in value["outputs"].items():
        path = marker.parent / relative
        if not path.is_file() or contract.artifact_record(path) != expected:
            raise RuntimeError(f"{phase} output authentication failed: {path}")
    return value


def _parent_completion(name: str) -> tuple[Path, dict[str, Any]]:
    marker = contract.resolve_source(name)
    value = json.loads(marker.read_text(encoding="utf-8"))
    if value.get("status") != "complete" or not value.get("outputs"):
        raise RuntimeError(f"invalid parent completion: {name}")
    for relative, expected in value["outputs"].items():
        path = marker.parent / relative
        if not path.is_file() or contract.artifact_record(path) != expected:
            raise RuntimeError(f"parent completion authentication failed: {path}")
    return marker, value


def _parent_output(marker: Path, completion: dict[str, Any], suffix: str) -> Path:
    matches = [relative for relative in completion["outputs"] if relative.endswith(suffix)]
    if len(matches) != 1:
        raise RuntimeError(f"expected one parent output ending in {suffix}")
    return marker.parent / matches[0]


def _base_columns(frame: pd.DataFrame) -> list[str]:
    marker, completion = _parent_completion("exp072_prepare_completion")
    audit = json.loads(_parent_output(marker, completion, "cohort_audit.json").read_text(encoding="utf-8"))
    columns = list(map(str, audit.get("feature_columns", [])))
    if len(columns) != 187 or len(columns) != len(set(columns)) or set(columns).difference(frame.columns):
        raise RuntimeError("Exp072 base feature contract changed")
    return columns


def _features(frame: pd.DataFrame) -> list[str]:
    columns = _base_columns(frame) + list(context.context_columns())
    if len(columns) != 247 or set(columns).difference(frame.columns):
        raise RuntimeError("Exp074 requires exactly 247 features")
    return columns


def _assert_no_test(root: Path) -> None:
    forbidden = [
        root / "metrics/prepared_test.csv", root / "metrics/.extract.complete.json",
        root / "metrics/.evaluate.complete.json", root / "cache/set03_parts",
    ]
    present = [str(path) for path in forbidden if path.exists()]
    if present:
        raise RuntimeError(f"set03 material exists before validation advance: {present}")


def _save_model(value: Any, path: Path) -> dict[str, int | str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        joblib.dump(value, temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return contract.artifact_record(path)


def _reference_predictions(split: str) -> pd.DataFrame:
    source = "exp072_validate_completion" if split == "validation" else "exp072_evaluate_completion"
    marker, completion = _parent_completion(source)
    parts = []
    for seed in model.SEEDS:
        suffix = f"validation_candidate_seed{seed}.csv" if split == "validation" else f"test_candidate_seed{seed}.csv"
        parts.append(pd.read_csv(_parent_output(marker, completion, suffix)))
    return pd.concat(parts, ignore_index=True)


def _align(candidate: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    keys = ["seed", "sequence_id", "set_id", "track_id", "video_id", "label"]
    merged = candidate.merge(
        reference.loc[:, keys + ["probability"]].rename(columns={"probability": "anchor_probability"}),
        on=keys, how="inner", validate="one_to_one", sort=False,
    )
    if len(merged) != len(candidate) or len(merged) != len(reference):
        raise RuntimeError("Exp074 and Exp072 predictions are not exactly aligned")
    return merged


def _metrics(aligned: pd.DataFrame, scope: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, deltas = [], []
    for seed, group in aligned.groupby("seed", sort=True):
        labels = group.label.to_numpy(int)
        candidate = model.compute_metrics(labels, group.probability.to_numpy(float))
        reference = model.compute_metrics(labels, group.anchor_probability.to_numpy(float))
        rows.extend([
            {"scope": scope, "seed": seed, "model": "candidate", **candidate},
            {"scope": scope, "seed": seed, "model": "reference", **reference},
        ])
        row = {"scope": scope, "seed": seed}
        for name in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "fpr", "fnr", "brier", "log_loss", "ece_10bin", "fp_ge_0p9_count", "fn_le_0p1_count"):
            row[f"delta_{name}"] = float(candidate[name] - reference[name])
        deltas.append(row)
    return pd.DataFrame(rows), pd.DataFrame(deltas)


def _summary(per_seed: pd.DataFrame) -> pd.DataFrame:
    metrics = ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "fpr", "fnr", "brier", "log_loss", "ece_10bin", "fp_ge_0p9_count", "fn_le_0p1_count")
    rows = []
    for (scope, name), group in per_seed.groupby(["scope", "model"], sort=True):
        row: dict[str, Any] = {"scope": scope, "model": name, "seed_count": len(group)}
        for metric_name in metrics:
            row[f"{metric_name}_mean"] = float(group[metric_name].mean())
            row[f"{metric_name}_std"] = float(group[metric_name].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def _transition_summary(aligned: pd.DataFrame) -> tuple[pd.DataFrame, float, float]:
    output = aligned.copy()
    candidate_correct = (output.probability >= .5).astype(int).eq(output.label.astype(int))
    reference_correct = (output.anchor_probability >= .5).astype(int).eq(output.label.astype(int))
    output["transition"] = np.select(
        [~reference_correct & candidate_correct, reference_correct & ~candidate_correct, reference_correct & candidate_correct],
        ["reference_wrong_candidate_correct", "reference_correct_candidate_wrong", "both_correct"],
        default="both_wrong",
    )
    counts = output.groupby(["seed", "transition"], sort=True).size().unstack(fill_value=0)
    corrections = float(counts.get("reference_wrong_candidate_correct", pd.Series(0, index=counts.index)).mean())
    harms = float(counts.get("reference_correct_candidate_wrong", pd.Series(0, index=counts.index)).mean())
    return output, corrections, harms


def run_prepare(output_root: Path = EXPERIMENT_ROOT) -> str:
    root = Path(output_root)
    if _completion_path(root, "prepare").exists():
        _authenticate(root, "prepare")
        _assert_no_test(root)
        return "complete"
    contract.load_registration()
    _assert_no_test(root)
    train = pd.read_csv(contract.resolve_source("prepared_train"))
    validation = pd.read_csv(contract.resolve_source("prepared_validation"))
    roads, coverage = context.load_development_context()
    if set(train.sequence_id) & set(validation.sequence_id):
        raise RuntimeError("train and validation sequence IDs overlap")
    expected_ids = set(train.sequence_id) | set(validation.sequence_id)
    if set(roads.sequence_id) != expected_ids:
        raise RuntimeError("road-relative development cohort does not exactly match Exp072")
    train = train.merge(roads, on="sequence_id", how="left", validate="one_to_one", sort=False)
    validation = validation.merge(roads, on="sequence_id", how="left", validate="one_to_one", sort=False)
    columns = _features(train)
    if columns != _features(validation) or not np.isfinite(train[columns].to_numpy(float)).all() or not np.isfinite(validation[columns].to_numpy(float)).all():
        raise RuntimeError("Exp074 development features are invalid")
    outputs = [root / "metrics/prepared_train.csv", root / "metrics/prepared_validation.csv"]
    contract.atomic_csv(outputs[0], train)
    contract.atomic_csv(outputs[1], validation)
    coverage_path = root / "metrics/context_coverage.json"
    provenance_path = root / "metrics/context_provenance.json"
    contract.atomic_json(coverage_path, coverage)
    contract.atomic_json(provenance_path, {
        name: {"path": str(contract.resolve_source(name)), **contract.artifact_record(contract.resolve_source(name))}
        for name in ("exp063_manifest", "exp063_representations", "exp063_contracts", "exp063_extract", "exp063_yolop")
    })
    outputs.extend([coverage_path, provenance_path])
    _publish(root, "prepare", outputs, feature_count=247, set03_materialized=False)
    _assert_no_test(root)
    return "complete"


def run_validate(output_root: Path = EXPERIMENT_ROOT) -> str:
    root = Path(output_root)
    if _completion_path(root, "validate").exists():
        completion = _authenticate(root, "validate")
        _assert_no_test(root)
        return str(completion["promotion_status"])
    _authenticate(root, "prepare")
    _assert_no_test(root)
    train = pd.read_csv(root / "metrics/prepared_train.csv")
    validation = pd.read_csv(root / "metrics/prepared_validation.csv")
    columns = _features(train)
    labels = train.label.to_numpy(int)
    parts, outputs, records = [], [], {}
    for seed in model.SEEDS:
        predictor = model.build_xgb(seed, labels)
        predictor.fit(train[columns].astype(float), labels)
        model_path = root / f"models/candidate_seed{seed}.joblib"
        records[str(seed)] = {"fit_rows": len(train), "feature_count": len(columns), **_save_model(predictor, model_path)}
        outputs.append(model_path)
        part = validation.loc[:, ["sequence_id", "set_id", "track_id", "video_id", "label"]].copy()
        part.insert(0, "seed", seed)
        part["probability"] = predictor.predict_proba(validation[columns].astype(float))[:, 1]
        prediction_path = root / f"predictions/validation_candidate_seed{seed}.csv"
        contract.atomic_csv(prediction_path, part)
        outputs.append(prediction_path)
        parts.append(part)
    manifest_path = root / "metrics/model_manifest_pre_gate.json"
    contract.atomic_json(manifest_path, {"fit_rule": contract.EXPECTED_REGISTRATION["fit_rule"], "models": records})
    outputs.append(manifest_path)
    aligned = _align(pd.concat(parts, ignore_index=True), _reference_predictions("validation"))
    per_seed, deltas = _metrics(aligned, "validation")
    transitions, corrections, harms = _transition_summary(aligned)
    gate = model.validation_gate(per_seed, corrections=corrections, harms=harms)
    gate.update({"threshold": .5, "reference": "Exp072_XGBoost", "set03_materialized": False, "model_manifest_pre_gate": contract.artifact_record(manifest_path)})
    paths = {
        "per_seed": root / "metrics/validation_per_seed_results.csv",
        "summary": root / "metrics/validation_summary_results.csv",
        "deltas": root / "metrics/validation_paired_deltas.csv",
        "transitions": root / "metrics/validation_error_transitions.csv",
        "gate": root / "metrics/validation_gate.json",
    }
    contract.atomic_csv(paths["per_seed"], per_seed)
    contract.atomic_csv(paths["summary"], _summary(per_seed))
    contract.atomic_csv(paths["deltas"], deltas)
    contract.atomic_csv(paths["transitions"], transitions)
    contract.atomic_json(paths["gate"], gate)
    outputs.extend(paths.values())
    _publish(root, "validate", outputs, promotion_status=gate["status"], set03_materialized=False)
    _assert_no_test(root)
    return str(gate["status"])


def _load_test(root: Path) -> pd.DataFrame:
    try:
        completion = _authenticate(root, "validate")
        gate = json.loads((root / "metrics/validation_gate.json").read_text(encoding="utf-8"))
    except RuntimeError as error:
        raise RuntimeError("validation advance is required before set03") from error
    if completion.get("promotion_status") != "advance" or gate.get("status") != "advance":
        raise RuntimeError("validation did not advance; set03 is forbidden")
    raise RuntimeError("authorized set03 road extraction has not completed")


def _load_models(root: Path) -> dict[int, Any]:
    gate = json.loads((root / "metrics/validation_gate.json").read_text(encoding="utf-8"))
    manifest_path = root / "metrics/model_manifest_pre_gate.json"
    if gate.get("model_manifest_pre_gate") != contract.artifact_record(manifest_path):
        raise RuntimeError("pre-gate model manifest changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    predictors = {}
    for seed in model.SEEDS:
        path = root / f"models/candidate_seed{seed}.joblib"
        expected = manifest["models"][str(seed)]
        actual = contract.artifact_record(path)
        if actual != {"sha256": expected["sha256"], "size_bytes": expected["size_bytes"]}:
            raise RuntimeError(f"frozen model changed for seed {seed}")
        predictors[seed] = joblib.load(path)
    return predictors


def _ensure_latency(root: Path) -> tuple[Path, Path, Path]:
    _authenticate(root, "validate")
    latency_path = root / "metrics/inference_latency.csv"
    complexity_path = root / "metrics/model_complexity.csv"
    context_path = root / "metrics/context_latency.json"
    if all(path.is_file() for path in (latency_path, complexity_path, context_path)):
        return latency_path, complexity_path, context_path
    validation = pd.read_csv(root / "metrics/prepared_validation.csv")
    candidate_columns = _features(validation)
    reference_columns = _base_columns(validation)
    predictors = _load_models(root)
    marker, completion = _parent_completion("exp072_validate_completion")
    latency, sizes = [], []
    for seed in model.SEEDS:
        reference_path = _parent_output(marker, completion, f"candidate_seed{seed}.joblib")
        reference = joblib.load(reference_path)
        latency.extend([
            model.benchmark_pipeline(reference, "Exp072_XGBoost", seed, validation[reference_columns].astype(float)),
            model.benchmark_pipeline(predictors[seed], "Exp074_Road_XGBoost", seed, validation[candidate_columns].astype(float)),
        ])
        sizes.extend([
            {"model": "Exp072_XGBoost", "seed": seed, "model_size_bytes": reference_path.stat().st_size, "feature_count": 187},
            {"model": "Exp074_Road_XGBoost", "seed": seed, "model_size_bytes": (root / f"models/candidate_seed{seed}.joblib").stat().st_size, "feature_count": 247},
        ])
    contract.atomic_csv(latency_path, pd.concat(latency, ignore_index=True))
    contract.atomic_csv(complexity_path, pd.DataFrame(sizes))
    settings = contract.EXPECTED_REGISTRATION["latency"]["summarizer"]
    timing = context.benchmark_summarizer(np.zeros((16, 20)), warmups=settings["warmups"], repetitions=settings["repetitions"])
    coverage = json.loads((root / "metrics/context_coverage.json").read_text(encoding="utf-8"))
    contract.atomic_json(context_path, {"summarizer": timing, "development_yolop_upstream": coverage["upstream_yolop"], "vitpose_biu_upstream": "separate_not_remeasured"})
    return latency_path, complexity_path, context_path


def _markdown(frame: pd.DataFrame) -> str:
    def cell(value: object) -> str:
        return (f"{value:.6f}" if isinstance(value, float) else str(value)).replace("|", "\\|")
    headers = [cell(column) for column in frame.columns]
    rows = [[cell(value) for value in row] for row in frame.itertuples(index=False, name=None)]
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |", *("| " + " | ".join(row) + " |" for row in rows)])


def run_report(output_root: Path = EXPERIMENT_ROOT) -> str:
    root = Path(output_root)
    validation = _authenticate(root, "validate")
    if validation["promotion_status"] == "advance":
        raise RuntimeError("Exp074 advanced; gated set03 extraction must run before final reporting")
    latency, complexity, context_latency = _ensure_latency(root)
    gate = json.loads((root / "metrics/validation_gate.json").read_text(encoding="utf-8"))
    report = root / "FINAL_REPORT.md"
    contract.atomic_text(report, "\n".join([
        "# Exp074 Final Report", "",
        "Validation-only controlled comparison of Exp072 against Exp072 plus 60 frozen road-relative YOLOP summaries.", "",
        "## Validation", "", _markdown(pd.read_csv(root / "metrics/validation_summary_results.csv")), "",
        _markdown(pd.read_csv(root / "metrics/validation_paired_deltas.csv")), "", "```json",
        json.dumps(gate, indent=2, sort_keys=True), "```", "", "## Inference Latency", "",
        _markdown(pd.read_csv(latency)), "", "## Outcome", "",
        "`NO ADVANCE`. Set03 road extraction and evaluation were not run.", "",
    ]))
    terminal = root / "VALIDATION_COMPLETE_NO_ADVANCE"
    contract.atomic_json(terminal, {"status": "no_advance", "set03_materialized": False})
    _publish(root, "report", [report, terminal, latency, complexity, context_latency], result="no_advance")
    return "no_advance"


def run_verify(output_root: Path = EXPERIMENT_ROOT) -> str:
    root = Path(output_root)
    _authenticate(root, "prepare")
    validation = _authenticate(root, "validate")
    _authenticate(root, "report")
    if validation["promotion_status"] != "no_advance":
        raise RuntimeError("advanced Exp074 requires the evaluation verifier")
    _assert_no_test(root)
    path = root / "metrics/verification_report.json"
    contract.atomic_json(path, {"status": "pass", "result": "no_advance", "checks": {"prepare": "pass", "validate": "pass", "report": "pass", "set03_absent": "pass"}})
    _publish(root, "verify", [path], result="no_advance")
    return "no_advance"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("prepare", "validate", "report", "verify", "all"), required=True)
    parser.add_argument("--output-root", type=Path, default=EXPERIMENT_ROOT)
    args = parser.parse_args()
    if args.phase in ("prepare", "all"):
        run_prepare(args.output_root)
    if args.phase in ("validate", "all"):
        run_validate(args.output_root)
    if args.phase in ("report", "all"):
        run_report(args.output_root)
    if args.phase in ("verify", "all"):
        run_verify(args.output_root)


if __name__ == "__main__":
    main()
