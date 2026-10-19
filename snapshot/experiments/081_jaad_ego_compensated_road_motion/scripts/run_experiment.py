"""Authenticated phase runner for Exp081."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = SCRIPT_DIR.parent


def _load_sibling(filename: str, name: str):
    path = SCRIPT_DIR / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load Exp081 module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


contract = _load_sibling("contract.py", "exp081_contract_for_runner")
jaad_data = _load_sibling("jaad_data.py", "exp081_data_for_runner")
extraction = _load_sibling("extract.py", "exp081_extract_for_runner")
model = _load_sibling("model.py", "exp081_model_for_runner")
latency_module = _load_sibling("latency.py", "exp081_latency_for_runner")


def _log(message: str) -> None:
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"[{timestamp}] {message}", flush=True)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    contract.atomic_write_text(path, frame.to_csv(index=False, lineterminator="\n"))


def _atomic_joblib(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=path.suffix, dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        joblib.dump(value, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sequence_digest(values: pd.Series | np.ndarray) -> str:
    import hashlib

    digest = hashlib.sha256()
    for value in np.asarray(values).astype(str):
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def run_audit(root: Path | str = EXPERIMENT_ROOT) -> dict[str, Any]:
    root = Path(root)
    registration = contract.load_registration()
    sources = {
        name: contract.artifact_record(contract.resolve_source(name))
        for name in sorted(registration["sources"])
    }
    adapter = extraction._load_twinlite_adapter()
    _, twinlite_load_audit = adapter.load_large_model(map_location="cpu")
    audit = {
        "experiment": registration["experiment"],
        "status": "pass",
        "test_accessed": False,
        "sources": sources,
        "registration": contract.artifact_record(contract.REGISTRATION_PATH),
        "twinlite_strict_cpu_load": twinlite_load_audit,
    }
    path = root / "metrics/provenance_audit.json"
    contract.atomic_write_json(path, audit)
    contract.publish_phase(root, "audit", [path], test_accessed=False)
    _log(f"audit complete: {len(sources)} authenticated sources")
    return audit


def run_extract_development(root: Path | str = EXPERIMENT_ROOT) -> dict[str, Any]:
    root = Path(root)
    contract.authenticate_phase(root, "audit")
    extractor = extraction.create_twinlite_extractor(device="cuda", half=True)
    outputs, gate = extraction.run_development_extraction(root, extractor)
    contract.publish_phase(
        root,
        "extract-development",
        outputs,
        test_accessed=False,
        quality_status=gate["status"],
    )
    _log(f"development extraction complete: quality={gate['status']}")
    return gate


def _load_development(root: Path) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    anchors = {
        "train": jaad_data.load_anchor("train"),
        "validation": jaad_data.load_anchor("validation"),
    }
    contexts = {
        split: extraction.load_context(
            root, split, anchor["sequence_id"].astype(str).to_numpy()
        )
        for split, anchor in anchors.items()
    }
    return anchors, contexts


def run_prepare(root: Path | str = EXPERIMENT_ROOT) -> dict[str, Any]:
    root = Path(root)
    marker = contract.authenticate_phase(root, "extract-development")
    if marker.get("quality_status") != "pass":
        raise RuntimeError("Exp081 extract-development quality gate did not pass")
    anchors, contexts = _load_development(root)
    parity: dict[str, Any] = {"test_accessed": False, "splits": {}}
    for split in ("train", "validation"):
        variants = model.assemble_variants(anchors[split], contexts[split])
        ids = anchors[split]["sequence_id"].astype(str)
        parity["splits"][split] = {
            "rows": len(ids),
            "sequence_sha256": _sequence_digest(ids),
            "dimensions": {
                name: len(model.feature_columns(frame))
                for name, frame in variants.items()
            },
            "labels_identical": all(
                np.array_equal(frame["label"], anchors[split]["label"])
                for frame in variants.values()
            ),
        }
    path = root / "metrics/cohort_parity.json"
    contract.atomic_write_json(path, parity)
    contract.publish_phase(root, "prepare", [path], test_accessed=False)
    _log("prepare complete: A/B/C/D cohort and dimensional parity passed")
    return parity


def _summary(per_seed: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
        "brier",
        "log_loss",
        "high_confidence_fp",
        "high_confidence_fn",
        "high_confidence_errors",
    ]
    rows = []
    for variant, group in per_seed.groupby("variant", sort=True):
        row: dict[str, Any] = {"variant": variant, "seeds": len(group)}
        for metric_name in metrics:
            values = group[metric_name].to_numpy(float)
            row[f"{metric_name}_mean"] = float(values.mean())
            row[f"{metric_name}_std"] = float(values.std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def _a_parity(per_seed: pd.DataFrame) -> dict[str, Any]:
    historical = pd.read_csv(contract.resolve_source("exp077_validation_results"))
    historical = historical.loc[historical["model"].eq("one_stage")].set_index("seed")
    current = per_seed.loc[per_seed["variant"].eq("A")].set_index("seed")
    metrics = ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "brier", "log_loss")
    differences = {
        str(seed): {
            metric_name: float(current.loc[seed, metric_name] - historical.loc[seed, metric_name])
            for metric_name in metrics
        }
        for seed in contract.load_registration()["seeds"]
    }
    maximum = max(abs(value) for row in differences.values() for value in row.values())
    return {
        "status": "pass" if maximum <= 1e-12 else "fail",
        "maximum_absolute_metric_difference": maximum,
        "tolerance": 1e-12,
        "differences": differences,
    }


def run_validate(root: Path | str = EXPERIMENT_ROOT) -> dict[str, Any]:
    root = Path(root)
    try:
        contract.authenticate_phase(root, "prepare")
    except RuntimeError as exc:
        raise RuntimeError("Exp081 validate requires authenticated prepare phase") from exc
    anchors, contexts = _load_development(root)
    train_variants = model.assemble_variants(anchors["train"], contexts["train"])
    validation_variants = model.assemble_variants(
        anchors["validation"], contexts["validation"]
    )
    labels_train = anchors["train"]["label"].to_numpy(int)
    labels_validation = anchors["validation"]["label"].to_numpy(int)
    prediction_rows: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    model_paths: list[Path] = []
    for seed in contract.load_registration()["seeds"]:
        for variant in ("A", "B", "C", "D"):
            _log(f"validate training variant={variant} seed={seed}")
            train_frame = train_variants[variant]
            validation_frame = validation_variants[variant]
            columns = model.feature_columns(train_frame)
            if columns != model.feature_columns(validation_frame):
                raise RuntimeError(f"Exp081 {variant} train/validation feature ABI differs")
            estimator = model.build_xgb(seed, labels_train)
            estimator.fit(train_frame.loc[:, columns], labels_train)
            probabilities = estimator.predict_proba(
                validation_frame.loc[:, columns]
            )[:, 1]
            model_path = root / "models" / f"variant_{variant}_seed{seed}.joblib"
            _atomic_joblib(model_path, estimator)
            model_paths.append(model_path)
            metrics = model.fixed_metrics(labels_validation, probabilities)
            metric_rows.append(
                {
                    "split": "validation",
                    "seed": seed,
                    "variant": variant,
                    "n": len(labels_validation),
                    "threshold": 0.5,
                    **metrics,
                }
            )
            prediction_rows.append(
                pd.DataFrame(
                    {
                        "sequence_id": validation_frame["sequence_id"].astype(str),
                        "label": labels_validation,
                        "seed": seed,
                        "variant": variant,
                        "probability": probabilities,
                        "prediction": (probabilities >= 0.5).astype(int),
                    }
                )
            )
            _log(
                f"validate complete variant={variant} seed={seed} "
                f"f1={metrics['f1']:.6f} pr_auc={metrics['pr_auc']:.6f}"
            )
    per_seed = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    summary = _summary(per_seed)
    comparisons = (("A", "B"), ("A", "C"), ("C", "D"), ("A", "D"))
    delta_rows: list[dict[str, Any]] = []
    transition_rows: list[pd.DataFrame] = []
    indexed_metrics = per_seed.set_index(["seed", "variant"])
    for seed in contract.load_registration()["seeds"]:
        seed_predictions = predictions.loc[predictions["seed"].eq(seed)]
        for left, right in comparisons:
            delta_rows.append(
                {
                    "seed": seed,
                    "comparison": f"{left}_to_{right}",
                    **{
                        f"delta_{metric_name}": float(
                            indexed_metrics.loc[(seed, right), metric_name]
                            - indexed_metrics.loc[(seed, left), metric_name]
                        )
                        for metric_name in (
                            "accuracy",
                            "precision",
                            "recall",
                            "f1",
                            "roc_auc",
                            "pr_auc",
                            "brier",
                            "log_loss",
                            "high_confidence_errors",
                        )
                    },
                }
            )
            left_rows = seed_predictions.loc[
                seed_predictions["variant"].eq(left)
            ].reset_index(drop=True)
            right_rows = seed_predictions.loc[
                seed_predictions["variant"].eq(right)
            ].reset_index(drop=True)
            transition_rows.append(
                model.paired_transitions(
                    left_rows["sequence_id"].to_numpy(),
                    left_rows["label"].to_numpy(),
                    left_rows["probability"].to_numpy(),
                    right_rows["probability"].to_numpy(),
                    left_name=left,
                    right_name=right,
                ).assign(seed=seed, comparison=f"{left}_to_{right}")
            )
    parity = _a_parity(per_seed)
    if parity["status"] != "pass":
        raise RuntimeError("Exp081 Variant A failed exact Exp077 validation parity")
    gate = model.validation_gate(per_seed)
    outputs = {
        "per_seed": root / "metrics/validation_per_seed_results.csv",
        "summary": root / "metrics/validation_summary_results.csv",
        "predictions": root / "predictions/validation_all_seeds.csv",
        "parity": root / "metrics/variant_A_exp077_parity.json",
        "gate": root / "metrics/validation_gate.json",
        "model_manifest": root / "metrics/model_manifest_pre_gate.json",
        "paired_deltas": root / "metrics/validation_paired_deltas.csv",
        "transitions": root / "metrics/validation_error_transitions.csv",
    }
    _atomic_csv(outputs["per_seed"], per_seed)
    _atomic_csv(outputs["summary"], summary)
    _atomic_csv(outputs["predictions"], predictions)
    _atomic_csv(outputs["paired_deltas"], pd.DataFrame(delta_rows))
    _atomic_csv(outputs["transitions"], pd.concat(transition_rows, ignore_index=True))
    contract.atomic_write_json(outputs["parity"], parity)
    contract.atomic_write_json(outputs["gate"], gate)
    contract.atomic_write_json(
        outputs["model_manifest"],
        {
            "frozen_before_test_access": True,
            "test_accessed": False,
            "models": [contract.artifact_record(path) for path in model_paths],
        },
    )
    contract.publish_phase(
        root,
        "validate",
        [*model_paths, *outputs.values()],
        test_accessed=False,
        gate_status=gate["gate_status"],
    )
    _log(f"validation gate: {gate['gate_status']}")
    return gate


def run_extract_test(root: Path | str = EXPERIMENT_ROOT) -> dict[str, Any]:
    root = Path(root)
    try:
        marker = contract.authenticate_phase(root, "validate")
    except RuntimeError as exc:
        raise RuntimeError("Exp081 extract-test requires authenticated validate phase") from exc
    if marker.get("gate_status") != "advance" or marker.get("test_accessed") is not False:
        raise RuntimeError("Exp081 validate gate did not authorize test extraction")
    anchor = jaad_data.load_anchor("test", allow_test=True)
    identities = extraction.load_test_identity_rows()
    if not np.array_equal(
        identities["sequence_id"].astype(str).to_numpy(),
        anchor["sequence_id"].astype(str).to_numpy(),
    ):
        raise RuntimeError("Exp081 test extraction identities differ from the anchor")
    extractor = extraction.create_twinlite_extractor(device="cuda", half=True)
    outputs = extraction.run_test_extraction(root, extractor, identities)
    contract.publish_phase(
        root,
        "extract-test",
        outputs,
        test_accessed=True,
        gate_status="advance",
    )
    result = {"status": "complete", "rows": len(anchor), "test_accessed": True}
    _log(f"test extraction complete: rows={len(anchor)}")
    return result


def _authenticate_frozen_models(root: Path) -> dict[str, dict[str, Any]]:
    path = root / "metrics/model_manifest_pre_gate.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise RuntimeError("Exp081 frozen model manifest is missing") from exc
    records: dict[str, dict[str, Any]] = {}
    for expected in manifest.get("models", []):
        actual = contract.artifact_record(expected["path"])
        if (
            actual["sha256"] != expected["sha256"]
            or actual["size_bytes"] != expected["size_bytes"]
        ):
            raise RuntimeError(f"Exp081 frozen model changed: {expected['path']}")
        records[Path(expected["path"]).name] = expected
    if len(records) != 12:
        raise RuntimeError("Exp081 frozen model manifest does not contain 12 models")
    return records


def run_evaluate(root: Path | str = EXPERIMENT_ROOT) -> dict[str, Any]:
    root = Path(root)
    marker = contract.authenticate_phase(root, "extract-test")
    if marker.get("gate_status") != "advance" or marker.get("test_accessed") is not True:
        raise RuntimeError("Exp081 evaluate requires authenticated test extraction")
    before = _authenticate_frozen_models(root)
    anchor = jaad_data.load_anchor("test", allow_test=True)
    context = extraction.load_context(
        root,
        "test",
        anchor["sequence_id"].astype(str).to_numpy(),
        manifest_name="test_extraction_manifest.json",
    )
    variants = model.assemble_variants(anchor, context)
    labels = anchor["label"].to_numpy(int)
    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[pd.DataFrame] = []
    by_seed: dict[tuple[int, str], np.ndarray] = {}
    for seed in contract.load_registration()["seeds"]:
        for variant in ("A", "D"):
            model_path = root / "models" / f"variant_{variant}_seed{seed}.joblib"
            estimator = joblib.load(model_path)
            frame = variants[variant]
            probabilities = estimator.predict_proba(
                frame.loc[:, model.feature_columns(frame)]
            )[:, 1]
            by_seed[(seed, variant)] = probabilities
            metrics = model.fixed_metrics(labels, probabilities)
            metric_rows.append(
                {
                    "split": "test",
                    "seed": seed,
                    "variant": variant,
                    "n": len(labels),
                    "threshold": 0.5,
                    **metrics,
                }
            )
            prediction_rows.append(
                pd.DataFrame(
                    {
                        "sequence_id": frame["sequence_id"].astype(str),
                        "label": labels,
                        "seed": seed,
                        "variant": variant,
                        "probability": probabilities,
                        "prediction": (probabilities >= 0.5).astype(int),
                    }
                )
            )
            _log(
                f"test evaluated variant={variant} seed={seed} "
                f"f1={metrics['f1']:.6f} pr_auc={metrics['pr_auc']:.6f}"
            )
    per_seed = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    transitions = pd.concat(
        [
            model.paired_transitions(
                anchor["sequence_id"].astype(str).to_numpy(),
                labels,
                by_seed[(seed, "A")],
                by_seed[(seed, "D")],
                left_name="A",
                right_name="D",
            ).assign(seed=seed)
            for seed in contract.load_registration()["seeds"]
        ],
        ignore_index=True,
    )
    outputs = {
        "per_seed": root / "metrics/test_per_seed_results.csv",
        "summary": root / "metrics/test_summary_results.csv",
        "predictions": root / "predictions/test_A_D_all_seeds.csv",
        "transitions": root / "metrics/test_A_to_D_transitions.csv",
    }
    _atomic_csv(outputs["per_seed"], per_seed)
    _atomic_csv(outputs["summary"], _summary(per_seed))
    _atomic_csv(outputs["predictions"], predictions)
    _atomic_csv(outputs["transitions"], transitions)
    after = _authenticate_frozen_models(root)
    if before != after:
        raise RuntimeError("Exp081 model manifest changed during test evaluation")
    contract.publish_phase(
        root,
        "evaluate",
        outputs.values(),
        test_accessed=True,
        evaluated_variants=["A", "D"],
    )
    return {"status": "complete", "variants": ["A", "D"]}


def _test_named_artifacts(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and "test" in path.name.lower()
    )


def run_latency(root: Path | str = EXPERIMENT_ROOT) -> Path:
    root = Path(root)
    contract.authenticate_phase(root, "validate")
    extractor = extraction.create_twinlite_extractor(device="cuda", half=True)
    rows = latency_module.benchmark_training_window(root, extractor)
    path = root / "metrics/inference_latency.csv"
    proxy_path = root / "metrics/pie_visual_proxy_telemetry_audit.csv"
    sample_path = root / "metrics/pie_visual_proxy_telemetry_sample.csv"
    _atomic_csv(path, rows)
    proxy_audit, proxy_sample = latency_module.pie_visual_proxy_telemetry_audit()
    _atomic_csv(proxy_path, proxy_audit)
    _atomic_csv(sample_path, proxy_sample)
    contract.publish_phase(
        root, "latency", [path, proxy_path, sample_path], test_accessed=False
    )
    _log("latency and train/validation-only PIE proxy audit complete")
    return path


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = list(map(str, frame.columns))
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    lines.extend(
        "| " + " | ".join(map(str, row)) + " |"
        for row in frame.itertuples(index=False, name=None)
    )
    return "\n".join(lines)


def report_text(bundle: dict[str, Any]) -> str:
    gate_status = str(bundle["gate_status"])
    validation = bundle["validation_summary"]
    lines = [
        "# Exp081 Final Report",
        "",
        "Exp081 evaluates an image-derived visual ego-motion proxy. It is not physical ego speed and cannot be interpreted as measured vehicle telemetry.",
        "",
        "## Protocol",
        "",
        "Clean JAAD behavior-pedestrian benchmark; 16 observed frames; TTE 30--60; fixed threshold 0.5; seeds 42/43/44.",
        "",
        "## Validation",
        "",
        _markdown_table(validation),
        "",
        f"Frozen validation decision: `{gate_status}`.",
    ]
    if bundle.get("test_summary") is not None:
        lines.extend(
            ["", "## Test", "", _markdown_table(bundle["test_summary"])]
        )
    if bundle.get("latency") is not None:
        lines.extend(
            [
                "",
                "## Efficiency",
                "",
                _markdown_table(bundle["latency"]),
                "",
                "The incremental timing excludes frame decode and ViTPose/anchor extraction; it is not a complete end-to-end claim.",
            ]
        )
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "JAAD test was historically exposed in separate repository work. Exp081 froze its design and validation gate without accessing that cohort during development.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_report(root: Path | str = EXPERIMENT_ROOT) -> Path:
    root = Path(root)
    marker = contract.authenticate_phase(root, "validate")
    if marker.get("gate_status") == "no_advance":
        forbidden = _test_named_artifacts(root)
        if forbidden:
            raise RuntimeError(f"Exp081 no-advance has forbidden test artifact: {forbidden[0]}")
    validation = pd.read_csv(root / "metrics/validation_summary_results.csv")
    latency_path = root / "metrics/inference_latency.csv"
    test_path = root / "metrics/test_summary_results.csv"
    has_test = marker.get("gate_status") == "advance" and test_path.is_file()
    bundle = {
        "gate_status": marker.get("gate_status"),
        "validation_summary": validation,
        "latency": pd.read_csv(latency_path) if latency_path.is_file() else None,
        "test_summary": pd.read_csv(test_path) if has_test else None,
    }
    report = root / "FINAL_REPORT.md"
    contract.atomic_write_text(report, report_text(bundle))
    outputs = [report]
    terminal: Path | None = None
    if marker.get("gate_status") == "no_advance":
        terminal = root / "VALIDATION_ONLY_COMPLETE"
        contract.atomic_write_text(terminal, "validation-only negative result\n")
    elif has_test:
        contract.authenticate_phase(root, "evaluate")
        terminal = root / "EXPERIMENT_COMPLETE"
        contract.atomic_write_text(terminal, "complete\n")
    if terminal is not None:
        outputs.append(terminal)
    contract.publish_phase(
        root,
        "report",
        outputs,
        test_accessed=has_test,
        gate_status=marker.get("gate_status"),
    )
    return report


def run_verify(root: Path | str = EXPERIMENT_ROOT) -> dict[str, Any]:
    root = Path(root)
    phases = ["audit", "extract-development", "prepare", "validate"]
    validate = contract.authenticate_phase(root, "validate")
    if (root / "metrics/.latency.complete.json").is_file():
        phases.append("latency")
    if validate.get("gate_status") == "advance" and (
        root / "metrics/.evaluate.complete.json"
    ).is_file():
        phases.extend(["extract-test", "evaluate"])
    if (root / "metrics/.report.complete.json").is_file():
        phases.append("report")
    authenticated = [contract.authenticate_phase(root, phase)["phase"] for phase in phases]
    if validate.get("gate_status") == "no_advance":
        forbidden = _test_named_artifacts(root)
        if forbidden:
            raise RuntimeError(f"Exp081 no-advance has forbidden test artifact: {forbidden[0]}")
    result = {
        "status": "pass",
        "gate_status": validate.get("gate_status"),
        "authenticated_phases": authenticated,
    }
    path = root / "metrics/verification.json"
    contract.atomic_write_json(path, result)
    contract.publish_phase(
        root,
        "verify",
        [path],
        test_accessed="evaluate" in authenticated,
        gate_status=validate.get("gate_status"),
    )
    return result


def run_verify_gate(root: Path | str = EXPERIMENT_ROOT) -> str:
    root = Path(root)
    marker = contract.authenticate_phase(root, "validate")
    _authenticate_frozen_models(root)
    status = str(marker.get("gate_status"))
    if status not in {"advance", "no_advance"}:
        raise RuntimeError("Exp081 validation marker has no frozen gate decision")
    print(status, flush=True)
    return status


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase",
        choices=(
            "audit",
            "extract-development",
            "prepare",
            "validate",
            "extract-test",
            "evaluate",
            "latency",
            "report",
            "verify",
            "verify-gate",
        ),
        help="Run exactly one authenticated Exp081 phase.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    functions = {
        "audit": run_audit,
        "extract-development": run_extract_development,
        "prepare": run_prepare,
        "validate": run_validate,
        "extract-test": run_extract_test,
        "evaluate": run_evaluate,
        "latency": run_latency,
        "report": run_report,
        "verify": run_verify,
        "verify-gate": run_verify_gate,
    }
    if args.phase not in functions:
        raise RuntimeError(f"Exp081 phase is not implemented yet: {args.phase}")
    functions[args.phase]()


if __name__ == "__main__":
    main()
