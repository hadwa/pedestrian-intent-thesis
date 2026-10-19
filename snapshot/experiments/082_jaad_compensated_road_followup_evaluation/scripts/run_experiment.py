"""Authenticated Exp082 validation-selected follow-up evaluation runner."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import joblib
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = SCRIPT_DIR.parent
ACTIVE_ROOT = EXPERIMENT_ROOT.parents[1]
PARENT_ROOT = ACTIVE_ROOT / "experiments/081_jaad_ego_compensated_road_motion"
REGISTRATION_PATH = EXPERIMENT_ROOT / "pre_registration.json"


def _read_json(path: Path | str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


EXPECTED_REGISTRATION = _read_json(REGISTRATION_PATH)


def load_registration() -> dict[str, Any]:
    registration = _read_json(REGISTRATION_PATH)
    if registration != EXPECTED_REGISTRATION:
        raise RuntimeError("Exp082 registration changed after import")
    return registration


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_record(path: Path | str) -> dict[str, Any]:
    artifact = Path(path).resolve()
    if not artifact.is_file():
        raise FileNotFoundError(artifact)
    return {
        "path": str(artifact),
        "size_bytes": artifact.stat().st_size,
        "sha256": sha256_file(artifact),
    }


def resolve_source(name: str) -> Path:
    source = load_registration()["sources"][name]
    path = Path(source["path"])
    if not path.is_absolute():
        path = ACTIVE_ROOT / path
    actual = artifact_record(path)
    if actual["size_bytes"] != source["size_bytes"] or actual["sha256"] != source["sha256"]:
        raise RuntimeError(f"Exp082 source authentication failed: {name}")
    return path.resolve()


def _atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_text(path: Path | str, value: str) -> None:
    _atomic_bytes(Path(path), value.encode("utf-8"))


def atomic_json(path: Path | str, value: Mapping[str, Any]) -> None:
    atomic_text(Path(path), json.dumps(value, indent=2, sort_keys=True) + "\n")


def atomic_csv(path: Path | str, value: pd.DataFrame) -> None:
    atomic_text(Path(path), value.to_csv(index=False, lineterminator="\n"))


def marker_path(root: Path | str, phase: str) -> Path:
    return Path(root) / "metrics" / f".{phase}.complete.json"


def publish_phase(
    root: Path | str,
    phase: str,
    outputs: Iterable[Path | str],
    *,
    test_accessed: bool,
) -> Path:
    marker = marker_path(root, phase)
    atomic_json(
        marker,
        {
            "phase": phase,
            "status": "complete",
            "test_accessed": test_accessed,
            "outputs": [artifact_record(path) for path in outputs],
        },
    )
    return marker


def authenticate_phase(root: Path | str, phase: str) -> dict[str, Any]:
    try:
        marker = _read_json(marker_path(root, phase))
        if marker.get("phase") != phase or marker.get("status") != "complete":
            raise RuntimeError("invalid phase state")
        for expected in marker.get("outputs", []):
            actual = artifact_record(expected["path"])
            if actual["sha256"] != expected["sha256"] or actual["size_bytes"] != expected["size_bytes"]:
                raise RuntimeError(f"changed output: {expected['path']}")
        return marker
    except (FileNotFoundError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise RuntimeError(f"Exp082 phase authentication failed: {phase}") from exc


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import authenticated module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _parent_modules():
    extraction = _load_module(resolve_source("exp081_extract"), "exp082_parent_extract")
    model = _load_module(resolve_source("exp081_model"), "exp082_parent_model")
    data = _load_module(resolve_source("exp081_jaad_data"), "exp082_parent_data")
    return extraction, model, data


def _log(message: str) -> None:
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"[{timestamp}] {message}", flush=True)


def _authenticate_parent_models() -> dict[tuple[str, int], Path]:
    manifest = _read_json(resolve_source("exp081_model_manifest"))
    if manifest.get("frozen_before_test_access") is not True or manifest.get("test_accessed") is not False:
        raise RuntimeError("Exp081 model manifest is not pre-test frozen")
    authenticated: dict[tuple[str, int], Path] = {}
    for expected in manifest.get("models", []):
        path = Path(expected["path"])
        actual = artifact_record(path)
        if actual["sha256"] != expected["sha256"] or actual["size_bytes"] != expected["size_bytes"]:
            raise RuntimeError(f"Exp081 model changed: {path}")
        parts = path.stem.split("_")
        variant = parts[1]
        seed = int(parts[2].replace("seed", ""))
        if variant in {"A", "D"}:
            authenticated[(variant, seed)] = path.resolve()
    expected_keys = {(variant, seed) for variant in ("A", "D") for seed in (42, 43, 44)}
    if set(authenticated) != expected_keys:
        raise RuntimeError("Exp082 did not authenticate exactly six A/D models")
    return authenticated


def _load_test_anchor(data_module: Any) -> pd.DataFrame:
    frame = pd.read_csv(resolve_source("test_anchor"))
    checked = data_module._validate_anchor(frame, "test")
    if len(checked) != load_registration()["test_rows"]:
        raise RuntimeError("Exp082 test anchor population changed")
    return checked


def align_identities(identities: pd.DataFrame, anchor_ids: np.ndarray) -> pd.DataFrame:
    """Require identical IDs, then restore the frozen anchor serialization order."""
    ids = identities["sequence_id"].astype(str)
    expected = np.asarray(anchor_ids).astype(str)
    if ids.duplicated().any() or len(set(expected)) != len(expected) or set(ids) != set(expected):
        raise RuntimeError("Exp082 split and anchor ID membership differ")
    indexed = identities.assign(sequence_id=ids).set_index("sequence_id", drop=False)
    return indexed.loc[expected].reset_index(drop=True)


def run_audit(root: Path | str = EXPERIMENT_ROOT) -> dict[str, Any]:
    root = Path(root)
    sources = {name: artifact_record(resolve_source(name)) for name in load_registration()["sources"]}
    parent_gate = _read_json(resolve_source("exp081_validation_gate"))
    if parent_gate.get("gate_status") != "no_advance":
        raise RuntimeError("Exp082 requires the preserved Exp081 no-advance gate")
    models = _authenticate_parent_models()
    audit = {
        "status": "pass",
        "experiment": load_registration()["experiment"],
        "selection_timing": load_registration()["selection_timing"],
        "parent_gate_status": parent_gate["gate_status"],
        "retraining": False,
        "authenticated_sources": sources,
        "authenticated_models": {
            f"{variant}{seed}": artifact_record(path)
            for (variant, seed), path in sorted(models.items())
        },
    }
    path = root / "metrics/provenance_audit.json"
    atomic_json(path, audit)
    publish_phase(root, "audit", [path], test_accessed=False)
    _log("audit complete: frozen Exp081 A/D models authenticated")
    return audit


def run_extract_test(root: Path | str = EXPERIMENT_ROOT) -> dict[str, Any]:
    root = Path(root)
    authenticate_phase(root, "audit")
    extraction, _, data = _parent_modules()
    anchor = _load_test_anchor(data)
    identities = pd.read_csv(resolve_source("test_split"), usecols=extraction.IDENTITY_COLUMNS)
    identities = extraction.validate_identity_rows(identities)
    identities = align_identities(
        identities, anchor["sequence_id"].astype(str).to_numpy()
    )
    extractor = extraction.create_twinlite_extractor(device="cuda", half=True)
    outputs = extraction.run_test_extraction(root, extractor, identities)
    publish_phase(root, "extract-test", outputs, test_accessed=True)
    _log(f"test extraction complete: {len(anchor)} exact windows")
    return {"status": "complete", "rows": len(anchor)}


METRIC_COLUMNS = (
    "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "brier",
    "log_loss", "tn", "fp", "fn", "tp", "high_confidence_fp",
    "high_confidence_fn", "high_confidence_errors",
)


def summarize_metrics(rows: pd.DataFrame) -> pd.DataFrame:
    metrics = [column for column in METRIC_COLUMNS if column in rows.columns]
    if not metrics:
        metrics = [
            column for column in rows.select_dtypes(include=[np.number]).columns
            if column not in {"seed", "n", "threshold"}
        ]
    records: list[dict[str, Any]] = []
    for variant, group in rows.groupby("variant", sort=True):
        record: dict[str, Any] = {"variant": variant, "seeds": len(group)}
        for metric in metrics:
            values = group[metric].to_numpy(float)
            record[f"{metric}_mean"] = float(values.mean())
            record[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        records.append(record)
    return pd.DataFrame(records)


def paired_deltas(rows: pd.DataFrame) -> pd.DataFrame:
    metrics = [column for column in METRIC_COLUMNS if column in rows.columns]
    if not metrics:
        metrics = [
            column for column in rows.select_dtypes(include=[np.number]).columns
            if column not in {"seed", "n", "threshold"}
        ]
    records: list[dict[str, Any]] = []
    for seed in sorted(rows["seed"].astype(int).unique()):
        indexed = rows.loc[rows["seed"].astype(int).eq(seed)].set_index("variant")
        if set(indexed.index) != {"A", "D"}:
            raise RuntimeError(f"Exp082 seed {seed} lacks exactly A and D")
        record: dict[str, Any] = {"seed": seed, "comparison": "D_minus_A"}
        for metric in metrics:
            record[f"delta_{metric}"] = float(indexed.loc["D", metric] - indexed.loc["A", metric])
        records.append(record)
    return pd.DataFrame(records)


def run_evaluate(root: Path | str = EXPERIMENT_ROOT) -> dict[str, Any]:
    root = Path(root)
    marker = authenticate_phase(root, "extract-test")
    if marker.get("test_accessed") is not True:
        raise RuntimeError("Exp082 evaluation requires authenticated test extraction")
    extraction, model, data = _parent_modules()
    models = _authenticate_parent_models()
    anchor = _load_test_anchor(data)
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
    transition_rows: list[pd.DataFrame] = []
    for seed in load_registration()["seeds"]:
        probabilities: dict[str, np.ndarray] = {}
        for variant in load_registration()["evaluated_variants"]:
            estimator = joblib.load(models[(variant, seed)])
            frame = variants[variant]
            probabilities[variant] = estimator.predict_proba(
                frame.loc[:, model.feature_columns(frame)]
            )[:, 1]
            values = model.fixed_metrics(labels, probabilities[variant])
            metric_rows.append(
                {"split": "test", "seed": seed, "variant": variant, "n": len(labels), "threshold": 0.5, **values}
            )
            prediction_rows.append(
                pd.DataFrame(
                    {
                        "sequence_id": frame["sequence_id"].astype(str),
                        "label": labels,
                        "seed": seed,
                        "variant": variant,
                        "probability": probabilities[variant],
                        "prediction": (probabilities[variant] >= 0.5).astype(int),
                    }
                )
            )
            _log(f"test {variant}{seed}: f1={values['f1']:.6f} pr_auc={values['pr_auc']:.6f}")
        transition_rows.append(
            model.paired_transitions(
                anchor["sequence_id"].astype(str).to_numpy(),
                labels,
                probabilities["A"],
                probabilities["D"],
                left_name="A",
                right_name="D",
            ).assign(seed=seed)
        )
    per_seed = pd.DataFrame(metric_rows)
    summary = summarize_metrics(per_seed)
    deltas = paired_deltas(per_seed)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    transitions = pd.concat(transition_rows, ignore_index=True)
    transition_summary = (
        transitions.groupby(["seed", "transition"], sort=True).size().rename("count").reset_index()
    )
    outputs = {
        "per_seed": root / "metrics/test_per_seed_results.csv",
        "summary": root / "metrics/test_summary_results.csv",
        "deltas": root / "metrics/test_paired_deltas.csv",
        "predictions": root / "predictions/test_A_D_all_seeds.csv",
        "transitions": root / "metrics/test_A_to_D_transitions.csv",
        "transition_summary": root / "metrics/test_A_to_D_transition_summary.csv",
    }
    for key, frame in (
        ("per_seed", per_seed), ("summary", summary), ("deltas", deltas),
        ("predictions", predictions), ("transitions", transitions),
        ("transition_summary", transition_summary),
    ):
        atomic_csv(outputs[key], frame)
    publish_phase(root, "evaluate", outputs.values(), test_accessed=True)
    return {"status": "complete", "variants": ["A", "D"]}


def _markdown(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in frame.itertuples(index=False, name=None))
    return "\n".join(lines)


def run_report(root: Path | str = EXPERIMENT_ROOT) -> Path:
    root = Path(root)
    authenticate_phase(root, "evaluate")
    summary = pd.read_csv(root / "metrics/test_summary_results.csv")
    deltas = pd.read_csv(root / "metrics/test_paired_deltas.csv")
    transition_summary = pd.read_csv(root / "metrics/test_A_to_D_transition_summary.csv")
    indexed = summary.set_index("variant")
    mean_delta_pr = float(indexed.loc["D", "pr_auc_mean"] - indexed.loc["A", "pr_auc_mean"])
    mean_delta_f1 = float(indexed.loc["D", "f1_mean"] - indexed.loc["A", "f1_mean"])
    mean_delta_roc = float(indexed.loc["D", "roc_auc_mean"] - indexed.loc["A", "roc_auc_mean"])
    report = "\n".join(
        [
            "# Exp082 Final Report",
            "",
            "This is a validation-selected follow-up evaluation. Exp081 remains `no_advance`; this report does not reinterpret its gate.",
            "",
            "## Protocol",
            "",
            "Clean JAAD; 1,501 test windows; 16 observed consecutive frames; TTE 30--60; fixed threshold 0.5; frozen Exp081 seeds 42/43/44; no retraining or retuning.",
            "",
            "## Test Results",
            "",
            _markdown(summary),
            "",
            f"Mean D-A deltas: F1 `{mean_delta_f1:+.6f}`, PR-AUC `{mean_delta_pr:+.6f}`, ROC-AUC `{mean_delta_roc:+.6f}`.",
            "",
            "## Per-seed D-A Deltas",
            "",
            _markdown(deltas),
            "",
            "## Fixed-0.5 Error Transitions",
            "",
            _markdown(transition_summary),
            "",
            "## Limitations",
            "",
            "The A/D comparison was selected after reviewing validation metrics, and JAAD test was historically exposed in earlier repository experiments. Treat this as transparent follow-up test evidence, not pristine external confirmation.",
            "",
        ]
    )
    path = root / "FINAL_REPORT.md"
    complete = root / "EXPERIMENT_COMPLETE"
    atomic_text(path, report)
    atomic_text(complete, "complete\n")
    publish_phase(root, "report", [path, complete], test_accessed=True)
    return path


def run_verify(root: Path | str = EXPERIMENT_ROOT) -> dict[str, Any]:
    root = Path(root)
    phases = ["audit", "extract-test", "evaluate", "report"]
    authenticated = [authenticate_phase(root, phase)["phase"] for phase in phases]
    models = _authenticate_parent_models()
    result = {
        "status": "pass",
        "test_accessed": True,
        "retraining": False,
        "authenticated_phases": authenticated,
        "authenticated_model_count": len(models),
    }
    path = root / "metrics/verification.json"
    atomic_json(path, result)
    publish_phase(root, "verify", [path], test_accessed=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("audit", "extract-test", "evaluate", "report", "verify"))
    args = parser.parse_args()
    phases = {
        "audit": run_audit,
        "extract-test": run_extract_test,
        "evaluate": run_evaluate,
        "report": run_report,
        "verify": run_verify,
    }
    phases[args.phase]()


if __name__ == "__main__":
    main()
