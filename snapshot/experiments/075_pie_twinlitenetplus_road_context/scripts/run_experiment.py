"""Fail-closed phase runner for Exp075."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import contract
import context


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]


def _load_local_module(name: str):
    path = SCRIPT_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_exp075_{name}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Exp075 module: {name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    sys.path.insert(0, str(SCRIPT_DIR))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(SCRIPT_DIR))
    return module


def base_feature_columns(frame: pd.DataFrame) -> list[str]:
    audit = json.loads(
        contract.resolve_source("exp072_cohort_audit").read_text(encoding="utf-8")
    )
    columns = list(map(str, audit.get("feature_columns", [])))
    if (
        len(columns) != 187
        or len(columns) != len(set(columns))
        or set(columns).difference(frame.columns)
    ):
        raise RuntimeError("Exp072 base feature contract changed")
    return columns


def join_base_and_context(
    base: pd.DataFrame,
    road: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    road_columns = list(context.context_columns())
    required = {"sequence_id", *road_columns}
    if required.difference(road.columns) or len(road) != len(base):
        raise RuntimeError("base/context population mismatch")
    base_ids = base["sequence_id"].astype(str).to_numpy()
    road_ids = road["sequence_id"].astype(str).to_numpy()
    if not np.array_equal(base_ids, road_ids):
        raise RuntimeError("base/context sequence order mismatch")
    context_values = road.loc[:, road_columns].to_numpy(dtype=np.float32)
    if not np.isfinite(context_values).all():
        raise RuntimeError("road context contains non-finite values")
    prepared = base.reset_index(drop=True).copy()
    prepared.loc[:, road_columns] = context_values
    features = base_feature_columns(prepared) + road_columns
    if len(features) != 247 or len(features) != len(set(features)):
        raise RuntimeError("Exp075 requires exactly 247 unique features")
    return prepared, features


def completion_path(root: Path, phase: str) -> Path:
    return Path(root) / "metrics" / f".{phase}.complete.json"


def publish_phase(
    root: Path,
    phase: str,
    outputs: Sequence[Path],
    **extra: Any,
) -> Path:
    marker = completion_path(root, phase)
    records = {
        os.path.relpath(Path(path), marker.parent): contract.artifact_record(Path(path))
        for path in outputs
    }
    if not records:
        raise ValueError("phase completion requires at least one output")
    contract.atomic_json(
        marker,
        {"phase": phase, "status": "complete", "outputs": records, **extra},
    )
    return marker


def authenticate_phase(root: Path, phase: str) -> dict[str, Any]:
    marker = completion_path(root, phase)
    try:
        value = json.loads(marker.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise RuntimeError(f"missing authenticated {phase} completion") from error
    if (
        value.get("phase") != phase
        or value.get("status") != "complete"
        or not value.get("outputs")
    ):
        raise RuntimeError(f"invalid {phase} completion")
    for relative, expected in value["outputs"].items():
        path = marker.parent / relative
        if not path.is_file() or contract.artifact_record(path) != expected:
            raise RuntimeError(f"{phase} output authentication failed: {path}")
    return value


def require_primary_cuda():
    """Refuse to publish primary perception artifacts without visible CUDA."""

    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("primary extraction requires a CUDA-visible GPU")
    return torch.device("cuda:0")


def run_audit(root: Path = EXPERIMENT_ROOT) -> dict[str, Any]:
    """Authenticate source/checkpoint and publish a CPU-safe audit."""

    import cv2
    import torch

    twinlite_adapter = _load_local_module("twinlite_adapter")

    contract.load_registration()
    source_root = contract.resolve_source("twinlite_model").parents[1]
    commit = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    expected_commit = contract.EXPECTED_REGISTRATION["twinlite"]["commit"]
    if commit != expected_commit:
        raise RuntimeError("TwinLiteNet submodule commit drifted")
    model, checkpoint_audit = twinlite_adapter.load_large_model(map_location="cpu")
    del model
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
    }
    audit = {
        "source_commit": commit,
        "checkpoint": checkpoint_audit,
        "checkpoint_artifact": contract.artifact_record(
            contract.resolve_source("twinlite_checkpoint")
        ),
        "paper_artifact": contract.artifact_record(
            contract.resolve_source("twinlite_paper")
        ),
        "strict_load": True,
        "primary_cache_published": False,
    }
    root = Path(root)
    environment_path = root / "metrics/environment.json"
    audit_path = root / "metrics/provenance_audit.json"
    contract.atomic_json(environment_path, environment)
    contract.atomic_json(audit_path, audit)
    publish_phase(root, "audit", [environment_path, audit_path])
    return audit


def run_extract_development(root: Path = EXPERIMENT_ROOT) -> dict[str, Any]:
    """Run primary train/validation extraction only after audit and CUDA gates."""

    extract_development = _load_local_module("extract_development")
    twinlite_adapter = _load_local_module("twinlite_adapter")

    authenticate_phase(root, "audit")
    device = require_primary_cuda()
    extractor = twinlite_adapter.TwinLiteExtractor(
        device=device,
        half=True,
    )
    train = extract_development.load_identity_rows("train")
    required = extract_development.required_frames(train, allowed_splits=("train",))
    set_id, video_id = next(iter(required))
    frame = next(iter(required[(set_id, video_id)]))
    image = extract_development.decode_exact_frame(set_id, video_id, frame)
    drivable, lane = extractor.predict([image])
    if (
        len(drivable) != 1
        or len(lane) != 1
        or drivable[0].shape != image.shape[:2]
        or lane[0].shape != image.shape[:2]
        or drivable[0].dtype != np.bool_
        or lane[0].dtype != np.bool_
    ):
        raise RuntimeError("CUDA smoke inference violated the mask ABI")
    result = extract_development.extract_development(extractor, output_root=root)
    publish_phase(
        root,
        "extract-development",
        list(map(Path, result["outputs"])),
        set03_accessed=False,
        target_labels_passed_to_perception=False,
    )
    return result


def run_prepare(root: Path = EXPERIMENT_ROOT) -> dict[str, Any]:
    """Join authenticated Exp075 context to exact Exp072 development rows."""

    authenticate_phase(root, "extract-development")
    root = Path(root)
    outputs: list[Path] = []
    audits: dict[str, object] = {}
    feature_columns: list[str] | None = None
    for split, source in (
        ("train", "exp072_prepared_train"),
        ("validation", "exp072_prepared_validation"),
    ):
        base = pd.read_csv(contract.resolve_source(source))
        road, audit = context.load_context(
            root, split, base["sequence_id"].astype(str).to_numpy()
        )
        prepared, features = join_base_and_context(base, road)
        if feature_columns is None:
            feature_columns = features
        elif features != feature_columns:
            raise RuntimeError("train/validation Exp075 feature order differs")
        expected_rows = int(contract.EXPECTED_REGISTRATION["cohorts"][split]["rows"])
        if len(prepared) != expected_rows:
            raise RuntimeError(f"Exp075 {split} row count drifted")
        output = root / "metrics" / f"prepared_{split}.csv"
        contract.atomic_csv(output, prepared)
        outputs.append(output)
        audits[split] = audit
    provenance = {
        "base_experiment": "Exp072",
        "base_features": 187,
        "context_features": 60,
        "total_features": 247,
        "feature_columns": feature_columns,
        "context_columns": list(context.context_columns()),
        "road_equation_source": contract.artifact_record(
            contract.resolve_source("exp063_representations")
        ),
        "checkpoint": contract.artifact_record(
            contract.resolve_source("twinlite_checkpoint")
        ),
        "source_commit": contract.EXPECTED_REGISTRATION["twinlite"]["commit"],
        "splits": audits,
        "set03_accessed": False,
    }
    provenance_path = root / "metrics/context_provenance.json"
    contract.atomic_json(provenance_path, provenance)
    outputs.append(provenance_path)
    current_coverage = pd.read_csv(
        _phase_output(root, "extract-development", "context_coverage.csv")
    )
    original_coverage = json.loads(
        contract.resolve_source("exp074_context_coverage").read_text(encoding="utf-8")
    )
    comparison = context.compare_development_coverage(
        current_coverage, original_coverage
    )
    comparison_path = root / "metrics/context_coverage_vs_exp074.csv"
    contract.atomic_csv(comparison_path, comparison)
    outputs.append(comparison_path)
    publish_phase(root, "prepare", outputs, set03_accessed=False)
    return provenance


def _phase_output(root: Path, phase: str, suffix: str) -> Path:
    completion = authenticate_phase(root, phase)
    matches = [relative for relative in completion["outputs"] if relative.endswith(suffix)]
    if len(matches) != 1:
        raise RuntimeError(f"expected one {phase} output ending in {suffix}")
    return completion_path(root, phase).parent / matches[0]


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


def _reference_predictions(
    experiment: str,
    split: str = "validation",
) -> pd.DataFrame:
    if experiment == "exp072":
        source = (
            "exp072_validate_completion"
            if split == "validation"
            else "exp072_evaluate_completion"
        )
    elif experiment == "exp074" and split == "validation":
        source = "exp074_validate_completion"
    else:
        raise ValueError("requested reference predictions do not exist")
    marker, completion = _parent_completion(source)
    parts = []
    for seed in (42, 43, 44):
        prefix = "validation" if split == "validation" else "test"
        parts.append(
            pd.read_csv(
                _parent_output(
                    marker,
                    completion,
                    f"{prefix}_candidate_seed{seed}.csv",
                )
            )
        )
    return pd.concat(parts, ignore_index=True)


def _align_predictions(candidate: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    keys = ["seed", "sequence_id", "set_id", "track_id", "video_id", "label"]
    merged = candidate.merge(
        reference.loc[:, keys + ["probability"]].rename(
            columns={"probability": "anchor_probability"}
        ),
        on=keys,
        how="inner",
        validate="one_to_one",
        sort=False,
    )
    if len(merged) != len(candidate) or len(merged) != len(reference):
        raise RuntimeError("Exp075/reference predictions are not exactly aligned")
    return merged


def _metric_tables(model_module, aligned: pd.DataFrame, reference_name: str):
    rows = []
    deltas = []
    for seed, group in aligned.groupby("seed", sort=True):
        labels = group["label"].to_numpy(dtype=int)
        candidate = model_module.compute_metrics(
            labels, group["probability"].to_numpy(dtype=float)
        )
        reference = model_module.compute_metrics(
            labels, group["anchor_probability"].to_numpy(dtype=float)
        )
        rows.extend(
            [
                {"seed": seed, "model": "candidate", **candidate},
                {"seed": seed, "model": reference_name, **reference},
            ]
        )
        delta = {"seed": seed, "reference": reference_name}
        for metric_name in (
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
        ):
            delta[f"delta_{metric_name}"] = float(
                candidate[metric_name] - reference[metric_name]
            )
        deltas.append(delta)
    return pd.DataFrame(rows), pd.DataFrame(deltas)


def _summary(per_seed: pd.DataFrame) -> pd.DataFrame:
    excluded = {"seed", "model"}
    metrics = [column for column in per_seed.columns if column not in excluded]
    rows = []
    for name, group in per_seed.groupby("model", sort=True):
        row: dict[str, Any] = {"model": name, "seed_count": len(group)}
        for metric_name in metrics:
            if pd.api.types.is_numeric_dtype(group[metric_name]):
                row[f"{metric_name}_mean"] = float(group[metric_name].mean())
                row[f"{metric_name}_std"] = float(group[metric_name].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def _atomic_joblib(value: Any, path: Path) -> None:
    import joblib

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        joblib.dump(value, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def run_validate(root: Path = EXPERIMENT_ROOT) -> dict[str, Any]:
    """Fit all frozen seeds, then inspect validation and apply the preregistered gate."""

    authenticate_phase(root, "prepare")
    root = Path(root)
    model_module = _load_local_module("model")
    train = pd.read_csv(_phase_output(root, "prepare", "prepared_train.csv"))
    provenance = json.loads(
        _phase_output(root, "prepare", "context_provenance.json").read_text(
            encoding="utf-8"
        )
    )
    features = list(map(str, provenance["feature_columns"]))
    if len(features) != 247 or set(features).difference(train.columns):
        raise RuntimeError("prepared Exp075 training feature contract changed")
    labels = train["label"].to_numpy(dtype=int)
    models: dict[int, Any] = {}
    model_records = []
    for seed in model_module.SEEDS:
        pipeline = model_module.build_xgb(seed, labels)
        pipeline.fit(train.loc[:, features], labels)
        path = root / "models" / f"candidate_seed{seed}.joblib"
        _atomic_joblib(pipeline, path)
        models[seed] = pipeline
        model_records.append({"seed": seed, "path": str(path), **contract.artifact_record(path)})
    model_manifest = {
        "models": model_records,
        "feature_columns": features,
        "feature_count": len(features),
        "validation_predictions_read": False,
    }
    model_manifest_path = root / "metrics/model_manifest_pre_gate.json"
    contract.atomic_json(model_manifest_path, model_manifest)
    complexity_path = root / "metrics/model_complexity.csv"
    contract.atomic_csv(
        complexity_path,
        pd.DataFrame(
            [
                {
                    "seed": record["seed"],
                    "feature_count": len(features),
                    "boosted_rounds": int(
                        models[int(record["seed"])]
                        .named_steps["model"]
                        .get_booster()
                        .num_boosted_rounds()
                    ),
                    "model_artifact_bytes": record["size_bytes"],
                }
                for record in model_records
            ]
        ),
    )

    validation = pd.read_csv(
        _phase_output(root, "prepare", "prepared_validation.csv")
    )
    candidate_parts = []
    prediction_paths = []
    identities = ["sequence_id", "set_id", "track_id", "video_id", "label"]
    for seed in model_module.SEEDS:
        probabilities = models[seed].predict_proba(validation.loc[:, features])[:, 1]
        predictions = validation.loc[:, identities].copy()
        predictions.insert(0, "seed", seed)
        predictions["probability"] = probabilities
        path = root / "predictions" / f"validation_candidate_seed{seed}.csv"
        contract.atomic_csv(path, predictions)
        prediction_paths.append(path)
        candidate_parts.append(predictions)
    candidate = pd.concat(candidate_parts, ignore_index=True)
    reference072 = _reference_predictions("exp072")
    reference074 = _reference_predictions("exp074")
    aligned072 = _align_predictions(candidate, reference072)
    aligned074 = _align_predictions(candidate, reference074)
    per_seed072, deltas072 = _metric_tables(model_module, aligned072, "reference")
    _, deltas074 = _metric_tables(model_module, aligned074, "exp074")
    transitions072 = model_module.error_transitions(aligned072)
    transitions074 = model_module.error_transitions(aligned074)
    counts = transitions072.groupby(["seed", "transition"], sort=True).size()
    corrections = float(
        np.mean(
            [counts.get((seed, "anchor_wrong_candidate_correct"), 0) for seed in model_module.SEEDS]
        )
    )
    harms = float(
        np.mean(
            [counts.get((seed, "anchor_correct_candidate_wrong"), 0) for seed in model_module.SEEDS]
        )
    )
    gate = model_module.validation_gate(
        per_seed072,
        corrections=corrections,
        harms=harms,
    )
    output_values = {
        "validation_per_seed_results.csv": per_seed072,
        "validation_summary_results.csv": _summary(per_seed072),
        "validation_deltas_vs_exp072.csv": deltas072,
        "validation_deltas_vs_exp074.csv": deltas074,
        "validation_error_transitions_vs_exp072.csv": transitions072,
        "validation_error_transitions_vs_exp074.csv": transitions074,
    }
    outputs = [model_manifest_path, complexity_path, *prediction_paths]
    for name, frame in output_values.items():
        path = root / "metrics" / name
        contract.atomic_csv(path, frame)
        outputs.append(path)
    gate_path = root / "metrics/validation_gate.json"
    contract.atomic_json(gate_path, gate)
    outputs.append(gate_path)
    publish_phase(root, "validate", outputs, gate_status=gate["status"])
    return gate


def require_validation_advance(root: Path = EXPERIMENT_ROOT) -> dict[str, Any]:
    """Authenticate the validation decision and frozen model hashes."""

    authenticate_phase(root, "validate")
    gate_path = _phase_output(root, "validate", "validation_gate.json")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("status") != "advance":
        raise RuntimeError("Exp075 validation did not advance to set03")
    manifest_path = _phase_output(root, "validate", "model_manifest_pre_gate.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if len(manifest.get("models", [])) != 3:
        raise RuntimeError("Exp075 frozen model manifest is incomplete")
    for record in manifest["models"]:
        path = Path(record["path"])
        expected = {"sha256": record["sha256"], "size_bytes": record["size_bytes"]}
        if not path.is_file() or contract.artifact_record(path) != expected:
            raise RuntimeError(f"Exp075 frozen model authentication failed: {path}")
    return {"gate": gate, "model_manifest": manifest}


def run_extract_test(root: Path = EXPERIMENT_ROOT) -> dict[str, Any]:
    """Run set03 perception only after authenticated validation advance."""

    require_validation_advance(root)
    device = require_primary_cuda()
    authenticate_phase(root, "extract-development")
    extraction_module = _load_local_module("extract_development")
    adapter = _load_local_module("twinlite_adapter")
    extractor = adapter.TwinLiteExtractor(device=device, half=True)
    result = extraction_module.extract_test(extractor, output_root=root)
    publish_phase(root, "extract-test", list(map(Path, result["outputs"])))
    return result


def run_evaluate(root: Path = EXPERIMENT_ROOT) -> dict[str, Any]:
    """Evaluate frozen models on set03 without fitting or threshold changes."""

    frozen = require_validation_advance(root)
    authenticate_phase(root, "extract-test")
    root = Path(root)
    model_module = _load_local_module("model")
    base = pd.read_csv(contract.resolve_source("exp072_prepared_test"))
    road, context_audit = context.load_context(
        root,
        "test",
        base["sequence_id"].astype(str).to_numpy(),
        manifest_name="test_extraction_manifest.json",
    )
    prepared, features = join_base_and_context(base, road)
    if features != frozen["model_manifest"]["feature_columns"]:
        raise RuntimeError("test feature order differs from frozen models")
    prepared_path = root / "metrics/prepared_test.csv"
    contract.atomic_csv(prepared_path, prepared)
    import joblib

    identities = ["sequence_id", "set_id", "track_id", "video_id", "label"]
    parts = []
    outputs: list[Path] = [prepared_path]
    model_after = []
    for record in frozen["model_manifest"]["models"]:
        seed = int(record["seed"])
        path = Path(record["path"])
        pipeline = joblib.load(path)
        probabilities = pipeline.predict_proba(prepared.loc[:, features])[:, 1]
        predictions = prepared.loc[:, identities].copy()
        predictions.insert(0, "seed", seed)
        predictions["probability"] = probabilities
        prediction_path = root / "predictions" / f"test_candidate_seed{seed}.csv"
        contract.atomic_csv(prediction_path, predictions)
        outputs.append(prediction_path)
        parts.append(predictions)
        model_after.append({"seed": seed, **contract.artifact_record(path)})
    candidate = pd.concat(parts, ignore_index=True)
    reference = _reference_predictions("exp072", split="test")
    aligned = _align_predictions(candidate, reference)
    per_seed, deltas = _metric_tables(model_module, aligned, "reference")
    transitions = model_module.error_transitions(aligned)
    frames = {
        "fixed_0p5_per_seed_results.csv": per_seed,
        "summary_results.csv": _summary(per_seed),
        "test_deltas_vs_exp072.csv": deltas,
        "test_error_transitions_vs_exp072.csv": transitions,
    }
    for name, frame in frames.items():
        path = root / "metrics" / name
        contract.atomic_csv(path, frame)
        outputs.append(path)
    after_path = root / "metrics/model_hashes_after_evaluation.json"
    contract.atomic_json(
        after_path,
        {"models": model_after, "unchanged": model_after == [
            {"seed": int(record["seed"]), "sha256": record["sha256"], "size_bytes": record["size_bytes"]}
            for record in frozen["model_manifest"]["models"]
        ]},
    )
    outputs.append(after_path)
    if not json.loads(after_path.read_text(encoding="utf-8"))["unchanged"]:
        raise RuntimeError("frozen model artifacts changed during evaluation")
    publish_phase(root, "evaluate", outputs, context_audit=context_audit)
    return {"per_seed": per_seed, "deltas": deltas}


def run_latency(root: Path = EXPERIMENT_ROOT) -> dict[str, Any]:
    """Benchmark cached prediction and online incremental 16-frame inference."""

    authenticate_phase(root, "validate")
    authenticate_phase(root, "extract-development")
    device = require_primary_cuda()
    root = Path(root)
    extraction = _load_local_module("extract_development")
    adapter = _load_local_module("twinlite_adapter")
    latency_module = _load_local_module("latency")
    import joblib
    import torch

    manifest = json.loads(
        _phase_output(root, "validate", "model_manifest_pre_gate.json").read_text(
            encoding="utf-8"
        )
    )
    record = next(value for value in manifest["models"] if int(value["seed"]) == 42)
    model_path = Path(record["path"])
    if contract.artifact_record(model_path) != {
        "sha256": record["sha256"],
        "size_bytes": record["size_bytes"],
    }:
        raise RuntimeError("latency model authentication failed")
    pipeline = joblib.load(model_path)
    features = list(map(str, manifest["feature_columns"]))
    train = pd.read_csv(_phase_output(root, "prepare", "prepared_train.csv"))
    row = train.sort_values("sequence_id", kind="stable").iloc[0]
    frames = tuple(range(int(row.start_frame), int(row.end_frame) + 1))
    images = [
        extraction.decode_exact_frame(str(row.set_id), str(row.video_id), frame)
        for frame in frames
    ]
    required = {frame: (str(row.ped_id),) for frame in frames}
    boxes = extraction.load_required_bboxes(
        str(row.set_id), str(row.video_id), required
    )
    bboxes = np.stack([(boxes[(str(row.ped_id), frame)]) for frame in frames])
    extractor = adapter.TwinLiteExtractor(device=device, half=True)
    parity = json.loads((root / "metrics/batch_parity.json").read_text(encoding="utf-8"))
    selected_batch = int(parity["selected_batch_size"])

    class NamedPredictor:
        def predict_proba(self, values):
            frame = pd.DataFrame(np.asarray(values), columns=features)
            return pipeline.predict_proba(frame)

    predictor = NamedPredictor()
    settings = contract.EXPECTED_REGISTRATION["latency"]
    online_settings = settings["online_incremental_window"]
    online = latency_module.benchmark_online_incremental(
        extractor,
        images,
        bboxes,
        row.loc[features[:187]].to_numpy(dtype=np.float32),
        predictor,
        batch_sizes=tuple(dict.fromkeys((1, selected_batch))),
        warmups=int(online_settings["warmups"]),
        repetitions=int(online_settings["repetitions"]),
        synchronize=torch.cuda.synchronize,
    )
    twin_settings = settings["twinlite"]
    perception = latency_module.benchmark_twinlite(
        extractor,
        images,
        inference_batch_size=selected_batch,
        warmups=int(twin_settings["warmups"]),
        repetitions=int(twin_settings["repetitions"]),
        synchronize=torch.cuda.synchronize,
    )
    drivable: list[np.ndarray] = []
    lane: list[np.ndarray] = []
    for start in range(0, 16, selected_batch):
        road_masks, lane_masks = extractor.predict(images[start : start + selected_batch])
        drivable.extend(road_masks)
        lane.extend(lane_masks)
    road_settings = settings["road_construction"]
    road_components = latency_module.benchmark_road_components(
        np.stack(drivable),
        np.stack(lane),
        bboxes,
        warmups=int(road_settings["warmups"]),
        repetitions=int(road_settings["repetitions"]),
    )
    cached = []
    for batch_size, key in ((1, "predictor_batch1"), (256, "predictor_batch256")):
        protocol = settings[key]
        source = train.loc[:, features].iloc[:batch_size].to_numpy(dtype=np.float32)
        cached.append(
            latency_module.benchmark_cached_predictor(
                predictor,
                source,
                batch_size=batch_size,
                warmups=int(protocol["warmups"]),
                repetitions=int(protocol["repetitions"]),
            )
        )
    rows = []
    for value in [*online, perception, *road_components, *cached]:
        value = value.copy()
        for key in ("included_stages", "excluded_stages"):
            if key in value:
                value[key] = json.dumps(value[key], separators=(",", ":"))
        value.update(
            {
                "seed": 42,
                "device": str(device),
                "precision": "float16"
                if value["component"] in {"online_incremental_window", "twinlitenet_masks"}
                else "cpu_float32",
            }
        )
        rows.append(value)
    latency_path = root / "metrics/inference_latency.csv"
    contract.atomic_csv(latency_path, pd.DataFrame(rows))
    scope = {
        "online_name": "online_incremental_window",
        "includes": settings["online_includes"],
        "excludes": settings["online_excludes"],
        "not_full_camera_to_decision": True,
        "observation_frames": 16,
        "sample_selection": "lexicographically_first_train_sequence",
        "sequence_id": str(row.sequence_id),
        "model_seed": 42,
    }
    scope_path = root / "metrics/latency_scope.json"
    contract.atomic_json(scope_path, scope)
    publish_phase(root, "latency", [latency_path, scope_path])
    return {"rows": rows, "scope": scope}


def terminal_marker_name(status: str, *, evaluated: bool) -> str:
    if status == "no_advance":
        if evaluated:
            raise RuntimeError("no-advance experiment has unexpected evaluation")
        return "VALIDATION_COMPLETE_NO_ADVANCE"
    if status == "advance":
        if not evaluated:
            raise RuntimeError("advanced experiment requires final evaluation")
        return "EXPERIMENT_COMPLETE"
    raise RuntimeError(f"unexpected validation status: {status}")


def evaluation_report_lines(
    summary: pd.DataFrame,
    deltas: pd.DataFrame,
    transitions: pd.DataFrame,
    latency: pd.DataFrame,
) -> list[str]:
    """Render compact final-test evidence without changing any result."""

    models = summary.set_index("model")
    if not {"candidate", "reference"}.issubset(models.index):
        raise RuntimeError("final summary is missing candidate/reference rows")
    required_deltas = {"delta_f1", "delta_pr_auc", "delta_roc_auc"}
    if not required_deltas.issubset(deltas.columns):
        raise RuntimeError("final delta table is incomplete")
    transition_counts = (
        transitions.groupby(["seed", "transition"], sort=True)
        .size()
        .rename("count")
        .reset_index()
    )
    candidate = models.loc["candidate"]
    reference = models.loc["reference"]
    f1_favorable = int((deltas["delta_f1"] > 0).sum())
    pr_favorable = int((deltas["delta_pr_auc"] > 0).sum())
    seed_count = len(deltas)
    interpretation = pd.DataFrame(
        [
            {
                "metric": metric,
                "candidate_mean": float(candidate[f"{metric}_mean"]),
                "reference_mean": float(reference[f"{metric}_mean"]),
                "candidate_minus_reference": float(
                    candidate[f"{metric}_mean"] - reference[f"{metric}_mean"]
                ),
            }
            for metric in ("f1", "pr_auc", "roc_auc")
        ]
    )
    return [
        "",
        "## Final Test Summary",
        "",
        "```csv",
        summary.to_csv(index=False).strip(),
        "```",
        "",
        "## Final Test Deltas Versus Exp072",
        "",
        "```csv",
        deltas.to_csv(index=False).strip(),
        "```",
        "",
        "## Final Test Error Transitions",
        "",
        "```csv",
        transition_counts.to_csv(index=False).strip(),
        "```",
        "",
        "## Inference Latency",
        "",
        "```csv",
        latency.to_csv(index=False).strip(),
        "```",
        "",
        "## Final Test Interpretation",
        "",
        "```csv",
        interpretation.to_csv(index=False).strip(),
        "```",
        "",
        f"- F1 favorable seeds: {f1_favorable}/{seed_count}",
        f"- PR-AUC favorable seeds: {pr_favorable}/{seed_count}",
    ]


def run_report(root: Path = EXPERIMENT_ROOT) -> dict[str, Any]:
    """Create a source-grounded terminal report for the allowed gate path."""

    authenticate_phase(root, "validate")
    authenticate_phase(root, "latency")
    root = Path(root)
    gate = json.loads(
        _phase_output(root, "validate", "validation_gate.json").read_text(
            encoding="utf-8"
        )
    )
    evaluated = completion_path(root, "evaluate").is_file()
    if gate["status"] == "advance":
        authenticate_phase(root, "evaluate")
    elif evaluated or completion_path(root, "extract-test").exists():
        raise RuntimeError("no-advance Exp075 contains forbidden set03 phase artifacts")
    marker_name = terminal_marker_name(gate["status"], evaluated=evaluated)
    validation = pd.read_csv(
        _phase_output(root, "validate", "validation_summary_results.csv")
    )
    delta072 = pd.read_csv(
        _phase_output(root, "validate", "validation_deltas_vs_exp072.csv")
    )
    delta074 = pd.read_csv(
        _phase_output(root, "validate", "validation_deltas_vs_exp074.csv")
    )
    latency_scope = json.loads(
        _phase_output(root, "latency", "latency_scope.json").read_text(
            encoding="utf-8"
        )
    )
    lines = [
        "# Exp075 Final Report",
        "",
        f"- Validation gate: `{gate['status']}`",
        f"- Terminal category: `{marker_name}`",
        "- Benchmark: PIE, 16 observed frames, TTE 30--60, overlap 0.6",
        "- Threshold: fixed 0.5",
        "- Predictor: exact Exp072 XGBoost recipe, seeds 42/43/44",
        "- Changed dimension: original YOLOP masks replaced by TwinLiteNet+ Large masks",
        "- Set03 evaluated: " + ("yes" if evaluated else "no"),
        "",
        "## Validation Summary",
        "",
        "```csv",
        validation.to_csv(index=False).strip(),
        "```",
        "",
        "## Paired Deltas Versus Exp072",
        "",
        "```csv",
        delta072.to_csv(index=False).strip(),
        "```",
        "",
        "## Paired Deltas Versus Exp074",
        "",
        "```csv",
        delta074.to_csv(index=False).strip(),
        "```",
    ]
    if evaluated:
        final_summary = pd.read_csv(
            _phase_output(root, "evaluate", "summary_results.csv")
        )
        final_deltas = pd.read_csv(
            _phase_output(root, "evaluate", "test_deltas_vs_exp072.csv")
        )
        final_transitions = pd.read_csv(
            _phase_output(root, "evaluate", "test_error_transitions_vs_exp072.csv")
        )
        latency = pd.read_csv(_phase_output(root, "latency", "inference_latency.csv"))
        lines.extend(
            evaluation_report_lines(
                final_summary, final_deltas, final_transitions, latency
            )
        )
    lines.extend(
        [
        "",
        "## Latency Scope",
        "",
        "The online incremental measurement includes "
        + ", ".join(latency_scope["includes"])
        + ". It excludes "
        + ", ".join(latency_scope["excludes"])
        + ", so it is not full camera-to-decision latency.",
        "",
        "## Interpretation Boundary",
        "",
        "Any Exp075-minus-Exp074 difference is evidence about the perception source under the same road representation. Advancement is determined only against Exp072. No architecture, threshold, representation, or gate was changed after validation.",
        ]
    )
    report_path = root / "FINAL_REPORT.md"
    contract.atomic_text(report_path, "\n".join(lines) + "\n")
    publish_phase(root, "report", [report_path], terminal_category=marker_name)
    return {"gate": gate, "evaluated": evaluated, "terminal_category": marker_name}


def run_verify(root: Path = EXPERIMENT_ROOT) -> dict[str, Any]:
    """Re-authenticate the full allowed path and create one terminal marker."""

    root = Path(root)
    phases = ["audit", "extract-development", "prepare", "validate", "latency", "report"]
    gate = json.loads(
        _phase_output(root, "validate", "validation_gate.json").read_text(
            encoding="utf-8"
        )
    )
    evaluated = gate["status"] == "advance"
    if evaluated:
        phases.extend(["extract-test", "evaluate"])
    for phase in phases:
        authenticate_phase(root, phase)
    if not evaluated:
        forbidden = [
            root / "cache/test_parts",
            root / "metrics/test_extraction_manifest.json",
            root / "metrics/prepared_test.csv",
            completion_path(root, "extract-test"),
            completion_path(root, "evaluate"),
        ]
        present = [str(path) for path in forbidden if path.exists()]
        if present:
            raise RuntimeError(f"no-advance path contains set03 artifacts: {present}")
    marker_name = terminal_marker_name(gate["status"], evaluated=evaluated)
    verification = {
        "status": "verified",
        "phases": phases,
        "validation_gate": gate["status"],
        "set03_evaluated": evaluated,
        "terminal_marker": marker_name,
        "checkpoint": contract.artifact_record(
            contract.resolve_source("twinlite_checkpoint")
        ),
        "source_commit": contract.EXPECTED_REGISTRATION["twinlite"]["commit"],
    }
    verification_path = root / "metrics/verification_report.json"
    contract.atomic_json(verification_path, verification)
    publish_phase(root, "verify", [verification_path])
    contract.atomic_text(root / marker_name, "verified\n")
    return verification


def main() -> None:
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
        ),
    )
    arguments = parser.parse_args()
    if arguments.phase == "audit":
        run_audit()
    elif arguments.phase == "extract-development":
        run_extract_development()
    elif arguments.phase == "prepare":
        run_prepare()
    elif arguments.phase == "validate":
        run_validate()
    elif arguments.phase == "extract-test":
        run_extract_test()
    elif arguments.phase == "evaluate":
        run_evaluate()
    elif arguments.phase == "latency":
        run_latency()
    elif arguments.phase == "report":
        run_report()
    else:
        run_verify()


if __name__ == "__main__":
    main()
