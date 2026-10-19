"""Resumable Exp063 preparation, training, and validation-only selection runner."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .contracts import ExperimentConfig, sha256_file
from .data import (
    ContextScaler,
    fit_context_scaler,
    load_trainval_context,
    transform_context,
)
from .extract_trainval import EXP062_ROOT, extract_trainval
from .model import HybridContextCorrector, predict_context_corrector


EXP_ROOT = Path(__file__).resolve().parents[1]
EXP062_CONTRACTS = importlib.import_module(
    "experiments.062_hybrid_tree_anchor_context_correction_skl152.scripts.contracts"
)

MetricBundle = EXP062_CONTRACTS.MetricBundle
ValidationGateDecision = EXP062_CONTRACTS.ValidationGateDecision
compute_binary_metrics = EXP062_CONTRACTS.compute_binary_metrics
evaluate_validation_gate = EXP062_CONTRACTS.evaluate_validation_gate
harmonic_f1_pr = EXP062_CONTRACTS.harmonic_f1_pr


def _exp062_runner():
    """Load training-only dependencies lazily outside the YOLOP runtime."""
    return importlib.import_module(
        "experiments.062_hybrid_tree_anchor_context_correction_skl152.scripts.runner"
    )


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, delete=False, encoding="utf-8"
    ) as handle:
        json.dump(dict(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)
    return path


def _atomic_csv(path: Path, frame: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, delete=False, encoding="utf-8", newline=""
    ) as handle:
        frame.to_csv(handle, index=False)
        temporary = Path(handle.name)
    os.replace(temporary, path)
    return path


def expected_run_keys() -> tuple[tuple[str, str, int], ...]:
    """Return the complete frozen 2x2x3 training grid in deterministic order."""
    config = ExperimentConfig()
    return tuple(
        (representation, anchor, seed)
        for representation in config.representations
        for anchor in config.anchor_families
        for seed in config.seeds
    )


def run_marker_name(representation: str, anchor: str, seed: int) -> str:
    return f"TRAINED__{representation}__{anchor}__seed{seed}.json"


def _scaler_path(representation: str, root: Path = EXP_ROOT) -> Path:
    return root / "metrics" / f"context_scaler_{representation}.json"


def _write_scaler(path: Path, scaler: ContextScaler, representation: str) -> Path:
    return _atomic_json(
        path,
        {
            "representation": representation,
            "fit_population": "training_rows_only",
            "ego_mean": scaler.ego_mean.tolist(),
            "ego_std": scaler.ego_std.tolist(),
            "scene_mean": scaler.scene_mean.tolist(),
            "scene_std": scaler.scene_std.tolist(),
        },
    )


def _load_scaler(path: Path, representation: str) -> ContextScaler:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("representation") != representation or payload.get(
        "fit_population"
    ) != "training_rows_only":
        raise RuntimeError("Exp063 context scaler identity mismatch")
    return ContextScaler(
        ego_mean=np.asarray(payload["ego_mean"], dtype=np.float32),
        ego_std=np.asarray(payload["ego_std"], dtype=np.float32),
        scene_mean=np.asarray(payload["scene_mean"], dtype=np.float32),
        scene_std=np.asarray(payload["scene_std"], dtype=np.float32),
    )


def _probability_vectors(anchor: str, seed: int):
    """Re-authenticate the completed Exp062 OOF/full-validation binding."""
    runner = _exp062_runner()
    spec = runner._bundle_spec(EXP062_ROOT, anchor, seed)
    grouped_oof = runner._load_grouped_oof(spec)
    vectors = runner.authenticate_probability_bundle(
        spec,
        grouped_oof=grouped_oof,
        expected_anchor=anchor,
        expected_seed=seed,
    )
    return spec, grouped_oof, vectors


def prepare(root: Path = EXP_ROOT) -> dict[str, Any]:
    """Bind extraction parts and fit one train-only scaler per representation."""
    marker = root / "PREPARED.json"
    if marker.exists():
        return json.loads(marker.read_text(encoding="utf-8"))
    extraction_manifest = root / "metrics/extraction_manifest.json"
    if not extraction_manifest.is_file():
        raise RuntimeError("Exp063 preparation requires completed corrected YOLOP extraction")
    extraction_payload = json.loads(extraction_manifest.read_text(encoding="utf-8"))
    if extraction_payload.get("set03_accessed") is not False or extraction_payload.get(
        "windows"
    ) != 5985:
        raise RuntimeError("Exp063 extraction manifest is incomplete or touched set03")
    artifacts: dict[str, str] = {
        "metrics/extraction_manifest.json": sha256_file(extraction_manifest)
    }
    for representation in ExperimentConfig().representations:
        data = load_trainval_context(representation)
        scaler = fit_context_scaler(data.ego, data.scene, data.train_mask)
        path = _write_scaler(_scaler_path(representation, root), scaler, representation)
        artifacts[str(path.relative_to(root))] = sha256_file(path)
    historical = EXP062_ROOT / "metrics/validation_per_seed_results.csv"
    if not historical.is_file():
        raise RuntimeError("Exp062 historical reference metrics are missing")
    artifacts["exp062_historical_validation_per_seed"] = sha256_file(historical)
    return json.loads(
        _atomic_json(
            marker,
            {
                "phase": "prepared",
                "set03_accessed": False,
                "fresh_runs": ExperimentConfig().fresh_run_count,
                "artifacts": artifacts,
            },
        ).read_text(encoding="utf-8")
    )


def train_one(
    representation: str,
    anchor: str,
    seed: int,
    *,
    device: str | None = None,
    root: Path = EXP_ROOT,
) -> dict[str, Any]:
    """Run one frozen residual job using authenticated Exp062 anchor vectors."""
    key = (representation, anchor, seed)
    if key not in expected_run_keys():
        raise ValueError(f"Training job is outside the frozen Exp063 grid: {key}")
    if not (root / "PREPARED.json").is_file():
        raise RuntimeError("Exp063 training requires PREPARED.json")
    marker = root / run_marker_name(*key)
    if marker.exists():
        payload = json.loads(marker.read_text(encoding="utf-8"))
        for artifact, digest in payload.get("artifacts", {}).items():
            path = root / artifact
            if not path.is_file() or sha256_file(path) != digest:
                raise RuntimeError(f"Completed Exp063 run artifact drifted: {path}")
        return payload

    data = load_trainval_context(representation)
    scaler_path = _scaler_path(representation, root)
    scaler = _load_scaler(scaler_path, representation)
    ego, scene = transform_context(data.ego, data.scene, scaler)
    spec, grouped_oof, _ = _probability_vectors(anchor, seed)
    checkpoint = root / "checkpoints" / f"{representation}__{anchor}__seed{seed}.pt"
    history_path = root / "metrics" / f"training_history_{representation}__{anchor}__seed{seed}.csv"
    log_path = root / "logs" / f"train_{representation}__{anchor}__seed{seed}.log"
    for path in (checkpoint, history_path, log_path):
        if path.exists():
            raise FileExistsError(f"Unbound partial training artifact exists: {path}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        f"phase=train representation={representation} anchor={anchor} seed={seed} start\n",
        encoding="utf-8",
    )
    result = _exp062_runner().run_authenticated_training(
        spec=spec,
        grouped_oof=grouped_oof,
        train_ego=ego[data.train_mask],
        train_yolop=scene[data.train_mask],
        train_labels=data.labels[data.train_mask],
        train_sequence_ids=data.rows.loc[data.train_mask, "sequence_id"].to_numpy(),
        validation_ego=ego[data.val_mask],
        validation_yolop=scene[data.val_mask],
        validation_labels=data.labels[data.val_mask],
        validation_sequence_ids=data.rows.loc[data.val_mask, "sequence_id"].to_numpy(),
        checkpoint_path=checkpoint,
        device=device,
    )
    _atomic_csv(history_path, pd.DataFrame(result.history))
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(
            f"phase=train complete best_epoch={result.best_epoch} best_harmonic={result.best_score}\n"
        )
    payload = {
        "phase": "trained",
        "representation": representation,
        "anchor_name": anchor,
        "seed": seed,
        "best_epoch": result.best_epoch,
        "best_validation_harmonic": result.best_score,
        "set03_accessed": False,
        "probability_binding_sha256": sha256_file(spec.manifest_path),
        "artifacts": {
            str(checkpoint.relative_to(root)): sha256_file(checkpoint),
            str(history_path.relative_to(root)): sha256_file(history_path),
            str(log_path.relative_to(root)): sha256_file(log_path),
            str(scaler_path.relative_to(root)): sha256_file(scaler_path),
        },
    }
    return json.loads(_atomic_json(marker, payload).read_text(encoding="utf-8"))


def assert_all_runs_complete(root: Path = EXP_ROOT) -> dict[str, dict[str, Any]]:
    """Require and authenticate all 12 independent run markers."""
    results: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for key in expected_run_keys():
        marker = root / run_marker_name(*key)
        if not marker.is_file():
            missing.append(marker.name)
            continue
        payload = json.loads(marker.read_text(encoding="utf-8"))
        if (
            payload.get("representation"),
            payload.get("anchor_name"),
            payload.get("seed"),
        ) != key:
            raise RuntimeError(f"Exp063 run marker identity drifted: {marker}")
        for artifact, digest in payload.get("artifacts", {}).items():
            path = root / artifact
            if not path.is_file() or sha256_file(path) != digest:
                raise RuntimeError(f"Exp063 run artifact authentication failed: {path}")
        results[marker.name] = payload
    if missing:
        raise RuntimeError(
            f"Exp063 validation requires all 12 run markers; missing {len(missing)}"
        )
    return results


def _metric_record(bundle: MetricBundle) -> dict[str, Any]:
    record = asdict(bundle)
    record["per_seed_harmonic"] = json.dumps(list(bundle.per_seed_harmonic))
    return record


def _aggregate(rows: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, MetricBundle]]:
    records: list[dict[str, Any]] = []
    bundles: dict[str, MetricBundle] = {}
    metric_columns = (
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
        "harmonic",
        "bce",
        "brier",
        "high_confidence_errors",
    )
    for (representation, anchor, variant), group in rows.groupby(
        ["representation", "anchor_name", "variant"], sort=False
    ):
        group = group.sort_values("seed")
        record: dict[str, Any] = {
            "representation": representation,
            "anchor_name": anchor,
            "variant": variant,
            "seed_count": len(group),
            "invariants_pass": bool(group["invariants_pass"].all()),
        }
        for column in metric_columns:
            record[column] = float(group[column].mean())
            record[f"{column}_std"] = float(group[column].std(ddof=0))
        per_seed = tuple(float(value) for value in group["harmonic"])
        record["per_seed_harmonic"] = json.dumps(list(per_seed))
        records.append(record)
        bundles[f"{representation}__{anchor}__{variant}"] = MetricBundle(
            f1=record["f1"],
            pr_auc=record["pr_auc"],
            roc_auc=record["roc_auc"],
            harmonic=record["harmonic"],
            high_confidence_errors=record["high_confidence_errors"],
            per_seed_harmonic=per_seed,
            invariants_pass=record["invariants_pass"],
        )
    return pd.DataFrame(records), bundles


def select_validation_candidate(
    candidates: Mapping[str, MetricBundle],
    anchors: Mapping[str, MetricBundle],
) -> tuple[str | None, dict[str, ValidationGateDecision]]:
    """Apply every inherited gate, then the frozen harmonic/PR/F1/HC order."""
    if set(candidates) != set(anchors) or not candidates:
        raise ValueError("Candidate and anchor gate mappings must share nonempty keys")
    decisions = {
        name: evaluate_validation_gate(candidate, anchors[name])
        for name, candidate in candidates.items()
    }
    eligible = [name for name, decision in decisions.items() if decision.eligible]
    if not eligible:
        return None, decisions
    eligible.sort(
        key=lambda name: (
            candidates[name].harmonic,
            candidates[name].pr_auc,
            candidates[name].f1,
            -candidates[name].high_confidence_errors,
        ),
        reverse=True,
    )
    return eligible[0], decisions


def write_validation_decision(
    root: Path,
    winner: str | None,
    decisions: Mapping[str, ValidationGateDecision],
    evidence: Mapping[str, str],
) -> Path:
    """Write exactly one mutually exclusive validation decision marker."""
    validation_only = root / "VALIDATION_ONLY_COMPLETE"
    final_freeze = root / "FROZEN_FINAL_MODEL.json"
    if validation_only.exists() or final_freeze.exists():
        raise FileExistsError("Exp063 validation decision already exists")
    payload = {
        "phase": "validation_decision",
        "winner": winner,
        "set03_accessed": False,
        "validation_gates": {
            name: {
                "eligible": decision.eligible,
                "failed_conditions": list(decision.failed_conditions),
            }
            for name, decision in decisions.items()
        },
        "evidence": dict(evidence),
    }
    return _atomic_json(final_freeze if winner is not None else validation_only, payload)


def validate(root: Path = EXP_ROOT, *, device: str | None = None) -> dict[str, Any]:
    """Evaluate all saved checkpoints on validation and apply the frozen gates."""
    assert_all_runs_complete(root)
    output_paths = (
        root / "metrics/validation_per_seed_results.csv",
        root / "metrics/validation_summary_results.csv",
        root / "metrics/paired_deltas.csv",
    )
    if any(path.exists() for path in output_paths):
        raise FileExistsError("Exp063 validation outputs already exist")
    import torch

    rows_out: list[dict[str, Any]] = []
    prediction_paths: dict[str, str] = {}
    for representation in ExperimentConfig().representations:
        data = load_trainval_context(representation)
        scaler = _load_scaler(_scaler_path(representation, root), representation)
        ego, scene = transform_context(data.ego, data.scene, scaler)
        validation_rows = data.rows.loc[data.val_mask, ["sequence_id", "label"]].reset_index(
            drop=True
        )
        for anchor in ExperimentConfig().anchor_families:
            for seed in ExperimentConfig().seeds:
                _, _, vectors = _probability_vectors(anchor, seed)
                checkpoint_path = (
                    root / "checkpoints" / f"{representation}__{anchor}__seed{seed}.pt"
                )
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                if checkpoint.get("seed") != seed or checkpoint.get(
                    "total_parameters"
                ) != ExperimentConfig().correction_parameter_count:
                    raise RuntimeError("Exp063 checkpoint identity/architecture mismatch")
                model = HybridContextCorrector()
                model.load_state_dict(checkpoint["model_state"], strict=True)
                prediction = predict_context_corrector(
                    model,
                    ego[data.val_mask],
                    scene[data.val_mask],
                    vectors.validation_probabilities,
                    device=device,
                )
                for variant, probabilities in (
                    ("anchor", vectors.validation_probabilities),
                    ("corrected", prediction["probability"]),
                ):
                    metric = compute_binary_metrics(data.labels[data.val_mask], probabilities)
                    rows_out.append(
                        {
                            "representation": representation,
                            "anchor_name": anchor,
                            "seed": seed,
                            "variant": variant,
                            "checkpoint_epoch": int(checkpoint["best_epoch"]),
                            **_metric_record(metric),
                        }
                    )
                prediction_path = (
                    root
                    / "predictions"
                    / f"validation_{representation}__{anchor}__seed{seed}.csv"
                )
                _atomic_csv(
                    prediction_path,
                    validation_rows.assign(
                        anchor_probability=vectors.validation_probabilities,
                        corrected_probability=prediction["probability"],
                        corrected_class_fixed_0p5=(prediction["probability"] >= 0.5).astype(
                            np.int64
                        ),
                        gate=prediction["gate"],
                        applied_correction=prediction["applied_correction"],
                    ),
                )
                prediction_paths[str(prediction_path.relative_to(root))] = sha256_file(
                    prediction_path
                )
    per_seed = pd.DataFrame(rows_out)
    summary, bundles = _aggregate(per_seed)
    _atomic_csv(output_paths[0], per_seed)
    _atomic_csv(output_paths[1], summary)

    historical_path = EXP062_ROOT / "metrics/validation_per_seed_results.csv"
    historical = pd.read_csv(historical_path)
    deltas: list[dict[str, Any]] = []
    for representation in ExperimentConfig().representations:
        for anchor in ExperimentConfig().anchor_families:
            for seed in ExperimentConfig().seeds:
                current = per_seed.loc[
                    per_seed["representation"].eq(representation)
                    & per_seed["anchor_name"].eq(anchor)
                    & per_seed["seed"].eq(seed)
                    & per_seed["variant"].eq("corrected")
                ].iloc[0]
                old = historical.loc[
                    historical["anchor_name"].eq(anchor)
                    & historical["seed"].eq(seed)
                    & historical["variant"].eq("corrected")
                ].iloc[0]
                deltas.append(
                    {
                        "comparison": f"historical_to_{representation}",
                        "anchor_name": anchor,
                        "seed": seed,
                        "f1_delta": float(current.f1 - old.f1),
                        "pr_auc_delta": float(current.pr_auc - old.pr_auc),
                        "roc_auc_delta": float(current.roc_auc - old.roc_auc),
                        "harmonic_delta": float(current.harmonic - old.harmonic),
                        "high_confidence_error_delta": float(
                            current.high_confidence_errors - old.high_confidence_errors
                        ),
                    }
                )
            grid = per_seed.loc[
                per_seed["representation"].eq("corrected_grid")
                & per_seed["anchor_name"].eq(anchor)
                & per_seed["variant"].eq("corrected")
            ].set_index("seed")
            road = per_seed.loc[
                per_seed["representation"].eq("road_relative")
                & per_seed["anchor_name"].eq(anchor)
                & per_seed["variant"].eq("corrected")
            ].set_index("seed")
            if representation == "road_relative":
                for seed in ExperimentConfig().seeds:
                    deltas.append(
                        {
                            "comparison": "corrected_grid_to_road_relative",
                            "anchor_name": anchor,
                            "seed": seed,
                            "f1_delta": float(road.loc[seed, "f1"] - grid.loc[seed, "f1"]),
                            "pr_auc_delta": float(
                                road.loc[seed, "pr_auc"] - grid.loc[seed, "pr_auc"]
                            ),
                            "roc_auc_delta": float(
                                road.loc[seed, "roc_auc"] - grid.loc[seed, "roc_auc"]
                            ),
                            "harmonic_delta": float(
                                road.loc[seed, "harmonic"] - grid.loc[seed, "harmonic"]
                            ),
                            "high_confidence_error_delta": float(
                                road.loc[seed, "high_confidence_errors"]
                                - grid.loc[seed, "high_confidence_errors"]
                            ),
                        }
                    )
    _atomic_csv(output_paths[2], pd.DataFrame(deltas))

    candidates: dict[str, MetricBundle] = {}
    anchors: dict[str, MetricBundle] = {}
    for representation in ExperimentConfig().representations:
        for anchor in ExperimentConfig().anchor_families:
            key = f"{representation}__{anchor}"
            candidates[key] = bundles[f"{key}__corrected"]
            anchors[key] = bundles[f"{key}__anchor"]
    winner, decisions = select_validation_candidate(candidates, anchors)
    evidence = {
        str(path.relative_to(root)): sha256_file(path) for path in output_paths
    }
    evidence.update(prediction_paths)
    decision_path = write_validation_decision(root, winner, decisions, evidence)
    validation_marker = _atomic_json(
        root / "VALIDATION_COMPLETE.json",
        {
            "phase": "validation_complete",
            "winner": winner,
            "decision_marker": decision_path.name,
            "decision_sha256": sha256_file(decision_path),
            "set03_accessed": False,
            "artifacts": evidence,
        },
    )
    return json.loads(validation_marker.read_text(encoding="utf-8"))


def train_all(*, device: str | None = None, root: Path = EXP_ROOT) -> None:
    """Resume the complete frozen grid in deterministic order."""
    for representation, anchor, seed in expected_run_keys():
        print(
            f"exp063_train representation={representation} anchor={anchor} seed={seed}",
            flush=True,
        )
        train_one(representation, anchor, seed, device=device, root=root)
    assert_all_runs_complete(root)
    _atomic_json(
        root / "TRAINING_COMPLETE.json",
        {
            "phase": "training_complete",
            "runs": len(expected_run_keys()),
            "set03_accessed": False,
        },
    )


def verify(root: Path = EXP_ROOT) -> dict[str, Any]:
    """Recompute validation metrics and authenticate the validation-only package."""
    assert_all_runs_complete(root)
    validation_marker = root / "VALIDATION_COMPLETE.json"
    if not validation_marker.is_file():
        raise RuntimeError("Exp063 verification requires VALIDATION_COMPLETE.json")
    payload = json.loads(validation_marker.read_text(encoding="utf-8"))
    if payload.get("set03_accessed") is not False or payload.get("winner") is not None:
        raise RuntimeError("Exp063 completed package is not the expected validation-only result")
    for artifact, digest in payload.get("artifacts", {}).items():
        path = root / artifact
        if not path.is_file() or sha256_file(path) != digest:
            raise RuntimeError(f"Exp063 validation artifact authentication failed: {path}")
    decision = root / str(payload.get("decision_marker"))
    if (
        decision.name != "VALIDATION_ONLY_COMPLETE"
        or not decision.is_file()
        or sha256_file(decision) != payload.get("decision_sha256")
    ):
        raise RuntimeError("Exp063 validation-only decision binding failed")
    per_seed = pd.read_csv(root / "metrics/validation_per_seed_results.csv")
    for representation, anchor, seed in expected_run_keys():
        prediction = pd.read_csv(
            root
            / "predictions"
            / f"validation_{representation}__{anchor}__seed{seed}.csv"
        )
        for variant, probability_column in (
            ("anchor", "anchor_probability"),
            ("corrected", "corrected_probability"),
        ):
            expected = per_seed.loc[
                per_seed["representation"].eq(representation)
                & per_seed["anchor_name"].eq(anchor)
                & per_seed["seed"].eq(seed)
                & per_seed["variant"].eq(variant)
            ]
            if len(expected) != 1:
                raise RuntimeError("Exp063 per-seed metric identity is incomplete")
            actual = compute_binary_metrics(
                prediction["label"].to_numpy(),
                prediction[probability_column].to_numpy(),
            )
            row = expected.iloc[0]
            for name in (
                "accuracy",
                "precision",
                "recall",
                "f1",
                "roc_auc",
                "pr_auc",
                "harmonic",
                "bce",
                "brier",
                "high_confidence_errors",
            ):
                if not np.isclose(float(row[name]), float(getattr(actual, name)), atol=1e-12):
                    raise RuntimeError(
                        f"Exp063 recomputed metric mismatch: {representation}/{anchor}/{seed}/{variant}/{name}"
                    )
            for name in (
                "tp",
                "tn",
                "fp",
                "fn",
                "high_confidence_fp",
                "high_confidence_fn",
            ):
                if int(row[name]) != int(getattr(actual, name)):
                    raise RuntimeError(
                        f"Exp063 recomputed count mismatch: {representation}/{anchor}/{seed}/{variant}/{name}"
                    )
    from .analyze import write_error_transitions, write_representation_diagnostics

    diagnostics = write_representation_diagnostics(root)
    transitions = write_error_transitions(root)
    complete = _atomic_json(
        root / "EXPERIMENT_COMPLETE",
        {
            "phase": "verified_validation_only_complete",
            "set03_accessed": False,
            "validation_marker_sha256": sha256_file(validation_marker),
            "decision_marker_sha256": sha256_file(decision),
            "representation_diagnostics_sha256": sha256_file(diagnostics),
            "error_transitions_sha256": sha256_file(transitions),
        },
    )
    return json.loads(complete.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    extract_parser = subparsers.add_parser("extract")
    extract_parser.add_argument("--device", default="cuda:0")
    subparsers.add_parser("prepare")
    train_parser = subparsers.add_parser("train-all")
    train_parser.add_argument("--device", default=None)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--device", default=None)
    subparsers.add_parser("verify")
    arguments = parser.parse_args()
    if arguments.command == "extract":
        extract_trainval(device=arguments.device)
    elif arguments.command == "prepare":
        prepare()
    elif arguments.command == "train-all":
        train_all(device=arguments.device)
    elif arguments.command == "validate":
        print(json.dumps(validate(device=arguments.device), indent=2, sort_keys=True))
    elif arguments.command == "verify":
        print(json.dumps(verify(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "MetricBundle",
    "assert_all_runs_complete",
    "expected_run_keys",
    "harmonic_f1_pr",
    "prepare",
    "run_marker_name",
    "select_validation_candidate",
    "train_all",
    "train_one",
    "validate",
    "verify",
    "write_validation_decision",
]
