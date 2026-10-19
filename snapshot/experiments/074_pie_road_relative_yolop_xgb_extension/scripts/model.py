"""Exact Exp072 estimator, metrics, gate, and predictor timing for Exp074."""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import contract  # noqa: E402


SEEDS = (42, 43, 44)
EPSILON = 1e-12


def _load_exp072():
    path = contract.resolve_source("exp072_model_core")
    spec = importlib.util.spec_from_file_location("_exp074_pinned_exp072_model", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import authenticated Exp072 model")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


EXP072 = _load_exp072()
compute_metrics = EXP072.compute_metrics
error_transitions = EXP072.error_transitions
cluster_bootstrap_deltas = EXP072.cluster_bootstrap_deltas


def build_xgb(seed: int, labels: np.ndarray):
    return EXP072.build_candidate(seed, labels)


def validation_gate(per_seed: pd.DataFrame, *, corrections: float, harms: float) -> dict[str, Any]:
    indexed = per_seed.set_index(["seed", "model"])
    rows = []
    for seed in SEEDS:
        try:
            candidate = indexed.loc[(seed, "candidate")]
            reference = indexed.loc[(seed, "reference")]
        except KeyError as error:
            raise ValueError("Exp074 gate requires paired seeds") from error
        rows.append({
            "seed": seed,
            "delta_f1": float(candidate.f1 - reference.f1),
            "delta_pr_auc": float(candidate.pr_auc - reference.pr_auc),
            "delta_roc_auc": float(candidate.roc_auc - reference.roc_auc),
            "candidate_fp": float(candidate.fp_ge_0p9_count),
            "candidate_fn": float(candidate.fn_le_0p1_count),
            "reference_fp": float(reference.fp_ge_0p9_count),
            "reference_fn": float(reference.fn_le_0p1_count),
        })
    paired = pd.DataFrame(rows)
    means = paired.mean(numeric_only=True)
    safety = bool((paired.delta_f1 >= -.01 - EPSILON).all() and (paired.delta_pr_auc >= -.01 - EPSILON).all())
    f1, pr, roc = float(means.delta_f1), float(means.delta_pr_auc), float(means.delta_roc_auc)
    candidate_total = float(means.candidate_fp + means.candidate_fn)
    reference_total = float(means.reference_fp + means.reference_fn)
    ratio = candidate_total / reference_total if reference_total else float("inf")
    common = safety and roc >= -.002 - EPSILON
    performance = bool(common and candidate_total <= reference_total + EPSILON and (
        (f1 >= .005 - EPSILON and pr >= -.005 - EPSILON)
        or (pr >= .005 - EPSILON and f1 >= -.005 - EPSILON)
    ))
    confidence = bool(
        common and f1 >= -.005 - EPSILON and pr >= -.005 - EPSILON
        and ratio <= .85 + EPSILON
        and means.candidate_fp <= means.reference_fp + EPSILON
        and means.candidate_fn <= means.reference_fn + EPSILON
        and float(corrections) > float(harms)
    )
    return {
        "status": "advance" if performance or confidence else "no_advance",
        "performance_path": performance, "confidence_path": confidence,
        "per_seed_safety": safety,
        "mean_deltas": {"f1": f1, "pr_auc": pr, "roc_auc": roc},
        "mean_high_confidence": {
            "candidate_fp": float(means.candidate_fp), "candidate_fn": float(means.candidate_fn),
            "candidate_combined": candidate_total, "reference_fp": float(means.reference_fp),
            "reference_fn": float(means.reference_fn), "reference_combined": reference_total,
            "combined_ratio": ratio,
        },
        "mean_transitions": {"corrections": float(corrections), "harms": float(harms)},
    }


def benchmark_pipeline(pipeline, name: str, seed: int, rows, *, protocol=None) -> pd.DataFrame:
    if protocol is None:
        frozen = contract.EXPECTED_REGISTRATION["latency"]
        protocol = {1: frozen["batch1"], 256: frozen["batch256"]}
    output = []
    for batch_size, settings in protocol.items():
        batch = rows.iloc[:batch_size] if hasattr(rows, "iloc") else rows[:batch_size]
        if len(batch) < batch_size:
            raise ValueError("insufficient latency rows")
        for _ in range(int(settings["warmups"])):
            pipeline.predict_proba(batch)
        elapsed = []
        for _ in range(int(settings["repetitions"])):
            started = time.perf_counter_ns()
            pipeline.predict_proba(batch)
            elapsed.append((time.perf_counter_ns() - started) / 1_000_000.0)
        values = np.asarray(elapsed)
        output.append({
            "model": name, "seed": seed, "batch_size": batch_size,
            "warmups": int(settings["warmups"]), "repetitions": int(settings["repetitions"]),
            "mean_ms": float(values.mean()), "median_ms": float(np.median(values)),
            "p95_ms": float(np.quantile(values, .95)), "per_sample_mean_ms": float(values.mean() / batch_size),
        })
    return pd.DataFrame(output)
