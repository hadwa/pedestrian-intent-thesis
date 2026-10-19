#!/usr/bin/env python3
"""Run Exp051 through the authenticated Exp050 runner with normalized fusion."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
REPO_ROOT = ROOT.parents[1]
EXP050_ROOT = ROOT.parents[0] / "050_mslstm_practical_behavior_replacement_pie_16f_tte30_60"
EXP050_RUNNER_PATH = EXP050_ROOT / "scripts" / "run_experiment.py"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import exp051_core as core  # noqa: E402


def _load_exp050_runner():
    spec = importlib.util.spec_from_file_location("_exp050_runner_for_exp051", EXP050_RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load Exp050 runner: {EXP050_RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


base = _load_exp050_runner()
base.core = core
base.ROOT = ROOT
base.REPO_ROOT = REPO_ROOT
base.BASE_CACHE = EXP050_ROOT / "cache" / "trainval_base"
base.TEST_CACHE = EXP050_ROOT / "cache" / "set03_final"
base.PREPARE_MARKER = ROOT / ".prepare.complete.json"
base.EVALUATION_MARKER = ROOT / ".evaluation.complete.json"
base.COMPLETE_MARKER = ROOT / "EXPERIMENT_COMPLETE"

_original_load_cache = base.load_cache


def _load_reused_cache(path: Path):
    candidate = Path(path)
    try:
        relative = candidate.relative_to(ROOT / "cache")
    except ValueError:
        return _original_load_cache(candidate)
    return _original_load_cache(EXP050_ROOT / "cache" / relative)


base.load_cache = _load_reused_cache


COMPARISON_METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc")


def exp050_paired_deltas(normalized: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    """Return Exp051 minus Exp050 metrics, paired by frozen variant and seed."""
    keys = ["variant", "seed"]
    left = baseline[keys + list(COMPARISON_METRICS)].copy()
    right = normalized[keys + list(COMPARISON_METRICS)].copy()
    merged = left.merge(right, on=keys, suffixes=("_exp050", "_exp051"), validate="one_to_one")
    if len(merged) != len(left) or len(merged) != len(right):
        raise ValueError("Exp050 and Exp051 metric identities do not match exactly.")
    output = merged[keys].copy()
    for metric in COMPARISON_METRICS:
        output[f"{metric}_exp050"] = merged[f"{metric}_exp050"]
        output[f"{metric}_exp051"] = merged[f"{metric}_exp051"]
        output[f"{metric}_delta"] = merged[f"{metric}_exp051"] - merged[f"{metric}_exp050"]
    return output.sort_values(keys).reset_index(drop=True)


def rewrite_report_identity(report: str) -> str:
    """Correct inherited Exp050 self-references while retaining baseline labels."""
    report = report.replace("# Exp050 Final Report", "# Exp051 Final Report")
    report = report.replace("designated Exp050 final evaluation", "designated Exp051 final evaluation")
    report = report.replace("but Exp050 was frozen before this evaluation", "but Exp051 was frozen before this evaluation")
    return report


def run_prepare() -> None:
    if base.PREPARE_MARKER.is_file():
        payload = json.loads(base.PREPARE_MARKER.read_text(encoding="utf-8"))
        base.authenticate_artifacts(ROOT, payload["artifacts"])
        base.log_line(ROOT / "logs/prepare.log", "status=already_complete_authenticated")
        return
    source_marker = EXP050_ROOT / ".prepare.complete.json"
    source_complete = EXP050_ROOT / "EXPERIMENT_COMPLETE"
    if not source_marker.is_file() or not source_complete.is_file():
        raise RuntimeError("Exp051 requires completed, authenticated Exp050 source artifacts.")
    source = json.loads(source_marker.read_text(encoding="utf-8"))
    if source.get("test_accessed") is not False:
        raise RuntimeError("The reused training cache must prove set03 was not accessed.")
    base.authenticate_artifacts(EXP050_ROOT, source["artifacts"])
    audit = {
        "source_experiment": 50,
        "source_prepare_marker": str(source_marker),
        "source_prepare_sha256": core.sha256_file(source_marker),
        "source_complete_sha256": core.sha256_file(source_complete),
        "trainval_metadata_sha256": core.sha256_file(base.BASE_CACHE / "metadata.json"),
        "test_cache_not_accessed_by_exp051_prepare": True,
        "fusion_change": "LayerNorm(512, elementwise_affine=False) before weighted sum",
    }
    audit_path = ROOT / "metrics/reused_cache_audit.json"
    base.atomic_json(audit, audit_path)
    payload = {
        "phase": "prepare",
        "created_at": base.utc_now(),
        "train_rows": core.EXPECTED_SPLIT_COUNTS["train"],
        "val_rows": core.EXPECTED_SPLIT_COUNTS["val"],
        "test_accessed": False,
        "pose_sha256": base.POSE_SHA256,
        "scalers": source["scalers"],
        "artifacts": {
            "metrics/reused_cache_audit.json": core.sha256_file(audit_path),
        },
    }
    base.atomic_json(payload, base.PREPARE_MARKER)
    base.log_line(ROOT / "logs/prepare.log", "phase=prepare status=complete test_accessed=false")


def _rewrite_evaluation_outputs() -> None:
    current = pd.read_csv(ROOT / "metrics/per_seed_results.csv")
    baseline = pd.read_csv(EXP050_ROOT / "metrics/per_seed_results.csv")
    deltas = exp050_paired_deltas(current, baseline)
    delta_path = ROOT / "metrics/normalized_vs_exp050_per_seed.csv"
    base.atomic_csv(deltas, delta_path)
    delta_columns = [f"{metric}_delta" for metric in COMPARISON_METRICS]
    summary = deltas.groupby("variant", sort=False)[delta_columns].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()
    summary_path = ROOT / "metrics/normalized_vs_exp050_summary.csv"
    base.atomic_csv(summary, summary_path)

    report_path = ROOT / "FINAL_REPORT.md"
    report = rewrite_report_identity(report_path.read_text(encoding="utf-8"))
    report = report.replace(
        "Fresh controlled A--F results",
        "Fresh normalized-fusion A--F results",
    )
    if "## Exp050 Paired Comparison" not in report:
        report += (
            "\nThe sole model change from Exp050 is parameter-free LayerNorm on each "
            "512-D stream context immediately before the unchanged learned weighted sum.\n"
            "\n## Exp050 Paired Comparison\n\n"
            + base.frame_to_markdown(summary)
            + "\n"
        )
    report_path.write_text(report, encoding="utf-8")
    marker = json.loads(base.EVALUATION_MARKER.read_text(encoding="utf-8"))
    marker["artifacts"]["FINAL_REPORT.md"] = core.sha256_file(report_path)
    marker["artifacts"]["metrics/normalized_vs_exp050_per_seed.csv"] = core.sha256_file(delta_path)
    marker["artifacts"]["metrics/normalized_vs_exp050_summary.csv"] = core.sha256_file(summary_path)
    marker["source_experiment"] = 50
    marker["fusion_change"] = "parameter-free LayerNorm before fusion"
    base.atomic_json(marker, base.EVALUATION_MARKER)


def run_evaluate() -> None:
    base.run_evaluate()
    _rewrite_evaluation_outputs()


def run_verify() -> None:
    base.assert_training_complete(ROOT)
    if not base.EVALUATION_MARKER.is_file():
        raise RuntimeError("Evaluation marker is missing.")
    evaluation = json.loads(base.EVALUATION_MARKER.read_text(encoding="utf-8"))
    base.authenticate_artifacts(ROOT, evaluation["artifacts"])
    base.COMPLETE_MARKER.write_text(
        json.dumps(
            {
                "experiment": 51,
                "completed_at": base.utc_now(),
                "training_runs": 18,
                "test_rows": int(evaluation["test_rows"]),
                "evaluation_marker_sha256": core.sha256_file(base.EVALUATION_MARKER),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print("Exp051 verification passed; EXPERIMENT_COMPLETE written.")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare")
    train = subparsers.add_parser("train")
    train.add_argument("--variant", choices=core.VARIANTS, required=True)
    train.add_argument("--seed", type=int, choices=core.SEEDS, required=True)
    subparsers.add_parser("train-all")
    subparsers.add_parser("evaluate")
    subparsers.add_parser("verify")
    subparsers.add_parser("gpu-smoke")
    subparsers.add_parser("status")
    args = parser.parse_args()
    if args.command == "prepare":
        run_prepare()
    elif args.command == "train":
        base.run_training(args.variant, args.seed)
    elif args.command == "train-all":
        base.run_train_all()
    elif args.command == "evaluate":
        run_evaluate()
    elif args.command == "verify":
        run_verify()
    elif args.command == "gpu-smoke":
        base.gpu_smoke()
    elif args.command == "status":
        base.status()


if __name__ == "__main__":
    main()
