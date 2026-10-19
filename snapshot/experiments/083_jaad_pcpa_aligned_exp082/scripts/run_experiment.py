"""Phase runner for Exp083."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = SCRIPT_DIR.parent


def _load(filename: str, name: str):
    path = SCRIPT_DIR / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


contract = _load("contract.py", "exp083_contract_for_runner")
cohort_module = _load("cohort.py", "exp083_cohort_for_runner")
coverage_module = _load("coverage.py", "exp083_coverage_for_runner")
context_extraction = _load(
    "context_extraction.py", "exp083_context_extraction_for_runner"
)
features_module = _load("features.py", "exp083_features_for_runner")
model_module = _load("model.py", "exp083_model_for_runner")


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    contract.atomic_write_text(path, frame.to_csv(index=False))


def run_protocol_audit(root: Path | str = EXPERIMENT_ROOT) -> dict:
    root = Path(root)
    api_path = contract.resolve_source("jaad_api")
    imdb, database = cohort_module.load_database_and_api(api_path)
    manifests = {}
    outputs = []
    for cohort_name, sample_type in cohort_module.COHORT_TO_SAMPLE_TYPE.items():
        for split in cohort_module.SPLITS:
            frame = cohort_module.build_cohort(
                sample_type,
                split,
                api_path=api_path,
                imdb=imdb,
                database=database,
            )
            manifests[(cohort_name, split)] = frame
            path = root / "manifests" / f"{cohort_name}_{split}.csv"
            _write_csv(path, frame)
            outputs.append(path)
    cohort_module.assert_split_disjoint(manifests)
    cohort_module.assert_approved_raw_frame_gaps(manifests)
    summary = cohort_module.summarize_manifests(manifests)
    summary_path = root / "metrics/cohort_summary.csv"
    _write_csv(summary_path, summary)
    outputs.append(summary_path)
    labels = (
        pd.concat(manifests.values(), ignore_index=True)
        .groupby(["cohort", "split", "label_source", "label"], dropna=False)
        .agg(windows=("sequence_id", "size"), tracks=("track_id", "nunique"))
        .reset_index()
    )
    labels_path = root / "metrics/label_source_summary.csv"
    _write_csv(labels_path, labels)
    outputs.append(labels_path)
    marker = contract.publish_phase(
        root,
        "protocol-audit",
        outputs,
        test_accessed=False,
        test_metrics_accessed=False,
        cohorts=["jaad_beh", "jaad_all"],
    )
    return {"marker": str(marker), "summary": summary.to_dict(orient="records")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "phase",
        choices=(
            "protocol-audit",
            "coverage-audit",
            "extract-development-context",
            "prepare-development",
            "train-validate",
            "extract-test-context",
            "prepare-test",
            "evaluate",
        ),
    )
    args = parser.parse_args()
    if args.phase == "protocol-audit":
        run_protocol_audit()
    elif args.phase == "coverage-audit":
        result = coverage_module.run_coverage_audit(EXPERIMENT_ROOT)
        print(result)
    elif args.phase == "extract-development-context":
        result = context_extraction.run_development_extraction(EXPERIMENT_ROOT)
        print(result)
    elif args.phase == "prepare-development":
        result = features_module.run_prepare_development(EXPERIMENT_ROOT)
        print(result)
    elif args.phase == "train-validate":
        result = model_module.run_train_validate(EXPERIMENT_ROOT)
        print(result)
    elif args.phase == "extract-test-context":
        result = context_extraction.run_test_extraction(EXPERIMENT_ROOT)
        print(result)
    elif args.phase == "prepare-test":
        result = features_module.run_prepare_test(EXPERIMENT_ROOT)
        print(result)
    elif args.phase == "evaluate":
        result = model_module.run_evaluate(EXPERIMENT_ROOT)
        print(result)


if __name__ == "__main__":
    main()
