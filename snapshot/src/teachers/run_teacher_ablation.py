"""Run MSLSTM teacher ablations against frozen split CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.benchmark.metrics import evaluate_with_validation_threshold
from src.teachers.mslstm_adapter import export_teacher_logits, load_teacher_predictions, prepare_mslstm_inputs, run_mslstm_training

DEFAULT_CONFIGS = [
    Path("configs/teacher_ablation_bbox_pose.yaml"),
    Path("configs/teacher_ablation_bbox_pose_yolop.yaml"),
    Path("configs/teacher_ablation_bbox_pose_ego.yaml"),
    Path("configs/teacher_ablation_full.yaml"),
]


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return config


def verify_files(config: dict[str, Any]) -> None:
    splits_dir = Path(config.get("splits_dir", "data/splits"))
    for split in ["train", "val", "test"]:
        path = splits_dir / f"{split}_sequences.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing frozen split file: {path}")


def run_config(config_path: Path) -> dict[str, Any]:
    config = read_yaml(config_path)
    verify_files(config)
    splits_dir = Path(config.get("splits_dir", "data/splits"))
    feature_roots = {name: Path(path) for name, path in config.get("feature_roots", {}).items()}
    active_streams = list(config.get("active_streams", []))
    prepared_dir = Path(config.get("outputs", {}).get("prepared_inputs", f"data/processed/teacher_inputs/{config_path.stem}"))

    for split in ["train", "val", "test"]:
        prepare_mslstm_inputs(splits_dir / f"{split}_sequences.csv", feature_roots, active_streams, prepared_dir / split)
    checkpoint = run_mslstm_training(config_path)

    prediction_path = Path(config.get("outputs", {}).get("predictions", f"results/predictions/{config_path.stem}_predictions.csv"))
    rows = []
    for split in ["val", "test"]:
        split_output = prediction_path.with_name(f"{prediction_path.stem}_{split}{prediction_path.suffix}")
        pred_df = export_teacher_logits(checkpoint, splits_dir / f"{split}_sequences.csv", split_output)
        rows.append(pred_df)
    preds = pd.concat(rows, ignore_index=True)
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(prediction_path, index=False)

    val = preds[preds["split"] == "val"]
    test = preds[preds["split"] == "test"]
    metrics = evaluate_with_validation_threshold(val["label"], val["teacher_prob"], test["label"], test["teacher_prob"])
    return {"teacher_name": config.get("teacher_name", config_path.stem), "config": str(config_path), **metrics}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run teacher ablation configs.")
    parser.add_argument("--configs", type=Path, nargs="*", default=DEFAULT_CONFIGS)
    parser.add_argument("--results_csv", type=Path, default=Path("results/teacher_results.csv"))
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dry_run:
        for config_path in args.configs:
            config = read_yaml(config_path)
            print(f"Would run teacher ablation: {config.get('teacher_name', config_path.stem)} streams={config.get('active_streams', [])}")
            verify_files(config)
        return

    result_rows = []
    for config_path in args.configs:
        metrics = run_config(config_path)
        result_rows.append(
            {
                "teacher_name": metrics["teacher_name"],
                "config": metrics["config"],
                **{f"val_{k}": v for k, v in metrics["validation"].items()},
                **{f"test_{k}": v for k, v in metrics["test"].items()},
            }
        )
    args.results_csv.parent.mkdir(parents=True, exist_ok=True)
    existing = pd.read_csv(args.results_csv) if args.results_csv.exists() and args.results_csv.stat().st_size else pd.DataFrame()
    pd.concat([existing, pd.DataFrame(result_rows)], ignore_index=True).to_csv(args.results_csv, index=False)
    print(f"Appended {len(result_rows)} teacher result rows to {args.results_csv}")


if __name__ == "__main__":
    main()
