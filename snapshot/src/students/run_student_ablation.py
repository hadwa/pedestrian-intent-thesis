"""Run configured student ablations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from src.students.train_student import dry_run, read_yaml, train

DEFAULT_CONFIGS = [
    Path("configs/student_biu.yaml"),
]


def read_name(path: Path) -> str:
    config = read_yaml(path)
    return str(config.get("student_name", path.stem))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run student ablation configs.")
    parser.add_argument("--configs", type=Path, nargs="*", default=DEFAULT_CONFIGS)
    parser.add_argument("--results_csv", type=Path, default=Path("results/student_results.csv"))
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dry_run:
        summaries = []
        for config_path in args.configs:
            report = dry_run(config_path)
            summaries.append(
                {
                    "config": str(config_path),
                    "student_name": report.student_name or read_name(config_path),
                    "feature_count": report.feature_count,
                    "row_counts": report.row_counts,
                    "nan_before": report.nan_before,
                    "nan_after": report.nan_after,
                    "validation_messages": report.validation_messages,
                }
            )
        print(json.dumps(summaries, indent=2))
        return

    rows = []
    for config_path in args.configs:
        metrics = train(config_path)
        rows.append(
            {
                "student_name": metrics["student_name"],
                "config": str(config_path),
                **{f"val_{k}": v for k, v in metrics["validation"].items()},
                **{f"test_{k}": v for k, v in metrics["test"].items()},
            }
        )
    args.results_csv.parent.mkdir(parents=True, exist_ok=True)
    existing = pd.read_csv(args.results_csv) if args.results_csv.exists() and args.results_csv.stat().st_size else pd.DataFrame()
    pd.concat([existing, pd.DataFrame(rows)], ignore_index=True).to_csv(args.results_csv, index=False)
    print(f"Appended {len(rows)} student result rows to {args.results_csv}")


if __name__ == "__main__":
    main()
