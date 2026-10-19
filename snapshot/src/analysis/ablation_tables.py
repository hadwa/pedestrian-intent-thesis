"""Create clean ablation summary tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def summarize_results(student_results: Path, teacher_results: Path, output_dir: Path) -> None:
    """Read student/teacher result CSVs and write compact summary tables."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if student_results.exists() and student_results.stat().st_size:
        students = pd.read_csv(student_results)
        students.to_csv(output_dir / "student_ablation_summary.csv", index=False)
    if teacher_results.exists() and teacher_results.stat().st_size:
        teachers = pd.read_csv(teacher_results)
        teachers.to_csv(output_dir / "teacher_ablation_summary.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ablation summary tables.")
    parser.add_argument("--student_results", type=Path, default=Path("results/student_results.csv"))
    parser.add_argument("--teacher_results", type=Path, default=Path("results/teacher_results.csv"))
    parser.add_argument("--output_dir", type=Path, default=Path("results/final_tables"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summarize_results(args.student_results, args.teacher_results, args.output_dir)
    print(f"Wrote summary tables to {args.output_dir}")


if __name__ == "__main__":
    main()
