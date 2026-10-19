"""Compare teacher and student errors by sequence_id."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def compute_error_overlap(teacher_predictions: Path, student_predictions: Path) -> dict[str, int]:
    """Compute a 2x2 correctness overlap table."""
    teacher = pd.read_csv(teacher_predictions)
    student = pd.read_csv(student_predictions)
    required_teacher = {"sequence_id", "label", "teacher_pred"}
    required_student = {"sequence_id", "label", "student_pred"}
    if missing := required_teacher - set(teacher.columns):
        raise ValueError(f"Teacher predictions missing columns: {sorted(missing)}")
    if missing := required_student - set(student.columns):
        raise ValueError(f"Student predictions missing columns: {sorted(missing)}")

    merged = teacher.merge(student, on="sequence_id", suffixes=("_teacher", "_student"), validate="one_to_one")
    if merged.empty:
        raise ValueError("No overlapping sequence_id values between teacher and student predictions.")
    if not (merged["label_teacher"].astype(int) == merged["label_student"].astype(int)).all():
        raise ValueError("Teacher and student labels disagree after sequence_id merge.")

    label = merged["label_teacher"].astype(int)
    teacher_correct = merged["teacher_pred"].astype(int) == label
    student_correct = merged["student_pred"].astype(int) == label
    return {
        "teacher_correct_student_correct": int((teacher_correct & student_correct).sum()),
        "teacher_correct_student_wrong": int((teacher_correct & ~student_correct).sum()),
        "teacher_wrong_student_correct": int((~teacher_correct & student_correct).sum()),
        "teacher_wrong_student_wrong": int((~teacher_correct & ~student_correct).sum()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute teacher/student error overlap.")
    parser.add_argument("--teacher_predictions", type=Path, required=True)
    parser.add_argument("--student_predictions", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(compute_error_overlap(args.teacher_predictions, args.student_predictions))


if __name__ == "__main__":
    main()
