"""Skeleton for future knowledge distillation training.

KD is intentionally not the main task yet. This file documents the required data
alignment and fails loudly until the supervised baselines and teacher exports exist.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Future KD training entry point.")
    parser.add_argument("--splits_dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--student_config", type=Path, required=True)
    parser.add_argument("--teacher_logits", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for split in ["train", "val", "test"]:
        path = args.splits_dir / f"{split}_sequences.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing frozen split file: {path}")
    if not args.teacher_logits.exists():
        raise FileNotFoundError(f"Missing teacher logits file: {args.teacher_logits}")
    raise NotImplementedError(
        "KD training will load frozen sequence CSVs, student features, and teacher "
        "logits aligned by sequence_id. Implement after supervised student and "
        "teacher baselines are frozen."
    )


if __name__ == "__main__":
    main()
