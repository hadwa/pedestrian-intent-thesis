"""CLI wrapper for exporting teacher logits aligned by sequence_id."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.teachers.mslstm_adapter import export_teacher_logits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export teacher logits for a frozen sequence CSV.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sequence_csv", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_teacher_logits(args.checkpoint, args.sequence_csv, args.output_path)
    print(f"Wrote teacher logits to {args.output_path}")


if __name__ == "__main__":
    main()
