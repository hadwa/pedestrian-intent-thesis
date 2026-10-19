"""CLI for building BIU feature tables keyed by frozen sequence_id."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.features.biu_features import (
    FeatureBuildReport,
    build_biu_feature_table,
    load_keypoint_index,
    validate_biu_feature_outputs,
)


def write_report(report: FeatureBuildReport, output_dir: Path, validation_messages: list[str]) -> Path:
    """Write the per-run feature generation report."""
    report.validation_messages = validation_messages
    report_path = output_dir / "biu_feature_report.md"
    nonzero_nan_counts = {k: v for k, v in sorted(report.feature_nan_counts.items()) if v}
    lines = [
        "# BIU Feature Table Generation Report",
        "",
        "## Inputs",
        "",
        f"- JSONL source: `{report.source_jsonl}`",
        f"- Splits directory: `{report.splits_dir}`",
        f"- Output directory: `{report.output_dir}`",
        "",
        "## Outputs",
        "",
        "- `biu_train.csv`",
        "- `biu_val.csv`",
        "- `biu_test.csv`",
        "",
        "## Row Counts",
        "",
    ]
    for split in ["train", "val", "test"]:
        lines.append(f"- {split}: split rows `{report.split_rows.get(split, 0)}`, output rows `{report.output_rows.get(split, 0)}`")
    lines.extend(
        [
            "",
            "## Missing And Quality Counts",
            "",
            f"- Missing required frames: `{report.missing_frame_count}`",
            f"- Invalid required-keypoint frames: `{report.invalid_keypoint_frames}`",
            f"- Low-confidence frames among required keypoints: `{report.low_confidence_frames}`",
            "",
            "## Sequence NaN Counts",
            "",
        ]
    )
    for split in ["train", "val", "test"]:
        lines.append(f"- {split}: `{report.sequence_nan_counts.get(split, 0)}` rows with at least one NaN")
    lines.extend(["", "## Feature NaN Counts", ""])
    if nonzero_nan_counts:
        for feature, count in nonzero_nan_counts.items():
            lines.append(f"- `{feature}`: `{count}`")
    else:
        lines.append("No NaN values were produced in BIU feature columns.")
    lines.extend(["", "## Missing Frame Examples", ""])
    if report.missing_frame_examples:
        for item in report.missing_frame_examples:
            lines.append(f"- `{item}`")
    else:
        lines.append("No missing frame examples; all required frames were found.")
    lines.extend(["", "## Validation", ""])
    for message in validation_messages:
        lines.append(f"- {message}")
    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build BIU feature tables aligned to frozen sequence_id CSVs.")
    parser.add_argument("--jsonl_path", type=Path, required=True)
    parser.add_argument("--splits_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = FeatureBuildReport(
        source_jsonl=str(args.jsonl_path),
        splits_dir=str(args.splits_dir),
        output_dir=str(args.output_dir),
    )
    keypoint_index = load_keypoint_index(args.jsonl_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val", "test"]:
        sequence_csv = args.splits_dir / f"{split}_sequences.csv"
        output_path = args.output_dir / f"biu_{split}.csv"
        build_biu_feature_table(sequence_csv, keypoint_index, output_path, report)

    validation_messages = validate_biu_feature_outputs(args.splits_dir, args.output_dir)
    report_path = write_report(report, args.output_dir, validation_messages)
    print(f"Wrote BIU feature tables to {args.output_dir}")
    print(f"Wrote BIU feature report to {report_path}")


if __name__ == "__main__":
    main()
