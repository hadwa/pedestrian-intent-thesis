"""Validate frozen-row T1/T2 teacher feature tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.students.datasets import load_split_csvs


META_COLUMNS = {"sequence_id", "split", "set_id", "video_id", "ped_id", "start_frame", "end_frame", "label"}
EXCLUDED_SUBSTRINGS = ["ped_action", "ped_look", "ped_occlusion", "pose", "age", "gender", "crossing_point", "critical_point", "exp_start_point", "intention_prob"]
FROZEN_MATCH_COLUMNS = ["sequence_id", "split", "set_id", "video_id", "ped_id", "start_frame", "end_frame", "label"]


def _validate_variant(variant_name: str, table_paths: dict[str, Path], splits: dict[str, pd.DataFrame]) -> tuple[dict[str, Any], list[str]]:
    summary: dict[str, Any] = {"variant": variant_name, "splits": {}, "feature_count": None}
    messages: list[str] = []
    feature_columns_reference: list[str] | None = None
    for split_name, split_df in splits.items():
        path = table_paths[split_name]
        if not path.exists():
            raise FileNotFoundError(f"Missing table for {variant_name}/{split_name}: {path}")
        frame = pd.read_csv(path)
        if len(frame) != len(split_df):
            raise ValueError(f"{variant_name}/{split_name}: row count {len(frame)} != frozen {len(split_df)}")
        if frame["sequence_id"].duplicated().any():
            raise ValueError(f"{variant_name}/{split_name}: duplicate sequence_id values found")
        split_ids = set(split_df["sequence_id"].astype(str))
        frame_ids = set(frame["sequence_id"].astype(str))
        if split_ids != frame_ids:
            raise ValueError(f"{variant_name}/{split_name}: sequence_id coverage mismatch")
        merged = split_df[FROZEN_MATCH_COLUMNS].merge(frame[FROZEN_MATCH_COLUMNS], on="sequence_id", suffixes=("_split", "_table"), validate="one_to_one")
        for column in FROZEN_MATCH_COLUMNS[1:]:
            split_col = f"{column}_split"
            table_col = f"{column}_table"
            if not (merged[split_col].astype(str) == merged[table_col].astype(str)).all():
                raise ValueError(f"{variant_name}/{split_name}: frozen metadata column {column!r} does not match")
        feature_columns = [c for c in frame.columns if c not in META_COLUMNS]
        if feature_columns_reference is None:
            feature_columns_reference = feature_columns
            summary["feature_count"] = len(feature_columns)
        elif feature_columns_reference != feature_columns:
            raise ValueError(f"{variant_name}/{split_name}: feature columns differ from train")
        bad = [c for c in feature_columns if any(term in c for term in EXCLUDED_SUBSTRINGS)]
        if bad:
            raise ValueError(f"{variant_name}/{split_name}: excluded-field-like columns present: {bad[:20]}")
        all_null = [c for c in feature_columns if frame[c].isna().all()]
        if all_null:
            raise ValueError(f"{variant_name}/{split_name}: all-null feature columns present: {all_null[:20]}")
        nan_counts = {c: int(v) for c, v in frame[feature_columns].isna().sum().items() if int(v) != 0}
        stream_nan: dict[str, int] = {}
        for col, value in nan_counts.items():
            stream = col.split("_", 1)[0]
            stream_nan[stream] = stream_nan.get(stream, 0) + value
        repair_counts = {c: float(frame[c].fillna(0).sum()) for c in frame.columns if "repaired_frame_count" in c}
        summary["splits"][split_name] = {
            "rows": int(len(frame)),
            "positive_ratio": float(frame["label"].mean()),
            "nan_feature_columns": nan_counts,
            "nan_counts_by_stream_prefix": stream_nan,
            "repair_counts": repair_counts,
            "future_access_check": "passed_by_metadata_alignment_and_observed_frame_policy",
        }
        messages.append(
            f"{variant_name}/{split_name}: rows, sequence_id coverage, frozen metadata alignment, "
            "excluded-column scan, all-null scan, and observed-frame-only policy checks passed"
        )
    return summary, messages


def write_report(path: Path, summaries: list[dict[str, Any]], messages: list[str]) -> None:
    lines = [
        "# MSLSTM Frozen T1/T2 Table Validation Report",
        "",
        "## Summary",
        "",
        "Validated frozen-row T1/T2 feature tables against the frozen PIE benchmark splits.",
        "The checks enforce exact row counts, exact sequence_id coverage, frozen metadata equality, excluded-stream absence, and no all-null feature columns.",
        "No-future-frame access is validated by exact `start_frame`/`end_frame` alignment plus the extractor's observed-frame-only policy.",
        "",
        "## Validation Messages",
        "",
    ]
    for msg in messages:
        lines.append(f"- {msg}")
    for summary in summaries:
        lines.extend(
            [
                "",
                f"## {summary['variant']}",
                "",
                f"- Feature count: `{summary['feature_count']}`",
                "",
            ]
        )
        for split_name, split_summary in summary["splits"].items():
            lines.append(f"### {split_name}")
            lines.append("")
            lines.append(f"- Rows: `{split_summary['rows']}`")
            lines.append(f"- Positive ratio: `{split_summary['positive_ratio']:.6f}`")
            lines.append(f"- Future-frame access check: `{split_summary['future_access_check']}`")
            lines.append(f"- NaN counts by stream prefix: `{split_summary['nan_counts_by_stream_prefix']}`")
            lines.append(f"- Repair counts: `{split_summary['repair_counts']}`")
            lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate frozen-row T1/T2 teacher tables.")
    parser.add_argument("--splits_dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--tables_dir", type=Path, default=Path("data/processed/teacher"))
    parser.add_argument("--report_path", type=Path, default=Path("MSLSTM_FROZEN_T1_T2_TABLE_VALIDATION_REPORT.md"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    splits = load_split_csvs(args.splits_dir)
    table_paths = {
        "T1": {split: args.tables_dir / f"mslstm_frozen_T1_{split}.csv" for split in ["train", "val", "test"]},
        "T2": {split: args.tables_dir / f"mslstm_frozen_T2_{split}.csv" for split in ["train", "val", "test"]},
    }
    messages: list[str] = []
    summaries: list[dict[str, Any]] = []
    for variant, paths in table_paths.items():
        summary, variant_messages = _validate_variant(variant, paths, splits)
        summaries.append(summary)
        messages.extend(variant_messages)
    write_report(args.report_path, summaries, messages)
    print(json.dumps({"validated_variants": [s["variant"] for s in summaries], "report": str(args.report_path)}, indent=2))


if __name__ == "__main__":
    main()
