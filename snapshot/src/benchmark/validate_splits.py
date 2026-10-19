"""Validate frozen benchmark split CSVs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.benchmark.build_pie_benchmark_sequences import (
    NEGATIVE_EVENT_FRAME,
    NEGATIVE_TIME_TO_EVENT,
    REQUIRED_COLUMNS,
    read_config,
)


def load_splits(splits_dir: Path) -> pd.DataFrame:
    frames = []
    for split in ["train", "val", "test"]:
        path = splits_dir / f"{split}_sequences.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing split file: {path}")
        df = pd.read_csv(path)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def validate(df: pd.DataFrame, config: dict[str, Any]) -> list[str]:
    messages: list[str] = []
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if df["sequence_id"].duplicated().any():
        dupes = df.loc[df["sequence_id"].duplicated(), "sequence_id"].head().tolist()
        raise ValueError(f"Duplicate sequence_id values across splits: {dupes}")
    messages.append("No duplicate sequence_id values across train/val/test.")

    obs_len = int(config["observation_length"])
    min_tte, max_tte = [int(x) for x in config["time_to_event_frames"]]
    checks = [
        ((df["observation_length"] == obs_len).all(), f"Every sequence has observation_length={obs_len}."),
        ((df["frame_step"] == 1).all(), "Every sequence has frame_step=1."),
        ((df["frame_sampling"] == "consecutive").all(), "Every sequence uses consecutive frame sampling."),
        (((df["end_frame"] - df["start_frame"] + 1) == obs_len).all(), "Every sequence spans exactly 16 frames."),
    ]
    for ok, msg in checks:
        if not ok:
            raise ValueError(f"Validation failed: {msg}")
        messages.append(msg)

    positives = df[df["label"] == 1]
    if not positives.empty:
        if not positives["time_to_event_frames"].between(min_tte, max_tte).all():
            raise ValueError("Positive crossing samples must be 30-60 frames before event_frame.")
        if not (positives["event_frame"] > 0).all():
            raise ValueError("Positive crossing samples must have positive event_frame values.")
    messages.append("Positive crossing samples satisfy the 30-60 frame time-to-event constraint.")

    negatives = df[df["label"] == 0]
    if not negatives.empty:
        if "track_never_crossing" not in negatives.columns:
            raise ValueError("Negative samples must include track_never_crossing provenance.")
        if not negatives["track_never_crossing"].astype(bool).all():
            raise ValueError("Negative samples must come only from never-crossing tracks.")
        if not (negatives["event_frame"] == NEGATIVE_EVENT_FRAME).all():
            raise ValueError(f"Negative samples must use event_frame={NEGATIVE_EVENT_FRAME}.")
        if not (negatives["time_to_event_frames"] == NEGATIVE_TIME_TO_EVENT).all():
            raise ValueError(f"Negative samples must use time_to_event_frames={NEGATIVE_TIME_TO_EVENT}.")
    messages.append("Negative samples come only from never-crossing tracks and use event_frame=-1.")

    expected_sets = {
        "train": set(config["split"]["train"]),
        "val": set(config["split"]["val"]),
        "test": set(config["split"]["test"]),
    }
    for split in ["train", "val", "test"]:
        split_df = df[df["split"] == split]
        if split_df.empty:
            raise ValueError(f"Split {split} is empty.")
        observed_sets = set(split_df["set_id"].astype(str).unique())
        invalid = observed_sets - expected_sets[split]
        if invalid:
            raise ValueError(f"Split {split} contains unexpected set_id values: {sorted(invalid)}")
    messages.append("Train, validation, and test splits are non-empty and preserve official set IDs.")
    return messages


def label_distribution(df: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    report: dict[str, dict[str, float | int]] = {}
    for split, split_df in df.groupby("split"):
        counts = split_df["label"].value_counts().to_dict()
        total = int(len(split_df))
        positive = int(counts.get(1, 0))
        report[str(split)] = {
            "total": total,
            "negative": int(counts.get(0, 0)),
            "positive": positive,
            "positive_ratio": float(positive / total) if total else 0.0,
        }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate frozen PIE split CSVs.")
    parser.add_argument("--splits_dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--config", type=Path, default=Path("configs/benchmark_pie.yaml"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = read_config(args.config)
    df = load_splits(args.splits_dir)
    messages = validate(df, config)
    distribution = label_distribution(df)

    lines = ["Frozen split validation report", "", *messages, "", "Label distribution:", json.dumps(distribution, indent=2, sort_keys=True)]
    report = "\n".join(lines) + "\n"
    print(report)
    output_path = args.splits_dir / "split_validation_report.txt"
    output_path.write_text(report, encoding="utf-8")
    print(f"Wrote validation report to {output_path}")


if __name__ == "__main__":
    main()
