"""Build frozen PIE benchmark sequence CSVs.

This module defines the canonical sequence interface for all future experiments.
Every later feature table, student prediction, teacher logit, and KD target must join
against these outputs by sequence_id.

First supported parser mode: jsonl_keypoints, for the processed BIU/PIE JSONL file
used by the previous repository. Raw PIE XML parsing is intentionally not guessed.
"""

from __future__ import annotations

import argparse
import ast
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

try:  # Keep the benchmark scripts runnable before the environment is installed.
    import yaml  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - exercised only in bare envs.
    yaml = None

REQUIRED_COLUMNS = [
    "sequence_id",
    "split",
    "set_id",
    "video_id",
    "ped_id",
    "track_id",
    "start_frame",
    "end_frame",
    "event_frame",
    "last_observed_frame",
    "time_to_event_frames",
    "label",
    "observation_length",
    "frame_step",
    "frame_sampling",
    "source_annotation_path",
]

JSONL_REQUIRED_FIELDS = {"set", "video", "pid", "frame_idx", "label"}
NEGATIVE_EVENT_FRAME = -1
NEGATIVE_TIME_TO_EVENT = -1
PARSER_STATS: dict[str, Any] = {}


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        items = [item.strip() for item in inner.split(",")]
        parsed = []
        for item in items:
            if (item.startswith("'") and item.endswith("'")) or (item.startswith('"') and item.endswith('"')):
                parsed.append(item[1:-1])
            else:
                parsed.append(_parse_scalar(item))
        return parsed
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value.strip('"')


def _fallback_safe_load(text: str) -> dict[str, Any]:
    """Small YAML subset reader for configs/benchmark_pie.yaml when PyYAML is absent."""
    result: dict[str, Any] = {}
    current_section: str | None = None
    current_list_key: str | None = None

    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()

        if indent == 0 and line.endswith(":"):
            current_section = line[:-1]
            result[current_section] = {}
            current_list_key = None
            continue
        if indent == 0 and ":" in line:
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if value:
                result[key] = _parse_scalar(value)
                current_section = None
                current_list_key = None
            else:
                result[key] = []
                current_section = key
                current_list_key = key
            continue
        if indent == 2 and current_section == "split" and ":" in line:
            key, value = line.split(":", 1)
            result["split"][key.strip()] = _parse_scalar(value.strip())
            continue
        if indent == 2 and current_section == "frame_sampling" and ":" in line:
            key, value = line.split(":", 1)
            result["frame_sampling"][key.strip()] = _parse_scalar(value.strip())
            continue
        if indent == 2 and line.startswith("- ") and current_list_key:
            result[current_list_key].append(_parse_scalar(line[2:]))
            continue

    return result


def _fallback_safe_dump(config: dict[str, Any]) -> str:
    """Small deterministic YAML writer for the benchmark config copy."""
    lines: list[str] = []
    for key, value in config.items():
        if isinstance(value, dict):
            lines.append(f"{key}:")
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, list):
                    rendered = "[" + ", ".join(str(x) for x in sub_value) + "]"
                elif isinstance(sub_value, bool):
                    rendered = str(sub_value).lower()
                else:
                    rendered = str(sub_value)
                lines.append(f"  {sub_key}: {rendered}")
        elif isinstance(value, list):
            lines.append(f"{key}:")
            for item in value:
                lines.append(f"  - {json.dumps(item) if isinstance(item, str) else item}")
        elif isinstance(value, bool):
            lines.append(f"{key}: {str(value).lower()}")
        else:
            lines.append(f"{key}: {value}")
    return "\n".join(lines) + "\n"


def read_config(path: Path) -> dict[str, Any]:
    """Load a YAML benchmark config."""
    text = path.read_text(encoding="utf-8")
    config = yaml.safe_load(text) if yaml is not None else _fallback_safe_load(text)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return config


def _write_config(config: dict[str, Any], path: Path) -> None:
    if yaml is not None:
        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, sort_keys=False)
    else:
        path.write_text(_fallback_safe_dump(config), encoding="utf-8")


def resolve_jsonl_path(pie_root: Path) -> Path:
    """Resolve the first supported processed JSONL parser input."""
    candidates = []
    if pie_root.is_file():
        candidates.append(pie_root)
    else:
        candidates.extend(
            [
                pie_root / "data_keypoints_all.jsonl",
                pie_root / "data" / "data_keypoints_all.jsonl",
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find data_keypoints_all.jsonl. Pass --pie_root as either the JSONL "
        "file itself or a directory containing data_keypoints_all.jsonl."
    )


def _detect_crossing_starts(frame_labels: list[tuple[int, int]]) -> list[int]:
    starts: list[int] = []
    prev = 0
    for frame_idx, label in frame_labels:
        if label == 1 and prev == 0:
            starts.append(frame_idx)
        prev = label
    return starts


def load_jsonl_keypoint_tracks(pie_root: Path) -> list[dict[str, Any]]:
    """Load processed BIU/PIE JSONL tracks grouped by (set, video, pid)."""
    jsonl_path = resolve_jsonl_path(pie_root)
    grouped: dict[tuple[str, str, str], list[tuple[int, int]]] = defaultdict(list)
    label_counts: Counter[int] = Counter()
    set_counts: Counter[str] = Counter()
    total_rows = 0

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            missing = JSONL_REQUIRED_FIELDS - set(row)
            if missing:
                raise ValueError(f"{jsonl_path}:{line_num} missing required fields: {sorted(missing)}")
            set_id = str(row["set"])
            video_id = str(row["video"])
            ped_id = str(row["pid"])
            frame_idx = int(row["frame_idx"])
            label = int(row["label"])
            if label not in (0, 1):
                raise ValueError(f"{jsonl_path}:{line_num} has non-binary label {label!r}")
            grouped[(set_id, video_id, ped_id)].append((frame_idx, label))
            label_counts[label] += 1
            set_counts[set_id] += 1
            total_rows += 1

    if not grouped:
        raise RuntimeError(f"No rows found in {jsonl_path}")

    tracks: list[dict[str, Any]] = []
    transition_counts: Counter[int] = Counter()
    duplicate_frame_tracks = 0
    for (set_id, video_id, ped_id), values in sorted(grouped.items()):
        values = sorted(values)
        frames = [frame for frame, _ in values]
        labels = [label for _, label in values]
        if len(frames) != len(set(frames)):
            duplicate_frame_tracks += 1
            raise ValueError(f"Duplicate frame_idx values in track {(set_id, video_id, ped_id)}")
        crossing_starts = _detect_crossing_starts(values)
        transition_counts[len(crossing_starts)] += 1
        tracks.append(
            {
                "set_id": set_id,
                "video_id": video_id,
                "ped_id": ped_id,
                "track_id": f"{set_id}_{video_id}_{ped_id}",
                "frames": frames,
                "labels": labels,
                "crossing_starts": crossing_starts,
                "event_frame": crossing_starts[0] if crossing_starts else NEGATIVE_EVENT_FRAME,
                "label": 1 if crossing_starts else 0,
                "track_never_crossing": not any(labels) and not crossing_starts,
                "num_crossing_transitions": len(crossing_starts),
                "source_annotation_path": str(jsonl_path),
                "parser_mode": "jsonl_keypoints",
            }
        )

    global PARSER_STATS
    PARSER_STATS = {
        "parser_mode": "jsonl_keypoints",
        "source_annotation_path": str(jsonl_path),
        "total_rows": int(total_rows),
        "set_row_counts": dict(sorted(set_counts.items())),
        "frame_label_counts": {str(k): int(v) for k, v in sorted(label_counts.items())},
        "num_tracks": int(len(tracks)),
        "transition_count_distribution": {str(k): int(v) for k, v in sorted(transition_counts.items())},
        "multi_transition_tracks": int(sum(1 for t in tracks if t["num_crossing_transitions"] > 1)),
        "never_crossing_tracks": int(sum(1 for t in tracks if t["track_never_crossing"])),
        "duplicate_frame_tracks": int(duplicate_frame_tracks),
        "negative_event_frame_value": NEGATIVE_EVENT_FRAME,
        "negative_time_to_event_value": NEGATIVE_TIME_TO_EVENT,
    }
    return tracks


def load_pie_tracks(pie_root: Path, parser_mode: str = "jsonl_keypoints") -> list[dict[str, Any]]:
    """Load PIE pedestrian tracks using an explicit parser mode."""
    if parser_mode != "jsonl_keypoints":
        raise NotImplementedError(
            f"Unsupported parser_mode={parser_mode!r}. The first supported mode is 'jsonl_keypoints'."
        )
    return load_jsonl_keypoint_tracks(pie_root)


def assign_split(set_id: str, config: dict[str, Any]) -> str:
    """Return train, val, or test for a PIE set id."""
    for split_name, set_ids in config["split"].items():
        if set_id in set_ids:
            return split_name
    raise ValueError(f"Set id {set_id!r} is not present in benchmark split config.")


def _window_step(config: dict[str, Any]) -> int:
    """Convert overlap into a deterministic sliding-window stride."""
    obs_len = int(config["observation_length"])
    overlap = float(config.get("overlap", 0.0))
    step = round(obs_len * (1.0 - overlap))
    return max(1, int(step))


def _assert_main_benchmark_config(config: dict[str, Any]) -> None:
    obs_len = int(config["observation_length"])
    frame_cfg = config["frame_sampling"]
    if obs_len != 16:
        raise ValueError("Main benchmark must use observation_length=16.")
    if frame_cfg.get("type") != "consecutive" or int(frame_cfg.get("step")) != 1:
        raise ValueError("Main benchmark must use consecutive raw frames with frame_step=1.")
    if bool(frame_cfg.get("skip_frames")):
        raise ValueError("Main benchmark forbids frame skipping.")


def _make_sequence_row(
    track: dict[str, Any],
    config: dict[str, Any],
    start_frame: int,
    end_frame: int,
    label: int,
    event_frame: int,
    time_to_event: int,
) -> dict[str, Any]:
    set_id = str(track["set_id"])
    video_id = str(track["video_id"])
    ped_id = str(track["ped_id"])
    sequence_id = f"{set_id}_{video_id}_{ped_id}_{start_frame}_{end_frame}"
    return {
        "sequence_id": sequence_id,
        "split": assign_split(set_id, config),
        "set_id": set_id,
        "video_id": video_id,
        "ped_id": ped_id,
        "track_id": str(track["track_id"]),
        "start_frame": int(start_frame),
        "end_frame": int(end_frame),
        "event_frame": int(event_frame),
        "last_observed_frame": int(end_frame),
        "time_to_event_frames": int(time_to_event),
        "label": int(label),
        "observation_length": int(config["observation_length"]),
        "frame_step": 1,
        "frame_sampling": "consecutive",
        "source_annotation_path": str(track["source_annotation_path"]),
        "parser_mode": str(track.get("parser_mode", "jsonl_keypoints")),
        "track_never_crossing": bool(track.get("track_never_crossing", False)),
        "num_crossing_transitions": int(track.get("num_crossing_transitions", 0)),
        "negative_event_policy": "event_frame=-1,time_to_event_frames=-1" if label == 0 else "positive_transition",
    }


def generate_sequences_for_track(track: dict[str, Any], config: dict[str, Any]) -> list[dict[str, Any]]:
    """Generate frozen benchmark observation windows for one parsed track.

    Positives use the first 0->1 crossing transition as event_frame. Tracks with
    multiple transitions are counted in parser stats but use only the first event.
    Negatives are generated only from never-crossing tracks. Negative event_frame
    and time_to_event_frames are both set to -1 by policy.
    """
    _assert_main_benchmark_config(config)
    required = {
        "set_id",
        "video_id",
        "ped_id",
        "track_id",
        "frames",
        "labels",
        "crossing_starts",
        "source_annotation_path",
        "track_never_crossing",
    }
    missing = required - set(track)
    if missing:
        raise ValueError(f"Track is missing required keys: {sorted(missing)}")

    obs_len = int(config["observation_length"])
    frames = sorted(int(x) for x in track["frames"])
    if len(frames) < obs_len:
        return []

    frame_set = set(frames)
    min_tte, max_tte = [int(x) for x in config["time_to_event_frames"]]
    sequences: list[dict[str, Any]] = []
    crossing_starts = [int(x) for x in track.get("crossing_starts", [])]

    if crossing_starts:
        event_frame = crossing_starts[0]
        for time_to_event in range(min_tte, max_tte + 1):
            end_frame = event_frame - time_to_event
            start_frame = end_frame - obs_len + 1
            if all(frame in frame_set for frame in range(start_frame, end_frame + 1)):
                sequences.append(_make_sequence_row(track, config, start_frame, end_frame, 1, event_frame, time_to_event))
        return sequences

    if not bool(track.get("track_never_crossing")):
        # A malformed track with no transition but at least one positive label should not
        # contribute negatives under Option A.
        return []

    stride = _window_step(config)
    for start_frame in range(frames[0], frames[-1] - obs_len + 2, stride):
        end_frame = start_frame + obs_len - 1
        if all(frame in frame_set for frame in range(start_frame, end_frame + 1)):
            sequences.append(
                _make_sequence_row(
                    track,
                    config,
                    start_frame,
                    end_frame,
                    0,
                    NEGATIVE_EVENT_FRAME,
                    NEGATIVE_TIME_TO_EVENT,
                )
            )
    return sequences


def validate_sequences(df: pd.DataFrame, config: dict[str, Any]) -> None:
    """Validate generated sequence rows before writing outputs."""
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Generated sequences are missing required columns: {missing}")
    if df.empty:
        raise ValueError("No sequences were generated. Check PIE parser and benchmark constraints.")
    if df["sequence_id"].duplicated().any():
        dupes = df.loc[df["sequence_id"].duplicated(), "sequence_id"].head().tolist()
        raise ValueError(f"Duplicate sequence_id values found: {dupes}")

    expected_splits = {"train", "val", "test"}
    observed_splits = set(df["split"].unique())
    if observed_splits != expected_splits:
        raise ValueError(f"Expected splits {expected_splits}, found {observed_splits}")
    for split in sorted(expected_splits):
        if df[df["split"] == split].empty:
            raise ValueError(f"Split {split!r} is empty.")

    obs_len = int(config["observation_length"])
    min_tte, max_tte = [int(x) for x in config["time_to_event_frames"]]
    if not (df["observation_length"] == obs_len).all():
        raise ValueError("All rows must use observation_length from benchmark config.")
    if not (df["frame_step"] == 1).all():
        raise ValueError("All rows must use frame_step=1.")
    if not (df["frame_sampling"] == "consecutive").all():
        raise ValueError("All rows must use consecutive frame sampling.")
    if not ((df["end_frame"] - df["start_frame"] + 1) == obs_len).all():
        raise ValueError("All rows must span exactly 16 raw consecutive frames.")

    positives = df[df["label"] == 1]
    if not positives.empty:
        if not positives["time_to_event_frames"].between(min_tte, max_tte).all():
            raise ValueError("Positive crossing samples must be 30-60 frames before event_frame.")
        if not (positives["event_frame"] > 0).all():
            raise ValueError("Positive crossing samples must have a positive event_frame.")

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


def _label_distribution(df: pd.DataFrame) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for split, split_df in df.groupby("split"):
        counts = split_df["label"].value_counts().to_dict()
        total = int(len(split_df))
        positives = int(counts.get(1, 0))
        report[str(split)] = {
            "total": total,
            "negative": int(counts.get(0, 0)),
            "positive": positives,
            "positive_ratio": float(positives / total) if total else 0.0,
        }
    return report


def write_outputs(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: dict[str, Any],
    output_dir: Path,
) -> None:
    """Write frozen split CSVs, class distribution, and frozen config copy."""
    output_dir.mkdir(parents=True, exist_ok=True)
    combined = pd.concat([train_df, val_df, test_df], ignore_index=True)
    validate_sequences(combined, config)

    train_df.to_csv(output_dir / "train_sequences.csv", index=False)
    val_df.to_csv(output_dir / "val_sequences.csv", index=False)
    test_df.to_csv(output_dir / "test_sequences.csv", index=False)

    distribution = _label_distribution(combined)
    distribution["_parser_stats"] = PARSER_STATS
    with (output_dir / "label_distribution.json").open("w", encoding="utf-8") as f:
        json.dump(distribution, f, indent=2, sort_keys=True)

    frozen_config = dict(config)
    frozen_config["parser_mode"] = "jsonl_keypoints"
    frozen_config["negative_event_policy"] = {
        "event_frame": NEGATIVE_EVENT_FRAME,
        "time_to_event_frames": NEGATIVE_TIME_TO_EVENT,
        "description": "Negatives are generated only from never-crossing tracks; event frame is not defined.",
    }
    _write_config(frozen_config, output_dir / "benchmark_config.yaml")


def build_sequences(pie_root: Path, config: dict[str, Any], parser_mode: str = "jsonl_keypoints") -> pd.DataFrame:
    """Load PIE tracks and generate a single DataFrame of benchmark sequences."""
    tracks = load_pie_tracks(pie_root, parser_mode=parser_mode)
    rows: list[dict[str, Any]] = []
    for track in tracks:
        rows.extend(generate_sequences_for_track(track, config))
    df = pd.DataFrame(rows)
    validate_sequences(df, config)
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build frozen PIE benchmark sequence CSVs.")
    parser.add_argument("--pie_root", type=Path, required=True, help="Path to data_keypoints_all.jsonl or its parent directory.")
    parser.add_argument("--config", type=Path, required=True, help="Path to benchmark_pie.yaml.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory for frozen split outputs.")
    parser.add_argument("--parser_mode", choices=["jsonl_keypoints"], default="jsonl_keypoints")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = read_config(args.config)
    df = build_sequences(args.pie_root, config, parser_mode=args.parser_mode)
    write_outputs(
        df[df["split"] == "train"].copy(),
        df[df["split"] == "val"].copy(),
        df[df["split"] == "test"].copy(),
        config,
        args.output_dir,
    )
    print(f"Wrote frozen benchmark split files to {args.output_dir} using parser_mode={args.parser_mode}")


if __name__ == "__main__":
    main()
