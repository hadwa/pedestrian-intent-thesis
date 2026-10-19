"""Fail-closed source-table lineage checks for the final confidence benchmark."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import pandas as pd


CLEAN_ROOT = Path("/home/abdelrahman/Hadwa/BIU-PID-Distillation-clean")
ARCHIVE_ROOT = Path("/home/abdelrahman/Hadwa/BIU-PID-Distillation")
SPLITS = ("train", "val", "test")
MANDATORY_CONFIGS = (("jaad", "30_60"), ("jaad", "15_30"), ("jaad", "5_15"), ("pie", "30_60"))
GLOBAL_CONFIDENCE_COLUMNS = (
    "lower_body_kp_conf_mean",
    "skeleton_conf_mean",
    "skeleton_conf_min",
)


def _source_specifications() -> dict[tuple[str, str], dict[str, str]]:
    jaad_script = CLEAN_ROOT / "scripts/experiments/jaad_biu_v2_benchmark_comparison.py"
    pie_script = CLEAN_ROOT / "scripts/experiments/pie_biu_v2_30_60_benchmark.py"
    return {
        ("jaad", "30_60"): {
            "benchmark_script": str(jaad_script),
            "feature_dir": str(ARCHIVE_ROOT / "data/jaad_features_event_16_30_60_overlap08_default"),
            "split_dir": str(ARCHIVE_ROOT / "data/jaad_splits_event_16_30_60_overlap08_default"),
            "lineage_metadata_path": str(ARCHIVE_ROOT / "data/jaad_splits_event_16_30_60_overlap08_default/metadata.json"),
            "reference_experiment": "009",
            "feature_table_kind": "jaad_cached_tables",
            "source_table_naming": "{split}_{table}.csv",
        },
        ("jaad", "15_30"): {
            "benchmark_script": str(jaad_script),
            "feature_dir": str(ARCHIVE_ROOT / "data/jaad_features_stage_horizon_audit/tte_15_30"),
            "split_dir": str(ARCHIVE_ROOT / "data/jaad_splits_stage_horizon_audit/tte_15_30"),
            "lineage_metadata_path": str(ARCHIVE_ROOT / "data/jaad_splits_stage_horizon_audit/tte_15_30/metadata.json"),
            "reference_experiment": "021/009 lineage",
            "feature_table_kind": "jaad_cached_tables",
            "source_table_naming": "{split}_{table}.csv",
        },
        ("jaad", "5_15"): {
            "benchmark_script": str(jaad_script),
            "feature_dir": str(ARCHIVE_ROOT / "data/jaad_features_stage_horizon_audit/tte_5_15"),
            "split_dir": str(ARCHIVE_ROOT / "data/jaad_splits_stage_horizon_audit/tte_5_15"),
            "lineage_metadata_path": str(ARCHIVE_ROOT / "data/jaad_splits_stage_horizon_audit/tte_5_15/metadata.json"),
            "reference_experiment": "009",
            "feature_table_kind": "jaad_cached_tables",
            "source_table_naming": "{split}_{table}.csv",
        },
        ("pie", "30_60"): {
            "benchmark_script": str(pie_script),
            "feature_dir": str(ARCHIVE_ROOT / "data/processed/features_wacv_pcpa_30_60_overlap06"),
            "split_dir": str(ARCHIVE_ROOT / "data/splits_pie_wacv_pcpa_30_60_overlap06"),
            "lineage_metadata_path": str(ARCHIVE_ROOT / "data/splits_pie_wacv_pcpa_30_60_overlap06/benchmark_config.yaml"),
            "reference_experiment": "012",
            "feature_table_kind": "pie_cached_tables",
            "source_table_naming": "{table}_{split}.csv",
        },
    }


def _require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required {description}: {path}")


def _sequence_ids(path: Path, description: str) -> set[str]:
    _require_file(path, description)
    try:
        ids = pd.read_csv(path, usecols=["sequence_id"])["sequence_id"]
    except ValueError as exc:
        raise ValueError(f"{description} lacks sequence_id: {path}") from exc
    if ids.isna().any() or ids.duplicated().any():
        raise ValueError(f"{description} has missing or duplicate sequence IDs: {path}")
    return set(ids.astype(str))


def _fingerprint(sequence_ids: set[str]) -> str:
    return sha256("\n".join(sorted(sequence_ids)).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _validate_jaad_lineage(config: tuple[str, str], path: Path) -> dict[str, str]:
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JAAD lineage metadata: {path}") from exc

    expected_tte = tuple(int(part) for part in config[1].split("_"))
    actual_tte = tuple(metadata.get("tte_range_frames", [metadata.get("tte_min"), metadata.get("tte_max")]))
    if actual_tte != expected_tte or float(metadata.get("overlap", -1)) != 0.8:
        raise ValueError(f"JAAD lineage metadata does not match {config}: {path}")
    if config[1] == "30_60" and metadata.get("observation_length") != 16:
        raise ValueError(f"JAAD 30_60 lineage lacks the required 16-frame observation length: {path}")
    return {
        "lineage_content_status": "verified",
        "lineage_dataset": "jaad",
        "lineage_tte_range": config[1],
        "lineage_overlap": "0.8",
    }


def _parse_pie_lineage(path: Path) -> dict[str, str | tuple[str, ...]]:
    values: dict[str, str | tuple[str, ...]] = {}
    tte_values: list[str] = []
    reading_tte = False
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if raw_line.startswith("time_to_event_frames:"):
            reading_tte = True
            continue
        if reading_tte and raw_line.startswith("- "):
            tte_values.append(raw_line[2:].strip())
            continue
        if raw_line and not raw_line[0].isspace() and ":" in raw_line:
            reading_tte = False
            key, value = raw_line.split(":", 1)
            values[key] = value.strip()
    values["time_to_event_frames"] = tuple(tte_values)
    return values


def _validate_pie_lineage(config: tuple[str, str], path: Path) -> dict[str, str]:
    metadata = _parse_pie_lineage(path)
    expected = {
        "dataset": "PIE",
        "benchmark_name": "pie_wacv_pcpa_30_60_overlap06",
        "observation_length": "16",
        "overlap": "0.6",
        "threshold_selection": "validation_f1",
        "normalization": "train_only",
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError(f"PIE lineage metadata does not match {config}: {path}")
    if metadata["time_to_event_frames"] != ("30", "60"):
        raise ValueError(f"PIE lineage has the wrong time-to-event range: {path}")
    return {
        "lineage_content_status": "verified",
        "lineage_dataset": "pie",
        "lineage_tte_range": "30_60",
        "lineage_overlap": "0.6",
    }


def _validate_lineage(config: tuple[str, str], path: Path) -> dict[str, str]:
    _require_file(path, "lineage metadata")
    details = _validate_pie_lineage(config, path) if config[0] == "pie" else _validate_jaad_lineage(config, path)
    details["lineage_metadata_sha256"] = _file_sha256(path)
    return details


def resolve_benchmark_sources() -> dict[tuple[str, str], dict[str, str]]:
    """Return only source tables whose files and split membership are verified."""
    resolved: dict[tuple[str, str], dict[str, str]] = {}
    for config, spec in _source_specifications().items():
        benchmark_script = Path(spec["benchmark_script"])
        split_dir = Path(spec["split_dir"])
        feature_dir = Path(spec["feature_dir"])
        _require_file(benchmark_script, "benchmark script")
        if not split_dir.is_dir() or not feature_dir.is_dir():
            raise FileNotFoundError(f"Missing source directory for {config}: split={split_dir}, feature={feature_dir}")

        source = dict(spec)
        source.update(_validate_lineage(config, Path(spec["lineage_metadata_path"])))
        for split in SPLITS:
            split_name = f"{split}_sequences.csv" if config[0] == "pie" else f"{split}.csv"
            split_path = split_dir / split_name
            split_ids = _sequence_ids(split_path, f"{config} {split} split")
            source[f"{split}_split_path"] = str(split_path)
            source[f"{split}_sequence_count"] = str(len(split_ids))
            source[f"{split}_sequence_id_sha256"] = _fingerprint(split_ids)

            for table in ("bbox", "biu", "kp", "biu_kp"):
                table_name = spec["source_table_naming"].format(split=split, table=table)
                table_path = feature_dir / table_name
                table_ids = _sequence_ids(table_path, f"{config} {split} {table} table")
                missing_split_ids = split_ids - table_ids
                if missing_split_ids:
                    raise ValueError(
                        f"{config} {split} {table} table is missing frozen split sequence IDs"
                    )
                key = "kp_xy" if table == "kp" else table
                source[f"{split}_{key}_path"] = str(table_path)
                source[f"{split}_{key}_matched_sequence_id_sha256"] = _fingerprint(split_ids)
                source[f"{split}_{key}_sequence_id_sha256"] = _fingerprint(table_ids)
                source[f"{split}_{key}_sequence_count"] = str(len(table_ids))
                source[f"{split}_{key}_extra_cached_sequence_count"] = str(len(table_ids - split_ids))
        source["source_identity_status"] = "verified"
        resolved[config] = source
    return resolved


def collect_global_confidence_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in GLOBAL_CONFIDENCE_COLUMNS if column in df.columns]


def _validate_key_columns(df: pd.DataFrame, view_name: str) -> None:
    missing = {"sequence_id", "split", "coord_norm_version"} - set(df.columns)
    if missing:
        raise ValueError(f"{view_name} is missing required lineage columns: {sorted(missing)}")
    if df["sequence_id"].isna().any() or df["sequence_id"].duplicated().any():
        raise ValueError(f"{view_name} has missing or duplicate sequence IDs")


def _is_confidence_column(column: str) -> bool:
    return column.startswith("kpconf__") or column in GLOBAL_CONFIDENCE_COLUMNS


def validate_kp_xy_vs_xyc_is_additive(
    kp_xy_df: pd.DataFrame, kp_xyc_df: pd.DataFrame | None = None
) -> None:
    """Reject paired views unless KP-XYC is KP-XY plus confidence columns only."""
    if kp_xyc_df is None:
        if "coord_norm_version" in kp_xy_df.columns:
            versions = kp_xy_df["coord_norm_version"]
            if versions.nunique() != 1 or not versions.eq("v2").all():
                raise ValueError("KP-XY vs KP-XYC is not additive")
        return

    _validate_key_columns(kp_xy_df, "KP-XY")
    _validate_key_columns(kp_xyc_df, "KP-XYC")

    xy = kp_xy_df.set_index("sequence_id", verify_integrity=True)
    xyc = kp_xyc_df.set_index("sequence_id", verify_integrity=True)
    if set(xy.index) != set(xyc.index):
        raise ValueError("KP-XY and KP-XYC sequence IDs do not match")
    xyc = xyc.loc[xy.index]

    if not xy["split"].equals(xyc["split"]):
        raise ValueError("KP-XY and KP-XYC split membership does not match")
    if not xy["coord_norm_version"].equals(xyc["coord_norm_version"]):
        raise ValueError("KP-XY and KP-XYC coordinate normalization does not match")
    if xy["coord_norm_version"].nunique() != 1 or not xy["coord_norm_version"].eq("v2").all():
        raise ValueError("KP-XY and KP-XYC must use canonical coord_norm_version v2")

    missing_base_columns = set(xy.columns) - set(xyc.columns)
    if missing_base_columns:
        raise ValueError(f"KP-XYC is missing KP-XY columns: {sorted(missing_base_columns)}")
    added_columns = set(xyc.columns) - set(xy.columns)
    non_confidence_additions = sorted(column for column in added_columns if not _is_confidence_column(column))
    if non_confidence_additions:
        raise ValueError(f"KP-XYC adds non-confidence columns: {non_confidence_additions}")
    if not xy.equals(xyc.loc[:, xy.columns]):
        raise ValueError("KP-XY versus KP-XYC has non-confidence column drift")
