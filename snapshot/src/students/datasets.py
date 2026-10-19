"""Dataset loading utilities for student experiments.

All student datasets must be loaded from frozen split CSVs and feature tables joined
strictly by sequence_id. Split membership, labels, and row counts are never changed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset

SPLITS = ["train", "val", "test"]

CORE_SPLIT_COLUMNS = [
    "sequence_id",
    "split",
    "label",
    "set_id",
    "video_id",
    "ped_id",
    "track_id",
    "start_frame",
    "end_frame",
]

FEATURE_METADATA_COLUMNS = set(CORE_SPLIT_COLUMNS) | {
    "event_frame",
    "last_observed_frame",
    "time_to_event_frames",
    "observation_length",
    "frame_step",
    "frame_sampling",
    "source_annotation_path",
    "parser_mode",
    "track_never_crossing",
    "num_crossing_transitions",
    "negative_event_policy",
}


@dataclass
class StudentLoaderReport:
    """Diagnostics from split-specific feature loading and preprocessing."""

    config_path: str = ""
    student_name: str = ""
    feature_table_paths: dict[str, str] = field(default_factory=dict)
    row_counts: dict[str, int] = field(default_factory=dict)
    feature_count: int = 0
    feature_columns: list[str] = field(default_factory=list)
    metadata_columns: list[str] = field(default_factory=list)
    nan_before: dict[str, int] = field(default_factory=dict)
    nan_after: dict[str, int] = field(default_factory=dict)
    feature_nan_before: dict[str, dict[str, int]] = field(default_factory=dict)
    feature_nan_after: dict[str, dict[str, int]] = field(default_factory=dict)
    validation_messages: list[str] = field(default_factory=list)


class TabularSequenceDataset(Dataset):
    """Torch dataset holding frozen sequence ids, numeric features, labels, and metadata."""

    def __init__(self, frame: pd.DataFrame, feature_columns: list[str], metadata_columns: list[str] | None = None) -> None:
        self.sequence_ids = frame["sequence_id"].astype(str).tolist()
        self.labels = frame["label"].astype(float).to_numpy()
        self.features = frame[feature_columns].astype(float).to_numpy()
        self.metadata = frame[metadata_columns or []].copy() if metadata_columns else pd.DataFrame(index=frame.index)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        x = torch.tensor(self.features[index], dtype=torch.float32)
        y = torch.tensor(self.labels[index], dtype=torch.float32)
        return x, y, self.sequence_ids[index]


def load_split_csvs(splits_dir: Path) -> dict[str, pd.DataFrame]:
    """Load frozen train/val/test sequence CSVs."""
    result = {}
    for split in SPLITS:
        path = splits_dir / f"{split}_sequences.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing frozen split file: {path}")
        df = pd.read_csv(path)
        if "sequence_id" not in df.columns:
            raise ValueError(f"Frozen split file {path} is missing sequence_id.")
        if df["sequence_id"].duplicated().any():
            raise ValueError(f"Frozen split file {path} contains duplicate sequence_id values.")
        result[split] = df
    return result


def load_feature_table(path: Path) -> pd.DataFrame:
    """Load one feature table and validate sequence_id presence/uniqueness."""
    if not path.exists():
        raise FileNotFoundError(f"Missing feature table: {path}")
    df = pd.read_csv(path)
    if "sequence_id" not in df.columns:
        raise ValueError(f"Feature table {path} is missing sequence_id.")
    if df["sequence_id"].duplicated().any():
        duplicates = df.loc[df["sequence_id"].duplicated(), "sequence_id"].head().tolist()
        raise ValueError(f"Feature table {path} contains duplicate sequence_id values, sample={duplicates}")
    return df


def resolve_split_feature_paths(config: dict[str, Any]) -> dict[str, dict[str, Path]]:
    """Resolve feature table paths as feature_name -> split -> path."""
    selected = set(config.get("features", []))
    configured = config.get("feature_tables", {})
    resolved: dict[str, dict[str, Path]] = {}
    for feature_name in selected:
        if feature_name not in configured:
            raise ValueError(f"Selected feature {feature_name!r} has no feature_tables entry.")
        value = configured[feature_name]
        if isinstance(value, dict):
            missing = [split for split in SPLITS if split not in value]
            if missing:
                raise ValueError(f"Feature {feature_name!r} is missing split-specific paths for {missing}")
            resolved[feature_name] = {split: Path(value[split]) for split in SPLITS}
        else:
            raise ValueError(
                f"Feature {feature_name!r} must use split-specific paths, e.g. train/val/test CSVs. Got {value!r}."
            )
    if not resolved:
        raise ValueError("No selected feature tables configured.")
    return resolved


def _feature_columns_from_table(feature_df: pd.DataFrame) -> list[str]:
    candidates = [col for col in feature_df.columns if col not in FEATURE_METADATA_COLUMNS]
    numeric = []
    for col in candidates:
        if pd.api.types.is_numeric_dtype(feature_df[col]):
            numeric.append(col)
    return numeric


def _prefixed_feature_frame(feature_name: str, feature_df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    out = feature_df[["sequence_id", *feature_columns]].copy()
    rename = {col: f"{feature_name}__{col}" for col in feature_columns}
    return out.rename(columns=rename)


def join_split_features(
    split_name: str,
    split_df: pd.DataFrame,
    feature_tables: dict[str, pd.DataFrame],
    feature_columns_by_name: dict[str, list[str]],
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Join split-specific feature tables to one frozen split by sequence_id."""
    merged = split_df.copy()
    joined_feature_columns: list[str] = []
    for feature_name, table in feature_tables.items():
        if len(table) != len(split_df):
            raise ValueError(f"{split_name}/{feature_name}: row count mismatch split={len(split_df)} feature={len(table)}")
        split_ids = set(split_df["sequence_id"].astype(str))
        feature_ids = set(table["sequence_id"].astype(str))
        if split_ids != feature_ids:
            missing = sorted(split_ids - feature_ids)[:5]
            extra = sorted(feature_ids - split_ids)[:5]
            raise ValueError(f"{split_name}/{feature_name}: sequence_id mismatch missing={missing} extra={extra}")
        if "label" in table.columns:
            labels = split_df[["sequence_id", "label"]].merge(
                table[["sequence_id", "label"]],
                on="sequence_id",
                suffixes=("_split", "_feature"),
                validate="one_to_one",
            )
            if not (labels["label_split"].astype(int) == labels["label_feature"].astype(int)).all():
                raise ValueError(f"{split_name}/{feature_name}: labels do not match frozen split CSV.")

        feature_columns = feature_columns_by_name[feature_name]
        prefixed = _prefixed_feature_frame(feature_name, table, feature_columns)
        before = len(merged)
        merged = merged.merge(prefixed, on="sequence_id", how="left", validate="one_to_one")
        if len(merged) != before:
            raise ValueError(f"{split_name}/{feature_name}: feature join changed row count.")
        joined_feature_columns.extend([f"{feature_name}__{col}" for col in feature_columns])

    metadata_columns = [col for col in merged.columns if col not in joined_feature_columns]
    return merged, joined_feature_columns, metadata_columns


def load_split_feature_frames(config: dict[str, Any], report: StudentLoaderReport | None = None) -> tuple[dict[str, pd.DataFrame], list[str], list[str]]:
    """Load frozen splits and split-specific feature tables with strict alignment checks."""
    splits = load_split_csvs(Path(config.get("splits_dir", "data/splits")))
    feature_paths = resolve_split_feature_paths(config)
    if report is not None:
        report.student_name = str(config.get("student_name", ""))
        report.feature_table_paths = {
            f"{name}.{split}": str(path)
            for name, split_paths in feature_paths.items()
            for split, path in split_paths.items()
        }

    loaded_tables: dict[str, dict[str, pd.DataFrame]] = {
        split: {name: load_feature_table(paths[split]) for name, paths in feature_paths.items()}
        for split in SPLITS
    }

    feature_columns_by_name: dict[str, list[str]] = {}
    for name in feature_paths:
        train_columns = _feature_columns_from_table(loaded_tables["train"][name])
        if not train_columns:
            raise ValueError(f"Feature table {name!r} has no numeric feature columns in train split.")
        for split in ["val", "test"]:
            split_columns = _feature_columns_from_table(loaded_tables[split][name])
            missing = [col for col in train_columns if col not in split_columns]
            if missing:
                raise ValueError(f"Feature table {name!r}/{split} missing train feature columns: {missing[:10]}")
        feature_columns_by_name[name] = train_columns

    joined: dict[str, pd.DataFrame] = {}
    feature_columns: list[str] | None = None
    metadata_columns: list[str] | None = None
    for split in SPLITS:
        frame, split_feature_columns, split_metadata_columns = join_split_features(
            split,
            splits[split],
            loaded_tables[split],
            feature_columns_by_name,
        )
        joined[split] = frame
        if feature_columns is None:
            feature_columns = split_feature_columns
            metadata_columns = split_metadata_columns
        elif feature_columns != split_feature_columns:
            raise ValueError(f"{split}: feature column order differs from train.")
        if report is not None:
            report.row_counts[split] = int(len(frame))

    assert feature_columns is not None and metadata_columns is not None
    if report is not None:
        report.feature_columns = feature_columns
        report.feature_count = len(feature_columns)
        report.metadata_columns = metadata_columns
        report.validation_messages.extend(
            [
                "Loaded split-specific feature CSVs.",
                "sequence_id exists and is unique in every split and feature table.",
                "Feature tables preserve exact frozen row counts per split.",
                "Labels match between frozen split CSVs and feature CSVs.",
            ]
        )
    return joined, feature_columns, metadata_columns


def fit_train_median_imputer(train_frame: pd.DataFrame, feature_columns: list[str]) -> pd.Series:
    """Fit numeric median imputation values on train only."""
    numeric = train_frame[feature_columns].apply(pd.to_numeric, errors="coerce")
    entirely_nan = numeric.columns[numeric.isna().all()].tolist()
    if entirely_nan:
        raise ValueError(f"Train feature columns are entirely NaN and cannot be imputed: {entirely_nan[:20]}")
    return numeric.median(axis=0)


def apply_median_imputer(frame: pd.DataFrame, feature_columns: list[str], medians: pd.Series) -> pd.DataFrame:
    """Apply train-fitted medians to one split without dropping rows."""
    out = frame.copy()
    numeric = out[feature_columns].apply(pd.to_numeric, errors="coerce")
    out[feature_columns] = numeric.fillna(medians)
    return out


def impute_feature_frames(
    frames: dict[str, pd.DataFrame],
    feature_columns: list[str],
    report: StudentLoaderReport | None = None,
) -> tuple[dict[str, pd.DataFrame], pd.Series]:
    """Fit train-only median imputation and apply it to train/val/test."""
    if report is not None:
        for split, frame in frames.items():
            before = frame[feature_columns].isna().sum()
            report.feature_nan_before[split] = {k: int(v) for k, v in before.items() if int(v) != 0}
            report.nan_before[split] = int(before.sum())

    medians = fit_train_median_imputer(frames["train"], feature_columns)
    imputed = {split: apply_median_imputer(frame, feature_columns, medians) for split, frame in frames.items()}

    if report is not None:
        for split, frame in imputed.items():
            after = frame[feature_columns].isna().sum()
            report.feature_nan_after[split] = {k: int(v) for k, v in after.items() if int(v) != 0}
            report.nan_after[split] = int(after.sum())
        report.validation_messages.append("Fitted numeric median imputation on train only and applied to val/test.")
        report.validation_messages.append("No rows were dropped during preprocessing.")
    return imputed, medians


def fit_train_normalizer(train_frame: pd.DataFrame, feature_columns: list[str]) -> tuple[pd.Series, pd.Series]:
    """Fit mean/std on training data only after imputation."""
    mean = train_frame[feature_columns].astype(float).mean()
    std = train_frame[feature_columns].astype(float).std().replace(0, 1.0).fillna(1.0)
    return mean, std


def apply_normalizer(frame: pd.DataFrame, feature_columns: list[str], mean: pd.Series, std: pd.Series) -> pd.DataFrame:
    """Apply train-fitted normalization statistics to a split frame."""
    out = frame.copy()
    out[feature_columns] = (out[feature_columns].astype(float) - mean) / std
    return out


def load_preprocessed_frames(
    config: dict[str, Any],
    report: StudentLoaderReport | None = None,
    normalize: bool = False,
) -> tuple[dict[str, pd.DataFrame], list[str], list[str], pd.Series]:
    """Load aligned features and apply train-only median imputation.

    Normalization is optional and also train-only. Dry-runs use normalize=False so
    reported NaN counts describe imputation only.
    """
    frames, feature_columns, metadata_columns = load_split_feature_frames(config, report)
    imputed, medians = impute_feature_frames(frames, feature_columns, report)
    if normalize:
        mean, std = fit_train_normalizer(imputed["train"], feature_columns)
        imputed = {split: apply_normalizer(frame, feature_columns, mean, std) for split, frame in imputed.items()}
    return imputed, feature_columns, metadata_columns, medians


def load_datasets(config: dict[str, Any]) -> tuple[dict[str, TabularSequenceDataset], list[str], dict[str, pd.DataFrame]]:
    """Load frozen splits, join split-specific features, impute, normalize, and return Torch datasets."""
    frames, feature_columns, metadata_columns, _ = load_preprocessed_frames(config, normalize=True)
    datasets = {split: TabularSequenceDataset(df, feature_columns, metadata_columns) for split, df in frames.items()}
    return datasets, feature_columns, frames
