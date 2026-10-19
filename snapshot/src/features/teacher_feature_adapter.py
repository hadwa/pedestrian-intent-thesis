"""Feature adapter helpers shared by teacher integration code."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def require_sequence_alignment(sequence_df: pd.DataFrame, feature_df: pd.DataFrame, feature_name: str) -> None:
    """Ensure a feature table can be joined to frozen sequences by sequence_id."""
    if "sequence_id" not in feature_df.columns:
        raise ValueError(f"Feature table {feature_name} is missing sequence_id.")
    missing = set(sequence_df["sequence_id"]) - set(feature_df["sequence_id"])
    if missing:
        sample = sorted(missing)[:5]
        raise ValueError(f"Feature table {feature_name} is missing {len(missing)} sequence_ids, sample={sample}")


def load_feature_tables(paths: Iterable[Path]) -> dict[str, pd.DataFrame]:
    """Load named feature tables from CSV paths."""
    tables = {}
    for path in paths:
        tables[path.stem] = pd.read_csv(path)
    return tables
