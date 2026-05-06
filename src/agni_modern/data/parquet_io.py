"""Parquet read/write helpers and schema/leakage validation checks."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from agni_modern.data.contracts import required_columns


def ensure_required_columns(df: pd.DataFrame) -> None:
    """Raise if required canonical columns are missing."""
    missing = [col for col in required_columns() if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def enforce_unique_patch_date(df: pd.DataFrame) -> None:
    """Ensure one row per (patch_id, reference_date)."""
    dupes = df.duplicated(subset=["patch_id", "reference_date"]).sum()
    if dupes > 0:
        raise ValueError(f"Found {dupes} duplicate (patch_id, reference_date) rows")


def enforce_leakage_guard(df: pd.DataFrame) -> None:
    """Ensure feature timestamps do not exceed reference date."""
    ref = pd.to_datetime(df["reference_date"])
    feat_max = pd.to_datetime(df["feature_max_timestamp"])
    invalid = (feat_max > ref).sum()
    if invalid > 0:
        raise ValueError(f"Leakage guard failed for {invalid} rows")


def write_partitioned_parquet(df: pd.DataFrame, output_path: str | Path, partition_cols: list[str] | None = None) -> None:
    """Persist a DataFrame as Parquet, optionally partitioned by columns."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if partition_cols:
        for col in partition_cols:
            if col not in df.columns:
                raise ValueError(f"Partition column '{col}' missing in dataframe")

    if partition_cols:
        df.to_parquet(path, index=False, partition_cols=partition_cols)
    else:
        df.to_parquet(path, index=False)


def read_parquet(path: str | Path) -> pd.DataFrame:
    """Load Parquet file/dataset into pandas DataFrame."""
    return pd.read_parquet(Path(path))
