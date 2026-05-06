"""Temporal and spatial holdout split strategies."""

from __future__ import annotations

import re

import pandas as pd


def temporal_holdout_split(df: pd.DataFrame, train_end: str, val_end: str, test_end: str):
    """Split rows by reference_date thresholds."""
    dt = pd.to_datetime(df["reference_date"])
    train = df[dt <= pd.Timestamp(train_end)].copy()
    val = df[(dt > pd.Timestamp(train_end)) & (dt <= pd.Timestamp(val_end))].copy()
    test = df[(dt > pd.Timestamp(val_end)) & (dt <= pd.Timestamp(test_end))].copy()
    return train, val, test


def spatial_holdout_split(df: pd.DataFrame, holdout_patch_prefixes: list[str]):
    """Split rows by patch_id prefixes for spatial generalization tests."""
    mask = df["patch_id"].astype(str).str.startswith(tuple(holdout_patch_prefixes))
    test = df[mask].copy()
    train = df[~mask].copy()
    return train, test


_ROW_RE = re.compile(r"^[^_]+_(\d+)_\d+$")


def auto_spatial_holdout_prefixes(
    patch_ids: pd.Series,
    holdout_fraction: float = 0.25,
) -> list[str]:
    """Auto-discover spatial holdout patch prefixes from grid-structured IDs.

    Expects patch IDs of the form ``<prefix>_<row>_<col>``.  Holds out the
    highest-numbered rows (geographic edge of the AOI) so that the model is
    tested on a contiguous spatial region it has never seen.

    Returns a list of ``<prefix>_<row>_`` strings suitable for
    :func:`spatial_holdout_split` or ``--spatial-holdout`` CLI args.
    """
    unique_ids = patch_ids.astype(str).unique()
    rows: set[int] = set()
    prefix = ""
    for pid in unique_ids:
        m = _ROW_RE.match(pid)
        if m:
            rows.add(int(m.group(1)))
            prefix = pid.rsplit("_", 2)[0]

    if not rows or not prefix:
        return []

    sorted_rows = sorted(rows)
    n_holdout = max(1, int(len(sorted_rows) * holdout_fraction))
    holdout_rows = sorted_rows[-n_holdout:]

    return [f"{prefix}_{r}_" for r in holdout_rows]
