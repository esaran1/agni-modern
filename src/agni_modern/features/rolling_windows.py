"""Backward-looking rolling temporal summaries for tabular models.

All windows look strictly backward in time per patch — no future leakage.
Rolling is computed on the full dataset before train/val/test split, which
is safe because each row's rolling stat only depends on that patch's
earlier observations.
"""

from __future__ import annotations

import pandas as pd


_ROLLING_WINDOWS = [3, 7]

_CANDIDATE_COLS = [
    "optical_ndvi",
    "optical_ndmi",
    "optical_nbr_prefire",
]


def add_temporal_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-patch backward rolling mean and std of key features.

    Only operates on columns that actually exist in the DataFrame.
    If none of the candidate columns are present (synthetic/placeholder mode),
    this is a no-op.
    """
    out = df.sort_values(["patch_id", "reference_date"]).reset_index(drop=True)

    present = [c for c in _CANDIDATE_COLS if c in out.columns]
    if not present:
        return out

    for col in present:
        grouped = out.groupby("patch_id")[col]
        for win in _ROLLING_WINDOWS:
            roll = grouped.rolling(window=win, min_periods=1).mean().reset_index(level=0, drop=True)
            out[f"temporal_{col}_rmean_{win}"] = roll
            roll_std = grouped.rolling(window=win, min_periods=1).std().reset_index(level=0, drop=True)
            out[f"temporal_{col}_rstd_{win}"] = roll_std.fillna(0.0)

    return out
