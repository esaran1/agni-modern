"""Dataset validation/reporting helpers for pilot builds."""

from __future__ import annotations

from typing import Any

import pandas as pd


def dataset_validation_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Return compact dataset quality summary for quick pilot checks."""
    if df.empty:
        return {
            "row_count": 0,
            "unique_patches": 0,
            "reference_date_min": None,
            "reference_date_max": None,
            "missingness": {},
            "positive_rate_y_occ_30d": None,
        }

    ref = pd.to_datetime(df["reference_date"])
    missingness = (df.isna().mean().sort_values(ascending=False).head(20)).to_dict()

    return {
        "row_count": int(len(df)),
        "unique_patches": int(df["patch_id"].nunique()),
        "reference_date_min": str(ref.min().date()),
        "reference_date_max": str(ref.max().date()),
        "missingness": {k: float(v) for k, v in missingness.items()},
        "positive_rate_y_occ_30d": float(df["y_occ_30d"].mean()) if "y_occ_30d" in df.columns else None,
    }
