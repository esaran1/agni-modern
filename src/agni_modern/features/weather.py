"""Weather aggregation features from ERA5-Land style variables.

The EE adapter already produces per-variable columns like
``weather_temperature_2m_mean_l60d``.  This module does NOT create
phantom zero-filled columns — it only applies transforms to columns
that actually exist.
"""

from __future__ import annotations

import pandas as pd


def add_weather_aggregations(df: pd.DataFrame, windows: list[int] | None = None) -> pd.DataFrame:
    """Pass-through: real weather features come from the EE adapter.

    Future extension point for cross-variable derived features
    (e.g. vapour pressure deficit from temperature + humidity).
    """
    return df
