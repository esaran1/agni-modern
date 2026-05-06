"""Land-cover summary feature builders (e.g., Dynamic World fractions).

No EE adapter for Dynamic World is wired up yet.  This module does NOT
inject phantom zero columns — it will compute real fractions once the
DynamicWorld adapter is implemented.
"""

from __future__ import annotations

import pandas as pd


def add_landcover_features(df: pd.DataFrame) -> pd.DataFrame:
    """Pass-through until a DynamicWorld adapter is wired up."""
    return df
