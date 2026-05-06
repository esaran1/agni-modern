"""Human-pressure features from WorldPop population and GHSL built environment.

No EE adapters for WorldPop or GHSL are wired up yet.  This module does
NOT inject phantom zero columns — it will compute real features once
those adapters are implemented.
"""

from __future__ import annotations

import pandas as pd


def add_human_pressure_features(df: pd.DataFrame) -> pd.DataFrame:
    """Pass-through until WorldPop/GHSL adapters are wired up."""
    return df
