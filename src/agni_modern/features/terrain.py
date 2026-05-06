"""Static terrain features such as elevation/slope/aspect.

Elevation comes from the SRTM EE adapter as ``terrain_elevation_mean``.
Slope and aspect require multi-pixel neighbourhood computation which
is not yet implemented in the single-point reducer — so they are NOT
injected as fake zeros.
"""

from __future__ import annotations

import pandas as pd


def add_terrain_features(df: pd.DataFrame) -> pd.DataFrame:
    """Pass-through: real terrain features come from the EE adapter."""
    return df
