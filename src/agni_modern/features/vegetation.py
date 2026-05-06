"""Vegetation index feature functions.

``build_spectral_features`` computes NDVI/NDMI/EVI from raw bands.
This module is a future extension point for additional derived
vegetation features (e.g. greenness anomalies, phenological timing).
It does NOT inject zero-filled fallback columns.
"""

from __future__ import annotations

import pandas as pd


def add_vegetation_indices(df: pd.DataFrame) -> pd.DataFrame:
    """Pass-through: spectral indices are computed in spectral.py."""
    return df
