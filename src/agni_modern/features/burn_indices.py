"""Burn-related index features (pre-fire context only at inference time).

Pre-fire NBR is computed in ``build_spectral_features`` from S2 bands
B8 and B12.  This module is a future extension point for additional
burn context features (e.g. fire weather index composites).
It does NOT inject zero-filled fallback columns.
"""

from __future__ import annotations

import pandas as pd


def add_burn_related_features(df: pd.DataFrame) -> pd.DataFrame:
    """Pass-through: pre-fire NBR is computed in spectral.py."""
    return df
