"""Spectral feature builders from optical bands.

Computes derived spectral indices from Sentinel-2 band mean columns
produced by the EE adapter (e.g. ``optical_b8_mean_l60d``).

When bands are missing (synthetic mode, heavy cloud cover), derived
indices are set to NaN — NOT zero.  Zero is a valid physical value
(e.g. NDVI = 0 means bare soil), so using it as a missing sentinel
would inject false information.  NaN propagates correctly through
XGBoost (native handling) and sklearn pipelines (SimpleImputer).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Normalized difference ratio that returns NaN when denominator is zero."""
    denom = denominator.replace(0, np.nan)
    return numerator / denom


def _find_band_col(df: pd.DataFrame, band: str) -> str | None:
    """Find the adapter column for a given S2 band (e.g. 'b8')."""
    for col in df.columns:
        if col.startswith(f"optical_{band}_mean_l"):
            return col
    return None


def build_spectral_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived spectral indices from Sentinel-2 band means.

    Only creates columns when the source bands exist.  Does NOT create
    constant-zero placeholder columns — downstream code and models must
    tolerate absent or NaN-valued index columns.
    """
    out = df.copy()

    b2 = _find_band_col(out, "b2")
    b4 = _find_band_col(out, "b4")
    b8 = _find_band_col(out, "b8")
    b11 = _find_band_col(out, "b11")
    b12 = _find_band_col(out, "b12")

    if b8 and b4:
        nir = out[b8]
        red = out[b4]
        out["optical_ndvi"] = _safe_ratio(nir - red, nir + red)

    if b8 and b11:
        nir = out[b8]
        swir1 = out[b11]
        out["optical_ndmi"] = _safe_ratio(nir - swir1, nir + swir1)

    if b8 and b4 and b2:
        nir = out[b8]
        red = out[b4]
        blue = out[b2]
        denom = nir + 6.0 * red - 7.5 * blue + 1.0
        out["optical_evi"] = 2.5 * _safe_ratio(nir - red, denom)

    if b8 and b12:
        nir = out[b8]
        swir2 = out[b12]
        out["optical_nbr_prefire"] = _safe_ratio(nir - swir2, nir + swir2)

    obs_col = [c for c in out.columns if c == "optical_s2_observation_count"]
    if obs_col:
        out["optical_s2_signal_available"] = (out[obs_col[0]] > 0).astype(int)

    return out
