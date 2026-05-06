"""Label sanity checks for consistency, null rates, and leakage alignment."""

from __future__ import annotations

import pandas as pd


def check_label_presence(df: pd.DataFrame) -> dict[str, float]:
    """Return missing-value rates for each label column."""
    label_cols = ["y_occ_7d", "y_occ_30d", "y_occ_60d", "y_sev_reg", "y_sev_cls", "y_sev_available"]
    return {col: float(df[col].isna().mean()) for col in label_cols if col in df.columns}


def check_severity_mask_consistency(df: pd.DataFrame) -> None:
    """Validate severity availability mask against severity targets."""
    if "y_sev_available" not in df.columns:
        raise ValueError("Missing y_sev_available column")

    invalid = df[(df["y_sev_available"] == 0) & (df["y_sev_reg"].notna() | df["y_sev_cls"].notna())]
    if not invalid.empty:
        raise ValueError("Rows with y_sev_available=0 must not contain severity labels")


def check_temporal_label_order(df: pd.DataFrame) -> None:
    """Ensure label window starts after reference date."""
    ref = pd.to_datetime(df["reference_date"])
    start = pd.to_datetime(df["label_window_start"])
    invalid = (start <= ref).sum()
    if invalid > 0:
        raise ValueError(f"Found {invalid} rows with non-future label windows")
