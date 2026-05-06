"""Canonical schema contract helpers for processed patch-date data."""

from __future__ import annotations

REQUIRED_KEY_COLUMNS = ["patch_id", "reference_date"]
REQUIRED_LABEL_COLUMNS = [
    "y_occ_7d",
    "y_occ_30d",
    "y_occ_60d",
    "y_sev_reg",
    "y_sev_cls",
    "y_sev_available",
    "label_window_start",
    "label_window_end",
]
REQUIRED_LEAKAGE_COLUMNS = ["feature_max_timestamp"]


def required_columns() -> list[str]:
    """Return the minimum required columns for canonical processed rows."""
    return REQUIRED_KEY_COLUMNS + REQUIRED_LABEL_COLUMNS + REQUIRED_LEAKAGE_COLUMNS
