"""Temporal sampling utilities for reference dates and lookback windows."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd


def generate_reference_dates(start: str, end: str, frequency: str = "7D") -> list[date]:
    """Generate reference dates for patch-date row construction."""
    dates = pd.date_range(start=start, end=end, freq=frequency)
    return [d.date() for d in dates]


def lookback_window(reference_date: date, lookback_days: int) -> tuple[date, date]:
    """Return inclusive lookback date range ending at reference date."""
    start = reference_date - timedelta(days=lookback_days)
    return start, reference_date


def future_label_window(reference_date: date, horizon_days: int) -> tuple[date, date]:
    """Return future interval used for horizon-specific occurrence labels."""
    start = reference_date + timedelta(days=1)
    end = reference_date + timedelta(days=horizon_days)
    return start, end
