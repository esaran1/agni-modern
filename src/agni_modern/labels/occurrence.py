"""Future-window occurrence label generation from active fire detections."""

from __future__ import annotations

from datetime import date

from agni_modern.data.temporal_sampling import future_label_window


def _label_from_count(value: object) -> int:
    try:
        return int(float(value) > 0.0)
    except (TypeError, ValueError):
        return 0


def build_occurrence_labels(reference_date: date, context: dict[str, object]) -> dict[str, object]:
    """Build binary occurrence labels for 7/30/60-day horizons.

    Uses VIIRS-derived future counts from context when present; otherwise
    falls back to zero labels for placeholder/synthetic-incomplete paths.
    """
    start_60, end_60 = future_label_window(reference_date, horizon_days=60)

    c7 = context.get("viirs_future_fire_count_7d")
    c30 = context.get("viirs_future_fire_count_30d")
    c60 = context.get("viirs_future_fire_count_60d")

    return {
        "y_occ_7d": _label_from_count(c7),
        "y_occ_30d": _label_from_count(c30),
        "y_occ_60d": _label_from_count(c60),
        "label_window_start": start_60,
        "label_window_end": end_60,
    }
