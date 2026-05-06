"""Severity label generation based on pre/post-fire composites and dNBR.

Severity labels are historical training targets, not inference-time features.
Post-fire imagery is permitted here for label construction.

dNBR resolution supports three input levels:
1. Direct ``dnbr`` key in context
2. ``prefire_nbr`` + ``postfire_nbr``
3. Raw bands: ``prefire_nir_mean``, ``prefire_swir2_mean``,
   ``postfire_nir_mean``, ``postfire_swir2_mean``
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Any


@dataclass(slots=True)
class SeverityThresholds:
    """Configurable thresholds for dNBR severity classes.

    Defaults follow the USGS burn severity classification:
      low:      dNBR <= 0.27
      moderate: 0.27 < dNBR <= 0.44
      high:     dNBR > 0.44
    """

    low_upper: float = 0.27
    moderate_upper: float = 0.44


# ---------------------------------------------------------------------------
# Core spectral helpers (unchanged)
# ---------------------------------------------------------------------------


def compute_nbr(nir: float, swir2: float) -> float:
    """Compute Normalized Burn Ratio from NIR and SWIR2."""
    denom = nir + swir2
    if denom == 0:
        return 0.0
    return (nir - swir2) / denom


def compute_dnbr(prefire_nbr: float, postfire_nbr: float) -> float:
    """Compute differenced NBR used as severity proxy."""
    return prefire_nbr - postfire_nbr


def severity_class_from_dnbr(dnbr: float, thresholds: SeverityThresholds | None = None) -> int:
    """Map dNBR to low/moderate/high classes encoded as 0/1/2."""
    thresholds = thresholds or SeverityThresholds()
    if dnbr <= thresholds.low_upper:
        return 0
    if dnbr <= thresholds.moderate_upper:
        return 1
    return 2


# ---------------------------------------------------------------------------
# Internal resolution helpers
# ---------------------------------------------------------------------------


def _safe_float(val: Any) -> float | None:
    """Coerce to float, returning None for missing/invalid values."""
    if val is None:
        return None
    try:
        f = float(val)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


def _resolve_dnbr(context: dict[str, Any]) -> float | None:
    """Attempt to resolve dNBR from context at decreasing levels of processing.

    Level 1 – direct ``dnbr`` value (e.g. pre-computed offline).
    Level 2 – ``prefire_nbr`` + ``postfire_nbr`` pair.
    Level 3 – raw bands: ``prefire_nir_mean``, ``prefire_swir2_mean``,
              ``postfire_nir_mean``, ``postfire_swir2_mean``.
    """
    direct = _safe_float(context.get("dnbr"))
    if direct is not None:
        return direct

    pre_nbr = _safe_float(context.get("prefire_nbr"))
    post_nbr = _safe_float(context.get("postfire_nbr"))
    if pre_nbr is not None and post_nbr is not None:
        return compute_dnbr(pre_nbr, post_nbr)

    pre_nir = _safe_float(context.get("prefire_nir_mean"))
    pre_swir2 = _safe_float(context.get("prefire_swir2_mean"))
    post_nir = _safe_float(context.get("postfire_nir_mean"))
    post_swir2 = _safe_float(context.get("postfire_swir2_mean"))
    if all(v is not None for v in [pre_nir, pre_swir2, post_nir, post_swir2]):
        return compute_dnbr(compute_nbr(pre_nir, pre_swir2), compute_nbr(post_nir, post_swir2))

    return None


def _has_fire_occurrence(context: dict[str, Any]) -> bool:
    """Return True if any occurrence horizon indicates fire."""
    for horizon in (7, 30, 60):
        val = context.get(f"y_occ_{horizon}d")
        if val is not None:
            try:
                if int(val) == 1:
                    return True
            except (TypeError, ValueError):
                pass
    return False


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def build_severity_labels(
    reference_date: date,
    context: dict[str, Any],
    thresholds: SeverityThresholds | None = None,
    min_dnbr: float = -0.1,
    max_dnbr: float = 1.3,
) -> dict[str, Any]:
    """Build severity regression and classification labels from context.

    Labels are conditional on fire occurrence.  If no fire occurred, or if
    dNBR data is not resolvable from the context, the row is marked as
    severity-unavailable (``y_sev_available=0``).

    Parameters
    ----------
    reference_date : date
        The reference date for this patch-date row (reserved for future
        temporal logic such as burn-date proximity weighting).
    context : dict
        Merged adapter outputs plus occurrence labels.  Must already contain
        ``y_occ_{7,30,60}d`` keys (built before severity in the pipeline).
    thresholds : SeverityThresholds, optional
        dNBR class boundaries.
    min_dnbr, max_dnbr : float
        Clipping bounds for the regression target.
    """
    _ = reference_date

    unavailable: dict[str, Any] = {
        "y_sev_reg": None,
        "y_sev_cls": None,
        "y_sev_available": 0,
    }

    if not _has_fire_occurrence(context):
        return unavailable

    dnbr = _resolve_dnbr(context)
    if dnbr is None:
        return unavailable

    thresholds = thresholds or SeverityThresholds()
    clipped = max(min_dnbr, min(dnbr, max_dnbr))

    return {
        "y_sev_reg": float(clipped),
        "y_sev_cls": severity_class_from_dnbr(dnbr, thresholds),
        "y_sev_available": 1,
    }
