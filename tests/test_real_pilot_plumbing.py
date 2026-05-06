from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from agni_modern.data.validation import dataset_validation_summary
from agni_modern.labels.occurrence import build_occurrence_labels
from agni_modern.labels.severity import (
    SeverityThresholds,
    build_severity_labels,
    compute_dnbr,
    compute_nbr,
)


# ---------------------------------------------------------------------------
# Occurrence label tests
# ---------------------------------------------------------------------------


def test_occurrence_labels_from_viirs_counts() -> None:
    labels = build_occurrence_labels(
        reference_date=date(2020, 8, 1),
        context={
            "viirs_future_fire_count_7d": 0.0,
            "viirs_future_fire_count_30d": 2.0,
            "viirs_future_fire_count_60d": 3.0,
        },
    )
    assert labels["y_occ_7d"] == 0
    assert labels["y_occ_30d"] == 1
    assert labels["y_occ_60d"] == 1


# ---------------------------------------------------------------------------
# Severity label tests — input level 1: direct dNBR
# ---------------------------------------------------------------------------


def test_severity_from_direct_dnbr_low() -> None:
    labels = build_severity_labels(
        reference_date=date(2020, 8, 1),
        context={"y_occ_30d": 1, "dnbr": 0.15},
    )
    assert labels["y_sev_available"] == 1
    assert labels["y_sev_reg"] == 0.15
    assert labels["y_sev_cls"] == 0  # low (<=0.27)


def test_severity_from_direct_dnbr_moderate() -> None:
    labels = build_severity_labels(
        reference_date=date(2020, 8, 1),
        context={"y_occ_30d": 1, "dnbr": 0.35},
    )
    assert labels["y_sev_available"] == 1
    assert labels["y_sev_reg"] == 0.35
    assert labels["y_sev_cls"] == 1  # moderate (0.27 < x <= 0.44)


def test_severity_from_direct_dnbr_high() -> None:
    labels = build_severity_labels(
        reference_date=date(2020, 8, 1),
        context={"y_occ_30d": 1, "dnbr": 0.60},
    )
    assert labels["y_sev_available"] == 1
    assert labels["y_sev_reg"] == 0.60
    assert labels["y_sev_cls"] == 2  # high (>0.44)


# ---------------------------------------------------------------------------
# Severity label tests — input level 2: pre/post-fire NBR pair
# ---------------------------------------------------------------------------


def test_severity_from_nbr_pair() -> None:
    labels = build_severity_labels(
        reference_date=date(2020, 8, 1),
        context={"y_occ_30d": 1, "prefire_nbr": 0.6, "postfire_nbr": 0.1},
    )
    assert labels["y_sev_available"] == 1
    assert abs(labels["y_sev_reg"] - 0.5) < 1e-6
    assert labels["y_sev_cls"] == 2  # high


# ---------------------------------------------------------------------------
# Severity label tests — input level 3: raw NIR/SWIR2 bands
# ---------------------------------------------------------------------------


def test_severity_from_raw_bands() -> None:
    pre_nir, pre_swir2 = 3000.0, 1000.0
    post_nir, post_swir2 = 1500.0, 2000.0
    expected_dnbr = compute_dnbr(
        compute_nbr(pre_nir, pre_swir2),
        compute_nbr(post_nir, post_swir2),
    )

    labels = build_severity_labels(
        reference_date=date(2020, 8, 1),
        context={
            "y_occ_60d": 1,
            "prefire_nir_mean": pre_nir,
            "prefire_swir2_mean": pre_swir2,
            "postfire_nir_mean": post_nir,
            "postfire_swir2_mean": post_swir2,
        },
    )
    assert labels["y_sev_available"] == 1
    assert abs(labels["y_sev_reg"] - expected_dnbr) < 1e-6


# ---------------------------------------------------------------------------
# Severity label tests — gating on fire occurrence
# ---------------------------------------------------------------------------


def test_severity_no_fire_returns_unavailable() -> None:
    labels = build_severity_labels(
        reference_date=date(2020, 8, 1),
        context={"y_occ_7d": 0, "y_occ_30d": 0, "y_occ_60d": 0, "dnbr": 0.5},
    )
    assert labels["y_sev_available"] == 0
    assert labels["y_sev_reg"] is None
    assert labels["y_sev_cls"] is None


def test_severity_fire_in_any_horizon_triggers() -> None:
    for h in (7, 30, 60):
        ctx = {"dnbr": 0.3}
        ctx[f"y_occ_{h}d"] = 1
        labels = build_severity_labels(reference_date=date(2020, 8, 1), context=ctx)
        assert labels["y_sev_available"] == 1, f"Should trigger for y_occ_{h}d"


def test_severity_missing_occurrence_keys() -> None:
    labels = build_severity_labels(
        reference_date=date(2020, 8, 1),
        context={"dnbr": 0.5},
    )
    assert labels["y_sev_available"] == 0


# ---------------------------------------------------------------------------
# Severity label tests — missing NBR data
# ---------------------------------------------------------------------------


def test_severity_fire_but_no_nbr_data() -> None:
    labels = build_severity_labels(
        reference_date=date(2020, 8, 1),
        context={"y_occ_30d": 1},
    )
    assert labels["y_sev_available"] == 0
    assert labels["y_sev_reg"] is None


def test_severity_nan_nbr_treated_as_missing() -> None:
    labels = build_severity_labels(
        reference_date=date(2020, 8, 1),
        context={"y_occ_30d": 1, "prefire_nbr": float("nan"), "postfire_nbr": 0.1},
    )
    assert labels["y_sev_available"] == 0


# ---------------------------------------------------------------------------
# Severity label tests — clipping and custom thresholds
# ---------------------------------------------------------------------------


def test_severity_dnbr_clipping_upper() -> None:
    labels = build_severity_labels(
        reference_date=date(2020, 8, 1),
        context={"y_occ_30d": 1, "dnbr": 2.0},
        max_dnbr=1.3,
    )
    assert labels["y_sev_reg"] == 1.3


def test_severity_dnbr_clipping_lower() -> None:
    labels = build_severity_labels(
        reference_date=date(2020, 8, 1),
        context={"y_occ_30d": 1, "dnbr": -0.5},
        min_dnbr=-0.1,
    )
    assert labels["y_sev_reg"] == -0.1


def test_severity_custom_thresholds() -> None:
    custom = SeverityThresholds(low_upper=0.20, moderate_upper=0.50)
    labels = build_severity_labels(
        reference_date=date(2020, 8, 1),
        context={"y_occ_30d": 1, "dnbr": 0.25},
        thresholds=custom,
    )
    assert labels["y_sev_cls"] == 1  # 0.20 < 0.25 <= 0.50 → moderate


# ---------------------------------------------------------------------------
# Resolution priority tests
# ---------------------------------------------------------------------------


def test_severity_resolution_prefers_direct_dnbr() -> None:
    labels = build_severity_labels(
        reference_date=date(2020, 8, 1),
        context={
            "y_occ_30d": 1,
            "dnbr": 0.10,
            "prefire_nbr": 0.6,
            "postfire_nbr": 0.1,
        },
    )
    assert abs(labels["y_sev_reg"] - 0.10) < 1e-6


# ---------------------------------------------------------------------------
# Validation summary tests
# ---------------------------------------------------------------------------


def test_dataset_validation_summary_fields() -> None:
    df = pd.DataFrame(
        {
            "patch_id": ["p1", "p2"],
            "reference_date": ["2020-08-01", "2020-08-15"],
            "y_occ_30d": [0, 1],
            "feature_max_timestamp": ["2020-08-01", "2020-08-15"],
        }
    )
    summary = dataset_validation_summary(df)
    assert summary["row_count"] == 2
    assert summary["unique_patches"] == 2
    assert "positive_rate_y_occ_30d" in summary
