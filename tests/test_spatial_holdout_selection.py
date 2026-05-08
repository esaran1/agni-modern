"""Tests for data-driven spatial holdout selection."""

from __future__ import annotations

import pandas as pd
import pytest

from agni_modern.evaluation.spatial_holdout_selection import (
    add_grid_coordinates,
    parse_grid_patch_id,
    scan_contiguous_row_bands,
    select_spatial_holdout,
)


def _grid_df() -> pd.DataFrame:
    rows = []
    for row in range(8):
        for col in range(4):
            for t in range(5):
                rows.append(
                    {
                        "patch_id": f"pilot_{row}_{col}",
                        "reference_date": pd.Timestamp("2023-01-01") + pd.Timedelta(days=t),
                        "y_occ_30d": 1 if row in {2, 3} and col in {1, 2} and t in {1, 3} else 0,
                        "y_sev_available": 1 if row == 3 and col == 2 and t == 3 else 0,
                    }
                )
    return pd.DataFrame(rows)


def test_parse_grid_patch_id() -> None:
    assert parse_grid_patch_id("pilot_12_4") == ("pilot", 12, 4)
    assert parse_grid_patch_id("central_kalimantan_3_10") == ("central_kalimantan", 3, 10)
    assert parse_grid_patch_id("bad-id") is None


def test_add_grid_coordinates_rejects_bad_ids() -> None:
    with pytest.raises(ValueError, match="Some patch IDs"):
        add_grid_coordinates(pd.DataFrame({"patch_id": ["pilot_1_2", "bad"]}))


def test_scan_contiguous_row_bands_finds_positive_band() -> None:
    candidates = scan_contiguous_row_bands(
        _grid_df(),
        target_col="y_occ_30d",
        min_rows=20,
        min_positive=4,
        min_prevalence=0.05,
        min_band_width=1,
        max_band_width=3,
        target_fraction=0.25,
    )
    assert candidates
    best = candidates[0]
    assert best.meets_criteria
    assert best.n_positive >= 4
    assert any(prefix in best.prefixes for prefix in ("pilot_2_", "pilot_3_"))


def test_selector_avoids_zero_positive_edge_rows() -> None:
    df = _grid_df()
    # Highest row is all-negative in this fixture; the selector should not choose
    # a naive edge holdout when a positive-containing band exists.
    selected = select_spatial_holdout(
        df,
        target_col="y_occ_30d",
        min_rows=20,
        min_positive=4,
        min_prevalence=0.05,
        min_band_width=1,
        max_band_width=2,
    )
    assert selected is not None
    assert selected.n_positive > 0
    assert "pilot_7_" not in selected.prefixes
