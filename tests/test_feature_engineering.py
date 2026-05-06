"""Tests for feature engineering modules — spectral indices, pipeline wiring.

Key properties verified:
  - Spectral indices computed from real bands have correct values and ranges
  - Missing bands produce NO phantom columns (not zero-filled)
  - Pass-through modules are truly no-ops on DataFrames without source data
  - infer_feature_columns excludes metadata/coordinate columns
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from agni_modern.features.spectral import build_spectral_features, _find_band_col
from agni_modern.features.vegetation import add_vegetation_indices
from agni_modern.features.burn_indices import add_burn_related_features
from agni_modern.features.weather import add_weather_aggregations
from agni_modern.features.terrain import add_terrain_features
from agni_modern.features.landcover import add_landcover_features
from agni_modern.features.human_pressure import add_human_pressure_features
from agni_modern.features.rolling_windows import add_temporal_rolling_features
from agni_modern.training.dataset import infer_feature_columns


def _make_s2_df(n: int = 20) -> pd.DataFrame:
    """Create a DataFrame mimicking real EE Sentinel-2 adapter output."""
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "patch_id": [f"p_{i}" for i in range(n)],
        "reference_date": pd.date_range("2023-08-01", periods=n, freq="D"),
        "optical_b2_mean_l60d": rng.uniform(100, 2000, n),
        "optical_b3_mean_l60d": rng.uniform(200, 2500, n),
        "optical_b4_mean_l60d": rng.uniform(200, 3000, n),
        "optical_b8_mean_l60d": rng.uniform(1000, 5000, n),
        "optical_b11_mean_l60d": rng.uniform(500, 3000, n),
        "optical_b12_mean_l60d": rng.uniform(300, 2500, n),
        "optical_s2_observation_count": rng.integers(0, 20, n).astype(float),
    })


class TestSpectralFeatures:
    def test_ndvi_computed(self):
        df = _make_s2_df()
        out = build_spectral_features(df)
        assert "optical_ndvi" in out.columns
        assert out["optical_ndvi"].notna().all()
        assert (out["optical_ndvi"] >= -1).all()
        assert (out["optical_ndvi"] <= 1).all()

    def test_ndmi_computed(self):
        df = _make_s2_df()
        out = build_spectral_features(df)
        assert "optical_ndmi" in out.columns
        assert (out["optical_ndmi"] >= -1).all()
        assert (out["optical_ndmi"] <= 1).all()

    def test_evi_computed(self):
        df = _make_s2_df()
        out = build_spectral_features(df)
        assert "optical_evi" in out.columns
        assert out["optical_evi"].notna().all()

    def test_nbr_prefire_computed(self):
        df = _make_s2_df()
        out = build_spectral_features(df)
        assert "optical_nbr_prefire" in out.columns
        assert (out["optical_nbr_prefire"] >= -1).all()
        assert (out["optical_nbr_prefire"] <= 1).all()

    def test_s2_signal_available(self):
        df = _make_s2_df()
        out = build_spectral_features(df)
        assert "optical_s2_signal_available" in out.columns

    def test_ndvi_formula_correctness(self):
        df = pd.DataFrame({
            "optical_b4_mean_l60d": [1000.0],
            "optical_b8_mean_l60d": [3000.0],
        })
        out = build_spectral_features(df)
        expected = (3000 - 1000) / (3000 + 1000)
        assert abs(out["optical_ndvi"].iloc[0] - expected) < 1e-6

    def test_missing_bands_produce_no_phantom_columns(self):
        """When bands are absent, spectral indices should NOT be created."""
        df = pd.DataFrame({"patch_id": ["p0"], "some_other_col": [1.0]})
        out = build_spectral_features(df)
        assert "optical_ndvi" not in out.columns
        assert "optical_nbr_prefire" not in out.columns

    def test_nan_band_propagates_nan(self):
        """When a band has NaN for a row, the derived index should be NaN."""
        df = pd.DataFrame({
            "optical_b4_mean_l60d": [1000.0, np.nan],
            "optical_b8_mean_l60d": [3000.0, 4000.0],
        })
        out = build_spectral_features(df)
        assert out["optical_ndvi"].notna().iloc[0]
        assert pd.isna(out["optical_ndvi"].iloc[1])

    def test_find_band_col(self):
        df = _make_s2_df()
        assert _find_band_col(df, "b8") == "optical_b8_mean_l60d"
        assert _find_band_col(df, "b99") is None


class TestPassThroughModules:
    """Verify that stub modules are true pass-throughs — no phantom columns."""

    def test_vegetation_passthrough(self):
        df = pd.DataFrame({"x": [1.0]})
        out = add_vegetation_indices(df)
        assert list(out.columns) == ["x"]

    def test_burn_passthrough(self):
        df = pd.DataFrame({"x": [1.0]})
        out = add_burn_related_features(df)
        assert list(out.columns) == ["x"]

    def test_weather_passthrough(self):
        df = pd.DataFrame({"x": [1.0]})
        out = add_weather_aggregations(df)
        assert list(out.columns) == ["x"]

    def test_terrain_passthrough(self):
        df = pd.DataFrame({"x": [1.0]})
        out = add_terrain_features(df)
        assert list(out.columns) == ["x"]

    def test_landcover_passthrough(self):
        df = pd.DataFrame({"x": [1.0]})
        out = add_landcover_features(df)
        assert list(out.columns) == ["x"]

    def test_human_pressure_passthrough(self):
        df = pd.DataFrame({"x": [1.0]})
        out = add_human_pressure_features(df)
        assert list(out.columns) == ["x"]


class TestRollingWindows:
    """Verify backward-looking rolling features."""

    def test_noop_without_candidate_cols(self):
        df = pd.DataFrame({
            "patch_id": ["a", "a", "b"],
            "reference_date": ["2023-01-01", "2023-01-02", "2023-01-01"],
            "foo": [1, 2, 3],
        })
        out = add_temporal_rolling_features(df)
        assert not any(c.startswith("temporal_") for c in out.columns)

    def test_produces_rolling_when_candidates_present(self):
        df = pd.DataFrame({
            "patch_id": ["a", "a", "a", "a"],
            "reference_date": pd.date_range("2023-08-01", periods=4, freq="D"),
            "optical_ndvi": [0.3, 0.5, 0.4, 0.6],
        })
        out = add_temporal_rolling_features(df)
        rolling_cols = [c for c in out.columns if c.startswith("temporal_")]
        assert len(rolling_cols) > 0
        assert "temporal_optical_ndvi_rmean_3" in rolling_cols

    def test_no_placeholder_columns(self):
        """The old placeholder columns should NOT exist."""
        df = pd.DataFrame({
            "patch_id": ["a"], "reference_date": ["2023-01-01"],
            "optical_ndvi": [0.5],
        })
        out = add_temporal_rolling_features(df)
        assert "temporal_row_index" not in out.columns
        assert "temporal_recent_count_30d_placeholder" not in out.columns


class TestInferFeatureColumns:
    """Verify that infer_feature_columns excludes metadata and coordinates."""

    def test_excludes_bounding_box(self):
        df = pd.DataFrame({
            "static_min_lon": [0.0], "static_max_lat": [1.0],
            "centroid_lat": [0.5], "centroid_lon": [0.5],
            "optical_ndvi": [0.3], "weather_temp": [25.0],
        })
        cols = infer_feature_columns(df)
        assert "static_min_lon" not in cols
        assert "static_max_lat" not in cols
        assert "centroid_lat" not in cols
        assert "centroid_lon" not in cols
        assert "optical_ndvi" in cols
        assert "weather_temp" in cols

    def test_excludes_legacy_placeholder_columns(self):
        df = pd.DataFrame({
            "temporal_row_index": [0, 1],
            "temporal_recent_count_30d_placeholder": [0, 1],
            "temporal_optical_ndvi_rmean_3": [0.1, 0.2],
            "optical_ndvi": [0.3, 0.4],
        })
        cols = infer_feature_columns(df)
        assert "temporal_row_index" not in cols
        assert "temporal_recent_count_30d_placeholder" not in cols
        assert "temporal_optical_ndvi_rmean_3" in cols
        assert "optical_ndvi" in cols

    def test_picks_up_real_prefixes(self):
        df = pd.DataFrame({
            "optical_ndvi": [0.3],
            "weather_temperature_2m_mean_l60d": [25.0],
            "terrain_elevation_mean": [500.0],
            "temporal_optical_ndvi_rmean_3": [0.4],
        })
        cols = infer_feature_columns(df)
        assert len(cols) == 4


class TestPipelineIntegration:
    """Full feature engineering chain as called by export_pipeline."""

    def test_full_chain_produces_derived_features(self):
        df = _make_s2_df()
        df["y_occ_30d"] = 0
        df["y_sev_available"] = 0
        df["y_sev_reg"] = None
        df["y_sev_cls"] = None

        out = build_spectral_features(df)
        out = add_vegetation_indices(out)
        out = add_burn_related_features(out)
        out = add_weather_aggregations(out)
        out = add_terrain_features(out)
        out = add_landcover_features(out)
        out = add_human_pressure_features(out)
        out = add_temporal_rolling_features(out)

        for col in ["optical_ndvi", "optical_ndmi", "optical_evi",
                     "optical_nbr_prefire", "optical_s2_signal_available"]:
            assert col in out.columns, f"Missing derived feature: {col}"

    def test_full_chain_no_phantom_columns(self):
        """Chain should not add any columns not backed by real data."""
        df = pd.DataFrame({
            "patch_id": ["p0"],
            "reference_date": ["2023-08-01"],
        })
        out = build_spectral_features(df)
        out = add_vegetation_indices(out)
        out = add_burn_related_features(out)
        out = add_weather_aggregations(out)
        out = add_terrain_features(out)
        out = add_landcover_features(out)
        out = add_human_pressure_features(out)

        assert set(out.columns) == {"patch_id", "reference_date"}
