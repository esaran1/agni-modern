"""Tests for probability calibration and winsorized logistic regression.

Key properties verified:
  - Winsorizer clips to fitted quantile bounds
  - Isotonic calibrator maps raw probabilities toward calibrated ones
  - Calibrator is saved/loaded correctly
  - Training path produces calibrated predictions and dual ECE metrics
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agni_modern.data.synthetic import SyntheticDataConfig, generate_synthetic_patch_date_table
from agni_modern.evaluation.splits import temporal_holdout_split
from agni_modern.models.tabular_baselines import Winsorizer
from agni_modern.training.calibration import (
    apply_calibrator,
    calibrator_path_for,
    fit_isotonic_calibrator,
    load_calibrator,
    save_calibrator,
    save_threshold,
    try_load_threshold,
)
from agni_modern.training.train_tabular import train_tabular_occurrence


class TestWinsorizer:
    def test_clips_extreme_values(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(200, 3))
        X[0, 0] = 100.0
        X[1, 1] = -100.0

        w = Winsorizer(lower_quantile=0.01, upper_quantile=0.99)
        w.fit(X)
        Xt = w.transform(X)

        assert Xt[0, 0] < 100.0
        assert Xt[1, 1] > -100.0

    def test_idempotent_within_bounds(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        w = Winsorizer(lower_quantile=0.0, upper_quantile=1.0)
        w.fit(X)
        Xt = w.transform(X)
        np.testing.assert_array_equal(X, Xt)

    def test_nan_handling(self):
        X = np.array([[1.0, np.nan], [2.0, 3.0], [4.0, 5.0]])
        w = Winsorizer()
        w.fit(X)
        Xt = w.transform(X)
        assert np.isnan(Xt[0, 1])


class TestIsotonicCalibrator:
    def test_calibrator_produces_valid_probabilities(self):
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, size=200)
        y_prob = rng.uniform(0, 1, size=200)

        cal = fit_isotonic_calibrator(y_true, y_prob)
        cal_probs = apply_calibrator(cal, y_prob)

        assert cal_probs.min() >= 0.0
        assert cal_probs.max() <= 1.0
        assert len(cal_probs) == len(y_prob)

    def test_save_and_load(self, tmp_path):
        rng = np.random.default_rng(99)
        y_true = rng.integers(0, 2, size=100)
        y_prob = rng.uniform(0, 1, size=100)

        cal = fit_isotonic_calibrator(y_true, y_prob)
        path = tmp_path / "test.calibrator.pkl"
        save_calibrator(cal, path)
        assert path.exists()

        loaded = load_calibrator(path)
        original = apply_calibrator(cal, y_prob)
        reloaded = apply_calibrator(loaded, y_prob)
        np.testing.assert_array_equal(original, reloaded)

    def test_calibrator_path_for(self):
        p = calibrator_path_for(Path("/models/occ.pkl"))
        assert p == Path("/models/occ.calibrator.pkl")

    def test_try_load_threshold_missing(self, tmp_path):
        assert try_load_threshold(tmp_path / "nope.threshold.json") is None

    def test_try_load_threshold_exactly_half(self, tmp_path):
        """Threshold 0.5 must load correctly (regression: must not confuse with 'missing')."""
        path = tmp_path / "m.threshold.json"
        save_threshold(0.5, 0.42, path)
        meta = try_load_threshold(path)
        assert meta is not None
        assert meta["threshold"] == 0.5
        assert meta["val_f1_at_threshold"] == 0.42


class TestCalibratedTraining:
    @pytest.fixture(scope="class")
    def trained(self, tmp_path_factory):
        tmp = tmp_path_factory.mktemp("cal")
        df = generate_synthetic_patch_date_table(
            SyntheticDataConfig(seed=42, num_patches=40, num_reference_dates=72)
        )
        train, val, test = temporal_holdout_split(df, "2021-12-31", "2022-03-31", "2022-06-30")

        model_path = tmp / "occ.pkl"
        metrics_path = tmp / "occ.json"
        preds_path = tmp / "occ_preds.parquet"

        metrics = train_tabular_occurrence(
            train, val, test, "xgb_occurrence",
            {"n_estimators": 30, "max_depth": 3},
            model_path, metrics_path,
            output_predictions_path=preds_path,
        )
        return {
            "metrics": metrics,
            "model_path": model_path,
            "preds_path": preds_path,
        }

    def test_calibrator_file_created(self, trained):
        cal_path = calibrator_path_for(trained["model_path"])
        assert cal_path.exists()

    def test_metrics_contain_dual_ece(self, trained):
        m = trained["metrics"]
        assert "ece_raw" in m
        assert "ece_calibrated" in m
        assert m["ece_raw"] >= 0
        assert m["ece_calibrated"] >= 0

    def test_predictions_have_calibrated_column(self, trained):
        pred_df = pd.read_parquet(trained["preds_path"])
        assert "y_prob" in pred_df.columns
        assert "y_prob_calibrated" in pred_df.columns
        assert (pred_df["y_prob_calibrated"] >= 0).all()
        assert (pred_df["y_prob_calibrated"] <= 1).all()

    def test_logreg_with_winsorizer_trains(self, tmp_path):
        """LogReg with Winsorizer pipeline should train without overflow."""
        df = generate_synthetic_patch_date_table(
            SyntheticDataConfig(seed=7, num_patches=40, num_reference_dates=72, start_date="2021-01-01")
        )
        train, val, test = temporal_holdout_split(df, "2021-12-31", "2022-03-31", "2022-06-30")

        model_path = tmp_path / "logreg.pkl"
        metrics_path = tmp_path / "logreg.json"

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            metrics = train_tabular_occurrence(
                train, val, test, "logreg",
                {"max_iter": 2000},
                model_path, metrics_path,
            )

        assert metrics["f1"] >= 0.0
        assert model_path.exists()
        assert calibrator_path_for(model_path).exists()
