"""Tests for research diagnostic curve generation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from agni_modern.visualization.model_curves import (
    plot_cumulative_gain_curves,
    plot_precision_recall_curves,
    plot_roc_curves,
    save_all_research_curves,
)


def _occ_predictions(n: int = 120) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(42)
    y_true = rng.binomial(1, 0.25, size=n)
    y_prob = np.clip(0.2 + 0.55 * y_true + rng.normal(0, 0.18, size=n), 0, 1)
    y_cal = np.clip(0.1 + 0.8 * y_prob, 0, 1)
    return {
        "xgb_occurrence": pd.DataFrame(
            {
                "patch_id": [f"pilot_{i % 8}_{i % 4}" for i in range(n)],
                "reference_date": pd.date_range("2023-01-01", periods=n, freq="D"),
                "y_true": y_true,
                "y_prob": y_prob,
                "y_prob_calibrated": y_cal,
                "y_pred": (y_prob >= 0.5).astype(int),
                "occurrence_decision_threshold": 0.42,
            }
        )
    }


def _severity_predictions(n: int = 40) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(7)
    y_true = rng.uniform(0.0, 0.5, size=n)
    return {
        "xgb_severity_reg": pd.DataFrame(
            {
                "patch_id": [f"pilot_{i % 8}_{i % 4}" for i in range(n)],
                "reference_date": pd.date_range("2023-01-01", periods=n, freq="D"),
                "y_true": y_true,
                "y_pred": np.clip(y_true + rng.normal(0, 0.05, size=n), 0, None),
            }
        )
    }


def test_roc_curve_plot_and_table(tmp_path: Path) -> None:
    out, table = plot_roc_curves(_occ_predictions(), tmp_path / "roc.png")
    assert out.exists()
    assert out.stat().st_size > 0
    assert {"model", "score", "curve", "x", "y", "threshold", "metric"} <= set(
        table.columns
    )
    assert set(table["score"]) == {"raw", "calibrated"}


def test_precision_recall_curve_plot_and_table(tmp_path: Path) -> None:
    out, table = plot_precision_recall_curves(_occ_predictions(), tmp_path / "pr.png")
    assert out.exists()
    assert out.stat().st_size > 0
    assert not table.empty
    assert (table["curve"] == "precision_recall").all()


def test_cumulative_gain_curve_plot_and_table(tmp_path: Path) -> None:
    out, table = plot_cumulative_gain_curves(_occ_predictions(), tmp_path / "gain.png")
    assert out.exists()
    assert out.stat().st_size > 0
    assert not table.empty
    assert table["fire_capture_rate"].between(0, 1).all()


def test_save_all_research_curves(tmp_path: Path) -> None:
    artifacts = save_all_research_curves(
        occurrence_predictions=_occ_predictions(),
        severity_predictions=_severity_predictions(),
        output_dir=tmp_path,
        prefix="pilot_test",
        calibration_bins=5,
    )
    assert len(artifacts.plots) == 6
    assert len(artifacts.tables) == 5
    for path in [*artifacts.plots, *artifacts.tables]:
        assert path.exists()
        assert path.stat().st_size > 0
