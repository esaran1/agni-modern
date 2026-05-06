"""Tests for the unified evaluation suite, calibration plots, and comparison table."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agni_modern.evaluation.splits import auto_spatial_holdout_prefixes
from agni_modern.evaluation.unified import (
    build_comparison_table,
    discover_prediction_files,
    evaluate_combined_risk,
    evaluate_occurrence_predictions,
    evaluate_severity_cls_predictions,
    evaluate_severity_reg_predictions,
    evaluate_spatial_subset,
    infer_task_type,
)
from agni_modern.visualization.calibration_plot import (
    plot_reliability_diagram,
    save_reliability_diagram,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def occ_pred_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 200
    y_true = rng.integers(0, 2, size=n)
    y_prob = np.clip(y_true * 0.6 + rng.normal(0.3, 0.15, size=n), 0, 1)
    return pd.DataFrame({
        "patch_id": [f"patch_{i % 20:03d}" for i in range(n)],
        "reference_date": pd.date_range("2022-01-01", periods=n, freq="D"),
        "y_true": y_true,
        "y_prob": y_prob,
        "y_pred": (y_prob >= 0.5).astype(int),
    })


@pytest.fixture()
def sev_cls_pred_df() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    n = 150
    return pd.DataFrame({
        "patch_id": [f"patch_{i % 15:03d}" for i in range(n)],
        "reference_date": pd.date_range("2022-01-01", periods=n, freq="D"),
        "y_true": rng.integers(0, 3, size=n),
        "y_pred": rng.integers(0, 3, size=n),
    })


@pytest.fixture()
def sev_reg_pred_df() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    n = 150
    y_true = rng.uniform(0.1, 0.9, size=n)
    return pd.DataFrame({
        "patch_id": [f"patch_{i % 15:03d}" for i in range(n)],
        "reference_date": pd.date_range("2022-01-01", periods=n, freq="D"),
        "y_true": y_true,
        "y_pred": y_true + rng.normal(0, 0.05, size=n),
    })


# ---------------------------------------------------------------------------
# Task type inference
# ---------------------------------------------------------------------------

def test_infer_occurrence(occ_pred_df: pd.DataFrame) -> None:
    assert infer_task_type(occ_pred_df) == "occurrence"


def test_infer_severity_cls_by_name(sev_cls_pred_df: pd.DataFrame) -> None:
    assert infer_task_type(sev_cls_pred_df, "xgb_severity_cls") == "severity_cls"


def test_infer_severity_reg_by_name(sev_reg_pred_df: pd.DataFrame) -> None:
    assert infer_task_type(sev_reg_pred_df, "xgb_severity_reg") == "severity_reg"


def test_infer_severity_cls_from_values(sev_cls_pred_df: pd.DataFrame) -> None:
    assert infer_task_type(sev_cls_pred_df) == "severity_cls"


def test_infer_severity_reg_from_values(sev_reg_pred_df: pd.DataFrame) -> None:
    assert infer_task_type(sev_reg_pred_df) == "severity_reg"


# ---------------------------------------------------------------------------
# Occurrence evaluation
# ---------------------------------------------------------------------------

def test_evaluate_occurrence(occ_pred_df: pd.DataFrame) -> None:
    metrics = evaluate_occurrence_predictions(occ_pred_df, topk_values=[50, 100])
    assert metrics["task"] == "occurrence"
    assert "f1" in metrics
    assert "roc_auc" in metrics
    assert "pr_auc" in metrics
    assert "ece_raw" in metrics
    assert metrics["n_samples"] == 200
    assert 0.0 <= metrics["prevalence"] <= 1.0
    assert "topk_recall_50" in metrics
    assert "topk_recall_100" in metrics


def test_occurrence_ece_bounded(occ_pred_df: pd.DataFrame) -> None:
    metrics = evaluate_occurrence_predictions(occ_pred_df)
    assert 0.0 <= metrics["ece_raw"] <= 1.0


def test_evaluate_occurrence_deployment_threshold_from_parquet() -> None:
    rng = np.random.default_rng(0)
    n = 80
    y_true = rng.integers(0, 2, size=n)
    y_prob = rng.uniform(0, 1, size=n)
    y_cal = np.clip(y_prob * 0.8 + 0.1, 0, 1)
    deploy_t = 0.35
    df = pd.DataFrame({
        "patch_id": [f"pilot_{i % 6}_{i % 4}" for i in range(n)],
        "reference_date": pd.date_range("2022-01-01", periods=n, freq="D"),
        "y_true": y_true,
        "y_prob": y_prob,
        "y_pred": (y_prob >= 0.5).astype(int),
        "y_prob_calibrated": y_cal,
        "occurrence_decision_threshold": deploy_t,
    })
    metrics = evaluate_occurrence_predictions(df)
    assert metrics["deployment_threshold"] == deploy_t
    assert "f1_calibrated_deployment" in metrics
    assert 0.0 <= metrics["f1_calibrated_deployment"] <= 1.0


def test_combined_risk_prefers_calibrated_probability() -> None:
    occ = pd.DataFrame({
        "patch_id": ["a"],
        "reference_date": [pd.Timestamp("2022-01-01")],
        "y_true": [1],
        "y_prob": [0.99],
        "y_prob_calibrated": [0.2],
        "y_pred": [1],
    })
    sev = pd.DataFrame({
        "patch_id": ["a"],
        "reference_date": [pd.Timestamp("2022-01-01")],
        "y_pred": [0.5],
    })
    m = evaluate_combined_risk(occ, sev)
    assert m["combined_risk_n"] == 1
    assert abs(m["expected_risk_mean"] - 0.1) < 1e-9


def test_auto_spatial_holdout_prefixes_pilot_grid() -> None:
    patch_ids = pd.Series([f"pilot_{r}_{c}" for r in range(8) for c in range(4)])
    prefixes = auto_spatial_holdout_prefixes(patch_ids, holdout_fraction=0.25)
    assert len(prefixes) >= 1
    assert all(isinstance(p, str) for p in prefixes)


# ---------------------------------------------------------------------------
# Severity classification evaluation
# ---------------------------------------------------------------------------

def test_evaluate_severity_cls(sev_cls_pred_df: pd.DataFrame) -> None:
    metrics = evaluate_severity_cls_predictions(sev_cls_pred_df)
    assert metrics["task"] == "severity_cls"
    assert "sev_macro_f1" in metrics
    assert metrics["n_samples"] == 150
    assert "class_0_precision" in metrics
    assert "class_0_recall" in metrics
    assert "class_0_support" in metrics


# ---------------------------------------------------------------------------
# Severity regression evaluation
# ---------------------------------------------------------------------------

def test_evaluate_severity_reg(sev_reg_pred_df: pd.DataFrame) -> None:
    metrics = evaluate_severity_reg_predictions(sev_reg_pred_df)
    assert metrics["task"] == "severity_reg"
    assert "sev_mae" in metrics
    assert "sev_rmse" in metrics
    assert "correlation" in metrics
    assert metrics["n_samples"] == 150
    assert metrics["sev_mae"] >= 0.0


# ---------------------------------------------------------------------------
# Combined risk
# ---------------------------------------------------------------------------

def test_evaluate_combined_risk(occ_pred_df: pd.DataFrame, sev_reg_pred_df: pd.DataFrame) -> None:
    metrics = evaluate_combined_risk(occ_pred_df, sev_reg_pred_df, topk_values=[10])
    assert metrics["task"] == "combined_risk"
    assert metrics["combined_risk_n"] > 0
    assert "expected_risk_mean" in metrics


def test_combined_risk_empty_join() -> None:
    occ = pd.DataFrame({
        "patch_id": ["a"], "reference_date": ["2022-01-01"], "y_true": [1], "y_prob": [0.8], "y_pred": [1]
    })
    sev = pd.DataFrame({
        "patch_id": ["b"], "reference_date": ["2022-02-01"], "y_true": [0.5], "y_pred": [0.4]
    })
    metrics = evaluate_combined_risk(occ, sev)
    assert metrics["combined_risk_n"] == 0


# ---------------------------------------------------------------------------
# Spatial holdout
# ---------------------------------------------------------------------------

def test_spatial_subset_occurrence(occ_pred_df: pd.DataFrame) -> None:
    metrics = evaluate_spatial_subset(occ_pred_df, ["patch_01"], "occurrence")
    assert "spatial_f1" in metrics or "spatial_holdout_n" in metrics


def test_spatial_subset_empty() -> None:
    df = pd.DataFrame({
        "patch_id": ["a"], "reference_date": ["2022-01-01"], "y_true": [1], "y_prob": [0.8]
    })
    metrics = evaluate_spatial_subset(df, ["zzz_"], "occurrence")
    assert metrics["spatial_holdout_n"] == 0


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def test_discover_prediction_files(tmp_path: Path) -> None:
    (tmp_path / "exp1_logreg_predictions.parquet").write_bytes(b"")
    (tmp_path / "exp1_xgb_predictions.parquet").write_bytes(b"")
    (tmp_path / "exp2_rf_predictions.parquet").write_bytes(b"")
    (tmp_path / "unrelated.json").write_bytes(b"")

    all_preds = discover_prediction_files(tmp_path)
    assert len(all_preds) == 3

    exp1_preds = discover_prediction_files(tmp_path, experiment_name="exp1")
    assert len(exp1_preds) == 2
    names = [n for n, _ in exp1_preds]
    assert "logreg" in names
    assert "xgb" in names


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def test_build_comparison_table() -> None:
    results = {
        "logreg": {"task": "occurrence", "f1": 0.75, "roc_auc": 0.80},
        "xgb_sev_reg": {"task": "severity_reg", "sev_mae": 0.05, "sev_rmse": 0.07},
    }
    table = build_comparison_table(results)
    assert len(table) == 2
    assert "logreg" in table.index
    assert "xgb_sev_reg" in table.index
    assert table.loc["logreg", "f1"] == 0.75


# ---------------------------------------------------------------------------
# Calibration plot
# ---------------------------------------------------------------------------

def test_plot_reliability_diagram() -> None:
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, size=100)
    y_prob = rng.uniform(0, 1, size=100)
    fig = plot_reliability_diagram(y_true, y_prob, bins=5)
    assert fig is not None
    assert len(fig.axes) == 2
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_save_reliability_diagram(tmp_path: Path) -> None:
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, size=100)
    y_prob = rng.uniform(0, 1, size=100)
    out = tmp_path / "cal.png"
    save_reliability_diagram(y_true, y_prob, out, bins=5)
    assert out.exists()
    assert out.stat().st_size > 0


# ---------------------------------------------------------------------------
# End-to-end: synthetic data → train → evaluate from predictions
# ---------------------------------------------------------------------------

def test_end_to_end_evaluation_from_synthetic(tmp_path: Path) -> None:
    """Train occurrence + severity on synthetic data, then evaluate from saved predictions."""
    from agni_modern.data.synthetic import SyntheticDataConfig, generate_synthetic_patch_date_table
    from agni_modern.evaluation.splits import temporal_holdout_split
    from agni_modern.training.train_tabular import (
        train_tabular_occurrence,
        train_tabular_severity_regression,
    )

    df = generate_synthetic_patch_date_table(
        SyntheticDataConfig(seed=42, num_patches=40, num_reference_dates=72)
    )
    train, val, test = temporal_holdout_split(df, "2021-12-31", "2022-03-31", "2022-06-30")

    occ_preds_path = tmp_path / "occ_preds.parquet"
    train_tabular_occurrence(
        train, val, test, "xgb_occurrence", {"n_estimators": 30, "max_depth": 3},
        tmp_path / "occ.pkl", tmp_path / "occ.json",
        output_predictions_path=occ_preds_path,
    )

    sev_preds_path = tmp_path / "sev_preds.parquet"
    train_tabular_severity_regression(
        train, val, test, "xgb_severity_reg",
        {"n_estimators": 30, "max_depth": 3, "objective": "reg:squarederror"},
        tmp_path / "sev.pkl", tmp_path / "sev.json",
        output_predictions_path=sev_preds_path,
    )

    occ_df = pd.read_parquet(occ_preds_path)
    sev_df = pd.read_parquet(sev_preds_path)

    occ_metrics = evaluate_occurrence_predictions(occ_df, topk_values=[50])
    assert occ_metrics["f1"] >= 0.0
    assert "ece_raw" in occ_metrics
    assert "topk_recall_50" in occ_metrics
    assert "occurrence_decision_threshold" in occ_df.columns
    assert "f1_calibrated_deployment" in occ_metrics

    sev_metrics = evaluate_severity_reg_predictions(sev_df)
    assert sev_metrics["sev_mae"] >= 0.0

    risk_metrics = evaluate_combined_risk(occ_df, sev_df, topk_values=[50])
    assert risk_metrics["combined_risk_n"] > 0

    table = build_comparison_table({
        "xgb_occurrence": occ_metrics,
        "xgb_severity_reg": sev_metrics,
        "combined_risk": risk_metrics,
    })
    assert len(table) == 3
