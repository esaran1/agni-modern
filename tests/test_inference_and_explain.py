"""Tests for inference pipeline, feature importance extraction, and SHAP."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agni_modern.data.synthetic import SyntheticDataConfig, generate_synthetic_patch_date_table
from agni_modern.evaluation.splits import temporal_holdout_split
from agni_modern.inference.predict import (
    build_prediction_table,
    compute_expected_risk,
    load_model,
    run_inference,
)
from agni_modern.training.dataset import infer_feature_columns
from agni_modern.training.train_tabular import (
    train_tabular_occurrence,
    train_tabular_severity_regression,
)
from agni_modern.visualization.feature_importance import (
    extract_feature_importances,
    plot_feature_importance,
    save_feature_importance_plot,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def trained_models(tmp_path_factory):
    """Train occurrence + severity models once for all tests in this module."""
    tmp = tmp_path_factory.mktemp("models")
    df = generate_synthetic_patch_date_table(
        SyntheticDataConfig(seed=42, num_patches=40, num_reference_dates=72)
    )
    train, val, test = temporal_holdout_split(df, "2021-12-31", "2022-03-31", "2022-06-30")

    occ_path = tmp / "occ.pkl"
    train_tabular_occurrence(
        train, val, test, "xgb_occurrence",
        {"n_estimators": 30, "max_depth": 3},
        occ_path, tmp / "occ.json",
    )

    sev_path = tmp / "sev.pkl"
    train_tabular_severity_regression(
        train, val, test, "xgb_severity_reg",
        {"n_estimators": 30, "max_depth": 3, "objective": "reg:squarederror"},
        sev_path, tmp / "sev.json",
    )

    return {
        "df": df,
        "train": train,
        "test": test,
        "occ_path": occ_path,
        "sev_path": sev_path,
    }


# ---------------------------------------------------------------------------
# 9a — Inference pipeline
# ---------------------------------------------------------------------------

def test_load_model(trained_models) -> None:
    occ = load_model("xgb_occurrence", trained_models["occ_path"])
    assert hasattr(occ, "predict_proba")
    sev = load_model("xgb_severity_reg", trained_models["sev_path"])
    assert hasattr(sev, "predict")


def test_load_model_severity_classifier(tmp_path) -> None:
    from agni_modern.models.xgboost_models import XGBoostSeverityClassifierWrapper

    path = tmp_path / "sev_cls.pkl"
    model = XGBoostSeverityClassifierWrapper(
        {"n_estimators": 1, "max_depth": 1, "objective": "multi:softprob", "num_class": 3}
    )
    model.save(path)
    loaded = load_model("xgb_severity_cls", path)
    assert hasattr(loaded, "predict")


def test_load_model_bad_name(tmp_path) -> None:
    with pytest.raises(ValueError, match="Unknown model"):
        load_model("nonexistent_model", tmp_path / "x.pkl")


def test_compute_expected_risk() -> None:
    p = pd.Series([0.8, 0.2, 0.5])
    s = pd.Series([0.6, 0.3, 0.4])
    risk = compute_expected_risk(p, s)
    np.testing.assert_allclose(risk, [0.48, 0.06, 0.20])


def test_run_inference(trained_models) -> None:
    df = trained_models["df"]
    occ = load_model("xgb_occurrence", trained_models["occ_path"])
    sev = load_model("xgb_severity_reg", trained_models["sev_path"])

    pred = run_inference(df.head(50), occ, sev)
    assert "p_fire" in pred.columns
    assert "severity_conditional" in pred.columns
    assert "expected_risk" in pred.columns
    assert "patch_id" in pred.columns
    assert "reference_date" in pred.columns
    assert len(pred) == 50
    assert (pred["p_fire"] >= 0).all() and (pred["p_fire"] <= 1).all()
    assert (pred["severity_conditional"] >= 0).all()
    np.testing.assert_allclose(
        pred["expected_risk"], pred["p_fire"] * pred["severity_conditional"]
    )


def test_run_inference_includes_lat_lon(trained_models) -> None:
    df = trained_models["df"].head(10)
    occ = load_model("xgb_occurrence", trained_models["occ_path"])
    sev = load_model("xgb_severity_reg", trained_models["sev_path"])
    pred = run_inference(df, occ, sev)
    assert "centroid_lat" in pred.columns
    assert "centroid_lon" in pred.columns


def test_build_prediction_table() -> None:
    patch_df = pd.DataFrame({
        "patch_id": ["a", "b"],
        "reference_date": ["2022-01-01", "2022-01-08"],
        "centroid_lat": [1.0, 2.0],
        "centroid_lon": [100.0, 101.0],
    })
    p_fire = pd.Series([0.9, 0.1])
    sev = pd.Series([0.5, 0.3])
    pred = build_prediction_table(patch_df, p_fire, sev)
    assert len(pred) == 2
    assert "expected_risk" in pred.columns
    assert "centroid_lat" in pred.columns


# ---------------------------------------------------------------------------
# 9b — Feature importance
# ---------------------------------------------------------------------------

def test_extract_feature_importances(trained_models) -> None:
    model = load_model("xgb_occurrence", trained_models["occ_path"])
    feature_cols = infer_feature_columns(trained_models["df"])
    imp = extract_feature_importances(model, feature_cols)
    assert isinstance(imp, pd.Series)
    assert len(imp) == len(feature_cols)
    assert (imp >= 0).all()
    assert imp.sum() > 0


def test_plot_feature_importance(trained_models) -> None:
    model = load_model("xgb_occurrence", trained_models["occ_path"])
    feature_cols = infer_feature_columns(trained_models["df"])
    imp = extract_feature_importances(model, feature_cols)
    fig = plot_feature_importance(imp, top_n=10)
    assert fig is not None
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_save_feature_importance_plot(trained_models, tmp_path) -> None:
    model = load_model("xgb_occurrence", trained_models["occ_path"])
    feature_cols = infer_feature_columns(trained_models["df"])
    imp = extract_feature_importances(model, feature_cols)
    out = tmp_path / "fi.png"
    save_feature_importance_plot(imp, out, top_n=10)
    assert out.exists()
    assert out.stat().st_size > 0


# ---------------------------------------------------------------------------
# 9c — SHAP (light test — only 50 samples)
# ---------------------------------------------------------------------------

def test_shap_compute_and_plot(trained_models, tmp_path) -> None:
    from agni_modern.visualization.shap_analysis import compute_shap_values, save_shap_plots

    model = load_model("xgb_occurrence", trained_models["occ_path"])
    feature_cols = infer_feature_columns(trained_models["df"])
    features = trained_models["test"][feature_cols]

    shap_vals = compute_shap_values(model, features, max_samples=50)
    assert shap_vals.values.shape[0] <= 50
    assert shap_vals.values.shape[1] == len(feature_cols)

    saved = save_shap_plots(shap_vals, tmp_path, prefix="test_shap", max_display=10)
    assert len(saved) == 2
    for p in saved:
        assert p.exists()
        assert p.stat().st_size > 0


def test_multiclass_shap_plots_accept_3d_values(tmp_path) -> None:
    from agni_modern.visualization.shap_analysis import save_shap_plots

    values = np.random.default_rng(0).normal(size=(8, 5, 3))
    base_values = np.random.default_rng(1).normal(size=(8, 3))
    data = np.random.default_rng(2).normal(size=(8, 5))
    shap_values = __import__("shap").Explanation(
        values=values,
        base_values=base_values,
        data=data,
        feature_names=[f"f{i}" for i in range(5)],
    )

    saved = save_shap_plots(shap_values, tmp_path, prefix="multiclass", max_display=5)
    assert len(saved) == 2
    for p in saved:
        assert p.exists()
        assert p.stat().st_size > 0
