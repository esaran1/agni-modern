"""Training utilities for tabular baselines and XGBoost models.

NaN handling is delegated to each model wrapper:
  - XGBoost handles NaN natively (learns optimal missing-value direction)
  - LogReg/RF use Pipeline with SimpleImputer(median) + Winsorizer/Scaler

Occurrence models additionally fit an isotonic probability calibrator on the
validation split and report both raw and calibrated metrics.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from agni_modern.evaluation.calibration import expected_calibration_error
from agni_modern.evaluation.metrics_occurrence import occurrence_metrics
from agni_modern.evaluation.metrics_severity import (
    severity_classification_metrics,
    severity_regression_metrics,
)
from agni_modern.models.tabular_baselines import LogisticRegressionWrapper, RandomForestWrapper
from agni_modern.models.xgboost_models import (
    XGBoostOccurrenceWrapper,
    XGBoostSeverityClassifierWrapper,
    XGBoostSeverityRegressorWrapper,
)
from agni_modern.training.calibration import (
    apply_calibrator,
    calibrator_path_for,
    find_optimal_f1_threshold,
    fit_isotonic_calibrator,
    save_calibrator,
    save_threshold,
    threshold_path_for,
)
from agni_modern.training.dataset import infer_feature_columns
from agni_modern.training.utils import save_metrics, set_global_seed

logger = logging.getLogger(__name__)


def _build_model(model_name: str, model_params: dict[str, object]):
    registry = {
        "logreg": LogisticRegressionWrapper,
        "random_forest": RandomForestWrapper,
        "xgb_occurrence": XGBoostOccurrenceWrapper,
        "xgb_severity_cls": XGBoostSeverityClassifierWrapper,
        "xgb_severity_reg": XGBoostSeverityRegressorWrapper,
    }
    cls = registry.get(model_name)
    if cls is None:
        raise ValueError(f"Unsupported tabular model: {model_name}")
    return cls(model_params)


def train_tabular_occurrence(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_name: str,
    model_params: dict[str, object],
    output_model_path: Path,
    output_metrics_path: Path,
    seed: int = 42,
    target_col: str = "y_occ_30d",
    output_predictions_path: Path | None = None,
) -> dict[str, float]:
    """Train a tabular occurrence model and persist artifacts + metrics.

    Also fits an isotonic calibrator on the validation split and reports
    both raw and calibrated ECE/metrics on the test split.
    """
    set_global_seed(seed)
    feature_cols = infer_feature_columns(train_df)

    if model_name.startswith("xgb") and "scale_pos_weight" not in model_params:
        n_pos = int(train_df[target_col].sum())
        n_neg = len(train_df) - n_pos
        if n_pos > 0:
            model_params = {**model_params, "scale_pos_weight": n_neg / n_pos}

    model = _build_model(model_name, model_params)
    model.fit(train_df, val_df, {"feature_cols": feature_cols, "target_col": target_col})

    # --- Raw test probabilities ---
    x_test = test_df[feature_cols]
    probs_raw = model.predict_proba(x_test)
    y_test = test_df[target_col].to_numpy()

    # --- Fit isotonic calibrator on validation set ---
    x_val = val_df[feature_cols]
    probs_val_raw = model.predict_proba(x_val)
    y_val = val_df[target_col].to_numpy()

    calibrator = fit_isotonic_calibrator(y_val, probs_val_raw)
    cal_path = calibrator_path_for(output_model_path)
    save_calibrator(calibrator, cal_path)

    probs_cal = apply_calibrator(calibrator, probs_raw)

    # --- Optimal threshold (selected on calibrated val probabilities) ---
    probs_val_cal = apply_calibrator(calibrator, probs_val_raw)
    opt_threshold, opt_val_f1 = find_optimal_f1_threshold(y_val, probs_val_cal)
    thresh_path = threshold_path_for(output_model_path)
    save_threshold(opt_threshold, opt_val_f1, thresh_path)

    # --- Metrics: raw (default 0.5 threshold) ---
    preds_raw = (probs_raw >= 0.5).astype(int)
    metrics = occurrence_metrics(y_true=y_test, y_score=probs_raw, y_pred=preds_raw)

    ece_raw = expected_calibration_error(y_true=y_test, y_prob=probs_raw)
    metrics["ece_raw"] = ece_raw

    # --- Metrics: calibrated (default 0.5 threshold) ---
    preds_cal = (probs_cal >= 0.5).astype(int)
    metrics_cal = occurrence_metrics(y_true=y_test, y_score=probs_cal, y_pred=preds_cal)

    ece_cal = expected_calibration_error(y_true=y_test, y_prob=probs_cal)
    metrics["ece_calibrated"] = ece_cal
    metrics["f1_calibrated"] = metrics_cal["f1"]
    metrics["roc_auc_calibrated"] = metrics_cal["roc_auc"]

    # --- Metrics: calibrated @ tuned threshold ---
    preds_tuned = (probs_cal >= opt_threshold).astype(int)
    metrics_tuned = occurrence_metrics(y_true=y_test, y_score=probs_cal, y_pred=preds_tuned)
    metrics["optimal_threshold"] = opt_threshold
    metrics["optimal_val_f1"] = opt_val_f1
    metrics["f1_at_optimal_threshold"] = metrics_tuned["f1"]

    # --- Diagnostics ---
    metrics["n_features"] = len(feature_cols)
    metrics["n_train"] = len(train_df)
    metrics["n_val"] = len(val_df)
    metrics["n_test"] = len(test_df)
    metrics["train_positive_rate"] = float(train_df[target_col].mean())
    metrics["test_positive_rate"] = float(test_df[target_col].mean())
    metrics["train_nan_rate"] = float(train_df[feature_cols].isna().mean().mean())

    model.save(output_model_path)
    save_metrics(metrics, output_metrics_path)

    if output_predictions_path is not None:
        output_predictions_path.parent.mkdir(parents=True, exist_ok=True)
        pred_df = test_df[["patch_id", "reference_date", target_col]].copy()
        pred_df = pred_df.rename(columns={target_col: "y_true"})
        pred_df["y_prob"] = probs_raw
        pred_df["y_prob_calibrated"] = probs_cal
        pred_df["y_pred"] = preds_raw
        pred_df["occurrence_decision_threshold"] = float(opt_threshold)
        pred_df["y_pred_calibrated_tuned"] = preds_tuned.astype(int)
        pred_df.to_parquet(output_predictions_path, index=False)

    return metrics


def _filter_severity_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows with available severity labels."""
    return df[df["y_sev_available"] == 1].copy()


def train_tabular_severity_classification(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_name: str,
    model_params: dict[str, object],
    output_model_path: Path,
    output_metrics_path: Path,
    seed: int = 42,
    target_col: str = "y_sev_cls",
    output_predictions_path: Path | None = None,
) -> dict[str, object]:
    """Train a severity classification model on y_sev_available==1 rows."""
    set_global_seed(seed)
    feature_cols = infer_feature_columns(train_df)

    train_sev = _filter_severity_rows(train_df)
    val_sev = _filter_severity_rows(val_df)
    test_sev = _filter_severity_rows(test_df)

    for label, split in [("train", train_sev), ("val", val_sev), ("test", test_sev)]:
        if len(split) == 0:
            raise ValueError(
                f"No severity-available rows in {label} split. "
                "Check that synthetic data produces fire-positive rows in all temporal splits."
            )

    y_train = train_sev[target_col].astype(int)
    train_sev = train_sev.copy()
    train_sev[target_col] = y_train

    model = _build_model(model_name, model_params)
    model.fit(train_sev, val_sev, {"feature_cols": feature_cols, "target_col": target_col})

    y_pred = model.predict(test_sev[feature_cols])
    y_true = test_sev[target_col].astype(int).to_numpy()
    metrics: dict[str, object] = severity_classification_metrics(y_true, y_pred)
    metrics["n_train"] = len(train_sev)
    metrics["n_val"] = len(val_sev)
    metrics["n_test"] = len(test_sev)
    metrics["n_features"] = len(feature_cols)
    metrics["class_distribution_train"] = dict(train_sev[target_col].value_counts().sort_index())
    metrics["class_distribution_test"] = dict(
        pd.Series(y_true).value_counts().sort_index().to_dict()
    )

    model.save(output_model_path)
    save_metrics(metrics, output_metrics_path)

    if output_predictions_path is not None:
        output_predictions_path.parent.mkdir(parents=True, exist_ok=True)
        pred_df = test_sev[["patch_id", "reference_date", target_col]].copy()
        pred_df = pred_df.rename(columns={target_col: "y_true"})
        pred_df["y_pred"] = y_pred
        pred_df.to_parquet(output_predictions_path, index=False)

    return metrics


def train_tabular_severity_regression(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_name: str,
    model_params: dict[str, object],
    output_model_path: Path,
    output_metrics_path: Path,
    seed: int = 42,
    target_col: str = "y_sev_reg",
    output_predictions_path: Path | None = None,
) -> dict[str, object]:
    """Train a severity regression model on y_sev_available==1 rows."""
    set_global_seed(seed)
    feature_cols = infer_feature_columns(train_df)

    train_sev = _filter_severity_rows(train_df)
    val_sev = _filter_severity_rows(val_df)
    test_sev = _filter_severity_rows(test_df)

    for label, split in [("train", train_sev), ("val", val_sev), ("test", test_sev)]:
        if len(split) == 0:
            raise ValueError(
                f"No severity-available rows in {label} split. "
                "Check that synthetic data produces fire-positive rows in all temporal splits."
            )

    model = _build_model(model_name, model_params)
    model.fit(train_sev, val_sev, {"feature_cols": feature_cols, "target_col": target_col})

    y_pred = model.predict(test_sev[feature_cols])
    y_true = test_sev[target_col].to_numpy()
    metrics: dict[str, object] = severity_regression_metrics(y_true, y_pred)
    metrics["n_train"] = len(train_sev)
    metrics["n_val"] = len(val_sev)
    metrics["n_test"] = len(test_sev)
    metrics["n_features"] = len(feature_cols)
    metrics["y_true_mean"] = float(np.mean(y_true))
    metrics["y_true_std"] = float(np.std(y_true))
    metrics["y_pred_mean"] = float(np.mean(y_pred))
    metrics["y_pred_std"] = float(np.std(y_pred))

    model.save(output_model_path)
    save_metrics(metrics, output_metrics_path)

    if output_predictions_path is not None:
        output_predictions_path.parent.mkdir(parents=True, exist_ok=True)
        pred_df = test_sev[["patch_id", "reference_date", target_col]].copy()
        pred_df = pred_df.rename(columns={target_col: "y_true"})
        pred_df["y_pred"] = y_pred
        pred_df.to_parquet(output_predictions_path, index=False)

    return metrics
