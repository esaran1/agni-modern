"""Inference pipeline for fire probability, severity, and expected risk outputs.

Loads trained occurrence and severity models, runs them on input features,
and produces the three-column operational output:
  p_fire, severity_conditional, expected_risk

If a calibrator file (``<model>.calibrator.pkl``) exists next to the
occurrence model, it is loaded automatically and applied to raw
probabilities, producing *calibrated* ``p_fire`` and ``expected_risk``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from agni_modern.models.base import ModelWrapper
from agni_modern.models.tabular_baselines import LogisticRegressionWrapper, RandomForestWrapper
from agni_modern.models.xgboost_models import (
    XGBoostOccurrenceWrapper,
    XGBoostSeverityClassifierWrapper,
    XGBoostSeverityRegressorWrapper,
)
from agni_modern.training.calibration import (
    apply_calibrator,
    calibrator_path_for,
    load_calibrator,
    threshold_path_for,
    try_load_threshold,
)
from agni_modern.training.dataset import infer_feature_columns
from agni_modern.training.utils import feature_cols_path_for, load_feature_cols

logger = logging.getLogger(__name__)

_OCCURRENCE_LOADERS: dict[str, type[ModelWrapper]] = {
    "logreg": LogisticRegressionWrapper,
    "random_forest": RandomForestWrapper,
    "xgb_occurrence": XGBoostOccurrenceWrapper,
}

_SEVERITY_LOADERS: dict[str, type[ModelWrapper]] = {
    "xgb_severity_cls": XGBoostSeverityClassifierWrapper,
    "xgb_severity_reg": XGBoostSeverityRegressorWrapper,
}


def load_model(model_name: str, model_path: Path) -> ModelWrapper:
    """Load a trained model wrapper from disk."""
    registry = {**_OCCURRENCE_LOADERS, **_SEVERITY_LOADERS}
    cls = registry.get(model_name)
    if cls is None:
        raise ValueError(
            f"Unknown model '{model_name}'. Available: {sorted(registry.keys())}"
        )
    return cls.load(model_path)


def compute_expected_risk(p_fire: pd.Series, severity_conditional: pd.Series) -> pd.Series:
    """Expected severity risk = P(fire) * conditional severity."""
    return p_fire * severity_conditional


def _resolve_feature_cols(df: pd.DataFrame, model_path: Path | None) -> list[str]:
    """Prefer the feature_cols sidecar saved at training time; fall back to inference.

    Using the sidecar guarantees the inference DataFrame is sliced into
    *exactly* the feature set (and order) the model was fit on, even if the
    Parquet has acquired new columns since training.
    """
    if model_path is not None:
        saved = load_feature_cols(feature_cols_path_for(model_path))
        if saved is not None:
            missing = [c for c in saved if c not in df.columns]
            if missing:
                raise ValueError(
                    "Inference DataFrame is missing features the model was trained on: "
                    f"{missing[:8]}{' ...' if len(missing) > 8 else ''}"
                )
            return saved
    return infer_feature_columns(df)


def run_inference(
    df: pd.DataFrame,
    occurrence_model: ModelWrapper,
    severity_model: ModelWrapper,
    occurrence_model_path: Path | None = None,
) -> pd.DataFrame:
    """Run both models on feature data and return a prediction table.

    Parameters
    ----------
    occurrence_model_path
        If provided, the function checks for a companion ``.calibrator.pkl``
        file and applies isotonic calibration to the raw fire probabilities.

    Returns a DataFrame with columns:
      patch_id, reference_date, p_fire, p_fire_raw (if calibrated),
      severity_conditional, expected_risk
      plus centroid_lat/lon if available.
    """
    feature_cols = _resolve_feature_cols(df, occurrence_model_path)
    features = df[feature_cols]

    p_fire_raw = occurrence_model.predict_proba(features)

    calibrator = None
    if occurrence_model_path is not None:
        cal_path = calibrator_path_for(occurrence_model_path)
        if cal_path.exists():
            calibrator = load_calibrator(cal_path)
            logger.info("Loaded isotonic calibrator from %s", cal_path)

    if calibrator is not None:
        p_fire = apply_calibrator(calibrator, p_fire_raw)
    else:
        p_fire = p_fire_raw

    severity_conditional = severity_model.predict(features)
    severity_conditional = np.clip(severity_conditional, 0.0, None)

    meta_cols = ["patch_id", "reference_date"]
    for optional in ("centroid_lat", "centroid_lon"):
        if optional in df.columns:
            meta_cols.append(optional)

    # --- Deployment threshold (F1-optimal on validation, post-calibration) ---
    threshold = 0.5
    if occurrence_model_path is not None:
        thresh_path = threshold_path_for(occurrence_model_path)
        tmeta = try_load_threshold(thresh_path)
        if tmeta is not None:
            threshold = tmeta["threshold"]
            logger.info(
                "Using deployment threshold %.4f (val F1=%.4f) from %s",
                threshold,
                tmeta["val_f1_at_threshold"],
                thresh_path,
            )

    out = df[meta_cols].copy()
    out["p_fire"] = p_fire
    if calibrator is not None:
        out["p_fire_raw"] = p_fire_raw
    out["fire_alert"] = (np.asarray(p_fire) >= threshold).astype(int)
    out["fire_alert_threshold"] = threshold
    out["severity_conditional"] = severity_conditional
    out["expected_risk"] = compute_expected_risk(out["p_fire"], out["severity_conditional"])
    return out


def build_prediction_table(
    patch_df: pd.DataFrame,
    p_fire: pd.Series,
    severity_conditional: pd.Series,
) -> pd.DataFrame:
    """Create standardized prediction table for downstream map generation."""
    meta_cols = ["patch_id", "reference_date"]
    for optional in ("centroid_lat", "centroid_lon"):
        if optional in patch_df.columns:
            meta_cols.append(optional)

    out = patch_df[meta_cols].copy()
    out["p_fire"] = p_fire.values
    out["severity_conditional"] = severity_conditional.values
    out["expected_risk"] = compute_expected_risk(out["p_fire"], out["severity_conditional"])
    return out
