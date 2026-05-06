"""Severity classification and regression metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error


def severity_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute severity classification metrics."""
    return {"sev_macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0))}


def severity_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute severity regression metrics."""
    return {
        "sev_mae": float(mean_absolute_error(y_true, y_pred)),
        "sev_rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }
