import numpy as np

from agni_modern.evaluation.metrics_occurrence import occurrence_metrics
from agni_modern.evaluation.metrics_severity import (
    severity_classification_metrics,
    severity_regression_metrics,
)


def test_occurrence_metrics() -> None:
    y_true = np.array([0, 1, 0, 1, 1])
    y_score = np.array([0.1, 0.8, 0.4, 0.7, 0.9])
    y_pred = (y_score >= 0.5).astype(int)
    m = occurrence_metrics(y_true, y_score, y_pred)
    assert "f1" in m
    assert "roc_auc" in m


def test_severity_metrics() -> None:
    y_true_cls = np.array([0, 1, 2, 1])
    y_pred_cls = np.array([0, 1, 1, 1])
    m_cls = severity_classification_metrics(y_true_cls, y_pred_cls)
    assert "sev_macro_f1" in m_cls

    y_true_reg = np.array([0.1, 0.3, 0.8])
    y_pred_reg = np.array([0.2, 0.4, 0.7])
    m_reg = severity_regression_metrics(y_true_reg, y_pred_reg)
    assert "sev_mae" in m_reg
    assert "sev_rmse" in m_reg
