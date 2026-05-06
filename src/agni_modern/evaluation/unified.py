"""Unified evaluation: load prediction Parquets, compute full metrics, build comparison."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve  # type: ignore

from agni_modern.evaluation.calibration import expected_calibration_error
from agni_modern.evaluation.metrics_occurrence import occurrence_metrics
from agni_modern.evaluation.metrics_severity import (
    severity_classification_metrics,
    severity_regression_metrics,
)
from agni_modern.evaluation.risk_ranking import topk_recall

_SEV_CLS_HINTS = ("severity_cls", "sev_cls")
_SEV_REG_HINTS = ("severity_reg", "sev_reg")


def _optimal_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    """Find the probability threshold that maximises F1.

    Returns (best_threshold, best_f1).
    """
    if len(np.unique(y_true)) < 2:
        return 0.5, 0.0
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    with np.errstate(divide="ignore", invalid="ignore"):
        f1_scores = np.where(
            (precision + recall) > 0,
            2 * precision * recall / (precision + recall),
            0.0,
        )
    idx = int(np.argmax(f1_scores[:-1]))
    return float(thresholds[idx]), float(f1_scores[idx])


def infer_task_type(pred_df: pd.DataFrame, model_name: str = "") -> str:
    """Infer task type from prediction columns and model name."""
    if "y_prob" in pred_df.columns:
        return "occurrence"

    hint = model_name.lower()
    if any(h in hint for h in _SEV_CLS_HINTS):
        return "severity_cls"
    if any(h in hint for h in _SEV_REG_HINTS):
        return "severity_reg"

    if "y_pred" in pred_df.columns:
        unique = pred_df["y_pred"].dropna().unique()
        if len(unique) <= 10 and all(float(v) == int(float(v)) for v in unique):
            return "severity_cls"
        return "severity_reg"

    raise ValueError(f"Cannot classify predictions for '{model_name}': missing y_prob and y_pred")


def evaluate_occurrence_predictions(
    pred_df: pd.DataFrame,
    calibration_bins: int = 10,
    topk_values: list[int] | None = None,
) -> dict[str, Any]:
    """Full occurrence evaluation from a saved predictions DataFrame.

    If ``y_prob_calibrated`` exists in *pred_df* (written by
    :func:`train_tabular_occurrence`), calibrated-ECE and
    calibrated-F1 are also included in the output.
    """
    y_true = pred_df["y_true"].to_numpy()
    y_prob = pred_df["y_prob"].to_numpy()
    y_pred = (
        pred_df["y_pred"].to_numpy()
        if "y_pred" in pred_df.columns
        else (y_prob >= 0.5).astype(int)
    )

    metrics: dict[str, Any] = occurrence_metrics(y_true=y_true, y_score=y_prob, y_pred=y_pred)
    metrics["task"] = "occurrence"
    metrics["ece_raw"] = expected_calibration_error(y_true=y_true, y_prob=y_prob, bins=calibration_bins)
    metrics["n_samples"] = len(pred_df)
    metrics["prevalence"] = float(y_true.mean())

    # Oracle thresholds: maximising F1 on *this* slice (typically test) — not for deployment.
    oracle_t_raw, oracle_f_raw = _optimal_f1_threshold(y_true, y_prob)
    metrics["optimal_threshold"] = oracle_t_raw
    metrics["optimal_f1"] = oracle_f_raw
    metrics["oracle_threshold_raw"] = oracle_t_raw
    metrics["oracle_f1_raw"] = oracle_f_raw

    for k in topk_values or [100, 500]:
        if k <= len(y_true):
            metrics[f"topk_recall_{k}"] = topk_recall(y_true=y_true, risk_score=y_prob, k=k)

    if "y_prob_calibrated" in pred_df.columns:
        y_prob_cal = pred_df["y_prob_calibrated"].to_numpy()
        metrics["ece_calibrated"] = expected_calibration_error(
            y_true=y_true, y_prob=y_prob_cal, bins=calibration_bins
        )
        y_pred_cal = (y_prob_cal >= 0.5).astype(int)
        cal_metrics = occurrence_metrics(y_true=y_true, y_score=y_prob_cal, y_pred=y_pred_cal)
        metrics["f1_calibrated"] = cal_metrics["f1"]
        metrics["roc_auc_calibrated"] = cal_metrics["roc_auc"]

        ot_cal, of_cal = _optimal_f1_threshold(y_true, y_prob_cal)
        metrics["oracle_threshold_calibrated"] = ot_cal
        metrics["oracle_f1_calibrated"] = of_cal

        if "occurrence_decision_threshold" in pred_df.columns:
            deploy_t = float(pred_df["occurrence_decision_threshold"].iloc[0])
            metrics["deployment_threshold"] = deploy_t
            y_deploy = (y_prob_cal >= deploy_t).astype(int)
            m_dep = occurrence_metrics(y_true=y_true, y_score=y_prob_cal, y_pred=y_deploy)
            metrics["f1_calibrated_deployment"] = m_dep["f1"]

    return metrics


def evaluate_severity_cls_predictions(pred_df: pd.DataFrame) -> dict[str, Any]:
    """Full severity classification evaluation."""
    y_true = pred_df["y_true"].astype(int).to_numpy()
    y_pred = pred_df["y_pred"].astype(int).to_numpy()

    metrics: dict[str, Any] = severity_classification_metrics(y_true, y_pred)
    metrics["task"] = "severity_cls"
    metrics["n_samples"] = len(pred_df)

    classes = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    for c in classes:
        tp = int(((y_true == c) & (y_pred == c)).sum())
        fp = int(((y_true != c) & (y_pred == c)).sum())
        fn = int(((y_true == c) & (y_pred != c)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        metrics[f"class_{c}_precision"] = prec
        metrics[f"class_{c}_recall"] = rec
        metrics[f"class_{c}_support"] = int((y_true == c).sum())

    return metrics


def evaluate_severity_reg_predictions(pred_df: pd.DataFrame) -> dict[str, Any]:
    """Full severity regression evaluation."""
    y_true = pred_df["y_true"].to_numpy().astype(float)
    y_pred = pred_df["y_pred"].to_numpy().astype(float)

    metrics: dict[str, Any] = severity_regression_metrics(y_true, y_pred)
    metrics["task"] = "severity_reg"
    metrics["n_samples"] = len(pred_df)
    metrics["y_true_mean"] = float(np.mean(y_true))
    metrics["y_true_std"] = float(np.std(y_true))
    metrics["y_pred_mean"] = float(np.mean(y_pred))
    metrics["y_pred_std"] = float(np.std(y_pred))
    if len(y_true) > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        metrics["correlation"] = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        metrics["correlation"] = 0.0

    return metrics


def evaluate_combined_risk(
    occ_pred_df: pd.DataFrame,
    sev_pred_df: pd.DataFrame,
    topk_values: list[int] | None = None,
) -> dict[str, Any]:
    """Evaluate combined expected risk = P(fire) * severity on joined rows.

    Joins on (patch_id, reference_date). Only rows present in both DataFrames
    are included — typically the fire-positive subset of the test set.
    """
    merged = occ_pred_df.merge(
        sev_pred_df[["patch_id", "reference_date", "y_pred"]].rename(
            columns={"y_pred": "sev_pred"}
        ),
        on=["patch_id", "reference_date"],
        how="inner",
    )

    if merged.empty:
        return {"task": "combined_risk", "combined_risk_n": 0}

    prob_col = "y_prob_calibrated" if "y_prob_calibrated" in merged.columns else "y_prob"
    expected_risk = (merged[prob_col] * merged["sev_pred"]).to_numpy()
    y_true = merged["y_true"].to_numpy()

    metrics: dict[str, Any] = {
        "task": "combined_risk",
        "combined_risk_n": len(merged),
        "expected_risk_mean": float(np.mean(expected_risk)),
        "expected_risk_std": float(np.std(expected_risk)),
    }

    for k in topk_values or [100, 500]:
        if k <= len(y_true):
            metrics[f"combined_topk_recall_{k}"] = topk_recall(
                y_true=y_true, risk_score=expected_risk, k=k
            )

    return metrics


def evaluate_spatial_subset(
    pred_df: pd.DataFrame,
    patch_prefixes: list[str],
    task_type: str,
    calibration_bins: int = 10,
) -> dict[str, Any]:
    """Evaluate on a spatial subset of predictions filtered by patch_id prefix."""
    mask = pred_df["patch_id"].astype(str).str.startswith(tuple(patch_prefixes))
    subset = pred_df[mask]

    if subset.empty:
        return {"spatial_holdout_n": 0}

    if task_type == "occurrence":
        metrics = evaluate_occurrence_predictions(subset, calibration_bins=calibration_bins, topk_values=[])
    elif task_type == "severity_cls":
        metrics = evaluate_severity_cls_predictions(subset)
    else:
        metrics = evaluate_severity_reg_predictions(subset)

    return {f"spatial_{k}": v for k, v in metrics.items()}


def discover_prediction_files(
    metrics_dir: Path,
    experiment_name: str | None = None,
) -> list[tuple[str, Path]]:
    """Find prediction Parquet files, returning (model_name, path) pairs.

    If *experiment_name* is given, only files matching
    ``<experiment_name>_<model_name>_predictions.parquet`` are returned.
    Otherwise all ``*_predictions.parquet`` files are scanned.
    """
    pattern = (
        f"{experiment_name}_*_predictions.parquet"
        if experiment_name
        else "*_predictions.parquet"
    )
    results: list[tuple[str, Path]] = []
    for p in sorted(metrics_dir.glob(pattern)):
        stem = p.stem.replace("_predictions", "")
        if experiment_name and stem.startswith(experiment_name + "_"):
            model_name = stem[len(experiment_name) + 1 :]
        else:
            model_name = stem
        results.append((model_name, p))
    return results


def build_comparison_table(all_results: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Build a single comparison DataFrame from all model evaluation dicts."""
    rows = []
    for model_name, metrics in all_results.items():
        row: dict[str, Any] = {"model": model_name}
        row.update(metrics)
        rows.append(row)
    df = pd.DataFrame(rows)
    if "model" in df.columns:
        df = df.set_index("model")
    return df
