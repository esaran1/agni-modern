"""Research-grade diagnostic curves for saved model predictions.

The functions in this module operate on prediction Parquets written by the
training scripts. They intentionally do not load model artifacts, so they are
safe and fast to run after any training job.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


@dataclass(slots=True)
class CurveArtifacts:
    """Paths written by the curve-generation pipeline."""

    plots: list[Path]
    tables: list[Path]


_MODEL_COLORS = {
    "logreg": "#4c78a8",
    "random_forest": "#f58518",
    "xgb_occurrence": "#54a24b",
    "transformer": "#b279a2",
}


def _score_columns(df: pd.DataFrame) -> list[tuple[str, str]]:
    """Return available score columns as ``(column, label_suffix)`` pairs."""
    cols = [("y_prob", "raw")]
    if "y_prob_calibrated" in df.columns:
        cols.append(("y_prob_calibrated", "calibrated"))
    return [(c, label) for c, label in cols if c in df.columns]


def _display_name(model_name: str, suffix: str) -> str:
    pretty = model_name.replace("_", " ")
    return f"{pretty} ({suffix})" if suffix != "raw" else pretty


def _model_color(model_name: str, suffix: str) -> str | None:
    base = _MODEL_COLORS.get(model_name)
    if base is None:
        return None
    return base if suffix == "raw" else None


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_roc_curves(
    occurrence_predictions: dict[str, pd.DataFrame],
    output_path: Path,
) -> tuple[Path, pd.DataFrame]:
    """Plot ROC curves for all occurrence models and available score columns."""
    fig, ax = plt.subplots(figsize=(7.5, 6))
    rows: list[dict[str, Any]] = []

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")

    for model_name, df in occurrence_predictions.items():
        y_true = df["y_true"].to_numpy()
        if len(np.unique(y_true)) < 2:
            continue
        for score_col, suffix in _score_columns(df):
            y_score = df[score_col].to_numpy()
            fpr, tpr, thresholds = roc_curve(y_true, y_score)
            roc_auc = roc_auc_score(y_true, y_score)
            label = f"{_display_name(model_name, suffix)} AUC={roc_auc:.3f}"
            ax.plot(
                fpr,
                tpr,
                linewidth=2,
                label=label,
                color=_model_color(model_name, suffix),
                linestyle="-" if suffix == "raw" else "--",
            )
            rows.extend(
                {
                    "model": model_name,
                    "score": suffix,
                    "curve": "roc",
                    "x": float(x),
                    "y": float(y),
                    "threshold": float(t),
                    "metric": float(roc_auc),
                }
                for x, y, t in zip(fpr, tpr, thresholds)
            )

    ax.set_title("ROC Curves - Fire Occurrence")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8, frameon=True)
    fig.tight_layout()
    return _save(fig, output_path), pd.DataFrame(rows)


def plot_precision_recall_curves(
    occurrence_predictions: dict[str, pd.DataFrame],
    output_path: Path,
) -> tuple[Path, pd.DataFrame]:
    """Plot precision-recall curves with prevalence baseline."""
    fig, ax = plt.subplots(figsize=(7.5, 6))
    rows: list[dict[str, Any]] = []

    prevalence_values = [
        float(df["y_true"].mean())
        for df in occurrence_predictions.values()
        if "y_true" in df.columns and len(df) > 0
    ]
    if prevalence_values:
        prevalence = float(np.mean(prevalence_values))
        ax.axhline(
            prevalence,
            color="black",
            linestyle="--",
            linewidth=1,
            label=f"Prevalence={prevalence:.3f}",
        )

    for model_name, df in occurrence_predictions.items():
        y_true = df["y_true"].to_numpy()
        if len(np.unique(y_true)) < 2:
            continue
        for score_col, suffix in _score_columns(df):
            y_score = df[score_col].to_numpy()
            precision, recall, thresholds = precision_recall_curve(y_true, y_score)
            pr_auc = auc(recall, precision)
            label = f"{_display_name(model_name, suffix)} AUC={pr_auc:.3f}"
            ax.plot(
                recall,
                precision,
                linewidth=2,
                label=label,
                color=_model_color(model_name, suffix),
                linestyle="-" if suffix == "raw" else "--",
            )
            padded_thresholds = np.r_[thresholds, np.nan]
            rows.extend(
                {
                    "model": model_name,
                    "score": suffix,
                    "curve": "precision_recall",
                    "x": float(x),
                    "y": float(y),
                    "threshold": None if np.isnan(t) else float(t),
                    "metric": float(pr_auc),
                }
                for x, y, t in zip(recall, precision, padded_thresholds)
            )

    ax.set_title("Precision-Recall Curves - Fire Occurrence")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8, frameon=True)
    fig.tight_layout()
    return _save(fig, output_path), pd.DataFrame(rows)


def plot_calibration_curves(
    occurrence_predictions: dict[str, pd.DataFrame],
    output_path: Path,
    bins: int = 10,
) -> tuple[Path, pd.DataFrame]:
    """Overlay reliability curves for all occurrence models."""
    fig, ax = plt.subplots(figsize=(7.5, 6))
    rows: list[dict[str, Any]] = []
    edges = np.linspace(0.0, 1.0, bins + 1)

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")

    for model_name, df in occurrence_predictions.items():
        y_true = df["y_true"].to_numpy()
        for score_col, suffix in _score_columns(df):
            y_score = df[score_col].to_numpy()
            xs: list[float] = []
            ys: list[float] = []
            for i in range(bins):
                left, right = edges[i], edges[i + 1]
                mask = (y_score >= left) & (y_score < right)
                if not np.any(mask):
                    continue
                mean_pred = float(y_score[mask].mean())
                obs_freq = float(y_true[mask].mean())
                xs.append(mean_pred)
                ys.append(obs_freq)
                rows.append(
                    {
                        "model": model_name,
                        "score": suffix,
                        "curve": "calibration",
                        "bin_left": float(left),
                        "bin_right": float(right),
                        "mean_prediction": mean_pred,
                        "observed_frequency": obs_freq,
                        "count": int(mask.sum()),
                    }
                )
            if xs:
                ax.plot(
                    xs,
                    ys,
                    marker="o",
                    linewidth=2,
                    markersize=4,
                    label=_display_name(model_name, suffix),
                    color=_model_color(model_name, suffix),
                    linestyle="-" if suffix == "raw" else "--",
                )

    ax.set_title("Reliability Curves - Fire Occurrence")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed fire frequency")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8, frameon=True)
    fig.tight_layout()
    return _save(fig, output_path), pd.DataFrame(rows)


def plot_threshold_tradeoff_curves(
    occurrence_predictions: dict[str, pd.DataFrame],
    output_path: Path,
) -> tuple[Path, pd.DataFrame]:
    """Plot precision, recall, and F1 as a function of decision threshold."""
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    rows: list[dict[str, Any]] = []

    for model_name, df in occurrence_predictions.items():
        y_true = df["y_true"].to_numpy().astype(int)
        for score_col, suffix in _score_columns(df):
            y_score = df[score_col].to_numpy()
            thresholds = np.linspace(0.01, 0.99, 99)
            precision_vals: list[float] = []
            recall_vals: list[float] = []
            f1_vals: list[float] = []
            for threshold in thresholds:
                y_pred = (y_score >= threshold).astype(int)
                tp = int(((y_true == 1) & (y_pred == 1)).sum())
                fp = int(((y_true == 0) & (y_pred == 1)).sum())
                fn = int(((y_true == 1) & (y_pred == 0)).sum())
                precision = tp / (tp + fp) if (tp + fp) else 0.0
                recall = tp / (tp + fn) if (tp + fn) else 0.0
                f1 = (
                    2 * precision * recall / (precision + recall)
                    if (precision + recall)
                    else 0.0
                )
                precision_vals.append(precision)
                recall_vals.append(recall)
                f1_vals.append(f1)
                rows.append(
                    {
                        "model": model_name,
                        "score": suffix,
                        "threshold": float(threshold),
                        "precision": float(precision),
                        "recall": float(recall),
                        "f1": float(f1),
                        "alerts": int(y_pred.sum()),
                    }
                )

            label = _display_name(model_name, suffix)
            style = "-" if suffix == "raw" else "--"
            color = _model_color(model_name, suffix)
            axes[0].plot(thresholds, precision_vals, label=label, linestyle=style, color=color)
            axes[1].plot(thresholds, recall_vals, label=label, linestyle=style, color=color)
            axes[2].plot(thresholds, f1_vals, label=label, linestyle=style, color=color)

            if "occurrence_decision_threshold" in df.columns and suffix == "calibrated":
                deployment_threshold = float(df["occurrence_decision_threshold"].iloc[0])
                for ax in axes:
                    ax.axvline(deployment_threshold, color="gray", alpha=0.25, linewidth=1)

    axes[0].set_ylabel("Precision")
    axes[1].set_ylabel("Recall")
    axes[2].set_ylabel("F1")
    axes[2].set_xlabel("Decision threshold")
    axes[0].set_title("Threshold Tradeoff Curves - Fire Occurrence")
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
    axes[0].legend(loc="best", fontsize=8, frameon=True)
    fig.tight_layout()
    return _save(fig, output_path), pd.DataFrame(rows)


def plot_cumulative_gain_curves(
    occurrence_predictions: dict[str, pd.DataFrame],
    output_path: Path,
) -> tuple[Path, pd.DataFrame]:
    """Plot cumulative fire capture as rows are ranked by predicted risk."""
    fig, ax = plt.subplots(figsize=(7.5, 6))
    rows: list[dict[str, Any]] = []

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random ranking")

    for model_name, df in occurrence_predictions.items():
        y_true = df["y_true"].to_numpy().astype(int)
        total_pos = int(y_true.sum())
        if total_pos == 0:
            continue
        for score_col, suffix in _score_columns(df):
            y_score = df[score_col].to_numpy()
            order = np.argsort(-y_score)
            ranked_true = y_true[order]
            capture = np.cumsum(ranked_true) / total_pos
            fraction_reviewed = np.arange(1, len(ranked_true) + 1) / len(ranked_true)
            label = _display_name(model_name, suffix)
            ax.plot(
                fraction_reviewed,
                capture,
                linewidth=2,
                label=label,
                color=_model_color(model_name, suffix),
                linestyle="-" if suffix == "raw" else "--",
            )
            rows.extend(
                {
                    "model": model_name,
                    "score": suffix,
                    "curve": "cumulative_gain",
                    "fraction_reviewed": float(frac),
                    "fire_capture_rate": float(cap),
                }
                for frac, cap in zip(fraction_reviewed, capture)
            )

    ax.set_title("Cumulative Fire Capture - Ranked by Risk")
    ax.set_xlabel("Fraction of patch-date rows reviewed")
    ax.set_ylabel("Fraction of true fires captured")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8, frameon=True)
    fig.tight_layout()
    return _save(fig, output_path), pd.DataFrame(rows)


def plot_severity_regression_diagnostics(
    severity_predictions: dict[str, pd.DataFrame],
    output_path: Path,
) -> tuple[Path | None, pd.DataFrame]:
    """Plot observed-vs-predicted severity and residual distributions."""
    if not severity_predictions:
        return None, pd.DataFrame()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    rows: list[dict[str, Any]] = []

    for model_name, df in severity_predictions.items():
        y_true = df["y_true"].to_numpy(dtype=float)
        y_pred = df["y_pred"].to_numpy(dtype=float)
        residual = y_pred - y_true
        axes[0].scatter(y_true, y_pred, alpha=0.75, s=28, label=model_name)
        axes[1].hist(residual, bins=20, alpha=0.55, label=model_name)
        rows.extend(
            {
                "model": model_name,
                "y_true": float(t),
                "y_pred": float(p),
                "residual": float(r),
            }
            for t, p, r in zip(y_true, y_pred, residual)
        )

    all_vals = np.concatenate(
        [
            np.r_[df["y_true"].to_numpy(dtype=float), df["y_pred"].to_numpy(dtype=float)]
            for df in severity_predictions.values()
        ]
    )
    lo, hi = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))
    axes[0].plot([lo, hi], [lo, hi], "k--", linewidth=1)
    axes[0].set_title("Severity Regression: Observed vs Predicted")
    axes[0].set_xlabel("Observed severity")
    axes[0].set_ylabel("Predicted severity")
    axes[0].grid(alpha=0.3)
    axes[0].legend(frameon=True)

    axes[1].axvline(0, color="black", linestyle="--", linewidth=1)
    axes[1].set_title("Severity Residual Distribution")
    axes[1].set_xlabel("Prediction residual")
    axes[1].set_ylabel("Count")
    axes[1].grid(alpha=0.3)
    axes[1].legend(frameon=True)

    fig.tight_layout()
    return _save(fig, output_path), pd.DataFrame(rows)


def save_all_research_curves(
    occurrence_predictions: dict[str, pd.DataFrame],
    severity_predictions: dict[str, pd.DataFrame],
    output_dir: Path,
    prefix: str,
    calibration_bins: int = 10,
) -> CurveArtifacts:
    """Generate all supported diagnostic plots and corresponding curve tables."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_paths: list[Path] = []
    table_paths: list[Path] = []

    if occurrence_predictions:
        roc_path, roc_df = plot_roc_curves(
            occurrence_predictions, output_dir / f"{prefix}_roc_curves.png"
        )
        pr_path, pr_df = plot_precision_recall_curves(
            occurrence_predictions, output_dir / f"{prefix}_precision_recall_curves.png"
        )
        cal_path, cal_df = plot_calibration_curves(
            occurrence_predictions,
            output_dir / f"{prefix}_calibration_curves.png",
            bins=calibration_bins,
        )
        thresh_path, thresh_df = plot_threshold_tradeoff_curves(
            occurrence_predictions, output_dir / f"{prefix}_threshold_tradeoffs.png"
        )
        gain_path, gain_df = plot_cumulative_gain_curves(
            occurrence_predictions, output_dir / f"{prefix}_cumulative_gain.png"
        )
        plot_paths.extend([roc_path, pr_path, cal_path, thresh_path, gain_path])

        curve_table = pd.concat([roc_df, pr_df], ignore_index=True)
        curve_table_path = output_dir / f"{prefix}_roc_pr_curve_points.csv"
        curve_table.to_csv(curve_table_path, index=False)

        cal_table_path = output_dir / f"{prefix}_calibration_bins.csv"
        cal_df.to_csv(cal_table_path, index=False)

        thresh_table_path = output_dir / f"{prefix}_threshold_tradeoffs.csv"
        thresh_df.to_csv(thresh_table_path, index=False)

        gain_table_path = output_dir / f"{prefix}_cumulative_gain.csv"
        gain_df.to_csv(gain_table_path, index=False)

        table_paths.extend(
            [curve_table_path, cal_table_path, thresh_table_path, gain_table_path]
        )

    sev_path, sev_df = plot_severity_regression_diagnostics(
        severity_predictions,
        output_dir / f"{prefix}_severity_regression_diagnostics.png",
    )
    if sev_path is not None:
        plot_paths.append(sev_path)
        sev_table_path = output_dir / f"{prefix}_severity_regression_diagnostics.csv"
        sev_df.to_csv(sev_table_path, index=False)
        table_paths.append(sev_table_path)

    return CurveArtifacts(plots=plot_paths, tables=table_paths)
