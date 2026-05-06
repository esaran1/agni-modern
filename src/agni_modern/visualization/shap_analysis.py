"""SHAP explainability for tree-based and linear models.

Designed to be laptop-safe: sample sizes are capped so SHAP computation
stays under ~2 GB RAM and finishes in seconds, not minutes.

Properly handles sklearn Pipeline-wrapped models by:
  1. Transforming features through preprocessing steps (imputer, scaler)
  2. Extracting the final estimator for the correct explainer type
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


DEFAULT_MAX_SAMPLES = 500


def _unwrap_pipeline(model):
    """Separate a model into (preprocessing_pipeline | None, estimator).

    For a Pipeline, returns the preprocessing steps and the final estimator
    separately so SHAP receives data in the correct feature space.
    """
    raw = model.model if hasattr(model, "model") else model

    if hasattr(raw, "named_steps"):
        steps = list(raw.named_steps.items())
        estimator = steps[-1][1]
        from sklearn.pipeline import Pipeline
        if len(steps) > 1:
            preprocessor = Pipeline(steps[:-1])
        else:
            preprocessor = None
        return preprocessor, estimator

    return None, raw


def compute_shap_values(
    model,
    X: pd.DataFrame,
    max_samples: int = DEFAULT_MAX_SAMPLES,
) -> shap.Explanation:
    """Compute SHAP values, handling Pipelines and both tree/linear models.

    Returns a shap.Explanation with feature names matching the original columns.
    """
    preprocessor, estimator = _unwrap_pipeline(model)

    if len(X) > max_samples:
        X = X.sample(n=max_samples, random_state=42)

    if preprocessor is not None:
        X_transformed = pd.DataFrame(
            preprocessor.transform(X), columns=X.columns, index=X.index,
        )
    else:
        X_transformed = X

    if hasattr(estimator, "feature_importances_"):
        explainer = shap.TreeExplainer(estimator)
    elif hasattr(estimator, "coef_"):
        explainer = shap.LinearExplainer(estimator, X_transformed)
    else:
        raise TypeError(
            f"No SHAP explainer available for {type(estimator).__name__}. "
            "Expected a tree model (feature_importances_) or linear model (coef_)."
        )

    return explainer(X_transformed)


def _collapse_multiclass_explanation(shap_values: shap.Explanation) -> shap.Explanation:
    """Collapse multi-output SHAP values to a 2D Explanation for plotting.

    SHAP returns shape (n_samples, n_features, n_outputs) for multiclass
    classifiers. Beeswarm/bar plots expect 2D values, so we aggregate over
    outputs using mean absolute attribution magnitude.
    """
    values = np.asarray(shap_values.values)
    if values.ndim <= 2:
        return shap_values

    collapsed_values = np.mean(np.abs(values), axis=-1)

    base_values = np.asarray(shap_values.base_values)
    if base_values.ndim > 1:
        base_values = np.mean(base_values, axis=-1)

    return shap.Explanation(
        values=collapsed_values,
        base_values=base_values,
        data=shap_values.data,
        feature_names=shap_values.feature_names,
    )


def plot_shap_summary(
    shap_values: shap.Explanation,
    title: str = "SHAP Summary",
    max_display: int = 20,
) -> plt.Figure:
    """Beeswarm summary plot."""
    shap_values = _collapse_multiclass_explanation(shap_values)
    fig, ax = plt.subplots(figsize=(10, max(5, 0.35 * max_display)))
    shap.plots.beeswarm(shap_values, max_display=max_display, show=False)
    plt.title(title)
    plt.tight_layout()
    return plt.gcf()


def plot_shap_bar(
    shap_values: shap.Explanation,
    title: str = "SHAP Feature Importance (mean |SHAP|)",
    max_display: int = 20,
) -> plt.Figure:
    """Bar chart of mean absolute SHAP values."""
    shap_values = _collapse_multiclass_explanation(shap_values)
    fig, ax = plt.subplots(figsize=(8, max(4, 0.3 * max_display)))
    shap.plots.bar(shap_values, max_display=max_display, show=False)
    plt.title(title)
    plt.tight_layout()
    return plt.gcf()


def save_shap_plots(
    shap_values: shap.Explanation,
    output_dir: Path,
    prefix: str = "shap",
    max_display: int = 20,
) -> list[Path]:
    """Save beeswarm + bar SHAP plots. Returns list of saved paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    beeswarm_path = output_dir / f"{prefix}_beeswarm.png"
    fig = plot_shap_summary(shap_values, title=f"{prefix} — SHAP Beeswarm", max_display=max_display)
    fig.savefig(str(beeswarm_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(beeswarm_path)

    bar_path = output_dir / f"{prefix}_bar.png"
    fig = plot_shap_bar(shap_values, title=f"{prefix} — Mean |SHAP|", max_display=max_display)
    fig.savefig(str(bar_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(bar_path)

    return saved
