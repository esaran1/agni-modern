"""Feature-importance extraction and plotting for tree-based models."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _unwrap_model(model):
    """Drill through ModelWrapper and sklearn Pipeline to find the estimator."""
    raw = model.model if hasattr(model, "model") else model
    if hasattr(raw, "named_steps"):
        candidates = [s for s in raw.named_steps.values() if hasattr(s, "feature_importances_") or hasattr(s, "coef_")]
        if candidates:
            raw = candidates[-1]
    return raw


def extract_feature_importances(model, feature_names: list[str]) -> pd.Series:
    """Extract feature importances from a fitted model.

    Works with XGBoost, scikit-learn RandomForest, Pipeline-wrapped
    estimators, and logistic regression (uses absolute coefficients).
    """
    raw = _unwrap_model(model)
    if hasattr(raw, "feature_importances_"):
        importances = np.asarray(raw.feature_importances_)
    elif hasattr(raw, "coef_"):
        importances = np.abs(raw.coef_).flatten()
    else:
        raise AttributeError(f"Model {type(raw).__name__} has no feature_importances_ or coef_")
    return pd.Series(importances, index=feature_names, name="importance")


def plot_feature_importance(
    importances: pd.Series,
    top_n: int = 30,
    title: str = "Top Feature Importances",
) -> plt.Figure:
    """Horizontal bar chart of top feature importances."""
    top = importances.sort_values(ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(8, max(4, 0.3 * len(top))))
    top.sort_values().plot(kind="barh", ax=ax, color="#4878cf")
    ax.set_title(title)
    ax.set_xlabel("Importance")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    return fig


def save_feature_importance_plot(
    importances: pd.Series,
    output_path: Path,
    top_n: int = 30,
    title: str = "Top Feature Importances",
) -> None:
    """Render and save feature importance plot."""
    fig = plot_feature_importance(importances, top_n=top_n, title=title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
