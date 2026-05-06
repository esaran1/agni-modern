"""Dependence plotting placeholders for feature-response diagnostics."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def plot_dependence(df: pd.DataFrame, feature_col: str, target_col: str):
    """Simple scatter dependence plot."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(df[feature_col], df[target_col], alpha=0.3, s=8)
    ax.set_xlabel(feature_col)
    ax.set_ylabel(target_col)
    ax.set_title(f"Dependence: {feature_col} vs {target_col}")
    return fig
