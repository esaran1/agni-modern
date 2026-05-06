"""Reliability diagram (calibration curve) visualization."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_reliability_diagram(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    bins: int = 10,
    title: str = "Reliability Diagram",
) -> plt.Figure:
    """Create a reliability diagram with calibration curve + prediction histogram."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    bin_centers: list[float] = []
    bin_accs: list[float] = []
    bin_counts: list[int] = []

    for i in range(bins):
        mask = (y_prob >= edges[i]) & (y_prob < edges[i + 1])
        count = int(mask.sum())
        bin_counts.append(count)
        if count == 0:
            bin_centers.append((edges[i] + edges[i + 1]) / 2)
            bin_accs.append(0.0)
            continue
        bin_centers.append(float(y_prob[mask].mean()))
        bin_accs.append(float(y_true[mask].mean()))

    fig, (ax_cal, ax_hist) = plt.subplots(
        2, 1, figsize=(7, 6), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )

    ax_cal.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect")
    non_empty = [(c, a) for c, a, n in zip(bin_centers, bin_accs, bin_counts) if n > 0]
    if non_empty:
        cx, cy = zip(*non_empty)
        ax_cal.plot(cx, cy, "o-", color="#4878cf", markersize=6, label="Model")
    ax_cal.set_ylabel("Observed frequency")
    ax_cal.set_title(title)
    ax_cal.legend(loc="lower right", frameon=True)
    ax_cal.set_xlim(0, 1)
    ax_cal.set_ylim(0, 1)
    ax_cal.grid(alpha=0.3)

    bar_x = [(edges[i] + edges[i + 1]) / 2 for i in range(bins)]
    ax_hist.bar(bar_x, bin_counts, width=0.8 / bins, color="#4878cf", alpha=0.6)
    ax_hist.set_xlabel("Predicted probability")
    ax_hist.set_ylabel("Count")
    ax_hist.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    return fig


def save_reliability_diagram(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_path: Path,
    bins: int = 10,
    title: str = "Reliability Diagram",
) -> None:
    """Render and save a reliability diagram to disk."""
    fig = plot_reliability_diagram(y_true, y_prob, bins=bins, title=title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
