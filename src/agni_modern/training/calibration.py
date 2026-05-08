"""Post-hoc probability calibration and threshold selection for occurrence models.

Standard approach (Zadrozny & Elkan 2002, Niculescu-Mizil & Caruana 2005):
  1. Train the base model on the train split.
  2. Fit an isotonic calibrator on the *validation* split's raw probabilities.
  3. Find the F1-optimal decision threshold on the *validation* split.
  4. At test/inference time, apply the calibrator and use the tuned threshold.

Companion files next to each model:
  - ``<model>.calibrator.pkl``  — fitted IsotonicRegression
  - ``<model>.threshold.json``  — ``{"threshold": float, "val_f1_at_threshold": float}``
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import precision_recall_curve

from agni_modern.training.utils import atomic_write_bytes, atomic_write_text


def calibrator_path_for(model_path: Path) -> Path:
    """Derive the calibrator file path from a model path."""
    return model_path.with_suffix(".calibrator.pkl")


def threshold_path_for(model_path: Path) -> Path:
    """Derive the threshold file path from a model path."""
    return model_path.with_suffix(".threshold.json")


def fit_isotonic_calibrator(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> IsotonicRegression:
    """Fit an isotonic regression calibrator from validation predictions.

    Parameters
    ----------
    y_true : array of {0, 1}
        Ground truth binary labels from the validation split.
    y_prob : array of float
        Raw predicted probabilities from the base model on the validation split.

    Returns
    -------
    IsotonicRegression
        Fitted calibrator that maps raw probabilities → calibrated probabilities.
    """
    calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    calibrator.fit(y_prob, y_true)
    return calibrator


def apply_calibrator(
    calibrator: IsotonicRegression,
    y_prob: np.ndarray,
) -> np.ndarray:
    """Apply a fitted calibrator to raw probabilities."""
    return calibrator.predict(y_prob)


def save_calibrator(calibrator: IsotonicRegression, path: Path) -> None:
    """Persist calibrator to disk atomically."""
    atomic_write_bytes(path, pickle.dumps(calibrator))


def load_calibrator(path: Path) -> IsotonicRegression:
    """Load calibrator from disk."""
    with path.open("rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Optimal decision threshold (selected on validation set)
# ---------------------------------------------------------------------------

def find_optimal_f1_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> tuple[float, float]:
    """Find the probability threshold that maximises F1 on a validation set.

    Returns ``(threshold, f1_at_threshold)``.
    Falls back to ``(0.5, 0.0)`` if the data has only one class.
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


def save_threshold(threshold: float, val_f1: float, path: Path) -> None:
    """Persist optimal threshold to JSON atomically."""
    atomic_write_text(
        path,
        json.dumps({"threshold": threshold, "val_f1_at_threshold": val_f1}, indent=2),
    )


def try_load_threshold(path: Path) -> dict[str, float] | None:
    """Load threshold metadata from JSON, or ``None`` if the file is absent."""
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "threshold": float(data["threshold"]),
        "val_f1_at_threshold": float(data.get("val_f1_at_threshold", 0.0)),
    }


def load_threshold(path: Path, default: float = 0.5) -> float:
    """Load threshold from JSON; returns *default* if the file is missing."""
    meta = try_load_threshold(path)
    return meta["threshold"] if meta is not None else default
