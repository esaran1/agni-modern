"""Top-k ranking utility metrics for operational prioritization."""

from __future__ import annotations

import numpy as np


def topk_recall(y_true: np.ndarray, risk_score: np.ndarray, k: int) -> float:
    """Recall among top-k highest-risk patches."""
    if k <= 0:
        raise ValueError("k must be positive")
    k = min(k, len(y_true))
    idx = np.argsort(-risk_score)[:k]
    positives = (y_true > 0).sum()
    if positives == 0:
        return 0.0
    return float((y_true[idx] > 0).sum() / positives)
