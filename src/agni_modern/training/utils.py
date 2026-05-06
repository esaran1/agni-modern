"""Training utility helpers for reproducibility and artifact persistence."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """Set deterministic seeds across numpy/python/torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _sanitize_for_json(obj: Any) -> Any:
    """Recursively coerce a metrics dict into JSON-safe Python primitives."""
    if isinstance(obj, dict):
        return {str(k): _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.ndarray,)):
        return _sanitize_for_json(obj.tolist())
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def save_metrics(metrics: dict[str, Any], output_path: Path) -> None:
    """Save metrics dictionary as JSON, converting NaN/Inf to null."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_sanitize_for_json(metrics), indent=2), encoding="utf-8")
