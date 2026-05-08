"""Training utility helpers for reproducibility and artifact persistence."""

from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """Set deterministic seeds across numpy/python/torch (CPU + CUDA).

    Also enables deterministic cuDNN convolutions and disables benchmark
    mode so repeated training runs produce bit-identical results when the
    underlying hardware is deterministic.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except AttributeError:
        # cudnn flags only exist when torch is built with CUDA; ignore on CPU-only builds.
        pass


def atomic_write_text(path: Path, contents: str, encoding: str = "utf-8") -> None:
    """Write text atomically by staging to ``path.tmp`` then renaming.

    Prevents partial files if the process is killed mid-write — important
    for long EE pipelines that share an output directory.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(contents, encoding=encoding)
    os.replace(tmp, path)


def atomic_write_bytes(path: Path, contents: bytes) -> None:
    """Binary equivalent of :func:`atomic_write_text`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(contents)
    os.replace(tmp, path)


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
    atomic_write_text(
        output_path,
        json.dumps(_sanitize_for_json(metrics), indent=2),
    )


def feature_cols_path_for(model_path: Path) -> Path:
    """Derive the feature-columns sidecar path from a model path."""
    return model_path.with_suffix(".feature_cols.json")


def save_feature_cols(feature_cols: list[str], path: Path) -> None:
    """Persist the ordered list of training features alongside a model.

    At inference time, the saved list MUST be used to select feature columns
    so the model never silently sees a different feature order than it was
    trained on.
    """
    atomic_write_text(path, json.dumps(list(feature_cols), indent=2))


def load_feature_cols(path: Path) -> list[str] | None:
    """Load feature-cols sidecar, returning ``None`` if absent."""
    if not path.exists():
        return None
    return list(json.loads(path.read_text(encoding="utf-8")))
