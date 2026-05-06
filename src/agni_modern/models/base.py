"""Base model wrapper classes."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class ModelWrapper:
    """Minimal base class shared by model wrappers."""

    def fit(self, train_df: Any, val_df: Any, config: dict[str, Any]) -> None:
        raise NotImplementedError

    def predict(self, df: Any) -> Any:
        raise NotImplementedError

    def predict_proba(self, df: Any) -> Any:
        raise NotImplementedError

    def save(self, path: Path) -> None:
        raise NotImplementedError

    @classmethod
    def load(cls, path: Path) -> "ModelWrapper":
        raise NotImplementedError
