"""Typed records and protocols shared across pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Protocol


@dataclass(slots=True)
class PatchRecord:
    """Represents one spatial modeling unit in the Indonesia patch grid."""

    patch_id: str
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float


@dataclass(slots=True)
class LabelRecord:
    """Canonical set of labels and metadata for a single patch-date row."""

    y_occ_7d: int | None = None
    y_occ_30d: int | None = None
    y_occ_60d: int | None = None
    y_sev_reg: float | None = None
    y_sev_cls: int | None = None
    y_sev_available: int = 0
    label_window_start: date | None = None
    label_window_end: date | None = None


@dataclass(slots=True)
class PredictionRecord:
    """Output schema for operational patch-level predictions."""

    patch_id: str
    reference_date: date
    p_fire: float
    severity_conditional: float
    expected_risk: float


class DataSourceAdapter(Protocol):
    """Interface for source-specific data fetchers."""

    def fetch_patch_timeseries(
        self,
        patch: PatchRecord,
        reference_date: date,
        lookback_days: int,
    ) -> dict[str, Any]:
        """Fetch source summaries for one patch and reference date."""


class LabelBuilder(Protocol):
    """Interface for label-generation components."""

    def build_labels(
        self,
        patch_id: str,
        reference_date: date,
        context: dict[str, Any],
    ) -> LabelRecord:
        """Generate leakage-safe labels for one patch-date sample."""


class BaseModelWrapper(Protocol):
    """Common interface across tabular and temporal model wrappers."""

    def fit(self, train_df: Any, val_df: Any, config: dict[str, Any]) -> None:
        """Fit model using train/validation subsets."""

    def predict(self, df: Any) -> Any:
        """Return deterministic point predictions."""

    def predict_proba(self, df: Any) -> Any:
        """Return probabilistic predictions where supported."""

    def save(self, path: Path) -> None:
        """Persist model artifacts."""

    @classmethod
    def load(cls, path: Path) -> "BaseModelWrapper":
        """Restore a saved model wrapper."""
