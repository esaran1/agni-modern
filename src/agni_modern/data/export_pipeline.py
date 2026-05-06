"""Dataset export orchestration for patch-date feature/label table creation."""

from __future__ import annotations

import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from agni_modern.data.parquet_io import (
    enforce_leakage_guard,
    enforce_unique_patch_date,
    ensure_required_columns,
    write_partitioned_parquet,
)
from agni_modern.data.sources import SourceAdapter
from agni_modern.features.burn_indices import add_burn_related_features
from agni_modern.features.human_pressure import add_human_pressure_features
from agni_modern.features.landcover import add_landcover_features
from agni_modern.features.rolling_windows import add_temporal_rolling_features
from agni_modern.features.spectral import build_spectral_features
from agni_modern.features.terrain import add_terrain_features
from agni_modern.features.vegetation import add_vegetation_indices
from agni_modern.features.weather import add_weather_aggregations
from agni_modern.labels.occurrence import build_occurrence_labels
from agni_modern.labels.severity import build_severity_labels
from agni_modern.utils.types import PatchRecord

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_RETRY_BACKOFF = 5.0


def _fetch_with_retry(
    adapter: SourceAdapter, patch: PatchRecord, reference_date: Any, lookback_days: int,
) -> dict[str, Any]:
    """Call adapter with exponential-backoff retries for transient EE errors."""
    for attempt in range(_MAX_RETRIES):
        try:
            return adapter.fetch_patch_timeseries(patch, reference_date, lookback_days)
        except Exception as exc:
            if attempt == _MAX_RETRIES - 1:
                logger.warning(
                    "Adapter %s failed after %d retries for %s @ %s: %s",
                    getattr(adapter, "source_name", "?"), _MAX_RETRIES,
                    patch.patch_id, reference_date, exc,
                )
                return {}
            wait = _RETRY_BACKOFF * (2 ** attempt)
            logger.info("Retry %d/%d for %s (waiting %.0fs): %s",
                        attempt + 1, _MAX_RETRIES, getattr(adapter, "source_name", "?"), wait, exc)
            time.sleep(wait)
    return {}


def _checkpoint_path(dataset_path: str | Path | None) -> Path | None:
    if dataset_path is None:
        return None
    return Path(str(dataset_path) + ".checkpoint.parquet")


def build_patch_date_table(
    patches: list[PatchRecord],
    reference_dates: list,
    lookback_days: int,
    source_adapters: list[SourceAdapter],
    progress: bool = True,
    checkpoint_every: int = 50,
    dataset_path: str | Path | None = None,
) -> pd.DataFrame:
    """Assemble one-row-per-patch-date table from source adapter summaries.

    Supports checkpoint/resume: if a ``.checkpoint.parquet`` file exists next
    to *dataset_path*, previously fetched rows are loaded and the loop skips
    them.  A new checkpoint is saved every *checkpoint_every* rows.
    """
    ckpt = _checkpoint_path(dataset_path)
    existing_keys: set[tuple[str, str]] = set()
    rows: list[dict[str, Any]] = []

    if ckpt is not None and ckpt.exists():
        prev = pd.read_parquet(ckpt)
        rows = prev.to_dict("records")
        existing_keys = {
            (str(r["patch_id"]), str(r["reference_date"])) for r in rows
        }
        logger.info("Resuming from checkpoint: %d rows already fetched", len(rows))
        sys.stderr.write(f"  Resuming from checkpoint ({len(rows)} rows cached)\n")

    total = len(patches) * len(reference_dates)
    done = len(rows)
    t0 = time.time()

    for patch in patches:
        for reference_date in reference_dates:
            key = (patch.patch_id, str(reference_date))
            if key in existing_keys:
                continue

            merged: dict[str, Any] = {
                "patch_id": patch.patch_id,
                "reference_date": reference_date,
                "feature_max_timestamp": reference_date,
                "static_min_lon": patch.min_lon,
                "static_min_lat": patch.min_lat,
                "static_max_lon": patch.max_lon,
                "static_max_lat": patch.max_lat,
                "centroid_lon": (patch.min_lon + patch.max_lon) / 2.0,
                "centroid_lat": (patch.min_lat + patch.max_lat) / 2.0,
            }

            for adapter in source_adapters:
                merged.update(
                    _fetch_with_retry(adapter, patch, reference_date, lookback_days)
                )

            merged.update(build_occurrence_labels(reference_date=reference_date, context=merged))
            merged.update(build_severity_labels(reference_date=reference_date, context=merged))
            rows.append(merged)

            done += 1
            if ckpt is not None and done % checkpoint_every == 0:
                pd.DataFrame(rows).to_parquet(ckpt, index=False)

            if progress and done % max(1, total // 20) == 0:
                elapsed = time.time() - t0
                new_rows = done - len(existing_keys)
                rate = new_rows / elapsed if elapsed > 0 else 0
                remaining = total - done
                eta = remaining / rate if rate > 0 else 0
                sys.stderr.write(
                    f"\r  [{done}/{total}] {done/total*100:.0f}%  "
                    f"({rate:.1f} rows/s, ETA {eta/60:.1f} min)"
                )
                sys.stderr.flush()

    if progress and total > 0:
        sys.stderr.write(f"\r  [{total}/{total}] 100%  ({time.time()-t0:.1f}s total)\n")
        sys.stderr.flush()

    if ckpt is not None and ckpt.exists():
        ckpt.unlink()
        logger.info("Checkpoint removed after successful completion")

    df = pd.DataFrame(rows)
    if not df.empty:
        df = build_spectral_features(df)
        df = add_vegetation_indices(df)
        df = add_burn_related_features(df)
        df = add_weather_aggregations(df)
        df = add_terrain_features(df)
        df = add_landcover_features(df)
        df = add_human_pressure_features(df)
        df = add_temporal_rolling_features(df)
        df["reference_year"] = pd.to_datetime(df["reference_date"]).dt.year
        df["label_window_start"] = pd.to_datetime(df["label_window_start"])
        df["label_window_end"] = pd.to_datetime(df["label_window_end"])
        df["feature_max_timestamp"] = pd.to_datetime(df["feature_max_timestamp"])
    return df


def validate_and_save_dataset(
    df: pd.DataFrame,
    output_path: str | Path,
    partition_cols: list[str],
    enforce_uniqueness: bool = True,
    enforce_leakage: bool = True,
) -> None:
    """Apply canonical checks and write dataset to Parquet."""
    ensure_required_columns(df)
    if enforce_uniqueness:
        enforce_unique_patch_date(df)
    if enforce_leakage:
        enforce_leakage_guard(df)

    write_partitioned_parquet(df, output_path, partition_cols=partition_cols)


def timestamped_run_id(prefix: str = "dataset") -> str:
    """Return stable run id for local artifact naming."""
    now = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{now}"
