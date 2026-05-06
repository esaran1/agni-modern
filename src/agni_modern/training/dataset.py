"""Dataset builders for shared Parquet rows (tabular + sequence views)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def load_processed_table(path: str) -> pd.DataFrame:
    """Load canonical patch-date table from Parquet."""
    return pd.read_parquet(path)


def infer_feature_columns(df: pd.DataFrame) -> list[str]:
    """Infer model feature columns using namespace prefixes.

    Excludes bounding-box / coordinate columns (``static_min_*``,
    ``static_max_*``, ``centroid_*``) to prevent spatial memorisation
    that would defeat spatial holdout evaluation.

    Also excludes known legacy placeholder/leak columns that may still
    exist in older Parquet datasets built before the feature pipeline was
    cleaned up. This lets us safely retrain from an old dataset artifact
    without waiting for a full EE rebuild.
    """
    prefixes = ("optical_", "weather_", "terrain_", "landcover_", "human_", "temporal_")
    exclude = {
        "centroid_lat",
        "centroid_lon",
        "temporal_row_index",
        "temporal_recent_count_30d_placeholder",
    }
    return [c for c in df.columns if c.startswith(prefixes) and c not in exclude]


@dataclass(slots=True)
class SequenceSample:
    """In-memory sequence sample used for transformer datasets."""

    x: np.ndarray
    y_occ: float
    y_sev: float
    y_sev_available: float


class PatchSequenceDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Build per-patch rolling temporal sequences from tabular rows."""

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        seq_len: int = 16,
        occ_col: str = "y_occ_30d",
        sev_col: str = "y_sev_reg",
        sev_mask_col: str = "y_sev_available",
    ) -> None:
        self.samples: list[SequenceSample] = []
        self.meta: list[dict[str, object]] = []
        self.seq_len = seq_len

        work = df.sort_values(["patch_id", "reference_date"]).copy()
        for _, group in work.groupby("patch_id"):
            values = group[feature_cols].fillna(0.0).to_numpy(dtype=np.float32)
            y_occ = group[occ_col].to_numpy(dtype=np.float32)
            y_sev = group[sev_col].fillna(0.0).to_numpy(dtype=np.float32)
            y_mask = group[sev_mask_col].to_numpy(dtype=np.float32)
            patch_ids = group["patch_id"].to_list()
            ref_dates = group["reference_date"].to_list()

            for idx in range(len(group)):
                start = max(0, idx - seq_len + 1)
                seq = values[start : idx + 1]
                if seq.shape[0] < seq_len:
                    pad = np.zeros((seq_len - seq.shape[0], seq.shape[1]), dtype=np.float32)
                    seq = np.vstack([pad, seq])
                self.samples.append(
                    SequenceSample(
                        x=seq,
                        y_occ=float(y_occ[idx]),
                        y_sev=float(y_sev[idx]),
                        y_sev_available=float(y_mask[idx]),
                    )
                )
                self.meta.append({"patch_id": patch_ids[idx], "reference_date": ref_dates[idx]})

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        return (
            torch.tensor(sample.x, dtype=torch.float32),
            torch.tensor(sample.y_occ, dtype=torch.float32),
            torch.tensor(sample.y_sev, dtype=torch.float32),
            torch.tensor(sample.y_sev_available, dtype=torch.float32),
        )
