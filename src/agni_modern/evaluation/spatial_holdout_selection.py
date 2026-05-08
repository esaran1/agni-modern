"""Select evaluable spatial holdout regions from grid-structured patch IDs.

The simple edge-row holdout can accidentally choose a region with no positive
events.  This module scans contiguous grid-row bands and ranks candidates by
whether they contain enough rows and fire-positive samples to support a useful
spatial subset evaluation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import pandas as pd

_GRID_PATCH_RE = re.compile(r"^(?P<prefix>.+)_(?P<row>\d+)_(?P<col>\d+)$")


@dataclass(frozen=True, slots=True)
class SpatialHoldoutCandidate:
    """Summary for one contiguous row-band holdout candidate."""

    prefixes: tuple[str, ...]
    row_start: int
    row_end: int
    n_rows: int
    n_patches: int
    n_positive: int
    prevalence: float
    n_severity_available: int
    severity_available_rate: float
    meets_criteria: bool
    score: float

    def as_dict(self) -> dict[str, Any]:
        """Return a serialisable row for CSV/reporting."""
        return {
            "prefixes": " ".join(self.prefixes),
            "row_start": self.row_start,
            "row_end": self.row_end,
            "n_rows": self.n_rows,
            "n_patches": self.n_patches,
            "n_positive": self.n_positive,
            "prevalence": self.prevalence,
            "n_severity_available": self.n_severity_available,
            "severity_available_rate": self.severity_available_rate,
            "meets_criteria": self.meets_criteria,
            "score": self.score,
        }


def parse_grid_patch_id(patch_id: str) -> tuple[str, int, int] | None:
    """Parse IDs of form ``<prefix>_<row>_<col>``.

    Returns ``(prefix, row, col)`` or ``None`` if the patch ID does not match.
    """
    match = _GRID_PATCH_RE.match(str(patch_id))
    if match is None:
        return None
    return (
        match.group("prefix"),
        int(match.group("row")),
        int(match.group("col")),
    )


def add_grid_coordinates(df: pd.DataFrame, patch_col: str = "patch_id") -> pd.DataFrame:
    """Add ``grid_prefix``, ``grid_row``, and ``grid_col`` columns from patch IDs."""
    parsed = df[patch_col].astype(str).map(parse_grid_patch_id)
    if parsed.isna().any():
        bad = df.loc[parsed.isna(), patch_col].astype(str).head(3).tolist()
        raise ValueError(
            "Some patch IDs do not match '<prefix>_<row>_<col>'; "
            f"examples: {bad}"
        )

    out = df.copy()
    coords = pd.DataFrame(parsed.tolist(), columns=["grid_prefix", "grid_row", "grid_col"], index=df.index)
    out[["grid_prefix", "grid_row", "grid_col"]] = coords
    return out


def _candidate_score(
    *,
    n_rows: int,
    n_positive: int,
    prevalence: float,
    n_severity_available: int,
    min_rows: int,
    min_positive: int,
    target_fraction: float,
    total_rows: int,
) -> float:
    """Score candidates, prioritising evaluability over exact size matching."""
    row_fraction = n_rows / total_rows if total_rows else 0.0
    size_penalty = abs(row_fraction - target_fraction)
    row_score = min(n_rows / max(min_rows, 1), 2.0)
    pos_score = min(n_positive / max(min_positive, 1), 3.0)
    severity_bonus = min(n_severity_available / 20.0, 1.0)
    prevalence_bonus = min(prevalence / 0.10, 1.0)
    return 3.0 * pos_score + row_score + severity_bonus + prevalence_bonus - size_penalty


def scan_contiguous_row_bands(
    df: pd.DataFrame,
    target_col: str,
    *,
    min_rows: int = 100,
    min_positive: int = 10,
    min_prevalence: float = 0.01,
    min_band_width: int = 1,
    max_band_width: int | None = None,
    target_fraction: float = 0.25,
    severity_available_col: str = "y_sev_available",
) -> list[SpatialHoldoutCandidate]:
    """Scan all contiguous row-band candidates and rank them.

    Parameters
    ----------
    df
        Dataset subset to scan. For temporal-spatial evaluation this should be
        the temporal test split, not the full dataset.
    target_col
        Binary occurrence label, e.g. ``y_occ_30d``.
    min_rows, min_positive, min_prevalence
        Criteria for a scientifically useful subset.
    min_band_width, max_band_width
        Number of contiguous grid rows per candidate.
    target_fraction
        Preferred fraction of rows in the holdout subset.
    """
    if df.empty:
        return []
    if target_col not in df.columns:
        raise ValueError(f"Missing target column: {target_col}")

    work = add_grid_coordinates(df)
    prefixes = sorted(work["grid_prefix"].astype(str).unique())
    if len(prefixes) != 1:
        raise ValueError(f"Expected one grid prefix, found: {prefixes}")
    prefix = prefixes[0]

    rows = sorted(int(r) for r in work["grid_row"].unique())
    max_width = max_band_width or max(1, int(len(rows) * target_fraction))
    max_width = max(min(max_width, len(rows)), min_band_width)
    total_rows = len(work)

    candidates: list[SpatialHoldoutCandidate] = []
    for width in range(min_band_width, max_width + 1):
        for start_idx in range(0, len(rows) - width + 1):
            band_rows = rows[start_idx : start_idx + width]
            subset = work[work["grid_row"].isin(band_rows)]
            n = len(subset)
            n_positive = int(subset[target_col].fillna(0).astype(int).sum())
            prevalence = float(n_positive / n) if n else 0.0
            n_patches = int(subset["patch_id"].nunique())
            if severity_available_col in subset.columns:
                n_sev = int(subset[severity_available_col].fillna(0).astype(int).sum())
            else:
                n_sev = 0
            sev_rate = float(n_sev / n) if n else 0.0
            meets = n >= min_rows and n_positive >= min_positive and prevalence >= min_prevalence
            prefixes_for_band = tuple(f"{prefix}_{row}_" for row in band_rows)
            candidates.append(
                SpatialHoldoutCandidate(
                    prefixes=prefixes_for_band,
                    row_start=int(min(band_rows)),
                    row_end=int(max(band_rows)),
                    n_rows=n,
                    n_patches=n_patches,
                    n_positive=n_positive,
                    prevalence=prevalence,
                    n_severity_available=n_sev,
                    severity_available_rate=sev_rate,
                    meets_criteria=meets,
                    score=_candidate_score(
                        n_rows=n,
                        n_positive=n_positive,
                        prevalence=prevalence,
                        n_severity_available=n_sev,
                        min_rows=min_rows,
                        min_positive=min_positive,
                        target_fraction=target_fraction,
                        total_rows=total_rows,
                    ),
                )
            )

    return sorted(
        candidates,
        key=lambda c: (c.meets_criteria, c.score, c.n_positive, c.n_rows),
        reverse=True,
    )


def select_spatial_holdout(
    df: pd.DataFrame,
    target_col: str,
    *,
    min_rows: int = 100,
    min_positive: int = 10,
    min_prevalence: float = 0.01,
    min_band_width: int = 1,
    max_band_width: int | None = None,
    target_fraction: float = 0.25,
    severity_available_col: str = "y_sev_available",
) -> SpatialHoldoutCandidate | None:
    """Return the best contiguous row-band holdout candidate."""
    candidates = scan_contiguous_row_bands(
        df,
        target_col,
        min_rows=min_rows,
        min_positive=min_positive,
        min_prevalence=min_prevalence,
        min_band_width=min_band_width,
        max_band_width=max_band_width,
        target_fraction=target_fraction,
        severity_available_col=severity_available_col,
    )
    return candidates[0] if candidates else None


def candidates_to_frame(candidates: list[SpatialHoldoutCandidate]) -> pd.DataFrame:
    """Convert candidate summaries to a DataFrame."""
    return pd.DataFrame([c.as_dict() for c in candidates])
