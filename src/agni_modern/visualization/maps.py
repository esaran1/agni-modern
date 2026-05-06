"""Map generation placeholders for patch-level risk visualization."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def save_map_ready_table(df: pd.DataFrame, output_path: Path) -> None:
    """Persist map-ready predictions table.

    TODO: export to GeoParquet/GeoJSON with geometry joins.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
