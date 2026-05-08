"""Select a spatial holdout with enough positives for meaningful evaluation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer

from agni_modern.evaluation.spatial_holdout_selection import (
    candidates_to_frame,
    scan_contiguous_row_bands,
)
from agni_modern.evaluation.splits import temporal_holdout_split
from agni_modern.utils.config_loader import load_experiment_config

app = typer.Typer(add_completion=False)


def _subset_for_scan(cfg, df: pd.DataFrame, split: str) -> pd.DataFrame:
    """Return the requested split for spatial-holdout candidate scanning."""
    if split == "all":
        return df.copy()

    train_df, val_df, test_df = temporal_holdout_split(
        df,
        train_end=cfg.split.train_end or "2021-12-31",
        val_end=cfg.split.val_end or "2022-12-31",
        test_end=cfg.split.test_end or "2023-12-31",
    )
    if split == "train":
        return train_df
    if split == "val":
        return val_df
    if split == "test":
        return test_df
    raise ValueError(f"Unsupported split: {split}")


@app.command()
def main(
    config: str,
    split: str = typer.Option(
        "test",
        "--split",
        help="Dataset split to scan: test, val, train, or all. Default: test.",
    ),
    horizon_days: int | None = typer.Option(
        None,
        "--horizon",
        help="Occurrence horizon. Defaults to cfg.task.horizon_days or 30.",
    ),
    min_rows: int = typer.Option(
        100,
        "--min-rows",
        help="Minimum patch-date rows required in the spatial subset.",
    ),
    min_positive: int = typer.Option(
        10,
        "--min-positive",
        help="Minimum positive occurrence labels required in the spatial subset.",
    ),
    min_prevalence: float = typer.Option(
        0.01,
        "--min-prevalence",
        help="Minimum positive rate required in the spatial subset.",
    ),
    min_band_width: int = typer.Option(
        1,
        "--min-band-width",
        help="Minimum number of contiguous grid rows per candidate.",
    ),
    max_band_width: int | None = typer.Option(
        None,
        "--max-band-width",
        help="Maximum number of contiguous grid rows per candidate. Defaults to target fraction.",
    ),
    target_fraction: float = typer.Option(
        0.25,
        "--target-fraction",
        help="Preferred fraction of rows in the spatial subset.",
    ),
    top_n: int = typer.Option(10, "--top-n", help="Number of candidates to print."),
    output_csv: str | None = typer.Option(
        None,
        "--output-csv",
        help="Optional path for candidate table. Defaults to outputs/metrics/<exp>_spatial_holdout_candidates.csv.",
    ),
    overrides: list[str] | None = typer.Option(None, "--set"),
) -> None:
    """Scan contiguous grid row bands and recommend a valid spatial holdout."""
    cfg = load_experiment_config(config, overrides=overrides)
    horizon = int(horizon_days or cfg.task.get("horizon_days", 30))
    target_col = f"y_occ_{horizon}d"

    df = pd.read_parquet(cfg.data.io.dataset_path)
    scan_df = _subset_for_scan(cfg, df, split)
    if scan_df.empty:
        typer.echo(f"No rows available for split='{split}'.")
        raise typer.Exit(code=1)

    typer.echo(
        f"Scanning split='{split}' | rows={len(scan_df)} | target={target_col} | "
        f"positive_rate={scan_df[target_col].mean():.4f}"
    )

    candidates = scan_contiguous_row_bands(
        scan_df,
        target_col=target_col,
        min_rows=min_rows,
        min_positive=min_positive,
        min_prevalence=min_prevalence,
        min_band_width=min_band_width,
        max_band_width=max_band_width,
        target_fraction=target_fraction,
    )
    if not candidates:
        typer.echo("No valid grid-row candidates found.")
        raise typer.Exit(code=1)

    table = candidates_to_frame(candidates)
    out_path = (
        Path(output_csv)
        if output_csv
        else Path(cfg.outputs.metrics_dir) / f"{cfg.experiment.name}_spatial_holdout_candidates.csv"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_path, index=False)

    typer.echo(f"\nCandidate table → {out_path}")
    typer.echo("\nTop candidates:")
    display_cols = [
        "prefixes",
        "n_rows",
        "n_patches",
        "n_positive",
        "prevalence",
        "n_severity_available",
        "meets_criteria",
        "score",
    ]
    typer.echo(table[display_cols].head(top_n).to_string(index=False))

    recommended = candidates[0]
    if not recommended.meets_criteria:
        typer.echo(
            "\nWARNING: No candidate met all thresholds. "
            "The recommendation below is the best available candidate; "
            "consider lowering thresholds or building a larger dataset."
        )

    args = " ".join(f"--spatial-holdout {p}" for p in recommended.prefixes)
    typer.echo("\nRecommended prefixes:")
    typer.echo("  " + " ".join(recommended.prefixes))
    typer.echo("\nRun evaluation with:")
    typer.echo(
        f"  OMP_NUM_THREADS=1 .venv311/bin/python scripts/evaluate.py {config} {args} "
        "--set eval.spatial_holdout_auto=false"
    )


if __name__ == "__main__":
    app()
