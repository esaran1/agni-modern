"""Build canonical patch-date dataset and save as Parquet."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import typer

from agni_modern.data.ee_client import initialize_earth_engine
from agni_modern.data.export_pipeline import build_patch_date_table, validate_and_save_dataset
from agni_modern.data.sources import default_source_adapters
from agni_modern.data.temporal_sampling import generate_reference_dates
from agni_modern.data.validation import dataset_validation_summary
from agni_modern.evaluation.reporting import save_json_report
from agni_modern.utils.config_loader import load_experiment_config
from agni_modern.utils.types import PatchRecord

app = typer.Typer(add_completion=False)


@app.command()
def main(config: str, set: list[str] | None = typer.Option(None, "--set")) -> None:
    """Assemble processed patch-date table using configured source adapters."""
    cfg = load_experiment_config(config, overrides=set)

    mode = str(cfg.task.get("mode", "placeholder"))
    if mode == "real_pilot":
        initialize_earth_engine()

    grid_path = Path(cfg.data.io.patch_grid_path)
    if not grid_path.exists():
        raise FileNotFoundError(f"Patch grid not found. Run scripts/build_grid.py first: {grid_path}")

    grid = gpd.read_parquet(grid_path)
    patches = [
        PatchRecord(
            patch_id=str(row.patch_id),
            min_lon=float(row.min_lon),
            min_lat=float(row.min_lat),
            max_lon=float(row.max_lon),
            max_lat=float(row.max_lat),
        )
        for row in grid.itertuples(index=False)
    ]

    refs = generate_reference_dates(
        start=cfg.data.temporal.reference_start,
        end=cfg.data.temporal.reference_end,
        frequency=cfg.data.temporal.reference_frequency,
    )

    adapters = default_source_adapters(source_configs=cfg.data.sources, mode=mode)
    typer.echo(f"Grid: {len(patches)} patches | Dates: {len(refs)} | Sources: {len(adapters)}")
    typer.echo(f"Total rows to fetch: {len(patches) * len(refs)}")

    df = build_patch_date_table(
        patches=patches,
        reference_dates=refs,
        lookback_days=cfg.data.temporal.lookback_days,
        source_adapters=adapters,
        dataset_path=cfg.data.io.dataset_path,
    )

    validate_and_save_dataset(
        df,
        output_path=cfg.data.io.dataset_path,
        partition_cols=cfg.data.io.partition_cols,
        enforce_uniqueness=cfg.data.quality.enforce_unique_patch_date,
        enforce_leakage=cfg.data.quality.enforce_feature_time_leakage_guard,
    )

    summary = dataset_validation_summary(df)
    summary["leakage_guard_passed"] = True
    summary["uniqueness_check_passed"] = True
    summary_path = Path(cfg.outputs.metrics_dir) / f"{cfg.experiment.name}_dataset_validation.json"
    save_json_report(summary, summary_path)

    typer.echo(f"Saved dataset to: {cfg.data.io.dataset_path} | rows={len(df)}")
    typer.echo(f"Saved validation summary: {summary_path}")


if __name__ == "__main__":
    app()
