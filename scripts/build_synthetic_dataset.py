"""Build a synthetic canonical patch-date dataset for architecture smoke tests."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from agni_modern.data.export_pipeline import validate_and_save_dataset
from agni_modern.data.synthetic import SyntheticDataConfig, generate_synthetic_patch_date_table
from agni_modern.utils.config_loader import load_experiment_config

app = typer.Typer(add_completion=False)


@app.command()
def main(config: str, set: list[str] | None = typer.Option(None, "--set")) -> None:
    """Generate, validate, and save synthetic Parquet dataset using experiment settings."""
    cfg = load_experiment_config(config, overrides=set)

    synth = cfg.task.get("synthetic", {})
    synth_cfg = SyntheticDataConfig(
        seed=int(synth.get("seed", cfg.experiment.seed)),
        num_patches=int(synth.get("num_patches", 48)),
        num_reference_dates=int(synth.get("num_reference_dates", 80)),
        start_date=str(synth.get("start_date", "2021-01-01")),
        frequency_days=int(synth.get("frequency_days", 7)),
    )

    df = generate_synthetic_patch_date_table(synth_cfg)

    validate_and_save_dataset(
        df,
        output_path=cfg.data.io.dataset_path,
        partition_cols=cfg.data.io.partition_cols,
        enforce_uniqueness=cfg.data.quality.enforce_unique_patch_date,
        enforce_leakage=cfg.data.quality.enforce_feature_time_leakage_guard,
    )

    # Save resolved config snapshot for reproducibility.
    metrics_dir = Path(cfg.outputs.metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = metrics_dir / f"{cfg.experiment.name}_resolved_config.json"
    snapshot_path.write_text(cfg.model_dump_json(indent=2), encoding="utf-8")

    occ_rate = float(df["y_occ_30d"].mean())
    typer.echo(
        f"Saved synthetic dataset to {cfg.data.io.dataset_path} | rows={len(df)} | y_occ_30d_rate={occ_rate:.3f}"
    )
    typer.echo(f"Saved config snapshot to {snapshot_path}")


if __name__ == "__main__":
    app()
