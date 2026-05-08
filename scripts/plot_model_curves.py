"""Generate research-grade diagnostic curves from saved prediction Parquets."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer

from agni_modern.evaluation.unified import discover_prediction_files, infer_task_type
from agni_modern.utils.config_loader import load_experiment_config
from agni_modern.visualization.model_curves import save_all_research_curves

app = typer.Typer(add_completion=False)


@app.command()
def main(
    config: str,
    output_dir: str | None = typer.Option(
        None,
        "--output-dir",
        help="Directory for plots/tables. Defaults to cfg.outputs.plots_dir.",
    ),
    scan_all: bool = typer.Option(
        False,
        "--scan-all",
        help="Scan all prediction files in metrics_dir instead of this experiment only.",
    ),
    overrides: list[str] | None = typer.Option(None, "--set"),
) -> None:
    """Create ROC/PR/calibration/threshold/ranking curves for current models."""
    cfg = load_experiment_config(config, overrides=overrides)
    metrics_dir = Path(cfg.outputs.metrics_dir)
    experiment_name = None if scan_all else cfg.experiment.name

    prediction_files = discover_prediction_files(metrics_dir, experiment_name=experiment_name)
    if not prediction_files:
        typer.echo(f"No prediction files found in {metrics_dir}")
        raise typer.Exit(code=1)

    occurrence_predictions: dict[str, pd.DataFrame] = {}
    severity_predictions: dict[str, pd.DataFrame] = {}

    typer.echo(f"Found {len(prediction_files)} prediction file(s)")
    for model_name, pred_path in prediction_files:
        pred_df = pd.read_parquet(pred_path)
        task = infer_task_type(pred_df, model_name)
        typer.echo(f"  {model_name}: {task}, n={len(pred_df)}")
        if task == "occurrence":
            occurrence_predictions[model_name] = pred_df
        elif task == "severity_reg":
            severity_predictions[model_name] = pred_df

    out_dir = Path(output_dir) if output_dir else Path(cfg.outputs.plots_dir)
    prefix = cfg.experiment.name if not scan_all else "all_experiments"
    artifacts = save_all_research_curves(
        occurrence_predictions=occurrence_predictions,
        severity_predictions=severity_predictions,
        output_dir=out_dir,
        prefix=prefix,
        calibration_bins=cfg.eval.calibration_bins,
    )

    typer.echo("\nSaved plots:")
    for path in artifacts.plots:
        typer.echo(f"  {path}")

    typer.echo("\nSaved curve tables:")
    for path in artifacts.tables:
        typer.echo(f"  {path}")


if __name__ == "__main__":
    app()
