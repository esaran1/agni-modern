"""Train multitask temporal transformer model."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer

from agni_modern.evaluation.splits import temporal_holdout_split
from agni_modern.training.train_transformer import train_transformer_multitask
from agni_modern.utils.config_loader import load_experiment_config

app = typer.Typer(add_completion=False)


@app.command()
def main(config: str, set: list[str] | None = typer.Option(None, "--set")) -> None:
    """Train temporal transformer and save checkpoint + metrics."""
    cfg = load_experiment_config(config, overrides=set)
    df = pd.read_parquet(cfg.data.io.dataset_path)

    train_df, val_df, test_df = temporal_holdout_split(
        df,
        train_end=cfg.split.train_end or "2021-12-31",
        val_end=cfg.split.val_end or "2022-12-31",
        test_end=cfg.split.test_end or "2023-12-31",
    )

    task_horizon = cfg.task.get("horizon_days")
    if task_horizon is None:
        horizons = cfg.task.get("horizons_days")
        task_horizon = horizons[0] if isinstance(horizons, list) and horizons else 30
    target_col = f"y_occ_{int(task_horizon)}d"

    params = {**cfg.model.params, **cfg.model.training, **cfg.model.loss}
    model_path = Path(cfg.outputs.model_dir) / f"{cfg.experiment.name}_transformer.pt"
    metrics_path = Path(cfg.outputs.metrics_dir) / f"{cfg.experiment.name}_transformer.json"
    preds_path = Path(cfg.outputs.metrics_dir) / f"{cfg.experiment.name}_transformer_predictions.parquet"

    metrics = train_transformer_multitask(
        train_df=train_df,
        val_df=val_df,
        output_model_path=model_path,
        output_metrics_path=metrics_path,
        params=params,
        seed=cfg.experiment.seed,
        test_df=test_df,
        target_col=target_col,
        device=cfg.train.device,
        output_predictions_path=preds_path,
    )

    typer.echo(f"Training complete — {metrics['total_epochs']} epochs, best_val_loss={metrics['best_val_loss']}")
    if "deployment_threshold" in metrics:
        typer.echo(
            f"Deployment threshold (val, calibrated): {metrics['deployment_threshold']:.4f}  "
            f"val_f1@t={metrics.get('val_f1_at_deployment_threshold', 0):.4f}"
        )
    if "test_f1" in metrics:
        typer.echo(
            f"Test: f1={metrics.get('test_f1')}  "
            f"roc_auc={metrics.get('test_roc_auc')}  "
            f"pr_auc={metrics.get('test_pr_auc')}"
        )
    if "test_f1_calibrated_deployment" in metrics:
        typer.echo(f"Test (calibrated @ deployment t): f1={metrics['test_f1_calibrated_deployment']}")


if __name__ == "__main__":
    app()
