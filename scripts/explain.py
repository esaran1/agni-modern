"""Extract feature importances and SHAP explanations from trained models.

Laptop-safe: SHAP is computed on a capped random sample (default 500 rows).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer

from agni_modern.inference.predict import load_model
from agni_modern.training.dataset import infer_feature_columns
from agni_modern.utils.config_loader import load_experiment_config
from agni_modern.visualization.feature_importance import (
    extract_feature_importances,
    save_feature_importance_plot,
)
from agni_modern.visualization.shap_analysis import (
    compute_shap_values,
    save_shap_plots,
)

app = typer.Typer(add_completion=False)


@app.command()
def main(
    config: str,
    model_name: str = typer.Option(..., "--model", help="Model name (xgb_occurrence, xgb_severity_reg, etc.)."),
    model_path: str = typer.Option(..., "--model-path", help="Path to trained model .pkl."),
    shap_samples: int = typer.Option(500, "--shap-samples", help="Max rows for SHAP computation."),
    top_n: int = typer.Option(20, "--top-n", help="Number of top features to display."),
    set: list[str] | None = typer.Option(None, "--set"),
) -> None:
    """Generate feature importance + SHAP plots for a trained model."""
    cfg = load_experiment_config(config, overrides=set)
    df = pd.read_parquet(cfg.data.io.dataset_path)
    feature_cols = infer_feature_columns(df)

    typer.echo(f"Loading model: {model_name} from {model_path}")
    model = load_model(model_name, Path(model_path))

    plots_dir = Path("outputs/plots")
    prefix = f"{cfg.experiment.name}_{model_name}"

    # --- Feature importance (built-in, instant) ---
    typer.echo("Extracting feature importances ...")
    importances = extract_feature_importances(model, feature_cols)
    fi_path = plots_dir / f"{prefix}_feature_importance.png"
    save_feature_importance_plot(importances, fi_path, top_n=top_n, title=f"{model_name} — Feature Importance")
    typer.echo(f"  Feature importance plot → {fi_path}")

    csv_path = plots_dir / f"{prefix}_feature_importance.csv"
    importances.sort_values(ascending=False).to_csv(csv_path, header=True)
    typer.echo(f"  Feature importance CSV  → {csv_path}")

    top5 = importances.sort_values(ascending=False).head(5)
    typer.echo(f"  Top 5: {', '.join(f'{n}={v:.4f}' for n, v in top5.items())}")

    # --- SHAP (capped sample, ~5-15 sec on laptop) ---
    features = df[feature_cols]
    n_shap = min(shap_samples, len(features))
    typer.echo(f"Computing SHAP on {n_shap} samples (capped at {shap_samples}) ...")

    shap_values = compute_shap_values(model, features, max_samples=shap_samples)
    saved = save_shap_plots(shap_values, plots_dir, prefix=prefix, max_display=top_n)
    for p in saved:
        typer.echo(f"  SHAP plot → {p}")

    typer.echo("Done.")


if __name__ == "__main__":
    app()
