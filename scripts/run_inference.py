"""Generate map-ready inference outputs from trained models."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer

from agni_modern.inference.predict import load_model, run_inference
from agni_modern.visualization.maps import save_map_ready_table
from agni_modern.utils.config_loader import load_experiment_config

app = typer.Typer(add_completion=False)


@app.command()
def main(
    config: str,
    occurrence_model_name: str = typer.Option("xgb_occurrence", "--occ-model", help="Occurrence model name (logreg, random_forest, xgb_occurrence)."),
    occurrence_model_path: str = typer.Option(..., "--occ-path", help="Path to trained occurrence model .pkl."),
    severity_model_name: str = typer.Option("xgb_severity_reg", "--sev-model", help="Severity model name."),
    severity_model_path: str = typer.Option(..., "--sev-path", help="Path to trained severity regressor .pkl."),
    start_date: str = typer.Option(None, "--start", help="Filter reference_date >= start. Omit to use all rows."),
    end_date: str = typer.Option(None, "--end", help="Filter reference_date <= end. Omit to use all rows."),
    output: str = typer.Option(None, "--output", help="Output Parquet path. Defaults to outputs/maps/<experiment>_predictions.parquet."),
    set: list[str] | None = typer.Option(None, "--set"),
) -> None:
    """Load trained occurrence + severity models, run inference, save map-ready table."""
    cfg = load_experiment_config(config, overrides=set)
    df = pd.read_parquet(cfg.data.io.dataset_path)

    if start_date or end_date:
        ref = pd.to_datetime(df["reference_date"])
        mask = pd.Series(True, index=df.index)
        if start_date:
            mask &= ref >= pd.Timestamp(start_date)
        if end_date:
            mask &= ref <= pd.Timestamp(end_date)
        df = df.loc[mask].copy()

    if df.empty:
        typer.echo("No rows for the requested date range.")
        raise typer.Exit(code=1)

    typer.echo(f"Loading occurrence model: {occurrence_model_name} from {occurrence_model_path}")
    occ_model = load_model(occurrence_model_name, Path(occurrence_model_path))

    typer.echo(f"Loading severity model: {severity_model_name} from {severity_model_path}")
    sev_model = load_model(severity_model_name, Path(severity_model_path))

    typer.echo(f"Running inference on {len(df)} rows ...")
    pred_df = run_inference(
        df, occ_model, sev_model,
        occurrence_model_path=Path(occurrence_model_path),
    )

    out_path = Path(output) if output else Path("outputs/maps") / f"{cfg.experiment.name}_predictions.parquet"
    save_map_ready_table(pred_df, out_path)

    typer.echo(f"Saved {len(pred_df)} predictions → {out_path}")
    if "p_fire_raw" in pred_df.columns:
        typer.echo("  [calibration applied — isotonic]")
        typer.echo(
            f"  p_fire_raw:  mean={pred_df['p_fire_raw'].mean():.4f}  "
            f"std={pred_df['p_fire_raw'].std():.4f}"
        )
    typer.echo(
        f"  p_fire:  mean={pred_df['p_fire'].mean():.4f}  "
        f"std={pred_df['p_fire'].std():.4f}"
    )
    if "fire_alert" in pred_df.columns:
        thresh = pred_df["fire_alert_threshold"].iloc[0]
        n_alert = int(pred_df["fire_alert"].sum())
        typer.echo(
            f"  fire_alert: {n_alert}/{len(pred_df)} patches "
            f"({100*n_alert/len(pred_df):.1f}%) at threshold={thresh:.4f}"
        )
    typer.echo(
        f"  severity: mean={pred_df['severity_conditional'].mean():.4f}  "
        f"std={pred_df['severity_conditional'].std():.4f}"
    )
    typer.echo(
        f"  risk:    mean={pred_df['expected_risk'].mean():.4f}  "
        f"std={pred_df['expected_risk'].std():.4f}"
    )


if __name__ == "__main__":
    app()
