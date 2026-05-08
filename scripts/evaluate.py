"""Unified evaluation of all trained models — occurrence, severity, combined risk.

Discovers saved prediction Parquet files, computes full metrics, builds a
side-by-side comparison table, and optionally generates calibration plots
and spatial holdout reports.

Spatial metrics filter the **saved test-split predictions** to patch IDs whose
prefix lies in the holdout set (manual ``split.holdout_regions``, CLI
``--spatial-holdout``, and/or ``eval.spatial_holdout_auto``).  This measures
generalisation to held-out grid rows within the same temporal test window, not
a separately trained spatial-only split.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer

from agni_modern.evaluation.reporting import save_json_report
from agni_modern.evaluation.spatial_holdout_selection import select_spatial_holdout
from agni_modern.evaluation.splits import (
    auto_spatial_holdout_prefixes,
    temporal_holdout_split,
)
from agni_modern.evaluation.unified import (
    build_comparison_table,
    discover_prediction_files,
    evaluate_combined_risk,
    evaluate_occurrence_predictions,
    evaluate_severity_cls_predictions,
    evaluate_severity_reg_predictions,
    evaluate_spatial_subset,
    infer_task_type,
)
from agni_modern.training.utils import _sanitize_for_json
from agni_modern.utils.config_loader import load_experiment_config
from agni_modern.visualization.calibration_plot import save_reliability_diagram


def _auto_spatial_prefixes(cfg) -> list[str]:
    """Return spatial-holdout patch prefixes derived from the test split.

    Prefers the contiguous-row-band selector
    (:func:`select_spatial_holdout`) which requires the candidate band to
    contain enough positive samples to be evaluable.  Falls back to the
    naive top-rows picker if the smart selector returns nothing (e.g. empty
    test split or a non-grid patch_id scheme).
    """
    horizon = int(cfg.task.get("horizon_days", 30))
    target_col = f"y_occ_{horizon}d"

    try:
        df = pd.read_parquet(cfg.data.io.dataset_path)
    except FileNotFoundError:
        return []

    if "patch_id" not in df.columns:
        return []

    train_end = cfg.split.train_end or "2021-12-31"
    val_end = cfg.split.val_end or "2022-12-31"
    test_end = cfg.split.test_end or "2023-12-31"
    _, _, test_df = temporal_holdout_split(df, train_end, val_end, test_end)

    if not test_df.empty and target_col in test_df.columns:
        candidate = select_spatial_holdout(
            test_df,
            target_col=target_col,
            target_fraction=cfg.eval.spatial_holdout_fraction,
        )
        if candidate is not None:
            if not candidate.meets_criteria:
                typer.echo(
                    "WARNING: best spatial holdout candidate did not meet "
                    f"min-positive/min-prevalence thresholds (n_pos={candidate.n_positive}, "
                    f"prev={candidate.prevalence:.3f}). Reporting metrics anyway — "
                    "treat spatial generalisation numbers with caution."
                )
            return list(candidate.prefixes)

    return auto_spatial_holdout_prefixes(
        df["patch_id"], holdout_fraction=cfg.eval.spatial_holdout_fraction,
    )

app = typer.Typer(add_completion=False)


@app.command()
def main(
    config: str,
    scan_all: bool = typer.Option(False, "--scan-all", help="Evaluate all experiments in metrics_dir, not just the one named in the config."),
    spatial_holdout: list[str] = typer.Option([], "--spatial-holdout", help="Patch-ID prefixes to use as spatial holdout subset."),
    overrides: list[str] | None = typer.Option(None, "--set"),
) -> None:
    """Evaluate all saved predictions for an experiment (or all experiments)."""
    cfg = load_experiment_config(config, overrides=overrides)
    metrics_dir = Path(cfg.outputs.metrics_dir)
    cal_bins = cfg.eval.calibration_bins
    topk_values = cfg.eval.top_k_values

    holdout_regions = list(dict.fromkeys([*spatial_holdout, *list(cfg.split.holdout_regions or [])]))
    if cfg.eval.spatial_holdout_auto:
        auto_prefixes = _auto_spatial_prefixes(cfg)
        if auto_prefixes:
            typer.echo(
                f"Auto spatial holdout ({cfg.eval.spatial_holdout_fraction:.0%} of grid rows, "
                f"smart-selected for positives): {auto_prefixes}"
            )
            holdout_regions = sorted(set(holdout_regions) | set(auto_prefixes))
        else:
            typer.echo(
                "spatial_holdout_auto is true but no grid-structured patch_ids were found; "
                "skipping auto prefixes (expected pattern prefix_row_col)."
            )

    experiment_name = None if scan_all else cfg.experiment.name
    predictions = discover_prediction_files(metrics_dir, experiment_name=experiment_name)

    if not predictions:
        typer.echo(f"No prediction files found in {metrics_dir}")
        raise typer.Exit(code=1)

    typer.echo(f"Found {len(predictions)} prediction file(s)")

    all_results: dict[str, dict] = {}
    occ_preds: dict[str, pd.DataFrame] = {}
    sev_reg_preds: dict[str, pd.DataFrame] = {}

    for model_name, pred_path in predictions:
        pred_df = pd.read_parquet(pred_path)
        task_type = infer_task_type(pred_df, model_name)
        typer.echo(f"  {model_name} ({task_type}, n={len(pred_df)})")

        if task_type == "occurrence":
            metrics = evaluate_occurrence_predictions(pred_df, cal_bins, topk_values)
            occ_preds[model_name] = pred_df
        elif task_type == "severity_cls":
            metrics = evaluate_severity_cls_predictions(pred_df)
        else:
            metrics = evaluate_severity_reg_predictions(pred_df)
            sev_reg_preds[model_name] = pred_df

        if holdout_regions:
            typer.echo(f"  spatial subset prefixes: {holdout_regions}")
            spatial = evaluate_spatial_subset(pred_df, holdout_regions, task_type, cal_bins)
            metrics.update(spatial)

        all_results[model_name] = metrics

        per_model_path = metrics_dir / f"{cfg.experiment.name}_{model_name}_evaluation.json"
        save_json_report(_sanitize_for_json(metrics), per_model_path)

    # --- Combined risk: pair each occurrence model with each severity regressor ---
    for occ_name, occ_df in occ_preds.items():
        for sev_name, sev_df in sev_reg_preds.items():
            combo_key = f"risk__{occ_name}+{sev_name}"
            risk_metrics = evaluate_combined_risk(occ_df, sev_df, topk_values)
            all_results[combo_key] = risk_metrics
            typer.echo(
                f"  {combo_key} (n={risk_metrics.get('combined_risk_n', 0)}, "
                f"mean_risk={risk_metrics.get('expected_risk_mean', 0):.4f})"
            )

    # --- Comparison table ---
    comparison = build_comparison_table(all_results)
    comparison_csv = metrics_dir / f"{cfg.experiment.name}_comparison.csv"
    comparison_json = metrics_dir / f"{cfg.experiment.name}_comparison.json"

    comparison.to_csv(comparison_csv)
    comparison.reset_index().to_json(comparison_json, orient="records", indent=2)
    typer.echo(f"\nComparison table → {comparison_csv}")

    # --- Calibration plots for occurrence models (raw + calibrated) ---
    plots_dir = Path("outputs/plots")
    for occ_name, occ_df in occ_preds.items():
        plot_path = plots_dir / f"{cfg.experiment.name}_{occ_name}_calibration.png"
        save_reliability_diagram(
            y_true=occ_df["y_true"].to_numpy(),
            y_prob=occ_df["y_prob"].to_numpy(),
            output_path=plot_path,
            bins=cal_bins,
            title=f"Calibration (raw) — {occ_name}",
        )
        typer.echo(f"Calibration plot (raw) → {plot_path}")

        if "y_prob_calibrated" in occ_df.columns:
            plot_path_cal = plots_dir / f"{cfg.experiment.name}_{occ_name}_calibration_post.png"
            save_reliability_diagram(
                y_true=occ_df["y_true"].to_numpy(),
                y_prob=occ_df["y_prob_calibrated"].to_numpy(),
                output_path=plot_path_cal,
                bins=cal_bins,
                title=f"Calibration (isotonic) — {occ_name}",
            )
            typer.echo(f"Calibration plot (isotonic) → {plot_path_cal}")

    # --- Pretty-print summary ---
    typer.echo("\n" + "=" * 72)
    typer.echo("EVALUATION SUMMARY")
    typer.echo("=" * 72)

    for model_name, metrics in all_results.items():
        task = metrics.get("task", "unknown")
        parts = [f"  {model_name} [{task}]"]
        if task == "occurrence":
            parts.append(f"f1={metrics.get('f1', 0):.4f}")
            parts.append(f"roc_auc={metrics.get('roc_auc', 0):.4f}")
            parts.append(f"pr_auc={metrics.get('pr_auc', 0):.4f}")
            ece_raw = metrics.get("ece_raw", metrics.get("ece", 0))
            parts.append(f"ece_raw={ece_raw:.4f}")
            if "ece_calibrated" in metrics:
                parts.append(f"ece_cal={metrics['ece_calibrated']:.4f}")
            if "f1_calibrated" in metrics:
                parts.append(f"f1_cal={metrics['f1_calibrated']:.4f}")
            if "f1_calibrated_deployment" in metrics:
                parts.append(
                    f"deploy_f1={metrics['f1_calibrated_deployment']:.4f}"
                    f"@t={metrics.get('deployment_threshold', 0):.3f}"
                )
            opt_t = metrics.get("optimal_threshold")
            opt_f = metrics.get("optimal_f1")
            if opt_t is not None and opt_f is not None:
                parts.append(f"oracle_raw_f1={opt_f:.4f}@{opt_t:.3f}")
            if "oracle_f1_calibrated" in metrics:
                parts.append(
                    f"oracle_cal_f1={metrics['oracle_f1_calibrated']:.4f}"
                    f"@{metrics.get('oracle_threshold_calibrated', 0):.3f}"
                )
        elif task == "severity_cls":
            parts.append(f"macro_f1={metrics.get('sev_macro_f1', 0):.4f}")
        elif task == "severity_reg":
            parts.append(f"mae={metrics.get('sev_mae', 0):.4f}")
            parts.append(f"rmse={metrics.get('sev_rmse', 0):.4f}")
            parts.append(f"corr={metrics.get('correlation', 0):.4f}")
        elif task == "combined_risk":
            parts.append(f"n={metrics.get('combined_risk_n', 0)}")
            parts.append(f"mean_risk={metrics.get('expected_risk_mean', 0):.4f}")
        if "spatial_n_samples" in metrics or "spatial_n" in metrics:
            sn = metrics.get("spatial_n_samples", metrics.get("spatial_n", 0))
            parts.append(f"spatial_n={sn}")
            if task == "occurrence":
                sp_prev = metrics.get("spatial_prevalence")
                sp_f1 = metrics.get("spatial_f1")
                sp_roc = metrics.get("spatial_roc_auc")
                sp_pr = metrics.get("spatial_pr_auc")

                def _fmt(v: object) -> str:
                    return f"{v:.4f}" if isinstance(v, (int, float)) else "n/a"

                parts.append(f"sp_prev={_fmt(sp_prev)}")
                parts.append(f"sp_f1={_fmt(sp_f1)}")
                parts.append(f"sp_roc={_fmt(sp_roc)}")
                parts.append(f"sp_pr={_fmt(sp_pr)}")
        typer.echo("  ".join(parts))

    typer.echo("=" * 72)


if __name__ == "__main__":
    app()
