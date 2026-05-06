"""Train tabular baseline model(s) for occurrence and/or severity prediction."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer
import yaml

from agni_modern.evaluation.splits import temporal_holdout_split
from agni_modern.training.train_tabular import (
    train_tabular_occurrence,
    train_tabular_severity_classification,
    train_tabular_severity_regression,
)
from agni_modern.utils.config_loader import load_experiment_config

app = typer.Typer(add_completion=False)

_MODEL_CONFIG_PATHS = {
    "logreg": Path("configs/models/logreg.yaml"),
    "random_forest": Path("configs/models/rf.yaml"),
    "xgb_occurrence": Path("configs/models/xgb_occurrence.yaml"),
    "xgb_severity_cls": Path("configs/models/xgb_severity_cls.yaml"),
    "xgb_severity_reg": Path("configs/models/xgb_severity_reg.yaml"),
}

_SEVERITY_TASK_TYPES = {"severity", "severity_cls", "severity_reg", "severity_both"}


def _load_model_params(model_name: str, fallback_params: dict[str, object]) -> dict[str, object]:
    path = _MODEL_CONFIG_PATHS.get(model_name)
    if path is None or not path.exists():
        return fallback_params
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    params = cfg.get("params", {})
    return params if isinstance(params, dict) else fallback_params


def _artifact_paths(
    cfg, model_name: str
) -> tuple[Path, Path, Path]:
    model_path = Path(cfg.outputs.model_dir) / f"{cfg.experiment.name}_{model_name}.pkl"
    metrics_path = Path(cfg.outputs.metrics_dir) / f"{cfg.experiment.name}_{model_name}.json"
    preds_path = (
        Path(cfg.outputs.metrics_dir) / f"{cfg.experiment.name}_{model_name}_predictions.parquet"
    )
    return model_path, metrics_path, preds_path


def _run_occurrence(
    cfg, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> None:
    horizon = int(cfg.task.get("horizon_days", 30))
    target_col = f"y_occ_{horizon}d"

    requested = cfg.task.get("baseline_models")
    model_names = (
        [str(m) for m in requested] if isinstance(requested, list) and requested else [cfg.model.name]
    )

    for model_name in model_names:
        params = _load_model_params(model_name, cfg.model.params)
        model_path, metrics_path, preds_path = _artifact_paths(cfg, model_name)

        metrics = train_tabular_occurrence(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            model_name=model_name,
            model_params=params,
            output_model_path=model_path,
            output_metrics_path=metrics_path,
            output_predictions_path=preds_path,
            seed=cfg.experiment.seed,
            target_col=target_col,
        )
        parts = [
            f"[occurrence] {model_name}",
            f"f1={metrics['f1']:.4f}",
            f"roc_auc={metrics['roc_auc']:.4f}",
            f"ece_raw={metrics.get('ece_raw', 0):.4f}",
        ]
        if "ece_calibrated" in metrics:
            parts.append(f"ece_cal={metrics['ece_calibrated']:.4f}")
        if "optimal_threshold" in metrics:
            parts.append(
                f"f1@tuned={metrics['f1_at_optimal_threshold']:.4f}"
                f"(t={metrics['optimal_threshold']:.3f})"
            )
        typer.echo(" — ".join(parts))


def _run_severity(
    cfg, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> None:
    task_type = cfg.task.get("type", "severity_both")
    requested = cfg.task.get("severity_models")
    if isinstance(requested, list) and requested:
        model_names = [str(m) for m in requested]
    else:
        model_names = []
        if task_type in {"severity", "severity_both", "severity_cls"}:
            model_names.append("xgb_severity_cls")
        if task_type in {"severity", "severity_both", "severity_reg"}:
            model_names.append("xgb_severity_reg")

    sev_col = "y_sev_available"
    has_severity = (
        sev_col in train_df.columns
        and int(train_df[sev_col].sum()) > 0
        and int(val_df[sev_col].sum()) > 0
    )
    if not has_severity:
        typer.echo(
            "[severity] Skipped — no severity-available rows in train/val splits. "
            "This is expected when pre/post-fire composites are not yet wired up."
        )
        return

    def _severity_class_count(df: pd.DataFrame) -> int:
        subset = df[df["y_sev_available"] == 1]
        if subset.empty:
            return 0
        return int(subset["y_sev_cls"].dropna().nunique())

    n_cls_train = _severity_class_count(train_df)
    n_cls_val = _severity_class_count(val_df)
    n_cls_test = _severity_class_count(test_df)

    for model_name in model_names:
        params = _load_model_params(model_name, cfg.model.params)
        model_path, metrics_path, preds_path = _artifact_paths(cfg, model_name)

        if model_name.endswith("_cls"):
            if n_cls_train < 2:
                typer.echo(
                    "[severity-cls] Skipped — train split has fewer than 2 severity classes. "
                    "A one-class classifier yields meaningless perfect metrics."
                )
                continue
            if n_cls_test < 2:
                typer.echo(
                    "[severity-cls] Warning — test split has fewer than 2 severity classes. "
                    "Reported macro-F1 will be weak evidence only."
                )
            metrics = train_tabular_severity_classification(
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                model_name=model_name,
                model_params=params,
                output_model_path=model_path,
                output_metrics_path=metrics_path,
                output_predictions_path=preds_path,
                seed=cfg.experiment.seed,
            )
            typer.echo(
                f"[severity-cls] {model_name} — "
                f"macro_f1={metrics['sev_macro_f1']:.4f}  "
                f"n_train={metrics['n_train']}  n_test={metrics['n_test']}"
            )
        else:
            metrics = train_tabular_severity_regression(
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                model_name=model_name,
                model_params=params,
                output_model_path=model_path,
                output_metrics_path=metrics_path,
                output_predictions_path=preds_path,
                seed=cfg.experiment.seed,
            )
            typer.echo(
                f"[severity-reg] {model_name} — "
                f"mae={metrics['sev_mae']:.4f}  rmse={metrics['sev_rmse']:.4f}  "
                f"n_train={metrics['n_train']}  n_test={metrics['n_test']}"
            )


@app.command()
def main(config: str, set: list[str] | None = typer.Option(None, "--set")) -> None:
    """Train baseline model(s) for occurrence and/or severity tasks."""
    cfg = load_experiment_config(config, overrides=set)
    df = pd.read_parquet(cfg.data.io.dataset_path)

    train_df, val_df, test_df = temporal_holdout_split(
        df,
        train_end=cfg.split.train_end or "2021-12-31",
        val_end=cfg.split.val_end or "2022-12-31",
        test_end=cfg.split.test_end or "2023-12-31",
    )

    task_type = cfg.task.get("type", "occurrence")

    if task_type == "occurrence":
        _run_occurrence(cfg, train_df, val_df, test_df)
    elif task_type in _SEVERITY_TASK_TYPES:
        _run_severity(cfg, train_df, val_df, test_df)
    elif task_type == "all":
        _run_occurrence(cfg, train_df, val_df, test_df)
        _run_severity(cfg, train_df, val_df, test_df)
    else:
        typer.echo(f"Unknown task type: {task_type}. Expected: occurrence | severity | severity_cls | severity_reg | severity_both | all")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
