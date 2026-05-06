from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from agni_modern.data.contracts import required_columns
from agni_modern.data.parquet_io import read_parquet, write_partitioned_parquet
from agni_modern.data.synthetic import SyntheticDataConfig, generate_synthetic_patch_date_table
from agni_modern.evaluation.splits import temporal_holdout_split
from agni_modern.training.train_tabular import (
    train_tabular_occurrence,
    train_tabular_severity_classification,
    train_tabular_severity_regression,
)
from scripts.train_baselines import _run_severity


def test_synthetic_generation_is_deterministic() -> None:
    cfg = SyntheticDataConfig(seed=123, num_patches=8, num_reference_dates=16)
    df_a = generate_synthetic_patch_date_table(cfg)
    df_b = generate_synthetic_patch_date_table(cfg)
    assert df_a.equals(df_b)


def test_synthetic_has_required_columns() -> None:
    df = generate_synthetic_patch_date_table(SyntheticDataConfig(seed=7, num_patches=6, num_reference_dates=12))
    for col in required_columns():
        assert col in df.columns


def test_synthetic_parquet_roundtrip(tmp_path: Path) -> None:
    df = generate_synthetic_patch_date_table(SyntheticDataConfig(seed=9, num_patches=5, num_reference_dates=8))
    out = tmp_path / "synthetic.parquet"
    write_partitioned_parquet(df, out, partition_cols=[])
    back = read_parquet(out)
    assert len(back) == len(df)


def test_synthetic_training_smoke(tmp_path: Path) -> None:
    df = generate_synthetic_patch_date_table(
        SyntheticDataConfig(seed=11, num_patches=10, num_reference_dates=40)
    )
    train, val, test = temporal_holdout_split(df, "2021-04-30", "2021-06-30", "2021-12-31")

    model_path = tmp_path / "syn_logreg.pkl"
    metrics_path = tmp_path / "syn_logreg_metrics.json"
    metrics = train_tabular_occurrence(
        train_df=train,
        val_df=val,
        test_df=test,
        model_name="logreg",
        model_params={"max_iter": 300},
        output_model_path=model_path,
        output_metrics_path=metrics_path,
        seed=11,
        target_col="y_occ_30d",
    )

    assert model_path.exists()
    assert metrics_path.exists()
    assert metrics["f1"] >= 0.0


# ---------------------------------------------------------------------------
# Severity baseline smoke tests
# ---------------------------------------------------------------------------

def _make_severity_splits():
    """Generate synthetic data with splits guaranteed to have severity rows."""
    df = generate_synthetic_patch_date_table(
        SyntheticDataConfig(seed=42, num_patches=40, num_reference_dates=72, start_date="2021-01-01")
    )
    train, val, test = temporal_holdout_split(df, "2021-12-31", "2022-03-31", "2022-06-30")
    assert (train["y_sev_available"] == 1).sum() > 0, "No severity rows in train"
    assert (val["y_sev_available"] == 1).sum() > 0, "No severity rows in val"
    assert (test["y_sev_available"] == 1).sum() > 0, "No severity rows in test"
    return train, val, test


def test_severity_classification_smoke(tmp_path: Path) -> None:
    """XGBoost severity classifier trains, saves artifacts, and returns macro-F1."""
    train, val, test = _make_severity_splits()

    model_path = tmp_path / "sev_cls.pkl"
    metrics_path = tmp_path / "sev_cls_metrics.json"
    preds_path = tmp_path / "sev_cls_preds.parquet"

    metrics = train_tabular_severity_classification(
        train_df=train,
        val_df=val,
        test_df=test,
        model_name="xgb_severity_cls",
        model_params={
            "objective": "multi:softprob",
            "num_class": 3,
            "eval_metric": "mlogloss",
            "n_estimators": 50,
            "learning_rate": 0.1,
            "max_depth": 4,
            "random_state": 42,
            "tree_method": "hist",
            "early_stopping_rounds": 10,
        },
        output_model_path=model_path,
        output_metrics_path=metrics_path,
        output_predictions_path=preds_path,
        seed=42,
    )

    assert model_path.exists()
    assert metrics_path.exists()
    assert preds_path.exists()

    assert "sev_macro_f1" in metrics
    assert metrics["sev_macro_f1"] >= 0.0
    assert metrics["n_train"] > 0
    assert metrics["n_test"] > 0
    assert "class_distribution_train" in metrics
    assert "class_distribution_test" in metrics

    saved = json.loads(metrics_path.read_text())
    assert "sev_macro_f1" in saved

    pred_df = pd.read_parquet(preds_path)
    assert "y_true" in pred_df.columns
    assert "y_pred" in pred_df.columns
    assert len(pred_df) == metrics["n_test"]


def test_severity_regression_smoke(tmp_path: Path) -> None:
    """XGBoost severity regressor trains, saves artifacts, and returns MAE/RMSE."""
    train, val, test = _make_severity_splits()

    model_path = tmp_path / "sev_reg.pkl"
    metrics_path = tmp_path / "sev_reg_metrics.json"
    preds_path = tmp_path / "sev_reg_preds.parquet"

    metrics = train_tabular_severity_regression(
        train_df=train,
        val_df=val,
        test_df=test,
        model_name="xgb_severity_reg",
        model_params={
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "n_estimators": 50,
            "learning_rate": 0.1,
            "max_depth": 4,
            "random_state": 42,
            "tree_method": "hist",
            "early_stopping_rounds": 10,
        },
        output_model_path=model_path,
        output_metrics_path=metrics_path,
        output_predictions_path=preds_path,
        seed=42,
    )

    assert model_path.exists()
    assert metrics_path.exists()
    assert preds_path.exists()

    assert "sev_mae" in metrics
    assert "sev_rmse" in metrics
    assert metrics["sev_mae"] >= 0.0
    assert metrics["sev_rmse"] >= 0.0
    assert metrics["n_train"] > 0
    assert metrics["n_test"] > 0
    assert "y_true_mean" in metrics
    assert "y_pred_mean" in metrics

    saved = json.loads(metrics_path.read_text())
    assert "sev_mae" in saved

    pred_df = pd.read_parquet(preds_path)
    assert "y_true" in pred_df.columns
    assert "y_pred" in pred_df.columns
    assert len(pred_df) == metrics["n_test"]


def test_severity_data_filtering() -> None:
    """Verify that severity training only uses y_sev_available==1 rows."""
    df = generate_synthetic_patch_date_table(
        SyntheticDataConfig(seed=42, num_patches=40, num_reference_dates=72)
    )
    total = len(df)
    sev_available = (df["y_sev_available"] == 1).sum()
    assert 0 < sev_available < total, "Expected partial severity availability"
    sev_rows = df[df["y_sev_available"] == 1]
    assert sev_rows["y_sev_reg"].notna().all(), "All severity-available rows should have y_sev_reg"
    assert sev_rows["y_sev_cls"].notna().all(), "All severity-available rows should have y_sev_cls"


def test_run_severity_skips_one_class_classifier(tmp_path: Path, capsys) -> None:
    df = generate_synthetic_patch_date_table(
        SyntheticDataConfig(seed=42, num_patches=40, num_reference_dates=72, start_date="2021-01-01")
    )
    df.loc[df["y_sev_available"] == 1, "y_sev_cls"] = 0
    train, val, test = temporal_holdout_split(df, "2021-12-31", "2022-03-31", "2022-06-30")

    class Dummy:
        pass

    cfg = Dummy()
    cfg.task = {"type": "severity_cls", "severity_models": ["xgb_severity_cls"]}
    cfg.model = Dummy()
    cfg.model.params = {}
    cfg.experiment = Dummy()
    cfg.experiment.name = "tmp"
    cfg.experiment.seed = 42
    cfg.outputs = Dummy()
    cfg.outputs.model_dir = str(tmp_path)
    cfg.outputs.metrics_dir = str(tmp_path)

    _run_severity(cfg, train, val, test)
    out = capsys.readouterr().out
    assert "Skipped" in out
    assert "fewer than 2 severity classes" in out
