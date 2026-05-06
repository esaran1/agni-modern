"""Smoke test: multitask transformer trains, saves calibration sidecars, writes predictions."""

from __future__ import annotations

from pathlib import Path

import pytest

from agni_modern.data.synthetic import SyntheticDataConfig, generate_synthetic_patch_date_table
from agni_modern.evaluation.splits import temporal_holdout_split
from agni_modern.training.calibration import calibrator_path_for, threshold_path_for
from agni_modern.training.train_transformer import train_transformer_multitask


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_transformer_saves_calibration_and_predictions(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")

    df = generate_synthetic_patch_date_table(
        SyntheticDataConfig(seed=42, num_patches=24, num_reference_dates=96, start_date="2021-01-01")
    )
    train, val, test = temporal_holdout_split(df, "2021-12-31", "2022-03-31", "2022-06-30")

    model_path = tmp_path / "tr.pt"
    metrics_path = tmp_path / "tr.json"
    preds_path = tmp_path / "tr_preds.parquet"

    params = {
        "d_model": 32,
        "nhead": 2,
        "num_encoder_layers": 1,
        "dim_feedforward": 64,
        "dropout": 0.1,
        "max_seq_len": 8,
        "batch_size": 32,
        "max_epochs": 4,
        "early_stopping_patience": 3,
        "grad_clip_norm": 1.0,
        "learning_rate": 3e-4,
        "weight_decay": 1e-4,
        "occurrence_weight": 1.0,
        "severity_weight": 0.3,
    }

    metrics = train_transformer_multitask(
        train_df=train,
        val_df=val,
        output_model_path=model_path,
        output_metrics_path=metrics_path,
        params=params,
        seed=42,
        test_df=test,
        target_col="y_occ_30d",
        device="cpu",
        output_predictions_path=preds_path,
        verbose=False,
    )

    assert model_path.exists()
    assert metrics_path.exists()
    assert preds_path.exists()
    assert metrics["total_epochs"] >= 1

    if len(val) > 0 and val["y_occ_30d"].nunique() >= 2:
        assert calibrator_path_for(model_path).exists()
        assert threshold_path_for(model_path).exists()
        import pandas as pd

        pred = pd.read_parquet(preds_path)
        assert "y_prob_calibrated" in pred.columns
        assert "occurrence_decision_threshold" in pred.columns
        assert "y_pred_calibrated_tuned" in pred.columns
