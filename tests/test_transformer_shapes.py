import json

import pandas as pd
import torch

from agni_modern.models.transformer import TemporalTransformer, TransformerConfig, multitask_loss


def test_transformer_forward_shapes() -> None:
    cfg = TransformerConfig(input_dim=12, max_seq_len=16)
    model = TemporalTransformer(cfg)
    x = torch.randn(4, 16, 12)
    out = model(x)
    assert out["occurrence_logit"].shape == (4,)
    assert out["severity_pred"].shape == (4,)


def test_multitask_loss_scalar() -> None:
    cfg = TransformerConfig(input_dim=12, max_seq_len=16)
    model = TemporalTransformer(cfg)
    x = torch.randn(4, 16, 12)
    out = model(x)
    loss = multitask_loss(
        outputs=out,
        y_occ=torch.tensor([0, 1, 0, 1], dtype=torch.float32),
        y_sev=torch.tensor([0.0, 0.2, 0.0, 0.8], dtype=torch.float32),
        sev_available_mask=torch.tensor([0, 1, 0, 1], dtype=torch.float32),
    )
    assert loss.ndim == 0


def test_transformer_training_smoke(tmp_path) -> None:
    """Multi-epoch training loop on synthetic data with test eval and predictions."""
    from agni_modern.data.synthetic import SyntheticDataConfig, generate_synthetic_patch_date_table
    from agni_modern.evaluation.splits import temporal_holdout_split
    from agni_modern.training.train_transformer import train_transformer_multitask

    df = generate_synthetic_patch_date_table(
        SyntheticDataConfig(seed=42, num_patches=6, num_reference_dates=30)
    )
    train, val, test = temporal_holdout_split(df, "2021-03-01", "2021-05-01", "2021-12-31")

    model_path = tmp_path / "model.pt"
    metrics_path = tmp_path / "metrics.json"
    preds_path = tmp_path / "preds.parquet"

    metrics = train_transformer_multitask(
        train_df=train,
        val_df=val,
        output_model_path=model_path,
        output_metrics_path=metrics_path,
        params={
            "max_seq_len": 4,
            "batch_size": 8,
            "max_epochs": 5,
            "early_stopping_patience": 3,
            "grad_clip_norm": 1.0,
            "d_model": 16,
            "nhead": 2,
            "num_encoder_layers": 1,
            "dim_feedforward": 32,
            "dropout": 0.0,
        },
        seed=42,
        test_df=test,
        target_col="y_occ_30d",
        output_predictions_path=preds_path,
        verbose=False,
    )

    assert model_path.exists()
    assert metrics_path.exists()

    assert "best_val_loss" in metrics
    assert "total_epochs" in metrics
    assert metrics["total_epochs"] >= 1
    assert isinstance(metrics.get("early_stopped"), bool)

    assert "history" in metrics
    assert isinstance(metrics["history"], list)
    assert len(metrics["history"]) == metrics["total_epochs"]
    first = metrics["history"][0]
    assert "train_loss" in first and "val_loss" in first and "lr" in first

    assert "test_f1" in metrics
    assert "test_loss" in metrics

    assert preds_path.exists()
    pred_df = pd.read_parquet(preds_path)
    assert "y_prob" in pred_df.columns
    assert "severity_pred" in pred_df.columns
    assert len(pred_df) > 0

    saved = json.loads(metrics_path.read_text())
    assert "best_val_loss" in saved


def test_early_stopping_triggers(tmp_path) -> None:
    """Verify early stopping fires before max_epochs with patience=1."""
    from agni_modern.data.synthetic import SyntheticDataConfig, generate_synthetic_patch_date_table
    from agni_modern.evaluation.splits import temporal_holdout_split
    from agni_modern.training.train_transformer import train_transformer_multitask

    df = generate_synthetic_patch_date_table(
        SyntheticDataConfig(seed=99, num_patches=4, num_reference_dates=30)
    )
    train, val, test = temporal_holdout_split(df, "2021-03-01", "2021-05-01", "2021-12-31")

    metrics = train_transformer_multitask(
        train_df=train,
        val_df=val,
        output_model_path=tmp_path / "m.pt",
        output_metrics_path=tmp_path / "m.json",
        params={
            "max_seq_len": 4,
            "batch_size": 16,
            "max_epochs": 50,
            "early_stopping_patience": 1,
            "learning_rate": 0.5,
            "d_model": 8,
            "nhead": 2,
            "num_encoder_layers": 1,
            "dim_feedforward": 16,
            "dropout": 0.0,
        },
        seed=99,
        verbose=False,
    )

    assert metrics["early_stopped"] is True, "Expected early stopping with high LR and patience=1"
    assert metrics["total_epochs"] < 50
