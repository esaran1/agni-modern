"""Regression tests for research-grade hardening fixes.

Covers:
  - Severity classification target uses the *clipped* dNBR so that
    ``y_sev_cls`` and ``y_sev_reg`` always agree on extreme values.
  - Transformer training applies positive-class weighting derived from the
    training prevalence (without it, BCE collapses on imbalanced data).
  - The ``PatchSequenceDataset`` emits a padding mask aligned with the
    front-padded sequences and uses median imputation rather than zero-fill.
  - Trained models drop a ``feature_cols.json`` sidecar that inference
    consumes to guarantee feature alignment between train and inference.
  - Atomic text/bytes writes do not leave partial files on disk.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from agni_modern.data.synthetic import SyntheticDataConfig, generate_synthetic_patch_date_table
from agni_modern.evaluation.splits import temporal_holdout_split
from agni_modern.inference.predict import load_model, run_inference
from agni_modern.labels.severity import (
    SeverityThresholds,
    build_severity_labels,
    severity_class_from_dnbr,
)
from agni_modern.models.transformer import TemporalTransformer, TransformerConfig, multitask_loss
from agni_modern.training.dataset import PatchSequenceDataset, infer_feature_columns
from agni_modern.training.train_tabular import train_tabular_occurrence
from agni_modern.training.utils import (
    atomic_write_bytes,
    atomic_write_text,
    feature_cols_path_for,
    load_feature_cols,
    save_feature_cols,
)


def test_severity_class_uses_clipped_dnbr() -> None:
    """Out-of-range dNBR must produce clipped reg AND consistent class."""
    thresholds = SeverityThresholds()
    extreme_dnbr = 5.0  # well above max_dnbr=1.3
    out = build_severity_labels(
        reference_date=pd.Timestamp("2022-06-15").date(),
        context={"y_occ_30d": 1, "dnbr": extreme_dnbr},
        thresholds=thresholds,
        min_dnbr=-0.1,
        max_dnbr=1.3,
    )
    assert out["y_sev_available"] == 1
    assert out["y_sev_reg"] == pytest.approx(1.3)
    assert out["y_sev_cls"] == severity_class_from_dnbr(1.3, thresholds)

    low = build_severity_labels(
        reference_date=pd.Timestamp("2022-06-15").date(),
        context={"y_occ_30d": 1, "dnbr": -2.0},
        thresholds=thresholds,
        min_dnbr=-0.1,
        max_dnbr=1.3,
    )
    assert low["y_sev_reg"] == pytest.approx(-0.1)
    assert low["y_sev_cls"] == severity_class_from_dnbr(-0.1, thresholds)


def test_patch_sequence_dataset_pad_mask_and_imputation() -> None:
    """Padding mask aligns with front-padding; NaNs are imputed via medians."""
    df = generate_synthetic_patch_date_table(
        SyntheticDataConfig(seed=1, num_patches=4, num_reference_dates=10)
    )
    feature_cols = infer_feature_columns(df)
    df.loc[df.index[:3], feature_cols[0]] = np.nan

    ds = PatchSequenceDataset(df, feature_cols, seq_len=8, occ_col="y_occ_30d")
    x, pad_mask, y_occ, y_sev, y_mask = ds[0]

    assert x.shape == (8, len(feature_cols))
    assert pad_mask.shape == (8,)
    assert pad_mask.dtype == torch.bool
    # First sample for any patch has only 1 real timestep: 7 padded + 1 real.
    assert int(pad_mask.sum().item()) == 7
    assert pad_mask[-1].item() is False
    # No NaNs survive imputation.
    assert not torch.isnan(x).any()


def test_multitask_loss_pos_weight_changes_gradient() -> None:
    """pos_weight must change the loss value on imbalanced labels."""
    cfg = TransformerConfig(input_dim=6, max_seq_len=4, d_model=8, nhead=2,
                            num_encoder_layers=1, dim_feedforward=16, dropout=0.0)
    torch.manual_seed(0)
    model = TemporalTransformer(cfg)
    x = torch.randn(8, 4, 6)
    out = model(x)
    y_occ = torch.tensor([0, 0, 0, 0, 0, 0, 0, 1], dtype=torch.float32)
    y_sev = torch.zeros(8)
    mask = torch.zeros(8)

    loss_unweighted = multitask_loss(out, y_occ, y_sev, mask).item()
    loss_weighted = multitask_loss(out, y_occ, y_sev, mask, pos_weight=7.0).item()
    assert loss_weighted != pytest.approx(loss_unweighted)


def test_feature_cols_sidecar_and_inference_alignment(tmp_path: Path) -> None:
    """Trained model writes feature_cols.json and inference consumes it."""
    df = generate_synthetic_patch_date_table(
        SyntheticDataConfig(seed=11, num_patches=20, num_reference_dates=40)
    )
    train, val, test = temporal_holdout_split(df, "2021-06-30", "2021-09-30", "2021-12-31")

    occ_path = tmp_path / "occ.pkl"
    sev_path = tmp_path / "sev.pkl"
    train_tabular_occurrence(
        train, val, test, "xgb_occurrence",
        {"n_estimators": 20, "max_depth": 3, "learning_rate": 0.1},
        occ_path, tmp_path / "occ.json",
    )
    from agni_modern.training.train_tabular import train_tabular_severity_regression
    train_tabular_severity_regression(
        train, val, test, "xgb_severity_reg",
        {"n_estimators": 20, "max_depth": 3, "learning_rate": 0.1,
         "objective": "reg:squarederror"},
        sev_path, tmp_path / "sev.json",
    )

    sidecar = feature_cols_path_for(occ_path)
    assert sidecar.exists()
    saved_cols = load_feature_cols(sidecar)
    assert saved_cols is not None and len(saved_cols) > 0

    inference_df = test.copy()
    # Add a column that did not exist at training time.
    inference_df["optical_brand_new_signal"] = 0.5
    occ_model = load_model("xgb_occurrence", occ_path)
    sev_model = load_model("xgb_severity_reg", sev_path)
    out = run_inference(inference_df, occ_model, sev_model, occurrence_model_path=occ_path)
    assert "p_fire" in out.columns
    assert "expected_risk" in out.columns
    assert len(out) == len(inference_df)


def test_save_feature_cols_round_trip(tmp_path: Path) -> None:
    cols = ["optical_a", "weather_b", "human_c"]
    path = tmp_path / "m.feature_cols.json"
    save_feature_cols(cols, path)
    assert load_feature_cols(path) == cols


def test_atomic_write_no_partial_file(tmp_path: Path) -> None:
    target = tmp_path / "metrics.json"
    atomic_write_text(target, json.dumps({"x": 1}))
    assert target.exists()
    assert json.loads(target.read_text()) == {"x": 1}
    assert not (tmp_path / "metrics.json.tmp").exists()

    binary_target = tmp_path / "blob.pkl"
    atomic_write_bytes(binary_target, b"hello")
    assert binary_target.read_bytes() == b"hello"
    assert not (tmp_path / "blob.pkl.tmp").exists()
