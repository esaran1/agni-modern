"""Training loop for temporal multitask transformer with early stopping."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from agni_modern.evaluation.metrics_occurrence import occurrence_metrics
from agni_modern.models.transformer import TemporalTransformer, TransformerConfig, multitask_loss
from agni_modern.training.calibration import (
    apply_calibrator,
    calibrator_path_for,
    find_optimal_f1_threshold,
    fit_isotonic_calibrator,
    load_calibrator,
    save_calibrator,
    save_threshold,
    threshold_path_for,
)
from agni_modern.training.dataset import PatchSequenceDataset, infer_feature_columns
from agni_modern.training.utils import save_metrics, set_global_seed


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _run_train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    occ_weight: float,
    sev_weight: float,
    grad_clip_norm: float,
    device: torch.device,
) -> float:
    """Run one training epoch and return average loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    for x, y_occ, y_sev, y_mask in loader:
        x = x.to(device)
        y_occ = y_occ.to(device)
        y_sev = y_sev.to(device)
        y_mask = y_mask.to(device)

        optimizer.zero_grad()
        outputs = model(x)
        loss = multitask_loss(outputs, y_occ, y_sev, y_mask, occ_weight, sev_weight)
        loss.backward()
        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def _evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    occ_weight: float,
    sev_weight: float,
    device: torch.device,
) -> tuple[float, dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate on a DataLoader.

    Returns
    -------
    avg_loss : float
    occ_metrics : dict
    probs : ndarray – occurrence probabilities per sample
    y_true : ndarray – ground-truth occurrence labels per sample
    sev_preds : ndarray – severity head predictions per sample
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_probs: list[torch.Tensor] = []
    all_true: list[torch.Tensor] = []
    all_sev: list[torch.Tensor] = []

    for x, y_occ, y_sev, y_mask in loader:
        x = x.to(device)
        y_occ = y_occ.to(device)
        y_sev = y_sev.to(device)
        y_mask = y_mask.to(device)

        outputs = model(x)
        loss = multitask_loss(outputs, y_occ, y_sev, y_mask, occ_weight, sev_weight)
        total_loss += loss.item()
        n_batches += 1

        all_probs.append(torch.sigmoid(outputs["occurrence_logit"]).cpu())
        all_true.append(y_occ.cpu())
        all_sev.append(outputs["severity_pred"].cpu())

    avg_loss = total_loss / max(n_batches, 1)

    if all_probs:
        probs = torch.cat(all_probs).numpy()
        true = torch.cat(all_true).numpy()
        sev_preds = torch.cat(all_sev).numpy()
        preds = (probs >= 0.5).astype(int)
        occ_m = occurrence_metrics(true, probs, preds)
    else:
        probs = np.array([])
        true = np.array([])
        sev_preds = np.array([])
        occ_m = {}

    return avg_loss, occ_m, probs, true, sev_preds


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def train_transformer_multitask(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    output_model_path: Path,
    output_metrics_path: Path,
    params: dict[str, object],
    seed: int = 42,
    *,
    test_df: pd.DataFrame | None = None,
    target_col: str = "y_occ_30d",
    device: str = "cpu",
    output_predictions_path: Path | None = None,
    verbose: bool = True,
) -> dict[str, object]:
    """Train multitask transformer with early stopping and per-epoch validation.

    Supports
    --------
    - Multi-epoch training with cosine-annealing LR schedule
    - Gradient clipping
    - Early stopping on validation multitask loss
    - Per-epoch occurrence metrics (F1, ROC-AUC, PR-AUC) on the validation set
    - Optional test-set evaluation with saved predictions
    """
    set_global_seed(seed)
    dev = torch.device(device)

    # ---- hyper-parameters from merged config dict ----
    feature_cols = infer_feature_columns(train_df)
    seq_len = int(params.get("max_seq_len", 16))
    batch_size = int(params.get("batch_size", 64))
    max_epochs = int(params.get("max_epochs", 40))
    patience = int(params.get("early_stopping_patience", 6))
    grad_clip_norm = float(params.get("grad_clip_norm", 1.0))
    lr = float(params.get("learning_rate", 3e-4))
    weight_decay = float(params.get("weight_decay", 1e-4))
    occ_weight = float(params.get("occurrence_weight", 1.0))
    sev_weight = float(params.get("severity_weight", 0.5))

    # ---- datasets / loaders ----
    train_ds = PatchSequenceDataset(train_df, feature_cols, seq_len=seq_len, occ_col=target_col)
    val_ds = PatchSequenceDataset(val_df, feature_cols, seq_len=seq_len, occ_col=target_col)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # ---- model / optimizer / scheduler ----
    model_cfg = TransformerConfig(
        input_dim=len(feature_cols),
        d_model=int(params.get("d_model", 128)),
        nhead=int(params.get("nhead", 4)),
        num_encoder_layers=int(params.get("num_encoder_layers", 3)),
        dim_feedforward=int(params.get("dim_feedforward", 256)),
        dropout=float(params.get("dropout", 0.1)),
        max_seq_len=seq_len,
    )
    model = TemporalTransformer(model_cfg).to(dev)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs)

    # ---- training loop ----
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, object]] = []

    for epoch in range(1, max_epochs + 1):
        train_loss = _run_train_epoch(
            model, train_loader, optimizer, occ_weight, sev_weight, grad_clip_norm, dev,
        )
        val_loss, val_occ_m, _, _, _ = _evaluate(model, val_loader, occ_weight, sev_weight, dev)
        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        record: dict[str, object] = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "lr": current_lr,
        }
        for k, v in val_occ_m.items():
            record[f"val_{k}"] = round(v, 6) if isinstance(v, float) and np.isfinite(v) else None
        history.append(record)

        if verbose:
            f1_str = f"{val_occ_m['f1']:.4f}" if "f1" in val_occ_m else "n/a"
            print(
                f"[epoch {epoch:>3d}/{max_epochs}]  "
                f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                f"val_f1={f1_str}  lr={current_lr:.2e}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch} (patience={patience})")
                break

    # ---- restore best checkpoint ----
    if best_state is not None:
        model.load_state_dict(best_state)
        model = model.to(dev)

    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_model_path)

    final: dict[str, object] = {
        "best_val_loss": round(best_val_loss, 6),
        "total_epochs": len(history),
        "early_stopped": epochs_no_improve >= patience,
    }

    # ---- validation-set isotonic calibration + deployment threshold (tabular parity) ----
    _, _, val_probs, val_true_arr, _ = _evaluate(
        model, val_loader, occ_weight, sev_weight, dev,
    )
    deployment_threshold = 0.5
    if len(val_probs) > 0 and len(np.unique(val_true_arr)) >= 2:
        occ_cal = fit_isotonic_calibrator(val_true_arr, val_probs)
        save_calibrator(occ_cal, calibrator_path_for(output_model_path))
        val_cal = apply_calibrator(occ_cal, val_probs)
        deployment_threshold, val_f1_at_t = find_optimal_f1_threshold(val_true_arr, val_cal)
        save_threshold(deployment_threshold, val_f1_at_t, threshold_path_for(output_model_path))
        final["deployment_threshold"] = round(float(deployment_threshold), 6)
        final["val_f1_at_deployment_threshold"] = round(float(val_f1_at_t), 6)
    else:
        if verbose and len(val_probs) > 0:
            print(
                "[transformer] Skipping isotonic calibration — validation split has a single "
                "occurrence class; threshold defaults to 0.5."
            )

    # ---- test-set evaluation ----
    if test_df is not None and not test_df.empty:
        test_ds = PatchSequenceDataset(test_df, feature_cols, seq_len=seq_len, occ_col=target_col)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
        test_loss, test_occ_m, test_probs, test_true, test_sev = _evaluate(
            model, test_loader, occ_weight, sev_weight, dev,
        )
        final["test_loss"] = round(test_loss, 6)
        for k, v in test_occ_m.items():
            final[f"test_{k}"] = round(v, 6) if isinstance(v, float) and np.isfinite(v) else None

        if output_predictions_path is not None and len(test_probs) > 0:
            output_predictions_path.parent.mkdir(parents=True, exist_ok=True)
            pred_df = pd.DataFrame(test_ds.meta)
            pred_df["y_true"] = test_true
            pred_df["y_prob"] = test_probs
            pred_df["y_pred"] = (test_probs >= 0.5).astype(int)
            pred_df["severity_pred"] = test_sev
            cal_path = calibrator_path_for(output_model_path)
            if cal_path.exists():
                occ_cal = load_calibrator(cal_path)
                test_cal = apply_calibrator(occ_cal, test_probs)
                pred_df["y_prob_calibrated"] = test_cal
                pred_df["occurrence_decision_threshold"] = float(deployment_threshold)
                pred_df["y_pred_calibrated_tuned"] = (
                    test_cal >= float(deployment_threshold)
                ).astype(int)
                m_dep = occurrence_metrics(
                    test_true,
                    test_cal,
                    pred_df["y_pred_calibrated_tuned"].to_numpy(),
                )
                final["test_f1_calibrated_deployment"] = round(float(m_dep["f1"]), 6)
            pred_df.to_parquet(output_predictions_path, index=False)

    final["history"] = history
    save_metrics(final, output_metrics_path)
    return final
