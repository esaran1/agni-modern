"""Temporal transformer model for multitask occurrence+severity forecasting."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(slots=True)
class TransformerConfig:
    """Config container for temporal transformer architecture."""

    input_dim: int
    d_model: int = 128
    nhead: int = 4
    num_encoder_layers: int = 3
    dim_feedforward: int = 256
    dropout: float = 0.1
    max_seq_len: int = 64


class PositionalEncoding(nn.Module):
    """Learned positional embeddings for temporal tokens."""

    def __init__(self, max_seq_len: int, d_model: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(max_seq_len, d_model)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, seq_len, _ = x.shape
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).repeat(batch_size, 1)
        return x + self.embedding(positions)


class TemporalTransformer(nn.Module):
    """Transformer encoder with occurrence and severity heads."""

    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.input_proj = nn.Linear(cfg.input_dim, cfg.d_model)
        self.pos_enc = PositionalEncoding(cfg.max_seq_len, cfg.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.num_encoder_layers)

        self.occurrence_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model // 2),
            nn.ReLU(),
            nn.Linear(cfg.d_model // 2, 1),
        )
        self.severity_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model // 2),
            nn.ReLU(),
            nn.Linear(cfg.d_model // 2, 1),
        )

    def forward(self, x: Tensor, padding_mask: Tensor | None = None) -> dict[str, Tensor]:
        """Forward pass.

        Args:
            x: [batch, seq_len, input_dim]
            padding_mask: [batch, seq_len], True for padded tokens.
        """
        h = self.input_proj(x)
        h = self.pos_enc(h)
        encoded = self.encoder(h, src_key_padding_mask=padding_mask)
        pooled = encoded[:, -1, :]

        occ_logit = self.occurrence_head(pooled).squeeze(-1)
        sev_pred = self.severity_head(pooled).squeeze(-1)

        return {"occurrence_logit": occ_logit, "severity_pred": sev_pred}


def multitask_loss(
    outputs: dict[str, Tensor],
    y_occ: Tensor,
    y_sev: Tensor,
    sev_available_mask: Tensor,
    occurrence_weight: float = 1.0,
    severity_weight: float = 0.5,
) -> Tensor:
    """Compute multitask loss with severity masked for non-fire rows."""
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss(reduction="none")

    occ_loss = bce(outputs["occurrence_logit"], y_occ.float())

    raw_sev_loss = mse(outputs["severity_pred"], y_sev.float())
    mask = sev_available_mask.float()
    sev_loss = (raw_sev_loss * mask).sum() / (mask.sum() + 1e-8)

    return occurrence_weight * occ_loss + severity_weight * sev_loss
