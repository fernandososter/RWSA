"""Variantes de arquitetura para o ramo de detecção de movimento (RSWA).

Espelha o padrão dos modelos de staging: um encoder CNN compartilhado
(``RSWAFeatureEncoder``) seguido de uma cabeça temporal intercambiável
(nenhuma / LSTM / BiLSTM / BiMamba) e a ``movement_head`` de cabeça única
que detecta "movement" (any = tônico OU fásico) por mini-época.

Todas as variantes têm o MESMO contrato de forward do ``RSWADetectionNet``:
    forward(emg_center: [B, T, C, N], mask: [B, T] | None)
        -> {"movement_logits": [B, T]}
para que sejam intercambiáveis no ``SleepStagingRSWASystem``.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..config import ModelConfig
from .mamba import MambaStack
from .rswa import RSWADetectionNet, RSWAFeatureEncoder


def _movement_head(d_in: int, dropout: float) -> nn.Sequential:
    """Cabeça única de movimento (mesma forma da usada no RSWADetectionNet)."""
    h = d_in // 2
    return nn.Sequential(
        nn.Linear(d_in, h),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(h, 1),
    )


class MovementCNN(nn.Module):
    """CNN-only: a CNN processa cada mini-época individualmente; não há
    comunicação temporal entre posições da sequência."""

    model_name = "cnn"

    def __init__(self, config: ModelConfig | None = None, *, use_se: bool = True) -> None:
        super().__init__()
        self.cfg = config or ModelConfig()
        self.encoder = RSWAFeatureEncoder(self.cfg, use_se)
        self.movement_head = _movement_head(self.cfg.d_model, self.cfg.dropout)

    def forward(self, emg_center: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        del mask
        z = self.encoder(emg_center)
        return {"movement_logits": self.movement_head(z).squeeze(-1)}

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MovementLSTM(nn.Module):
    """CNN + LSTM ou CNN + BiLSTM sobre as features de mini-época."""

    model_name = "cnn_lstm"

    def __init__(
        self,
        config: ModelConfig | None = None,
        *,
        hidden_size: int | None = None,
        num_layers: int = 1,
        bidirectional: bool = True,
        use_se: bool = True,
    ) -> None:
        super().__init__()
        self.cfg = config or ModelConfig()
        self.encoder = RSWAFeatureEncoder(self.cfg, use_se)
        self.hidden_size = hidden_size if hidden_size is not None else self.cfg.d_model // 2
        self.bidirectional = bidirectional
        self.temporal = nn.LSTM(
            input_size=self.cfg.d_model,
            hidden_size=self.hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=self.cfg.dropout if num_layers > 1 else 0.0,
        )
        temporal_output_dim = self.hidden_size * (2 if bidirectional else 1)
        self.movement_head = _movement_head(temporal_output_dim, self.cfg.dropout)

    def forward(self, emg_center: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        z = self.encoder(emg_center)
        temporal_features, _ = self.temporal(z)
        logits = self.movement_head(temporal_features).squeeze(-1)
        if mask is not None:
            logits = logits.masked_fill(~mask, 0.0)
        return {"movement_logits": logits}

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MovementBiMamba(RSWADetectionNet):
    """CNN + BiMamba — arquitetura padrão (idêntica ao RSWADetectionNet)."""

    model_name = "cnn_bimamba"
