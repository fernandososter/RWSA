"""Variantes de arquitetura para o ramo RSWA multi-head.

Espelha o padrão dos modelos de staging: um encoder CNN compartilhado
(``RSWAFeatureEncoder``) seguido de uma cabeça temporal intercambiável
(nenhuma / LSTM / BiLSTM / BiMamba) e 3 heads independentes por mini-época:
``tonic_logits``, ``phasic_logits`` e ``any_logits``.

Todas as variantes têm o MESMO contrato do ``RSWADetectionNet``:
    forward(emg_center: [B, T, C, N], mask: [B, T] | None, stage_probs: [B, T, 5] | None)
        -> {"tonic_logits": [B, T], "phasic_logits": [B, T], "any_logits": [B, T]}
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..config import ModelConfig
from .mamba import MambaStack
from .rswa import RSWADetectionNet, RSWAFeatureEncoder, _rswa_head


def _stage_fusion(
    cfg: ModelConfig,
    *,
    use_stage_conditioning: bool,
) -> nn.Module | None:
    if not use_stage_conditioning:
        return None
    return nn.Sequential(
        nn.Linear(
            cfg.d_model + int(cfg.staging_num_classes),
            cfg.d_model,
            bias=False,
        ),
        nn.LayerNorm(cfg.d_model),
        nn.ReLU(inplace=True),
        nn.Dropout(cfg.dropout),
    )


def _apply_stage_conditioning(
    z: torch.Tensor,
    stage_probs: torch.Tensor | None,
    *,
    stage_fusion: nn.Module | None,
    stage_context_dim: int,
) -> torch.Tensor:
    if stage_fusion is None or stage_probs is None:
        return z
    if stage_probs.shape[:2] != z.shape[:2]:
        raise ValueError(
            "stage_probs deve alinhar com (B,T) das features RSWA; "
            f"recebeu {tuple(stage_probs.shape)} para features {tuple(z.shape)}."
        )
    if stage_probs.shape[-1] != stage_context_dim:
        raise ValueError(
            "stage_probs deve ter "
            f"{stage_context_dim} classes de estagio; recebeu "
            f"{stage_probs.shape[-1]}."
        )
    return stage_fusion(torch.cat([z, stage_probs.to(z.dtype)], dim=-1))


class _RSWAHeadMixin:
    def _build_heads(self, d_in: int, dropout: float) -> None:
        self.tonic_head = _rswa_head(d_in, dropout)
        self.phasic_head = _rswa_head(d_in, dropout)
        self.any_head = _rswa_head(d_in, dropout)

    def _pack_outputs(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "tonic_logits": self.tonic_head(z).squeeze(-1),
            "phasic_logits": self.phasic_head(z).squeeze(-1),
            "any_logits": self.any_head(z).squeeze(-1),
        }

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MovementCNN(nn.Module, _RSWAHeadMixin):
    """CNN-only: a CNN processa cada mini-época individualmente."""

    model_name = "cnn"

    def __init__(
        self,
        config: ModelConfig | None = None,
        *,
        use_se: bool = True,
        stage_conditioning: bool | None = None,
    ) -> None:
        super().__init__()
        self.cfg = config or ModelConfig()
        self.encoder = RSWAFeatureEncoder(self.cfg, use_se)
        self.use_stage_conditioning = bool(
            self.cfg.rswa_stage_conditioning
            if stage_conditioning is None
            else stage_conditioning
        )
        self.stage_context_dim = int(self.cfg.staging_num_classes)
        self.stage_fusion = _stage_fusion(
            self.cfg,
            use_stage_conditioning=self.use_stage_conditioning,
        )
        self._build_heads(self.cfg.d_model, self.cfg.dropout)

    def forward(
        self,
        emg_center: torch.Tensor,
        mask: torch.Tensor | None = None,
        stage_probs: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        del mask
        z = self.encoder(emg_center)
        z = _apply_stage_conditioning(
            z,
            stage_probs,
            stage_fusion=self.stage_fusion,
            stage_context_dim=self.stage_context_dim,
        )
        return self._pack_outputs(z)


class MovementLSTM(nn.Module, _RSWAHeadMixin):
    """CNN + LSTM ou CNN + BiLSTM sobre as features de mini-época."""

    model_name = "cnn_lstm"

    def __init__(
        self,
        config: ModelConfig | None = None,
        *,
        hidden_size: int | None = None,
        num_layers: int = 1,
        bidirectional: bool = True,
        rnn_cls: type[nn.Module] = nn.LSTM,
        use_se: bool = True,
        stage_conditioning: bool | None = None,
    ) -> None:
        super().__init__()
        self.cfg = config or ModelConfig()
        self.encoder = RSWAFeatureEncoder(self.cfg, use_se)
        self.use_stage_conditioning = bool(
            self.cfg.rswa_stage_conditioning
            if stage_conditioning is None
            else stage_conditioning
        )
        self.stage_context_dim = int(self.cfg.staging_num_classes)
        self.stage_fusion = _stage_fusion(
            self.cfg,
            use_stage_conditioning=self.use_stage_conditioning,
        )
        self.hidden_size = hidden_size if hidden_size is not None else self.cfg.d_model // 2
        self.bidirectional = bidirectional
        self.rnn_cls = rnn_cls
        self.temporal = self.rnn_cls(
            input_size=self.cfg.d_model,
            hidden_size=self.hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=self.cfg.dropout if num_layers > 1 else 0.0,
        )
        temporal_output_dim = self.hidden_size * (2 if bidirectional else 1)
        self._build_heads(temporal_output_dim, self.cfg.dropout)

    def forward(
        self,
        emg_center: torch.Tensor,
        mask: torch.Tensor | None = None,
        stage_probs: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        z = self.encoder(emg_center)
        z = _apply_stage_conditioning(
            z,
            stage_probs,
            stage_fusion=self.stage_fusion,
            stage_context_dim=self.stage_context_dim,
        )
        temporal_features, _ = self.temporal(z)
        out = self._pack_outputs(temporal_features)
        if mask is not None:
            for key, value in out.items():
                out[key] = value.masked_fill(~mask, 0.0)
        return out


class MovementGRU(MovementLSTM):
    """CNN + GRU ou CNN + BiGRU sobre as features de mini-época."""

    model_name = "cnn_gru"

    def __init__(
        self,
        config: ModelConfig | None = None,
        *,
        hidden_size: int | None = None,
        num_layers: int = 1,
        bidirectional: bool = True,
        use_se: bool = True,
        stage_conditioning: bool | None = None,
    ) -> None:
        super().__init__(
            config=config,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            rnn_cls=nn.GRU,
            use_se=use_se,
            stage_conditioning=stage_conditioning,
        )


class MovementMamba(nn.Module, _RSWAHeadMixin):
    """CNN + Mamba/BiMamba sobre as features de mini-época."""

    model_name = "cnn_mamba"

    def __init__(
        self,
        config: ModelConfig | None = None,
        *,
        bidirectional: bool = False,
        use_se: bool = True,
        stage_conditioning: bool | None = None,
    ) -> None:
        super().__init__()
        self.cfg = config or ModelConfig()
        self.encoder = RSWAFeatureEncoder(self.cfg, use_se)
        self.use_stage_conditioning = bool(
            self.cfg.rswa_stage_conditioning
            if stage_conditioning is None
            else stage_conditioning
        )
        self.stage_context_dim = int(self.cfg.staging_num_classes)
        self.stage_fusion = _stage_fusion(
            self.cfg,
            use_stage_conditioning=self.use_stage_conditioning,
        )
        self.bidirectional = bidirectional
        self.temporal = MambaStack(
            self.cfg.d_model,
            self.cfg.rswa_mamba_layers,
            self.cfg.d_state,
            self.cfg.dropout,
            self.bidirectional,
        )
        self._build_heads(self.cfg.d_model, self.cfg.dropout)

    def forward(
        self,
        emg_center: torch.Tensor,
        mask: torch.Tensor | None = None,
        stage_probs: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        z = self.encoder(emg_center)
        z = _apply_stage_conditioning(
            z,
            stage_probs,
            stage_fusion=self.stage_fusion,
            stage_context_dim=self.stage_context_dim,
        )
        temporal_features = self.temporal(z, mask)
        return self._pack_outputs(temporal_features)


class MovementBiMamba(MovementMamba):
    """CNN + BiMamba — arquitetura padrão do ramo RSWA multi-head."""

    model_name = "cnn_bimamba"

    def __init__(
        self,
        config: ModelConfig | None = None,
        *,
        use_se: bool = True,
        stage_conditioning: bool | None = None,
    ) -> None:
        super().__init__(
            config=config,
            bidirectional=True,
            use_se=use_se,
            stage_conditioning=stage_conditioning,
        )
