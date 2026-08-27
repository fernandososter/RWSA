from __future__ import annotations

import inspect
from dataclasses import replace

import torch
import torch.nn as nn

from ..config import ModelConfig
from .mamba import MambaStack
from .rswa import RSWAFeatureEncoder
from .staging_encoder import StagingCNNEncoder
from .staging import SleepStagingNet
from .rswa import RSWADetectionNet


def _expand_stage_time(
    stage_tensor: torch.Tensor,
    target_length: int,
) -> torch.Tensor:
    source_length = int(stage_tensor.shape[1])
    if source_length == target_length:
        return stage_tensor
    if source_length <= 0 or target_length <= 0:
        raise ValueError("source_length e target_length precisam ser positivos.")
    if target_length % source_length != 0:
        raise ValueError(
            "Não foi possível alinhar staging ao RSWA: "
            f"T_stage={source_length}, T_rswa={target_length}."
        )
    factor = target_length // source_length
    return stage_tensor.repeat_interleave(factor, dim=1)


class SleepStagingRSWASystem(nn.Module):
    def __init__(self, staging_model=None, rswa_model=None):
        super().__init__()
        self.staging_model = staging_model or SleepStagingNet()
        if rswa_model is None:
            base_cfg = getattr(self.staging_model, "cfg", ModelConfig())
            rswa_cfg = replace(base_cfg, rswa_stage_conditioning=True)
            self.rswa_model = RSWADetectionNet(config=rswa_cfg, stage_conditioning=True)
        else:
            self.rswa_model = rswa_model
        self.cfg = getattr(
            self.staging_model,
            "cfg",
            getattr(self.rswa_model, "cfg", ModelConfig()),
        )
        self.use_stage_conditioning = bool(
            getattr(self.rswa_model, "use_stage_conditioning", False)
        )
        self.detach_stage_probs = bool(
            getattr(self.cfg, "rswa_stage_conditioning_detach", True)
        )
        self._rswa_accepts_stage_probs = (
            "stage_probs" in inspect.signature(self.rswa_model.forward).parameters
        )
        self.last_shape_info: dict[str, tuple[int, ...]] = {}

    def forward(
        self,
        signals,
        emg_center,
        mask=None,
        staging_mask=None,
        rswa_mask=None,
    ):
        if signals.shape[0] != emg_center.shape[0]:
            raise ValueError(
                "signals e emg_center precisam alinhar ao menos em B; "
                f"recebido {tuple(signals.shape)} e {tuple(emg_center.shape)}."
            )
        staging_mask = mask if staging_mask is None else staging_mask
        rswa_mask = mask if rswa_mask is None else rswa_mask
        staging_logits = self.staging_model(signals, staging_mask)
        stage_probs = torch.softmax(staging_logits, dim=-1)
        stage_context = stage_probs.detach() if self.detach_stage_probs else stage_probs
        rswa_kwargs = {"mask": rswa_mask}
        if self.use_stage_conditioning and self._rswa_accepts_stage_probs:
            rswa_kwargs["stage_probs"] = _expand_stage_time(
                stage_context,
                emg_center.shape[1],
            )
        rswa_out = self.rswa_model(emg_center, **rswa_kwargs)
        rswa_shapes = getattr(getattr(self.rswa_model, "encoder", None), "last_shape_info", {})
        self.last_shape_info = {
            "staging_embedding": tuple(stage_probs.shape),
            **rswa_shapes,
        }
        return {
            "staging_logits": staging_logits,
            "stage_probs": stage_probs,
            **rswa_out,
        }

    def n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def _rswa_head(d_in: int, dropout: float) -> nn.Sequential:
    h = d_in // 2
    return nn.Sequential(
        nn.Linear(d_in, h),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(h, 1),
    )


class SharedBiMambaJointSystem(nn.Module):
    """Sistema conjunto com um único tronco temporal bidirecional.

    Fluxo:
      1. ``StagingCNNEncoder`` extrai features EEG/EOG por mini-época.
      2. ``RSWAFeatureEncoder`` extrai features EMG por mini-época.
      3. As duas representações são fundidas por concatenação + projeção.
      4. Um único ``MambaStack`` bidirecional modela a sequência temporal.
      5. Cabeças independentes produzem logits de staging e RSWA.
    """

    def __init__(
        self,
        config: ModelConfig | None = None,
        *,
        use_se: bool = True,
    ) -> None:
        super().__init__()

        self.cfg = config or ModelConfig()
        self.staging_encoder = StagingCNNEncoder(
            config=self.cfg,
            use_se=use_se,
        )
        self.rswa_encoder = RSWAFeatureEncoder(
            config=self.cfg,
            use_se=use_se,
        )

        fused_dim = self.staging_encoder.output_dim + self.cfg.d_model
        self.fusion = nn.Sequential(
            nn.Linear(
                fused_dim,
                self.cfg.d_model,
                bias=False,
            ),
            nn.LayerNorm(self.cfg.d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(self.cfg.dropout),
        )
        self.temporal = MambaStack(
            self.cfg.d_model,
            self.cfg.staging_mamba_layers,
            self.cfg.d_state,
            self.cfg.dropout,
            bidirectional=True,
        )

        self.staging_classifier = nn.Sequential(
            nn.Linear(
                self.cfg.d_model,
                self.cfg.d_model // 2,
            ),
            nn.ReLU(inplace=True),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(
                self.cfg.d_model // 2,
                self.cfg.staging_num_classes,
            ),
        )
        self.tonic_head = _rswa_head(self.cfg.d_model, self.cfg.dropout)
        self.phasic_head = _rswa_head(self.cfg.d_model, self.cfg.dropout)
        self.any_head = _rswa_head(self.cfg.d_model, self.cfg.dropout)

        self.use_stage_conditioning = False
        self.detach_stage_probs = False
        self.last_shape_info: dict[str, tuple[int, ...]] = {}

    def forward(
        self,
        signals: torch.Tensor,
        emg_center: torch.Tensor,
        mask: torch.Tensor | None = None,
        staging_mask: torch.Tensor | None = None,
        rswa_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if signals.shape[:2] != emg_center.shape[:2]:
            raise ValueError(
                "SharedBiMambaJointSystem exige a mesma grade temporal para staging e RSWA; "
                f"recebido {tuple(signals.shape)} e {tuple(emg_center.shape)}."
            )
        if staging_mask is not None and rswa_mask is not None and staging_mask.shape != rswa_mask.shape:
            raise ValueError(
                "SharedBiMambaJointSystem exige máscaras temporais com a mesma shape "
                f"quando ambas são fornecidas: {tuple(staging_mask.shape)} vs {tuple(rswa_mask.shape)}."
            )
        staging_features = self.staging_encoder(signals)
        rswa_features = self.rswa_encoder(emg_center)
        if staging_features.shape[:2] != rswa_features.shape[:2]:
            raise ValueError(
                "staging_features e rswa_features precisam alinhar em (B,T); "
                f"recebido {tuple(staging_features.shape)} e {tuple(rswa_features.shape)}."
            )
        fused_features = self.fusion(
            torch.cat(
                [staging_features, rswa_features],
                dim=-1,
            )
        )
        temporal_mask = rswa_mask if rswa_mask is not None else mask
        temporal_features = self.temporal(fused_features, temporal_mask)
        staging_logits = self.staging_classifier(temporal_features)
        rswa_shapes = getattr(self.rswa_encoder, "last_shape_info", {})
        self.last_shape_info = {
            **rswa_shapes,
            "staging_embedding": tuple(staging_features.shape),
            "pre_mamba_embedding": tuple(fused_features.shape),
        }
        return {
            "staging_logits": staging_logits,
            "stage_probs": torch.softmax(staging_logits, dim=-1),
            "tonic_logits": self.tonic_head(temporal_features).squeeze(-1),
            "phasic_logits": self.phasic_head(temporal_features).squeeze(-1),
            "any_logits": self.any_head(temporal_features).squeeze(-1),
        }

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
