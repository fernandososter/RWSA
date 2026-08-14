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

    def forward(self, signals, emg_center, mask=None):
        staging_logits = self.staging_model(signals, mask)
        stage_probs = torch.softmax(staging_logits, dim=-1)
        stage_context = stage_probs.detach() if self.detach_stage_probs else stage_probs
        rswa_kwargs = {"mask": mask}
        if self.use_stage_conditioning and self._rswa_accepts_stage_probs:
            rswa_kwargs["stage_probs"] = stage_context
        return {
            "staging_logits": staging_logits,
            "stage_probs": stage_probs,
            **self.rswa_model(emg_center, **rswa_kwargs),
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

    def forward(
        self,
        signals: torch.Tensor,
        emg_center: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        staging_features = self.staging_encoder(signals)
        rswa_features = self.rswa_encoder(emg_center)
        fused_features = self.fusion(
            torch.cat(
                [staging_features, rswa_features],
                dim=-1,
            )
        )
        temporal_features = self.temporal(fused_features, mask)
        staging_logits = self.staging_classifier(temporal_features)
        return {
            "staging_logits": staging_logits,
            "stage_probs": torch.softmax(staging_logits, dim=-1),
            "tonic_logits": self.tonic_head(temporal_features).squeeze(-1),
            "phasic_logits": self.phasic_head(temporal_features).squeeze(-1),
            "any_logits": self.any_head(temporal_features).squeeze(-1),
        }

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
