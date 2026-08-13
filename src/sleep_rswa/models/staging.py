from __future__ import annotations

import torch
import torch.nn as nn

from ..config import ModelConfig
from .mamba import MambaStack
from .staging_base import BaseStagingModel
from .staging_encoder import StagingCNNEncoder


class SleepStagingMamba(BaseStagingModel):
    """
    Modelo CNN + Mamba/BiMamba para sleep staging.

    Mantém o mesmo comportamento do SleepStagingNet anterior.
    """

    model_name = "cnn_mamba"

    def __init__(
        self,
        config: ModelConfig | None = None,
        *,
        bidirectional: bool = False,
        use_se: bool = True,
        num_classes: int = 5,
    ) -> None:
        super().__init__()

        self.cfg = config or ModelConfig()
        self.num_classes = num_classes
        self.bidirectional = bidirectional

        self.encoder = StagingCNNEncoder(
            config=self.cfg,
            use_se=use_se,
        )

        self.temporal = MambaStack(
            self.encoder.output_dim,
            self.cfg.staging_mamba_layers,
            self.cfg.d_state,
            self.cfg.dropout,
            self.bidirectional,
        )

        self.classifier = nn.Sequential(
            nn.Linear(
                self.encoder.output_dim,
                self.encoder.output_dim // 2,
            ),
            nn.ReLU(inplace=True),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(
                self.encoder.output_dim // 2,
                num_classes,
            ),
        )

    def forward(
        self,
        signals: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        features = self.encoder(signals)

        temporal_features = self.temporal(
            features,
            mask,
        )

        return self.classifier(temporal_features)


class SleepStagingBiMamba(SleepStagingMamba):
    """Alias bidirecional para o ablation."""

    model_name = "cnn_bimamba"

    def __init__(
        self,
        config: ModelConfig | None = None,
        *,
        use_se: bool = True,
        num_classes: int = 5,
    ) -> None:
        super().__init__(
            config=config,
            bidirectional=True,
            use_se=use_se,
            num_classes=num_classes,
        )


# Alias temporário para não quebrar scripts antigos.
SleepStagingNet = SleepStagingBiMamba
