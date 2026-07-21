from __future__ import annotations

import torch
import torch.nn as nn

from ..config import ModelConfig
from .staging_base import BaseStagingModel
from .staging_encoder import StagingCNNEncoder


class SleepStagingCNN(BaseStagingModel):
    """
    Modelo CNN-only.

    A CNN processa cada mini-época individualmente.
    Não existe comunicação temporal entre posições da sequência.

    Entrada:
        [B, T, C, N]

    Saída:
        [B, T, num_classes]
    """

    model_name = "cnn"

    def __init__(
        self,
        config: ModelConfig | None = None,
        *,
        use_se: bool = True,
        num_classes: int = 5,
    ) -> None:
        super().__init__()

        self.cfg = config or ModelConfig()
        self.num_classes = num_classes

        self.encoder = StagingCNNEncoder(
            config=self.cfg,
            use_se=use_se,
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
        del mask

        features = self.encoder(signals)

        return self.classifier(features)