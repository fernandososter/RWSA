from __future__ import annotations

import torch
import torch.nn as nn

from ..config import ModelConfig
from .staging_base import BaseStagingModel
from .staging_encoder import StagingCNNEncoder


class SleepStagingLSTM(BaseStagingModel):
    """
    Modelo CNN + LSTM ou CNN + BiLSTM.

    Entrada:
        [B, T, C, N]

    Saída:
        [B, T, num_classes]
    """

    model_name = "cnn_lstm"

    def __init__(
        self,
        config: ModelConfig | None = None,
        *,
        hidden_size: int | None = None,
        num_layers: int = 1,
        bidirectional: bool = True,
        use_se: bool = True,
        num_classes: int = 5,
    ) -> None:
        super().__init__()

        self.cfg = config or ModelConfig()
        self.num_classes = num_classes
        self.hidden_size = (
            hidden_size
            if hidden_size is not None
            else self.cfg.d_model // 2
        )
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.encoder = StagingCNNEncoder(
            config=self.cfg,
            use_se=use_se,
        )

        self.temporal = nn.LSTM(
            input_size=self.encoder.output_dim,
            hidden_size=self.hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=(
                self.cfg.dropout
                if num_layers > 1
                else 0.0
            ),
        )

        temporal_output_dim = self.hidden_size * (
            2 if bidirectional else 1
        )

        self.classifier = nn.Sequential(
            nn.Linear(
                temporal_output_dim,
                temporal_output_dim // 2,
            ),
            nn.ReLU(inplace=True),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(
                temporal_output_dim // 2,
                num_classes,
            ),
        )

    def forward(
        self,
        signals: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        features = self.encoder(signals)

        temporal_features, _ = self.temporal(features)

        logits = self.classifier(temporal_features)

        if mask is not None:
            logits = logits.masked_fill(
                ~mask.unsqueeze(-1),
                0.0,
            )

        return logits