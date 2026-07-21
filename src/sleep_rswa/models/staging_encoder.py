from __future__ import annotations

import torch
import torch.nn as nn

from ..config import ModelConfig
from .common import (
    MultiKernelCNNBranch,
    SEBlock,
    make_group_norm,
)


class StagingCNNEncoder(nn.Module):
    """
    Extrator CNN compartilhado pelos modelos de sleep staging.

    Entrada:
        x: [batch, sequence, channels, samples]

    Saída:
        features: [batch, sequence, d_model]
    """

    def __init__(
        self,
        config: ModelConfig | None = None,
        *,
        use_se: bool = True,
    ) -> None:
        super().__init__()

        self.cfg = config or ModelConfig()

        self.branches = nn.ModuleList(
            [
                MultiKernelCNNBranch(
                    in_ch=self.cfg.eeg_in_channels,
                    out_ch=self.cfg.branch_filters,
                    kernels=self.cfg.eeg_kernels,
                    n_layers=self.cfg.cnn_layers,
                    drop=self.cfg.dropout,
                ),
                MultiKernelCNNBranch(
                    in_ch=self.cfg.eog_in_channels,
                    out_ch=self.cfg.branch_filters,
                    kernels=self.cfg.eog_kernels,
                    n_layers=self.cfg.cnn_layers,
                    drop=self.cfg.dropout,
                ),
            ]
        )

        merged_channels = self.cfg.branch_filters * 2

        self.se_global = (
            SEBlock(merged_channels)
            if use_se
            else nn.Identity()
        )

        self.branch_projection = nn.Sequential(
            nn.Conv1d(
                merged_channels,
                self.cfg.d_model,
                kernel_size=1,
                bias=False,
            ),
            make_group_norm(self.cfg.d_model),
            nn.ReLU(inplace=True),
        )

        self.spatial_block = nn.Sequential(
            nn.Conv1d(
                self.cfg.d_model,
                self.cfg.d_model,
                kernel_size=3,
                padding=1,
                groups=self.cfg.d_model,
                bias=False,
            ),
            nn.Conv1d(
                self.cfg.d_model,
                self.cfg.d_model,
                kernel_size=1,
                bias=False,
            ),
            make_group_norm(self.cfg.d_model),
            nn.ReLU(inplace=True),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

    @property
    def output_dim(self) -> int:
        return self.cfg.d_model

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                "StagingCNNEncoder esperava entrada "
                "[batch, sequence, channels, samples], "
                f"mas recebeu shape={tuple(x.shape)}."
            )

        batch_size, sequence_length, channels, samples = x.shape

        expected_channels = (
            self.cfg.eeg_in_channels
            + self.cfg.eog_in_channels
        )

        if channels != expected_channels:
            raise ValueError(
                f"Esperados {expected_channels} canais de staging, "
                f"mas foram recebidos {channels}."
            )

        # Cada mini-época é processada independentemente pela CNN.
        x = x.reshape(
            batch_size * sequence_length,
            channels,
            samples,
        )

        eeg_end = self.cfg.eeg_in_channels
        eog_end = eeg_end + self.cfg.eog_in_channels

        eeg = x[:, :eeg_end, :]
        eog = x[:, eeg_end:eog_end, :]

        eeg_features = self.branches[0](eeg)
        eog_features = self.branches[1](eog)

        features = torch.cat(
            [eeg_features, eog_features],
            dim=1,
        )

        features = self.se_global(features)
        features = self.branch_projection(features)
        features = self.spatial_block(features)
        features = self.pool(features).squeeze(-1)

        return features.reshape(
            batch_size,
            sequence_length,
            self.output_dim,
        )

    def n_params(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )