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

        # A extração multiescala ocorre apenas na primeira etapa; as camadas
        # profundas usam kernels pequenos compartilhados para evitar que os
        # maiores caminhos operem predominantemente sobre padding após vários
        # poolings sucessivos.
        self.branches = nn.ModuleList(
            [
                MultiKernelCNNBranch(
                    in_ch=self.cfg.eeg_in_channels,
                    out_ch=self.cfg.branch_filters,
                    kernels=self.cfg.eeg_kernels,
                    n_layers=1,
                    drop=self.cfg.dropout,
                ),
                MultiKernelCNNBranch(
                    in_ch=self.cfg.eog_in_channels,
                    out_ch=self.cfg.branch_filters,
                    kernels=self.cfg.eog_kernels,
                    n_layers=1,
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

        self.refine = nn.Sequential(
            nn.Conv1d(
                merged_channels,
                merged_channels,
                kernel_size=5,
                padding=2,
                bias=False,
            ),
            make_group_norm(merged_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(
                merged_channels,
                self.cfg.d_model,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            make_group_norm(self.cfg.d_model),
            nn.ReLU(inplace=True),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)
        self._auto_chunk_min_samples = 1200
        self._auto_chunk_size = 16
        self._auto_chunk_sample_budget = 24_000

    @property
    def output_dim(self) -> int:
        return self.cfg.d_model

    def _encode_flat_epochs(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
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
        features = self.refine(features)
        return self.pool(features).squeeze(-1)

    def _resolve_chunk_size(
        self,
        n_items: int,
        samples: int,
    ) -> int:
        total_points = int(n_items) * int(samples)
        if n_items <= self._auto_chunk_size:
            return 0
        if samples >= self._auto_chunk_min_samples:
            return self._auto_chunk_size
        if total_points >= self._auto_chunk_sample_budget:
            return self._auto_chunk_size
        return self._auto_chunk_size

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
        chunk_size = self._resolve_chunk_size(
            batch_size * sequence_length,
            samples,
        )
        if chunk_size <= 0:
            features = self._encode_flat_epochs(x)
        else:
            chunks: list[torch.Tensor] = []
            for start in range(0, x.shape[0], chunk_size):
                stop = min(start + chunk_size, x.shape[0])
                chunks.append(self._encode_flat_epochs(x[start:stop]))
            features = torch.cat(chunks, dim=0)

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
