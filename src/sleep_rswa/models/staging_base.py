from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseStagingModel(nn.Module, ABC):
    """
    Contrato comum para todos os modelos de staging.

    Todos os modelos devem receber:

        signals: [B, T, C, N]
        mask:    [B, T] ou None

    E retornar:

        logits:  [B, T, num_classes]
    """

    model_name: str = "base"

    @abstractmethod
    def forward(
        self,
        signals: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    def n_params(
        self,
        *,
        trainable_only: bool = True,
    ) -> int:
        parameters = self.parameters()

        if trainable_only:
            parameters = (
                parameter
                for parameter in parameters
                if parameter.requires_grad
            )

        return sum(
            parameter.numel()
            for parameter in parameters
        )