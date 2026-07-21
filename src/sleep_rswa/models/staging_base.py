from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn


class BaseStagingModel(nn.Module, ABC):
    """
    Interface comum para modelos de sleep staging.

    Entrada:
        signals: [B, T, C, N]
        mask:    [B, T] ou None

    Saída:
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

    def model_summary(self) -> dict[str, Any]:
        """
        Retorna informações básicas e reutilizáveis sobre o modelo.
        """
        return {
            "model_name": self.model_name,
            "class_name": self.__class__.__name__,
            "trainable_parameters": self.n_params(
                trainable_only=True
            ),
            "total_parameters": self.n_params(
                trainable_only=False
            ),
        }