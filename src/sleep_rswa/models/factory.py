from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch.nn as nn

from ..config import ModelConfig
from .staging import SleepStagingBiMamba
from .staging_cnn import SleepStagingCNN
from .staging_lstm import SleepStagingLSTM


StagingBuilder = Callable[..., nn.Module]


_STAGING_MODELS: dict[str, StagingBuilder] = {
    "cnn": SleepStagingCNN,
    "cnn_lstm": SleepStagingLSTM,
    "cnn_bilstm": SleepStagingLSTM,
    "cnn_bimamba": SleepStagingBiMamba,
}


def available_staging_models() -> tuple[str, ...]:
    return tuple(sorted(_STAGING_MODELS))


def register_staging_model(
    name: str,
    builder: StagingBuilder,
    *,
    overwrite: bool = False,
) -> None:
    normalized_name = name.strip().lower()

    if not normalized_name:
        raise ValueError(
            "O nome do modelo não pode ser vazio."
        )

    if (
        normalized_name in _STAGING_MODELS
        and not overwrite
    ):
        raise ValueError(
            f"O modelo '{normalized_name}' já está registrado."
        )

    _STAGING_MODELS[normalized_name] = builder


def build_staging_model(
    name: str,
    *,
    config: ModelConfig | None = None,
    **model_kwargs: Any,
) -> nn.Module:
    normalized_name = name.strip().lower()

    try:
        builder = _STAGING_MODELS[normalized_name]
    except KeyError as error:
        available = ", ".join(
            available_staging_models()
        )

        raise ValueError(
            f"Modelo de staging desconhecido: '{name}'. "
            f"Modelos disponíveis: {available}."
        ) from error

    # cnn_lstm e cnn_bilstm usam a mesma classe,
    # mudando apenas o parâmetro bidirectional.
    if normalized_name == "cnn_lstm":
        model_kwargs.setdefault(
            "bidirectional",
            False,
        )

    if normalized_name == "cnn_bilstm":
        model_kwargs.setdefault(
            "bidirectional",
            True,
        )

    return builder(
        config=config,
        **model_kwargs,
    )