from __future__ import annotations

import torch
import torch.nn as nn


class ExplicitBidirectionalRNN(nn.Module):
    """Empilha duas RNNs unidirecionais explícitas: forward e backward.

    A direção backward processa a sequência invertida no tempo e depois
    desfaz a inversão antes da concatenação final.
    """

    def __init__(
        self,
        rnn_cls: type[nn.Module],
        *,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layer_dropout = dropout if num_layers > 1 else 0.0
        self.rnn_cls = rnn_cls
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = layer_dropout

        self.fwd = self.rnn_cls(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=layer_dropout,
        )
        self.bwd = self.rnn_cls(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=layer_dropout,
        )

    @staticmethod
    def _flip_time(x: torch.Tensor) -> torch.Tensor:
        return torch.flip(x, dims=[1])

    def _forward_direction(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...] | torch.Tensor]:
        return self.fwd(x)

    def _backward_direction(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...] | torch.Tensor]:
        y, state = self.bwd(self._flip_time(x))
        return self._flip_time(y), state

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[tuple[torch.Tensor, ...] | torch.Tensor, tuple[torch.Tensor, ...] | torch.Tensor]]:
        forward_out, forward_state = self._forward_direction(x)
        backward_out, backward_state = self._backward_direction(x)
        output = torch.cat(
            [forward_out, backward_out],
            dim=-1,
        )
        return output, (forward_state, backward_state)

