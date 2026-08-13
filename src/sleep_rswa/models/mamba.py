import torch
import torch.nn as nn
try:
    from mamba_ssm import Mamba as MambaOfficial
    MAMBA_IMPORT_ERROR=None
except ImportError as exc:
    MambaOfficial=None
    MAMBA_IMPORT_ERROR=exc


def mamba_backend_name() -> str:
    return "mamba_ssm" if MambaOfficial is not None else "gru_fallback"


def mamba_backend_detail() -> str:
    if MambaOfficial is not None:
        return "mamba_ssm importado com sucesso."
    if MAMBA_IMPORT_ERROR is None:
        return "mamba_ssm indisponivel; usando fallback GRU."
    return (
        "mamba_ssm indisponivel; usando fallback GRU. "
        f"Motivo do import: {type(MAMBA_IMPORT_ERROR).__name__}: {MAMBA_IMPORT_ERROR}"
    )

class _FallbackDirectionalBlock(nn.Module):
    """Fallback unidirecional portátil quando mamba-ssm não está instalado."""

    def __init__(self,d_model):
        super().__init__(); self.rnn=nn.GRU(d_model,d_model,batch_first=True)

    def forward(self,x):
        y,_=self.rnn(x); return y


class MambaBlock(nn.Module):
    """Bloco unidirecional explícito."""

    def __init__(self,d_model,d_state=16,dropout=0.1):
        super().__init__(); self.norm=nn.LayerNorm(d_model); self.drop=nn.Dropout(dropout)
        if MambaOfficial is None:
            self.sequence_impl="gru_fallback"; self.core=_FallbackDirectionalBlock(d_model)
        else:
            self.sequence_impl="mamba"; self.core=MambaOfficial(d_model=d_model,d_state=d_state)

    def extra_repr(self):
        return f"sequence_impl={self.sequence_impl}"

    def forward(self,x):
        z=self.norm(x)
        y=self.core(z)
        return x+self.drop(y)

class BidirMambaBlock(nn.Module):
    """Bloco bidirecional explícito.

    Quando ``mamba_ssm`` está disponível, executa dois módulos independentes:
      1. ``forward_out = fwd_mamba(x)``
      2. ``backward_out = flip(bwd_mamba(flip(x, dim=1)), dim=1)``

    O fallback mantém a mesma topologia conceitual, mas usando GRUs
    unidirecionais em vez de Mamba.
    """

    def __init__(self,d_model,d_state=16,dropout=0.1):
        super().__init__(); self.norm=nn.LayerNorm(d_model); self.drop=nn.Dropout(dropout)
        if MambaOfficial is None:
            self.sequence_impl="gru_fallback"; self.fwd=_FallbackDirectionalBlock(d_model); self.bwd=_FallbackDirectionalBlock(d_model)
        else:
            self.sequence_impl="mamba"; self.fwd=MambaOfficial(d_model=d_model,d_state=d_state); self.bwd=MambaOfficial(d_model=d_model,d_state=d_state)

    def extra_repr(self):
        return f"sequence_impl={self.sequence_impl}"

    @staticmethod
    def _flip_time(x):
        return torch.flip(x,[1])

    def _forward_direction(self,z):
        return self.fwd(z)

    def _backward_direction(self,z):
        return self._flip_time(self.bwd(self._flip_time(z)))

    @staticmethod
    def _combine_directions(forward_out,backward_out):
        return forward_out+backward_out

    def forward(self,x):
        z=self.norm(x)
        forward_out=self._forward_direction(z)
        backward_out=self._backward_direction(z)
        y=self._combine_directions(forward_out,backward_out)
        return x+self.drop(y)

class MambaStack(nn.Module):
    def __init__(self,d_model,n_layers=1,d_state=16,dropout=0.1,bidirectional=True):
        block_cls=BidirMambaBlock if bidirectional else MambaBlock
        super().__init__(); self.blocks=nn.ModuleList([block_cls(d_model,d_state,dropout) for _ in range(n_layers)]); self.norm_out=nn.LayerNorm(d_model)
    def forward(self,x,mask=None):
        for block in self.blocks: x=block(x)
        x=self.norm_out(x)
        return x if mask is None else x*mask.unsqueeze(-1).to(x.dtype)
