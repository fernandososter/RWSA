import torch
import torch.nn as nn
try:
    from mamba_ssm import Mamba as MambaOfficial
except ImportError:
    MambaOfficial=None

class FallbackSequenceBlock(nn.Module):
    """Fallback bidirecional portátil quando mamba-ssm não está instalado."""
    def __init__(self,d_model,dropout=0.1):
        super().__init__(); self.norm=nn.LayerNorm(d_model); self.rnn=nn.GRU(d_model,d_model//2,batch_first=True,bidirectional=True); self.drop=nn.Dropout(dropout)
    def forward(self,x):
        y,_=self.rnn(self.norm(x)); return x+self.drop(y)

class BidirMambaBlock(nn.Module):
    def __init__(self,d_model,d_state=16,dropout=0.1):
        super().__init__(); self.norm=nn.LayerNorm(d_model); self.drop=nn.Dropout(dropout)
        if MambaOfficial is None:
            self.fallback=FallbackSequenceBlock(d_model,dropout); self.fwd=self.bwd=None
        else:
            self.fallback=None; self.fwd=MambaOfficial(d_model=d_model,d_state=d_state); self.bwd=MambaOfficial(d_model=d_model,d_state=d_state)
    def forward(self,x):
        if self.fallback is not None: return self.fallback(x)
        z=self.norm(x); y=self.fwd(z)+torch.flip(self.bwd(torch.flip(z,[1])),[1]); return x+self.drop(y)

class MambaStack(nn.Module):
    def __init__(self,d_model,n_layers=1,d_state=16,dropout=0.1):
        super().__init__(); self.blocks=nn.ModuleList([BidirMambaBlock(d_model,d_state,dropout) for _ in range(n_layers)]); self.norm_out=nn.LayerNorm(d_model)
    def forward(self,x,mask=None):
        for block in self.blocks: x=block(x)
        x=self.norm_out(x)
        return x if mask is None else x*mask.unsqueeze(-1).to(x.dtype)
