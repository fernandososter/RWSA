import torch
import torch.nn as nn
from ..config import ModelConfig
from .common import MultiKernelCNNBranch,SEBlock,make_group_norm
from .mamba import MambaStack

class RSWAFeatureEncoder(nn.Module):
    def __init__(self,config=None,use_se=True):
        super().__init__(); cfg=config or ModelConfig(); self.cfg=cfg
        self.branch=MultiKernelCNNBranch(cfg.rswa_emg_in_channels,cfg.rswa_emg_filters,cfg.emg_kernels,cfg.cnn_layers,cfg.dropout)
        self.se=SEBlock(cfg.rswa_emg_filters) if use_se else nn.Identity()
        self.proj=nn.Sequential(nn.Conv1d(cfg.rswa_emg_filters,cfg.d_model,1,bias=False),make_group_norm(cfg.d_model),nn.ReLU(inplace=True))
        self.spatial=nn.Sequential(nn.Conv1d(cfg.d_model,cfg.d_model,3,padding=1,groups=cfg.d_model,bias=False),nn.Conv1d(cfg.d_model,cfg.d_model,1,bias=False),make_group_norm(cfg.d_model),nn.ReLU(inplace=True)); self.pool=nn.AdaptiveAvgPool1d(1)
    def forward(self,x):
        b,t,c,n=x.shape; z=x.reshape(b*t,c,n); z=self.pool(self.spatial(self.proj(self.se(self.branch(z))))).squeeze(-1); return z.reshape(b,t,-1)

class RSWADetectionNet(nn.Module):
    def __init__(self,config=None,use_se=True):
        super().__init__(); cfg=config or ModelConfig(); self.encoder=RSWAFeatureEncoder(cfg,use_se); self.temporal=MambaStack(cfg.d_model,cfg.rswa_mamba_layers,cfg.d_state,cfg.dropout); h=cfg.d_model//2
        self.tonic_head=nn.Sequential(nn.Linear(cfg.d_model,h),nn.ReLU(inplace=True),nn.Dropout(cfg.dropout),nn.Linear(h,1))
        self.phasic_head=nn.Sequential(nn.Linear(cfg.d_model,h),nn.ReLU(inplace=True),nn.Dropout(cfg.dropout),nn.Linear(h,1))
    def forward(self,emg_center,mask=None):
        z=self.temporal(self.encoder(emg_center),mask); return {"tonic_logits":self.tonic_head(z).squeeze(-1),"phasic_logits":self.phasic_head(z).squeeze(-1)}
    def n_params(self): return sum(p.numel() for p in self.parameters() if p.requires_grad)
