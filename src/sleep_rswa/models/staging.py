import torch
import torch.nn as nn
from ..config import ModelConfig
from .common import MultiKernelCNNBranch,SEBlock,make_group_norm
from .mamba import MambaStack

class SleepStagingNet(nn.Module):
    def __init__(self,config=None,use_se=True):
        super().__init__(); cfg=config or ModelConfig(); self.cfg=cfg
        self.branches=nn.ModuleList([
          MultiKernelCNNBranch(cfg.eeg_in_channels,cfg.branch_filters,cfg.eeg_kernels,cfg.cnn_layers,cfg.dropout),
          MultiKernelCNNBranch(cfg.eog_in_channels,cfg.branch_filters,cfg.eog_kernels,cfg.cnn_layers,cfg.dropout)])
        merged=cfg.branch_filters*2; self.se_global=SEBlock(merged) if use_se else nn.Identity()
        self.branch_proj=nn.Sequential(nn.Conv1d(merged,cfg.d_model,1,bias=False),make_group_norm(cfg.d_model),nn.ReLU(inplace=True))
        self.spatial=nn.Sequential(nn.Conv1d(cfg.d_model,cfg.d_model,3,padding=1,groups=cfg.d_model,bias=False),nn.Conv1d(cfg.d_model,cfg.d_model,1,bias=False),make_group_norm(cfg.d_model),nn.ReLU(inplace=True))
        self.pool=nn.AdaptiveAvgPool1d(1); self.temporal=MambaStack(cfg.d_model,cfg.staging_mamba_layers,cfg.d_state,cfg.dropout)
        self.stage_head=nn.Sequential(nn.Linear(cfg.d_model,cfg.d_model//2),nn.ReLU(inplace=True),nn.Dropout(cfg.dropout),nn.Linear(cfg.d_model//2,5))
    def forward(self,x,mask=None):
        b,t,c,n=x.shape; z=x.reshape(b*t,c,n); eeg=z[:,:3]; eog=z[:,3:4]
        z=torch.cat([self.branches[0](eeg),self.branches[1](eog)],1); z=self.se_global(z); z=self.spatial(self.branch_proj(z)); z=self.pool(z).squeeze(-1).reshape(b,t,-1)
        return self.stage_head(self.temporal(z,mask))
    def n_params(self): return sum(p.numel() for p in self.parameters() if p.requires_grad)
