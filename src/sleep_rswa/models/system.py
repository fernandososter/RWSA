import torch.nn as nn
from .staging import SleepStagingNet
from .rswa import RSWADetectionNet
class SleepStagingRSWASystem(nn.Module):
    def __init__(self,staging_model=None,rswa_model=None):
        super().__init__(); self.staging_model=staging_model or SleepStagingNet(); self.rswa_model=rswa_model or RSWADetectionNet()
    def forward(self,signals,emg_center,mask=None):
        return {"staging_logits":self.staging_model(signals,mask),**self.rswa_model(emg_center,mask)}
    def n_params(self): return sum(p.numel() for p in self.parameters() if p.requires_grad)
