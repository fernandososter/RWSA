import torch
import torch.nn as nn
class StagingLoss(nn.Module):
    def __init__(self,class_weights=None): super().__init__(); self.ce=nn.CrossEntropyLoss(weight=class_weights,reduction="none",ignore_index=-1)
    def forward(self,logits,targets,mask):
        loss=self.ce(logits.reshape(-1,5),targets.reshape(-1)).reshape_as(targets); return loss[mask].mean()
class RSWALoss(nn.Module):
    """BCE unica sobre a cabeca 'movement' (any = tonico OU fasico)."""
    def __init__(self,movement_pos_weight=None):
        super().__init__(); self.movement=nn.BCEWithLogitsLoss(pos_weight=movement_pos_weight,reduction="none")
    def forward(self,outputs,movement_targets,mask):
        ml=self.movement(outputs["movement_logits"],movement_targets); return ml[mask].mean()
