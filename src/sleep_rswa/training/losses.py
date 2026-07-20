import torch
import torch.nn as nn
class StagingLoss(nn.Module):
    def __init__(self,class_weights=None): super().__init__(); self.ce=nn.CrossEntropyLoss(weight=class_weights,reduction="none",ignore_index=-1)
    def forward(self,logits,targets,mask):
        loss=self.ce(logits.reshape(-1,5),targets.reshape(-1)).reshape_as(targets); return loss[mask].mean()
class RSWALoss(nn.Module):
    def __init__(self,tonic_pos_weight=None,phasic_pos_weight=None):
        super().__init__(); self.tonic=nn.BCEWithLogitsLoss(pos_weight=tonic_pos_weight,reduction="none"); self.phasic=nn.BCEWithLogitsLoss(pos_weight=phasic_pos_weight,reduction="none")
    def forward(self,outputs,tonic_targets,phasic_targets,mask):
        tl=self.tonic(outputs["tonic_logits"],tonic_targets); pl=self.phasic(outputs["phasic_logits"],phasic_targets); return (tl[mask].mean()+pl[mask].mean())/2
