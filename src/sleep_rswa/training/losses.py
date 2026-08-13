import torch
import torch.nn as nn
class StagingLoss(nn.Module):
    def __init__(self,class_weights=None): super().__init__(); self.ce=nn.CrossEntropyLoss(weight=class_weights,reduction="none",ignore_index=-1)
    def forward(self,logits,targets,mask):
        loss=self.ce(logits.reshape(-1,5),targets.reshape(-1)).reshape_as(targets); return loss[mask].mean()
class RSWALoss(nn.Module):
    """BCE independente por cabeca: tonic_head, phasic_head, any_head.

    Cada cabeca tem seu proprio pos_weight (a prevalencia de tonico, fasico
    e "any" e bem diferente) e sua propria mascara de validade -- mas hoje
    as tres compartilham a mesma mascara `rswa_valid` (validade = mini-epoca
    escorada, nao confianca do rotulo). Retorna a soma das tres losses
    (cada uma media so sobre as posicoes validas) e cada termo individual,
    para logging/monitoramento por cabeca.
    """
    def __init__(self,tonic_pos_weight=None,phasic_pos_weight=None,any_pos_weight=None):
        super().__init__()
        self.tonic=nn.BCEWithLogitsLoss(pos_weight=tonic_pos_weight,reduction="none")
        self.phasic=nn.BCEWithLogitsLoss(pos_weight=phasic_pos_weight,reduction="none")
        self.any=nn.BCEWithLogitsLoss(pos_weight=any_pos_weight,reduction="none")
    def forward(self,outputs,tonic_targets,phasic_targets,any_targets,mask):
        tl=self.tonic(outputs["tonic_logits"],tonic_targets)[mask].mean()
        pl=self.phasic(outputs["phasic_logits"],phasic_targets)[mask].mean()
        al=self.any(outputs["any_logits"],any_targets)[mask].mean()
        total=tl+pl+al
        return total,{"tonic_loss":tl.detach(),"phasic_loss":pl.detach(),"any_loss":al.detach()}
