import numpy as np
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, f1_score

def staging_metrics(targets,preds):
    return {"f1_macro":f1_score(targets,preds,average="macro",zero_division=0),"kappa":cohen_kappa_score(targets,preds),"balanced_accuracy":balanced_accuracy_score(targets,preds)}

def rswa_metrics(movement_targets,movement_preds):
    """F1/kappa de um unico alvo 'movement' (any = tonico OU fasico).

    Retorna 'movement_f1'/'movement_kappa' e mantem os aliases historicos
    'rswa_f1_macro'/'rswa_kappa_macro' apontando para o MESMO valor unico,
    para nao quebrar codigo de monitor/logging que ainda os referencia.
    """
    f1=float(f1_score(movement_targets,movement_preds,zero_division=0))
    k=float(cohen_kappa_score(movement_targets,movement_preds))
    return {"movement_f1":f1,"movement_kappa":k,"rswa_f1_macro":f1,"rswa_kappa_macro":k}
