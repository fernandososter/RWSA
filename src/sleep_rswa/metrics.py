import numpy as np
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, f1_score

def staging_metrics(targets,preds):
    return {"f1_macro":f1_score(targets,preds,average="macro",zero_division=0),"kappa":cohen_kappa_score(targets,preds),"balanced_accuracy":balanced_accuracy_score(targets,preds)}

def _binary_f1_kappa(targets,preds):
    f1=float(f1_score(targets,preds,zero_division=0))
    k=float(cohen_kappa_score(targets,preds))
    return f1,k


def rswa_metrics(tonic_targets,tonic_preds,phasic_targets,phasic_preds,any_targets,any_preds):
    """F1/kappa independentes para as 3 cabecas (tonic/phasic/any).

    Cada cabeca e um alvo binario multi-rotulo separado (nao mutuamente
    exclusivo em teoria, embora no rotulo de treino cada mini-epoca so
    deva pertencer a uma categoria). Mantem os aliases historicos
    'rswa_f1_macro'/'rswa_kappa_macro' como a MEDIA macro das 3 cabecas,
    para nao quebrar codigo de monitor/logging que ainda os referencia;
    'movement_f1'/'movement_kappa' (uniao das 3 = qualquer movimento
    anotado) tambem sao mantidos pelo mesmo motivo.
    """
    tonic_f1,tonic_k=_binary_f1_kappa(tonic_targets,tonic_preds)
    phasic_f1,phasic_k=_binary_f1_kappa(phasic_targets,phasic_preds)
    any_f1,any_k=_binary_f1_kappa(any_targets,any_preds)

    movement_targets=((np.asarray(tonic_targets)>0.5)|(np.asarray(phasic_targets)>0.5)|(np.asarray(any_targets)>0.5)).astype(np.int64)
    movement_preds=((np.asarray(tonic_preds)>0.5)|(np.asarray(phasic_preds)>0.5)|(np.asarray(any_preds)>0.5)).astype(np.int64)
    movement_f1,movement_k=_binary_f1_kappa(movement_targets,movement_preds)

    f1_macro=(tonic_f1+phasic_f1+any_f1)/3.0
    kappa_macro=(tonic_k+phasic_k+any_k)/3.0

    return {
        "tonic_f1":tonic_f1,"tonic_kappa":tonic_k,
        "phasic_f1":phasic_f1,"phasic_kappa":phasic_k,
        "any_f1":any_f1,"any_kappa":any_k,
        "movement_f1":movement_f1,"movement_kappa":movement_k,
        "rswa_f1_macro":f1_macro,"rswa_kappa_macro":kappa_macro,
    }
