import numpy as np
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, f1_score

def staging_metrics(targets,preds):
    return {"f1_macro":f1_score(targets,preds,average="macro",zero_division=0),"kappa":cohen_kappa_score(targets,preds),"balanced_accuracy":balanced_accuracy_score(targets,preds)}

def rswa_metrics(tonic_targets,tonic_preds,phasic_targets,phasic_preds):
    tf1=f1_score(tonic_targets,tonic_preds,zero_division=0); pf1=f1_score(phasic_targets,phasic_preds,zero_division=0)
    tk=cohen_kappa_score(tonic_targets,tonic_preds); pk=cohen_kappa_score(phasic_targets,phasic_preds)
    return {"tonic_f1":tf1,"phasic_f1":pf1,"rswa_f1_macro":float(np.mean([tf1,pf1])),"tonic_kappa":tk,"phasic_kappa":pk,"rswa_kappa_macro":float(np.nanmean([tk,pk]))}
