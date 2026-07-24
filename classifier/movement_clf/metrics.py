"""
Metricas para o detector de movimento (modulo isolado).

- Nivel de mini-epoca: PR-AUC, F1, precision, recall a um dado limiar.
- Nivel de evento: funde mini-epocas positivas em eventos e mede recall de
  evento (fracao de eventos anotados tocados por alguma predicao positiva) e
  falsos alarmes por hora (eventos preditos sem sobreposicao com anotacao).
"""
from __future__ import annotations

import numpy as np

from .dataio import EPOCH_SEC, events_from_binary


def pr_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Average precision (area sob a curva precision-recall)."""
    from sklearn.metrics import average_precision_score
    if y_true.sum() == 0:
        return float("nan")
    return float(average_precision_score(y_true, scores))


def epoch_metrics(y_true: np.ndarray, scores: np.ndarray, thr: float) -> dict:
    from sklearn.metrics import precision_score, recall_score, f1_score
    pred = (scores >= thr).astype(int)
    yt = y_true.astype(int)
    return {
        "threshold": round(float(thr), 4),
        "precision": round(float(precision_score(yt, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(yt, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(yt, pred, zero_division=0)), 4),
        "pr_auc": round(pr_auc(yt, scores), 4),
        "n_pos": int(yt.sum()),
        "n_pred_pos": int(pred.sum()),
    }


def _intervals(mask: np.ndarray) -> list[tuple[int, int]]:
    """Lista de (inicio, fim) inclusivos de runs True."""
    mask = np.asarray(mask).astype(bool)
    out = []
    i = 0
    T = len(mask)
    while i < T:
        if mask[i]:
            j = i
            while j + 1 < T and mask[j + 1]:
                j += 1
            out.append((i, j))
            i = j + 1
        else:
            i += 1
    return out


def event_metrics(y_true: np.ndarray, scores: np.ndarray, thr: float,
                  hours: float) -> dict:
    """Recall de evento e falsos alarmes por hora.

    Um evento anotado conta como detectado se QUALQUER mini-epoca predita
    positiva o sobrepoe. Um evento predito e falso alarme se NAO sobrepoe
    nenhum evento anotado.
    """
    pred = (scores >= thr).astype(bool)
    true = y_true.astype(bool)
    true_ev = _intervals(true)
    pred_ev = _intervals(pred)

    hits = 0
    for a, b in true_ev:
        if pred[a:b + 1].any():
            hits += 1
    fa = 0
    for a, b in pred_ev:
        if not true[a:b + 1].any():
            fa += 1

    n_true = len(true_ev)
    return {
        "n_true_events": n_true,
        "n_pred_events": len(pred_ev),
        "event_recall": round(hits / n_true, 4) if n_true else float("nan"),
        "false_alarms": fa,
        "false_alarms_per_hour": round(fa / hours, 3) if hours > 0 else float("nan"),
    }


def best_f1_threshold(y_true: np.ndarray, scores: np.ndarray,
                      grid: np.ndarray | None = None) -> tuple[float, float]:
    """Varre limiares e devolve (thr, f1) do melhor F1 por mini-epoca."""
    from sklearn.metrics import f1_score
    if grid is None:
        grid = np.linspace(0.05, 0.95, 19)
    yt = y_true.astype(int)
    best_thr, best_f1 = 0.5, -1.0
    for t in grid:
        f1 = f1_score(yt, (scores >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, float(t)
    return best_thr, float(best_f1)
