"""
Validacao (leave-one-subject-out ou k-fold agrupado por sujeito) do detector
de movimento.

Sem --n-folds: LOSO classico — treina em N-1 exames, testa no 1 restante,
roda N vezes (um fold por sujeito). Viavel para poucos exames (ex. 4).

Com --n-folds K: agrupa os sujeitos em K folds (balanceados por prevalencia
de movimento via distribuicao round-robin/"snake"), treina em K-1 grupos,
testa no grupo restante, roda K vezes. Necessario quando o numero de exames
cresce, pois o tempo por fold escala com o tamanho do treino — LOSO puro com
dezenas de exames vira caro demais (cada fold treina em quase todos os
exames). Nenhum sujeito de teste aparece no treino do seu proprio fold, com
K sujeitos ou 1 — a garantia de "sujeito nunca visto" e identica.

Salva:
  outputs/loso_predictions.npz   scores out-of-fold por sujeito (y, score, stages, hours)
  outputs/loso_history.json      curvas de treino por fold (uma entrada por FOLD, nao por sujeito,
                                  quando --n-folds agrupa varios sujeitos por fold)
Uso:
  python classifier/train_loso.py [--device auto|cpu|cuda|cuda:N]              # LOSO puro (1 sujeito/fold)
  python classifier/train_loso.py --n-folds 6 [--device ...] [--seed 0]        # k-fold agrupado
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from classifier.movement_clf.dataio import load_dir
from classifier.movement_clf.dataset import make_loaders
from classifier.movement_clf.engine import TrainConfig, train_one, predict_scores, resolve_device

DATA = HERE / "data"
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)


def make_subject_folds(exams, n_folds: int, seed: int = 0):
    """Agrupa exames em n_folds grupos de teste, balanceados por prevalencia
    de movimento (n_positivos/n_epochs), via distribuicao "snake" (round-robin
    alternando o sentido a cada volta pelos n_folds baldes).

    Devolve lista de n_folds listas de indices em `exams` (grupo de teste de
    cada fold). Cada exame aparece em exatamente 1 grupo de teste.
    """
    if n_folds < 2:
        raise ValueError(f"--n-folds deve ser >= 2 (recebido {n_folds})")
    n = len(exams)
    if n_folds > n:
        raise ValueError(f"--n-folds={n_folds} maior que o numero de exames ({n})")

    prevalence = [float(e.movement.mean()) if e.n_epochs else 0.0 for e in exams]
    order = sorted(range(n), key=lambda i: prevalence[i], reverse=True)

    buckets = [[] for _ in range(n_folds)]
    pos, direction = 0, 1
    for idx in order:
        buckets[pos].append(idx)
        pos += direction
        if pos == n_folds:
            pos, direction = n_folds - 1, -1
        elif pos == -1:
            pos, direction = 0, 1
    return buckets


def main():
    import functools
    global print
    print = functools.partial(print, flush=True)

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="auto", help="auto (default), cpu, cuda ou cuda:N")
    ap.add_argument("--n-folds", type=int, default=None,
                     help="numero de folds agrupados por sujeito (default: LOSO puro, "
                          "1 fold por sujeito). Use p/ datasets grandes (ex. --n-folds 6).")
    ap.add_argument("--seed", type=int, default=0, help="seed do agrupamento em folds (default 0)")
    args = ap.parse_args()
    device = resolve_device(args.device)

    cfg = TrainConfig()
    exams = load_dir(DATA, require_labels=True)
    subjects = [e.subject_id for e in exams]
    n_folds = args.n_folds or len(exams)
    mode = "LOSO puro" if n_folds == len(exams) else f"k-fold agrupado (k={n_folds})"
    print(f"exames: {subjects} | modo={mode} | params/modelo config window={cfg.window_epochs} | device={device}")

    test_groups = make_subject_folds(exams, n_folds, seed=args.seed)

    preds = {}
    history = {}
    t0 = time.time()
    for fi, test_idx in enumerate(test_groups):
        test_exs = [exams[i] for i in test_idx]
        train_ex = [e for j, e in enumerate(exams) if j not in test_idx]
        test_sids = [e.subject_id for e in test_exs]
        print(f"\n=== FOLD {fi+1}/{n_folds} — teste={test_sids} "
              f"treino={[e.subject_id for e in train_ex]} ===")
        tr, va, tl, vl = make_loaders(train_ex, test_exs,
                                      window_epochs=cfg.window_epochs,
                                      batch_size=cfg.batch_size)
        model, hist, _, best_auc = train_one(tl, vl, cfg, device=device, verbose=True)

        # scores out-of-fold por sujeito de teste (val_loader concatena todos
        # os sujeitos de test_exs na mesma ordem; reparticiona por n_epochs)
        y_all, score_all = predict_scores(model, vl, device=device)
        offset = 0
        for ex in test_exs:
            n = ex.n_epochs
            y = y_all[offset:offset + n]
            score = score_all[offset:offset + n]
            offset += n
            preds[ex.subject_id] = dict(y=y.astype(np.float32), score=score.astype(np.float32),
                                        stages=ex.stages.astype(np.int64),
                                        hours=float(ex.hours))
        fold_key = "+".join(test_sids)
        history[fold_key] = dict(history=hist, best_val_pr_auc=best_auc, test_subjects=test_sids)
        print(f"  fold {fold_key}: PR-AUC={best_auc:.4f} (pooled sobre {len(test_sids)} sujeito(s) de teste)")

    # salva npz (arrays por sujeito com prefixo) — formato identico ao LOSO puro,
    # evaluate_loso.py nao precisa mudar
    flat = {}
    for sid, d in preds.items():
        flat[f"{sid}__y"] = d["y"]
        flat[f"{sid}__score"] = d["score"]
        flat[f"{sid}__stages"] = d["stages"]
        flat[f"{sid}__hours"] = np.array([d["hours"]], dtype=np.float32)
    np.savez_compressed(OUT / "loso_predictions.npz", subjects=np.array(list(preds.keys())), **flat)
    json.dump(history, open(OUT / "loso_history.json", "w"), indent=2)
    print(f"\nconcluido em {time.time()-t0:.0f}s -> {OUT/'loso_predictions.npz'}")


if __name__ == "__main__":
    main()
