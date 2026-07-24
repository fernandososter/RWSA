"""
Validacao leave-one-subject-out do detector de movimento.

Treina em 3 exames, testa no 4o, rotaciona. Salva:
  outputs/loso_predictions.npz   scores out-of-fold por sujeito (y, score, stages, hours)
  outputs/loso_history.json      curvas de treino por fold
Uso: python classifier/train_loso.py
"""
from __future__ import annotations

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
from classifier.movement_clf.engine import TrainConfig, train_one, predict_scores

DATA = HERE / "data"
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)


def main():
    import functools
    global print
    print = functools.partial(print, flush=True)
    cfg = TrainConfig()
    exams = load_dir(DATA, require_labels=True)
    subjects = [e.subject_id for e in exams]
    print(f"exames: {subjects} | params/modelo config window={cfg.window_epochs}")

    preds = {}
    history = {}
    t0 = time.time()
    for i, test_ex in enumerate(exams):
        train_ex = [e for j, e in enumerate(exams) if j != i]
        sid = test_ex.subject_id
        print(f"\n=== FOLD {i+1}/4 — teste={sid} treino={[e.subject_id for e in train_ex]} ===")
        tr, va, tl, vl = make_loaders(train_ex, [test_ex],
                                      window_epochs=cfg.window_epochs,
                                      batch_size=cfg.batch_size)
        model, hist, _, best_auc = train_one(tl, vl, cfg, device="cpu", verbose=True)
        y, score = predict_scores(model, vl, device="cpu")
        preds[sid] = dict(y=y.astype(np.float32), score=score.astype(np.float32),
                          stages=test_ex.stages.astype(np.int64),
                          hours=float(test_ex.hours))
        history[sid] = dict(history=hist, best_val_pr_auc=best_auc)
        print(f"  fold {sid}: PR-AUC={best_auc:.4f}")

    # salva npz (arrays por sujeito com prefixo)
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
