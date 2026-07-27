"""
Treino final do detector de movimento nos 4 exames juntos.

Para validacao interna e early stopping, separa temporalmente 15% do fim de
cada exame como conjunto de validacao (nao ha 5o sujeito). Salva:
  outputs/movement_cnn_final.pt   {state_dict, cfg, window_epochs, threshold, meta}

O limiar de operacao e lido de outputs/operating_point.json se existir
(gerado pelo passo de avaliacao); caso contrario usa 0.5.
Uso: python classifier/train_final.py [--device auto|cpu|cuda|cuda:N]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from classifier.movement_clf.dataio import load_dir, Exam
from classifier.movement_clf.dataset import build_tensors, subsample_negatives
from classifier.movement_clf.engine import TrainConfig, train_one, resolve_device

DATA = HERE / "data"
OUT = HERE / "outputs"


def split_tail(ex: Exam, frac: float = 0.15):
    """Divide um exame em (treino_inicio, val_fim) por tempo."""
    T = ex.n_epochs
    cut = int(T * (1 - frac))
    tr = Exam(ex.subject_id, ex.emg[:cut], ex.stages[:cut], ex.movement[:cut], ex.path)
    va = Exam(ex.subject_id, ex.emg[cut:], ex.stages[cut:], ex.movement[cut:], ex.path)
    return tr, va


def main():
    import functools
    global print
    print = functools.partial(print, flush=True)

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="auto", help="auto (default), cpu, cuda ou cuda:N")
    args = ap.parse_args()
    device = resolve_device(args.device)

    cfg = TrainConfig()
    exams = load_dir(DATA, require_labels=True)
    tr_exams, va_exams = zip(*[split_tail(e) for e in exams])

    from torch.utils.data import TensorDataset
    Xtr, ytr = build_tensors(list(tr_exams), cfg.window_epochs)
    Xtr, ytr = subsample_negatives(Xtr, ytr, neg_per_pos=cfg.neg_per_pos)
    Xva, yva = build_tensors(list(va_exams), cfg.window_epochs)
    tl = DataLoader(TensorDataset(Xtr, ytr), batch_size=cfg.batch_size, shuffle=True)
    vl = DataLoader(TensorDataset(Xva, yva), batch_size=cfg.batch_size, shuffle=False)

    print(f"treino final: {len(Xtr)} janelas / val {len(Xva)} janelas | device={device}")
    model, hist, best_state, best_auc = train_one(tl, vl, cfg, device=device, verbose=True)
    print(f"val PR-AUC final: {best_auc:.4f}")

    op = OUT / "operating_point.json"
    threshold = json.load(open(op))["threshold"] if op.exists() else 0.5

    # state_dict sempre em CPU no checkpoint -> portavel entre maquinas com/sem GPU
    # (evita que torch.load numa maquina sem CUDA falhe ao tentar mapear tensores p/ GPU).
    ckpt = {
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "window_epochs": cfg.window_epochs,
        "cfg": cfg.__dict__,
        "threshold": float(threshold),
        "meta": {
            "subjects": [e.subject_id for e in exams],
            "val_pr_auc": float(best_auc),
            "emg_channel_index": 4,
            "trained_on_device": device,
            "note": "movimento binario, noite toda; entrada EMG z-scored por exame",
        },
    }
    torch.save(ckpt, OUT / "movement_cnn_final.pt")
    json.dump(hist, open(OUT / "final_history.json", "w"), indent=2)
    print(f"salvo -> {OUT/'movement_cnn_final.pt'} (threshold={threshold})")


if __name__ == "__main__":
    main()
