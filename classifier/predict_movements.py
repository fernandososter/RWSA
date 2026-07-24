"""
Inferencia do detector de movimento: um novo .pt -> CSV de anotacoes.

Le um exame .pt (mesmo formato do preprocessamento: signals[T,5,300],
canal 4 = EMG do mento), roda a CNN, aplica o limiar de operacao, funde
mini-epocas positivas adjacentes em eventos e escreve um CSV com colunas:

    subject_id, onset_s, duration_s, type, score

`type` e sempre 'movement'. `score` e o score medio das mini-epocas do evento.
Este CSV segue o mesmo formato dos seus arquivos *_rswa.csv e pode ser usado
como pre-anotacao para revisao humana / rodar junto com o estagiamento.

Uso:
    python classifier/predict_movements.py EXAME.pt [-o SAIDA.csv]
                 [--model CKPT.pt] [--threshold 0.5] [--min-epochs 1]

Este modulo NAO importa nada de src/sleep_rswa.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from classifier.movement_clf.dataio import load_exam, zscore_emg, events_from_binary, EPOCH_SEC
from classifier.movement_clf.dataset import build_tensors
from classifier.movement_clf.model import MovementCNN

DEFAULT_MODEL = HERE / "outputs" / "movement_cnn_final.pt"


def load_model(ckpt_path: Path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    window = ckpt.get("window_epochs", 5)
    model = MovementCNN(window_epochs=window)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


@torch.no_grad()
def score_exam(model, exam, window_epochs: int, batch_size: int = 512) -> np.ndarray:
    """Score por mini-epoca (probabilidade de movimento) para o exame inteiro."""
    from torch.utils.data import DataLoader, TensorDataset
    X, y = build_tensors([exam], window_epochs=window_epochs)
    loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=False)
    scores = []
    for x, _ in loader:
        scores.append(torch.sigmoid(model(x)).cpu().numpy())
    return np.concatenate(scores)


def predict_to_csv(pt_path, out_csv, model_path=DEFAULT_MODEL, threshold=None,
                   min_epochs: int = 1, verbose: bool = True):
    model, ckpt = load_model(Path(model_path))
    window = ckpt.get("window_epochs", 5)
    if threshold is None:
        threshold = ckpt.get("threshold", 0.5)

    # exame novo pode nao ter rotulos -> require_labels=False
    exam = load_exam(pt_path, require_labels=False)
    scores = score_exam(model, exam, window)
    mask = scores >= threshold

    events = events_from_binary(mask, scores=scores, subject_id=exam.subject_id,
                                etype="movement")
    # filtra eventos curtos demais (min_epochs mini-epocas)
    min_dur = min_epochs * EPOCH_SEC
    events = [e for e in events if e["duration_s"] >= min_dur - 1e-6]

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["subject_id", "onset_s", "duration_s", "type", "score"])
        w.writeheader()
        for e in events:
            w.writerow(e)

    if verbose:
        print(f"{exam.subject_id}: {exam.n_epochs} mini-epocas ({exam.hours:.1f}h), "
              f"limiar={threshold:.3f} -> {int(mask.sum())} mini-epocas positivas, "
              f"{len(events)} eventos -> {out_csv}")
    return events, scores


def main():
    ap = argparse.ArgumentParser(description="Detecta movimento num .pt e gera CSV de anotacoes.")
    ap.add_argument("pt", help="arquivo .pt do exame")
    ap.add_argument("-o", "--out", default=None, help="CSV de saida (default: <exame>_movimentos.csv)")
    ap.add_argument("--model", default=str(DEFAULT_MODEL), help="checkpoint do modelo")
    ap.add_argument("--threshold", type=float, default=None, help="limiar (default: do checkpoint)")
    ap.add_argument("--min-epochs", type=int, default=1, help="duracao minima do evento em mini-epocas")
    args = ap.parse_args()

    pt = Path(args.pt)
    out = args.out or str(pt.with_name(pt.stem + "_movimentos.csv"))
    predict_to_csv(pt, out, model_path=args.model, threshold=args.threshold,
                   min_epochs=args.min_epochs)


if __name__ == "__main__":
    main()
