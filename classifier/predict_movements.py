"""
Inferencia do detector de movimento: um novo .pt -> CSV de anotacoes.

Le um exame .pt (mesmo formato do preprocessamento: signals[T,5,300],
canal 4 = EMG do mento), roda a CNN, aplica o limiar de operacao, funde
mini-epocas positivas adjacentes em eventos e escreve um CSV com colunas:

    subject_id, onset_s, duration_s, type, score

`type` e sempre 'movement'. `score` e o score medio das mini-epocas do evento.
Este CSV segue o mesmo formato dos seus arquivos *_rswa.csv e pode ser usado
como pre-anotacao para revisao humana / rodar junto com o estagiamento.

Por padrao `onset_s` esta no referencial do .pt (mini-epoca 0 = inicio do
tensor). Para gravar em tempo do EDF (segundos desde o inicio da gravacao),
informe o horario de inicio do EDF (--meas-date) e o horario de inicio do
hipnograma/estagiamento (--hipno-start) — mesma convencao de offset usada em
view/app.py (annot_start = hipno_start - meas_date, com correcao de meia-
noite). Se ja souber o offset em segundos, use --annot-start diretamente
(tem prioridade sobre o par --meas-date/--hipno-start). Sem nenhum desses,
`onset_s` sai em tempo do .pt, como sempre.

Opcionalmente (--tonic-phasic, ligado por padrao), roda tambem a regra
deterministica de duracao+amplitude (classifier/movement_clf/tonic_phasic.py)
restrita aos trechos ja marcados como movimento pela CNN, e grava um
SEGUNDO csv (<out>_tonico_fasico.csv) com colunas:

    subject_id, onset_s, duration_s, type, peak_ratio, needs_review

`type` e 'phasic' (rotulo final) ou 'tonic_candidate' (NUNCA final --
needs_review=True sempre). Ver RELATORIO.md para a justificativa: a regra
nao consegue, matematicamente, separar tonus tonico real de artefato de
movimento sustentado usando so duracao+amplitude do EMG; por isso todo
candidato a tonico exige confirmacao visual antes de virar rotulo.

Uso:
    python classifier/predict_movements.py EXAME.pt [-o SAIDA.csv]
                 [--model CKPT.pt] [--threshold 0.5] [--min-epochs 1]
                 [--device auto|cpu|cuda|cuda:N]
                 [--meas-date HH:MM:SS --hipno-start HH:MM:SS]
                 [--annot-start SEGUNDOS]
                 [--no-tonic-phasic]

Este modulo NAO importa nada de src/sleep_rswa nem de view/ — o par
meas-date/hipno-start e resolvido localmente, sem ler view/exam_config.json
nem view/mat/.
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
from classifier.movement_clf.engine import resolve_device
from classifier.movement_clf.tonic_phasic import classify_tonic_phasic, restrict_to_movement

DEFAULT_MODEL = HERE / "outputs" / "movement_cnn_final.pt"


def predict_tonic_phasic(exam, movement_mask, annot_start: float | None = None):
    """Roda a regra deterministica de fasico/candidato-tonico sobre o EMG bruto,
    restrita aos trechos que a CNN ja marcou como movimento.

    exam: Exam (de load_exam), usa exam.emg [T,300] -> flatten para EMG continuo.
    movement_mask: [T] bool, mascara de movimento por mini-epoca (mask do CNN).
    Devolve lista de eventos com onset_s/duration_s/type/peak_ratio/needs_review.
    """
    flat = exam.emg.reshape(-1).astype(float)  # [T*300], fs=100Hz, mesmo referencial do exam
    events = classify_tonic_phasic(flat)
    events = restrict_to_movement(events, movement_mask, epoch_sec=EPOCH_SEC)
    for e in events:
        e["subject_id"] = exam.subject_id
        e["needs_review"] = (e["type"] == "tonic_candidate")
    if annot_start is not None:
        for e in events:
            e["onset_s"] = round(e["onset_s"] + annot_start, 3)
    return events


def _hms_to_sec(s):
    """'HH:MM:SS' (ou 'HH:MM') -> segundos desde a meia-noite. None se invalido.

    Mesma logica de view/app.py::_hms_to_sec, reimplementada aqui para o
    modulo classifier/ ficar autocontido (nao importa nada de view/).
    """
    if s is None:
        return None
    parts = str(s).strip().replace(",", ":").split(":")
    try:
        parts = [float(p) for p in parts if p != ""]
    except ValueError:
        return None
    if not parts:
        return None
    h = parts[0]
    m = parts[1] if len(parts) > 1 else 0.0
    sec = parts[2] if len(parts) > 2 else 0.0
    return h * 3600.0 + m * 60.0 + sec


def resolve_annot_start(meas_date=None, hipno_start=None, annot_start=None):
    """Resolve o offset onset_pt -> onset_edf, em segundos.

    Prioridade: --annot-start explicito > --meas-date + --hipno-start > None
    (sem offset -> onset_s sai em tempo do .pt).
    """
    if annot_start is not None:
        return float(annot_start)
    meas_sec = _hms_to_sec(meas_date)
    hipno_sec = _hms_to_sec(hipno_start)
    if meas_sec is None or hipno_sec is None:
        return None
    off = hipno_sec - meas_sec
    if off < 0:
        off += 86400.0  # gravacao cruza a meia-noite
    return round(off, 3)


def load_model(ckpt_path: Path, device: str = "cpu"):
    # o checkpoint guarda state_dict em CPU (train_final.py garante isso), entao
    # map_location="cpu" aqui e sempre seguro mesmo se quem treinou usou CUDA;
    # so depois movemos o modelo para o device pedido.
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    window = ckpt.get("window_epochs", 5)
    model = MovementCNN(window_epochs=window)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model, ckpt


@torch.no_grad()
def score_exam(model, exam, window_epochs: int, batch_size: int = 512, device: str = "cpu") -> np.ndarray:
    """Score por mini-epoca (probabilidade de movimento) para o exame inteiro."""
    from torch.utils.data import DataLoader, TensorDataset
    X, y = build_tensors([exam], window_epochs=window_epochs)
    loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=False)
    scores = []
    for x, _ in loader:
        x = x.to(device)
        scores.append(torch.sigmoid(model(x)).cpu().numpy())
    return np.concatenate(scores)


def predict_to_csv(pt_path, out_csv, model_path=DEFAULT_MODEL, threshold=None,
                   min_epochs: int = 1, verbose: bool = True, device: str = "cpu",
                   annot_start: float | None = None, tonic_phasic: bool = True,
                   out_csv_tonic_phasic=None):
    """Roda a inferencia e grava o CSV de eventos.

    annot_start: offset em segundos (onset_edf = onset_s_pt + annot_start).
    Se informado, `onset_s` no CSV sai em tempo do EDF; senao, sai em tempo
    do .pt (mini-epoca 0 = inicio do tensor), como antes. Use
    resolve_annot_start(...) para obter esse valor a partir de
    --meas-date/--hipno-start ou passar --annot-start diretamente.

    tonic_phasic: se True (default), roda tambem a regra deterministica de
    duracao+amplitude (movement_clf.tonic_phasic) restrita aos trechos de
    movimento, e grava um segundo CSV em out_csv_tonic_phasic (default:
    <out_csv>_tonico_fasico.csv). 'phasic' e rotulo final; 'tonic_candidate'
    NUNCA e final e sempre tem needs_review=True (ver RELATORIO.md).
    """
    model, ckpt = load_model(Path(model_path), device=device)
    window = ckpt.get("window_epochs", 5)
    if threshold is None:
        threshold = ckpt.get("threshold", 0.5)

    # exame novo pode nao ter rotulos -> require_labels=False
    exam = load_exam(pt_path, require_labels=False)
    scores = score_exam(model, exam, window, device=device)
    mask = scores >= threshold

    events = events_from_binary(mask, scores=scores, subject_id=exam.subject_id,
                                etype="movement")
    # filtra eventos curtos demais (min_epochs mini-epocas)
    min_dur = min_epochs * EPOCH_SEC
    events = [e for e in events if e["duration_s"] >= min_dur - 1e-6]

    if annot_start is not None:
        for e in events:
            e["onset_s"] = round(e["onset_s"] + annot_start, 3)

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["subject_id", "onset_s", "duration_s", "type", "score"])
        w.writeheader()
        for e in events:
            w.writerow(e)

    tp_events = None
    if tonic_phasic:
        tp_events = predict_tonic_phasic(exam, mask, annot_start=annot_start)
        out_tp = Path(out_csv_tonic_phasic) if out_csv_tonic_phasic else \
            out_csv.with_name(out_csv.stem + "_tonico_fasico.csv")
        out_tp.parent.mkdir(parents=True, exist_ok=True)
        with open(out_tp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["subject_id", "onset_s", "duration_s",
                                              "type", "peak_ratio", "needs_review"])
            w.writeheader()
            for e in tp_events:
                w.writerow(e)

    if verbose:
        time_ref = "tempo EDF" if annot_start is not None else "tempo .pt"
        print(f"{exam.subject_id}: {exam.n_epochs} mini-epocas ({exam.hours:.1f}h), "
              f"limiar={threshold:.3f} -> {int(mask.sum())} mini-epocas positivas, "
              f"{len(events)} eventos ({time_ref}) -> {out_csv}")
        if tonic_phasic:
            n_phasic = sum(1 for e in tp_events if e["type"] == "phasic")
            n_cand = sum(1 for e in tp_events if e["type"] == "tonic_candidate")
            print(f"  tonico/fasico: {n_phasic} fasico (final), "
                  f"{n_cand} candidato-tonico (precisa revisao) -> {out_tp}")
    return events, scores, tp_events


def main():
    ap = argparse.ArgumentParser(description="Detecta movimento num .pt e gera CSV de anotacoes.")
    ap.add_argument("pt", help="arquivo .pt do exame")
    ap.add_argument("-o", "--out", default=None, help="CSV de saida (default: <exame>_movimentos.csv)")
    ap.add_argument("--model", default=str(DEFAULT_MODEL), help="checkpoint do modelo")
    ap.add_argument("--threshold", type=float, default=None, help="limiar (default: do checkpoint)")
    ap.add_argument("--min-epochs", type=int, default=1, help="duracao minima do evento em mini-epocas")
    ap.add_argument("--device", default="auto", help="auto (default), cpu, cuda ou cuda:N")
    ap.add_argument("--meas-date", default=None,
                     help="horario de inicio do EDF, 'HH:MM:SS' (junto com --hipno-start, "
                          "grava onset_s em tempo do EDF)")
    ap.add_argument("--hipno-start", default=None,
                     help="horario de inicio do hipnograma/estagiamento, 'HH:MM:SS'")
    ap.add_argument("--annot-start", type=float, default=None,
                     help="offset onset_pt->onset_edf em segundos, se ja souber (tem "
                          "prioridade sobre --meas-date/--hipno-start)")
    ap.add_argument("--no-tonic-phasic", action="store_true",
                     help="nao gerar o CSV de sub-classificacao fasico/candidato-tonico")
    args = ap.parse_args()
    device = resolve_device(args.device)

    annot_start = resolve_annot_start(meas_date=args.meas_date, hipno_start=args.hipno_start,
                                       annot_start=args.annot_start)
    if (args.meas_date or args.hipno_start) and annot_start is None:
        print("[aviso] --meas-date/--hipno-start informado(s) mas incompleto ou invalido "
              "-> onset_s sairah em tempo do .pt.", file=sys.stderr)

    pt = Path(args.pt)
    out = args.out or str(pt.with_name(pt.stem + "_movimentos.csv"))
    predict_to_csv(pt, out, model_path=args.model, threshold=args.threshold,
                   min_epochs=args.min_epochs, device=device, annot_start=annot_start,
                   tonic_phasic=not args.no_tonic_phasic)


if __name__ == "__main__":
    main()
