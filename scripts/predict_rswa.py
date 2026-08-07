"""
Inferência do RSWADetectionNet (3 cabeças): um novo .pt -> CSV de eventos.

Lê um exame .pt no mesmo formato usado no treino (signals[T,C,300] com
canal de EMG do mento, ou emg_signals/emg separado -- ver
sleep_rswa.data.SubjectData / load_subject_file), roda o modelo nas
mini-épocas válidas (REM + confiança, por padrão -- mesma máscara do
treino; use --all-stages para rodar em todas as fases), aplica o limiar
de operação de cada cabeça (tônico/fásico/any, independentes -- uma
mini-época pode ser positiva em mais de uma cabeça), funde mini-épocas
positivas adjacentes DA MESMA CABEÇA em eventos, e escreve um CSV único
com colunas:

    subject_id, onset_s, duration_s, type, score

`type` é 'tonic', 'phasic' ou 'any' (a cabeça que gerou o evento). `score`
é a probabilidade média das mini-épocas do evento. Mesmo formato dos
*_revisado.csv / *_amostra_revisao.csv existentes -- pode ser usado como
pré-anotação para revisão humana.

`onset_s`/`duration_s` estão no referencial do .pt (mini-época 0 = início
do tensor), mesma convenção de classifier/movement_clf/dataio.py::
events_from_binary.

Uso:
    python scripts/predict_rswa.py EXAME.pt -o SAIDA.csv \\
        --checkpoint runs/rswa/.../fold_0/checkpoints/best.pt

    # Ensemble de múltiplos folds (média de probabilidade, mesma lógica de
    # scripts/evaluate_per_subject.py):
    python scripts/predict_rswa.py EXAME.pt -o SAIDA.csv \\
        --checkpoint runs/rswa/.../fold_0/checkpoints/best.pt \\
        --checkpoint runs/rswa/.../fold_1/checkpoints/best.pt \\
        --tonic-threshold 0.4

    # Vários exames de um diretório, um CSV por exame em --out-dir:
    python scripts/predict_rswa.py --data-dir classifier/data --out-dir classifier/predictions \\
        --checkpoint runs/rswa/.../fold_0/checkpoints/best.pt
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from sleep_rswa import ModelConfig, RSWADetectionNet, SleepAnalysisDataset, collate_sleep_analysis_exams  # noqa: E402
from sleep_rswa.data import load_subject_file, load_subject_directory  # noqa: E402
from sleep_rswa.training import load_checkpoint, resolve_device  # noqa: E402
from classifier.movement_clf.dataio import events_from_binary, EPOCH_SEC  # noqa: E402

_HEADS = ("tonic", "phasic", "any")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pt", nargs="?", default=None, help="Arquivo .pt de um único exame.")
    ap.add_argument("--data-dir", type=Path, default=None,
                     help="Alternativa a `pt`: diretório com vários .pt, gera um CSV por exame.")
    ap.add_argument("-o", "--out", default=None, help="CSV de saída (modo arquivo único; default: <exame>_rswa_pred.csv).")
    ap.add_argument("--out-dir", type=Path, default=None, help="Diretório de saída (modo --data-dir).")
    ap.add_argument("--checkpoint", action="append", required=True, dest="checkpoints",
                     help="Caminho para um checkpoint (best.pt). Repita para ensemble por média de probabilidade.")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--tonic-threshold", type=float, default=None)
    ap.add_argument("--phasic-threshold", type=float, default=None)
    ap.add_argument("--any-threshold", type=float, default=None)
    ap.add_argument("--min-epochs", type=int, default=1, help="Duração mínima do evento em mini-épocas (default: 1 = 3s).")
    ap.add_argument("--min-confidence", type=float, default=0.0)
    ap.add_argument("--all-stages", action="store_true", help="Roda em todas as fases (não só REM).")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--d-model", type=int, default=None,
                     help="Deve casar com o D_MODEL usado no treino do checkpoint (ver runs/.../run.json). "
                          "Se omitido, usa o default de ModelConfig (256).")
    args = ap.parse_args()
    if args.pt is None and args.data_dir is None:
        ap.error("informe um .pt posicional OU --data-dir.")
    return args


def _resolve_thresholds(args) -> dict[str, float]:
    return {
        "tonic": args.tonic_threshold if args.tonic_threshold is not None else args.threshold,
        "phasic": args.phasic_threshold if args.phasic_threshold is not None else args.threshold,
        "any": args.any_threshold if args.any_threshold is not None else args.threshold,
    }


def score_subject(models, subject, device, amp, min_confidence, rem_mask_only):
    """Roda o(s) modelo(s) no sujeito inteiro. Retorna dict {head: (idx, prob)}
    onde ``idx`` são os índices de mini-época válidos (no referencial do
    tensor completo) e ``prob`` a probabilidade média do ensemble nesses
    índices. Mini-épocas fora de ``idx`` não são preditas (não entram na
    máscara de validade do treino) e ficam fora dos eventos gerados.
    """
    ds = SleepAnalysisDataset([subject], min_confidence=min_confidence, rem_mask_only=rem_mask_only)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_sleep_analysis_exams)
    batch = next(iter(loader))
    emg = batch["emg_center"].to(device)
    padding_mask = batch["padding_mask"].to(device)
    valid_mask = (batch["rswa_valid"].to(device) & padding_mask)[0]
    idx = torch.nonzero(valid_mask, as_tuple=False).flatten().detach().cpu().numpy()

    probs_sum = {h: None for h in _HEADS}
    for model in models:
        model.eval()
        with torch.no_grad():
            ctx = torch.autocast(device_type=device.type, enabled=amp) if device.type == "cuda" else _nullctx()
            with ctx:
                outputs = model(emg, mask=padding_mask)
        for h in _HEADS:
            p = torch.sigmoid(outputs[f"{h}_logits"].float()).detach().cpu().numpy()[0]
            probs_sum[h] = p if probs_sum[h] is None else probs_sum[h] + p

    n_models = len(models)
    return {h: (idx, probs_sum[h][idx] / n_models) for h in _HEADS}, subject.n_epochs


class _nullctx:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


def events_for_subject(models, subject, thresholds, device, amp, min_confidence, rem_mask_only, min_epochs):
    """Gera a lista de eventos (dict subject_id/onset_s/duration_s/type/score)
    para as 3 cabeças de um sujeito, fundindo mini-épocas positivas
    adjacentes DA MESMA CABEÇA via ``events_from_binary`` (mesmo algoritmo
    usado em classifier/predict_movements.py e classifier/sample_for_review.py).
    """
    per_head, n_epochs = score_subject(models, subject, device, amp, min_confidence, rem_mask_only)
    min_dur = min_epochs * EPOCH_SEC
    all_events = []
    for h in _HEADS:
        idx, prob = per_head[h]
        mask_full = np.zeros(n_epochs, dtype=bool)
        prob_full = np.zeros(n_epochs, dtype=np.float32)
        mask_full[idx] = prob >= thresholds[h]
        prob_full[idx] = prob
        events = events_from_binary(mask_full, scores=prob_full, subject_id=subject.subject_id, etype=h)
        events = [e for e in events if e["duration_s"] >= min_dur - 1e-6]
        all_events.extend(events)
    all_events.sort(key=lambda e: e["onset_s"])
    return all_events, len(per_head["tonic"][0])


def write_csv(events, out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["subject_id", "onset_s", "duration_s", "type", "score"])
        w.writeheader()
        w.writerows(events)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    thresholds = _resolve_thresholds(args)
    rem_mask_only = not args.all_stages

    model_cfg = ModelConfig(d_model=args.d_model) if args.d_model is not None else ModelConfig()
    print(f"Carregando {len(args.checkpoints)} checkpoint(s) em {device} (d_model={model_cfg.d_model})...")
    models = []
    for ckpt_path in args.checkpoints:
        model = RSWADetectionNet(config=model_cfg).to(device)
        try:
            load_checkpoint(ckpt_path, model, device)
        except RuntimeError as e:
            raise RuntimeError(
                f"Falha ao carregar {ckpt_path}: mismatch de arquitetura. Se este checkpoint foi treinado "
                f"com D_MODEL diferente do default (256), passe --d-model <valor> (confira em "
                f"runs/.../run.json ou no log de treino). Erro original: {e}"
            ) from e
        models.append(model)

    if args.pt is not None:
        pt_path = Path(args.pt)
        subject = load_subject_file(pt_path)
        events, n_valid = events_for_subject(models, subject, thresholds, device, not args.no_amp,
                                              args.min_confidence, rem_mask_only, args.min_epochs)
        out_csv = Path(args.out) if args.out else pt_path.with_name(pt_path.stem + "_rswa_pred.csv")
        write_csv(events, out_csv)
        by_type = {}
        for e in events:
            by_type[e["type"]] = by_type.get(e["type"], 0) + 1
        print(f"{subject.subject_id}: {n_valid} mini-épocas avaliadas -> {len(events)} eventos {by_type} -> {out_csv}")
    else:
        subjects = load_subject_directory(args.data_dir)
        out_dir = args.out_dir or (args.data_dir.parent / "predictions")
        out_dir.mkdir(parents=True, exist_ok=True)
        total = 0
        for subject in subjects:
            events, n_valid = events_for_subject(models, subject, thresholds, device, not args.no_amp,
                                                  args.min_confidence, rem_mask_only, args.min_epochs)
            out_csv = out_dir / f"{subject.subject_id}_rswa_pred.csv"
            write_csv(events, out_csv)
            by_type = {}
            for e in events:
                by_type[e["type"]] = by_type.get(e["type"], 0) + 1
            print(f"  {subject.subject_id:15s}: {n_valid:6d} mini-épocas -> {len(events):4d} eventos {by_type} -> {out_csv.name}")
            total += len(events)
        print(f"\nTotal: {total} eventos em {len(subjects)} exame(s), em {out_dir}")


if __name__ == "__main__":
    main()
