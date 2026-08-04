"""
Avaliacao: operador de energia de Teager-Kaiser (TKEO) vs. envelope RMS,
ambos com o MESMO pos-processamento (limiar duplo/histerese + fusao de
lacunas + classificacao por duracao), testados contra os 10 exames
sinteticos com ground truth exato em testes/data/.

Isolado: usa apenas tkeo_rule.py (deste mesmo diretorio) + torch/numpy/
pandas. Nao importa nada de classifier/, src/sleep_rswa/ nem testes/src/limiar/.

Metrica: identica ao teste 1 -- correspondencia por evento via sobreposicao
temporal, TP/FP/FN por tipo (fasico/tonico), precisao/recall/F1.

Metrica adicional (especifica deste teste): razao de separacao
sinal-ruido -- mediana do sinal de energia (TKEO ou RMS) DENTRO dos eventos
verdadeiros dividida pela mediana do sinal FORA de qualquer evento
verdadeiro. Quantifica o quanto cada pre-processamento separa evento de
repouso, independente do limiar escolhido.

Saidas (em testes/src/tkeo/results/):
  per_subject_metrics.csv
  summary_metrics.csv
  separation_ratio.csv
  tkeo_vs_rms_comparison.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tkeo_rule import (
    detect_events_tkeo, detect_events_rms, tkeo_energy_signal, rms_envelope, FS,
)

DATA_DIR = Path(__file__).resolve().parents[2] / "data"    # testes/data
RESULTS_DIR = Path(__file__).resolve().parent / "results"  # testes/src/tkeo/results
EMG_CHANNEL_INDEX = 4
METHODS = ["rms", "tkeo"]
EVENT_TYPES = ["phasic", "tonic"]


def load_synth_exam(pt_path: Path):
    obj = torch.load(pt_path, map_location="cpu", weights_only=False)
    emg = obj["signals"][:, EMG_CHANNEL_INDEX, :].numpy().astype(np.float64).reshape(-1)
    return emg


def load_ground_truth(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["end_s"] = df["onset_s"] + df["duration_s"]
    return df


def match_events(gt_df: pd.DataFrame, detected: list, event_type: str) -> dict:
    gt_sub = gt_df[gt_df["type"] == event_type]
    det_sub = [d for d in detected if d.type == event_type]

    gt_matched = np.zeros(len(gt_sub), dtype=bool)
    det_matched = np.zeros(len(det_sub), dtype=bool)
    gt_starts = gt_sub["onset_s"].to_numpy()
    gt_ends = gt_sub["end_s"].to_numpy()

    for j, d in enumerate(det_sub):
        d_start, d_end = d.onset_s, d.onset_s + d.duration_s
        overlaps = (gt_starts < d_end) & (gt_ends > d_start)
        idxs = np.where(overlaps)[0]
        if len(idxs) > 0:
            det_matched[j] = True
            gt_matched[idxs] = True

    tp = int(gt_matched.sum())
    fn = int((~gt_matched).sum())
    fp = int((~det_matched).sum())
    return {"tp": tp, "fp": fp, "fn": fn}


def prf1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 and not (np.isnan(precision) or np.isnan(recall)) else np.nan)
    return precision, recall, f1


def separation_ratio(energy: np.ndarray, gt_df: pd.DataFrame, fs: int = FS) -> float:
    """Mediana(energia dentro de eventos verdadeiros) / Mediana(energia fora
    de qualquer evento verdadeiro) -- quanto maior, melhor a separacao
    intrinseca do sinal de energia, antes de qualquer escolha de limiar."""
    n = len(energy)
    in_event = np.zeros(n, dtype=bool)
    for _, row in gt_df.iterrows():
        s = int(round(row["onset_s"] * fs))
        e = int(round(row["end_s"] * fs))
        in_event[max(0, s):min(n, e)] = True
    med_in = np.median(energy[in_event]) if in_event.any() else np.nan
    med_out = np.median(energy[~in_event]) if (~in_event).any() else np.nan
    return float(med_in / med_out) if med_out > 0 else np.nan


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    subject_stems = sorted(p.stem for p in DATA_DIR.glob("synth*.pt"))
    assert len(subject_stems) > 0, f"nenhum synth*.pt encontrado em {DATA_DIR}"

    rows = []
    sep_rows = []
    for stem in subject_stems:
        emg = load_synth_exam(DATA_DIR / f"{stem}.pt")
        gt_df = load_ground_truth(DATA_DIR / f"{stem}_events.csv")

        detected_by_method = {
            "rms": detect_events_rms(emg, fs=FS, apply_merge_gaps=True),
            "tkeo": detect_events_tkeo(emg, fs=FS, apply_merge_gaps=True),
        }
        for method in METHODS:
            detected = detected_by_method[method]
            for etype in EVENT_TYPES:
                m = match_events(gt_df, detected, etype)
                precision, recall, f1 = prf1(m["tp"], m["fp"], m["fn"])
                rows.append({
                    "subject": stem, "method": method, "event_type": etype,
                    "tp": m["tp"], "fp": m["fp"], "fn": m["fn"],
                    "n_gt": int((gt_df["type"] == etype).sum()),
                    "n_detected": sum(1 for d in detected if d.type == etype),
                    "precision": precision, "recall": recall, "f1": f1,
                })

        # razao de separacao sinal-ruido, independente do limiar
        rms_energy = rms_envelope(emg, win_sec=0.1, fs=FS)
        tkeo_energy = tkeo_energy_signal(emg, fs=FS)
        sep_rows.append({"subject": stem, "method": "rms", "separation_ratio": separation_ratio(rms_energy, gt_df, fs=FS)})
        sep_rows.append({"subject": stem, "method": "tkeo", "separation_ratio": separation_ratio(tkeo_energy, gt_df, fs=FS)})

    per_subject = pd.DataFrame(rows)
    per_subject.to_csv(RESULTS_DIR / "per_subject_metrics.csv", index=False)

    sep_df = pd.DataFrame(sep_rows)
    sep_df.to_csv(RESULTS_DIR / "separation_ratio.csv", index=False)

    agg = (per_subject.groupby(["method", "event_type"])
           .agg(tp=("tp", "sum"), fp=("fp", "sum"), fn=("fn", "sum"),
                n_gt=("n_gt", "sum"), n_detected=("n_detected", "sum"))
           .reset_index())
    agg["precision"], agg["recall"], agg["f1"] = zip(*agg.apply(
        lambda r: prf1(r["tp"], r["fp"], r["fn"]), axis=1))
    agg.to_csv(RESULTS_DIR / "summary_metrics.csv", index=False)

    sep_agg = sep_df.groupby("method")["separation_ratio"].agg(["median", "mean", "std"]).reset_index()
    sep_agg.to_csv(RESULTS_DIR / "separation_ratio_summary.csv", index=False)

    print(per_subject.to_string(index=False))
    print()
    print("=== RESUMO AGREGADO (todos os 10 exames sinteticos, 2h cada) ===")
    print(agg.to_string(index=False))
    print()
    print("=== RAZAO DE SEPARACAO SINAL-RUIDO (mediana dentro de eventos / mediana fora) ===")
    print(sep_agg.to_string(index=False))
    return per_subject, agg, sep_df, sep_agg


if __name__ == "__main__":
    main()
