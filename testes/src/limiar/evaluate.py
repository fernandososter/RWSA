"""
Avaliacao: limiar simples vs. limiar duplo (histerese) para isolar e
classificar eventos tonicos/fasicos, testado contra os 10 exames sinteticos
com ground truth exato em testes/data/ (ver testes/generate_synthetic_data.py).

Isolado: usa apenas threshold_rule.py (deste mesmo diretorio) + torch/numpy/
pandas. Nao importa nada de classifier/ ou src/sleep_rswa/.

Metrica: correspondencia por evento (nao por mini-epoca). Para cada tipo
(fasico, tonico), um evento verdadeiro conta como TP se ao menos um evento
detectado DO MESMO TIPO se sobrepoe a ele no tempo; caso contrario e FN. Um
evento detectado do tipo T conta como FP se nao se sobrepoe a nenhum evento
verdadeiro do tipo T. Overlap = qualquer intersecao temporal > 0 -- eventos
tonicos/fasicos verdadeiros nunca se sobrepoem entre si (by construction do
gerador), entao esta regra simples e nao-ambigua.

Saidas (em testes/src/limiar/results/):
  per_subject_metrics.csv       -- TP/FP/FN/precision/recall/f1 por (exame, metodo, tipo)
  summary_metrics.csv           -- agregado (todos os exames somados) por (metodo, tipo)
  threshold_comparison.png      -- figura comparando precisao/recall/F1, simples vs duplo, fasico vs tonico
  
  
  
FUNCIONAMENTO: 

Etapa 0 — pré-processamento (igual para os dois métodos)

Envelope RMS: o EMG bruto é elevado ao quadrado e suavizado por uma janela deslizante de 0,1s, depois tira-se a raiz — isso dá uma curva de amplitude que sobe durante contração muscular e cai no repouso, sem a oscilação bipolar do sinal bruto.
Baseline local: em vez de um único valor fixo de "repouso" para o exame inteiro, calcula-se o percentil 10 do envelope dentro de uma janela local de 120s, deslizando ao longo do tempo. Isso segue o mesmo nível de ruído de fundo do paciente, que varia com a postura, o estágio de sono e a impedância do eletrodo ao longo da noite.

"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from threshold_rule import detect_events, raw_mask, segments_from_mask, merge_gaps, MERGE_GAP_S

DATA_DIR = Path(__file__).resolve().parents[2] / "data"   # testes/src/limiar/../.. -> testes / data
RESULTS_DIR = Path(__file__).resolve().parent / "results"  # testes/src/limiar/results
EMG_CHANNEL_INDEX = 4
FS = 100
METHODS = ["single", "double"]
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
    """Correspondencia por sobreposicao temporal, restrita a `event_type`.

    Retorna {tp, fp, fn} (contagem de eventos, nao de amostras).
    """
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

    tp = int(gt_matched.sum())          # eventos verdadeiros capturados por >=1 deteccao correta
    fn = int((~gt_matched).sum())       # eventos verdadeiros sem nenhuma deteccao correspondente
    fp = int((~det_matched).sum())      # deteccoes deste tipo sem evento verdadeiro correspondente
    return {"tp": tp, "fp": fp, "fn": fn}


def prf1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 and not (np.isnan(precision) or np.isnan(recall)) else np.nan)
    return precision, recall, f1


def fragmentation_stats(emg: np.ndarray, method: str, fs: int = FS) -> dict:
    """Mede a fragmentacao INTRINSECA de cada metodo de limiar, isolada do
    pos-processamento de merge_gaps -- ver docstring de detect_events."""
    _, _, mask = raw_mask(emg, method=method, fs=fs)
    segs_raw = segments_from_mask(mask)
    mask_merged = merge_gaps(mask, gap_samples=int(round(MERGE_GAP_S * fs)))
    segs_merged = segments_from_mask(mask_merged)
    durs_raw = np.array([(e - s) / fs for s, e in segs_raw])
    return {
        "n_segments_raw": len(segs_raw),
        "n_segments_after_merge_gaps": len(segs_merged),
        "n_short_fragments_raw_lt_0.3s": int((durs_raw < 0.3).sum()) if len(durs_raw) else 0,
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    subject_stems = sorted(p.stem for p in DATA_DIR.glob("synth*.pt"))
    assert len(subject_stems) > 0, f"nenhum synth*.pt encontrado em {DATA_DIR}"

    rows = []
    frag_rows = []
    for stem in subject_stems:
        # pega os eventos ja carregados do ground truth (csv) e o EMG sintetico (pt)
        emg = load_synth_exam(DATA_DIR / f"{stem}.pt")
        gt_df = load_ground_truth(DATA_DIR / f"{stem}_events.csv")

        for method in METHODS:
            # metricas de evento (com merge_gaps -- pipeline "real")
            detected = detect_events(emg, method=method, fs=FS, apply_merge_gaps=True)
            n_unclassified = sum(1 for d in detected if d.type == "unclassified")
            for etype in EVENT_TYPES:
                m = match_events(gt_df, detected, etype)
                precision, recall, f1 = prf1(m["tp"], m["fp"], m["fn"])
                rows.append({
                    "subject": stem, "method": method, "event_type": etype,
                    "tp": m["tp"], "fp": m["fp"], "fn": m["fn"],
                    "n_gt": int((gt_df["type"] == etype).sum()),
                    "n_detected": sum(1 for d in detected if d.type == etype),
                    "n_unclassified_total": n_unclassified,
                    "precision": precision, "recall": recall, "f1": f1,
                })

            # fragmentacao intrinseca (sem merge_gaps) -- mostra a vantagem
            # estrutural do limiar duplo antes do pos-processamento mascara-la
            frag = fragmentation_stats(emg, method=method, fs=FS)
            frag_rows.append({"subject": stem, "method": method, **frag})

    per_subject = pd.DataFrame(rows)
    per_subject.to_csv(RESULTS_DIR / "per_subject_metrics.csv", index=False)

    frag_df = pd.DataFrame(frag_rows)
    frag_df.to_csv(RESULTS_DIR / "fragmentation_stats.csv", index=False)

    agg = (per_subject.groupby(["method", "event_type"])
           .agg(tp=("tp", "sum"), fp=("fp", "sum"), fn=("fn", "sum"),
                n_gt=("n_gt", "sum"), n_detected=("n_detected", "sum"))
           .reset_index())
    agg["precision"], agg["recall"], agg["f1"] = zip(*agg.apply(
        lambda r: prf1(r["tp"], r["fp"], r["fn"]), axis=1))
    agg.to_csv(RESULTS_DIR / "summary_metrics.csv", index=False)

    frag_agg = frag_df.groupby("method")[
        ["n_segments_raw", "n_segments_after_merge_gaps", "n_short_fragments_raw_lt_0.3s"]
    ].sum().reset_index()
    frag_agg.to_csv(RESULTS_DIR / "fragmentation_summary.csv", index=False)

    print(per_subject.to_string(index=False))
    print()
    print("=== RESUMO AGREGADO (todos os 10 exames sinteticos, 2h cada) ===")
    print(agg.to_string(index=False))
    print()
    print("=== FRAGMENTACAO INTRINSECA (antes de merge_gaps) ===")
    print(frag_agg.to_string(index=False))
    return per_subject, agg, frag_df, frag_agg


if __name__ == "__main__":
    main()

