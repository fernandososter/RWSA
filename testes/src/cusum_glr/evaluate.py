"""
Avaliacao: tres variantes de testes sequenciais/deteccao de ponto de mudanca
para isolar e classificar eventos tonicos/fasicos --
  1. CUSUM classico de Page (falha estruturalmente, ver docstring de cusum_glr_rule.py)
  2. GLR multi-escala (janela finita, mas ainda funde clusters dos exames sinteticos)
  3. CUSUM com esquecimento / leaky CUSUM (corrige a falha estrutural do classico)
-- testadas contra os 10 exames sinteticos com ground truth exato em testes/data/.

Isolado: usa apenas cusum_glr_rule.py (deste mesmo diretorio) + torch/numpy/
pandas. Nao importa nada de classifier/, src/sleep_rswa/ nem dos testes
anteriores (testes/src/limiar/, testes/src/tkeo/).

Diagnostico critico incluido (nao so metricas de evento): duracao MAXIMA de
qualquer segmento detectado por exame. O maior evento verdadeiro nos exames
sinteticos e um tonico de <=45s (ver TONIC_DUR_RANGE do gerador) -- qualquer
segmento detectado muito maior que isso e um sinal inequivoco de fusao
patologica (o acumulador nunca resetou), independente do que as metricas de
evento (que podem, por coincidencia, ainda registrar alguns TPs dentro do
blob) sugerissem isoladamente.

Saidas (em testes/src/cusum_glr/results/):
  per_subject_metrics.csv
  summary_metrics.csv
  max_duration_diagnostic.csv
  cusum_glr_comparison.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cusum_glr_rule import (
    detect_events_cusum, detect_events_glr, detect_events_cusum_leaky, FS,
    CUSUM_K, CUSUM_H, GLR_H, CUSUM_LEAKY_K, CUSUM_LEAKY_H, CUSUM_LEAKY_RHO,
)

DATA_DIR = Path(__file__).resolve().parents[2] / "data"    # testes/data
RESULTS_DIR = Path(__file__).resolve().parent / "results"  # testes/src/cusum_glr/results
EMG_CHANNEL_INDEX = 4
EVENT_TYPES = ["phasic", "tonic"]
# duracao maxima fisiologicamente plausivel nos exames sinteticos (TONIC_DUR_RANGE=16-45s,
# ver testes/generate_synthetic_data.py) -- qualquer deteccao > 2x isso e um blob degenerado
MAX_PLAUSIBLE_DUR_S = 90.0

METHODS = {
    "cusum_classic": lambda emg: detect_events_cusum(emg, fs=FS, k=CUSUM_K, h=CUSUM_H),
    "glr_multiscale": lambda emg: detect_events_glr(emg, fs=FS, h=GLR_H),
    "cusum_leaky": lambda emg: detect_events_cusum_leaky(emg, fs=FS, k=CUSUM_LEAKY_K, h=CUSUM_LEAKY_H, rho=CUSUM_LEAKY_RHO),
}


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


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    subject_stems = sorted(p.stem for p in DATA_DIR.glob("synth*.pt"))
    assert len(subject_stems) > 0, f"nenhum synth*.pt encontrado em {DATA_DIR}"

    rows = []
    diag_rows = []
    for stem in subject_stems:
        emg = load_synth_exam(DATA_DIR / f"{stem}.pt")
        gt_df = load_ground_truth(DATA_DIR / f"{stem}_events.csv")

        for method_name, detect_fn in METHODS.items():
            detected = detect_fn(emg)
            max_dur = max((d.duration_s for d in detected), default=0.0)
            diag_rows.append({
                "subject": stem, "method": method_name,
                "n_detected_total": len(detected),
                "max_detected_duration_s": round(max_dur, 1),
                "degenerate_blob": max_dur > MAX_PLAUSIBLE_DUR_S,
            })
            for etype in EVENT_TYPES:
                m = match_events(gt_df, detected, etype)
                precision, recall, f1 = prf1(m["tp"], m["fp"], m["fn"])
                rows.append({
                    "subject": stem, "method": method_name, "event_type": etype,
                    "tp": m["tp"], "fp": m["fp"], "fn": m["fn"],
                    "n_gt": int((gt_df["type"] == etype).sum()),
                    "n_detected": sum(1 for d in detected if d.type == etype),
                    "precision": precision, "recall": recall, "f1": f1,
                })

    per_subject = pd.DataFrame(rows)
    per_subject.to_csv(RESULTS_DIR / "per_subject_metrics.csv", index=False)

    diag_df = pd.DataFrame(diag_rows)
    diag_df.to_csv(RESULTS_DIR / "max_duration_diagnostic.csv", index=False)

    agg = (per_subject.groupby(["method", "event_type"])
           .agg(tp=("tp", "sum"), fp=("fp", "sum"), fn=("fn", "sum"),
                n_gt=("n_gt", "sum"), n_detected=("n_detected", "sum"))
           .reset_index())
    agg["precision"], agg["recall"], agg["f1"] = zip(*agg.apply(
        lambda r: prf1(r["tp"], r["fp"], r["fn"]), axis=1))
    agg.to_csv(RESULTS_DIR / "summary_metrics.csv", index=False)

    print(per_subject.to_string(index=False))
    print()
    print("=== RESUMO AGREGADO (todos os 10 exames sinteticos, 2h cada) ===")
    print(agg.to_string(index=False))
    print()
    print("=== DIAGNOSTICO: duracao maxima detectada por metodo (blob degenerado se > 90s) ===")
    print(diag_df.groupby("method")["max_detected_duration_s"].agg(["max", "mean"]).reset_index().to_string(index=False))
    return per_subject, agg, diag_df


if __name__ == "__main__":
    main()
