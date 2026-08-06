"""
Avaliacao intervalar das tres variantes de testes sequenciais/deteccao de
ponto de mudanca (CUSUM classico, GLR multi-escala, CUSUM com esquecimento)
em exames reais -- MESMA metodologia usada em testes/src/limiar/evaluate.py
e testes/src/tkeo/evaluate.py apos a revisao do usuario:

1. Le cada evento anotado no arquivo <exame>_revisado.csv.
2. Converte onset/duration para tempo do .pt usando annot_start quando
   disponivel em classifier/labels/exam_config.json.
3. Recorta o EMG apenas no intervalo anotado.
4. Executa o detector (cusum_classic / glr_multiscale / cusum_leaky) nesse
   recorte, sem varrer o exame inteiro -- baseline robusta (mediana+MAD) e
   os estatisticos sequenciais sao calculados apenas dentro do recorte.
5. Classificacao por duracao + amplitude identica aos outros dois testes:
   fasico 0.1-5.0s, any (ambigua) 5.0-15.0s (exclusive-exclusive), tonico
   >=15.0s, todos exigindo score (pico do ENVELOPE RMS / baseline robusta
   local mu0, nao mais o estatistico g/Lambda) >= 2.0.

Isolado: usa apenas cusum_glr_rule.py (deste mesmo diretorio) + torch/numpy/
pandas. Nao importa nada de classifier/, src/sleep_rswa/ nem dos outros
testes (reimplementado aqui para manter a pasta autocontida).

NOTA: o recorte por intervalo anotado muda o regime de operacao do CUSUM
classico e do GLR em relacao a avaliacao original (exame sintetico
completo, 2h): a janela de baseline robusta (120s) e maior que muitos
intervalos anotados, entao a normalizacao z[n] pode degenerar em janelas
curtas -- isso e esperado e nao e um bug (ver comentarios no proprio
cusum_glr_rule.py sobre a limitacao estrutural do CUSUM classico/GLR em
janelas curtas ou com poucos ciclos de "recorte-quieto").

Saidas (em testes/src/cusum_glr/results/), MESMO FORMATO de
testes/src/limiar/results/ e testes/src/tkeo/results/ para permitir
comparacao direta entre os tres testes:
  interval_analysis.csv   -- um resumo por intervalo anotado
  detected_events.csv     -- lista completa dos eventos detectados dentro
                             de cada intervalo (onset/duration/type/score)
  per_subject_metrics.csv -- agregados por exame
  summary_metrics.csv     -- agregados globais
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cusum_glr_rule import (
    detect_events_cusum, detect_events_glr, detect_events_cusum_leaky,
    CUSUM_K, CUSUM_H, GLR_H, CUSUM_LEAKY_K, CUSUM_LEAKY_H, CUSUM_LEAKY_RHO,
)

DATA_DIR = Path(__file__).resolve().parents[2] / "data_real"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
CONFIG_PATH = Path(__file__).resolve().parents[3] / "classifier" / "labels" / "exam_config.json"
EMG_CHANNEL_INDEX = 4
FS = 100
METHODS = ["cusum_classic", "glr_multiscale", "cusum_leaky"]
EVENT_TYPES = ["phasic", "tonic", "any"]

DETECT_FNS = {
    "cusum_classic": lambda emg, fs: detect_events_cusum(emg, fs=fs, k=CUSUM_K, h=CUSUM_H, apply_merge_gaps=True),
    "glr_multiscale": lambda emg, fs: detect_events_glr(emg, fs=fs, h=GLR_H, apply_merge_gaps=True),
    "cusum_leaky": lambda emg, fs: detect_events_cusum_leaky(emg, fs=fs, k=CUSUM_LEAKY_K, h=CUSUM_LEAKY_H, rho=CUSUM_LEAKY_RHO, apply_merge_gaps=True),
}


def load_exam_emg(pt_path: Path):
    obj = torch.load(pt_path, map_location="cpu", weights_only=False)
    return obj["signals"][:, EMG_CHANNEL_INDEX, :].numpy().astype("float64").reshape(-1)


def load_offsets(config_path: Path) -> dict[str, float | None]:
    if not config_path.exists():
        return {}
    cfg = json.loads(config_path.read_text())
    return {exam: meta.get("annot_start") for exam, meta in cfg.items()}


def load_ground_truth(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["end_s"] = df["onset_s"] + df["duration_s"]
    df["interval_id"] = range(1, len(df) + 1)
    return df


def resolve_annot_start(subject: str, offsets: dict[str, float | None]) -> float:
    value = offsets.get(subject)
    return float(value) if value is not None else 0.0


def clip_interval_to_exam(start_s: float, end_s: float, n_samples: int, fs: int = FS) -> tuple[int, int]:
    i0 = max(0, int(round(start_s * fs)))
    i1 = min(n_samples, int(round(end_s * fs)))
    return i0, i1


def detect_events_in_interval(
    emg: object,
    start_s_pt: float,
    end_s_pt: float,
    method: str,
    fs: int = FS,
) -> tuple[list[dict], dict]:
    i0, i1 = clip_interval_to_exam(start_s_pt, end_s_pt, len(emg), fs=fs)
    clipped_start_s = i0 / fs
    clipped_end_s = i1 / fs

    meta = {
        "analysis_start_s_pt": round(clipped_start_s, 3),
        "analysis_end_s_pt": round(clipped_end_s, 3),
        "analysis_duration_s": round(max(0.0, clipped_end_s - clipped_start_s), 3),
        "was_clipped": (abs(clipped_start_s - start_s_pt) > 1e-9) or (abs(clipped_end_s - end_s_pt) > 1e-9),
        "is_empty": i0 >= i1,
    }
    if i0 >= i1:
        return [], meta

    local_events = DETECT_FNS[method](emg[i0:i1], fs)
    detected = []
    for event_idx, ev in enumerate(local_events, start=1):
        onset_s = clipped_start_s + ev.onset_s
        end_s = onset_s + ev.duration_s
        detected.append({
            "event_index_in_interval": event_idx,
            "event_type": ev.type,
            "onset_s": round(onset_s, 3),
            "end_s": round(end_s, 3),
            "duration_s": round(ev.duration_s, 3),
            "score": ev.score,
        })
    return detected, meta


def classify_interval_status(gt_type: str, counts: dict[str, int]) -> str:
    if counts.get(gt_type, 0) > 0:
        return "same_type_detected"
    if sum(counts.values()) == 0:
        return "none_detected"
    if counts.get("any", 0) > 0 and counts.get("phasic", 0) == 0 and counts.get("tonic", 0) == 0:
        return "only_ambiguous"
    return "other_type_detected"


def aggregate_interval_metrics(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    grouped = (df.groupby(group_cols)
               .agg(
                   n_intervals=("interval_id", "count"),
                   n_same_type_detected=("same_type_detected", "sum"),
                   n_any_detected=("any_detected", "sum"),
                   n_none_detected=("none_detected", "sum"),
                   n_only_ambiguous=("only_ambiguous", "sum"),
                   n_other_type_detected=("other_type_detected", "sum"),
                   n_clipped_intervals=("was_clipped", "sum"),
                   mean_n_detected_total=("n_detected_total", "mean"),
                   mean_n_detected_phasic=("n_detected_phasic", "mean"),
                   mean_n_detected_tonic=("n_detected_tonic", "mean"),
                   mean_n_detected_ambiguous=("n_detected_ambiguous", "mean"),
               )
               .reset_index())

    grouped["same_type_rate"] = grouped["n_same_type_detected"] / grouped["n_intervals"]
    grouped["any_detection_rate"] = grouped["n_any_detected"] / grouped["n_intervals"]
    grouped["mean_n_detected_total"] = grouped["mean_n_detected_total"].round(3)
    grouped["mean_n_detected_phasic"] = grouped["mean_n_detected_phasic"].round(3)
    grouped["mean_n_detected_tonic"] = grouped["mean_n_detected_tonic"].round(3)
    grouped["mean_n_detected_ambiguous"] = grouped["mean_n_detected_ambiguous"].round(3)
    grouped["same_type_rate"] = grouped["same_type_rate"].round(6)
    grouped["any_detection_rate"] = grouped["any_detection_rate"].round(6)
    return grouped


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    offsets = load_offsets(CONFIG_PATH)
    subject_stems = sorted(p.stem for p in DATA_DIR.glob("*.pt"))
    assert subject_stems, f"nenhum .pt encontrado em {DATA_DIR}"

    interval_rows = []
    detected_rows = []

    for stem in subject_stems:
        emg = load_exam_emg(DATA_DIR / f"{stem}.pt")
        gt_df = load_ground_truth(DATA_DIR / f"{stem}_revisado.csv")
        annot_start = resolve_annot_start(stem, offsets)

        for row in gt_df.itertuples(index=False):
            gt_onset_s = float(row.onset_s)
            gt_end_s = float(row.end_s)
            gt_duration_s = float(row.duration_s)
            gt_type = str(row.type)
            start_s_pt = gt_onset_s - annot_start
            end_s_pt = gt_end_s - annot_start

            for method in METHODS:
                detected, meta = detect_events_in_interval(
                    emg=emg,
                    start_s_pt=start_s_pt,
                    end_s_pt=end_s_pt,
                    method=method,
                    fs=FS,
                )

                counts = {etype: sum(1 for d in detected if d["event_type"] == etype) for etype in EVENT_TYPES}
                status = classify_interval_status(gt_type, counts)

                interval_rows.append({
                    "subject": stem,
                    "interval_id": int(row.interval_id),
                    "method": method,
                    "gt_type": gt_type,
                    "csv_onset_s": round(gt_onset_s, 3),
                    "csv_end_s": round(gt_end_s, 3),
                    "csv_duration_s": round(gt_duration_s, 3),
                    "annot_start_used": round(annot_start, 3),
                    "pt_onset_s": round(start_s_pt, 3),
                    "pt_end_s": round(end_s_pt, 3),
                    **meta,
                    "n_detected_total": len(detected),
                    "n_detected_phasic": counts["phasic"],
                    "n_detected_tonic": counts["tonic"],
                    "n_detected_ambiguous": counts["any"],
                    "detected_types": "|".join(sorted({d["event_type"] for d in detected})) if detected else "",
                    "status": status,
                    "same_type_detected": status == "same_type_detected",
                    "any_detected": len(detected) > 0,
                    "none_detected": status == "none_detected",
                    "only_ambiguous": status == "only_ambiguous",
                    "other_type_detected": status == "other_type_detected",
                })

                for det in detected:
                    detected_rows.append({
                        "subject": stem,
                        "interval_id": int(row.interval_id),
                        "method": method,
                        "gt_type": gt_type,
                        "csv_onset_s": round(gt_onset_s, 3),
                        "csv_end_s": round(gt_end_s, 3),
                        "annot_start_used": round(annot_start, 3),
                        "analysis_start_s_pt": meta["analysis_start_s_pt"],
                        "analysis_end_s_pt": meta["analysis_end_s_pt"],
                        "analysis_duration_s": meta["analysis_duration_s"],
                        **det,
                    })

    interval_df = pd.DataFrame(interval_rows)
    detected_df = pd.DataFrame(detected_rows)
    per_subject = aggregate_interval_metrics(interval_df, ["subject", "method", "gt_type"])
    summary = aggregate_interval_metrics(interval_df, ["method", "gt_type"])

    interval_df.to_csv(RESULTS_DIR / "interval_analysis.csv", index=False)
    detected_df.to_csv(RESULTS_DIR / "detected_events.csv", index=False)
    per_subject.to_csv(RESULTS_DIR / "per_subject_metrics.csv", index=False)
    summary.to_csv(RESULTS_DIR / "summary_metrics.csv", index=False)

    print("=== ANALISE POR INTERVALO ANOTADO (CUSUM/GLR) ===")
    print(f"{len(interval_df)} linhas gravadas em {RESULTS_DIR / 'interval_analysis.csv'}")
    print(f"{len(detected_df)} eventos gravados em {RESULTS_DIR / 'detected_events.csv'}")
    print()
    print("=== RESUMO AGREGADO ===")
    print(summary.to_string(index=False))
    return interval_df, detected_df, per_subject, summary


if __name__ == "__main__":
    main()
