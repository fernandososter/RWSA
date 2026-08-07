"""
Avaliacao intervalar do metodo de limiar (single vs. double) em exames reais.

Novo fluxo:
1. Le cada evento anotado no arquivo <exame>_revisado.csv.
2. Converte onset/duration para tempo do .pt usando annot_start quando
   disponivel em classifier/labels/exam_config.json.
3. Recorta o EMG apenas no intervalo anotado.
4. Executa o detector nesse recorte, sem varrer o exame inteiro.
5. Registra:
   - um resumo por intervalo anotado (interval_analysis.csv)
   - a lista completa dos eventos detectados dentro de cada intervalo
     (detected_events.csv)
   - agregados por exame e globais (per_subject_metrics.csv e
     summary_metrics.csv)

Observacao importante:
- A baseline local e os limiares sao calculados apenas dentro do recorte do
  intervalo anotado, conforme solicitado.
- O CSV revisado nao e usado para "guiar" a deteccao dentro do recorte; ele
  apenas define qual janela temporal deve ser analisada.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from threshold_rule import K_OFF, K_OFF_HOLD_S, K_ON, K_SINGLE, detect_events

DATA_DIR = Path(__file__).resolve().parents[2] / "data_real"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
CONFIG_PATH = Path(__file__).resolve().parents[3] / "classifier" / "labels" / "exam_config.json"
EMG_CHANNEL_INDEX = 4
FS = 100
METHODS = ["single", "double"]
EVENT_TYPES = ["phasic", "tonic", "any"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Avalia o detector de limiar em intervalos anotados de exames reais."
    )
    parser.add_argument("--k-single", type=float, default=K_SINGLE,
                        help="Multiplicador do baseline no metodo single.")
    parser.add_argument("--k-on", type=float, default=K_ON,
                        help="Multiplicador do baseline para ligar evento no metodo double.")
    parser.add_argument("--k-off", type=float, default=K_OFF,
                        help="Multiplicador do baseline para desligar evento no metodo double.")
    parser.add_argument("--off-hold-s", type=float, default=K_OFF_HOLD_S,
                        help="Tempo minimo abaixo de k_off para desligar no metodo double.")
    return parser.parse_args()


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
    k_single: float = K_SINGLE,
    k_on: float = K_ON,
    k_off: float = K_OFF,
    off_hold_s: float = K_OFF_HOLD_S,
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

    local_events = detect_events(
        emg[i0:i1],
        method=method,
        fs=fs,
        apply_merge_gaps=True,
        k_single=k_single,
        k_on=k_on,
        k_off=k_off,
        off_hold_s=off_hold_s,
    )
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
    args = parse_args()
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
                    k_single=args.k_single,
                    k_on=args.k_on,
                    k_off=args.k_off,
                    off_hold_s=args.off_hold_s,
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
                    "k_single_used": round(args.k_single, 6),
                    "k_on_used": round(args.k_on, 6),
                    "k_off_used": round(args.k_off, 6),
                    "off_hold_s_used": round(args.off_hold_s, 6),
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
                        "k_single_used": round(args.k_single, 6),
                        "k_on_used": round(args.k_on, 6),
                        "k_off_used": round(args.k_off, 6),
                        "off_hold_s_used": round(args.off_hold_s, 6),
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

    print("=== ANALISE POR INTERVALO ANOTADO ===")
    print(f"{len(interval_df)} linhas gravadas em {RESULTS_DIR / 'interval_analysis.csv'}")
    print(f"{len(detected_df)} eventos gravados em {RESULTS_DIR / 'detected_events.csv'}")
    print()
    print("=== PARAMETROS USADOS ===")
    print(
        f"k_single={args.k_single:.6f} | "
        f"k_on={args.k_on:.6f} | "
        f"k_off={args.k_off:.6f} | "
        f"off_hold_s={args.off_hold_s:.6f}"
    )
    print()
    print("=== RESUMO AGREGADO ===")
    print(summary.to_string(index=False))
    return interval_df, detected_df, per_subject, summary


if __name__ == "__main__":
    main()
