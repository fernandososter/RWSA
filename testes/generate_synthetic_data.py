"""
Gerador de exames sinteticos com eventos tonicos/fasicos de EMG, para uso
EXCLUSIVO na pasta testes/ (nao importa nada de classifier/ ou src/sleep_rswa/,
para nao misturar com o codigo de producao).

Motivacao (ver discussao do projeto): rotulos manuais revisados (tonic_labels/
phasic_labels dos exames reais em classifier/data/*.pt) sao conhecidos por
ter inconsistencia inter/intra-revisor e NAO podem ser usados para validar
regras deterministicas ou o classificador -- qualquer teste de acerto de
duracao/onset precisa de ground truth garantido por construcao. Este script
gera esse ground truth.

Saida (em testes/data/):
  synth01.pt ... synth10.pt
      dict no MESMO schema dos .pt de producao (ver classifier/movement_clf/dataio.py):
        signals       Tensor[T, 5, 300] float32   (canais: F2-F4, C4-A1, P4-O2, ROC-LOC, EMG1-EMG2)
        sleep_stages  Tensor[T]         int64      (0=W,1=N1,2=N2,3=N3,4=REM)
        tonic_labels  Tensor[T]         float32    {0,1} por mini-epoca (regra: >=50% da epoca coberta por evento tonico)
        phasic_labels Tensor[T]         float32    {0,1} por mini-epoca (regra: >=50% da epoca coberta por evento fasico)
        channel_mask  Tensor[5]         bool       (todos True; sem canais ausentes no sintetico)
        channel_names list[5]          str
  synth01_events.csv ... synth10_events.csv
      ground truth EXATO (nao quantizado em mini-epoca), uma linha por evento:
        onset_s, duration_s, type   (type em {"tonic","phasic"})
      Este manifesto é o que deve ser usado para validar o script de
      inferencia (onset_s/duration_s/type/score previstos vs. aqui).

Parametros do sinal EMG (canal index 4) escaneados/validados em rodadas
anteriores deste projeto (ver discussao no historico): amplitude de burst
~3x a amplitude basal, ruido gaussiano de fundo, fs=100Hz.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

FS = 100                      # Hz, igual ao pipeline de producao
EPOCH_SEC = 3.0
SAMPLES_PER_EPOCH = 300        # = FS * EPOCH_SEC
CHANNEL_NAMES = ["F2-F4", "C4-A1", "P4-O2", "ROC-LOC", "EMG1-EMG2"]
EMG_CHANNEL_INDEX = 4

# --- parametros de amplitude do EMG sintetico (mesma escala usada nas
#     validacoes sinteticas anteriores deste projeto) ---
BASELINE_AMP = 3e-6
ON_AMP = 9e-6                 # ~3x baseline durante evento
NOISE_STD = 1.2e-6

# --- duracao dos eventos ---
PHASIC_DUR_RANGE = (0.6, 4.5)   # s (criterio Lapierre/Montplaisir: 0.1-5s)
TONIC_DUR_RANGE = (16.0, 45.0)  # s (bem acima do corte tonico ~>=15-20s)

# --- espacamento entre eventos ---
GAP_LOOSE_RANGE = (3.0, 25.0)     # s, entre eventos "soltos"
GAP_CLUSTER_RANGE = (2.0, 8.0)    # s, dentro de um cluster de movimento denso
CLUSTER_N_RANGE = (2, 5)          # numero de sub-eventos por cluster (uniforme discreto, exclusivo no topo)
P_KIND = {"phasic": 0.45, "tonic": 0.20, "cluster": 0.35}


@dataclass
class SynthEvent:
    onset_s: float
    duration_s: float
    type: str  # "phasic" | "tonic"


def synth_burst(rng: np.random.Generator, dur_s: float, fs: int = FS) -> np.ndarray:
    """Plato de amplitude elevada com envelope levemente flutuante + ruido."""
    n = int(round(dur_s * fs))
    ramp_n = max(1, int(0.05 * fs))  # rampa de 50ms de subida/descida (evita degrau abrupto)
    env = np.full(n, ON_AMP, dtype=np.float64)
    if n > 2 * ramp_n:
        env[:ramp_n] *= np.linspace(0.0, 1.0, ramp_n)
        env[-ramp_n:] *= np.linspace(1.0, 0.0, ramp_n)
    env = env * (0.6 + 0.4 * rng.random(n))
    carrier = rng.normal(0, 1, n)
    return env * np.abs(carrier)


def synth_quiet(rng: np.random.Generator, dur_s: float, fs: int = FS) -> np.ndarray:
    n = int(round(dur_s * fs))
    return rng.normal(0, NOISE_STD, n) + BASELINE_AMP * 0.3 * np.abs(rng.normal(0, 1, n))


def build_synth_emg_recording(rng: np.random.Generator, duration_hr: float, fs: int = FS):
    """Gera 1 canal EMG continuo com eventos tonicos/fasicos isolados e em
    clusters (movimento denso, gaps curtos de 2-8s) -- cenario adversarial
    que reproduz o padrao de contaminacao de baseline visto nos exames reais.

    Retorna (sinal[N], eventos: list[SynthEvent], duracao_total_s).
    """
    total_n = int(duration_hr * 3600 * fs)
    sig = np.zeros(total_n)
    events: list[SynthEvent] = []
    pos = 0
    kinds = list(P_KIND.keys())
    probs = list(P_KIND.values())

    def emit(kind: str):
        nonlocal pos
        dur = rng.uniform(*PHASIC_DUR_RANGE) if kind == "phasic" else rng.uniform(*TONIC_DUR_RANGE)
        n = int(round(dur * fs))
        n = min(n, total_n - pos)
        if n <= 0:
            return False
        onset_s = pos / fs
        sig[pos:pos + n] = synth_burst(rng, n / fs, fs=fs)
        events.append(SynthEvent(onset_s=onset_s, duration_s=n / fs, type=kind))
        pos += n
        return True

    while pos < total_n - fs * 5:
        gap_s = rng.uniform(*GAP_LOOSE_RANGE)
        gap_n = int(gap_s * fs)
        end_gap = min(total_n, pos + gap_n)
        sig[pos:end_gap] = synth_quiet(rng, (end_gap - pos) / fs, fs=fs)
        pos = end_gap
        if pos >= total_n - fs * 5:
            break

        kind = rng.choice(kinds, p=probs)
        if kind in ("phasic", "tonic"):
            emit(kind)
        else:  # cluster: 2-4 eventos com gaps curtos entre si
            n_sub = rng.integers(*CLUSTER_N_RANGE)
            for _ in range(n_sub):
                sub_kind = rng.choice(["phasic", "tonic"], p=[0.5, 0.5])
                ok = emit(sub_kind)
                if not ok:
                    break
                gap_sub = rng.uniform(*GAP_CLUSTER_RANGE)
                gn = int(gap_sub * fs)
                gn = min(gn, total_n - pos)
                if gn > 0:
                    sig[pos:pos + gn] = synth_quiet(rng, gn / fs, fs=fs)
                    pos += gn

    sig = sig[:pos]
    return sig, events, pos / fs


def build_synth_hypnogram(rng: np.random.Generator, n_epochs: int, epoch_sec: float = EPOCH_SEC) -> np.ndarray:
    """Hipnograma sintetico simplificado (nao clinico): ciclos de sono
    tipicos W->N1->N2->N3->N2->REM repetidos, com REM crescendo ao longo da
    noite. So usado para popular sleep_stages com algo nao-degenerado --
    o alvo de movimento do classificador NAO depende do estagio (ver
    dataio.py: "a NOITE TODA, sem mascara de REM"), entao esta parte nao
    precisa ser fisiologicamente precisa.
    """
    stages = []
    cycle_min = 90.0
    epochs_per_cycle = int(cycle_min * 60 / epoch_sec)
    n_cycles = max(1, int(np.ceil(n_epochs / epochs_per_cycle)) + 1)
    seq_template = [0, 1, 2, 3, 2, 4]  # W,N1,N2,N3,N2,REM
    for c in range(n_cycles):
        rem_frac = min(0.35, 0.08 + 0.05 * c)  # REM cresce ao longo da noite
        fracs = {0: 0.03, 1: 0.07, 2: 0.35, 3: 0.20, 4: rem_frac}
        # normaliza fracs para somar 1 (usa so os estagios do template, mas com "N2" repetido)
        remaining = 1.0 - fracs[0] - fracs[1] - fracs[3] - fracs[4]
        fracs[2] = max(0.05, remaining)
        for st in seq_template:
            n_st = max(1, int(round(fracs[st] * epochs_per_cycle / (2 if st == 2 else 1))))
            stages.extend([st] * n_st)
    stages = np.array(stages[:n_epochs], dtype=np.int64)
    if len(stages) < n_epochs:
        stages = np.concatenate([stages, np.full(n_epochs - len(stages), 0, dtype=np.int64)])
    return stages


def epochs_from_events(events: list[SynthEvent], n_epochs: int, epoch_sec: float = EPOCH_SEC,
                        overlap_frac: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Converte eventos continuos em rotulos por mini-epoca.

    Regra (Lapierre/Montplaisir, ver Frauscher 2013): uma mini-epoca recebe
    o rotulo do tipo do evento se a FRACAO coberta pelo evento nessa epoca
    for >= overlap_frac (default 50%).
    """
    tonic = np.zeros(n_epochs, dtype=np.float32)
    phasic = np.zeros(n_epochs, dtype=np.float32)
    for ev in events:
        e_start, e_end = ev.onset_s, ev.onset_s + ev.duration_s
        m_start = int(np.floor(e_start / epoch_sec))
        m_end = int(np.ceil(e_end / epoch_sec))
        for m in range(max(0, m_start), min(n_epochs, m_end + 1)):
            ep_start, ep_end = m * epoch_sec, (m + 1) * epoch_sec
            overlap = max(0.0, min(e_end, ep_end) - max(e_start, ep_start))
            if overlap / epoch_sec >= overlap_frac:
                if ev.type == "tonic":
                    tonic[m] = 1.0
                else:
                    phasic[m] = 1.0
    return tonic, phasic


def build_filler_channel(rng: np.random.Generator, n_samples: int, std: float = 15e-6) -> np.ndarray:
    """Canal generico (EEG/EOG) sem morfologia clinica -- apenas ruido
    colorido de baixa frequencia. NAO usado pelo classificador de movimento
    (dataio.py so le o canal EMG, index 4); existe apenas para manter o
    schema dos .pt compativel com o pipeline de producao.
    """
    white = rng.normal(0, std, n_samples)
    kernel = np.ones(15) / 15.0
    return np.convolve(white, kernel, mode="same").astype(np.float32)


def build_synth_exam(seed: int, duration_hr: float = 2.0, fs: int = FS) -> tuple[dict, list[SynthEvent]]:
    rng = np.random.default_rng(seed)
    emg_raw, events, dur_s = build_synth_emg_recording(rng, duration_hr=duration_hr, fs=fs)

    n_epochs = int(len(emg_raw) // SAMPLES_PER_EPOCH)
    emg_raw = emg_raw[: n_epochs * SAMPLES_PER_EPOCH]
    emg_epochs = emg_raw.reshape(n_epochs, SAMPLES_PER_EPOCH).astype(np.float32)

    # descarta eventos truncados pelo corte em n_epochs*EPOCH_SEC ANTES de
    # computar os rotulos por epoca -- caso contrario a ultima mini-epoca
    # pode ficar marcada por um evento que nao aparece no manifesto CSV
    # (bug de contorno detectado em teste de consistencia .pt vs CSV).
    total_dur_s = n_epochs * EPOCH_SEC
    events = [e for e in events if e.onset_s + e.duration_s <= total_dur_s]

    tonic, phasic = epochs_from_events(events, n_epochs, epoch_sec=EPOCH_SEC)
    stages = build_synth_hypnogram(rng, n_epochs)

    other_channels = np.stack([
        build_filler_channel(rng, n_epochs * SAMPLES_PER_EPOCH).reshape(n_epochs, SAMPLES_PER_EPOCH)
        for _ in range(4)
    ], axis=1)  # [T, 4, 300]

    signals = np.zeros((n_epochs, 5, SAMPLES_PER_EPOCH), dtype=np.float32)
    signals[:, :4, :] = other_channels
    signals[:, EMG_CHANNEL_INDEX, :] = emg_epochs

    # rswa_labels/rswa_conf nao sao lidos por dataio.py (so usado por outros
    # scripts do projeto, ex. view/app.py) -- inclusos apenas por completude
    # de schema. rswa_conf=1.0 (sem incerteza: rotulo sintetico e exato).
    obj = {
        "signals": torch.from_numpy(signals),
        "sleep_stages": torch.from_numpy(stages),
        "tonic_labels": torch.from_numpy(tonic),
        "phasic_labels": torch.from_numpy(phasic),
        "channel_mask": torch.ones(5, dtype=torch.bool),
        "channel_names": list(CHANNEL_NAMES),
        "rswa_labels": torch.zeros(n_epochs, dtype=torch.int64),
        "rswa_conf": torch.ones(n_epochs, dtype=torch.float32),
        "subject_id": f"synth_{seed:02d}",
    }
    return obj, events


def save_exam(obj: dict, events: list[SynthEvent], out_pt: Path, out_csv: Path) -> None:
    torch.save(obj, out_pt)
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["onset_s", "duration_s", "type"])
        for e in events:
            writer.writerow([f"{e.onset_s:.3f}", f"{e.duration_s:.3f}", e.type])


DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "data"


def main(n_exams: int = 10, duration_hr: float = 2.0, out_dir: str | Path | None = None, seed0: int = 7000):
    out_dir = Path(out_dir) if out_dir is not None else DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for i in range(1, n_exams + 1):
        seed = seed0 + i
        obj, events = build_synth_exam(seed=seed, duration_hr=duration_hr)
        stem = f"synth{i:02d}"
        save_exam(obj, events, out_dir / f"{stem}.pt", out_dir / f"{stem}_events.csv")
        n_tonic = sum(1 for e in events if e.type == "tonic")
        n_phasic = sum(1 for e in events if e.type == "phasic")
        summary.append({
            "file": f"{stem}.pt", "seed": seed, "n_epochs": int(obj["tonic_labels"].shape[0]),
            "hours": round(obj["tonic_labels"].shape[0] * EPOCH_SEC / 3600, 3),
            "n_tonic_events": n_tonic, "n_phasic_events": n_phasic,
            "n_tonic_epochs": int(obj["tonic_labels"].sum().item()),
            "n_phasic_epochs": int(obj["phasic_labels"].sum().item()),
        })
        print(f"{stem}.pt  seed={seed}  {summary[-1]['hours']}h  "
              f"tonic_ev={n_tonic} phasic_ev={n_phasic}  "
              f"tonic_epochs={summary[-1]['n_tonic_epochs']} phasic_epochs={summary[-1]['n_phasic_epochs']}")

    with open(out_dir / "manifest_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    return summary


if __name__ == "__main__":
    main()
