"""
Teste 1: Limiar simples vs. limiar duplo (histerese) para isolar e
classificar eventos tonicos/fasicos no envelope RMS do EMG.

Modulo ISOLADO dentro de testes/ -- nao importa nada de classifier/ ou
src/sleep_rswa/. Reimplementa apenas o minimo necessario (envelope RMS,
baseline local por percentil, deteccao de segmentos) para que este teste
nao dependa do estado do codigo de producao.

Duas variantes de deteccao, mesma classificacao por duracao a partir daqui:

1) LIMIAR SIMPLES (single threshold)
   Um so corte: ativo enquanto envelope(t) > k * baseline(t).
   Risco conhecido (motivacao do teste 2): ruido de alta frequencia que
   flutua em torno do limiar fragmenta um unico evento sustentado em varios
   segmentos curtos ("chattering"), e vice-versa pode fundir dois eventos
   proximos se o envelope nunca cai abaixo do limiar entre eles.

2) LIMIAR DUPLO / HISTERESE (double threshold, estilo trigger de Schmitt)
   Dois cortes: k_on (mais alto, exige evidencia forte para COMECAR um
   evento) e k_off (mais baixo, so exige evidencia fraca para MANTER o
   evento ja iniciado). Uma vez ativo, o segmento so termina quando o
   envelope cai abaixo de k_off. Isso e a tecnica classica usada em
   deteccao de onset de EMG (ex. Hodges & Bui) e em deteccao de bordas
   (Canny) exatamente para reduzir fragmentacao por ruido perto do limiar,
   sem exigir evidencia forte para SUSTENTAR (so para COMECAR) um evento.

Apos a deteccao de segmentos (por qualquer um dos dois metodos), a
classificacao fasico/tonico e por duracao, replicando os cortes ja
validados no pipeline de producao (classifier/movement_clf/tonic_phasic.py):
  fasico  : 0.5s <= duracao <= 5.0s   -> rotulo final
  tonico  : duracao >= 16.0s          -> rotulo final (aqui, sem a ressalva
                                          de revisao humana da producao --
                                          este teste usa GROUND TRUTH
                                          SINTETICO exato, entao pode-se
                                          avaliar "tonico" diretamente)
  zona morta (5s < duracao < 16s) e sub-0.5s: nao classificados (descartados
                                          da contagem de fasico/tonico; ver
                                          avaliacao para como isso e tratado)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FS = 100                 # Hz -- igual ao gerador sintetico em testes/generate_synthetic_data.py
EPOCH_SEC = 3.0
SAMPLES_PER_EPOCH = 300

# --- parametros de deteccao (mesma escala validada no pipeline de producao) ---
BASELINE_WIN_S = 120.0
BASELINE_PCT = 10
MERGE_GAP_S = 1.0          # funde lacunas curtas (<= isso) entre segmentos ativos
K_SINGLE = 2.5             # limiar simples
K_ON = 3.0                 # limiar duplo: corte de INICIO (mais exigente)
K_OFF = 1.5                # limiar duplo: corte de MANUTENCAO (menos exigente)

# --- classificacao por duracao (mesmos cortes do pipeline de producao) ---
PHASIC_LO_S = 0.5
PHASIC_HI_S = 5.0
TONIC_MIN_DUR_S = 16.0


def rms_envelope(x: np.ndarray, win_sec: float = 0.1, fs: int = FS) -> np.ndarray:
    """Envelope RMS de janela deslizante (mesmo comprimento da entrada)."""
    win = max(1, int(round(win_sec * fs)))
    x2 = x.astype(np.float64) ** 2
    kernel = np.ones(win) / win
    ms = np.convolve(x2, kernel, mode="same")
    return np.sqrt(ms)


def rolling_baseline(env: np.ndarray, win_sec: float = BASELINE_WIN_S,
                      pct: float = BASELINE_PCT, fs: int = FS) -> np.ndarray:
    """Baseline local: percentil `pct` dentro de uma janela rolante centrada
    de win_sec segundos (implementacao por blocos, igual ao pipeline de
    producao -- ver classifier/movement_clf/tonic_phasic.py)."""
    win = max(1, int(round(win_sec * fs)))
    n = len(env)
    half = win // 2
    step = max(1, win // 4)
    edges = list(range(0, n, step)) + [n]
    block_vals, block_centers = [], []
    for s in edges[:-1]:
        e = min(n, s + step)
        lo = max(0, s - half)
        hi = min(n, e + half)
        block_vals.append(np.percentile(env[lo:hi], pct))
        block_centers.append((s + e) / 2)
    return np.interp(np.arange(n), block_centers, block_vals)


def merge_gaps(mask: np.ndarray, gap_samples: int) -> np.ndarray:
    """Funde lacunas curtas (<= gap_samples) entre segmentos True."""
    mask = mask.copy()
    n = len(mask)
    i = 0
    while i < n:
        if not mask[i]:
            j = i
            while j < n and not mask[j]:
                j += 1
            gap_len = j - i
            has_before = i > 0 and mask[i - 1]
            has_after = j < n and mask[j]
            if has_before and has_after and gap_len <= gap_samples:
                mask[i:j] = True
            i = j
        else:
            i += 1
    return mask


def segments_from_mask(mask: np.ndarray) -> list[tuple[int, int]]:
    """Devolve lista de (start_sample, end_sample_exclusive) para runs True."""
    n = len(mask)
    segs = []
    i = 0
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            segs.append((i, j + 1))
            i = j + 1
        else:
            i += 1
    return segs


def single_threshold_mask(env: np.ndarray, baseline: np.ndarray, k: float = K_SINGLE) -> np.ndarray:
    """Limiar simples: ativo enquanto env > k * baseline."""
    return env > (k * baseline)


def double_threshold_mask(env: np.ndarray, baseline: np.ndarray,
                           k_on: float = K_ON, k_off: float = K_OFF) -> np.ndarray:
    """Limiar duplo / histerese (Schmitt trigger):
    - Precisa cruzar k_on*baseline para ATIVAR um segmento (evidencia forte).
    - Uma vez ativo, so DESATIVA quando cai abaixo de k_off*baseline
      (evidencia fraca basta para manter -- evita fragmentar um evento
      sustentado por flutuacoes de ruido perto do limiar).

    Implementacao vetorizada: calcula onde k_on e cruzado (ignition points),
    e para cada ignicao estende o segmento enquanto env permanecer acima de
    k_off, usando os limites dos segmentos de "env > k_off*baseline".
    """
    above_off = env > (k_off * baseline)
    above_on = env > (k_on * baseline)

    off_segs = segments_from_mask(above_off)
    mask = np.zeros_like(above_off)
    for s, e in off_segs:
        # este segmento (definido pelo limiar baixo) so conta se contiver
        # pelo menos 1 amostra que cruzou o limiar alto (ignicao)
        if np.any(above_on[s:e]):
            mask[s:e] = True
    return mask


@dataclass
class DetectedEvent:
    onset_s: float
    duration_s: float
    type: str          # "phasic" | "tonic" | "unclassified"
    score: float       # peak_env / mean_baseline dentro do segmento (proxy deterministico de confianca)


def segments_to_events(segs: list[tuple[int, int]], env: np.ndarray, baseline: np.ndarray,
                        fs: int = FS, phasic_lo_s: float = PHASIC_LO_S,
                        phasic_hi_s: float = PHASIC_HI_S,
                        tonic_min_dur_s: float = TONIC_MIN_DUR_S) -> list[DetectedEvent]:
    """Converte segmentos (amostras) em eventos classificados por duracao.

    Segmentos com duracao < phasic_lo_s sao descartados (ruido/micro-
    flutuacao). Segmentos na zona morta (phasic_hi_s < dur < tonic_min_dur_s)
    sao mantidos com type="unclassified" -- no pipeline de producao isso
    vira "tonic_candidate" (revisao humana); aqui, como o ground truth
    sintetico so contem eventos EXATAMENTE nas faixas fasico/tonico, um
    segmento na zona morta e sempre um erro do detector (fragmentacao ou
    fusao), nunca um evento real ambiguo -- por isso a avaliacao trata
    "unclassified" como nem fasico nem tonico (conta contra recall de ambos
    se corresponder a um evento real, e nao conta como FP de nenhum tipo
    classificado).
    """
    events = []
    for s, e in segs:
        dur_s = (e - s) / fs
        if dur_s < phasic_lo_s:
            continue
        onset_s = s / fs
        score = float(np.max(env[s:e]) / np.mean(baseline[s:e]))
        if phasic_lo_s <= dur_s <= phasic_hi_s:
            etype = "phasic"
        elif dur_s >= tonic_min_dur_s:
            etype = "tonic"
        else:
            etype = "unclassified"
        events.append(DetectedEvent(onset_s=onset_s, duration_s=dur_s, type=etype, score=round(score, 3)))
    return events


def raw_mask(emg_flat: np.ndarray, method: str, fs: int = FS) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Envelope, baseline e mascara BRUTA (antes de qualquer fusao de
    lacunas) -- usado para medir fragmentacao intrinseca de cada metodo,
    isolada do efeito de merge_gaps."""
    env = rms_envelope(emg_flat, win_sec=0.1, fs=fs)
    baseline = rolling_baseline(env, win_sec=BASELINE_WIN_S, pct=BASELINE_PCT, fs=fs)
    if method == "single":
        mask = single_threshold_mask(env, baseline, k=K_SINGLE)
    elif method == "double":
        mask = double_threshold_mask(env, baseline, k_on=K_ON, k_off=K_OFF)
    else:
        raise ValueError(f"metodo desconhecido: {method!r} (use 'single' ou 'double')")
    return env, baseline, mask


def detect_events(emg_flat: np.ndarray, method: str, fs: int = FS,
                   apply_merge_gaps: bool = True) -> list[DetectedEvent]:
    """Pipeline completo: envelope -> baseline -> mascara (metodo escolhido)
    -> fusao de lacunas curtas (opcional) -> segmentos -> classificacao por duracao.

    method: "single" (limiar simples) ou "double" (limiar duplo/histerese).
    apply_merge_gaps: se False, pula a fusao de lacunas curtas -- usado para
        medir o efeito ISOLADO de cada metodo de limiar, sem o
        pos-processamento que absorve fragmentacao (ver testes/src/limiar/
        results/ e o relatorio: merge_gaps mascara boa parte da diferenca
        entre os dois metodos no nivel de evento classificado).
    """
    env, baseline, mask = raw_mask(emg_flat, method=method, fs=fs)
    if apply_merge_gaps:
        mask = merge_gaps(mask, gap_samples=int(round(MERGE_GAP_S * fs)))
    segs = segments_from_mask(mask)
    return segments_to_events(segs, env, baseline, fs=fs)
