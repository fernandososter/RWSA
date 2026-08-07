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
classificacao fasico/any/tonico e por duracao E amplitude (revisao vs.
exames reais, testes/src/limiar/evaluate.py):
  fasico  : 0.1s <= duracao <= 5.0s    E  score >= 2.0 (pico/baseline)
  any     : 5.0s <  duracao < 15.0s    E  score >= 2.0  -- faixa intermediaria
                                          ambigua (nao conta como acerto de
                                          fasico nem de tonico na avaliacao;
                                          substitui a antiga "zona morta")
  tonico  : duracao >= 15.0s           E  score >= 2.0
  score < 2.0 (amplitude insuficiente) OU duracao < 0.1s: descartado, nao
                                          vira evento de nenhum tipo.
  score = max(envelope) / media(baseline) dentro do segmento (mesma metrica
          ja usada como campo "score" de cada DetectedEvent).
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
K_OFF_HOLD_S = 0.0         # limiar duplo: tempo minimo abaixo de k_off para DESLIGAR

# --- classificacao por duracao + amplitude (cortes atualizados pelo usuario) ---
PHASIC_LO_S = 0.1
PHASIC_HI_S = 5.0
TONIC_MIN_DUR_S = 15.0          # faixa "any" (ambigua) fica entre PHASIC_HI_S e TONIC_MIN_DUR_S
MIN_AMPLITUDE_RATIO = 2.0        # score (pico/baseline) minimo para contar como evento de qualquer tipo


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
                           k_on: float = K_ON, k_off: float = K_OFF,
                           off_hold_s: float = K_OFF_HOLD_S,
                           fs: int = FS) -> np.ndarray:
    """Limiar duplo / histerese (Schmitt trigger):
    - Precisa cruzar k_on*baseline para ATIVAR um segmento (evidencia forte).
    - Uma vez ativo, so DESATIVA quando permanece abaixo de k_off*baseline
      por pelo menos off_hold_s segundos consecutivos. Quedas breves abaixo
      de k_off sao absorvidas como parte do mesmo evento.

    Implementacao por maquina de estados para suportar "desligamento
    confirmado no tempo":
    - inactive -> active ao cruzar k_on
    - active -> inactive apenas apos off_hold_s consecutivos abaixo de k_off

    Se a queda abaixo de k_off for confirmada, o fim do evento e marcado de
    forma retroativa no PRIMEIRO instante da queda (nao no instante da
    confirmacao), para nao inflar artificialmente a duracao do segmento.
    """
    n = len(env)
    mask = np.zeros(n, dtype=bool)
    off_hold_samples = max(0, int(round(off_hold_s * fs)))

    active = False
    off_count = 0
    off_start: int | None = None

    for i in range(n):
        on_threshold = k_on * baseline[i]
        off_threshold = k_off * baseline[i]

        if not active:
            if env[i] > on_threshold:
                active = True
                mask[i] = True
            continue

        if env[i] > off_threshold:
            if off_start is not None:
                # Queda curta abaixo de k_off: reabsorve o vale no mesmo evento.
                mask[off_start:i] = True
                off_start = None
                off_count = 0
            mask[i] = True
            continue

        if off_hold_samples == 0:
            active = False
            off_count = 0
            off_start = None
            continue

        if off_start is None:
            off_start = i
        off_count += 1

        if off_count >= off_hold_samples:
            active = False
            off_count = 0
            off_start = None

    if active and off_start is not None:
        # O sinal terminou durante uma queda curta nao confirmada: mantem o
        # trecho abaixo de k_off como parte do evento.
        mask[off_start:n] = True

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
                        tonic_min_dur_s: float = TONIC_MIN_DUR_S,
                        min_amplitude_ratio: float = MIN_AMPLITUDE_RATIO) -> list[DetectedEvent]:
    """Converte segmentos (amostras) em eventos classificados por duracao E amplitude.

    Um segmento so vira evento se: duracao >= phasic_lo_s E score >= min_amplitude_ratio
    (score = pico do envelope / media do baseline dentro do segmento). Caso
    contrario e descartado (nem fasico, nem any, nem tonico).

    Faixas de duracao (aplicadas so apos passar no filtro de amplitude):
      fasico : phasic_lo_s  <= dur <= phasic_hi_s
      tonico : dur >= tonic_min_dur_s
      any    : phasic_hi_s < dur < tonic_min_dur_s -- faixa intermediaria
               ambigua (substitui a antiga "zona morta"/"unclassified"): no
               ground truth revisado isto e sempre um erro de fragmentacao/
               fusao do detector ou um evento genuinamente ambiguo, nunca
               contado como acerto de fasico nem de tonico na avaliacao.
    """
    events = []
    for s, e in segs:
        dur_s = (e - s) / fs
        if dur_s < phasic_lo_s:
            continue
        score = float(np.max(env[s:e]) / np.mean(baseline[s:e]))
        if score < min_amplitude_ratio:
            continue
        onset_s = s / fs
        if phasic_lo_s <= dur_s <= phasic_hi_s:
            etype = "phasic"
        elif dur_s >= tonic_min_dur_s:
            etype = "tonic"
        else:
            etype = "any"
        events.append(DetectedEvent(onset_s=onset_s, duration_s=dur_s, type=etype, score=round(score, 3)))
    return events


def raw_mask(emg_flat: np.ndarray, method: str, fs: int = FS,
             k_single: float = K_SINGLE,
             k_on: float = K_ON,
             k_off: float = K_OFF,
             off_hold_s: float = K_OFF_HOLD_S) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Envelope, baseline e mascara BRUTA (antes de qualquer fusao de
    lacunas) -- usado para medir fragmentacao intrinseca de cada metodo,
    isolada do efeito de merge_gaps."""
    env = rms_envelope(emg_flat, win_sec=0.1, fs=fs)
    baseline = rolling_baseline(env, win_sec=BASELINE_WIN_S, pct=BASELINE_PCT, fs=fs)
    if method == "single":
        mask = single_threshold_mask(env, baseline, k=k_single)
    elif method == "double":
        mask = double_threshold_mask(env, baseline, k_on=k_on, k_off=k_off, off_hold_s=off_hold_s, fs=fs)
    else:
        raise ValueError(f"metodo desconhecido: {method!r} (use 'single' ou 'double')")
    return env, baseline, mask


def detect_events(emg_flat: np.ndarray, method: str, fs: int = FS,
                   apply_merge_gaps: bool = True,
                   k_single: float = K_SINGLE,
                   k_on: float = K_ON,
                   k_off: float = K_OFF,
                   off_hold_s: float = K_OFF_HOLD_S) -> list[DetectedEvent]:
    """Pipeline completo: envelope -> baseline -> mascara (metodo escolhido)
    -> fusao de lacunas curtas (opcional) -> segmentos -> classificacao por duracao.

    method: "single" (limiar simples) ou "double" (limiar duplo/histerese).
    apply_merge_gaps: se False, pula a fusao de lacunas curtas -- usado para
        medir o efeito ISOLADO de cada metodo de limiar, sem o
        pos-processamento que absorve fragmentacao (ver testes/src/limiar/
        results/ e o relatorio: merge_gaps mascara boa parte da diferenca
        entre os dois metodos no nivel de evento classificado).
    k_single: limiar multiplicativo do metodo "single".
    k_on: no metodo "double", limiar multiplicativo para INICIAR o evento.
    k_off: no metodo "double", limiar multiplicativo para ENCERRAR o evento.
    off_hold_s: no metodo "double", exige que o sinal permaneca abaixo de
        k_off por esse tempo antes de desligar. Em 0.0s, reproduz o
        comportamento antigo de desligamento imediato.
    """
    env, baseline, mask = raw_mask(
        emg_flat,
        method=method,
        fs=fs,
        k_single=k_single,
        k_on=k_on,
        k_off=k_off,
        off_hold_s=off_hold_s,
    )
    if apply_merge_gaps:
        mask = merge_gaps(mask, gap_samples=int(round(MERGE_GAP_S * fs)))
    segs = segments_from_mask(mask)
    return segments_to_events(segs, env, baseline, fs=fs)
