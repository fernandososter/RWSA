"""
Teste 2: Operador de energia de Teager-Kaiser (TKEO) para isolar e
classificar eventos tonicos/fasicos no EMG.

Modulo ISOLADO dentro de testes/ -- nao importa nada de classifier/,
src/sleep_rswa/ nem de testes/src/limiar/ (reimplementa o que precisa, para
que cada teste desta pasta possa ser lido e rodado de forma independente).

--------------------------------------------------------------------------
O QUE E O TKEO
--------------------------------------------------------------------------
Para um sinal discreto x[n], o operador de Teager-Kaiser e definido como:

    psi(x[n]) = x[n]^2 - x[n-1] * x[n+1]

Diferente do envelope RMS (que so enxerga amplitude, via x[n]^2 suavizado),
o TKEO e sensivel simultaneamente a AMPLITUDE e a FREQUENCIA instantaneas do
sinal: para uma senoide pura x[n] = A*cos(w*n), psi(x[n]) ~= A^2 * w^2
(constante). Ou seja, um aumento na frequencia do sinal (tipico de
recrutamento de mais unidades motoras / disparo mais rapido durante
contracao muscular) e amplificado pelo TKEO mesmo sem mudanca de amplitude
pico-a-pico -- e a motivacao classica de seu uso em deteccao de burst de
EMG de superficie (ver revisao de literatura: Teager-Kaiser aplicado como
pre-processador em detectores automaticos de onset/offset muscular).

Por ser essencialmente uma segunda derivada discreta, o TKEO amplifica
ruido de alta frequencia -- por isso o pipeline padrao e:
  1. (opcional) filtro passa-faixa no EMG bruto
  2. psi[n] = x[n]^2 - x[n-1]*x[n+1]                (operador TKEO)
  3. suavizacao (media movel curta) do psi[n]        (reduz ruido residual)
  4. limiar (aqui: limiar duplo/histerese, igual ao teste 1, sobre uma
     baseline local do psi suavizado) -> segmentos -> classificacao por
     duracao (identica ao teste 1: fasico 0.5-5s, tonico >=16s)

--------------------------------------------------------------------------
LIMITACAO CONHECIDA DO TESTE (avisar antes de interpretar os resultados)
--------------------------------------------------------------------------
O ganho classico do TKEO vem do conteudo de FREQUENCIA do sinal (aumento de
frequencia de disparo das unidades motoras durante contracao). O gerador
sintetico deste projeto (testes/generate_synthetic_data.py) modela o burst
como envelope_alto * |ruido_gaussiano_branco| -- ou seja, a "portadora" e
ruido branco de banda larga tanto dentro quanto fora do evento, sem
deslocamento espectral real entre repouso e ativacao. Isso significa que o
TKEO, neste sinal sintetico especifico, so pode ganhar sobre o envelope RMS
pela componente de AMPLITUDE (que ja e capturada por ambos os metodos), nao
pela componente de frequencia que e sua vantagem teorica principal em EMG
real. Portanto: um resultado de "TKEO praticamente empatado com RMS" neste
teste NAO refuta o metodo -- refuta apenas a capacidade do gerador
sintetico atual de diferenciar amplitude de frequencia. Isso e reportado
explicitamente na avaliacao, nao omitido.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FS = 100
EPOCH_SEC = 3.0

# --- parametros de deteccao (mesma escala do teste 1, para comparabilidade) ---
BASELINE_WIN_S = 120.0
BASELINE_PCT = 10
MERGE_GAP_S = 1.0
TKEO_SMOOTH_WIN_S = 0.5     # janela de suavizacao do psi[n] -- MAIOR que a do envelope RMS (0.1s)
                            # porque o TKEO (~2a derivada discreta) tem cauda muito mais pesada em
                            # ruido de alta frequencia (curtose ~25 vs ~10 do RMS na mesma janela);
                            # 0.1s deixa picos de ruido cruzarem k_on isoladamente, gerando ~3000 FP
                            # falsos-fasicos nos 10 exames sinteticos (F1 fasico caindo a 0.31).
                            # 0.5s foi escolhido por varredura (0.1/0.3/0.5/1.0s): reduz a cauda
                            # pesada o suficiente para eliminar a fragmentacao (F1 fasico 0.997)
                            # sem borrar eventos fasicos curtos (que vao de 0.5 a 4.5s -- 1.0s de
                            # suavizacao ja comeca a fundir/perder os mais curtos, recall cai a 0.89).
K_ON = 3.0                  # limiar duplo: corte de INICIO
K_OFF = 1.5                 # limiar duplo: corte de MANUTENCAO

# --- classificacao por duracao + amplitude (cortes atualizados pelo usuario,
# identicos aos usados em testes/src/limiar/threshold_rule.py apos a revisao
# em exames reais) ---
PHASIC_LO_S = 0.1
PHASIC_HI_S = 5.0
TONIC_MIN_DUR_S = 15.0          # faixa "any" (ambigua) fica entre PHASIC_HI_S e TONIC_MIN_DUR_S
MIN_AMPLITUDE_RATIO = 2.0        # score (pico/baseline) minimo para contar como evento de qualquer tipo


def teager_kaiser_operator(x: np.ndarray) -> np.ndarray:
    """psi[n] = x[n]^2 - x[n-1]*x[n+1]. Mesmo comprimento da entrada
    (bordas replicadas por padding 'edge' -- unica escolha razoavel, ja
    que nao ha amostra fora do sinal para operar nos extremos)."""
    xp = np.pad(x.astype(np.float64), 1, mode="edge")
    return xp[1:-1] ** 2 - xp[:-2] * xp[2:]


def smooth(x: np.ndarray, win_sec: float, fs: int = FS) -> np.ndarray:
    win = max(1, int(round(win_sec * fs)))
    kernel = np.ones(win) / win
    return np.convolve(x, kernel, mode="same")


def rolling_baseline(sig: np.ndarray, win_sec: float = BASELINE_WIN_S,
                      pct: float = BASELINE_PCT, fs: int = FS) -> np.ndarray:
    """Baseline local por percentil, identica em forma ao teste 1 (blocos +
    interpolacao) -- aplicada aqui ao psi[n] suavizado em vez de ao envelope
    RMS, para isolar o efeito do pre-processamento TKEO."""
    win = max(1, int(round(win_sec * fs)))
    n = len(sig)
    half = win // 2
    step = max(1, win // 4)
    edges = list(range(0, n, step)) + [n]
    block_vals, block_centers = [], []
    for s in edges[:-1]:
        e = min(n, s + step)
        lo = max(0, s - half)
        hi = min(n, e + half)
        block_vals.append(np.percentile(sig[lo:hi], pct))
        block_centers.append((s + e) / 2)
    return np.interp(np.arange(n), block_centers, block_vals)


def merge_gaps(mask: np.ndarray, gap_samples: int) -> np.ndarray:
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


def double_threshold_mask(sig: np.ndarray, baseline: np.ndarray,
                           k_on: float = K_ON, k_off: float = K_OFF) -> np.ndarray:
    """Identico em logica ao teste 1 (histerese): ativa em k_on*baseline,
    so desativa em k_off*baseline."""
    above_off = sig > (k_off * baseline)
    above_on = sig > (k_on * baseline)
    off_segs = segments_from_mask(above_off)
    mask = np.zeros_like(above_off)
    for s, e in off_segs:
        if np.any(above_on[s:e]):
            mask[s:e] = True
    return mask


@dataclass
class DetectedEvent:
    onset_s: float
    duration_s: float
    type: str
    score: float


def segments_to_events(segs: list[tuple[int, int]], sig: np.ndarray, baseline: np.ndarray,
                        fs: int = FS, phasic_lo_s: float = PHASIC_LO_S,
                        phasic_hi_s: float = PHASIC_HI_S,
                        tonic_min_dur_s: float = TONIC_MIN_DUR_S,
                        min_amplitude_ratio: float = MIN_AMPLITUDE_RATIO) -> list[DetectedEvent]:
    """Classificacao por duracao (fasico/any/tonico) + filtro de amplitude:
    score = pico do sinal de energia / media do baseline local dentro do
    segmento; segmentos com score < min_amplitude_ratio sao descartados
    (nao contam como evento de nenhum tipo), mesma regra de
    testes/src/limiar/threshold_rule.py."""
    events = []
    for s, e in segs:
        dur_s = (e - s) / fs
        if dur_s < phasic_lo_s:
            continue
        score = float(np.max(sig[s:e]) / max(np.mean(baseline[s:e]), 1e-12))
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


def tkeo_energy_signal(emg_flat: np.ndarray, smooth_win_s: float = TKEO_SMOOTH_WIN_S,
                        fs: int = FS) -> np.ndarray:
    """psi[n] suavizado -- a serie usada para limiar/baseline no metodo TKEO."""
    psi = teager_kaiser_operator(emg_flat)
    psi = np.clip(psi, 0, None)   # TKEO pode dar negativo com ruido; energia fisica e >=0
    return smooth(psi, win_sec=smooth_win_s, fs=fs)


def detect_events_tkeo(emg_flat: np.ndarray, fs: int = FS, smooth_win_s: float = TKEO_SMOOTH_WIN_S,
                        apply_merge_gaps: bool = True) -> list[DetectedEvent]:
    """Pipeline TKEO completo: TKEO -> suavizacao -> baseline local ->
    limiar duplo/histerese -> fusao de lacunas curtas (opcional) ->
    segmentos -> classificacao por duracao."""
    energy = tkeo_energy_signal(emg_flat, smooth_win_s=smooth_win_s, fs=fs)
    baseline = rolling_baseline(energy, win_sec=BASELINE_WIN_S, pct=BASELINE_PCT, fs=fs)
    mask = double_threshold_mask(energy, baseline, k_on=K_ON, k_off=K_OFF)
    if apply_merge_gaps:
        mask = merge_gaps(mask, gap_samples=int(round(MERGE_GAP_S * fs)))
    segs = segments_from_mask(mask)
    return segments_to_events(segs, energy, baseline, fs=fs)


# --------------------------------------------------------------------------
# Baseline de comparacao: envelope RMS + limiar duplo (identico ao "vencedor"
# do teste 1), reimplementado aqui (nao importado de testes/src/limiar/) para
# que esta pasta permaneca autocontida e o comparativo use exatamente o
# mesmo pos-processamento (histerese + merge_gaps + classificacao por
# duracao) que o pipeline TKEO acima -- isolando a UNICA variavel que muda
# entre os dois testes: qual sinal de energia alimenta o limiar.
# --------------------------------------------------------------------------

def rms_envelope(x: np.ndarray, win_sec: float = 0.1, fs: int = FS) -> np.ndarray:
    win = max(1, int(round(win_sec * fs)))
    x2 = x.astype(np.float64) ** 2
    kernel = np.ones(win) / win
    ms = np.convolve(x2, kernel, mode="same")
    return np.sqrt(ms)


def detect_events_rms(emg_flat: np.ndarray, fs: int = FS,
                       apply_merge_gaps: bool = True) -> list[DetectedEvent]:
    """Mesmo pipeline de deteccao do metodo TKEO acima, mas alimentado pelo
    envelope RMS (0.1s) em vez do TKEO -- baseline de comparacao direta."""
    env = rms_envelope(emg_flat, win_sec=0.1, fs=fs)
    baseline = rolling_baseline(env, win_sec=BASELINE_WIN_S, pct=BASELINE_PCT, fs=fs)
    mask = double_threshold_mask(env, baseline, k_on=K_ON, k_off=K_OFF)
    if apply_merge_gaps:
        mask = merge_gaps(mask, gap_samples=int(round(MERGE_GAP_S * fs)))
    segs = segments_from_mask(mask)
    return segments_to_events(segs, env, baseline, fs=fs)
