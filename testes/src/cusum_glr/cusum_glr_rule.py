"""
Teste 3: testes estatisticos sequenciais de deteccao de ponto de mudanca
(CUSUM de Page e uma aproximacao de GLR multi-escala) para isolar e
classificar eventos tonicos/fasicos no EMG.

Modulo ISOLADO dentro de testes/ -- nao importa nada de classifier/,
src/sleep_rswa/ nem dos testes anteriores (testes/src/limiar/,
testes/src/tkeo/). Reimplementa o envelope RMS e a baseline robusta
localmente para que esta pasta possa ser lida e rodada de forma
independente.

--------------------------------------------------------------------------
POR QUE ESTA FAMILIA E DIFERENTE DAS DUAS ANTERIORES
--------------------------------------------------------------------------
Limiar simples/duplo (teste 1) e TKEO (teste 2) decidem amostra-a-amostra
se o envelope esta "alto" comparado a uma referencia local. Nenhum dos dois
faz um teste de hipotese explicito sobre SE HOUVE MUDANCA DE REGIME -- eles
comparam um NIVEL a um CORTE.

CUSUM e GLR fazem a pergunta operacionalmente diferente e mais proxima da
definicao real de "tonico": "a media do sinal mudou de regime, de forma
SUSTENTADA, em algum ponto recente?" -- acumulando evidencia estatistica ao
longo do tempo em vez de julgar cada amostra isoladamente. Isso e a
motivacao da literatura citada na revisao deste projeto: CUSUM com modelo
de Markov oculto foi usado para detectar mudancas de tendencia sustentada
em sinal fisiologico continuo (pressao arterial), estruturalmente
equivalente a separar "elevacao de tonus sustentada" de flutuacao de fundo.

--------------------------------------------------------------------------
VARIANTE 1 -- CUSUM DE PAGE (1954), com baseline robusta local
--------------------------------------------------------------------------
1. Envelope RMS do EMG (mesma janela de 0.1s dos testes 1 e 2).
2. Baseline robusta local: mediana (mu0) e desvio robusto via MAD
   (sigma0 = 1.4826 * MAD, escala consistente com o desvio-padrao para
   dados gaussianos) numa janela rolante de 120s -- mesma janela e mesma
   ideia de robustez a contaminacao por eventos vizinhos ja usada nos
   testes 1 e 2 (percentil 10), agora com um estimador de dispersao
   explicito, que o CUSUM exige.
3. Inovacao padronizada: z[n] = (env[n] - mu0[n]) / sigma0[n].
4. Estatistica de Page (CUSUM unilateral, para detectar so ELEVACOES,
   que e o que define um evento EMG):
       g[n] = max(0, g[n-1] + z[n] - k)
   `k` (folga, em unidades de sigma) e o parametro classico de Page: por
   convencao, ajustado a aproximadamente a metade do menor desvio que se
   quer detectar de forma otima. `h` (limiar de decisao, em unidades de g)
   controla o trade-off classico deteccao-tardia vs. falso-alarme (ARL).
5. Deteccao: um segmento se INICIA no ultimo instante em que g[n] esteve
   em zero antes de cruzar h (retrocesso ao ponto de mudanca real -- o
   CUSUM so ALARMA com atraso, mas o proprio valor acumulado aponta para
   tras o inicio real da mudanca). O segmento so TERMINA quando g[n]
   retorna a zero -- este e um mecanismo de histerese *automatico*, sem
   parametro extra k_off: enquanto a media do sinal nao voltar a baseline,
   as inovacoes negativas nao conseguem zerar o acumulador tao rapido
   quanto um unico cruzamento de limiar simples faria.

--------------------------------------------------------------------------
VARIANTE 2 -- GLR MULTI-ESCALA (aproximacao computavel)
--------------------------------------------------------------------------
O GLR teorico para deteccao de mudanca de media com magnitude e instante
DESCONHECIDOS, dentro de uma janela de M amostras terminando em n, e:
    Lambda[n] = max_{1<=m<=M} ( sum_{i=n-m+1}^{n} z[i] )^2 / (2*m)
-- o maximo, sobre todos os tamanhos de sub-janela m, da estatistica de
razao de verossimilhança generalizada para "as ultimas m amostras tem media
diferente de zero" (z ja padronizado localmente, entao z ~ N(0,1) sob H0).

Calcular isso para TODO m em 1..M a cada amostra e caro (O(n*M)); a solucao
pratica adotada aqui -- e documentada como tal, nao escondida -- e avaliar
o maximo apenas num conjunto pequeno de escalas espacadas geometricamente
(ex.: 0.1, 0.2, 0.4, 0.8, ..., ~3s), a mesma logica de "deteccao
multi-resolucao" que a revisao de literatura ja apontou para variantes
recentes do TKEO. Isso e uma aproximacao ao GLR completo, nao o GLR exato:
captura mudancas de duracao proxima a qualquer uma das escalas testadas,
mas pode ser levemente subotimo para duracoes muito entre duas escalas
consecutivas.

Ao contrario do CUSUM (que tem histerese automatica via seu proprio reset),
o GLR aqui usa limiar SIMPLES sobre Lambda[n] (sem k_off separado) -- o
proprio agregamento em janela ja suaviza o estatistico o suficiente para
nao fragmentar como o limiar simples de amostra-a-amostra do teste 1
(hipotese a verificar empiricamente na avaliacao, nao assumida).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FS = 100
EPOCH_SEC = 3.0

# --- baseline robusta (mesma janela dos testes 1 e 2, estimador diferente) ---
BASELINE_WIN_S = 120.0
MERGE_GAP_S = 1.0

# --- CUSUM de Page (classico) ---
# NOTA: varredura em grade (k em [0.3,10], h em [1,8], 4 exames) NAO encontrou
# nenhuma combinacao que evite o blob degenerado (ver VARIANTE 3 abaixo) com
# F1 aceitavel simultaneamente para fasico e tonico -- este e o melhor
# compromisso qualitativo encontrado (ainda com blob), usado para
# DOCUMENTAR a falha estrutural na avaliacao, nao como configuracao
# recomendada (usar CUSUM_LEAKY_* para uso real).
CUSUM_K = 0.75          # folga em unidades de sigma (~metade do menor desvio "alvo")
CUSUM_H = 6.0           # limiar de decisao em unidades de g (trade-off deteccao/falso-alarme)

# --- GLR multi-escala ---
# NOTA: mesma limitacao estrutural do CUSUM classico (fusao de clusters densos),
# atenuada mas nao eliminada pela janela finita (ate 3s) -- ver avaliacao.
GLR_SCALES_S = [0.1, 0.2, 0.4, 0.8, 1.6, 3.0]   # escalas geometricas, ~0.1 a 3s
GLR_H = 9.0             # limiar de decisao sobre Lambda[n] -- valor com melhor F1 tonico (0.98)
                        # na varredura, AINDA com blob (max ~800s); ver nota acima e avaliacao.

# --- classificacao por duracao + amplitude (cortes atualizados pelo usuario,
# identicos aos usados em testes/src/limiar/threshold_rule.py e
# testes/src/tkeo/tkeo_rule.py apos a revisao em exames reais) ---
PHASIC_LO_S = 0.1
PHASIC_HI_S = 5.0
TONIC_MIN_DUR_S = 15.0          # faixa "any" (ambigua) fica entre PHASIC_HI_S e TONIC_MIN_DUR_S
MIN_AMPLITUDE_RATIO = 2.0        # score (pico do envelope / media do baseline local) minimo


def rms_envelope(x: np.ndarray, win_sec: float = 0.1, fs: int = FS) -> np.ndarray:
    win = max(1, int(round(win_sec * fs)))
    x2 = x.astype(np.float64) ** 2
    kernel = np.ones(win) / win
    ms = np.convolve(x2, kernel, mode="same")
    return np.sqrt(ms)


def robust_rolling_baseline(env: np.ndarray, win_sec: float = BASELINE_WIN_S,
                             fs: int = FS) -> tuple[np.ndarray, np.ndarray]:
    """Baseline robusta local: mediana (mu0) e desvio via MAD (sigma0),
    em blocos + interpolacao (mesma estrutura de implementacao dos testes
    1 e 2, agora estimando tambem a dispersao, exigida pelo CUSUM/GLR)."""
    win = max(1, int(round(win_sec * fs)))
    n = len(env)
    half = win // 2
    step = max(1, win // 4)
    edges = list(range(0, n, step)) + [n]
    mu_vals, sigma_vals, centers = [], [], []
    for s in edges[:-1]:
        e = min(n, s + step)
        lo = max(0, s - half)
        hi = min(n, e + half)
        block = env[lo:hi]
        med = np.median(block)
        mad = np.median(np.abs(block - med))
        sigma = 1.4826 * mad
        mu_vals.append(med)
        sigma_vals.append(max(sigma, 1e-12))   # evita divisao por zero em blocos totalmente planos
        centers.append((s + e) / 2)
    mu0 = np.interp(np.arange(n), centers, mu_vals)
    sigma0 = np.interp(np.arange(n), centers, sigma_vals)
    return mu0, sigma0


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


@dataclass
class DetectedEvent:
    onset_s: float
    duration_s: float
    type: str
    score: float


def segments_to_events(segs: list[tuple[int, int]], env: np.ndarray, baseline: np.ndarray,
                        fs: int = FS, phasic_lo_s: float = PHASIC_LO_S,
                        phasic_hi_s: float = PHASIC_HI_S,
                        tonic_min_dur_s: float = TONIC_MIN_DUR_S,
                        min_amplitude_ratio: float = MIN_AMPLITUDE_RATIO) -> list[DetectedEvent]:
    """Classificacao por duracao (fasico/any/tonico) + filtro de amplitude:
    score = pico do ENVELOPE RMS / media do baseline robusto (mu0) local
    dentro do segmento -- nao mais o proprio estatistico CUSUM/GLR (g ou
    Lambda), para que o campo `score` seja comparavel entre os tres testes
    (limiar/tkeo/cusum_glr), conforme pedido pelo usuario. Segmentos com
    score < min_amplitude_ratio sao descartados (nao contam como evento de
    nenhum tipo)."""
    events = []
    for s, e in segs:
        dur_s = (e - s) / fs
        if dur_s < phasic_lo_s:
            continue
        score = float(np.max(env[s:e]) / max(np.mean(baseline[s:e]), 1e-12))
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


def cusum_page_statistic(z: np.ndarray, k: float = CUSUM_K) -> np.ndarray:
    """g[n] = max(0, g[n-1] + z[n] - k) -- recursao classica de Page.
    Implementacao sequencial (nao vetorizavel por natureza: g depende do
    proprio passado), mas O(n) e rapida o suficiente em numpy puro."""
    g = np.empty_like(z)
    acc = 0.0
    for i in range(len(z)):
        acc = max(0.0, acc + z[i] - k)
        g[i] = acc
    return g


def cusum_mask_with_backtrack(g: np.ndarray, h: float = CUSUM_H) -> np.ndarray:
    """Segmento ATIVO desde o ultimo zero de g antes do alarme (retrocesso
    ao ponto de mudanca real) at o proximo retorno a zero (recovery
    automatico -- ver docstring do modulo)."""
    n = len(g)
    mask = np.zeros(n, dtype=bool)
    above = g > h
    if not above.any():
        return mask
    zero = g <= 1e-9
    # para cada instante, indice do ultimo zero <= este instante (busca vetorizada via forward-fill)
    last_zero_idx = np.where(zero, np.arange(n), -1)
    np.maximum.accumulate(last_zero_idx, out=last_zero_idx)
    alarm_segs = segments_from_mask(above)
    for s, e in alarm_segs:
        onset = last_zero_idx[s] + 1 if last_zero_idx[s] >= 0 else 0
        # fim: proximo zero de g apos o alarme (ou fim do sinal, se nunca voltar a zero)
        tail = np.where(zero[e:])[0]
        end = e + tail[0] + 1 if len(tail) > 0 else n
        mask[onset:end] = True
    return mask


def detect_events_cusum(emg_flat: np.ndarray, fs: int = FS,
                         k: float = CUSUM_K, h: float = CUSUM_H,
                         apply_merge_gaps: bool = True) -> list[DetectedEvent]:
    env = rms_envelope(emg_flat, win_sec=0.1, fs=fs)
    mu0, sigma0 = robust_rolling_baseline(env, win_sec=BASELINE_WIN_S, fs=fs)
    z = (env - mu0) / sigma0
    g = cusum_page_statistic(z, k=k)
    mask = cusum_mask_with_backtrack(g, h=h)
    if apply_merge_gaps:
        mask = merge_gaps(mask, gap_samples=int(round(MERGE_GAP_S * fs)))
    segs = segments_from_mask(mask)
    return segments_to_events(segs, env, mu0, fs=fs)


def glr_multiscale_statistic(z: np.ndarray, scales_s: list[float] = GLR_SCALES_S,
                              fs: int = FS) -> np.ndarray:
    """Lambda[n] = max sobre um conjunto pequeno de escalas m (amostras) de
    (soma dos ultimos m valores de z)^2 / (2*m) -- aproximacao multi-escala
    do GLR completo (ver docstring do modulo). Vetorizado via cumsum:
    soma_{n-m+1..n} z = C[n] - C[n-m], com C = cumsum(z)."""
    n = len(z)
    C = np.concatenate([[0.0], np.cumsum(z)])   # C[i] = soma de z[0..i-1]
    lam = np.zeros(n)
    for scale_s in scales_s:
        m = max(1, int(round(scale_s * fs)))
        if m >= n:
            continue
        # para indice i (0-based, correspondente a amostra n=i), soma das ultimas m amostras
        # (indices i-m+1 .. i) = C[i+1] - C[i+1-m]
        diff = np.empty(n)
        diff[:m] = np.nan   # sem historico suficiente ainda -- fica de fora do max (tratado abaixo)
        diff[m:] = C[m + 1:n + 1] - C[1:n - m + 1]
        contrib = (diff ** 2) / (2 * m)
        contrib = np.nan_to_num(contrib, nan=0.0)
        lam = np.maximum(lam, contrib)
    return lam


def detect_events_glr(emg_flat: np.ndarray, fs: int = FS, h: float = GLR_H,
                       apply_merge_gaps: bool = True) -> list[DetectedEvent]:
    env = rms_envelope(emg_flat, win_sec=0.1, fs=fs)
    mu0, sigma0 = robust_rolling_baseline(env, win_sec=BASELINE_WIN_S, fs=fs)
    z = (env - mu0) / sigma0
    lam = glr_multiscale_statistic(z, scales_s=GLR_SCALES_S, fs=fs)
    mask = lam > h
    if apply_merge_gaps:
        mask = merge_gaps(mask, gap_samples=int(round(MERGE_GAP_S * fs)))
    segs = segments_from_mask(mask)
    return segments_to_events(segs, env, mu0, fs=fs)


# --------------------------------------------------------------------------
# VARIANTE 3 -- CUSUM "COM ESQUECIMENTO" (leaky/windowed CUSUM)
# --------------------------------------------------------------------------
# ACHADO EMPIRICO (documentado aqui, nao omitido): o CUSUM classico de Page
# (variante 1 acima) FALHA ESTRUTURALMENTE no cenario adversarial deste
# projeto (clusters de eventos com gaps curtos de 2-8s entre sub-eventos).
# A causa raiz nao e calibracao de k/h: e que a taxa de CRESCIMENTO do
# acumulador durante um evento forte (z medio ~1.3, aumentando g em
# ~1.0/amostra a 100Hz) e da mesma ordem de grandeza que a taxa de
# DECAIMENTO durante os gaps quietos entre sub-eventos do cluster (z medio
# ~-0.7 a -1.0/amostra) -- e os gaps de 2-8s nunca sao longos o bastante
# para "pagar" a divida acumulada durante um evento tonico de 16-45s. O
# resultado e que g cresce de forma praticamente monotonica ao longo de
# TODO o exame sintetico de 2h (verificado: g vai de 0 a >10^5 sem nunca
# retornar a zero), fundindo o exame inteiro num unico "evento" de
# duracao maior que a gravacao. Nenhuma combinacao de (k,h) testada
# (varredura k em [0.3,10], h em [1,8]) produziu deteccoes de duracao
# fisiologicamente plausivel (<90s) com F1 aceitavel simultaneamente para
# fasico e tonico -- ou o acumulador nunca reseta (blob gigante) ou k tao
# grande que o metodo perde sensibilidade a eventos fasicos curtos.
#
# A correcao classica de controle de processo para este problema e o CUSUM
# "com esquecimento" (leaky/windowed CUSUM, tambem descrito como hibrido
# EWMA-CUSUM na literatura de deteccao de mudanca): introduzir um fator de
# esquecimento rho<1 na recursao,
#     g[n] = max(0, rho * g[n-1] + z[n] - k)
# que limita a "memoria" efetiva do acumulador a aproximadamente
# 1/(1-rho) amostras -- por construcao, a divida acumulada decai
# geometricamente mesmo sem gaps quietos, resolvendo o problema estrutural
# acima sem abandonar a familia CUSUM. rho=1 recupera exatamente o CUSUM
# classico de Page (variante 1).
#
# Parametros escolhidos por varredura em grade (rho, k, h) num subconjunto
# de 4 exames sinteticos, validados depois nos 10 exames completos:
CUSUM_LEAKY_RHO = 0.95
CUSUM_LEAKY_K = 0.20
CUSUM_LEAKY_H = 2.5


def cusum_leaky_statistic(z: np.ndarray, k: float = CUSUM_LEAKY_K,
                           rho: float = CUSUM_LEAKY_RHO) -> np.ndarray:
    """g[n] = max(0, rho*g[n-1] + z[n] - k) -- CUSUM de Page com fator de
    esquecimento rho, ver docstring da secao acima. rho=1.0 recupera
    cusum_page_statistic exatamente."""
    g = np.empty_like(z)
    acc = 0.0
    for i in range(len(z)):
        acc = max(0.0, rho * acc + z[i] - k)
        g[i] = acc
    return g


def detect_events_cusum_leaky(emg_flat: np.ndarray, fs: int = FS,
                               k: float = CUSUM_LEAKY_K, h: float = CUSUM_LEAKY_H,
                               rho: float = CUSUM_LEAKY_RHO,
                               apply_merge_gaps: bool = True) -> list[DetectedEvent]:
    env = rms_envelope(emg_flat, win_sec=0.1, fs=fs)
    mu0, sigma0 = robust_rolling_baseline(env, win_sec=BASELINE_WIN_S, fs=fs)
    z = (env - mu0) / sigma0
    g = cusum_leaky_statistic(z, k=k, rho=rho)
    mask = cusum_mask_with_backtrack(g, h=h)
    if apply_merge_gaps:
        mask = merge_gaps(mask, gap_samples=int(round(MERGE_GAP_S * fs)))
    segs = segments_from_mask(mask)
    return segments_to_events(segs, env, mu0, fs=fs)
