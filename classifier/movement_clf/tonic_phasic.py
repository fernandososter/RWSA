"""
Sub-classificacao fasico/tonico-candidato dentro de trechos de movimento.

Modulo isolado (nao importa nada de src/sleep_rswa). Recebe o EMG bruto do
mento (1 canal, fs=100Hz) e um mask de movimento por mini-epoca (vindo da
MovementCNN) e aplica uma regra DETERMINISTICA de duracao+amplitude,
calculada sobre o envelope RMS do sinal bruto (nao sobre o score da CNN --
o score tem resolucao temporal insuficiente porque a janela de contexto de
`window_epochs` mini-epocas "vaza" score para epocas vizinhas e infla a
duracao aparente de qualquer evento em ~100x; ver discussao no relatorio).

Design validado com dados sinteticos de ground-truth garantido por
construcao (nao com os rotulos revisados manualmente, que podem ter
inconsistencia entre revisores):

  - FASICO (burst 0.5-5s acima de k x baseline local): rotulo FINAL,
    determinstico. Precisao/recall ~0.90/0.89 vs. tonico e trem-de-fasico.
  - CANDIDATO A TONICO (segmento >=16s, cobre mais da metade de uma epoca
    de 30s -- OU segmento na "zona morta" 5-16s, ver abaixo -- OU trecho
    de movimento (CNN) sem segmento de ativacao RMS suficiente, ver
    `ensure_movement_coverage`): NUNCA e rotulo final. A regra nao
    consegue, matematicamente, distinguir tonus tonico real de artefato
    de movimento sustentado (ex.: o paciente se mexendo, tossindo,
    ajustando a posicao) -- ambos tem a MESMA estatistica de duracao e
    amplitude por definicao. Testado contra dados sinteticos: 95.9% do
    artefato sustentado tambem cai nessa faixa. Por isso, candidato-tonico
    sempre exige confirmacao visual do revisor antes de virar rotulo
    tonico.

CORRECAO (2026-08-03): a versao original desta regra DESCARTAVA
silenciosamente (a) qualquer segmento de ativacao com duracao entre
phasic_hi_s (5s) e tonic_min_dur_s (16s) -- "zona morta" -- e (b) qualquer
trecho marcado como movimento pela CNN em que o envelope RMS nunca cruza
o limiar por tempo suficiente para gerar um segmento de ativacao. Medido
em dados reais (rbd1): 11.5% dos eventos do CSV primario de movimento nao
tinham NENHUMA linha correspondente no CSV tonico/fasico -- silenciosamente
ausentes da revisao, nao apenas mal classificados. Ambos os casos agora
sao sempre emitidos como 'tonic_candidate' (needs_review=True) em vez de
descartados: (a) via classify_tonic_phasic (zona morta reclassificada) e
(b) via ensure_movement_coverage (garante cobertura total do movement_mask
da CNN, cobrindo o gap de sensibilidade entre CNN e a regra de amplitude).
O objetivo NAO e classificar corretamente esses casos residuais -- e
garantir que nenhum evento do CSV primario desapareca silenciosamente do
CSV de sub-classificacao; o revisor sempre ve os dois.

O uso pretendido restringe a regra aos trechos ja sinalizados como
movimento pela MovementCNN (o localizador grosseiro "aqui ha movimento"),
e so entao aplica a regra de duracao sobre o envelope bruto dentro (e ao
redor de) cada trecho -- a CNN filtra onde olhar, a regra decide o que e.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

FS = 100
EPOCH_SEC = 3.0
SAMPLES_PER_EPOCH = 300

# Parametros da regra, validados em dados sinteticos (ver RELATORIO)
K_THRESH = 2.5          # limiar de ativacao = k x baseline local
BASELINE_WIN_S = 120.0  # janela rolante para o baseline local (p10)
BASELINE_PCT = 10       # percentil usado como baseline dentro da janela
MERGE_GAP_S = 1.0       # funde lacunas <= isso (ruido de micro-flutuacao)
PHASIC_LO_S = 0.5
PHASIC_HI_S = 5.0
TONIC_MIN_DUR_S = 16.0  # > metade de uma epoca de 30s (15s) + margem


def rms_envelope(x: np.ndarray, win_sec: float = 0.1, fs: int = FS) -> np.ndarray:
    """Envelope RMS de janela deslizante (mesmo comprimento da entrada)."""
    win = max(1, int(round(win_sec * fs)))
    x2 = x.astype(np.float64) ** 2
    kernel = np.ones(win) / win
    ms = np.convolve(x2, kernel, mode="same")
    return np.sqrt(ms)


def rolling_baseline(env: np.ndarray, win_sec: float = BASELINE_WIN_S,
                      pct: float = BASELINE_PCT, fs: int = FS) -> np.ndarray:
    """Baseline local: percentil `pct` dentro de uma janela rolante de win_sec.

    Usa baseline local (nao o percentil da noite inteira) porque o baseline
    global mistura o tonus de repouso real com os raros instantes de
    silencio absoluto da noite toda, gerando um limiar artificialmente
    baixo que nao separa tonico de artefato/fasico (ver RELATORIO).
    """
    win = max(1, int(round(win_sec * fs)))
    n = len(env)
    baseline = np.empty(n, dtype=np.float64)
    half = win // 2
    # implementacao simples por blocos (rapida o suficiente para uso em
    # inferencia de um exame por vez; ~horas de sinal a 100Hz)
    step = max(1, win // 4)
    edges = list(range(0, n, step)) + [n]
    block_vals = []
    block_centers = []
    for s in edges[:-1]:
        e = min(n, s + step)
        lo = max(0, s - half)
        hi = min(n, e + half)
        block_vals.append(np.percentile(env[lo:hi], pct))
        block_centers.append((s + e) / 2)
    baseline = np.interp(np.arange(n), block_centers, block_vals)
    return baseline


def _merge_gaps(mask: np.ndarray, gap_samples: int) -> np.ndarray:
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


def _segments_from_mask(mask: np.ndarray):
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


@dataclass
class DurationRuleConfig:
    k_thresh: float = K_THRESH
    baseline_win_s: float = BASELINE_WIN_S
    baseline_pct: float = BASELINE_PCT
    merge_gap_s: float = MERGE_GAP_S
    phasic_lo_s: float = PHASIC_LO_S
    phasic_hi_s: float = PHASIC_HI_S
    tonic_min_dur_s: float = TONIC_MIN_DUR_S
    fs: int = FS


def classify_tonic_phasic(emg_flat: np.ndarray, cfg: DurationRuleConfig | None = None):
    """Aplica a regra deterministica de duracao+amplitude ao EMG bruto inteiro.

    emg_flat: sinal EMG continuo [N] amostras (fs=100Hz por padrao), ja no
    referencial temporal do exame inteiro (concatenacao das mini-epocas).

    Devolve lista de dicts: {onset_s, duration_s, type, peak_ratio}
      type = "phasic"          -> rotulo final
      type = "tonic_candidate" -> NUNCA final, precisa revisao humana

    Nenhum segmento de ativacao (RMS >= k x baseline, apos fusao de
    lacunas curtas) e descartado: fora da faixa fasico (0.5-5s) e fora da
    faixa tonico-candidato (>=16s) cai na "zona morta" (5-16s), que
    tambem vira 'tonic_candidate' -- ver nota de correcao no docstring do
    modulo. Isso e deliberadamente conservador (nao classifica bem a zona
    morta), mas garante que a duracao do segmento nunca some para zero
    linhas no CSV de saida.
    """
    cfg = cfg or DurationRuleConfig()
    fs = cfg.fs
    env = rms_envelope(emg_flat, fs=fs)
    baseline = rolling_baseline(env, win_sec=cfg.baseline_win_s, pct=cfg.baseline_pct, fs=fs)
    baseline = np.maximum(baseline, 1e-8)

    above = env >= (cfg.k_thresh * baseline)
    gap_samples = int(round(cfg.merge_gap_s * fs))
    fused = _merge_gaps(above, gap_samples)

    events = []
    for s, e in _segments_from_mask(fused):
        dur_s = (e - s) / fs
        onset_s = s / fs
        peak_ratio = float(np.max(env[s:e]) / np.mean(baseline[s:e]))
        if dur_s < cfg.phasic_lo_s:
            # sub-0.5s: ruido/micro-flutuacao, nao um evento de movimento.
            # Descartado aqui de proposito -- restrict_to_movement() ainda
            # pode gerar um tonic_candidate de cobertura via
            # ensure_movement_coverage() se isso deixar um trecho de
            # movimento (CNN) sem NENHUM segmento de ativacao suficiente.
            continue
        if cfg.phasic_lo_s <= dur_s <= cfg.phasic_hi_s:
            etype = "phasic"
        else:
            # >=16s (candidato-tonico "classico") OU na zona morta 5-16s
            # (nem fasico nem tonico-candidato "classico" pela duracao, mas
            # nao pode ser descartado -- ver nota de correcao no docstring
            # do modulo). Ambos caem em tonic_candidate, sempre com revisao.
            etype = "tonic_candidate"
        events.append({
            "onset_s": round(float(onset_s), 3),
            "duration_s": round(float(dur_s), 3),
            "type": etype,
            "peak_ratio": round(peak_ratio, 3),
        })
    return events


def restrict_to_movement(events: list[dict], movement_mask: np.ndarray,
                          epoch_sec: float = EPOCH_SEC,
                          min_overlap_frac: float = 0.3) -> list[dict]:
    """Mantem so eventos que se sobrepoem a mini-epocas marcadas como
    movimento pela MovementCNN (localizador grosseiro).

    movement_mask: [T] bool/0-1, uma entrada por mini-epoca de 3s.
    min_overlap_frac: fracao minima da duracao do evento que precisa cair
    dentro de mini-epocas de movimento para o evento ser mantido.
    """
    movement_mask = np.asarray(movement_mask).astype(bool)
    T = len(movement_mask)
    kept = []
    for ev in events:
        e0 = ev["onset_s"] / epoch_sec
        e1 = (ev["onset_s"] + ev["duration_s"]) / epoch_sec
        m0 = max(0, int(np.floor(e0)))
        m1 = min(T, int(np.ceil(e1)))
        if m1 <= m0:
            continue
        overlap_epochs = movement_mask[m0:m1]
        if overlap_epochs.size == 0:
            continue
        if overlap_epochs.mean() >= min_overlap_frac:
            kept.append(ev)
    return kept


def _epoch_runs(mask: np.ndarray):
    """Runs contiguos (m0, m1) [indices de mini-epoca, m1 exclusivo] onde mask e True."""
    n = len(mask)
    runs = []
    i = 0
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            runs.append((i, j + 1))
            i = j + 1
        else:
            i += 1
    return runs


def ensure_movement_coverage(kept_events: list[dict], movement_mask: np.ndarray,
                              epoch_sec: float = EPOCH_SEC,
                              min_overlap_frac: float = 0.3) -> list[dict]:
    """Garante que TODO trecho de movimento sinalizado pela CNN tenha pelo
    menos um evento correspondente no CSV de sub-classificacao.

    Motivacao (ver nota de correcao no docstring do modulo): a regra de
    duracao+amplitude pode nao gerar NENHUM segmento de ativacao dentro de
    um trecho que a CNN marcou como movimento (o pico do envelope RMS pode
    nunca cruzar k x baseline por tempo suficiente, mesmo com score da CNN
    alto). Sem esta funcao, esses trechos desaparecem silenciosamente do
    CSV tonico/fasico -- o revisor nunca os ve, mesmo estando no CSV
    primario. Aqui, cada run continuo de mini-epocas de movimento que nao
    tem NENHUM evento de `kept_events` sobrepondo-o (mesmo criterio de
    overlap de restrict_to_movement) recebe um evento 'tonic_candidate' de
    cobertura, com needs_review=True e peak_ratio=None (sinaliza que e um
    placeholder de cobertura, nao uma estimativa real de amplitude --
    o revisor decide visualmente o que e).
    """
    movement_mask = np.asarray(movement_mask).astype(bool)
    T = len(movement_mask)
    covered = np.zeros(T, dtype=bool)
    for ev in kept_events:
        e0 = ev["onset_s"] / epoch_sec
        e1 = (ev["onset_s"] + ev["duration_s"]) / epoch_sec
        m0 = max(0, int(np.floor(e0)))
        m1 = min(T, int(np.ceil(e1)))
        if m1 > m0:
            covered[m0:m1] = True

    extra = []
    for m0, m1 in _epoch_runs(movement_mask):
        run_len = m1 - m0
        covered_frac = covered[m0:m1].mean() if run_len else 1.0
        if covered_frac >= min_overlap_frac:
            continue
        extra.append({
            "onset_s": round(float(m0 * epoch_sec), 3),
            "duration_s": round(float(run_len * epoch_sec), 3),
            "type": "tonic_candidate",
            "peak_ratio": None,
        })
    return kept_events + extra
