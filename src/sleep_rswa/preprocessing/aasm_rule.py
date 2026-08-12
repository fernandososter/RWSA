"""
Regra RSWA conforme os criterios da AASM (2023a) para atividade muscular em
sono REM -- tonica, fasica e "any" (qualquer atividade).

Modulo ISOLADO dentro de src/sleep_rswa/preprocessing/: nao importa nada de
testes/src/limiar/ nem de classifier/. Reimplementa apenas o minimo
necessario (envelope RMS, deteccao de segmentos por limiar simples de
amplitude) para nao depender do estado do detector de producao
(auto_rswa.py / threshold_rule.py), que usa uma METODOLOGIA DIFERENTE
(limiar duplo/histerese contra baseline local rolante, classificacao por
segmento isolado). Este modulo implementa a definicao textual da AASM, para
poder medir o impacto de trocar de metodologia -- ver
docs/relatorio_impacto_regra_aasm.md.

Resumo dos 3 criterios (epoca R = epoca de 30s com estagio REM):

(i) TONICA (excessive sustained muscle activity):
    Epoca R em que a SOMA das duracoes de segmentos de amplitude >= 2x o
    nivel de atonia REM, cada segmento com duracao > 5s, cobre >= 50% da
    epoca (>= 15s de 30s). Multiplos segmentos >5s podem somar-se.
    Decisao e por EPOCA de 30s (nao por mini-epoca de 3s isolada).

(ii) FASICA (excessive transient muscle activity):
    Epoca R dividida em 10 mini-epocas de 3s. Fasica=1 na epoca se >=5 das
    10 mini-epocas (50%) contem pelo menos um burst de 0.1-5.0s de
    amplitude >= 2x o nivel de atonia REM. Decisao e por EPOCA de 30s.

(iii) ANY (any chin EMG activity):
    Qualquer atividade de amplitude >= 2x o nivel de atonia REM,
    independente da duracao (inclui segmentos de 5-15s que nao contam nem
    para tonico isolado nem fasico). E SUPERSET de tonico e fasico -- toda
    mini-epoca marcada tonic=1 ou phasic=1 tambem tem any=1, mais quaisquer
    mini-epocas com atividade de amplitude suficiente mas que nao fecham o
    criterio de nenhum dos dois (ex.: um unico burst de 8s isolado, sem
    atingir 50% da epoca de 30s nem caber na faixa fasica).
    Decisao e por MINI-EPOCA de 3s (nao ha agregacao por epoca de 30s aqui
    -- "any" e definido no nivel de atividade presente, nao de contagem).

Referencia de amplitude (nivel de atonia REM):
    - Preferencial: `rem_baseline_uv` do exame (percentil 10 do envelope RMS
      do EMG dentro das mini-epocas REM -- ja calculado e gravado no .pt por
      rem_baseline.compute_rem_baseline).
    - Fallback (quando o exame nao tem atonia detectavel em R -- ex. RSWA
      tonica contaminando a maior parte do REM): menor amplitude observada
      no NREM, aproximada aqui pelo percentil 10 do envelope RMS do EMG
      dentro das mini-epocas NREM (estagios N1/N2/N3), via
      compute_nrem_baseline_uv(). LIMITACAO: poucos exames no dataset atual
      tem REM sem nenhuma atonia detectavel para validar este fallback (ver
      relatorio de impacto).
"""
from __future__ import annotations

import numpy as np

FS = 100
EPOCH_SEC = 3.0                 # duracao da mini-epoca (mesma unidade do .pt)
SAMPLES_PER_EPOCH = 300
MACRO_EPOCH_SEC = 30.0          # duracao da epoca R da AASM
MINI_PER_MACRO = int(round(MACRO_EPOCH_SEC / EPOCH_SEC))  # 10

REM_STAGE = 4
NREM_STAGES = (1, 2, 3)          # N1, N2, N3 (W=0 e propositalmente excluido)

MIN_AMPLITUDE_RATIO = 2.0        # amplitude minima = 2x o nivel de atonia REM
                                  # (configuravel via min_amplitude_ratio em
                                  # apply_aasm_rule/label_exam_with_aasm_rule e
                                  # via aasm_min_amplitude_ratio em preprocess_exam/
                                  # --aasm-min-amplitude-ratio no CLI; ver varredura
                                  # empirica em docs/relatorio_impacto_regra_aasm.md)
TONIC_SEGMENT_MIN_S = 5.0        # cada segmento tonico somado deve ter > 5s
TONIC_EPOCH_COVERAGE = 0.5       # soma dos segmentos >5s deve cobrir >= 50% da epoca R
PHASIC_LO_S = 0.1
PHASIC_HI_S = 5.0
PHASIC_MIN_MINI_FRACTION = 0.5   # >= 5 de 10 mini-epocas com burst fasico

# Piso de duracao para o criterio "any". O texto da AASM (2023a) define
# "any chin EMG activity" SEM piso de duracao ("without regard to the
# duration of the activity"), mas isso, aplicado literalmente a um limiar
# simples de amplitude sobre o envelope RMS, faz "any" herdar 100% da
# sensibilidade do detector a ruido de linha de base: em dados reais,
# 50-66% dos segmentos brutos que cruzam o limiar de amplitude duram
# menos que PHASIC_LO_S (alguns de apenas 1 amostra = 10ms) -- oscilacao
# do envelope no ruido de fundo, nao atividade muscular visivel. Sem
# piso, isso faz "any" marcar como positivo qualquer trecho onde o
# tracado simplesmente sobe um pouco, sem corresponder a um evento real.
# Usamos o mesmo piso ja validado para o burst fasico (PHASIC_LO_S=0.1s)
# como corte de plausibilidade minima -- nao para reinterpretar o texto
# da AASM (que continua sem limite SUPERIOR aqui: um segmento de 5-15s,
# por exemplo, ainda conta como "any" mesmo nao contando isoladamente
# nem para tonico nem para fasico), apenas para excluir cruzamentos de
# limiar mais curtos do que a resolucao temporal minima de um burst real.
ANY_MIN_SEG_S = PHASIC_LO_S      # 0.1s -- piso de plausibilidade para "any"

# Tolerancia de gap para fusao de segmentos antes da classificacao tonic/
# phasic/any. O corte de limiar SIMPLES (sem histerese) usado em
# _segments_above_threshold fragmenta atividade continua real em muitos
# segmentos curtos sempre que o envelope RMS cruza o limiar por uma unica
# amostra de ruido (tipicamente 10-160ms) mesmo durante elevacao sustentada
# real -- confirmado em dados reais (rbd9: 52 segmentos de 10ms-3s, gaps
# todos <=160ms, somando 34.2s de 36s de janela). Isso faz atividade que
# deveria disparar TONICO (segmento continuo >5s) nunca acumular duracao
# suficiente, e em vez disso vazar como muitas mini-epocas FASICAS
# consecutivas -- um artefato de fragmentacao do detector, nao o padrao
# fisiologico real. RBDtector (Sarkar/Manoli 2022 -- ferramenta open-source
# que implementa a pontuacao visual SINBAR/AASM) documenta a mesma correcao:
# um intervalo de silencio de ate 250ms nao termina o evento. Fundimos aqui
# segmentos separados por gap <= MERGE_GAP_S ANTES de calcular duracoes para
# os 3 criterios -- isso nao e uma reinterpretacao do texto da AASM (que nao
# fala de fusao), e sim uma correcao de artefato de discretizacao do
# detector de segmentos, na mesma categoria do piso ANY_MIN_SEG_S acima.
MERGE_GAP_S = 0.25                # 250ms -- convencao RBDtector/SINBAR


def rms_envelope(x: np.ndarray, win_sec: float = 0.1, fs: int = FS) -> np.ndarray:
    """Envelope RMS de janela deslizante (mesmo comprimento da entrada).
    Identico a rms_envelope() em testes/src/limiar/threshold_rule.py e
    rem_envelope_rms() em rem_baseline.py -- mantida como copia local
    (nao importada) para preservar o isolamento entre modulos."""
    win = max(1, int(round(win_sec * fs)))
    x2 = x.astype(np.float64) ** 2
    kernel = np.ones(win) / win
    ms = np.convolve(x2, kernel, mode="same")
    return np.sqrt(ms)


def compute_nrem_baseline_uv(
    signals: np.ndarray,          # (T, N_CHANNELS, n_samples) float, unidades BRUTAS (V)
    sleep_stages: np.ndarray,     # (T,)
    *,
    emg_channel_index: int = 4,
    nrem_stages: tuple[int, ...] = NREM_STAGES,
    pct: float = 10.0,
    win_sec: float = 0.1,
    fs: int = FS,
    volts_to_microvolts: float = 1e6,
) -> tuple[float, int]:
    """Fallback do nivel de atonia quando o REM nao tem atonia detectavel:
    percentil `pct` do envelope RMS do EMG dentro das mini-epocas NREM
    (N1/N2/N3). Mesma estatistica/formula de rem_baseline.compute_rem_baseline,
    aplicada ao NREM em vez do REM.

    Retorna (baseline_uv, n_epochs_usadas). baseline_uv = nan se nao houver
    nenhuma mini-epoca NREM no exame.
    """
    sleep_stages = np.asarray(sleep_stages)
    nrem_mask = np.isin(sleep_stages, nrem_stages)
    n_nrem = int(nrem_mask.sum())
    if n_nrem == 0:
        return float("nan"), 0

    emg_nrem = np.asarray(signals)[nrem_mask, emg_channel_index, :].astype(np.float64)
    envelopes = [rms_envelope(row, win_sec=win_sec, fs=fs) for row in emg_nrem]
    env_concat = np.concatenate(envelopes)
    baseline_v = float(np.percentile(env_concat, pct))
    return baseline_v * volts_to_microvolts, n_nrem


def resolve_atonia_baseline_uv(
    rem_baseline_uv: float,
    rem_baseline_n_epochs: int,
    signals: np.ndarray,
    sleep_stages: np.ndarray,
    *,
    emg_channel_index: int = 4,
    min_rem_epochs: int = 1,
) -> tuple[float, str]:
    """Escolhe o nivel de atonia a usar: REM (preferencial) ou fallback NREM.

    Usa o fallback quando rem_baseline_uv e NaN, <=0, ou nao-finito, ou
    quando ha poucas mini-epocas REM (`rem_baseline_n_epochs < min_rem_epochs`)
    para confiar na estatistica do REM.

    Retorna (baseline_uv, source) onde source in {"rem", "nrem_fallback", "unavailable"}.
    """
    if (
        np.isfinite(rem_baseline_uv)
        and rem_baseline_uv > 0
        and rem_baseline_n_epochs >= min_rem_epochs
    ):
        return float(rem_baseline_uv), "rem"

    nrem_uv, n_nrem = compute_nrem_baseline_uv(
        signals, sleep_stages, emg_channel_index=emg_channel_index
    )
    if np.isfinite(nrem_uv) and nrem_uv > 0:
        return nrem_uv, "nrem_fallback"
    return float("nan"), "unavailable"


def _segments_above_threshold(env_uv: np.ndarray, threshold_uv: float) -> list[tuple[int, int]]:
    """Runs contiguos (amostra) onde env_uv >= threshold_uv. Limiar SIMPLES
    de amplitude (a AASM nao especifica histerese k_on/k_off -- e um corte
    unico contra o nivel fixo de atonia)."""
    mask = env_uv >= threshold_uv
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


def _merge_close_segments(
    segs: list[tuple[int, int]], gap_samples: int
) -> list[tuple[int, int]]:
    """Funde segmentos consecutivos separados por um gap <= gap_samples
    (amostras abaixo do limiar) em um unico segmento continuo. Ver
    MERGE_GAP_S acima para a justificativa (correcao de fragmentacao do
    detector de limiar simples, convencao RBDtector/SINBAR de 250ms).

    `segs` deve estar ordenado por inicio (garantido por
    `_segments_above_threshold`). gap_samples <= 0 desativa a fusao
    (retorna `segs` inalterado).
    """
    if gap_samples <= 0 or len(segs) <= 1:
        return list(segs)
    merged = [segs[0]]
    for s, e in segs[1:]:
        last_s, last_e = merged[-1]
        if s - last_e <= gap_samples:
            merged[-1] = (last_s, max(last_e, e))
        else:
            merged.append((s, e))
    return merged


def _macro_epoch_bounds(macro_idx: int, fs: int = FS,
                         macro_epoch_sec: float = MACRO_EPOCH_SEC) -> tuple[int, int]:
    n_samples = int(round(macro_epoch_sec * fs))
    return macro_idx * n_samples, (macro_idx + 1) * n_samples


def classify_macro_epoch(
    env_uv: np.ndarray,           # (n_samples_macro,) envelope RMS do EMG, em uV, DENTRO da epoca de 30s
    threshold_uv: float,
    *,
    fs: int = FS,
    mini_per_macro: int = MINI_PER_MACRO,
    epoch_sec: float = EPOCH_SEC,
    tonic_segment_min_s: float = TONIC_SEGMENT_MIN_S,
    tonic_epoch_coverage: float = TONIC_EPOCH_COVERAGE,
    phasic_lo_s: float = PHASIC_LO_S,
    phasic_hi_s: float = PHASIC_HI_S,
    phasic_min_mini_fraction: float = PHASIC_MIN_MINI_FRACTION,
    any_min_seg_s: float = ANY_MIN_SEG_S,
    merge_gap_s: float = MERGE_GAP_S,
) -> dict:
    """Aplica os 3 criterios da AASM a UMA epoca de 30s (estagio R) ja
    isolada. `env_uv` deve ter exatamente mini_per_macro*epoch_sec*fs
    amostras (30s * 100Hz = 3000, por padrao).

    `merge_gap_s`: segmentos acima do limiar separados por um gap
    <= merge_gap_s sao fundidos em um so ANTES de calcular duracoes para
    os 3 criterios (ver MERGE_GAP_S). Passe 0 para desativar a fusao.

    Retorna dict:
      tonic          : bool -- criterio tonico da epoca (aplica-se a toda a epoca)
      phasic         : bool -- criterio fasico da epoca (aplica-se a toda a epoca)
      any_mini       : (mini_per_macro,) bool -- criterio any, por mini-epoca de 3s
      n_phasic_mini  : int  -- quantas das mini_per_macro mini-epocas tem burst fasico
      tonic_coverage_s : float -- soma das duracoes dos segmentos >5s usados no criterio tonico
    """
    raw_segs = _segments_above_threshold(env_uv, threshold_uv)
    gap_samples = int(round(merge_gap_s * fs))
    segs = _merge_close_segments(raw_segs, gap_samples)
    durations_s = [(e - s) / fs for s, e in segs]

    # --- tonico: soma dos segmentos > tonic_segment_min_s, cobertura >= 50% da epoca ---
    tonic_segments_s = [d for d in durations_s if d > tonic_segment_min_s]
    tonic_coverage_s = float(sum(tonic_segments_s))
    macro_dur_s = mini_per_macro * epoch_sec
    tonic = tonic_coverage_s >= (tonic_epoch_coverage * macro_dur_s)

    # --- fasico: >=50% das mini-epocas de 3s contem burst 0.1-5.0s ---
    n_mini_samples = int(round(epoch_sec * fs))
    phasic_mini = np.zeros(mini_per_macro, dtype=bool)
    for s, e in segs:
        dur_s = (e - s) / fs
        if not (phasic_lo_s <= dur_s <= phasic_hi_s):
            continue
        m0 = s // n_mini_samples
        m1 = (e - 1) // n_mini_samples
        for m in range(max(0, m0), min(mini_per_macro - 1, m1) + 1):
            phasic_mini[m] = True
    n_phasic_mini = int(phasic_mini.sum())
    phasic = n_phasic_mini >= round(phasic_min_mini_fraction * mini_per_macro)

    # --- any: qualquer atividade >= limiar, por mini-epoca (superset de
    # tonic/phasic; nao ha piso SUPERIOR de duracao -- um segmento longo
    # que ja disparou tonic tambem conta para any). Piso INFERIOR de
    # any_min_seg_s exclui cruzamentos de limiar mais curtos que a
    # resolucao minima de um burst real (ruido de envelope), que o texto
    # da AASM nao antecipa ao dizer "sem considerar a duracao" -- ver
    # ANY_MIN_SEG_S acima.
    any_mini = np.zeros(mini_per_macro, dtype=bool)
    for (s, e), dur_s in zip(segs, durations_s):
        if dur_s < any_min_seg_s:
            continue
        m0 = s // n_mini_samples
        m1 = (e - 1) // n_mini_samples
        for m in range(max(0, m0), min(mini_per_macro - 1, m1) + 1):
            any_mini[m] = True

    return {
        "tonic": bool(tonic),
        "phasic": bool(phasic),
        "any_mini": any_mini,
        "n_phasic_mini": n_phasic_mini,
        "tonic_coverage_s": tonic_coverage_s,
    }


def apply_aasm_rule(
    signals: np.ndarray,          # (T, N_CHANNELS, n_samples) float, mini-epocas de 3s, unidades BRUTAS (V)
    sleep_stages: np.ndarray,     # (T,) int64, estagio por mini-epoca (constante dentro de cada bloco de 10)
    rem_baseline_uv: float,
    rem_baseline_n_epochs: int,
    *,
    emg_channel_index: int = 4,
    rem_stage: int = REM_STAGE,
    min_amplitude_ratio: float = MIN_AMPLITUDE_RATIO,
    fs: int = FS,
    epoch_sec: float = EPOCH_SEC,
    mini_per_macro: int = MINI_PER_MACRO,
    volts_to_microvolts: float = 1e6,
    merge_gap_s: float = MERGE_GAP_S,
) -> dict[str, np.ndarray | float | str]:
    """Aplica a regra AASM completa a um exame inteiro, produzindo rotulos
    por MINI-EPOCA de 3s no mesmo schema do .pt (tonic_labels/phasic_labels/
    any_labels), compativel com o consumido pelo restante do pipeline.

    `merge_gap_s`: ver `classify_macro_epoch` -- passado por epoca.

    Pressupoe que T e multiplo de mini_per_macro e que sleep_stages e
    constante dentro de cada bloco de mini_per_macro mini-epocas (garantido
    por preprocess.py: stages_mini = np.repeat(stages_30s, n_mini_per_epoch)).
    Blocos fora de estagio R (qualquer estagio != rem_stage) recebem
    tonic=phasic=any=0 -- a regra da AASM so se aplica a epocas de sono REM.

    Retorna dict com:
      tonic_labels, phasic_labels, any_labels : (T,) float32 {0,1}
      atonia_baseline_uv : float -- nivel de atonia usado (REM ou fallback NREM)
      atonia_source      : str   -- "rem" | "nrem_fallback" | "unavailable"
      n_rem_macro_epochs  : int  -- quantas epocas de 30s de estagio R foram avaliadas
      n_tonic_macro_epochs, n_phasic_macro_epochs : int -- contagem de epocas R positivas
    """
    signals = np.asarray(signals)
    sleep_stages = np.asarray(sleep_stages)
    T = signals.shape[0]
    if T % mini_per_macro != 0:
        raise ValueError(
            f"T={T} nao e multiplo de mini_per_macro={mini_per_macro}; "
            "apply_aasm_rule espera mini-epocas ja agrupadas em blocos de 30s."
        )

    baseline_uv, atonia_source = resolve_atonia_baseline_uv(
        rem_baseline_uv, rem_baseline_n_epochs, signals, sleep_stages,
        emg_channel_index=emg_channel_index,
    )

    tonic_labels = np.zeros(T, dtype=np.float32)
    phasic_labels = np.zeros(T, dtype=np.float32)
    any_labels = np.zeros(T, dtype=np.float32)

    n_rem_macro = 0
    n_tonic_macro = 0
    n_phasic_macro = 0

    if atonia_source == "unavailable":
        return {
            "tonic_labels": tonic_labels, "phasic_labels": phasic_labels, "any_labels": any_labels,
            "atonia_baseline_uv": float("nan"), "atonia_source": atonia_source,
            "n_rem_macro_epochs": 0, "n_tonic_macro_epochs": 0, "n_phasic_macro_epochs": 0,
        }

    threshold_uv = min_amplitude_ratio * baseline_uv
    n_macro = T // mini_per_macro
    emg = signals[:, emg_channel_index, :].astype(np.float64)  # (T, n_samples_mini)

    for macro_idx in range(n_macro):
        m0 = macro_idx * mini_per_macro
        m1 = m0 + mini_per_macro
        stage_block = sleep_stages[m0:m1]
        if not np.all(stage_block == rem_stage):
            continue  # regra da AASM so aplica dentro de estagio R
        n_rem_macro += 1

        emg_macro = emg[m0:m1].reshape(-1)  # concatena as 10 mini-epocas em ordem temporal
        env_uv = rms_envelope(emg_macro, win_sec=0.1, fs=fs) * volts_to_microvolts

        result = classify_macro_epoch(
            env_uv, threshold_uv, fs=fs, mini_per_macro=mini_per_macro, epoch_sec=epoch_sec,
            merge_gap_s=merge_gap_s,
        )
        any_block = result["any_mini"].copy()
        if result["tonic"]:
            tonic_labels[m0:m1] = 1.0
            n_tonic_macro += 1
            any_block[:] = True   # any e SUPERSET: epoca tonica -> any=1 na epoca toda
        if result["phasic"]:
            phasic_labels[m0:m1] = 1.0
            n_phasic_macro += 1
            any_block[:] = True   # any e SUPERSET: epoca fasica -> any=1 na epoca toda
        any_labels[m0:m1] = any_block.astype(np.float32)

    return {
        "tonic_labels": tonic_labels,
        "phasic_labels": phasic_labels,
        "any_labels": any_labels,
        "atonia_baseline_uv": baseline_uv,
        "atonia_source": atonia_source,
        "n_rem_macro_epochs": n_rem_macro,
        "n_tonic_macro_epochs": n_tonic_macro,
        "n_phasic_macro_epochs": n_phasic_macro,
    }


LABEL_SOURCE = "aasm_rule_v1"

# AVISO DE CALIBRACAO (ver docs/relatorio_impacto_regra_aasm.md, secao 4):
# com atonia_pct=10.0 (mesmo percentil ja usado em rem_baseline_uv), o
# criterio de amplitude "*>=2x o nivel de atonia*" satura -- 87-100% das
# epocas R viram phasic/any nos 5 exames testados, porque a mediana do
# envelope RMS em REM ja fica 2.3-2.7x acima do percentil 10 so por
# variabilidade normal do sinal. NAO usar rswa_source="aasm" em producao
# sem antes recalibrar (testar atonia_pct mais alto, ex. 25-50) e revalidar
# contra os CSVs revisados. O parametro atonia_pct abaixo existe exatamente
# para essa recalibracao nao exigir nova integracao.
AASM_CALIBRATION_WARNING = (
    "rswa_source='aasm': percentil de atonia atual (10.0, herdado de "
    "rem_baseline_uv) satura phasic/any (~87-100% das epocas REM positivas "
    "nos exames de referencia) -- ver docs/relatorio_impacto_regra_aasm.md "
    "secao 4 antes de usar em treinamento. Ajuste atonia_pct para recalibrar."
)


def label_exam_with_aasm_rule(
    signals: np.ndarray,
    sleep_stages: np.ndarray,
    rem_baseline_uv: float,
    rem_baseline_n_epochs: int,
    *,
    emg_channel_index: int = 4,
    rem_stage: int = REM_STAGE,
    min_amplitude_ratio: float = MIN_AMPLITUDE_RATIO,
    fs: int = FS,
    epoch_sec: float = EPOCH_SEC,
    mini_per_macro: int = MINI_PER_MACRO,
    atonia_pct: float | None = None,
    merge_gap_s: float = MERGE_GAP_S,
) -> dict[str, np.ndarray | float | int | str]:
    """Wrapper de `apply_aasm_rule` que devolve o MESMO schema de chaves
    produzido por `auto_label_rswa_from_signals` (auto_rswa.py) e por
    `rasterize_rswa_annotations` (rswa_labels.py), para poder ser usado como
    terceira opcao de `rswa_source` em `preprocess_exam` sem alterar o
    formato do .pt.

    `atonia_pct`: se None (default), usa `rem_baseline_uv` tal como recebido
    (percentil 10, mesmo campo ja gravado no .pt). Se informado, RECALCULA
    o nivel de atonia REM internamente com esse percentil (ignorando o
    `rem_baseline_uv` recebido) -- usar apos a recalibracao descrita em
    AASM_CALIBRATION_WARNING, sem precisar modificar a assinatura desta
    funcao nem o chamador em preprocess.py.

    Retorna dict com as mesmas chaves de auto_label_rswa_from_signals:
      tonic_labels, phasic_labels, any_labels : (T,) float32 {0,1}
      rswa_labels : (T,) int64 {0,1,2,3} (0=nada,1=fasico,2=tonico,3=ambos;
                    NAO inclui "any", mesma convencao de rswa_labels.py)
      rswa_conf   : (T,) float32 {0,1} -- validade (1.0 onde stage != -1)
      tonic_cov, phasic_cov, any_cov : (T,) float32 {0,1} -- aqui SAO os
                    proprios labels (decisao binaria da AASM, nao fracao de
                    cobertura continua como em rswa_labels.py/auto_rswa.py;
                    mantidas por compatibilidade de schema)
      label_source : str -- "aasm_rule_v1"
      atonia_baseline_uv, atonia_source, atonia_pct_used,
      n_rem_macro_epochs, n_tonic_macro_epochs, n_phasic_macro_epochs : diagnostico
    """
    signals = np.asarray(signals)
    sleep_stages = np.asarray(sleep_stages)

    if atonia_pct is not None:
        rem_mask = sleep_stages == rem_stage
        if int(rem_mask.sum()) > 0:
            emg_rem = signals[rem_mask, emg_channel_index, :].astype(np.float64)
            envelopes = [rms_envelope(row, win_sec=0.1, fs=fs) for row in emg_rem]
            env_concat = np.concatenate(envelopes)
            rem_baseline_uv = float(np.percentile(env_concat, atonia_pct)) * 1e6
            rem_baseline_n_epochs = int(rem_mask.sum())
        else:
            rem_baseline_uv, rem_baseline_n_epochs = float("nan"), 0

    result = apply_aasm_rule(
        signals, sleep_stages, rem_baseline_uv, rem_baseline_n_epochs,
        emg_channel_index=emg_channel_index, rem_stage=rem_stage,
        min_amplitude_ratio=min_amplitude_ratio, fs=fs, epoch_sec=epoch_sec,
        mini_per_macro=mini_per_macro, merge_gap_s=merge_gap_s,
    )

    tonic_labels = result["tonic_labels"]
    phasic_labels = result["phasic_labels"]
    any_labels = result["any_labels"]
    rswa_labels_int = (phasic_labels.astype(np.int64) * 1) + (tonic_labels.astype(np.int64) * 2)
    rswa_conf = (sleep_stages != -1).astype(np.float32)

    return {
        "tonic_labels": tonic_labels,
        "phasic_labels": phasic_labels,
        "any_labels": any_labels,
        "rswa_labels": rswa_labels_int,
        "rswa_conf": rswa_conf,
        "tonic_cov": tonic_labels.copy(),
        "phasic_cov": phasic_labels.copy(),
        "any_cov": any_labels.copy(),
        "label_source": LABEL_SOURCE,
        "atonia_baseline_uv": result["atonia_baseline_uv"],
        "atonia_source": result["atonia_source"],
        "atonia_pct_used": float(atonia_pct) if atonia_pct is not None else 10.0,
        "n_rem_macro_epochs": result["n_rem_macro_epochs"],
        "n_tonic_macro_epochs": result["n_tonic_macro_epochs"],
        "n_phasic_macro_epochs": result["n_phasic_macro_epochs"],
        "min_amplitude_ratio_used": float(min_amplitude_ratio),
    }
