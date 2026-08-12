"""
ecg_gating.py — Deteccao de picos-R no ECG e gating (interpolacao) da janela
correspondente no EMG, para remover contaminacao por artefato cardiaco
("bleed-through" do complexo QRS no canal de EMG de mento).

Contexto (docs/relatorio_impacto_regra_aasm.md, Secao 12): confirmou-se
visualmente e quantitativamente, num subconjunto de mini-epocas
`any`-positivas nos 5 exames de referencia, que uma fracao pequena mas
sistematica (0.6-3.5%) exibe picos de EMG sincronizados com o complexo QRS
do ECG (frequencia cardiaca fisiologica, 40-150 bpm). O padrao-ouro na
literatura (RBDtector, protocolos de scoring PSG) e fazer gating/subtracao
sincronizada por ECG: remover ou interpolar uma janela estreita em torno de
cada pico-R antes de qualquer deteccao de amplitude no EMG.

Este modulo implementa esse gating como um passo OPCIONAL no
pre-processamento (preprocess.py::preprocess_exam), habilitado por default
apos validacao quantitativa (ver quantify_gating_effect.py e Secao 12 do
relatorio), mas desativavel via ecg_gate=False para reproduzir o
comportamento anterior (sem ECG).

Uso tipico (dentro de preprocess_exam, apos filtragem por FILTER_PARAMS):

    if ecg_present and emg_present:
        r_peaks = detect_r_peaks(ecg_uv, fs)
        emg_gated, n_gated = gate_emg_by_ecg(
            emg_uv, r_peaks, fs, window_pre_s=0.06, window_post_s=0.15,
        )

Janela ASSIMETRICA (Secao 12.4 do relatorio): a validacao inicial usava uma
janela simetrica de +-50ms, mas a inspecao do perfil medio de |EMG| em
torno do pico-R (pooled sobre ~1400 batimentos REM dos 5 exames de
referencia) mostrou que o artefato de QRS/T-wave no EMG NAO e simetrico --
comeca a subir ANTES do pico-R (onda P / limiar de deteccao) e persiste
POR MAIS TEMPO depois dele (T-wave), com a extensao significativa tipica
em torno de [-0.03s, +0.10s] e cauda ocasional ate ~+0.14s. A janela
simetrica de +-50ms capturava so ~47% da energia do artefato acima do
baseline, deixando artefato residual visivel do lado direito do pico-R
(ver `ins8.pt`, mini-epocas REM no modo revisao). Os defaults atuais
(`window_pre_s=0.06`, `window_post_s=0.15`) cobrem >=95% dessa energia nos
5 exames de referencia.
"""
from __future__ import annotations

import numpy as np


def detect_r_peaks(
    ecg: np.ndarray,
    fs: float,
    *,
    method: str = "neurokit",
) -> np.ndarray:
    """
    Detecta picos-R num sinal de ECG 1D.

    Usa `neurokit2.ecg_peaks` (mesmo detector usado na quantificacao de
    contaminacao cardiaca da Secao 12). Retorna indices (int) dos picos-R,
    clipados para o intervalo valido [0, len(ecg)-1]. Se `neurokit2` nao
    conseguir processar o sinal (ex.: muito curto, sem batimento detectavel),
    retorna array vazio em vez de propagar excecao -- gating downstream deve
    ser tolerante a "nenhum pico-R encontrado".

    Parameters
    ----------
    ecg : np.ndarray
        Sinal de ECG 1D, qualquer unidade (neurokit2 normaliza internamente).
    fs : float
        Taxa de amostragem em Hz.
    method : str
        Metodo de deteccao passado a neurokit2.ecg_peaks (default "neurokit").

    Returns
    -------
    np.ndarray[int]
        Indices (amostras) dos picos-R detectados, em ordem crescente.
    """
    ecg = np.asarray(ecg, dtype=np.float64).reshape(-1)
    if ecg.size < int(fs * 0.5):  # menos de 0.5s: sem batimento detectavel
        return np.array([], dtype=np.int64)
    try:
        import neurokit2 as nk

        _, info = nk.ecg_peaks(ecg, sampling_rate=int(round(fs)), method=method)
        r_peaks = np.asarray(info.get("ECG_R_Peaks", []), dtype=np.int64)
    except Exception:
        return np.array([], dtype=np.int64)

    if r_peaks.size == 0:
        return r_peaks
    r_peaks = r_peaks[(r_peaks >= 0) & (r_peaks < ecg.size)]
    return np.sort(r_peaks)


def apply_ecg_gating_to_raw(
    raw,
    emg_ch_name: str | None,
    ecg_candidates: list[str],
    *,
    window_pre_s: float = 0.06,
    window_post_s: float = 0.15,
    verbose: bool = False,
) -> dict:
    """
    Localiza um canal de ECG em `raw.ch_names` (dentre `ecg_candidates`),
    detecta picos-R nele e aplica gating (interpolacao, ver gate_emg_by_ecg)
    IN-PLACE no canal de EMG `emg_ch_name`. Remove o canal de ECG de `raw`
    ao final (raw.drop_channels), para que ele nao apareca na matriz de
    sinais final salva no .pt -- o pipeline de producao nunca usou o ECG
    como canal de entrada do modelo, apenas como sinal auxiliar de gating.

    Deve ser chamado DEPOIS do crop (annot_start/annot_end) e ANTES do loop
    de filtragem por FILTER_PARAMS (preprocess.py etapa 4), para que a
    deteccao de picos-R veja o ECG o mais bruto possivel (sem filtro passa-
    banda que possa distorcer a morfologia do QRS) e para que o EMG gated
    seja o que efetivamente entra no filtro/reamostragem/epocamento.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Objeto Raw ja com os canais de interesse (EMG + ECG, se presente)
        selecionados via raw.pick(). Modificado in-place.
    emg_ch_name : str | None
        Nome real do canal de EMG em raw.ch_names (None se EMG ausente no
        exame -- nesse caso nada e feito).
    ecg_candidates : list[str]
        Lista de nomes alternativos aceitos para o canal de ECG (ver
        PSGConfig.ECG_CANDIDATES em config.py), comparacao case-insensitive.
    window_pre_s : float
        Extensao da janela de gating ANTES do pico-R, em segundos (ver
        gate_emg_by_ecg).
    window_post_s : float
        Extensao da janela de gating DEPOIS do pico-R, em segundos (ver
        gate_emg_by_ecg). Maior que `window_pre_s` por default -- o
        artefato de EMG persiste mais tempo apos o pico-R (T-wave) do que
        antes (Secao 12.4 do relatorio).
    verbose : bool
        Se True, imprime uma linha de diagnostico.

    Returns
    -------
    dict
        Diagnostico da operacao:
        {
          "ecg_gate_applied":  bool   -- True somente se ECG encontrado, EMG
                                          presente e >=1 pico-R detectado
          "ecg_channel_found": str | None
          "n_r_peaks":         int
          "n_gated_samples":   int
          "frac_gated":        float  -- n_gated_samples / n_samples_emg
          "window_pre_s":      float
          "window_post_s":     float
          "reason_skipped":    str | None  -- "emg_channel_absent",
                                "ecg_channel_not_found" ou
                                "no_r_peaks_detected" quando ecg_gate_applied=False
        }
    """
    from .channels import find_channel

    diag = {
        "ecg_gate_applied": False,
        "ecg_channel_found": None,
        "n_r_peaks": 0,
        "n_gated_samples": 0,
        "frac_gated": 0.0,
        "window_pre_s": float(window_pre_s),
        "window_post_s": float(window_post_s),
        "reason_skipped": None,
    }

    if emg_ch_name is None or emg_ch_name not in raw.ch_names:
        diag["reason_skipped"] = "emg_channel_absent"
        return diag

    ecg_ch_name = find_channel(raw.ch_names, ecg_candidates)
    if ecg_ch_name is None:
        diag["reason_skipped"] = "ecg_channel_not_found"
        return diag

    diag["ecg_channel_found"] = ecg_ch_name
    fs = float(raw.info["sfreq"])

    ecg_data = raw.get_data(picks=[ecg_ch_name])[0]
    r_peaks = detect_r_peaks(ecg_data, fs)
    diag["n_r_peaks"] = int(len(r_peaks))

    if len(r_peaks) == 0:
        diag["reason_skipped"] = "no_r_peaks_detected"
        raw.drop_channels([ecg_ch_name])
        return diag

    emg_idx = raw.ch_names.index(emg_ch_name)
    emg_data = raw.get_data(picks=[emg_ch_name])[0]
    emg_gated, n_gated = gate_emg_by_ecg(
        emg_data, r_peaks, fs,
        window_pre_s=window_pre_s, window_post_s=window_post_s,
    )

    raw._data[emg_idx, :] = emg_gated
    raw.drop_channels([ecg_ch_name])

    diag["ecg_gate_applied"] = True
    diag["n_gated_samples"] = int(n_gated)
    diag["frac_gated"] = float(n_gated / len(emg_data)) if len(emg_data) else 0.0

    if verbose:
        print(f"  [ECG GATING] canal={ecg_ch_name} r_picos={diag['n_r_peaks']} "
              f"amostras_gated={diag['n_gated_samples']} "
              f"({diag['frac_gated']*100:.2f}% do canal EMG)")

    return diag


def gate_emg_by_ecg(
    emg: np.ndarray,
    r_peaks: np.ndarray,
    fs: float,
    *,
    window_pre_s: float = 0.06,
    window_post_s: float = 0.15,
) -> tuple[np.ndarray, int]:
    """
    Remove a contaminacao de QRS/T-wave do EMG por interpolacao linear numa
    janela ASSIMETRICA centrada em cada pico-R (gating).

    Para cada pico-R em `r_peaks`, substitui as amostras de `emg` no
    intervalo [r_peak - window_pre_s, r_peak + window_post_s] por uma
    interpolacao linear entre os dois valores nas bordas da janela -- em
    vez de zerar (que criaria uma queda espuria e artificial no envelope
    RMS), a interpolacao preserva o nivel local do sinal sem introduzir a
    espicula sincronizada ao ECG. Janelas de picos-R adjacentes que se
    sobrepoem sao fundidas automaticamente (processadas em ordem, cada uma
    usa os valores JA interpolados da anterior como borda).

    A janela e assimetrica por default (`window_post_s > window_pre_s`)
    porque o perfil medio de |EMG| em torno do pico-R (validado em ~1400
    batimentos REM pooled sobre os 5 exames de referencia, Secao 12.4 do
    relatorio) mostra que o artefato de bleed-through cardiaco no EMG
    persiste mais tempo DEPOIS do pico-R (onda T) do que antes; uma janela
    simetrica de +-50ms deixava escapar artefato residual visivel no lado
    direito do pico-R.

    Parameters
    ----------
    emg : np.ndarray
        Sinal de EMG 1D (mesma taxa de amostragem que `fs`, tipicamente ja
        filtrado por FILTER_PARAMS["emg"]).
    r_peaks : np.ndarray[int]
        Indices dos picos-R (saida de detect_r_peaks), na mesma base de
        amostragem que `emg`.
    fs : float
        Taxa de amostragem em Hz.
    window_pre_s : float
        Extensao da janela ANTES do pico-R, em segundos (default 0.06s).
    window_post_s : float
        Extensao da janela DEPOIS do pico-R, em segundos (default 0.15s --
        cobre >=95% da energia do artefato acima do baseline nos 5 exames
        de referencia; ver Secao 12.4).

    Returns
    -------
    emg_gated : np.ndarray
        Copia de `emg` com as janelas de picos-R interpoladas.
    n_gated_samples : int
        Numero total de amostras efetivamente substituidas (uniao das
        janelas, sem contagem duplicada em caso de sobreposicao).
    """
    #MARK - Aqui ocorre a remocaco do ECG no EMG, por interpolacao linear entre os pontos de borda da janela de gating
    emg = np.asarray(emg, dtype=np.float64).reshape(-1).copy()
    n = emg.size
    if r_peaks is None or len(r_peaks) == 0:
        return emg, 0

    half_w_pre = max(1, int(round(window_pre_s * fs)))
    half_w_post = max(1, int(round(window_post_s * fs)))
    gated_mask = np.zeros(n, dtype=bool)

    for peak in r_peaks:
        i0 = max(0, int(peak) - half_w_pre)
        i1 = min(n - 1, int(peak) + half_w_post)
        if i1 <= i0:
            continue
        left_val = emg[i0]
        right_val = emg[i1]
        n_pts = i1 - i0 + 1
        emg[i0:i1 + 1] = np.linspace(left_val, right_val, n_pts)
        gated_mask[i0:i1 + 1] = True

    return emg, int(gated_mask.sum())
