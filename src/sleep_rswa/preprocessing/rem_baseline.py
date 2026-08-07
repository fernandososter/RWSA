"""
Basal (baseline) de EMG na fase REM -- calculado uma vez por exame e gravado
no .pt para uso posterior por detectores de eventos tonico/any/fasico.

Chamar dentro de preprocess_exam, LOGO APOS rasterize_rswa_annotations (etapa
10) e ANTES do dict de retorno -- mesmo ponto do pipeline onde tonic_labels/
phasic_labels ja estao disponiveis, embora este calculo NAO dependa deles.

Definicao (decidida com o usuario em 2026-08-06, ver plano_rem_baseline.md):
  - Unidade: microvolts BRUTOS (sem normalizacao). O .pt hoje grava o EMG tal
    como veio do EDF (Volts, pos-filtro), sem nenhum z-score -- este calculo
    usa esse mesmo sinal bruto e so converte a escala para uV na saida, para
    ficar interpretavel sem precisar saber qual formula de normalizacao um
    consumidor futuro vai usar (ha DUAS formulas de z-score independentes no
    projeto -- src/sleep_rswa/data.py e classifier/movement_clf/dataio.py --
    e este campo nao deve ficar acoplado a nenhuma das duas).
  - Estatistica: percentil 10 (mesmo BASELINE_PCT ja validado nos testes
    deterministicos em testes/src/limiar/threshold_rule.py) do envelope RMS
    do canal EMG, calculado SOMENTE sobre as mini-epocas em que
    sleep_stages == rem_stage. Percentil (em vez de media) e robusto a
    mini-epocas com evento tonico/fasico sustentado dentro do REM, sem
    precisar dos rotulos tonic_labels/phasic_labels -- funciona igual em
    exames revisados (treino) e em exames novos sem revisao (inferencia).
  - Envelope RMS: janela deslizante de 0.1s, calculado independentemente
    dentro de cada mini-epoca de 3s (nao ha continuidade temporal assumida
    entre mini-epocas REM, que tipicamente NAO sao contiguas no exame).

Este modulo grava o valor no .pt; NAO altera nenhum detector de eventos.
Os tres testes deterministicos (testes/src/limiar, tkeo, cusum_glr) e a rede
RSWADetectionNet continuam com suas baselines locais atuais -- decidir usar
este campo como baseline de deteccao e uma etapa separada, ainda nao feita.
"""
from __future__ import annotations

import numpy as np


def rem_envelope_rms(x: np.ndarray, win_sec: float = 0.1, fs: int = 100) -> np.ndarray:
    """Envelope RMS de janela deslizante (mesmo comprimento da entrada).
    Identica a rms_envelope() em testes/src/limiar/threshold_rule.py -- mantida
    como copia local (nao importada) para preservar o isolamento entre
    src/sleep_rswa/preprocessing e testes/."""
    win = max(1, int(round(win_sec * fs)))
    x2 = x.astype(np.float64) ** 2
    kernel = np.ones(win) / win
    ms = np.convolve(x2, kernel, mode="same")
    return np.sqrt(ms)


def compute_rem_baseline(
    signals: np.ndarray,        # (T, N_CHANNELS, n_samples) float32 -- unidades BRUTAS (V)
    sleep_stages: np.ndarray,   # (T,) int64
    *,
    emg_channel_index: int = 4,
    rem_stage: int = 4,
    pct: float = 10.0,
    win_sec: float = 0.1,
    fs: int = 100,
    volts_to_microvolts: float = 1e6,
) -> dict[str, float]:
    """
    Calcula o basal de EMG na fase REM para um exame.

    Retorna:
        rem_baseline_uv       : float -- percentil `pct` do envelope RMS do
                                  EMG dentro das mini-epocas REM, em microvolts.
                                  NaN se o exame nao tiver nenhuma mini-epoca REM
                                  (nao ocorreu em nenhum dos 60 exames atuais,
                                  mas o consumidor deve tratar o caso).
        rem_baseline_n_epochs : int -- quantas mini-epocas REM entraram no
                                  calculo (diagnostico / QC).
    """
    sleep_stages = np.asarray(sleep_stages)
    rem_mask = sleep_stages == rem_stage
    n_rem = int(rem_mask.sum())

    if n_rem == 0:
        return {"rem_baseline_uv": float("nan"), "rem_baseline_n_epochs": 0}

    emg_rem = np.asarray(signals)[rem_mask, emg_channel_index, :].astype(np.float64)

    envelopes = [rem_envelope_rms(row, win_sec=win_sec, fs=fs) for row in emg_rem]
    env_concat = np.concatenate(envelopes)

    baseline_v = float(np.percentile(env_concat, pct))
    baseline_uv = baseline_v * volts_to_microvolts

    return {"rem_baseline_uv": baseline_uv, "rem_baseline_n_epochs": n_rem}
