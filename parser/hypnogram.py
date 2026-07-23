"""
Leitura e alinhamento do hipnograma (.mat gerado em Octave) ao inicio do EDF.

Convertido do notebook Parser_Exames (celula 15).
"""
from __future__ import annotations

from typing import Optional

import mne
import numpy as np
from scipy.io import loadmat


# Codigos numericos do hyp[:, 0] (R&K) -> descricao MNE.
STAGE_MAP = {
    0: "Sleep stage W",
    1: "Sleep stage N1",
    2: "Sleep stage N2",
    3: "Sleep stage N3",
    4: "Sleep stage N3",   # R&K S4 combinado com N3
    5: "Sleep stage R",
    7: "Movement time",
}


def _parse_stage_value(val: str) -> Optional[int]:
    """Converte a descricao textual de uma annotation MNE em codigo de estagio."""
    v = val.strip().upper()
    aliases = {
        "A": 0, "W": 0, "SLEEP STAGE W": 0, "WAKE": 0,
        "N1": 1, "SLEEP STAGE N1": 1,
        "N2": 2, "SLEEP STAGE N2": 2,
        "N3": 3, "SLEEP STAGE N3": 3,
        "SLEEP STAGE R": 4, "REM": 4, "R": 4,
    }
    return aliases.get(v, None)


def calculate_annotation_offset(raw, start_time) -> float:
    """
    Deslocamento (s) entre o inicio do EDF (raw.info['meas_date']) e o inicio
    do hipnograma (start_time do .mat). Corrige passagem pela meia-noite.
    """
    meas_date = raw.info["meas_date"]
    if meas_date is None:
        raise ValueError(
            "O EDF nao possui raw.info['meas_date']; "
            "nao e possivel alinhar as anotacoes pelo horario."
        )

    edf_start_seconds = (
        meas_date.hour * 3600
        + meas_date.minute * 60
        + meas_date.second
        + meas_date.microsecond / 1e6
    )
    annotation_start_seconds = (
        float(start_time["h"]) * 3600
        + float(start_time["m"]) * 60
        + float(start_time["s"])
    )
    offset = annotation_start_seconds - edf_start_seconds

    if offset < -12 * 3600:
        offset += 24 * 3600
    elif offset > 12 * 3600:
        offset -= 24 * 3600
    return float(offset)


def load_aligned_hyp_annotations(raw, mat_path, include_movement: bool = False):
    """
    Carrega hyp_<subject>.mat e cria mne.Annotations alinhadas ao inicio do EDF.

    Retorna (annotations, hyp, alignment).
    """
    data = loadmat(mat_path, simplify_cells=True)

    if "hyp" not in data:
        raise KeyError(f"A variavel 'hyp' nao foi encontrada em {mat_path}.")
    if "start_time" not in data:
        raise KeyError(f"A variavel 'start_time' nao foi encontrada em {mat_path}.")

    hyp = np.asarray(data["hyp"], dtype=float)
    start_time = data["start_time"]

    if hyp.ndim != 2 or hyp.shape[1] < 2:
        raise ValueError(f"Formato invalido da matriz hyp: {hyp.shape}")

    stages = hyp[:, 0].astype(int)
    relative_onsets = hyp[:, 1].astype(float)

    unknown_stages = set(np.unique(stages)) - set(STAGE_MAP)
    if unknown_stages:
        raise ValueError(f"Codigos de estagio desconhecidos: {unknown_stages}")

    offset = calculate_annotation_offset(raw, start_time)
    aligned_onsets = relative_onsets + offset

    keep = np.ones(len(stages), dtype=bool)
    if not include_movement:
        keep &= stages != 7

    stages_kept = stages[keep]
    onsets_kept = aligned_onsets[keep]

    descriptions = np.asarray(
        [STAGE_MAP[stage] for stage in stages_kept], dtype=str,
    )
    durations = np.full(len(onsets_kept), 30.0, dtype=float)

    annotations = mne.Annotations(
        onset=onsets_kept,
        duration=durations,
        description=descriptions,
        orig_time=None,
    )

    edf_duration = raw.n_times / raw.info["sfreq"]
    first_annotation_onset = (
        float(onsets_kept[0]) if len(onsets_kept) else np.nan
    )
    last_annotation_end = (
        float(onsets_kept[-1] + durations[-1]) if len(onsets_kept) else np.nan
    )

    alignment = {
        "offset_seconds": offset,
        "first_annotation_onset": first_annotation_onset,
        "last_annotation_end": last_annotation_end,
        "edf_duration": edf_duration,
        "initial_unscored_seconds": first_annotation_onset,
        "final_unscored_seconds": edf_duration - last_annotation_end,
        "number_of_annotations": len(annotations),
        "start_time": start_time,
    }
    return annotations, hyp, alignment
