"""
Leitura e alinhamento do hipnograma ao inicio do EDF.

Suporta dois caminhos:
  1. arquivo externo (.mat gerado em Octave ou .edf de hipnograma);
  2. anotacoes de estagiamento internas do proprio EDF+ bruto.

Convertido do notebook Parser_Exames (celula 15), com extensoes para EDF+.
"""
from __future__ import annotations

from pathlib import Path
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
        "N1": 1, "SLEEP STAGE N1": 1, "SLEEP STAGE 1": 1,
        "N2": 2, "SLEEP STAGE N2": 2, "SLEEP STAGE 2": 2,
        "N3": 3, "SLEEP STAGE N3": 3, "SLEEP STAGE 3": 3, "SLEEP STAGE 4": 3,
        "SLEEP STAGE R": 4, "REM": 4, "R": 4,
        "ARTEFATO": -1, "ARTEFACT": -1, "SLEEP STAGE ?": -1, "?": -1, "UNSCORED": -1,
    }
    return aliases.get(v, None)


def _canonical_stage_description(stage_code: int) -> str:
    mapping = {
        -1: "UNSCORED",
        0: "Sleep stage W",
        1: "Sleep stage N1",
        2: "Sleep stage N2",
        3: "Sleep stage N3",
        4: "Sleep stage R",
    }
    return mapping[stage_code]


def _normalize_stage_annotations(
    annotations: mne.Annotations,
    *,
    raw_duration: float,
    include_movement: bool = False,
):
    """
    Filtra apenas anotacoes de estagiamento reconhecidas e normaliza duracoes.

    Para EDF+ de estagiamento, e comum cada estagio aparecer como marcador
    pontual a cada 30 s (duracao zero). Nesses casos, assume-se 30 s.
    """
    kept_onsets = []
    kept_durations = []
    kept_descriptions = []

    for ann in annotations:
        stage_code = _parse_stage_value(str(ann["description"]))
        if stage_code is None:
            continue
        if not include_movement and stage_code == 7:
            continue
        duration = float(ann["duration"])
        if duration <= 0:
            duration = 30.0
        kept_onsets.append(float(ann["onset"]))
        kept_durations.append(duration)
        kept_descriptions.append(_canonical_stage_description(stage_code))

    if not kept_onsets:
        raise ValueError("Nenhuma anotacao de estagiamento reconhecida foi encontrada.")

    annotations_out = mne.Annotations(
        onset=kept_onsets,
        duration=kept_durations,
        description=kept_descriptions,
        orig_time=None,
    )
    first_annotation_onset = float(kept_onsets[0])
    last_annotation_end = float(kept_onsets[-1] + kept_durations[-1])
    alignment = {
        "offset_seconds": 0.0,
        "first_annotation_onset": first_annotation_onset,
        "last_annotation_end": last_annotation_end,
        "edf_duration": float(raw_duration),
        "initial_unscored_seconds": first_annotation_onset,
        "final_unscored_seconds": float(raw_duration) - last_annotation_end,
        "number_of_annotations": len(annotations_out),
        "source": "edf_annotations",
    }
    return annotations_out, None, alignment


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
    Carrega o hipnograma e cria mne.Annotations alinhadas ao inicio do EDF.

    Retorna (annotations, hyp, alignment).
    """
    mat_path = Path(mat_path)
    suffix = mat_path.suffix.lower()

    if suffix == ".edf":
        annotations = mne.read_annotations(str(mat_path))
        return _normalize_stage_annotations(
            annotations,
            raw_duration=raw.n_times / raw.info["sfreq"],
            include_movement=include_movement,
        )

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


def load_hyp_annotations_from_raw(raw, include_movement: bool = False):
    """
    Usa as anotacoes internas do proprio EDF+ como hipnograma.
    """
    return _normalize_stage_annotations(
        raw.annotations,
        raw_duration=raw.n_times / raw.info["sfreq"],
        include_movement=include_movement,
    )
