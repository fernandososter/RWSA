import csv
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import mne
import numpy as np
from scipy.io import loadmat
from dataclasses import dataclass, field

class PathConfig:
    EDF_DIR = Path("/Volumes/HD_EXTERNO/Workspace/AI/USP/dataset/capslpdb-1.0.0/")
    MAT_DIR = Path(__file__).parent.parent.parent / "data" / "mat"
    CONVERTED_DIR = EDF_DIR / "converted"


STAGE_MAP = {
    0: "Sleep stage W",
    1: "Sleep stage N1",
    2: "Sleep stage N2",
    3: "Sleep stage N3",
    4: "Sleep stage N3",  # R&K S4 combinado com N3
    5: "Sleep stage R",
    7: "Movement time",
}

CSV_ANNOTATION_COLUMNS = {
    "subject_id",
    "onset_s",
    "duration_s",
    "type",
}




def parser(
    edf_path: Path,
    subject_id: str,
    annotations_csv_path: Optional[Path] = None,
    annotations_csv_dir: Optional[Path] = None,
):

    try:
        converted_dir = PathConfig.EDF_DIR / "converted"
        
        scores_path = find_mat_file(subject_id,converted_dir)

        
        print(f"[SCORE FILE]: {scores_path}")
        raw = mne.io.read_raw_edf(str(edf_path),preload=True,verbose="ERROR")
        print(f"[RAW FILE DONE]")

        annotations, hyp, alignment = (
            load_aligned_hyp_annotations(raw=raw,mat_path=scores_path,include_movement=False)
        )

        raw.set_annotations(annotations,emit_warning=False)

        old = raw.annotations.copy()

        if annotations_csv_path is None and annotations_csv_dir is not None:
            annotations_csv_path = find_annotations_csv_file(
                subject_id=subject_id,
                csv_dir=annotations_csv_dir,
            )

        if annotations_csv_path is not None:
            new = load_subject_annotations_from_csv(
                csv_path=annotations_csv_path,
                subject_id=subject_id,
                orig_time=old.orig_time,
            )

            if len(new):
                raw.set_annotations(old + new, emit_warning=False)
                print(
                    f"[CSV ANNOTATIONS] {len(new)} anotações adicionadas "
                    f"para {subject_id}"
                )
            else:
                print(
                    f"[CSV ANNOTATIONS] nenhuma anotação encontrada "
                    f"para {subject_id} em {annotations_csv_path}"
                )
        elif annotations_csv_dir is not None:
            print(
                f"[CSV FILE] nenhum arquivo encontrado para {subject_id} "
                f"em {annotations_csv_dir}"
            )
        print(f"[ANNOTATION TOTAL] {len(raw.annotations)}")
        print(
            "[ANNOTATION COUNTS] "
            f"{count_annotations_by_description(raw.annotations)}"
        )
        
        
    

    except Exception as error:
        print(f"[ERRO ANOTAÇÃO] - {error}")


def load_subject_annotations_from_csv(
    csv_path: Path,
    subject_id: str,
    orig_time,
) -> mne.Annotations:
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"Arquivo CSV não encontrado: {csv_path}")

    rows = []

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing_columns = CSV_ANNOTATION_COLUMNS - fieldnames

        if missing_columns:
            raise ValueError(
                f"CSV inválido: colunas ausentes {sorted(missing_columns)}. "
                f"Esperado: {sorted(CSV_ANNOTATION_COLUMNS)}"
            )

        for line_number, row in enumerate(reader, start=2):
            row_subject_id = str(row.get("subject_id", "")).strip()
            if row_subject_id != subject_id:
                continue

            description = str(row.get("type", "")).strip()
            if not description:
                raise ValueError(
                    f"Linha {line_number}: campo 'type' vazio para {subject_id}."
                )

            try:
                onset = float(row["onset_s"])
                duration = float(row["duration_s"])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Linha {line_number}: onset_s/duration_s inválidos "
                    f"para {subject_id}."
                ) from error

            if onset < 0:
                raise ValueError(
                    f"Linha {line_number}: onset_s deve ser >= 0, recebeu {onset}."
                )

            if duration < 0:
                raise ValueError(
                    f"Linha {line_number}: duration_s deve ser >= 0, recebeu {duration}."
                )

            rows.append((onset, duration, description))

    rows.sort(key=lambda item: item[0])

    return mne.Annotations(
        onset=[row[0] for row in rows],
        duration=[row[1] for row in rows],
        description=[row[2] for row in rows],
        orig_time=orig_time,
    )


def count_annotations_by_description(
    annotations: mne.Annotations,
) -> Dict[str, int]:
    counts = Counter(str(description) for description in annotations.description)
    return dict(sorted(counts.items()))


def find_annotations_csv_file(
    subject_id: str,
    csv_dir: Path,
) -> Optional[Path]:
    csv_dir = Path(csv_dir)

    if not csv_dir.exists():
        raise FileNotFoundError(f"Diretório CSV não encontrado: {csv_dir}")

    if not csv_dir.is_dir():
        raise NotADirectoryError(f"Caminho CSV não é uma pasta: {csv_dir}")

    csv_path = csv_dir / f"{subject_id}_rswa.csv"
    if csv_path.exists():
        return csv_path

    return None


def find_mat_file(subject_id: str, scores_dir: Path) -> Optional[Path]:
    """
    Procura arquivo de scoring com mesmo stem do EDF em scores_dir.
    Aceita extensões: -annot.fif, _sleepscoring.edf
    """

    for ext in ["-annot.fif", "_sleepscoring.edf","-Hypnogram.edf",".mat"]:
        p = scores_dir / f"hyp_{subject_id}{ext}"
        if p.exists():
            return p
    return None


def load_aligned_hyp_annotations(
    raw,
    mat_path,
    include_movement=False,
):
    """
    Carrega o hyp_<subject>.mat e cria anotações alinhadas ao EDF.

    Returns
    -------
    annotations : mne.Annotations
        Anotações alinhadas ao início do EDF.

    hyp : numpy.ndarray
        Matriz original salva pelo script Octave.

    alignment : dict
        Informações utilizadas para validar o alinhamento.
    """
    data = loadmat(
        mat_path,
        simplify_cells=True,
    )

    if "hyp" not in data:
        raise KeyError(
            f"A variável 'hyp' não foi encontrada em {mat_path}."
        )

    if "start_time" not in data:
        raise KeyError(
            f"A variável 'start_time' não foi encontrada em {mat_path}."
        )

    hyp = np.asarray(data["hyp"], dtype=float)
    start_time = data["start_time"]

    if hyp.ndim != 2 or hyp.shape[1] < 2:
        raise ValueError(
            f"Formato inválido da matriz hyp: {hyp.shape}"
        )

    stages = hyp[:, 0].astype(int)
    relative_onsets = hyp[:, 1].astype(float)

    unknown_stages = set(np.unique(stages)) - set(STAGE_MAP)

    if unknown_stages:
        raise ValueError(
            f"Códigos de estágio desconhecidos: {unknown_stages}"
        )

    offset = calculate_annotation_offset(
        raw,
        start_time,
    )

    aligned_onsets = relative_onsets + offset

    keep = np.ones(len(stages), dtype=bool)

    if not include_movement:
        keep &= stages != 7

    stages_kept = stages[keep]
    onsets_kept = aligned_onsets[keep]

    descriptions = np.asarray(
        [STAGE_MAP[stage] for stage in stages_kept],
        dtype=str,
    )

    durations = np.full(
        len(onsets_kept),
        30.0,
        dtype=float,
    )

    annotations = mne.Annotations(
        onset=onsets_kept,
        duration=durations,
        description=descriptions,
        orig_time=None,
    )

    edf_duration = raw.n_times / raw.info["sfreq"]

    first_annotation_onset = (
        float(onsets_kept[0])
        if len(onsets_kept)
        else np.nan
    )

    last_annotation_end = (
        float(onsets_kept[-1] + durations[-1])
        if len(onsets_kept)
        else np.nan
    )

    alignment = {
        "offset_seconds": offset,
        "first_annotation_onset": first_annotation_onset,
        "last_annotation_end": last_annotation_end,
        "edf_duration": edf_duration,
        "initial_unscored_seconds": first_annotation_onset,
        "final_unscored_seconds": (
            edf_duration - last_annotation_end
        ),
        "number_of_annotations": len(annotations),
        "start_time": start_time,
    }

    return annotations, hyp, alignment




def calculate_annotation_offset(raw, start_time):
    """
    Calcula o deslocamento entre o início do EDF e a primeira
    anotação do hipnograma.
    """
    meas_date = raw.info["meas_date"]

    if meas_date is None:
        raise ValueError(
            "O EDF não possui raw.info['meas_date']; "
            "não é possível alinhar as anotações pelo horário."
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

    # Corrige passagem pela meia-noite.
    if offset < -12 * 3600:
        offset += 24 * 3600
    elif offset > 12 * 3600:
        offset -= 24 * 3600

    return float(offset)

def is_raw_edf(path: Path) -> bool:
    """
    Retorna True apenas para arquivos EDF de sinal bruto.

    Regras de exclusão:
      - Extensão não é exatamente ".edf" (ex: .edf.st, .edf.gz)
      - Nome contém "scor", "staging", "hypno", "annot", "event"
        (ex: subject-scoring.edf, subject_staging.edf)

    Exemplos:
      subject001.edf          → True   ✓
      subject001-scoring.edf  → False  ✗
      subject001.edf.st       → False  ✗
      subject001_staging.edf  → False  ✗
      subject001.hypno.edf    → False  ✗
    """
    if path.suffix.lower() != ".edf":
        return False

    name_lower = path.stem.lower()
    excluded_keywords = ["scor", "staging", "hypno", "annot", "event",
                         "label", "stage", "sleep_stage"]
    for kw in excluded_keywords:
        if kw in name_lower:
            return False

    return True

def list_raw_edfs(edf_dir: Path) -> List[Path]:
    """
    Lista apenas os EDFs brutos de um diretório, ignorando variantes
    de scoring/anotação.
    """
    all_edfs  = list(edf_dir.glob("*.edf"))
    raw_edfs  = [p for p in all_edfs if is_raw_edf(p)]
    skipped   = len(all_edfs) - len(raw_edfs)

    print(f"[preprocess] {len(all_edfs)} arquivos .edf encontrados em {edf_dir.name}/")
    if skipped > 0:
        skipped_names = [p.name for p in all_edfs if not is_raw_edf(p)]
        print(f"             {skipped} ignorados (scoring/anotação): "
              f"{skipped_names[:5]}{'...' if skipped > 5 else ''}")
    print(f"             {len(raw_edfs)} EDFs brutos para processar")

    return sorted(raw_edfs)


if __name__ == "__main__":

    print("iniciando parser de EDFs brutos para anotação de hipnograma...")
    raw_edfs = list_raw_edfs(PathConfig.EDF_DIR)
    if not raw_edfs:
        print("[preprocess] Nenhum EDF bruto encontrado. Verifique o diretório.")

    parser(
        Path("/Volumes/HD_EXTERNO/Workspace/AI/USP/dataset/capslpdb-1.0.0/rbd1.edf"), 
        "rbd1",
        annotations_csv_path=Path("/Volumes/HD_EXTERNO/Workspace/AI/USP/dataset/capslpdb-1.0.0-annotations/rbd1_rswa.csv")
    )
    #for i, edf_path in enumerate(raw_edfs):
    #    subject_id = edf_path.stem
    #    parser(edf_path, subject_id)
