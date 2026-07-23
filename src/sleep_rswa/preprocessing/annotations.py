"""
Leitura das anotacoes de RSWA a partir do CSV (<subject>_rswa.csv).

Duas responsabilidades:
  1. load_subject_annotations_from_csv -> mne.Annotations (para inspecao/plot no raw)
  2. rasterize_rswa_annotations (em rswa_labels.py) -> rotulos por mini-epoca

Convertido do notebook Parser_Exames (celula 12).
"""
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Dict, Optional

import mne

from .config import CSV_ANNOTATION_COLUMNS


def find_annotations_csv_file(subject_id: str, csv_dir: Path) -> Optional[Path]:
    """Procura <subject>_rswa.csv em csv_dir. None se ausente."""
    csv_dir = Path(csv_dir)
    if not csv_dir.exists():
        raise FileNotFoundError(f"Diretorio CSV nao encontrado: {csv_dir}")
    if not csv_dir.is_dir():
        raise NotADirectoryError(f"Caminho CSV nao e uma pasta: {csv_dir}")

    csv_path = csv_dir / f"{subject_id}_rswa.csv"
    return csv_path if csv_path.exists() else None


def load_subject_annotations_from_csv(
    csv_path: Path,
    subject_id: str,
    orig_time,
) -> mne.Annotations:
    """
    Le eventos (onset_s, duration_s, type) do CSV para o sujeito e devolve
    mne.Annotations (onsets no referencial do EDF bruto). Valida colunas e valores.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Arquivo CSV nao encontrado: {csv_path}")

    rows = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing_columns = CSV_ANNOTATION_COLUMNS - fieldnames
        if missing_columns:
            raise ValueError(
                f"CSV invalido: colunas ausentes {sorted(missing_columns)}. "
                f"Esperado: {sorted(CSV_ANNOTATION_COLUMNS)}"
            )

        for line_number, row in enumerate(reader, start=2):
            if str(row.get("subject_id", "")).strip() != subject_id:
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
                    f"Linha {line_number}: onset_s/duration_s invalidos "
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
        onset=[r[0] for r in rows],
        duration=[r[1] for r in rows],
        description=[r[2] for r in rows],
        orig_time=orig_time,
    )


def count_annotations_by_description(annotations: mne.Annotations) -> Dict[str, int]:
    counts = Counter(str(d) for d in annotations.description)
    return dict(sorted(counts.items()))
