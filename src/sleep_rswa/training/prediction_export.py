from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

_STAGE_LABELS = ("W", "N1", "N2", "N3", "REM")
_RSWA_HEADS = ("tonic", "phasic", "any", "movement")


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> Path:
    _ensure_parent(path)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)
    return path


def save_staging_predictions_csv(
    path: str | Path,
    result: Mapping[str, Any],
    *,
    split: str,
    fold: int | None = None,
    source: str | None = None,
) -> Path:
    path = Path(path)
    probs = np.asarray(result["probabilities"], dtype=np.float32)
    expected = np.asarray(result["expected"], dtype=np.int64)
    prediction = np.asarray(result["prediction"], dtype=np.int64)
    subject_id = np.asarray(result["subject_id"], dtype=object)
    mini_epoch_index = np.asarray(result["mini_epoch_index"], dtype=np.int64)

    fieldnames = [
        "split",
        "source",
        "fold",
        "subject_id",
        "mini_epoch_index",
        "expected",
        "prediction",
        "correct",
        "predicted_probability",
    ] + [f"prob_{stage}" for stage in _STAGE_LABELS]

    rows: list[dict[str, Any]] = []
    for i in range(expected.shape[0]):
        pred_idx = int(prediction[i])
        row = {
            "split": split,
            "source": source,
            "fold": fold,
            "subject_id": str(subject_id[i]),
            "mini_epoch_index": int(mini_epoch_index[i]),
            "expected": int(expected[i]),
            "prediction": pred_idx,
            "correct": int(expected[i] == prediction[i]),
            "predicted_probability": float(probs[i, pred_idx]),
        }
        for stage_index, stage_name in enumerate(_STAGE_LABELS):
            row[f"prob_{stage_name}"] = float(probs[i, stage_index])
        rows.append(row)
    return _write_csv(path, fieldnames, rows)


def save_rswa_predictions_csv(
    path: str | Path,
    result: Mapping[str, Any],
    *,
    split: str,
    fold: int | None = None,
    source: str | None = None,
) -> Path:
    path = Path(path)
    subject_id = np.asarray(result["subject_id"], dtype=object)
    mini_epoch_index = np.asarray(result["mini_epoch_index"], dtype=np.int64)
    n_samples = int(mini_epoch_index.shape[0])

    fieldnames = [
        "split",
        "source",
        "fold",
        "subject_id",
        "mini_epoch_index",
    ]
    for head in _RSWA_HEADS:
        fieldnames.extend(
            [
                f"{head}_expected",
                f"{head}_prediction",
                f"{head}_probability",
                f"{head}_correct",
            ]
        )

    rows: list[dict[str, Any]] = []
    expected = {head: np.asarray(result[f"{head}_expected"], dtype=np.int64) for head in _RSWA_HEADS}
    prediction = {head: np.asarray(result[f"{head}_prediction"], dtype=np.int64) for head in _RSWA_HEADS}
    probability = {head: np.asarray(result[f"{head}_probability"], dtype=np.float32) for head in _RSWA_HEADS}

    for i in range(n_samples):
        row = {
            "split": split,
            "source": source,
            "fold": fold,
            "subject_id": str(subject_id[i]),
            "mini_epoch_index": int(mini_epoch_index[i]),
        }
        for head in _RSWA_HEADS:
            row[f"{head}_expected"] = int(expected[head][i])
            row[f"{head}_prediction"] = int(prediction[head][i])
            row[f"{head}_probability"] = float(probability[head][i])
            row[f"{head}_correct"] = int(expected[head][i] == prediction[head][i])
        rows.append(row)
    return _write_csv(path, fieldnames, rows)
