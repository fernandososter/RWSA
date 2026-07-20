from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from ..data import SubjectData


def _subject_targets(subject: SubjectData, task: str) -> np.ndarray:
    if task == "staging":
        labels = subject.sleep_stages.detach().cpu().numpy().reshape(-1)
        return labels[labels >= 0].astype(np.int64, copy=False)
    if task == "rswa":
        labels = subject.rswa_labels.detach().cpu().numpy().reshape(-1)
        return labels[labels >= 0].astype(np.int64, copy=False)
    raise ValueError(f"Task desconhecida: {task!r}")


def stratified_group_folds(
    subjects: Sequence[SubjectData],
    *,
    n_splits: int = 5,
    seed: int = 42,
    task: str = "staging",
) -> Iterator[tuple[int, list[SubjectData], list[SubjectData]]]:
    """Gera folds estratificados em nível de mini-época e agrupados por sujeito.

    Cada mini-época fornece o rótulo usado na estratificação, enquanto ``groups``
    garante que todas as mini-épocas de um mesmo sujeito permaneçam no mesmo fold.
    """
    subjects = list(subjects)
    if len(subjects) < n_splits:
        raise ValueError(
            f"n_splits={n_splits} é maior que o número de sujeitos ({len(subjects)})."
        )

    y_parts: list[np.ndarray] = []
    group_parts: list[np.ndarray] = []
    for index, subject in enumerate(subjects):
        targets = _subject_targets(subject, task)
        if targets.size == 0:
            raise ValueError(f"{subject.subject_id}: nenhum rótulo válido para {task}.")
        y_parts.append(targets)
        group_parts.append(np.full(targets.shape[0], index, dtype=np.int32))

    y = np.concatenate(y_parts)
    groups = np.concatenate(group_parts)
    x = np.zeros(y.shape[0], dtype=np.uint8)

    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )
    for fold, (train_idx, val_idx) in enumerate(splitter.split(x, y, groups)):
        train_subject_indices = sorted(set(groups[train_idx].tolist()))
        val_subject_indices = sorted(set(groups[val_idx].tolist()))
        train_subjects = [subjects[i] for i in train_subject_indices]
        val_subjects = [subjects[i] for i in val_subject_indices]
        yield fold, train_subjects, val_subjects
