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


def stratified_group_holdout(
    subjects: Sequence[SubjectData],
    *,
    test_fraction: float,
    seed: int = 42,
    task: str = "staging",
) -> tuple[list[SubjectData], list[SubjectData]]:
    """Separa um conjunto de teste fixo a nível de sujeito, antes da CV.

    A separação usa StratifiedGroupKFold internamente (mesma lógica dos folds):
    escolhe ``n_splits = round(1 / test_fraction)`` e usa a partição de
    validação do primeiro fold como teste. Isso garante que:
      - nenhum sujeito aparece simultaneamente no teste e no pool de CV
        (agrupamento por sujeito);
      - a distribuição de estágios do teste é aproximadamente a do dataset
        (estratificação em nível de mini-época).

    Retorna ``(train_pool, test_subjects)``. O ``train_pool`` é o que deve ser
    passado a :func:`stratified_group_folds` para a CV.
    """
    subjects = list(subjects)
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction deve estar entre 0 e 1 (exclusivo).")

    n_splits = max(2, round(1.0 / test_fraction))
    if len(subjects) < n_splits:
        raise ValueError(
            f"test_fraction={test_fraction} exige ao menos {n_splits} sujeitos, "
            f"mas há apenas {len(subjects)}. Aumente test_fraction ou use "
            f"--test-dir para um conjunto de teste externo."
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

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_idx, test_idx = next(iter(splitter.split(x, y, groups)))
    train_subject_indices = sorted(set(groups[train_idx].tolist()))
    test_subject_indices = sorted(set(groups[test_idx].tolist()))
    train_pool = [subjects[i] for i in train_subject_indices]
    test_subjects = [subjects[i] for i in test_subject_indices]
    return train_pool, test_subjects
