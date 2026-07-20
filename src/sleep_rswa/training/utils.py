from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from ..data import SubjectData, load_subject_directory


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA foi solicitada, mas não está disponível.")
    return device


def split_subjects(
    subjects: Sequence[SubjectData],
    val_fraction: float,
    seed: int,
) -> tuple[list[SubjectData], list[SubjectData]]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction deve estar entre 0 e 1.")
    if len(subjects) < 2:
        raise ValueError("São necessários ao menos dois sujeitos para divisão treino/validação.")

    indices = list(range(len(subjects)))
    random.Random(seed).shuffle(indices)
    n_val = max(1, round(len(indices) * val_fraction))
    n_val = min(n_val, len(indices) - 1)
    val_indices = set(indices[:n_val])

    train = [subject for i, subject in enumerate(subjects) if i not in val_indices]
    val = [subject for i, subject in enumerate(subjects) if i in val_indices]
    return train, val


def load_train_val_subjects(
    *,
    data_dir: str | Path | None,
    train_dir: str | Path | None,
    val_dir: str | Path | None,
    val_fraction: float,
    seed: int,
) -> tuple[list[SubjectData], list[SubjectData]]:
    if train_dir or val_dir:
        if not train_dir or not val_dir:
            raise ValueError("Use --train-dir e --val-dir juntos.")
        return load_subject_directory(train_dir), load_subject_directory(val_dir)

    if data_dir is None:
        raise ValueError("Informe --data-dir ou o par --train-dir/--val-dir.")

    return split_subjects(load_subject_directory(data_dir), val_fraction, seed)


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    metrics: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "metrics": metrics,
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state)
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


def write_history(path: str | Path, history: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
