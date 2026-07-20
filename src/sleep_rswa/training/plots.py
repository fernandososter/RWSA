from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

def plot_training_curves(
    history: list[dict[str, float]],
    output_path: str | Path,
    *,
    f1_key: str,
    kappa_key: str,
    title: str,
) -> Path:
    if not history:
        raise ValueError("Histórico vazio; não é possível gerar os gráficos.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = [int(row["epoch"]) for row in history]

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(10, 11),
        sharex=True,
    )

    # Loss
    axes[0].plot(
        epochs,
        [row["train_loss"] for row in history],
        label="Train loss",
    )
    axes[0].plot(
        epochs,
        [row["val_loss"] for row in history],
        label="Validation loss",
    )
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    # F1
    axes[1].plot(
        epochs,
        [row[f"train_{f1_key}"] for row in history],
        label="Train F1",
    )
    axes[1].plot(
        epochs,
        [row[f"val_{f1_key}"] for row in history],
        label="Validation F1",
    )
    axes[1].set_ylabel("F1")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    # Cohen's Kappa
    axes[2].plot(
        epochs,
        [row[f"train_{kappa_key}"] for row in history],
        label="Train Kappa",
    )
    axes[2].plot(
        epochs,
        [row[f"val_{kappa_key}"] for row in history],
        label="Validation Kappa",
    )
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Cohen's Kappa")
    axes[2].legend()
    axes[2].grid(alpha=0.25)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    return output_path

def plot_confusion_matrix(
    expected: Sequence[int] | np.ndarray,
    prediction: Sequence[int] | np.ndarray,
    output_path: str | Path,
    *,
    labels: Sequence[int],
    display_labels: Sequence[str],
    title: str,
    normalize: str | None = None,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matrix = confusion_matrix(expected, prediction, labels=list(labels), normalize=normalize)
    fig, ax = plt.subplots(figsize=(8, 7))
    disp = ConfusionMatrixDisplay(matrix, display_labels=list(display_labels))
    disp.plot(ax=ax, values_format=".2f" if normalize else "d", colorbar=True)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path
