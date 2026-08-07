"""Descrição de dados por split e fases de teste held-out (movimento e staging).

Centraliza o que ``train_rswa``/``train_joint`` precisam além da CV:

- :func:`describe_split` — resumo de dados de um split (nº de exames,
  distribuição de estágios em %, distribuição de movimento em %), consistente
  com a máscara de validade que o ``SleepAnalysisDataset`` usa no treino.
- :func:`evaluate_movement_test_set` / :func:`evaluate_staging_test_set` —
  avaliam o conjunto de TESTE held-out com os checkpoints de cada fold,
  reportando métricas por fold e o ENSEMBLE (média das probabilidades entre
  folds). Nenhum fold viu o teste no treino nem na seleção de checkpoint, então
  o ensemble é uma estimativa honesta de generalização.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, f1_score

from .engine import collect_rswa_predictions, collect_staging_predictions
from .plots import plot_confusion_matrix
from .utils import load_checkpoint


def describe_split(dataset: Any) -> dict[str, Any]:
    """Resumo de dados de um split (a partir de um ``SleepAnalysisDataset``).

    Retorna nº de exames, distribuição de estágios do sono (contagem + %) e
    distribuição de movimento 'any' (tônico OU fásico) dentro da máscara de
    validade RSWA do dataset.
    """
    return {
        "exams": len(dataset),
        "stage_distribution": dataset.stage_distribution().as_dict(),
        "movement": dataset.movement_distribution(),
    }


def format_split_description(name: str, desc: Mapping[str, Any]) -> str:
    """Uma linha legível para o log a partir de :func:`describe_split`."""
    stages = " | ".join(
        f"{stage}={values['count']:,}({values['percentage']:.1f}%)"
        for stage, values in desc["stage_distribution"].items()
    )
    m = desc["movement"]
    return (
        f"{name}: exames={desc['exams']} | "
        f"estágios[{stages}] | "
        f"movimento={m['movement_positive']:,}/{m['evaluable_mini_epochs']:,} "
        f"({m['pct_movement_of_evaluable']:.2f}% das avaliáveis; "
        f"avaliáveis={m['pct_evaluable_of_total']:.1f}% do total de "
        f"{m['total_mini_epochs']:,} mini-épocas)"
    )


def _binary_metrics(expected: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "f1": float(f1_score(expected, prediction, zero_division=0)),
        "kappa": float(cohen_kappa_score(expected, prediction)),
        "n_samples": int(expected.size),
        "n_positives": int(expected.sum()),
    }


_HEADS = ("tonic", "phasic", "any")


def evaluate_movement_test_set(
    *,
    test_loader: Any,
    fold_checkpoints: list[dict[str, Any]],
    build_model: Callable[[], torch.nn.Module],
    device: torch.device,
    logger: Any,
    figures_dir: Path,
    amp: bool = True,
    threshold: float | dict[str, float] = 0.5,
) -> dict[str, Any]:
    """Avalia as 3 cabecas (tonic/phasic/any) no teste held-out, por fold + ensemble.

    ``fold_checkpoints``: lista de ``{"fold": int, "best_checkpoint": Path}``.
    ``threshold`` pode ser um float unico (mesmo limiar nas 3 cabecas) ou um
    dict ``{"tonic": t, "phasic": t, "any": t}`` com o limiar selecionado por
    cabeca (ver seleção de limiar por cabeça). O ensemble alinha os folds por
    chave estável (subject_id#mini_epoch_index) e faz a média das
    probabilidades antes do threshold -- por cabeça, independentemente.
    Retorna um dict com uma entrada por cabeça (``tonic``/``phasic``/``any``)
    e o alias histórico ``movement`` (união das 3, mesma semântica de antes).
    """
    if isinstance(threshold, dict):
        thr = {h: float(threshold[h]) for h in _HEADS}
    else:
        thr = {h: float(threshold) for h in _HEADS}

    figures_dir.mkdir(parents=True, exist_ok=True)
    labels = [0, 1]
    display_labels = ["Negative", "Positive"]

    logger.info("=" * 80)
    logger.info(f"FASE DE TESTE (tonic/phasic/any) | folds={len(fold_checkpoints)}")
    logger.info("=" * 80)

    per_fold: dict[str, list[dict[str, float]]] = {h: [] for h in (*_HEADS, "movement")}
    prob_sum: dict[str, np.ndarray | None] = {h: None for h in (*_HEADS, "movement")}
    ref_expected: dict[str, np.ndarray | None] = {h: None for h in (*_HEADS, "movement")}
    ref_keys: np.ndarray | None = None

    for entry in fold_checkpoints:
        fold = entry["fold"]
        checkpoint_path = Path(entry["best_checkpoint"])
        if not checkpoint_path.exists():
            logger.info(f"Fold {fold}: best.pt ausente ({checkpoint_path}); pulando no teste.")
            continue

        model = build_model().to(device)
        load_checkpoint(checkpoint_path, model, device)
        result = collect_rswa_predictions(model, test_loader, device, amp=amp, threshold=thr)

        keys = np.array(
            [f"{s}#{i}" for s, i in zip(result["subject_id"], result["mini_epoch_index"])],
            dtype=object,
        )

        for h in (*_HEADS, "movement"):
            expected = result[f"{h}_expected"]
            prediction = result[f"{h}_prediction"]
            probs = result[f"{h}_probability"]

            m = _binary_metrics(expected, prediction)
            logger.info(f"TESTE fold {fold} ({h}): f1={m['f1']:.4f} kappa={m['kappa']:.4f} "
                        f"pos={m['n_positives']}/{m['n_samples']}")
            per_fold[h].append({"fold": int(fold), **m})

            plot_confusion_matrix(
                expected, prediction, figures_dir / f"confusion_matrix_{h}_test_fold_{fold}.png",
                labels=labels, display_labels=display_labels, title=f"{h.capitalize()} TEST - Fold {fold}",
            )

            if prob_sum[h] is None:
                prob_sum[h] = probs.copy()
                ref_expected[h] = expected
            else:
                if ref_keys is not None and not np.array_equal(keys, ref_keys):
                    order = {k: j for j, k in enumerate(keys)}
                    idx = np.array([order[k] for k in ref_keys], dtype=np.int64)
                    probs = probs[idx]
                prob_sum[h] = prob_sum[h] + probs

        ref_keys = keys

    test_summary: dict[str, Any] = {"n_subjects": len(test_loader.dataset)}

    for h in (*_HEADS, "movement"):
        head_summary: dict[str, Any] = {"per_fold": per_fold[h]}
        if prob_sum[h] is not None and ref_expected[h] is not None:
            n_folds = len(per_fold[h])
            ensemble_pred = (prob_sum[h] / n_folds >= thr.get(h, 0.5)).astype(np.int64)
            m = _binary_metrics(ref_expected[h], ensemble_pred)
            logger.info(f"TESTE ENSEMBLE {h} ({n_folds} folds): f1={m['f1']:.4f} kappa={m['kappa']:.4f}")

            plot_confusion_matrix(
                ref_expected[h], ensemble_pred, figures_dir / f"confusion_matrix_{h}_test_ensemble.png",
                labels=labels, display_labels=display_labels, title=f"{h.capitalize()} TEST - Ensemble",
            )
            plot_confusion_matrix(
                ref_expected[h], ensemble_pred, figures_dir / f"confusion_matrix_{h}_test_ensemble_normalized.png",
                labels=labels, display_labels=display_labels, normalize="true",
                title=f"{h.capitalize()} TEST - Ensemble normalizada",
            )
            f1_vals = np.asarray([f["f1"] for f in per_fold[h]], dtype=np.float64)
            head_summary["per_fold_f1_mean"] = float(f1_vals.mean()) if f1_vals.size else None
            head_summary["per_fold_f1_std"] = float(f1_vals.std(ddof=1)) if f1_vals.size > 1 else 0.0
            head_summary["ensemble"] = {"n_folds": n_folds, **m}
        test_summary[h] = head_summary

    return test_summary


def evaluate_staging_test_set(
    *,
    test_loader: Any,
    fold_checkpoints: list[dict[str, Any]],
    build_model: Callable[[], torch.nn.Module],
    device: torch.device,
    logger: Any,
    figures_dir: Path,
    amp: bool = True,
) -> dict[str, Any]:
    """Avalia o staging no teste held-out com o best.pt de cada fold + ensemble
    (média das probabilidades softmax, alinhadas por mini-época)."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    labels = [0, 1, 2, 3, 4]
    display_labels = ["W", "N1", "N2", "N3", "REM"]

    logger.info("=" * 80)
    logger.info(f"FASE DE TESTE (staging) | folds={len(fold_checkpoints)}")
    logger.info("=" * 80)

    per_fold: list[dict[str, float]] = []
    prob_sum: np.ndarray | None = None
    ref_expected: np.ndarray | None = None
    ref_keys: np.ndarray | None = None

    for entry in fold_checkpoints:
        fold = entry["fold"]
        checkpoint_path = Path(entry["best_checkpoint"])
        if not checkpoint_path.exists():
            logger.info(f"Fold {fold}: staging best.pt ausente ({checkpoint_path}); pulando no teste.")
            continue

        model = build_model().to(device)
        load_checkpoint(checkpoint_path, model, device)
        result = collect_staging_predictions(model, test_loader, device, amp=amp)

        expected = result["expected"]
        prediction = result["prediction"]
        probs = result["probabilities"]
        keys = np.array(
            [f"{s}#{i}" for s, i in zip(result["subject_id"], result["mini_epoch_index"])],
            dtype=object,
        )

        f1 = float(f1_score(expected, prediction, average="macro", zero_division=0))
        kappa = float(cohen_kappa_score(expected, prediction))
        bacc = float(balanced_accuracy_score(expected, prediction))
        logger.info(f"TESTE fold {fold} (staging): f1_macro={f1:.4f} kappa={kappa:.4f} balanced_acc={bacc:.4f}")
        per_fold.append({"fold": int(fold), "f1_macro": f1, "kappa": kappa,
                         "balanced_accuracy": bacc, "n_samples": int(expected.size)})

        plot_confusion_matrix(
            expected, prediction, figures_dir / f"confusion_matrix_staging_test_fold_{fold}.png",
            labels=labels, display_labels=display_labels, title=f"Staging TEST - Fold {fold}",
        )

        if prob_sum is None:
            prob_sum = probs.copy()
            ref_expected = expected
            ref_keys = keys
        else:
            if not np.array_equal(keys, ref_keys):
                order = {k: j for j, k in enumerate(keys)}
                idx = np.array([order[k] for k in ref_keys], dtype=np.int64)
                probs = probs[idx]
            prob_sum = prob_sum + probs

    test_summary: dict[str, Any] = {"n_subjects": len(test_loader.dataset), "per_fold": per_fold}

    if prob_sum is not None and ref_expected is not None:
        n_folds = len(per_fold)
        ensemble_pred = prob_sum.argmax(axis=1).astype(np.int64)
        ens_f1 = float(f1_score(ref_expected, ensemble_pred, average="macro", zero_division=0))
        ens_kappa = float(cohen_kappa_score(ref_expected, ensemble_pred))
        ens_bacc = float(balanced_accuracy_score(ref_expected, ensemble_pred))
        logger.info(f"TESTE ENSEMBLE staging ({n_folds} folds): f1_macro={ens_f1:.4f} "
                    f"kappa={ens_kappa:.4f} balanced_acc={ens_bacc:.4f}")

        plot_confusion_matrix(
            ref_expected, ensemble_pred, figures_dir / "confusion_matrix_staging_test_ensemble.png",
            labels=labels, display_labels=display_labels, title="Staging TEST - Ensemble",
        )
        plot_confusion_matrix(
            ref_expected, ensemble_pred, figures_dir / "confusion_matrix_staging_test_ensemble_normalized.png",
            labels=labels, display_labels=display_labels, normalize="true", title="Staging TEST - Ensemble normalizada",
        )
        f1_vals = np.asarray([f["f1_macro"] for f in per_fold], dtype=np.float64)
        test_summary["per_fold_f1_macro_mean"] = float(f1_vals.mean()) if f1_vals.size else None
        test_summary["per_fold_f1_macro_std"] = float(f1_vals.std(ddof=1)) if f1_vals.size > 1 else 0.0
        test_summary["ensemble"] = {"n_folds": n_folds, "n_samples": int(ref_expected.size),
                                    "f1_macro": ens_f1, "kappa": ens_kappa, "balanced_accuracy": ens_bacc}

    return test_summary
