from __future__ import annotations

from collections.abc import Iterable
from contextlib import nullcontext
from typing import Any
from tqdm import tqdm
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from ..distribution import StageDistribution

from ..metrics import rswa_metrics, staging_metrics
from .losses import RSWALoss, StagingLoss


def _autocast_context(device: torch.device, enabled: bool):
    if not enabled:
        return nullcontext()
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _safe_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def _binary_distribution(labels: np.ndarray) -> dict[str, dict[str, float | int]]:
    """Distribuição binária de movimento no formato de ``StageDistribution.as_dict``:
    ``{"Negative": {count, percentage}, "Positive": {count, percentage}}``.
    Reaproveita ``print_stage_distribution``/``format_stage_distribution``.
    """
    labels = np.asarray(labels).reshape(-1).astype(np.int64)
    total = int(labels.size)
    positives = int((labels == 1).sum())
    negatives = total - positives
    return {
        "Negative": {
            "count": negatives,
            "percentage": (100.0 * negatives / total) if total else 0.0,
        },
        "Positive": {
            "count": positives,
            "percentage": (100.0 * positives / total) if total else 0.0,
        },
    }


def run_staging_epoch(
    model: torch.nn.Module,
    loader: Iterable[dict[str, Any]],
    criterion: StagingLoss,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    amp: bool = True,
    grad_clip: float | None = 1.0,
    prediction_logger: Any | None = None,
    epoch: int | None = None,
) -> dict[str, Any]:
    
    training = optimizer is not None
    model.train(training)
    losses: list[float] = []
    all_targets: list[torch.Tensor] = []
    all_predictions: list[torch.Tensor] = []

    target_distribution = StageDistribution()
    prediction_distribution = StageDistribution()

    if prediction_logger is not None:
        if training:
            raise ValueError("prediction_logger deve ser usado somente na validação.")
        if epoch is None:
            raise ValueError("epoch é obrigatório quando prediction_logger é informado.")
        prediction_logger.start_epoch(epoch)

  
    for batch in tqdm(loader, desc="Running staging epoch", unit="batch"):
        signals = batch["signals"].to(device, non_blocking=True)
        targets = batch["sleep_stages"].to(device, non_blocking=True)
        padding_mask = batch["padding_mask"].to(device, non_blocking=True)
        valid_mask = batch["staging_valid"].to(device, non_blocking=True) & padding_mask

        if not valid_mask.any():
            continue

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with _autocast_context(device, amp):
                logits = model(signals, mask=padding_mask)
                loss = criterion(logits, targets, valid_mask)

            if training:
                loss.backward()
                if grad_clip is not None:
                    clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

        predictions = logits.argmax(dim=-1)
        losses.append(float(loss.detach().cpu()))
        all_targets.append(targets[valid_mask].detach().cpu())
        all_predictions.append(predictions[valid_mask].detach().cpu())

        target_distribution.update(
            targets,
            mask=valid_mask,
        )

        prediction_distribution.update(
            predictions,
            mask=valid_mask,
        )

        if prediction_logger is not None:
            prediction_logger.log_staging_batch(
                subject_ids=batch["subject_ids"],
                valid_mask=valid_mask,
                expected=targets,
                prediction=predictions,
            )

    if prediction_logger is not None:
        prediction_logger.end_epoch()

    if not all_targets:
        raise RuntimeError("Nenhum rótulo válido de staging foi encontrado nesta época.")

    targets_np = torch.cat(all_targets).numpy()
    predictions_np = torch.cat(all_predictions).numpy()
    result = staging_metrics(targets_np, predictions_np)
    
    metrics: dict[str, Any] = {
        "loss": _safe_mean(losses),
        **{
            key: float(value)
            for key, value in result.items()
        },
        "target_distribution": target_distribution.as_dict(),
        "prediction_distribution": prediction_distribution.as_dict(),
    }

    return metrics


_HEADS = ("tonic", "phasic", "any")


def run_rswa_epoch(
    model: torch.nn.Module,
    loader: Iterable[dict[str, Any]],
    criterion: RSWALoss,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    amp: bool = True,
    grad_clip: float | None = 1.0,
    threshold: float | dict[str, float] = 0.5,
) -> dict[str, float]:
    """Roda uma epoca de treino/validacao das 3 cabecas (tonic/phasic/any).

    ``threshold`` pode ser um unico float (aplicado as 3 cabecas -- usado
    durante o treino so para monitorar F1/kappa) ou um dict
    ``{"tonic": t, "phasic": t, "any": t}`` com limiares por cabeca
    (usado na avaliacao final, apos a selecao de limiar por cabeca).
    """
    if isinstance(threshold, dict):
        thr = {h: float(threshold[h]) for h in _HEADS}
    else:
        thr = {h: float(threshold) for h in _HEADS}

    training = optimizer is not None
    model.train(training)
    losses: list[float] = []
    head_losses: dict[str, list[float]] = {h: [] for h in _HEADS}
    targets_all: dict[str, list[torch.Tensor]] = {h: [] for h in _HEADS}
    preds_all: dict[str, list[torch.Tensor]] = {h: [] for h in _HEADS}

    for batch in tqdm(loader, desc="Running RSWA epoch", unit="batch"):
        emg = batch["emg_center"].to(device, non_blocking=True)
        tonic_targets = batch["tonic_labels"].to(device, non_blocking=True)
        phasic_targets = batch["phasic_labels"].to(device, non_blocking=True)
        any_targets = batch["any_labels"].to(device, non_blocking=True)
        padding_mask = batch["padding_mask"].to(device, non_blocking=True)
        valid_mask = batch["rswa_valid"].to(device, non_blocking=True) & padding_mask

        if not valid_mask.any():
            continue

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with _autocast_context(device, amp):
                outputs = model(emg, mask=padding_mask)
                loss, per_head = criterion(
                    outputs, tonic_targets, phasic_targets, any_targets, valid_mask
                )

            if training:
                loss.backward()
                if grad_clip is not None:
                    clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

        losses.append(float(loss.detach().cpu()))
        for h in _HEADS:
            head_losses[h].append(float(per_head[f"{h}_loss"].cpu()))

        head_targets = {"tonic": tonic_targets, "phasic": phasic_targets, "any": any_targets}
        for h in _HEADS:
            preds = (torch.sigmoid(outputs[f"{h}_logits"]) >= thr[h]).long()
            targets_all[h].append(head_targets[h][valid_mask].long().detach().cpu())
            preds_all[h].append(preds[valid_mask].detach().cpu())

    if not targets_all["tonic"]:
        raise RuntimeError(
            "Nenhum rótulo RSWA válido foi encontrado. Verifique rswa_conf, "
            "min_confidence e rem_mask_only."
        )

    targets_np = {h: torch.cat(targets_all[h]).numpy() for h in _HEADS}
    preds_np = {h: torch.cat(preds_all[h]).numpy() for h in _HEADS}
    result = rswa_metrics(
        targets_np["tonic"], preds_np["tonic"],
        targets_np["phasic"], preds_np["phasic"],
        targets_np["any"], preds_np["any"],
    )
    metrics: dict[str, Any] = {
        "loss": _safe_mean(losses),
        **{k: float(v) for k, v in result.items()},
    }
    for h in _HEADS:
        metrics[f"{h}_loss"] = _safe_mean(head_losses[h])
        metrics[f"{h}_target_distribution"] = _binary_distribution(targets_np[h])
        metrics[f"{h}_prediction_distribution"] = _binary_distribution(preds_np[h])
    return metrics


def evaluate_joint(
    model: torch.nn.Module,
    loader: Iterable[dict[str, Any]],
    staging_criterion: StagingLoss,
    rswa_criterion: RSWALoss,
    device: torch.device,
    amp: bool = True,
    threshold: float | dict[str, float] = 0.5,
) -> dict[str, float]:
    if isinstance(threshold, dict):
        thr = {h: float(threshold[h]) for h in _HEADS}
    else:
        thr = {h: float(threshold) for h in _HEADS}

    model.eval()
    stage_losses: list[float] = []
    rswa_losses: list[float] = []
    stage_targets_all: list[torch.Tensor] = []
    stage_preds_all: list[torch.Tensor] = []
    targets_all: dict[str, list[torch.Tensor]] = {h: [] for h in _HEADS}
    preds_all: dict[str, list[torch.Tensor]] = {h: [] for h in _HEADS}

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating joint", unit="batch"):
            signals = batch["signals"].to(device, non_blocking=True)
            emg = batch["emg_center"].to(device, non_blocking=True)
            padding_mask = batch["padding_mask"].to(device, non_blocking=True)
            stage_targets = batch["sleep_stages"].to(device, non_blocking=True)
            tonic_targets = batch["tonic_labels"].to(device, non_blocking=True)
            phasic_targets = batch["phasic_labels"].to(device, non_blocking=True)
            any_targets = batch["any_labels"].to(device, non_blocking=True)
            stage_valid = batch["staging_valid"].to(device, non_blocking=True) & padding_mask
            rswa_valid = batch["rswa_valid"].to(device, non_blocking=True) & padding_mask

            with _autocast_context(device, amp):
                outputs = model(signals, emg, mask=padding_mask)

            if stage_valid.any():
                stage_loss = staging_criterion(
                    outputs["staging_logits"], stage_targets, stage_valid
                )
                stage_preds = outputs["staging_logits"].argmax(dim=-1)
                stage_losses.append(float(stage_loss.cpu()))
                stage_targets_all.append(stage_targets[stage_valid].cpu())
                stage_preds_all.append(stage_preds[stage_valid].cpu())

            if rswa_valid.any():
                rswa_loss, _ = rswa_criterion(
                    outputs, tonic_targets, phasic_targets, any_targets, rswa_valid
                )
                rswa_losses.append(float(rswa_loss.cpu()))
                head_targets = {"tonic": tonic_targets, "phasic": phasic_targets, "any": any_targets}
                for h in _HEADS:
                    preds = (torch.sigmoid(outputs[f"{h}_logits"]) >= thr[h]).long()
                    targets_all[h].append(head_targets[h][rswa_valid].long().cpu())
                    preds_all[h].append(preds[rswa_valid].cpu())

    metrics: dict[str, Any] = {}
    if stage_targets_all:
        stage_t = torch.cat(stage_targets_all).numpy()
        stage_p = torch.cat(stage_preds_all).numpy()
        stage = staging_metrics(stage_t, stage_p)
        metrics.update({f"staging_{k}": float(v) for k, v in stage.items()})
        metrics["staging_loss"] = _safe_mean(stage_losses)
        st_target = StageDistribution()
        st_pred = StageDistribution()
        st_target.update(torch.as_tensor(stage_t))
        st_pred.update(torch.as_tensor(stage_p))
        metrics["staging_target_distribution"] = st_target.as_dict()
        metrics["staging_prediction_distribution"] = st_pred.as_dict()
    if targets_all["tonic"]:
        targets_np = {h: torch.cat(targets_all[h]).numpy() for h in _HEADS}
        preds_np = {h: torch.cat(preds_all[h]).numpy() for h in _HEADS}
        rswa = rswa_metrics(
            targets_np["tonic"], preds_np["tonic"],
            targets_np["phasic"], preds_np["phasic"],
            targets_np["any"], preds_np["any"],
        )
        metrics.update({f"rswa_{k}": float(v) for k, v in rswa.items()})
        metrics["rswa_loss"] = _safe_mean(rswa_losses)
        for h in _HEADS:
            metrics[f"{h}_target_distribution"] = _binary_distribution(targets_np[h])
            metrics[f"{h}_prediction_distribution"] = _binary_distribution(preds_np[h])
    return metrics


def collect_rswa_predictions(
    model: torch.nn.Module,
    loader: Iterable[dict[str, Any]],
    device: torch.device,
    *,
    amp: bool = True,
    threshold: float | dict[str, float] = 0.5,
) -> dict[str, np.ndarray]:
    """Coleta predicoes das 3 cabecas (tonic/phasic/any), por mini-epoca valida.

    Retorna, para cada cabeca h em {tonic,phasic,any}:
      ``{h}_expected``, ``{h}_probability``, ``{h}_prediction`` (limiar por
      cabeca via ``threshold[h]`` se dict, senao o mesmo float para as 3).
    Mais ``subject_id``/``mini_epoch_index``, alinhados por linha, e os
    aliases historicos ``movement_expected``/``movement_probability``/
    ``movement_prediction`` (uniao das 3 cabecas) para compat.
    """
    if isinstance(threshold, dict):
        thr = {h: float(threshold[h]) for h in _HEADS}
    else:
        thr = {h: float(threshold) for h in _HEADS}

    model.eval()
    expected: dict[str, list[np.ndarray]] = {h: [] for h in _HEADS}
    probability: dict[str, list[np.ndarray]] = {h: [] for h in _HEADS}
    subject_ids: list[str] = []
    mini_indices: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            emg = batch["emg_center"].to(device, non_blocking=True)
            padding_mask = batch["padding_mask"].to(device, non_blocking=True)
            valid_mask = batch["rswa_valid"].to(device, non_blocking=True) & padding_mask
            if not valid_mask.any():
                continue
            with _autocast_context(device, amp):
                outputs = model(emg, mask=padding_mask)

            valid_cpu = valid_mask.detach().cpu()
            probs_cpu = {h: torch.sigmoid(outputs[f"{h}_logits"].float()).detach().cpu() for h in _HEADS}
            targets_cpu = {
                "tonic": batch["tonic_labels"].detach().cpu(),
                "phasic": batch["phasic_labels"].detach().cpu(),
                "any": batch["any_labels"].detach().cpu(),
            }
            batch_subject_ids = batch["subject_ids"]

            for b, subject_id in enumerate(batch_subject_ids):
                idx = torch.nonzero(valid_cpu[b], as_tuple=False).flatten()
                if idx.numel() == 0:
                    continue
                for h in _HEADS:
                    expected[h].append(targets_cpu[h][b, idx].numpy().astype(np.int64, copy=False))
                    probability[h].append(probs_cpu[h][b, idx].numpy().astype(np.float32, copy=False))
                subject_ids.extend([str(subject_id)] * int(idx.numel()))
                mini_indices.append(idx.numpy().astype(np.int64, copy=False))

    if not expected["tonic"]:
        raise RuntimeError("Nenhuma predição RSWA válida foi encontrada.")

    result: dict[str, np.ndarray] = {
        "subject_id": np.asarray(subject_ids, dtype=object),
        "mini_epoch_index": np.concatenate(mini_indices),
    }
    prob_by_head: dict[str, np.ndarray] = {}
    exp_by_head: dict[str, np.ndarray] = {}
    for h in _HEADS:
        exp_arr = np.concatenate(expected[h])
        prob_arr = np.concatenate(probability[h])
        exp_by_head[h] = exp_arr
        prob_by_head[h] = prob_arr
        result[f"{h}_expected"] = exp_arr
        result[f"{h}_probability"] = prob_arr
        result[f"{h}_prediction"] = (prob_arr >= thr[h]).astype(np.int64, copy=False)

    # Aliases historicos "movement" = uniao das 3 cabecas.
    movement_expected = (
        (exp_by_head["tonic"] > 0) | (exp_by_head["phasic"] > 0) | (exp_by_head["any"] > 0)
    ).astype(np.int64, copy=False)
    movement_prediction = (
        (result["tonic_prediction"] > 0) | (result["phasic_prediction"] > 0) | (result["any_prediction"] > 0)
    ).astype(np.int64, copy=False)
    result["movement_expected"] = movement_expected
    result["movement_prediction"] = movement_prediction
    result["movement_probability"] = np.maximum.reduce(
        [prob_by_head["tonic"], prob_by_head["phasic"], prob_by_head["any"]]
    )
    return result


def collect_staging_predictions(
    model: torch.nn.Module,
    loader: Iterable[dict[str, Any]],
    device: torch.device,
    *,
    amp: bool = True,
    num_classes: int = 5,
) -> dict[str, np.ndarray]:
    """Coleta predições de staging de um modelo já treinado num loader.

    Diferente de :func:`run_staging_epoch`, não calcula loss nem depende de
    critério: percorre o loader em modo avaliação e devolve, por mini-época
    válida, o rótulo esperado, a predição (argmax) e as **probabilidades**
    softmax por classe. As probabilidades permitem ensemble (média entre folds)
    antes do argmax final.

    Retorna arrays alinhados por mini-época válida:
      - ``expected``      [N]         int64
      - ``prediction``    [N]         int64
      - ``probabilities`` [N, C]      float32
      - ``subject_id``    [N]         str (object array)
      - ``mini_epoch_index`` [N]      int64
    """
    model.eval()
    expected: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    subject_ids: list[str] = []
    mini_indices: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            signals = batch["signals"].to(device, non_blocking=True)
            targets = batch["sleep_stages"].to(device, non_blocking=True)
            padding_mask = batch["padding_mask"].to(device, non_blocking=True)
            valid_mask = batch["staging_valid"].to(device, non_blocking=True) & padding_mask
            if not valid_mask.any():
                continue
            with _autocast_context(device, amp):
                logits = model(signals, mask=padding_mask)
            probs = torch.softmax(logits.float(), dim=-1)

            valid_cpu = valid_mask.detach().cpu()
            probs_cpu = probs.detach().cpu()
            targets_cpu = targets.detach().cpu()
            batch_subject_ids = batch["subject_ids"]

            for b, subject_id in enumerate(batch_subject_ids):
                idx = torch.nonzero(valid_cpu[b], as_tuple=False).flatten()
                if idx.numel() == 0:
                    continue
                expected.append(targets_cpu[b, idx].numpy().astype(np.int64, copy=False))
                probabilities.append(probs_cpu[b, idx].numpy().astype(np.float32, copy=False))
                subject_ids.extend([str(subject_id)] * int(idx.numel()))
                mini_indices.append(idx.numpy().astype(np.int64, copy=False))

    if not expected:
        raise RuntimeError("Nenhuma predição válida de staging foi encontrada.")

    prob_arr = np.concatenate(probabilities, axis=0)
    return {
        "expected": np.concatenate(expected),
        "prediction": prob_arr.argmax(axis=1).astype(np.int64, copy=False),
        "probabilities": prob_arr,
        "subject_id": np.asarray(subject_ids, dtype=object),
        "mini_epoch_index": np.concatenate(mini_indices),
    }

