from __future__ import annotations

from collections.abc import Iterable
from contextlib import nullcontext
from typing import Any
from tqdm import tqdm
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from sleep_rswa.training.stage_distribution import (
    StageDistribution,
)

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


def run_rswa_epoch(
    model: torch.nn.Module,
    loader: Iterable[dict[str, Any]],
    criterion: RSWALoss,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    amp: bool = True,
    grad_clip: float | None = 1.0,
    threshold: float = 0.5,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    losses: list[float] = []
    tonic_targets_all: list[torch.Tensor] = []
    tonic_preds_all: list[torch.Tensor] = []
    phasic_targets_all: list[torch.Tensor] = []
    phasic_preds_all: list[torch.Tensor] = []

    for batch in tqdm(loader, desc="Running RSWA epoch", unit="batch"):
        emg = batch["emg_center"].to(device, non_blocking=True)
        tonic_targets = batch["tonic_labels"].to(device, non_blocking=True)
        phasic_targets = batch["phasic_labels"].to(device, non_blocking=True)
        padding_mask = batch["padding_mask"].to(device, non_blocking=True)
        valid_mask = batch["rswa_valid"].to(device, non_blocking=True) & padding_mask

        if not valid_mask.any():
            continue

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with _autocast_context(device, amp):
                outputs = model(emg, mask=padding_mask)
                loss = criterion(outputs, tonic_targets, phasic_targets, valid_mask)

            if training:
                loss.backward()
                if grad_clip is not None:
                    clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

        tonic_preds = (torch.sigmoid(outputs["tonic_logits"]) >= threshold).long()
        phasic_preds = (torch.sigmoid(outputs["phasic_logits"]) >= threshold).long()

        losses.append(float(loss.detach().cpu()))
        tonic_targets_all.append(tonic_targets[valid_mask].long().detach().cpu())
        tonic_preds_all.append(tonic_preds[valid_mask].detach().cpu())
        phasic_targets_all.append(phasic_targets[valid_mask].long().detach().cpu())
        phasic_preds_all.append(phasic_preds[valid_mask].detach().cpu())

    if not tonic_targets_all:
        raise RuntimeError(
            "Nenhum rótulo RSWA válido foi encontrado. Verifique rswa_conf, "
            "min_confidence e rem_mask_only."
        )

    result = rswa_metrics(
        torch.cat(tonic_targets_all).numpy(),
        torch.cat(tonic_preds_all).numpy(),
        torch.cat(phasic_targets_all).numpy(),
        torch.cat(phasic_preds_all).numpy(),
    )
    return {"loss": _safe_mean(losses), **{k: float(v) for k, v in result.items()}}


def evaluate_joint(
    model: torch.nn.Module,
    loader: Iterable[dict[str, Any]],
    staging_criterion: StagingLoss,
    rswa_criterion: RSWALoss,
    device: torch.device,
    amp: bool = True,
    threshold: float = 0.5,
) -> dict[str, float]:
    model.eval()
    stage_losses: list[float] = []
    rswa_losses: list[float] = []
    stage_targets_all: list[torch.Tensor] = []
    stage_preds_all: list[torch.Tensor] = []
    tonic_targets_all: list[torch.Tensor] = []
    tonic_preds_all: list[torch.Tensor] = []
    phasic_targets_all: list[torch.Tensor] = []
    phasic_preds_all: list[torch.Tensor] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating joint", unit="batch"):
            signals = batch["signals"].to(device, non_blocking=True)
            emg = batch["emg_center"].to(device, non_blocking=True)
            padding_mask = batch["padding_mask"].to(device, non_blocking=True)
            stage_targets = batch["sleep_stages"].to(device, non_blocking=True)
            tonic_targets = batch["tonic_labels"].to(device, non_blocking=True)
            phasic_targets = batch["phasic_labels"].to(device, non_blocking=True)
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
                rswa_loss = rswa_criterion(outputs, tonic_targets, phasic_targets, rswa_valid)
                tonic_preds = (torch.sigmoid(outputs["tonic_logits"]) >= threshold).long()
                phasic_preds = (torch.sigmoid(outputs["phasic_logits"]) >= threshold).long()
                rswa_losses.append(float(rswa_loss.cpu()))
                tonic_targets_all.append(tonic_targets[rswa_valid].long().cpu())
                tonic_preds_all.append(tonic_preds[rswa_valid].cpu())
                phasic_targets_all.append(phasic_targets[rswa_valid].long().cpu())
                phasic_preds_all.append(phasic_preds[rswa_valid].cpu())

    metrics: dict[str, float] = {}
    if stage_targets_all:
        stage = staging_metrics(
            torch.cat(stage_targets_all).numpy(), torch.cat(stage_preds_all).numpy()
        )
        metrics.update({f"staging_{k}": float(v) for k, v in stage.items()})
        metrics["staging_loss"] = _safe_mean(stage_losses)
    if tonic_targets_all:
        rswa = rswa_metrics(
            torch.cat(tonic_targets_all).numpy(),
            torch.cat(tonic_preds_all).numpy(),
            torch.cat(phasic_targets_all).numpy(),
            torch.cat(phasic_preds_all).numpy(),
        )
        metrics.update({f"rswa_{k}": float(v) for k, v in rswa.items()})
        metrics["rswa_loss"] = _safe_mean(rswa_losses)
    return metrics


def collect_rswa_predictions(
    model: torch.nn.Module,
    loader: Iterable[dict[str, Any]],
    device: torch.device,
    *,
    amp: bool = True,
    threshold: float = 0.5,
) -> dict[str, np.ndarray]:
    model.eval()
    tonic_expected: list[torch.Tensor] = []
    tonic_prediction: list[torch.Tensor] = []
    phasic_expected: list[torch.Tensor] = []
    phasic_prediction: list[torch.Tensor] = []

    with torch.no_grad():
        for batch in loader:
            emg = batch["emg_center"].to(device, non_blocking=True)
            padding_mask = batch["padding_mask"].to(device, non_blocking=True)
            valid_mask = batch["rswa_valid"].to(device, non_blocking=True) & padding_mask
            if not valid_mask.any():
                continue
            with _autocast_context(device, amp):
                outputs = model(emg, mask=padding_mask)
            tonic_pred = (torch.sigmoid(outputs["tonic_logits"]) >= threshold).long()
            phasic_pred = (torch.sigmoid(outputs["phasic_logits"]) >= threshold).long()
            tonic_expected.append(batch["tonic_labels"].to(device)[valid_mask].long().cpu())
            tonic_prediction.append(tonic_pred[valid_mask].cpu())
            phasic_expected.append(batch["phasic_labels"].to(device)[valid_mask].long().cpu())
            phasic_prediction.append(phasic_pred[valid_mask].cpu())

    if not tonic_expected:
        raise RuntimeError("Nenhuma predição RSWA válida foi encontrada.")
    return {
        "tonic_expected": torch.cat(tonic_expected).numpy(),
        "tonic_prediction": torch.cat(tonic_prediction).numpy(),
        "phasic_expected": torch.cat(phasic_expected).numpy(),
        "phasic_prediction": torch.cat(phasic_prediction).numpy(),
    }

