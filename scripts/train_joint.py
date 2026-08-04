from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from sleep_rswa import (
    RSWADetectionNet,
    SleepAnalysisDataset,
    SleepStagingNet,
    SleepStagingRSWASystem,
    collate_sleep_analysis_exams,
)
from sleep_rswa.data import load_subject_directory
from sleep_rswa.training import (
    ExperimentLogger,
    RSWALoss,
    StagingLoss,
    collect_rswa_predictions,
    collect_staging_predictions,
    evaluate_joint,
    load_checkpoint,
    plot_confusion_matrix,
    resolve_device,
    save_checkpoint,
    seed_everything,
    stratified_group_folds,
)

GREEN  = "\033[92m"
YELLOW = "\033[93m"
RESET  = "\033[0m"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Treina staging e RSWA no mesmo DataLoader com StratifiedGroupKFold."
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--fold", type=int, default=None, help="Executa apenas este fold; padrão: todos.")
    parser.add_argument(
        "--stratify-by",
        choices=["staging", "rswa"],
        default="staging",
        help=(
            "Rótulo usado para estratificar os folds. O split é por sujeito e o "
            "joint treina as duas cabeças no mesmo split, então só um rótulo "
            "pode guiar a estratificação."
        ),
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr-staging", type=float, default=1e-4)
    parser.add_argument("--lr-rswa", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--all-stages", action="store_true")
    parser.add_argument("--movement-pos-weight", type=float)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument(
        "--monitor",
        choices=["joint_mean_f1", "staging_f1_macro", "rswa_movement_f1", "rswa_movement_kappa"],
        default="joint_mean_f1",
        help="Métrica de validação usada para selecionar o melhor checkpoint de cada fold.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--run-dir", type=Path, default=Path("runs/joint"))
    parser.add_argument("--experiment-name", default="joint_stratified_kfold")
    parser.add_argument("--notes", default=None)
    parser.add_argument("--tags", nargs="*", default=[])
    return parser.parse_args()


def make_loader(subjects, args, shuffle, device):
    ds = SleepAnalysisDataset(subjects, min_confidence=args.min_confidence, rem_mask_only=not args.all_stages)
    return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.num_workers,
                      collate_fn=collate_sleep_analysis_exams, pin_memory=device.type == "cuda",
                      persistent_workers=args.num_workers > 0)


def joint_monitor_value(monitor: str, val_metrics: dict[str, float]) -> float:
    """Valor da métrica monitorada. 'joint_mean_f1' = média de staging_f1_macro e movement_f1."""
    if monitor == "joint_mean_f1":
        scores = [
            val_metrics.get("staging_f1_macro", float("nan")),
            val_metrics.get("rswa_movement_f1", float("nan")),
        ]
        scores = [x for x in scores if x == x]
        return sum(scores) / len(scores) if scores else float("-inf")
    return float(val_metrics.get(monitor, float("-inf")))


def plot_joint_curves(history: list[dict[str, float]], output_path: Path, *, title: str) -> Path:
    """Curvas de treino do joint: losses (staging/rswa) e F1 de validação (staging/movement)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [int(r["epoch"]) for r in history]
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(epochs, [r["train_staging_loss"] for r in history], label="Train staging loss")
    axes[0].plot(epochs, [r.get("val_staging_loss", float("nan")) for r in history], label="Val staging loss")
    axes[0].plot(epochs, [r["train_rswa_loss"] for r in history], label="Train movement loss")
    axes[0].plot(epochs, [r.get("val_rswa_loss", float("nan")) for r in history], label="Val movement loss")
    axes[0].set_ylabel("Loss"); axes[0].legend(); axes[0].grid(alpha=0.25)

    axes[1].plot(epochs, [r.get("val_staging_f1_macro", float("nan")) for r in history], label="Val staging F1 (macro)")
    axes[1].plot(epochs, [r.get("val_rswa_movement_f1", float("nan")) for r in history], label="Val movement F1")
    axes[1].set_ylabel("F1"); axes[1].set_xlabel("Epoch"); axes[1].legend(); axes[1].grid(alpha=0.25)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)
    subjects = load_subject_directory(args.data_dir)
    folds = list(stratified_group_folds(subjects, n_splits=args.n_splits, seed=args.seed, task=args.stratify_by))
    if args.fold is not None:
        folds = [item for item in folds if item[0] == args.fold]
        if not folds:
            raise ValueError(f"Fold {args.fold} não existe para n_splits={args.n_splits}.")

    with ExperimentLogger(
        task="joint", experiment_name=args.experiment_name, root_dir=args.run_dir,
        device=device, args=vars(args), notes=args.notes, tags=args.tags,
    ) as logger:
        logger.info(f"Dispositivo: {device}")
        logger.info(f"Sujeitos: {len(subjects)} | n_splits={args.n_splits} | estratificação={args.stratify_by}")

        fold_summaries = []
        for fold, train_subjects, val_subjects in folds:
            seed_everything(args.seed + fold)
            fold_dir = logger.run_dir / f"fold_{fold}"
            checkpoint_dir = fold_dir / "checkpoints"
            figures_dir = fold_dir / "figures"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            figures_dir.mkdir(parents=True, exist_ok=True)

            train_loader = make_loader(train_subjects, args, True, device)
            val_loader = make_loader(val_subjects, args, False, device)

            staging_model = SleepStagingNet().to(device)
            rswa_model = RSWADetectionNet().to(device)
            system = SleepStagingRSWASystem(staging_model, rswa_model).to(device)
            staging_loss_fn = StagingLoss()
            movement_weight = torch.tensor(args.movement_pos_weight, device=device) if args.movement_pos_weight else None
            rswa_loss_fn = RSWALoss(movement_pos_weight=movement_weight)
            staging_optimizer = torch.optim.AdamW(
                staging_model.parameters(), lr=args.lr_staging, weight_decay=args.weight_decay
            )
            rswa_optimizer = torch.optim.AdamW(
                rswa_model.parameters(), lr=args.lr_rswa, weight_decay=args.weight_decay
            )

            logger.log_subject_split(train_subjects, val_subjects, filename=f"fold_{fold}_split.json")
            logger.info(f"Fold {fold}: treino={len(train_subjects)} validação={len(val_subjects)}")

            best_metric = float("-inf")
            best_epoch = 0
            stale = 0
            best_metrics: dict[str, float] = {}
            history: list[dict[str, float]] = []

            for epoch in range(1, args.epochs + 1):
                epoch_start = perf_counter()
                train_start = perf_counter()
                system.train()
                stage_loss_sum = 0.0
                rswa_loss_sum = 0.0
                stage_batches = 0
                rswa_batches = 0

                for batch in train_loader:
                    signals = batch["signals"].to(device, non_blocking=True)
                    emg = batch["emg_center"].to(device, non_blocking=True)
                    padding_mask = batch["padding_mask"].to(device, non_blocking=True)
                    stage_targets = batch["sleep_stages"].to(device, non_blocking=True)
                    movement_targets = batch["movement_labels"].to(device, non_blocking=True)
                    stage_valid = batch["staging_valid"].to(device, non_blocking=True) & padding_mask
                    rswa_valid = batch["rswa_valid"].to(device, non_blocking=True) & padding_mask

                    if stage_valid.any():
                        staging_optimizer.zero_grad(set_to_none=True)
                        with torch.autocast(
                            device_type="cuda", dtype=torch.bfloat16,
                            enabled=(not args.no_amp and device.type == "cuda"),
                        ):
                            stage_logits = staging_model(signals, mask=padding_mask)
                            stage_loss = staging_loss_fn(stage_logits, stage_targets, stage_valid)
                        stage_loss.backward()
                        clip_grad_norm_(staging_model.parameters(), args.grad_clip)
                        staging_optimizer.step()
                        stage_loss_sum += float(stage_loss.detach().cpu())
                        stage_batches += 1

                    if rswa_valid.any():
                        rswa_optimizer.zero_grad(set_to_none=True)
                        with torch.autocast(
                            device_type="cuda", dtype=torch.bfloat16,
                            enabled=(not args.no_amp and device.type == "cuda"),
                        ):
                            rswa_outputs = rswa_model(emg, mask=padding_mask)
                            rswa_loss = rswa_loss_fn(rswa_outputs, movement_targets, rswa_valid)
                        rswa_loss.backward()
                        clip_grad_norm_(rswa_model.parameters(), args.grad_clip)
                        rswa_optimizer.step()
                        rswa_loss_sum += float(rswa_loss.detach().cpu())
                        rswa_batches += 1

                train_time = perf_counter() - train_start
                val_start = perf_counter()
                val_metrics = evaluate_joint(
                    system, val_loader, staging_loss_fn, rswa_loss_fn, device,
                    amp=not args.no_amp, threshold=args.threshold,
                )
                val_time = perf_counter() - val_start

                row = {
                    "fold": fold,
                    "epoch": epoch,
                    "train_time_sec": train_time,
                    "val_time_sec": val_time,
                    "epoch_time_sec": perf_counter() - epoch_start,
                    "staging_learning_rate": staging_optimizer.param_groups[0]["lr"],
                    "rswa_learning_rate": rswa_optimizer.param_groups[0]["lr"],
                    "train_staging_loss": stage_loss_sum / max(stage_batches, 1),
                    "train_rswa_loss": rswa_loss_sum / max(rswa_batches, 1),
                    **{f"val_{key}": value for key, value in val_metrics.items()},
                }
                history.append(row)
                logger.log_epoch(row)
                logger.info(
                    f"fold={fold} ep={epoch:03d} train={train_time:.1f}s val={val_time:.1f}s "
                    f"train_stg_loss={row['train_staging_loss']:.4f} "
                    f"train_rswa_loss={row['train_rswa_loss']:.4f} | "
                    f"{YELLOW}"
                    f"val_stg_f1={val_metrics.get('staging_f1_macro', float('nan')):.4f} "
                    f"val_stg_kappa={val_metrics.get('staging_kappa', float('nan')):.4f} "
                    f"{GREEN}"
                    f"val_movement_f1={val_metrics.get('rswa_movement_f1', float('nan')):.4f} "
                    f"val_movement_kappa={val_metrics.get('rswa_movement_kappa', float('nan')):.4f}"
                    f"{RESET}"
                )

                save_checkpoint(
                    checkpoint_dir / "staging_last.pt", model=staging_model,
                    optimizer=staging_optimizer, epoch=epoch, metrics=val_metrics,
                    extra={"task": "staging", "trained_with": "joint", "fold": fold, "run_id": logger.run_id},
                )
                save_checkpoint(
                    checkpoint_dir / "rswa_last.pt", model=rswa_model,
                    optimizer=rswa_optimizer, epoch=epoch, metrics=val_metrics,
                    extra={"task": "rswa", "trained_with": "joint", "fold": fold, "run_id": logger.run_id},
                )

                current_metric = joint_monitor_value(args.monitor, val_metrics)
                if current_metric > best_metric:
                    best_metric = current_metric
                    best_epoch = epoch
                    stale = 0
                    best_metrics = dict(val_metrics)
                    save_checkpoint(
                        checkpoint_dir / "staging_best.pt", model=staging_model,
                        optimizer=staging_optimizer, epoch=epoch, metrics=val_metrics,
                        extra={"task": "staging", "trained_with": "joint", "fold": fold,
                               "monitor": args.monitor, "monitor_value": current_metric, "run_id": logger.run_id},
                    )
                    save_checkpoint(
                        checkpoint_dir / "rswa_best.pt", model=rswa_model,
                        optimizer=rswa_optimizer, epoch=epoch, metrics=val_metrics,
                        extra={"task": "rswa", "trained_with": "joint", "fold": fold,
                               "monitor": args.monitor, "monitor_value": current_metric, "run_id": logger.run_id},
                    )
                    logger.info(
                        f"Fold {fold}: novo melhor checkpoint na época {epoch}, "
                        f"{args.monitor}={current_metric:.4f}"
                    )
                else:
                    stale += 1

                if stale >= args.patience:
                    logger.info(f"Fold {fold}: early stopping na época {epoch}.")
                    break

            plot_joint_curves(history, figures_dir / "training_curves.png", title=f"Joint - Fold {fold}")

            # Matrizes de confusão no melhor checkpoint (staging 5 classes + movement binário).
            load_checkpoint(checkpoint_dir / "staging_best.pt", staging_model, device)
            load_checkpoint(checkpoint_dir / "rswa_best.pt", rswa_model, device)
            stage_pred = collect_staging_predictions(staging_model, val_loader, device, amp=not args.no_amp)
            move_pred = collect_rswa_predictions(rswa_model, val_loader, device, amp=not args.no_amp, threshold=args.threshold)
            plot_confusion_matrix(
                stage_pred["expected"], stage_pred["prediction"],
                figures_dir / "confusion_matrix_staging.png",
                labels=[0, 1, 2, 3, 4], display_labels=["W", "N1", "N2", "N3", "REM"],
                title=f"Staging confusion matrix - Fold {fold}",
            )
            plot_confusion_matrix(
                stage_pred["expected"], stage_pred["prediction"],
                figures_dir / "confusion_matrix_staging_normalized.png",
                labels=[0, 1, 2, 3, 4], display_labels=["W", "N1", "N2", "N3", "REM"],
                title=f"Staging normalized confusion matrix - Fold {fold}", normalize="true",
            )
            plot_confusion_matrix(
                move_pred["movement_expected"], move_pred["movement_prediction"],
                figures_dir / "confusion_matrix_movement.png", labels=[0, 1],
                display_labels=["Negative", "Positive"], title=f"Movement confusion matrix - Fold {fold}",
            )
            plot_confusion_matrix(
                move_pred["movement_expected"], move_pred["movement_prediction"],
                figures_dir / "confusion_matrix_movement_normalized.png", labels=[0, 1],
                display_labels=["Negative", "Positive"], title=f"Movement normalized confusion matrix - Fold {fold}", normalize="true",
            )

            fold_summaries.append(
                {
                    "fold": fold,
                    "best_epoch": best_epoch,
                    "monitor": args.monitor,
                    "best_monitor_value": best_metric,
                    "best_val_staging_f1_macro": best_metrics.get("staging_f1_macro"),
                    "best_val_staging_kappa": best_metrics.get("staging_kappa"),
                    "best_val_movement_f1": best_metrics.get("rswa_movement_f1"),
                    "best_val_movement_kappa": best_metrics.get("rswa_movement_kappa"),
                }
            )

        staging_f1_values = np.asarray(
            [f["best_val_staging_f1_macro"] for f in fold_summaries if f["best_val_staging_f1_macro"] is not None],
            dtype=np.float64,
        )
        movement_f1_values = np.asarray(
            [f["best_val_movement_f1"] for f in fold_summaries if f["best_val_movement_f1"] is not None],
            dtype=np.float64,
        )

        def _mean_std(values: np.ndarray) -> dict[str, float | None]:
            if values.size == 0:
                return {"mean": None, "std": None}
            return {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
            }

        logger.finalize(
            status="completed",
            summary={
                "folds": fold_summaries,
                "cross_validation": {
                    "n_folds": len(fold_summaries),
                    "stratify_by": args.stratify_by,
                    "staging_f1_macro": _mean_std(staging_f1_values),
                    "movement_f1": _mean_std(movement_f1_values),
                },
            },
        )


if __name__ == "__main__":
    main()
