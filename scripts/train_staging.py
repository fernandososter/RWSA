from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
)

from sleep_rswa import SleepAnalysisDataset, SleepStagingNet, collate_sleep_analysis_exams
from sleep_rswa.data import load_subject_directory
from sleep_rswa.training import (
    ExperimentLogger,
    StagingLoss,
    ValidationPredictionLogger,
    load_checkpoint,
    plot_confusion_matrix,
    plot_training_curves,
    resolve_device,
    run_staging_epoch,
    save_checkpoint,
    seed_everything,
    stratified_group_folds,
)


GREEN  = "\033[92m"
YELLOW = "\033[93m"
RESET  = "\033[0m"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treina staging com StratifiedGroupKFold.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--fold", type=int, default=None, help="Executa apenas este fold; padrão: todos.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--run-dir", type=Path, default=Path("runs/staging"))
    parser.add_argument("--experiment-name", default="staging_stratified_kfold")
    parser.add_argument("--notes", default=None)
    parser.add_argument("--tags", nargs="*", default=[])
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--class-weights", type=float, nargs=5, default=None)
    parser.add_argument("--monitor", choices=["f1_macro", "kappa"],default="f1_macro",  help="Métrica de validação usada para selecionar o melhor checkpoint.")
    return parser.parse_args()


def make_loader(subjects, args, shuffle, device):
    return DataLoader(
        SleepAnalysisDataset(subjects),
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        collate_fn=collate_sleep_analysis_exams,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)
    subjects = load_subject_directory(args.data_dir)
    folds = list(stratified_group_folds(subjects, n_splits=args.n_splits, seed=args.seed, task="staging"))
    if args.fold is not None:
        folds = [item for item in folds if item[0] == args.fold]
        if not folds:
            raise ValueError(f"Fold {args.fold} não existe para n_splits={args.n_splits}.")

    with ExperimentLogger(
        task="staging",
        experiment_name=args.experiment_name,
        root_dir=args.run_dir,
        device=device,
        args=vars(args),
        notes=args.notes,
        tags=args.tags,
    ) as logger:
        
        
        fold_summaries = []
        # out of folds - usado para acumular as previsões de validação de todos os folds para avaliação final
        all_oof_expected: list[int] = []
        all_oof_predictions: list[int] = []


        for fold, train_subjects, val_subjects in folds:
            seed_everything(args.seed + fold)
            fold_dir = logger.run_dir / f"fold_{fold}"
            checkpoint_dir = fold_dir / "checkpoints"
            figures_dir = fold_dir / "figures"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            figures_dir.mkdir(parents=True, exist_ok=True)

            train_loader = make_loader(train_subjects, args, True, device)
            val_loader = make_loader(val_subjects, args, False, device)
            model = SleepStagingNet().to(device)
            weights = torch.tensor(args.class_weights, dtype=torch.float32, device=device) if args.class_weights else None
            criterion = StagingLoss(class_weights=weights)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            prediction_logger = ValidationPredictionLogger(fold_dir, fold=fold)

            logger.info(f"Fold {fold}: treino={len(train_subjects)} validação={len(val_subjects)}")
            logger.log_subject_split(train_subjects, val_subjects, filename=f"fold_{fold}_split.json")
            
            best_metric = float("-inf")
            best_epoch = 0
            stale = 0
            best_metrics: dict[str, float] = {}

            history: list[dict[str, float]] = []

            for epoch in tqdm(range(1, args.epochs + 1), desc=f"Fold {fold} training", unit="epoch"):


                epoch_start = perf_counter()
                train_start = perf_counter()
                train_metrics = run_staging_epoch(model, train_loader, criterion, device, optimizer, amp=not args.no_amp, grad_clip=args.grad_clip)
                train_time = perf_counter() - train_start
                val_start = perf_counter()
                val_metrics = run_staging_epoch(model, val_loader, criterion, device, amp=not args.no_amp, prediction_logger=prediction_logger, epoch=epoch)
                val_time = perf_counter() - val_start
                row = {
                    "fold": fold,
                    "epoch": epoch,
                    "train_time_sec": train_time,
                    "val_time_sec": val_time,
                    "epoch_time_sec": perf_counter() - epoch_start,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    **{f"train_{k}": v for k, v in train_metrics.items()},
                    **{f"val_{k}": v for k, v in val_metrics.items()},
                }
                history.append(row)
                logger.log_epoch(row)
                
                logger.info(
                    f"ep={epoch:03d} -- "
                    f"{GREEN}"
                    f"train_loss={train_metrics['loss']:.4f} "
                    f"train_f1={train_metrics['f1_macro']:.4f} "
                    f"train_kappa={train_metrics['kappa']:.4f}"
                    f"{RESET} -- "
                    f"{YELLOW}"
                    f"val_loss={val_metrics['loss']:.4f} "
                    f"val_f1={val_metrics['f1_macro']:.4f} "
                    f"val_kappa={val_metrics['kappa']:.4f}"
                    f"{RESET}"
                )
                
                save_checkpoint(checkpoint_dir / "last.pt", model=model, optimizer=optimizer, epoch=epoch, metrics=val_metrics, extra={"fold": fold})
               
                current_metric = float(val_metrics[args.monitor])

                if current_metric > best_metric:
                    best_metric = current_metric
                    best_epoch = epoch
                    stale = 0
                    best_metrics = dict(val_metrics)

                    save_checkpoint( checkpoint_dir / "best.pt", model=model, optimizer=optimizer,
                        epoch=epoch, metrics=val_metrics,
                        extra={
                            "fold": fold,
                            "monitor": args.monitor,
                            "monitor_value": current_metric,
                        },
                    )

                    logger.info( f"Fold {fold}: novo melhor checkpoint "
                        f"na época {epoch}, "
                        f"{args.monitor}={current_metric:.4f}"
                    )
                else:
                    stale += 1

                if stale >= args.patience:
                    logger.info(f"Fold {fold}: early stopping na época {epoch}.")
                    break

            parquet_path = prediction_logger.close()
            plot_training_curves( history, figures_dir / "training_curves.png", f1_key="f1_macro",  kappa_key="kappa", title=f"Staging - Fold {fold}")
            predictions = pd.read_parquet(parquet_path, filters=[("epoch", "=", best_epoch)])
          
            all_oof_expected.extend(predictions["expected"].to_numpy().astype(int).tolist())
            all_oof_predictions.extend(predictions["prediction"].to_numpy().astype(int).tolist())

            plot_confusion_matrix(
                predictions["expected"].to_numpy(), predictions["prediction"].to_numpy(),
                figures_dir / "confusion_matrix_best_epoch.png",
                labels=[0, 1, 2, 3, 4], display_labels=["W", "N1", "N2", "N3", "REM"],
                title=f"Staging confusion matrix - Fold {fold} - Epoch {best_epoch}",
            )
            plot_confusion_matrix(
                predictions["expected"].to_numpy(), predictions["prediction"].to_numpy(),
                figures_dir / "confusion_matrix_best_epoch_normalized.png",
                labels=[0, 1, 2, 3, 4], display_labels=["W", "N1", "N2", "N3", "REM"],
                title=f"Staging normalized confusion matrix - Fold {fold} - Epoch {best_epoch}", normalize="true",
            )


            
            fold_summaries.append(
                {
                    "fold": fold,
                    "best_epoch": best_epoch,
                    "monitor": args.monitor,
                    "best_monitor_value": best_metric,
                    "best_val_loss": best_metrics.get("loss"),
                    "best_val_f1_macro": best_metrics.get("f1_macro"),
                    "best_val_kappa": best_metrics.get("kappa"),
                    "best_val_balanced_accuracy": best_metrics.get(
                        "balanced_accuracy"
                    ),
                }
            )



        
        oof_expected = np.asarray(all_oof_expected, dtype=np.int64)
        oof_predictions = np.asarray( all_oof_predictions, dtype=np.int64)

        # gerando métricas globais de validação usando todas as previsões de validação de todos os folds
        global_f1_macro = f1_score( oof_expected, oof_predictions, average="macro", zero_division=0)
        global_kappa = cohen_kappa_score( oof_expected, oof_predictions )
        global_balanced_accuracy = balanced_accuracy_score( oof_expected, oof_predictions )

        global_figures_dir = logger.run_dir / "figures"
        global_figures_dir.mkdir( parents=True,exist_ok=True)

        plot_confusion_matrix( oof_expected, oof_predictions, global_figures_dir / "confusion_matrix_oof.png",
            labels=[0, 1, 2, 3, 4], display_labels=["W", "N1", "N2", "N3", "REM"], normalize=None,
            title="Staging - Out-of-fold confusion matrix")

        plot_confusion_matrix( oof_expected, oof_predictions, global_figures_dir / "confusion_matrix_oof_normalized.png",
            labels=[0, 1, 2, 3, 4], display_labels=["W", "N1", "N2", "N3", "REM"], normalize="true",
            title=(
                "Staging - Out-of-fold normalized "
                "confusion matrix"
            ),
        )

        
        fold_f1_values = np.asarray(
            [ fold["best_val_f1_macro"] for fold in fold_summaries], dtype=np.float64,
        )

        fold_f1_mean = float(fold_f1_values.mean())
        fold_f1_std = float(  fold_f1_values.std(ddof=1)
            if len(fold_f1_values) > 1
            else 0.0
        )

        logger.finalize(
            status="completed",
            summary={
                "folds": fold_summaries,
                "cross_validation": {
                    "n_folds": len(fold_summaries),
                    "f1_macro_mean": fold_f1_mean,
                    "f1_macro_std": fold_f1_std,
                },
                "out_of_fold": {
                    "n_samples": int(oof_expected.size),
                    "f1_macro": float(global_f1_macro),
                    "kappa": float(global_kappa),
                    "balanced_accuracy": float(
                        global_balanced_accuracy
                    ),
                },
            },
        )


if __name__ == "__main__":
    main()
