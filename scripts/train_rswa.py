from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import torch
from torch.utils.data import DataLoader

from sleep_rswa import RSWADetectionNet, SleepAnalysisDataset, collate_sleep_analysis_exams
from sleep_rswa.data import load_subject_directory
from sleep_rswa.training import (
    ExperimentLogger,
    RSWALoss,
    collect_rswa_predictions,
    load_checkpoint,
    plot_confusion_matrix,
    plot_training_curves,
    resolve_device,
    run_rswa_epoch,
    save_checkpoint,
    seed_everything,
    stratified_group_folds,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treina RSWA com StratifiedGroupKFold.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--fold", type=int, default=None, help="Executa apenas este fold; padrão: todos.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--all-stages", action="store_true")
    parser.add_argument("--tonic-pos-weight", type=float)
    parser.add_argument("--phasic-pos-weight", type=float)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--run-dir", type=Path, default=Path("runs/rswa"))
    parser.add_argument("--experiment-name", default="rswa_stratified_kfold")
    parser.add_argument("--notes", default=None)
    parser.add_argument("--tags", nargs="*", default=[])
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--monitor", choices=["rswa_f1_macro", "rswa_kappa_macro"], default="rswa_f1_macro", help="Métrica usada para selecionar o melhor checkpoint.")

    return parser.parse_args()


def make_loader(subjects, args, shuffle, device):
    ds = SleepAnalysisDataset(subjects, min_confidence=args.min_confidence, rem_mask_only=not args.all_stages)
    return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.num_workers,
                      collate_fn=collate_sleep_analysis_exams, pin_memory=device.type == "cuda",
                      persistent_workers=args.num_workers > 0)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)
    subjects = load_subject_directory(args.data_dir)
    folds = list(stratified_group_folds(subjects, n_splits=args.n_splits, seed=args.seed, task="rswa"))
    if args.fold is not None:
        folds = [item for item in folds if item[0] == args.fold]
        if not folds:
            raise ValueError(f"Fold {args.fold} não existe para n_splits={args.n_splits}.")

    with ExperimentLogger(task="rswa", experiment_name=args.experiment_name, root_dir=args.run_dir,
                          device=device, args=vars(args), notes=args.notes, tags=args.tags) as logger:
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
            model = RSWADetectionNet().to(device)
            tonic_weight = torch.tensor(args.tonic_pos_weight, device=device) if args.tonic_pos_weight else None
            phasic_weight = torch.tensor(args.phasic_pos_weight, device=device) if args.phasic_pos_weight else None
            criterion = RSWALoss(tonic_pos_weight=tonic_weight, phasic_pos_weight=phasic_weight)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            logger.log_subject_split(train_subjects, val_subjects, filename=f"fold_{fold}_split.json")
            
            best_metric = float("-inf")
            best_epoch = 0
            stale = 0
            best_metrics: dict[str, float] = {}

            history: list[dict[str, float]] = []

            for epoch in range(1, args.epochs + 1):
                epoch_start = perf_counter()
                train_start = perf_counter()
                train_metrics = run_rswa_epoch(model, train_loader, criterion, device, optimizer, amp=not args.no_amp,
                                               grad_clip=args.grad_clip, threshold=args.threshold)
                train_time = perf_counter() - train_start
                val_start = perf_counter()
                val_metrics = run_rswa_epoch(model, val_loader, criterion, device, amp=not args.no_amp,
                                             threshold=args.threshold)
                val_time = perf_counter() - val_start
                row = {"fold": fold, "epoch": epoch, "train_time_sec": train_time, "val_time_sec": val_time,
                       "epoch_time_sec": perf_counter() - epoch_start, "learning_rate": optimizer.param_groups[0]["lr"],
                       **{f"train_{k}": v for k, v in train_metrics.items()},
                       **{f"val_{k}": v for k, v in val_metrics.items()}}
                history.append(row)
                logger.log_epoch(row)
                
                logger.info(
                    f"fold={fold} "
                    f"ep={epoch:03d} "
                    f"train_loss={train_metrics['loss']:.4f} "
                    f"val_loss={val_metrics['loss']:.4f} "
                    f"train_f1={train_metrics['rswa_f1_macro']:.4f} "
                    f"val_f1={val_metrics['rswa_f1_macro']:.4f} "
                    f"train_kappa={train_metrics['rswa_kappa_macro']:.4f} "
                    f"val_kappa={val_metrics['rswa_kappa_macro']:.4f} "
                    f"val_tonic_kappa={val_metrics['tonic_kappa']:.4f} "
                    f"val_phasic_kappa={val_metrics['phasic_kappa']:.4f}"
                )

                current_metric = float(val_metrics[args.monitor])

                if current_metric > best_metric:
                    best_metric = current_metric
                    best_epoch = epoch
                    stale = 0
                    best_metrics = dict(val_metrics)

                    save_checkpoint(
                        checkpoint_dir / "best.pt",
                        model=model,
                        optimizer=optimizer,
                        epoch=epoch,
                        metrics=val_metrics,
                        extra={
                            "fold": fold,
                            "monitor": args.monitor,
                            "monitor_value": current_metric,
                        },
                    )

                    logger.info(
                        f"Fold {fold}: novo melhor checkpoint "
                        f"na época {epoch}, "
                        f"{args.monitor}={current_metric:.4f}"
                    )
                else:
                    stale += 1

                if stale >= args.patience:
                    logger.info(f"Fold {fold}: early stopping na época {epoch}.")
                    break

            plot_training_curves( history, figures_dir / "training_curves.png", f1_key="rswa_f1_macro", kappa_key="rswa_kappa_macro", title=f"RSWA - Fold {fold}")

            load_checkpoint(checkpoint_dir / "best.pt", model, device)
            final = collect_rswa_predictions(model, val_loader, device, amp=not args.no_amp, threshold=args.threshold)
            for name, display in (("tonic", "Tonic"), ("phasic", "Phasic")):
                plot_confusion_matrix(final[f"{name}_expected"], final[f"{name}_prediction"],
                                      figures_dir / f"confusion_matrix_{name}.png", labels=[0, 1],
                                      display_labels=["Negative", "Positive"], title=f"{display} confusion matrix - Fold {fold}")
                plot_confusion_matrix(final[f"{name}_expected"], final[f"{name}_prediction"],
                                      figures_dir / f"confusion_matrix_{name}_normalized.png", labels=[0, 1],
                                      display_labels=["Negative", "Positive"], title=f"{display} normalized confusion matrix - Fold {fold}", normalize="true")
            
            fold_summaries.append(
                {
                    "fold": fold,
                    "best_epoch": best_epoch,
                    "monitor": args.monitor,
                    "best_monitor_value": best_metric,
                    "best_val_loss": best_metrics.get("loss"),
                    "best_val_rswa_f1_macro": best_metrics.get(
                        "rswa_f1_macro"
                    ),
                    "best_val_rswa_kappa_macro": best_metrics.get(
                        "rswa_kappa_macro"
                    ),
                    "best_val_tonic_f1": best_metrics.get("tonic_f1"),
                    "best_val_phasic_f1": best_metrics.get("phasic_f1"),
                    "best_val_tonic_kappa": best_metrics.get(
                        "tonic_kappa"
                    ),
                    "best_val_phasic_kappa": best_metrics.get(
                        "phasic_kappa"
                    ),
                }
            )

        logger.finalize(status="completed", summary={"folds": fold_summaries})


if __name__ == "__main__":
    main()
