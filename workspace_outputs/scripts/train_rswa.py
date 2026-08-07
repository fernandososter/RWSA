from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from sklearn.metrics import cohen_kappa_score, f1_score

import torch
from torch.utils.data import DataLoader

from sleep_rswa import RSWADetectionNet, SleepAnalysisDataset, collate_sleep_analysis_exams
from sleep_rswa.data import load_subject_directory
from sleep_rswa.utils import (
    format_stage_distribution,
    print_movement_distribution,
    print_split_summary,
    print_stage_distribution,
)
from sleep_rswa.training import (
    ExperimentLogger,
    RSWALoss,
    collect_rswa_predictions,
    describe_split,
    evaluate_movement_test_set,
    format_split_description,
    load_checkpoint,
    plot_confusion_matrix,
    plot_training_curves,
    resolve_device,
    run_rswa_epoch,
    save_checkpoint,
    seed_everything,
    stratified_group_folds,
    stratified_group_holdout,
)


GREEN  = "\033[92m"
YELLOW = "\033[93m"
RESET  = "\033[0m"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treina RSWA com StratifiedGroupKFold.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--fold", type=int, default=None, help="Executa apenas este fold; padrão: todos.")
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.2,
        help=(
            "Fração de sujeitos separada como conjunto de TESTE fixo, antes da "
            "validação cruzada. Estratificado por movimento (rswa) e agrupado "
            "por sujeito. Use 0 para desativar a fase de teste. Ignorado se "
            "--test-dir for dado."
        ),
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=None,
        help=(
            "Diretório com .pt de um conjunto de teste EXTERNO. Se informado, "
            "--data-dir é usado inteiro para a CV e este para o teste final."
        ),
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.5, help="Limiar de decisão default, aplicado às 3 cabeças se os limiares específicos não forem dados.")
    parser.add_argument("--tonic-threshold", type=float, default=None, help="Limiar específico da cabeça tônica (default: --threshold).")
    parser.add_argument("--phasic-threshold", type=float, default=None, help="Limiar específico da cabeça fásica (default: --threshold).")
    parser.add_argument("--any-threshold", type=float, default=None, help="Limiar específico da cabeça any (default: --threshold).")
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--all-stages", action="store_true")
    parser.add_argument("--tonic-pos-weight", type=float, help="pos_weight da BCE da cabeça tônica (prevalência tipicamente mais baixa que fásica).")
    parser.add_argument("--phasic-pos-weight", type=float, help="pos_weight da BCE da cabeça fásica.")
    parser.add_argument("--any-pos-weight", type=float, help="pos_weight da BCE da cabeça any (faixa de duração ambígua 5-15s; prevalência mais baixa das 3).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--run-dir", type=Path, default=Path("runs/rswa"))
    parser.add_argument("--experiment-name", default="rswa_stratified_kfold")
    parser.add_argument("--notes", default=None)
    parser.add_argument("--tags", nargs="*", default=[])
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument(
        "--log-movement-distribution", action="store_true",
        help="Mostra target/prediction de movimento (Negative/Positive) por época, para cada cabeça.",
    )
    parser.add_argument(
        "--monitor",
        choices=["tonic_f1", "phasic_f1", "any_f1", "rswa_f1_macro", "rswa_kappa_macro", "movement_f1", "movement_kappa"],
        default="rswa_f1_macro",
        help="Métrica usada para selecionar o melhor checkpoint. rswa_f1_macro/kappa_macro = média das 3 cabeças.",
    )

    return parser.parse_args()


def _resolve_thresholds(args) -> dict[str, float]:
    return {
        "tonic": args.tonic_threshold if args.tonic_threshold is not None else args.threshold,
        "phasic": args.phasic_threshold if args.phasic_threshold is not None else args.threshold,
        "any": args.any_threshold if args.any_threshold is not None else args.threshold,
    }


def make_loader(subjects, args, shuffle, device):
    ds = SleepAnalysisDataset(subjects, min_confidence=args.min_confidence, rem_mask_only=not args.all_stages)
    return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.num_workers,
                      collate_fn=collate_sleep_analysis_exams, pin_memory=device.type == "cuda",
                      persistent_workers=args.num_workers > 0)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)
    all_subjects = load_subject_directory(args.data_dir)

    # ── Conjunto de TESTE fixo (held-out), separado ANTES da CV ────────────
    # Prioridade: --test-dir (externo) > --test-fraction (holdout do data-dir).
    # Nunca entra na CV nem na seleção de checkpoint; usado só no fim, uma vez.
    test_subjects: list = []
    if args.test_dir is not None:
        test_subjects = load_subject_directory(args.test_dir)
        cv_subjects = all_subjects
    elif args.test_fraction and args.test_fraction > 0.0:
        cv_subjects, test_subjects = stratified_group_holdout(
            all_subjects, test_fraction=args.test_fraction, seed=args.seed, task="rswa"
        )
    else:
        cv_subjects = all_subjects

    subjects = cv_subjects
    folds = list(stratified_group_folds(subjects, n_splits=args.n_splits, seed=args.seed, task="rswa"))
    if args.fold is not None:
        folds = [item for item in folds if item[0] == args.fold]
        if not folds:
            raise ValueError(f"Fold {args.fold} não existe para n_splits={args.n_splits}.")

    with ExperimentLogger(task="rswa", experiment_name=args.experiment_name, root_dir=args.run_dir,
                          device=device, args=vars(args), notes=args.notes, tags=args.tags) as logger:
        fold_summaries = []
        fold_checkpoints: list[dict[str, Any]] = []
        all_oof_expected: dict[str, list[int]] = {"tonic": [], "phasic": [], "any": []}
        all_oof_predictions: dict[str, list[int]] = {"tonic": [], "phasic": [], "any": []}
        data_report: dict[str, Any] = {"folds": []}

        logger.info(
            f"Sujeitos: total={len(all_subjects)} | CV={len(subjects)} | teste={len(test_subjects)}"
        )
        if test_subjects:
            logger.log_subject_split(subjects, test_subjects, filename="test_split.json")
            test_dataset = make_loader(test_subjects, args, False, device).dataset
            test_desc = describe_split(test_dataset)
            data_report["test"] = test_desc
            logger.info(format_split_description("TESTE (held-out)", test_desc))
            print_stage_distribution(
                "TESTE (held-out) - Stage distribution",
                test_dataset.stage_distribution().as_dict(),
            )
            print_movement_distribution(
                "TESTE (held-out) - Movement distribution",
                test_dataset.movement_distribution(),
            )

        for fold, train_subjects, val_subjects in folds:
            seed_everything(args.seed + fold)
            fold_dir = logger.run_dir / f"fold_{fold}"
            checkpoint_dir = fold_dir / "checkpoints"
            figures_dir = fold_dir / "figures"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            figures_dir.mkdir(parents=True, exist_ok=True)
            train_loader = make_loader(train_subjects, args, True, device)
            val_loader = make_loader(val_subjects, args, False, device)

            # ── Documentação de dados do fold (exames, % estágios, % movimento) ──
            train_desc = describe_split(train_loader.dataset)
            val_desc = describe_split(val_loader.dataset)
            fold_data = {"fold": fold, "train": train_desc, "validation": val_desc}
            data_report["folds"].append(fold_data)
            logger.write_json(f"fold_{fold}/data_description.json", fold_data)
            logger.info(format_split_description(f"Fold {fold} TREINO", train_desc))
            logger.info(format_split_description(f"Fold {fold} VALIDAÇÃO", val_desc))

            print()
            print("=" * 80)
            print(f"FOLD {fold}/{args.n_splits}")
            print("=" * 80)

            print_stage_distribution(
                f"Fold {fold} - Train stage distribution",
                train_loader.dataset.stage_distribution().as_dict(),
            )
            print_movement_distribution(
                f"Fold {fold} - Train movement distribution",
                train_loader.dataset.movement_distribution(),
            )
            print_stage_distribution(
                f"Fold {fold} - Validation stage distribution",
                val_loader.dataset.stage_distribution().as_dict(),
            )
            print_movement_distribution(
                f"Fold {fold} - Validation movement distribution",
                val_loader.dataset.movement_distribution(),
            )

            print_split_summary(
                split_name="Train",
                subjects=train_subjects,
                dataset=train_loader.dataset,
                loader=train_loader,
            )
            print_split_summary(
                split_name="Validation",
                subjects=val_subjects,
                dataset=val_loader.dataset,
                loader=val_loader,
            )

            model = RSWADetectionNet().to(device)
            tonic_weight = torch.tensor(args.tonic_pos_weight, device=device) if args.tonic_pos_weight else None
            phasic_weight = torch.tensor(args.phasic_pos_weight, device=device) if args.phasic_pos_weight else None
            any_weight = torch.tensor(args.any_pos_weight, device=device) if args.any_pos_weight else None
            criterion = RSWALoss(tonic_pos_weight=tonic_weight, phasic_pos_weight=phasic_weight, any_pos_weight=any_weight)
            thresholds = _resolve_thresholds(args)
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
                                               grad_clip=args.grad_clip, threshold=thresholds)
                train_time = perf_counter() - train_start
                val_start = perf_counter()
                val_metrics = run_rswa_epoch(model, val_loader, criterion, device, amp=not args.no_amp,
                                             threshold=thresholds)
                val_time = perf_counter() - val_start
                scalar_train = {k: v for k, v in train_metrics.items() if isinstance(v, (int, float))}
                scalar_val = {k: v for k, v in val_metrics.items() if isinstance(v, (int, float))}
                row = {"fold": fold, "epoch": epoch, "train_time_sec": train_time, "val_time_sec": val_time,
                       "epoch_time_sec": perf_counter() - epoch_start, "learning_rate": optimizer.param_groups[0]["lr"],
                       **{f"train_{k}": v for k, v in scalar_train.items()},
                       **{f"val_{k}": v for k, v in scalar_val.items()}}
                history.append(row)
                logger.log_epoch(row)
                
                logger.info(
                    f"fold={fold} ep={epoch:03d} -- "
                    f"{GREEN}"
                    f"train_loss={train_metrics['loss']:.4f} "
                    f"train_f1(t/p/a)={train_metrics['tonic_f1']:.3f}/{train_metrics['phasic_f1']:.3f}/{train_metrics['any_f1']:.3f} "
                    f"train_f1_macro={train_metrics['rswa_f1_macro']:.4f}"
                    f"{RESET} -- "
                    f"{YELLOW}"
                    f"val_loss={val_metrics['loss']:.4f} "
                    f"val_f1(t/p/a)={val_metrics['tonic_f1']:.3f}/{val_metrics['phasic_f1']:.3f}/{val_metrics['any_f1']:.3f} "
                    f"val_f1_macro={val_metrics['rswa_f1_macro']:.4f}"
                    f"{RESET}"
                )

                if args.log_movement_distribution:
                    for head in ("tonic", "phasic", "any"):
                        train_targets = format_stage_distribution(train_metrics[f"{head}_target_distribution"])
                        train_predictions = format_stage_distribution(train_metrics[f"{head}_prediction_distribution"])
                        val_targets = format_stage_distribution(val_metrics[f"{head}_target_distribution"])
                        val_predictions = format_stage_distribution(val_metrics[f"{head}_prediction_distribution"])
                        logger.info(f"{GREEN}train_{head}_targets[{train_targets}]{RESET}")
                        logger.info(f"{GREEN}train_{head}_predictions[{train_predictions}]{RESET}")
                        logger.info(f"{YELLOW}val_{head}_targets[{val_targets}]{RESET}")
                        logger.info(f"{YELLOW}val_{head}_predictions[{val_predictions}]{RESET}")

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
            final = collect_rswa_predictions(model, val_loader, device, amp=not args.no_amp, threshold=thresholds)
            # Acumula predições de validação do melhor checkpoint para o OOF global, por cabeça.
            for head in ("tonic", "phasic", "any"):
                all_oof_expected[head].extend(final[f"{head}_expected"].astype(int).tolist())
                all_oof_predictions[head].extend(final[f"{head}_prediction"].astype(int).tolist())
                plot_confusion_matrix(final[f"{head}_expected"], final[f"{head}_prediction"],
                                      figures_dir / f"confusion_matrix_{head}.png", labels=[0, 1],
                                      display_labels=["Negative", "Positive"], title=f"{head.capitalize()} confusion matrix - Fold {fold}")
                plot_confusion_matrix(final[f"{head}_expected"], final[f"{head}_prediction"],
                                      figures_dir / f"confusion_matrix_{head}_normalized.png", labels=[0, 1],
                                      display_labels=["Negative", "Positive"], title=f"{head.capitalize()} normalized confusion matrix - Fold {fold}", normalize="true")

            fold_checkpoints.append({"fold": fold, "best_checkpoint": checkpoint_dir / "best.pt"})
            fold_summaries.append(
                {
                    "fold": fold,
                    "best_epoch": best_epoch,
                    "monitor": args.monitor,
                    "best_monitor_value": best_metric,
                    "best_val_loss": best_metrics.get("loss"),
                    "best_val_tonic_f1": best_metrics.get("tonic_f1"),
                    "best_val_phasic_f1": best_metrics.get("phasic_f1"),
                    "best_val_any_f1": best_metrics.get("any_f1"),
                    "best_val_rswa_f1_macro": best_metrics.get("rswa_f1_macro"),
                    "best_val_rswa_kappa_macro": best_metrics.get("rswa_kappa_macro"),
                }
            )

        # ── Métricas out-of-fold (validação agregada de todos os folds), por cabeça ────
        global_figures_dir = logger.run_dir / "figures"
        global_figures_dir.mkdir(parents=True, exist_ok=True)
        out_of_fold: dict[str, Any] = {}
        for head in ("tonic", "phasic", "any"):
            oof_expected = np.asarray(all_oof_expected[head], dtype=np.int64)
            oof_predictions = np.asarray(all_oof_predictions[head], dtype=np.int64)
            if not oof_expected.size:
                continue
            plot_confusion_matrix(oof_expected, oof_predictions, global_figures_dir / f"confusion_matrix_{head}_oof.png",
                                  labels=[0, 1], display_labels=["Negative", "Positive"],
                                  title=f"{head.capitalize()} - Out-of-fold confusion matrix")
            plot_confusion_matrix(oof_expected, oof_predictions, global_figures_dir / f"confusion_matrix_{head}_oof_normalized.png",
                                  labels=[0, 1], display_labels=["Negative", "Positive"], normalize="true",
                                  title=f"{head.capitalize()} - Out-of-fold normalized confusion matrix")
            out_of_fold[head] = {
                "n_samples": int(oof_expected.size),
                "n_positives": int(oof_expected.sum()),
                "f1": float(f1_score(oof_expected, oof_predictions, zero_division=0)),
                "kappa": float(cohen_kappa_score(oof_expected, oof_predictions)),
            }
            logger.info(f"OUT-OF-FOLD ({head}): f1={out_of_fold[head]['f1']:.4f} kappa={out_of_fold[head]['kappa']:.4f}")

        # ── Fase de TESTE (held-out, nunca visto na CV): ensemble dos folds ─
        test_summary: dict[str, Any] | None = None
        if test_subjects and fold_checkpoints:
            test_loader = make_loader(test_subjects, args, False, device)
            test_summary = evaluate_movement_test_set(
                test_loader=test_loader, fold_checkpoints=fold_checkpoints,
                build_model=lambda: RSWADetectionNet(), device=device, logger=logger,
                figures_dir=logger.run_dir / "test", amp=not args.no_amp, threshold=thresholds,
            )

        logger.write_json("data_description.json", data_report)

        cv_stats: dict[str, Any] = {"n_folds": len(fold_summaries)}
        for head in ("tonic", "phasic", "any"):
            fold_f1_values = np.asarray(
                [f[f"best_val_{head}_f1"] for f in fold_summaries if f.get(f"best_val_{head}_f1") is not None],
                dtype=np.float64,
            )
            cv_stats[f"{head}_f1_mean"] = float(fold_f1_values.mean()) if fold_f1_values.size else None
            cv_stats[f"{head}_f1_std"] = float(fold_f1_values.std(ddof=1)) if fold_f1_values.size > 1 else 0.0

        logger.finalize(
            status="completed",
            summary={
                "folds": fold_summaries,
                "cross_validation": cv_stats,
                "out_of_fold": out_of_fold,
                "test": test_summary,
                "data_description": data_report,
            },
        )


if __name__ == "__main__":
    main()
