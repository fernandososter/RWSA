from __future__ import annotations
from collections.abc import Iterable, Mapping
import argparse
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from sleep_rswa.utils import (
    print_experiment_summary,
    print_split_summary,
    print_stage_distribution,
)

from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
)

from sleep_rswa import (
    SleepAnalysisDataset,
    available_staging_models,
    build_staging_model,
    collate_sleep_analysis_exams,
)
from sleep_rswa.data import load_subject_directory
from sleep_rswa.training import (
    ExperimentLogger,
    StagingLoss,
    ValidationPredictionLogger,
    collect_staging_predictions,
    load_checkpoint,
    plot_confusion_matrix,
    plot_training_curves,
    resolve_device,
    run_staging_epoch,
    save_checkpoint,
    seed_everything,
    stratified_group_folds,
    stratified_group_holdout,
)


GREEN  = "\033[92m"
YELLOW = "\033[93m"
RESET  = "\033[0m"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treina staging com StratifiedGroupKFold.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--fold", type=int, default=None, help="Executa apenas este fold; padrão: todos.")
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.2,
        help=(
            "Fração de sujeitos separada como conjunto de TESTE fixo, antes da "
            "validação cruzada. Estratificado por estágio e agrupado por sujeito. "
            "Use 0 para desativar a fase de teste. Ignorado se --test-dir for dado."
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
    parser.add_argument("--model", choices=available_staging_models(),default="cnn_bimamba",help="Arquitetura usada no experimento.")
    parser.add_argument("--lstm-hidden-size", type=int, default=None)
    parser.add_argument( "--lstm-layers", type=int, default=1)
    parser.add_argument("--summary", action="store_true", help=("Mostra um resumo do experimento e a estrutura completa ")) 

    parser.add_argument( "--log-stage-distribution", action="store_true",
        help=(
            "Registra a distribuição dos rótulos e das "
            "predições em cada época."
        ),
    )

    return parser.parse_args()


def make_loader(subjects, args, shuffle, device):

    dataset = SleepAnalysisDataset(subjects)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        collate_fn=collate_sleep_analysis_exams,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    return loader



def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)
    all_subjects = load_subject_directory(args.data_dir)

    # ── Separação do conjunto de TESTE fixo (antes da CV) ──────────────────
    # Prioridade: --test-dir (externo) > --test-fraction (holdout do data-dir).
    # O test_subjects nunca entra na CV nem na seleção de checkpoint.
    test_subjects: list = []
    if args.test_dir is not None:
        test_subjects = load_subject_directory(args.test_dir)
        cv_subjects = all_subjects
    elif args.test_fraction and args.test_fraction > 0.0:
        cv_subjects, test_subjects = stratified_group_holdout(
            all_subjects,
            test_fraction=args.test_fraction,
            seed=args.seed,
            task="staging",
        )
    else:
        cv_subjects = all_subjects

    subjects = cv_subjects
    folds = list(stratified_group_folds(subjects, n_splits=args.n_splits, seed=args.seed, task="staging"))
    if args.fold is not None:
        folds = [item for item in folds if item[0] == args.fold]
        if not folds:
            raise ValueError(f"Fold {args.fold} não existe para n_splits={args.n_splits}.")

    if args.experiment_name == "staging_stratified_kfold":
        args.experiment_name = (
            f"staging_{args.model}_stratified_kfold"
        )

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
        # checkpoints por fold p/ avaliação no conjunto de teste (ensemble)
        fold_checkpoints: list[dict[str, Any]] = []

        logger.info(
            f"Sujeitos: total={len(all_subjects)} | "
            f"CV={len(subjects)} | teste={len(test_subjects)}"
        )
        if test_subjects:
            logger.log_subject_split(
                subjects, test_subjects, filename="test_split.json"
            )

        for fold, train_subjects, val_subjects in folds:
            seed_everything(args.seed + fold)

            fold_dir = logger.run_dir / f"fold_{fold}"
            checkpoint_dir = fold_dir / "checkpoints"
            figures_dir = fold_dir / "figures"

            checkpoint_dir.mkdir(
                parents=True,
                exist_ok=True,
            )
            figures_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            train_loader = make_loader(
                train_subjects,
                args,
                True,
                device,
            )

            val_loader = make_loader(
                val_subjects,
                args,
                False,
                device,
            )

            model_kwargs = {}

            if args.model in {"cnn_lstm", "cnn_bilstm"}:
                model_kwargs.update(
                    {
                        "hidden_size": args.lstm_hidden_size,
                        "num_layers": args.lstm_layers,
                    }
                )

            model = build_staging_model(
                args.model,
                **model_kwargs,
            ).to(device)

            if args.summary:
                if fold == folds[0][0]:
                    print_experiment_summary(
                        model=model,
                        model_name=args.model,
                        experiment_name=args.experiment_name,
                        device=str(device),
                        training_config={
                            "epochs": args.epochs,
                            "batch_size": args.batch_size,
                            "learning_rate": args.lr,
                            "weight_decay": args.weight_decay,
                            "num_workers": args.num_workers,
                        },
                        cross_validation_config={
                            "n_splits": args.n_splits,
                            "seed": args.seed,
                        },
                    )

                print()
                print("=" * 80)
                print(f"FOLD {fold}/{args.n_splits}")
                print("=" * 80)

                print_stage_distribution(
                    f"Fold {fold} - Train stage distribution",
                    train_loader.dataset.stage_distribution().as_dict(),
                )

                print_stage_distribution(
                    f"Fold {fold} - Validation stage distribution",
                    val_loader.dataset.stage_distribution().as_dict(),
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
                
            
            logger.info(
                f"Modelo: {args.model} | "
                f"parâmetros treináveis: {model.n_params():,}"
            )


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

            for epoch in range(1, args.epochs + 1):

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
                    **{
                        f"train_{key}": value
                        for key, value in scalar_metrics(train_metrics).items()
                    },
                    **{
                        f"val_{key}": value
                        for key, value in scalar_metrics(val_metrics).items()
                    },
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

                if args.log_stage_distribution:
                    train_targets = format_stage_distribution(
                        train_metrics["target_distribution"]
                    )

                    train_predictions = format_stage_distribution(
                        train_metrics["prediction_distribution"]
                    )

                    val_targets = format_stage_distribution(
                        val_metrics["target_distribution"]
                    )

                    val_predictions = format_stage_distribution(
                        val_metrics["prediction_distribution"]
                    )

                    logger.info(
                        f"ep={epoch:03d} "
                        f"train_targets[{train_targets}]"
                    )

                    logger.info(
                        f"ep={epoch:03d} "
                        f"train_predictions[{train_predictions}]"
                    )

                    logger.info(
                        f"ep={epoch:03d} "
                        f"val_targets[{val_targets}]"
                    )

                    logger.info(
                        f"ep={epoch:03d} "
                        f"val_predictions[{val_predictions}]"
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


            
            fold_checkpoints.append(
                {
                    "fold": fold,
                    "best_checkpoint": checkpoint_dir / "best.pt",
                    "model_kwargs": dict(model_kwargs),
                    "best_epoch": best_epoch,
                }
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

        # ── Fase de TESTE (conjunto held-out, nunca visto na CV) ───────────
        test_summary: dict[str, Any] | None = None
        if test_subjects:
            test_summary = evaluate_test_set(
                test_subjects=test_subjects,
                fold_checkpoints=fold_checkpoints,
                args=args,
                device=device,
                logger=logger,
                figures_dir=logger.run_dir / "test",
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
                "test": test_summary,
            },
        )

def evaluate_test_set(
    *,
    test_subjects: list,
    fold_checkpoints: list[dict[str, Any]],
    args: argparse.Namespace,
    device: torch.device,
    logger: ExperimentLogger,
    figures_dir: Path,
) -> dict[str, Any]:
    """Avalia o conjunto de teste held-out com os checkpoints de cada fold.

    Para cada fold, recarrega o best.pt e coleta as probabilidades no teste.
    Reporta métricas por fold e o ENSEMBLE (média das probabilidades entre os
    folds sobre as mesmas mini-épocas), com matrizes de confusão. O ensemble é
    a predição recomendada: usa todos os folds sem que nenhum tenha visto o
    teste no treino nem na seleção de checkpoint.
    """
    figures_dir.mkdir(parents=True, exist_ok=True)
    display_labels = ["W", "N1", "N2", "N3", "REM"]
    labels = [0, 1, 2, 3, 4]

    test_loader = make_loader(test_subjects, args, False, device)

    logger.info("=" * 80)
    logger.info(f"FASE DE TESTE | sujeitos={len(test_subjects)} | folds={len(fold_checkpoints)}")
    logger.info("=" * 80)

    per_fold: list[dict[str, float]] = []
    prob_sum: np.ndarray | None = None
    ref_expected: np.ndarray | None = None
    ref_keys: np.ndarray | None = None

    for entry in fold_checkpoints:
        fold = entry["fold"]
        checkpoint_path = entry["best_checkpoint"]
        if not Path(checkpoint_path).exists():
            logger.info(f"Fold {fold}: best.pt ausente ({checkpoint_path}); pulando no teste.")
            continue

        model = build_staging_model(args.model, **entry["model_kwargs"]).to(device)
        load_checkpoint(checkpoint_path, model, device)

        result = collect_staging_predictions(model, test_loader, device, amp=not args.no_amp)

        expected = result["expected"]
        prediction = result["prediction"]
        probs = result["probabilities"]
        # chave estável por mini-época p/ alinhar os folds antes de somar probs
        keys = np.array(
            [f"{s}#{i}" for s, i in zip(result["subject_id"], result["mini_epoch_index"])],
            dtype=object,
        )

        f1 = f1_score(expected, prediction, average="macro", zero_division=0)
        kappa = cohen_kappa_score(expected, prediction)
        bacc = balanced_accuracy_score(expected, prediction)
        logger.info(
            f"{YELLOW}TESTE fold {fold}: f1_macro={f1:.4f} "
            f"kappa={kappa:.4f} balanced_acc={bacc:.4f}{RESET}"
        )
        per_fold.append(
            {
                "fold": int(fold),
                "f1_macro": float(f1),
                "kappa": float(kappa),
                "balanced_accuracy": float(bacc),
                "n_samples": int(expected.size),
            }
        )

        plot_confusion_matrix(
            expected, prediction,
            figures_dir / f"confusion_matrix_test_fold_{fold}.png",
            labels=labels, display_labels=display_labels,
            title=f"Staging TEST - Fold {fold}",
        )

        # acumula probabilidades para o ensemble (alinhado por chave)
        if prob_sum is None:
            prob_sum = probs.copy()
            ref_expected = expected
            ref_keys = keys
        else:
            if not np.array_equal(keys, ref_keys):
                # reordena este fold para casar com a referência
                order = {k: j for j, k in enumerate(keys)}
                idx = np.array([order[k] for k in ref_keys], dtype=np.int64)
                probs = probs[idx]
            prob_sum = prob_sum + probs

    test_summary: dict[str, Any] = {
        "n_subjects": len(test_subjects),
        "per_fold": per_fold,
    }

    if prob_sum is not None and ref_expected is not None:
        ensemble_pred = prob_sum.argmax(axis=1).astype(np.int64)
        ens_f1 = f1_score(ref_expected, ensemble_pred, average="macro", zero_division=0)
        ens_kappa = cohen_kappa_score(ref_expected, ensemble_pred)
        ens_bacc = balanced_accuracy_score(ref_expected, ensemble_pred)
        logger.info(
            f"{GREEN}TESTE ENSEMBLE ({len(per_fold)} folds): "
            f"f1_macro={ens_f1:.4f} kappa={ens_kappa:.4f} "
            f"balanced_acc={ens_bacc:.4f}{RESET}"
        )

        plot_confusion_matrix(
            ref_expected, ensemble_pred,
            figures_dir / "confusion_matrix_test_ensemble.png",
            labels=labels, display_labels=display_labels,
            title="Staging TEST - Ensemble (média das probabilidades)",
        )
        plot_confusion_matrix(
            ref_expected, ensemble_pred,
            figures_dir / "confusion_matrix_test_ensemble_normalized.png",
            labels=labels, display_labels=display_labels, normalize="true",
            title="Staging TEST - Ensemble normalizada",
        )

        f1_vals = np.asarray([f["f1_macro"] for f in per_fold], dtype=np.float64)
        test_summary["per_fold_f1_macro_mean"] = float(f1_vals.mean()) if f1_vals.size else None
        test_summary["per_fold_f1_macro_std"] = (
            float(f1_vals.std(ddof=1)) if f1_vals.size > 1 else 0.0
        )
        test_summary["ensemble"] = {
            "n_samples": int(ref_expected.size),
            "n_folds": len(per_fold),
            "f1_macro": float(ens_f1),
            "kappa": float(ens_kappa),
            "balanced_accuracy": float(ens_bacc),
        }

    return test_summary


def scalar_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        key: float(value)
        for key, value in metrics.items()
        if isinstance(value, (int, float))
    }

def format_stage_distribution(
    distribution: Mapping[
        str,
        Mapping[str, Any],
    ],
    ) -> str:
    parts = []

    for stage, values in distribution.items():
        count = int(values["count"])
        percentage = float(values["percentage"])

        parts.append(
            f"{stage}={count:,}({percentage:.1f}%)"
        )

    return " | ".join(parts)

if __name__ == "__main__":
    main()
