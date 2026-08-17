from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter
from typing import Any

from sklearn.metrics import cohen_kappa_score, f1_score

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, WeightedRandomSampler

from sleep_rswa.config import ModelConfig
from sleep_rswa import (
    available_staging_models,
    build_movement_model,
    build_staging_model,
    RSWADetectionNet,
    SharedBiMambaJointSystem,
    SleepAnalysisDataset,
    SleepStagingNet,
    SleepStagingRSWASystem,
    collate_sleep_analysis_exams,
)
from sleep_rswa.data import load_subject_directory
from sleep_rswa.distribution import StageDistribution
from sleep_rswa.training.engine import _binary_distribution
from sleep_rswa.utils import (
    format_stage_distribution,
    print_movement_distribution,
    print_split_summary,
    print_stage_distribution,
)
from sleep_rswa.training import (
    ExperimentLogger,
    RSWALoss,
    StagingLoss,
    collect_rswa_predictions,
    collect_staging_predictions,
    describe_split,
    evaluate_joint,
    evaluate_movement_test_set,
    evaluate_staging_test_set,
    format_split_description,
    load_checkpoint,
    plot_confusion_matrix,
    ResourceMonitor,
    resolve_device,
    save_rswa_predictions_csv,
    save_staging_predictions_csv,
    save_checkpoint,
    seed_everything,
    stratified_group_folds,
    stratified_group_holdout,
)
from sleep_rswa.models.mamba import mamba_backend_detail, mamba_backend_name

GREEN  = "\033[92m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
PURPLE = "\033[95m"
ORANGE = "\033[38;5;214m"
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
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.2,
        help=(
            "Fração de sujeitos separada como conjunto de TESTE fixo, antes da "
            "validação cruzada, estratificada por --test-stratify-by. Use 0 para "
            "desativar. Ignorado se --test-dir for dado."
        ),
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=None,
        help="Diretório .pt de teste EXTERNO; se dado, --data-dir vai inteiro para a CV.",
    )
    parser.add_argument(
        "--test-stratify-by",
        choices=["staging", "rswa"],
        default="rswa",
        help=(
            "Rótulo que estratifica o holdout de TESTE. Padrão 'rswa' porque "
            "movimento é raro e o teste precisa de positivos suficientes para "
            "um F1 de movimento estável."
        ),
    )
    parser.add_argument(
        "--model", choices=available_staging_models(), default="cnn_bimamba",
        help="Arquitetura aplicada a AMBOS os ramos (staging e movimento/RSWA).",
    )
    parser.add_argument(
        "--joint-topology",
        choices=["shared", "separate"],
        default="separate",
        help=(
            "Topologia conjunta. 'shared' habilita apenas o sistema com BiMamba "
            "compartilhada entre staging e RSWA; 'separate' usa ramos temporais "
            "independentes."
        ),
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr-staging", type=float, default=1e-4)
    parser.add_argument("--lr-rswa", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.5, help="Limiar default aplicado às 3 cabeças (tonic/phasic/any) do ramo RSWA.")
    parser.add_argument("--tonic-threshold", type=float, default=None)
    parser.add_argument("--phasic-threshold", type=float, default=None)
    parser.add_argument("--any-threshold", type=float, default=None)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--all-stages", action="store_true")
    parser.add_argument("--tonic-pos-weight", type=float)
    parser.add_argument("--phasic-pos-weight", type=float)
    parser.add_argument("--any-pos-weight", type=float)
    parser.add_argument(
        "--rswa-target-mode",
        choices=["final", "aasm_proto"],
        default="final",
        help="Usa labels finais ou alvos intermediarios AASM para tonic/phasic.",
    )
    parser.add_argument(
        "--rswa-postprocess-mode",
        choices=["none", "aasm_simple"],
        default="none",
        help="Pos-processamento aplicado nas predições RSWA na validacao/inferencia.",
    )
    parser.add_argument(
        "--rswa-use-baseline-relative-channel",
        action="store_true",
        help="Adiciona ao ramo RSWA um segundo canal |EMG| / rem_baseline_uv.",
    )
    parser.add_argument(
        "--rswa-use-rms-relative-channel",
        action="store_true",
        help=(
            "Adiciona ao ramo RSWA um canal extra com envelope RMS de 100 ms "
            "relativo ao basal de atonia. Exige --rswa-use-baseline-relative-channel."
        ),
    )
    parser.add_argument(
        "--use-emg-subwindow-features",
        action="store_true",
        help=(
            "Ativa o encoder local de sub-janelas do EMG (features RMS/MAV/STD "
            "em resolução mais fina dentro de cada mini-época de 3 s)."
        ),
    )
    parser.add_argument(
        "--emg-subwindow-ms",
        type=int,
        default=250,
        help="Tamanho, em ms, das sub-janelas intramini-época do ramo EMG.",
    )
    parser.add_argument(
        "--oversample-tonic-subjects",
        action="store_true",
        help="Aumenta a frequência de sujeitos que contêm ao menos um evento tônico no train_loader.",
    )
    parser.add_argument(
        "--tonic-subject-weight",
        type=float,
        default=5.0,
        help="Peso aplicado a sujeitos com tonic>0 quando --oversample-tonic-subjects está ativo.",
    )
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument(
        "--log-movement-distribution", action="store_true",
        help="Mostra target/prediction de staging e movimento (train e val) por época.",
    )
    parser.add_argument(
        "--monitor",
        choices=[
            "joint_mean_f1", "staging_f1_macro",
            "rswa_tonic_f1", "rswa_phasic_f1", "rswa_any_f1",
            "rswa_rswa_f1_macro", "rswa_rswa_kappa_macro",
            "rswa_movement_f1", "rswa_movement_kappa",
        ],
        default="joint_mean_f1",
        help="Métrica de validação usada para selecionar o melhor checkpoint de cada fold. rswa_rswa_f1_macro = média das 3 cabeças (tonic/phasic/any).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--resource-log-interval-sec",
        type=float,
        default=30.0,
        help="Intervalo, em segundos, para registrar uso de CPU/RAM/GPU em resource_usage.csv.",
    )
    parser.add_argument("--run-dir", type=Path, default=Path("runs/joint"))
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--notes", default=None)
    parser.add_argument("--tags", nargs="*", default=[])
    return parser.parse_args()


def _resolve_thresholds(args) -> dict[str, float]:
    return {
        "tonic": args.tonic_threshold if args.tonic_threshold is not None else args.threshold,
        "phasic": args.phasic_threshold if args.phasic_threshold is not None else args.threshold,
        "any": args.any_threshold if args.any_threshold is not None else args.threshold,
    }


def make_loader(subjects, args, shuffle, device):
    if args.rswa_target_mode == "aasm_proto":
        missing = [
            subject.subject_id for subject in subjects
            if subject.tonic_proto_labels is None or subject.phasic_proto_labels is None
        ]
        if missing:
            raise ValueError(
                "rswa_target_mode='aasm_proto' exige .pt com tonic_proto_labels/phasic_proto_labels. "
                f"Exemplos ausentes: {', '.join(missing[:5])}"
            )
    ds = SleepAnalysisDataset(
        subjects,
        min_confidence=args.min_confidence,
        rem_mask_only=not args.all_stages,
        rswa_target_mode=args.rswa_target_mode,
        use_baseline_relative_channel=args.rswa_use_baseline_relative_channel,
        use_rms_relative_channel=args.rswa_use_rms_relative_channel,
    )
    sampler = None
    if shuffle and args.oversample_tonic_subjects:
        weights = torch.as_tensor(
            [
                args.tonic_subject_weight if _subject_has_tonic(subject) else 1.0
                for subject in ds.subjects
            ],
            dtype=torch.double,
        )
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True,
        )
    return DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=(shuffle and sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=collate_sleep_analysis_exams,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )


def _subject_has_tonic(subject) -> bool:
    if subject.tonic_labels is not None:
        return bool((subject.tonic_labels > 0.5).any().item())
    return bool((subject.rswa_labels == 2).any().item())


def _count_tonic_subjects(subjects) -> int:
    return sum(1 for subject in subjects if _subject_has_tonic(subject))


def joint_monitor_value(monitor: str, val_metrics: dict[str, float]) -> float:
    """Valor da métrica monitorada. 'joint_mean_f1' = média de staging_f1_macro e rswa_f1_macro
    (média das 3 cabeças tonic/phasic/any)."""
    if monitor == "joint_mean_f1":
        scores = [
            val_metrics.get("staging_f1_macro", float("nan")),
            val_metrics.get("rswa_rswa_f1_macro", float("nan")),
        ]
        scores = [x for x in scores if x == x]
        return sum(scores) / len(scores) if scores else float("-inf")
    return float(val_metrics.get(monitor, float("-inf")))


def _use_shared_joint_system(model_name: str, joint_topology: str) -> bool:
    if joint_topology == "shared":
        if model_name != "cnn_bimamba":
            raise ValueError(
                "--joint-topology shared está disponível apenas para --model cnn_bimamba."
            )
        return True
    return False


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
    axes[1].plot(epochs, [r.get("val_rswa_tonic_f1", float("nan")) for r in history], label="Val tonic F1")
    axes[1].plot(epochs, [r.get("val_rswa_phasic_f1", float("nan")) for r in history], label="Val phasic F1")
    axes[1].plot(epochs, [r.get("val_rswa_any_f1", float("nan")) for r in history], label="Val any F1")
    axes[1].set_ylabel("F1"); axes[1].set_xlabel("Epoch"); axes[1].legend(); axes[1].grid(alpha=0.25)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _print_model_summary(name: str, model: torch.nn.Module, logger: Any) -> None:
    n_params = (
        model.n_params()
        if hasattr(model, "n_params")
        else sum(p.numel() for p in model.parameters() if p.requires_grad)
    )
    header = f"[MODEL] {name} | trainable_params={n_params:,}"
    print(header)
    print(model)
    logger.info(header)
    logger.info(str(model))


def main() -> None:
    args = parse_args()
    if args.rswa_use_rms_relative_channel and not args.rswa_use_baseline_relative_channel:
        raise ValueError(
            "--rswa-use-rms-relative-channel exige --rswa-use-baseline-relative-channel."
        )
    shared_joint = _use_shared_joint_system(args.model, args.joint_topology)
    if args.experiment_name is None:
        args.experiment_name = f"joint_{args.model}_stratified_kfold"
    seed_everything(args.seed)
    device = resolve_device(args.device)
    all_subjects = load_subject_directory(args.data_dir)
    rswa_model_cfg = ModelConfig(
        rswa_stage_conditioning=True,
        rswa_emg_in_channels=(
            3
            if args.rswa_use_baseline_relative_channel and args.rswa_use_rms_relative_channel
            else (2 if args.rswa_use_baseline_relative_channel else 1)
        ),
        rswa_use_baseline_relative_channel=args.rswa_use_baseline_relative_channel,
        rswa_use_rms_relative_channel=args.rswa_use_rms_relative_channel,
        use_emg_subwindow_features=args.use_emg_subwindow_features,
        emg_subwindow_ms=args.emg_subwindow_ms,
    )

    # ── Conjunto de TESTE fixo (held-out), separado ANTES da CV ────────────
    test_subjects: list = []
    if args.test_dir is not None:
        test_subjects = load_subject_directory(args.test_dir)
        cv_subjects = all_subjects
    elif args.test_fraction and args.test_fraction > 0.0:
        cv_subjects, test_subjects = stratified_group_holdout(
            all_subjects, test_fraction=args.test_fraction, seed=args.seed, task=args.test_stratify_by
        )
    else:
        cv_subjects = all_subjects

    subjects = cv_subjects
    folds = list(stratified_group_folds(subjects, n_splits=args.n_splits, seed=args.seed, task=args.stratify_by))
    if args.fold is not None:
        folds = [item for item in folds if item[0] == args.fold]
        if not folds:
            raise ValueError(f"Fold {args.fold} não existe para n_splits={args.n_splits}.")

    with ExperimentLogger(
        task="joint", experiment_name=args.experiment_name, root_dir=args.run_dir,
        device=device, args=vars(args), notes=args.notes, tags=args.tags,
    ) as logger:
        with ResourceMonitor(
            logger.run_dir / "resource_usage.csv",
            device=device,
            interval_sec=args.resource_log_interval_sec,
        ):
            logger.info(f"Dispositivo: {device}")
            logger.info(f"Modelo (staging + movimento): {args.model}")
            logger.info(
                "Topologia conjunta: "
                + (
                    "encoder EEG/EOG + encoder EMG + fusao + BiMamba compartilhado"
                    if shared_joint
                    else "staging e RSWA com troncos temporais separados"
                )
            )
            logger.info(
                f"Backend temporal BiMamba: {mamba_backend_name()} | "
                f"{mamba_backend_detail()}"
            )
            logger.info(
                f"RSWA experimental: target_mode={args.rswa_target_mode} "
                f"postprocess_mode={args.rswa_postprocess_mode} "
                f"use_baseline_relative_channel={args.rswa_use_baseline_relative_channel} "
                f"use_rms_relative_channel={args.rswa_use_rms_relative_channel} "
                f"use_emg_subwindow_features={args.use_emg_subwindow_features} "
                f"emg_subwindow_ms={args.emg_subwindow_ms}"
            )
            logger.info(
                f"Sujeitos: total={len(all_subjects)} | CV={len(subjects)} | teste={len(test_subjects)} | "
                f"n_splits={args.n_splits} | estratificação CV={args.stratify_by} | teste={args.test_stratify_by}"
            )

            fold_summaries = []
            staging_checkpoints: list[dict[str, Any]] = []
            rswa_checkpoints: list[dict[str, Any]] = []
            data_report: dict[str, Any] = {"folds": []}

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
                predictions_dir = fold_dir / "predictions"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                figures_dir.mkdir(parents=True, exist_ok=True)
                predictions_dir.mkdir(parents=True, exist_ok=True)
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
                train_tonic_subjects = _count_tonic_subjects(train_subjects)
                val_tonic_subjects = _count_tonic_subjects(val_subjects)
                logger.info(
                    f"Fold {fold}: sujeitos com tonic -> treino={train_tonic_subjects}/{len(train_subjects)} "
                    f"validação={val_tonic_subjects}/{len(val_subjects)} | "
                    f"oversample_tonic_subjects={args.oversample_tonic_subjects} "
                    f"tonic_subject_weight={args.tonic_subject_weight:.2f}"
                )

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
                    split_name="Train", subjects=train_subjects,
                    dataset=train_loader.dataset, loader=train_loader,
                )
                print_split_summary(
                    split_name="Validation", subjects=val_subjects,
                    dataset=val_loader.dataset, loader=val_loader,
                )

                if shared_joint:
                    system = SharedBiMambaJointSystem(config=rswa_model_cfg).to(device)
                    staging_model = None
                    rswa_model = None
                    _print_model_summary("shared_joint_system", system, logger)
                    _print_model_summary("shared_joint_staging_encoder", system.staging_encoder, logger)
                    _print_model_summary("shared_joint_rswa_encoder", system.rswa_encoder, logger)
                    _print_model_summary("shared_joint_temporal", system.temporal, logger)
                else:
                    staging_model = build_staging_model(args.model, config=rswa_model_cfg).to(device)
                    rswa_model = build_movement_model(
                        args.model, config=rswa_model_cfg, stage_conditioning=True
                    ).to(device)
                    system = SleepStagingRSWASystem(staging_model, rswa_model).to(device)
                    _print_model_summary("staging_model", staging_model, logger)
                    _print_model_summary("rswa_model", rswa_model, logger)
                _print_model_summary("joint_system", system, logger)
                staging_loss_fn = StagingLoss()
                tonic_weight = torch.tensor(args.tonic_pos_weight, device=device) if args.tonic_pos_weight else None
                phasic_weight = torch.tensor(args.phasic_pos_weight, device=device) if args.phasic_pos_weight else None
                any_weight = torch.tensor(args.any_pos_weight, device=device) if args.any_pos_weight else None
                rswa_loss_fn = RSWALoss(tonic_pos_weight=tonic_weight, phasic_pos_weight=phasic_weight, any_pos_weight=any_weight)
                thresholds = _resolve_thresholds(args)
                if shared_joint:
                    joint_optimizer = torch.optim.AdamW(
                        [
                            {
                                "params": list(system.staging_encoder.parameters())
                                + list(system.staging_classifier.parameters()),
                                "lr": args.lr_staging,
                            },
                            {
                                "params": list(system.rswa_encoder.parameters())
                                + list(system.fusion.parameters())
                                + list(system.temporal.parameters())
                                + list(system.tonic_head.parameters())
                                + list(system.phasic_head.parameters())
                                + list(system.any_head.parameters()),
                                "lr": args.lr_rswa,
                            },
                        ],
                        weight_decay=args.weight_decay,
                    )
                    staging_optimizer = None
                    rswa_optimizer = None
                else:
                    staging_optimizer = torch.optim.AdamW(
                        staging_model.parameters(), lr=args.lr_staging, weight_decay=args.weight_decay
                    )
                    rswa_optimizer = torch.optim.AdamW(
                        rswa_model.parameters(), lr=args.lr_rswa, weight_decay=args.weight_decay
                    )
                    joint_optimizer = None

                logger.log_subject_split(train_subjects, val_subjects, filename=f"fold_{fold}_split.json")
                logger.info(f"Fold {fold}: treino={len(train_subjects)} validação={len(val_subjects)}")

                best_metric = float("-inf")
                best_epoch = 0
                stale = 0
                best_metrics: dict[str, float] = {}
                history: list[dict[str, float]] = []
                shape_logged = False

                for epoch in range(1, args.epochs + 1):
                    epoch_start = perf_counter()
                    train_start = perf_counter()
                    system.train()
                    stage_loss_sum = 0.0
                    rswa_loss_sum = 0.0
                    rswa_head_loss_sums = {"tonic": 0.0, "phasic": 0.0, "any": 0.0}
                    stage_batches = 0
                    rswa_batches = 0
                    tr_stage_targets: list[torch.Tensor] = []
                    tr_stage_preds: list[torch.Tensor] = []
                    tr_move_targets: list[torch.Tensor] = []
                    tr_move_preds: list[torch.Tensor] = []
                    tr_head_targets: dict[str, list[torch.Tensor]] = {h: [] for h in ("tonic", "phasic", "any")}
                    tr_head_preds: dict[str, list[torch.Tensor]] = {h: [] for h in ("tonic", "phasic", "any")}

                    for batch in train_loader:
                        signals = batch["signals"].to(device, non_blocking=True)
                        emg = batch["emg_center"].to(device, non_blocking=True)
                        padding_mask = batch["padding_mask"].to(device, non_blocking=True)
                        stage_targets = batch["sleep_stages"].to(device, non_blocking=True)
                        tonic_targets = batch["tonic_targets"].to(device, non_blocking=True)
                        phasic_targets = batch["phasic_targets"].to(device, non_blocking=True)
                        any_targets = batch["any_targets"].to(device, non_blocking=True)
                        tonic_labels = batch["tonic_labels"].to(device, non_blocking=True)
                        phasic_labels = batch["phasic_labels"].to(device, non_blocking=True)
                        any_labels = batch["any_labels"].to(device, non_blocking=True)
                        stage_valid = batch["staging_valid"].to(device, non_blocking=True) & padding_mask
                        rswa_valid = batch["rswa_valid"].to(device, non_blocking=True) & padding_mask

                        if not stage_valid.any() and not rswa_valid.any():
                            continue

                        if joint_optimizer is not None:
                            joint_optimizer.zero_grad(set_to_none=True)
                        else:
                            staging_optimizer.zero_grad(set_to_none=True)
                            rswa_optimizer.zero_grad(set_to_none=True)
                        with torch.autocast(
                            device_type="cuda", dtype=torch.bfloat16,
                            enabled=(not args.no_amp and device.type == "cuda"),
                        ):
                            outputs = system(signals, emg, mask=padding_mask)
                        if not shape_logged:
                            shape_info = getattr(system, "last_shape_info", None)
                            if shape_info:
                                ordered_keys = [
                                    "emg_raw",
                                    "emg_subwindows",
                                    "emg_subwindow_features",
                                    "emg_local_embedding",
                                    "emg_cnn_embedding",
                                    "fused_emg_embedding",
                                    "staging_embedding",
                                    "pre_mamba_embedding",
                                ]
                                for key in ordered_keys:
                                    if key in shape_info:
                                        logger.info(f"shape[{key}]={shape_info[key]}")
                                for key, value in shape_info.items():
                                    if key not in ordered_keys:
                                        logger.info(f"shape[{key}]={value}")
                                shape_logged = True

                        if stage_valid.any():
                            stage_loss = staging_loss_fn(
                                outputs["staging_logits"], stage_targets, stage_valid
                            )
                            stage_loss_sum += float(stage_loss.detach().cpu())
                            stage_batches += 1
                            tr_stage_targets.append(stage_targets[stage_valid].detach().cpu())
                            tr_stage_preds.append(
                                outputs["staging_logits"].detach().argmax(dim=-1)[stage_valid].cpu()
                            )
                        else:
                            stage_loss = None

                        if rswa_valid.any():
                            rswa_loss, per_head = rswa_loss_fn(
                                outputs, tonic_targets, phasic_targets, any_targets, rswa_valid
                            )
                            rswa_loss_sum += float(rswa_loss.detach().cpu())
                            for head in ("tonic", "phasic", "any"):
                                rswa_head_loss_sums[head] += float(
                                    per_head[f"{head}_loss"].detach().cpu()
                                )
                            rswa_batches += 1
                            move_targets = (
                                (tonic_labels > 0.5) | (phasic_labels > 0.5) | (any_labels > 0.5)
                            ).float()
                            move_preds = (
                                (torch.sigmoid(outputs["tonic_logits"].detach()) >= thresholds["tonic"])
                                | (torch.sigmoid(outputs["phasic_logits"].detach()) >= thresholds["phasic"])
                                | (torch.sigmoid(outputs["any_logits"].detach()) >= thresholds["any"])
                            ).long()
                            tr_move_targets.append(move_targets[rswa_valid].long().detach().cpu())
                            tr_move_preds.append(move_preds[rswa_valid].cpu())
                            head_targets = {
                                "tonic": tonic_labels,
                                "phasic": phasic_labels,
                                "any": any_labels,
                            }
                            for head in ("tonic", "phasic", "any"):
                                head_preds = (
                                    torch.sigmoid(outputs[f"{head}_logits"].detach())
                                    >= thresholds[head]
                                ).long()
                                tr_head_targets[head].append(
                                    head_targets[head][rswa_valid].long().detach().cpu()
                                )
                                tr_head_preds[head].append(head_preds[rswa_valid].cpu())
                        else:
                            rswa_loss = None

                        total_loss = None
                        if stage_loss is not None:
                            total_loss = stage_loss
                        if rswa_loss is not None:
                            total_loss = rswa_loss if total_loss is None else (total_loss + rswa_loss)
                        if total_loss is None:
                            continue
                        total_loss.backward()
                        if joint_optimizer is not None:
                            clip_grad_norm_(system.parameters(), args.grad_clip)
                            joint_optimizer.step()
                        else:
                            clip_grad_norm_(staging_model.parameters(), args.grad_clip)
                            clip_grad_norm_(rswa_model.parameters(), args.grad_clip)
                            staging_optimizer.step()
                            rswa_optimizer.step()

                    train_time = perf_counter() - train_start

                    train_dist: dict[str, Any] = {}
                    if tr_stage_targets:
                        st_t = StageDistribution(); st_p = StageDistribution()
                        st_t.update(torch.cat(tr_stage_targets)); st_p.update(torch.cat(tr_stage_preds))
                        train_dist["staging_target_distribution"] = st_t.as_dict()
                        train_dist["staging_prediction_distribution"] = st_p.as_dict()
                    if tr_move_targets:
                        train_dist["movement_target_distribution"] = _binary_distribution(
                            torch.cat(tr_move_targets).numpy())
                        train_dist["movement_prediction_distribution"] = _binary_distribution(
                            torch.cat(tr_move_preds).numpy())
                    for head in ("tonic", "phasic", "any"):
                        if tr_head_targets[head]:
                            train_dist[f"{head}_target_distribution"] = _binary_distribution(
                                torch.cat(tr_head_targets[head]).numpy()
                            )
                            train_dist[f"{head}_prediction_distribution"] = _binary_distribution(
                                torch.cat(tr_head_preds[head]).numpy()
                            )

                    val_start = perf_counter()
                    val_metrics = evaluate_joint(
                        system, val_loader, staging_loss_fn, rswa_loss_fn, device,
                        amp=not args.no_amp, threshold=thresholds,
                        postprocess_mode=(
                            None if args.rswa_postprocess_mode == "none"
                            else args.rswa_postprocess_mode
                        ),
                    )
                    val_time = perf_counter() - val_start

                    row = {
                        "fold": fold,
                        "epoch": epoch,
                        "train_time_sec": train_time,
                        "val_time_sec": val_time,
                        "epoch_time_sec": perf_counter() - epoch_start,
                        "staging_learning_rate": (
                            joint_optimizer.param_groups[0]["lr"]
                            if joint_optimizer is not None
                            else staging_optimizer.param_groups[0]["lr"]
                        ),
                        "rswa_learning_rate": (
                            joint_optimizer.param_groups[1]["lr"]
                            if joint_optimizer is not None
                            else rswa_optimizer.param_groups[0]["lr"]
                        ),
                        "train_staging_loss": stage_loss_sum / max(stage_batches, 1),
                        "train_rswa_loss": rswa_loss_sum / max(rswa_batches, 1),
                        "train_rswa_tonic_loss": rswa_head_loss_sums["tonic"] / max(rswa_batches, 1),
                        "train_rswa_phasic_loss": rswa_head_loss_sums["phasic"] / max(rswa_batches, 1),
                        "train_rswa_any_loss": rswa_head_loss_sums["any"] / max(rswa_batches, 1),
                        **{f"val_{key}": value for key, value in val_metrics.items()
                           if isinstance(value, (int, float))},
                    }
                    history.append(row)
                    logger.log_epoch(row)
                    logger.info(
                        f"fold={fold} ep={epoch:03d} train={train_time:.1f}s val={val_time:.1f}s "
                        f"{GREEN}"
                        f"train_stg_loss={row['train_staging_loss']:.4f} "
                        f"train_rswa_loss={row['train_rswa_loss']:.4f} "
                        f"train_rswa_head_loss(t/p/a)="
                        f"{row['train_rswa_tonic_loss']:.4f}/"
                        f"{row['train_rswa_phasic_loss']:.4f}/"
                        f"{row['train_rswa_any_loss']:.4f}"
                        f"{RESET} | "
                        f"{YELLOW}"
                        f"val_stg_f1={val_metrics.get('staging_f1_macro', float('nan')):.4f} "
                        f"val_stg_kappa={val_metrics.get('staging_kappa', float('nan')):.4f} "
                        f"val_rswa_head_loss(t/p/a)="
                        f"{val_metrics.get('rswa_tonic_loss', float('nan')):.4f}/"
                        f"{val_metrics.get('rswa_phasic_loss', float('nan')):.4f}/"
                        f"{val_metrics.get('rswa_any_loss', float('nan')):.4f} "
                        f"val_f1(t/p/a)={val_metrics.get('rswa_tonic_f1', float('nan')):.3f}/"
                        f"{val_metrics.get('rswa_phasic_f1', float('nan')):.3f}/"
                        f"{val_metrics.get('rswa_any_f1', float('nan')):.3f} "
                        f"val_f1_macro={val_metrics.get('rswa_rswa_f1_macro', float('nan')):.4f}"
                        f"{RESET}"
                    )

                    if args.log_movement_distribution:
                        def _emit(tag, dist, color, key):
                            if key in dist:
                                logger.info(f"{color}{tag}[{format_stage_distribution(dist[key])}]{RESET}")
                        _emit("train_stage_targets", train_dist, GREEN, "staging_target_distribution")
                        _emit("train_stage_predictions", train_dist, GREEN, "staging_prediction_distribution")
                        _emit("train_move_targets", train_dist, GREEN, "movement_target_distribution")
                        _emit("train_move_predictions", train_dist, GREEN, "movement_prediction_distribution")
                        _emit("val_stage_targets", val_metrics, YELLOW, "staging_target_distribution")
                        _emit("val_stage_predictions", val_metrics, YELLOW, "staging_prediction_distribution")
                        _emit("val_move_targets", val_metrics, YELLOW, "movement_target_distribution")
                        _emit("val_move_predictions", val_metrics, YELLOW, "movement_prediction_distribution")
                        _emit("train_tonic_targets", train_dist, BLUE, "tonic_target_distribution")
                        _emit("train_tonic_predictions", train_dist, BLUE, "tonic_prediction_distribution")
                        _emit("val_tonic_targets", val_metrics, BLUE, "tonic_target_distribution")
                        _emit("val_tonic_predictions", val_metrics, BLUE, "tonic_prediction_distribution")
                        _emit("train_phasic_targets", train_dist, PURPLE, "phasic_target_distribution")
                        _emit("train_phasic_predictions", train_dist, PURPLE, "phasic_prediction_distribution")
                        _emit("val_phasic_targets", val_metrics, PURPLE, "phasic_target_distribution")
                        _emit("val_phasic_predictions", val_metrics, PURPLE, "phasic_prediction_distribution")
                        _emit("train_any_targets", train_dist, ORANGE, "any_target_distribution")
                        _emit("train_any_predictions", train_dist, ORANGE, "any_prediction_distribution")
                        _emit("val_any_targets", val_metrics, ORANGE, "any_target_distribution")
                        _emit("val_any_predictions", val_metrics, ORANGE, "any_prediction_distribution")

                    if joint_optimizer is not None:
                        save_checkpoint(
                            checkpoint_dir / "joint_last.pt",
                            model=system,
                            optimizer=joint_optimizer,
                            epoch=epoch,
                            metrics=val_metrics,
                            extra={
                                "task": "joint",
                                "trained_with": "shared_bimamba",
                                "fold": fold,
                                "run_id": logger.run_id,
                            },
                        )
                    else:
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
                        if joint_optimizer is not None:
                            save_checkpoint(
                                checkpoint_dir / "joint_best.pt",
                                model=system,
                                optimizer=joint_optimizer,
                                epoch=epoch,
                                metrics=val_metrics,
                                extra={
                                    "task": "joint",
                                    "trained_with": "shared_bimamba",
                                    "fold": fold,
                                    "monitor": args.monitor,
                                    "monitor_value": current_metric,
                                    "run_id": logger.run_id,
                                },
                            )
                        else:
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

                if joint_optimizer is not None:
                    load_checkpoint(checkpoint_dir / "joint_best.pt", system, device)
                    stage_pred = collect_staging_predictions(system, val_loader, device, amp=not args.no_amp)
                else:
                    load_checkpoint(checkpoint_dir / "staging_best.pt", staging_model, device)
                    load_checkpoint(checkpoint_dir / "rswa_best.pt", rswa_model, device)
                    stage_pred = collect_staging_predictions(staging_model, val_loader, device, amp=not args.no_amp)
                move_pred = collect_rswa_predictions(
                    system, val_loader, device, amp=not args.no_amp, threshold=thresholds,
                    postprocess_mode=(
                        None if args.rswa_postprocess_mode == "none"
                        else args.rswa_postprocess_mode
                    ),
                )
                save_staging_predictions_csv(
                    predictions_dir / "validation_staging_best.csv",
                    stage_pred,
                    split="validation",
                    fold=int(fold),
                    source="best_checkpoint",
                )
                save_rswa_predictions_csv(
                    predictions_dir / "validation_rswa_best.csv",
                    move_pred,
                    split="validation",
                    fold=int(fold),
                    source="best_checkpoint",
                )
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
                for head in ("tonic", "phasic", "any", "movement"):
                    plot_confusion_matrix(
                        move_pred[f"{head}_expected"], move_pred[f"{head}_prediction"],
                        figures_dir / f"confusion_matrix_{head}.png", labels=[0, 1],
                        display_labels=["Negative", "Positive"], title=f"{head.capitalize()} confusion matrix - Fold {fold}",
                    )
                    plot_confusion_matrix(
                        move_pred[f"{head}_expected"], move_pred[f"{head}_prediction"],
                        figures_dir / f"confusion_matrix_{head}_normalized.png", labels=[0, 1],
                        display_labels=["Negative", "Positive"], title=f"{head.capitalize()} normalized confusion matrix - Fold {fold}", normalize="true",
                    )

                if joint_optimizer is not None:
                    staging_checkpoints.append({"fold": fold, "best_checkpoint": checkpoint_dir / "joint_best.pt"})
                    rswa_checkpoints.append({"fold": fold, "best_checkpoint": checkpoint_dir / "joint_best.pt"})
                else:
                    staging_checkpoints.append({"fold": fold, "best_checkpoint": checkpoint_dir / "staging_best.pt"})
                    rswa_checkpoints.append(
                        {
                            "fold": fold,
                            "staging_checkpoint": checkpoint_dir / "staging_best.pt",
                            "rswa_checkpoint": checkpoint_dir / "rswa_best.pt",
                        }
                    )
                fold_summaries.append(
                    {
                        "fold": fold,
                        "best_epoch": best_epoch,
                        "monitor": args.monitor,
                        "best_monitor_value": best_metric,
                        "best_val_staging_f1_macro": best_metrics.get("staging_f1_macro"),
                        "best_val_staging_kappa": best_metrics.get("staging_kappa"),
                        "best_val_tonic_f1": best_metrics.get("rswa_tonic_f1"),
                        "best_val_phasic_f1": best_metrics.get("rswa_phasic_f1"),
                        "best_val_any_f1": best_metrics.get("rswa_any_f1"),
                        "best_val_rswa_f1_macro": best_metrics.get("rswa_rswa_f1_macro"),
                        "best_val_rswa_kappa_macro": best_metrics.get("rswa_rswa_kappa_macro"),
                    }
                )

            # ── Fase de TESTE (held-out): ensemble dos folds, staging e movimento ─
            staging_test_summary: dict[str, Any] | None = None
            movement_test_summary: dict[str, Any] | None = None
            if test_subjects and staging_checkpoints:
                test_loader = make_loader(test_subjects, args, False, device)
                test_predictions_dir = logger.run_dir / "test" / "predictions"
                staging_test_summary = evaluate_staging_test_set(
                    test_loader=test_loader, fold_checkpoints=staging_checkpoints,
                    build_model=(
                        (lambda: SharedBiMambaJointSystem(config=rswa_model_cfg))
                        if shared_joint
                        else (lambda: build_staging_model(args.model, config=rswa_model_cfg))
                    ),
                    device=device, logger=logger,
                    figures_dir=logger.run_dir / "test",
                    predictions_dir=test_predictions_dir,
                    amp=not args.no_amp,
                )
                movement_test_summary = evaluate_movement_test_set(
                    test_loader=test_loader, fold_checkpoints=rswa_checkpoints,
                    build_model=(
                        (lambda: SharedBiMambaJointSystem(config=rswa_model_cfg))
                        if shared_joint
                        else (
                            lambda: SleepStagingRSWASystem(
                                build_staging_model(args.model, config=rswa_model_cfg),
                                build_movement_model(args.model, config=rswa_model_cfg, stage_conditioning=True),
                            )
                        )
                    ),
                    device=device, logger=logger,
                    figures_dir=logger.run_dir / "test",
                    predictions_dir=test_predictions_dir,
                    amp=not args.no_amp, threshold=thresholds,
                    postprocess_mode=(
                        None if args.rswa_postprocess_mode == "none"
                        else args.rswa_postprocess_mode
                    ),
                )

            logger.write_json("data_description.json", data_report)

            staging_f1_values = np.asarray(
                [f["best_val_staging_f1_macro"] for f in fold_summaries if f["best_val_staging_f1_macro"] is not None],
                dtype=np.float64,
            )
            head_f1_values = {
                head: np.asarray(
                    [f[f"best_val_{head}_f1"] for f in fold_summaries if f.get(f"best_val_{head}_f1") is not None],
                    dtype=np.float64,
                )
                for head in ("tonic", "phasic", "any")
            }

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
                        "tonic_f1": _mean_std(head_f1_values["tonic"]),
                        "phasic_f1": _mean_std(head_f1_values["phasic"]),
                        "any_f1": _mean_std(head_f1_values["any"]),
                    },
                    "test": {
                        "stratify_by": args.test_stratify_by,
                        "staging": staging_test_summary,
                        "movement": movement_test_summary,
                    },
                    "data_description": data_report,
                },
            )


if __name__ == "__main__":
    main()
