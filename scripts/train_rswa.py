from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from sleep_rswa import RSWADetectionNet, SleepAnalysisDataset, collate_sleep_analysis_exams
from sleep_rswa.training import (
    RSWALoss,
    load_train_val_subjects,
    resolve_device,
    run_rswa_epoch,
    save_checkpoint,
    seed_everything,
    write_history,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treina o modelo de detecção de RSWA.")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--train-dir", type=Path)
    parser.add_argument("--val-dir", type=Path)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--all-stages", action="store_true", help="Não restringe o RSWA às miniépocas REM.")
    parser.add_argument("--tonic-pos-weight", type=float)
    parser.add_argument("--phasic-pos-weight", type=float)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/rswa"))
    parser.add_argument("--patience", type=int, default=15)
    return parser.parse_args()


def make_loader(dataset: SleepAnalysisDataset, args: argparse.Namespace, shuffle: bool, device: torch.device) -> DataLoader:
    return DataLoader(
        dataset,
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
    train_subjects, val_subjects = load_train_val_subjects(
        data_dir=args.data_dir,
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )

    ds_kwargs = {"min_confidence": args.min_confidence, "rem_mask_only": not args.all_stages}
    train_loader = make_loader(SleepAnalysisDataset(train_subjects, **ds_kwargs), args, True, device)
    val_loader = make_loader(SleepAnalysisDataset(val_subjects, **ds_kwargs), args, False, device)

    model = RSWADetectionNet().to(device)
    tonic_weight = None if args.tonic_pos_weight is None else torch.tensor(args.tonic_pos_weight, device=device)
    phasic_weight = None if args.phasic_pos_weight is None else torch.tensor(args.phasic_pos_weight, device=device)
    criterion = RSWALoss(tonic_weight, phasic_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(f"Dispositivo: {device}")
    print(f"Treino: {len(train_subjects)} sujeitos | Validação: {len(val_subjects)} sujeitos")
    print(f"Parâmetros treináveis: {model.n_params():,}")

    history: list[dict[str, float | int]] = []
    best_f1 = float("-inf")
    stale_epochs = 0

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_rswa_epoch(
            model, train_loader, criterion, device, optimizer,
            amp=not args.no_amp, grad_clip=args.grad_clip, threshold=args.threshold,
        )
        val_metrics = run_rswa_epoch(
            model, val_loader, criterion, device,
            amp=not args.no_amp, threshold=args.threshold,
        )
        row = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(row)
        print(
            f"ep={epoch:03d} "
            f"train loss={train_metrics['loss']:.4f} rswa_f1={train_metrics['rswa_f1_macro']:.4f} "
            f"tonic={train_metrics['tonic_f1']:.4f} phasic={train_metrics['phasic_f1']:.4f} | "
            f"val loss={val_metrics['loss']:.4f} rswa_f1={val_metrics['rswa_f1_macro']:.4f} "
            f"tonic={val_metrics['tonic_f1']:.4f} phasic={val_metrics['phasic_f1']:.4f}"
        )

        save_checkpoint(
            args.checkpoint_dir / "last.pt",
            model=model, optimizer=optimizer, epoch=epoch, metrics=val_metrics,
            extra={"task": "rswa", "args": vars(args)},
        )
        if val_metrics["rswa_f1_macro"] > best_f1:
            best_f1 = val_metrics["rswa_f1_macro"]
            stale_epochs = 0
            save_checkpoint(
                args.checkpoint_dir / "best.pt",
                model=model, optimizer=optimizer, epoch=epoch, metrics=val_metrics,
                extra={"task": "rswa", "args": vars(args)},
            )
        else:
            stale_epochs += 1

        write_history(args.checkpoint_dir / "history.json", history)
        if stale_epochs >= args.patience:
            print(f"Early stopping após {args.patience} épocas sem melhora.")
            break


if __name__ == "__main__":
    main()
