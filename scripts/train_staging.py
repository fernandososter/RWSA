from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from sleep_rswa import SleepAnalysisDataset, SleepStagingNet, collate_sleep_analysis_exams
from sleep_rswa.training import (
    StagingLoss,
    load_train_val_subjects,
    resolve_device,
    run_staging_epoch,
    save_checkpoint,
    seed_everything,
    write_history,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treina o modelo de estagiamento do sono.")
    parser.add_argument("--data-dir", type=Path, help="Diretório único com arquivos .pt.")
    parser.add_argument("--train-dir", type=Path, help="Diretório de treino com arquivos .pt.")
    parser.add_argument("--val-dir", type=Path, help="Diretório de validação com arquivos .pt.")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda ou cuda:0.")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/staging"))
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument(
        "--class-weights",
        type=float,
        nargs=5,
        metavar=("W", "N1", "N2", "N3", "REM"),
        default=None,
    )
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
    train_loader = make_loader(SleepAnalysisDataset(train_subjects), args, True, device)
    val_loader = make_loader(SleepAnalysisDataset(val_subjects), args, False, device)

    model = SleepStagingNet().to(device)
    class_weights = None
    if args.class_weights is not None:
        class_weights = torch.tensor(args.class_weights, dtype=torch.float32, device=device)
    criterion = StagingLoss(class_weights=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(f"Dispositivo: {device}")
    print(f"Treino: {len(train_subjects)} sujeitos | Validação: {len(val_subjects)} sujeitos")
    print(f"Parâmetros treináveis: {model.n_params():,}")

    history: list[dict[str, float | int]] = []
    best_f1 = float("-inf")
    stale_epochs = 0

    for epoch in tqdm(range(1, args.epochs + 1), desc="Treinando", unit="época"):
        train_metrics = run_staging_epoch(
            model, train_loader, criterion, device, optimizer,
            amp=not args.no_amp, grad_clip=args.grad_clip,
        )
        val_metrics = run_staging_epoch(
            model, val_loader, criterion, device,
            amp=not args.no_amp,
        )
        row = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(row)
        print(
            f"ep={epoch:03d} "
            f"train loss={train_metrics['loss']:.4f} f1={train_metrics['f1_macro']:.4f} k={train_metrics['kappa']:.4f} | "
            f"val loss={val_metrics['loss']:.4f} f1={val_metrics['f1_macro']:.4f} k={val_metrics['kappa']:.4f}"
        )

        save_checkpoint(
            args.checkpoint_dir / "last.pt",
            model=model, optimizer=optimizer, epoch=epoch, metrics=val_metrics,
            extra={"task": "staging", "args": vars(args)},
        )
        if val_metrics["f1_macro"] > best_f1:
            best_f1 = val_metrics["f1_macro"]
            stale_epochs = 0
            save_checkpoint(
                args.checkpoint_dir / "best.pt",
                model=model, optimizer=optimizer, epoch=epoch, metrics=val_metrics,
                extra={"task": "staging", "args": vars(args)},
            )
        else:
            stale_epochs += 1

        write_history(args.checkpoint_dir / "history.json", history)
        if stale_epochs >= args.patience:
            print(f"Early stopping após {args.patience} épocas sem melhora.")
            break


if __name__ == "__main__":
    main()
