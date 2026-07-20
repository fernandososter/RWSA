from __future__ import annotations

import argparse
from pathlib import Path

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
from sleep_rswa.training import (
    RSWALoss,
    StagingLoss,
    evaluate_joint,
    load_train_val_subjects,
    resolve_device,
    save_checkpoint,
    seed_everything,
    write_history,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Treina staging e RSWA no mesmo DataLoader com otimizadores independentes."
    )
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--train-dir", type=Path)
    parser.add_argument("--val-dir", type=Path)
    parser.add_argument("--val-fraction", type=float, default=0.2)
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/joint"))
    return parser.parse_args()


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
    dataset_kwargs = {
        "min_confidence": args.min_confidence,
        "rem_mask_only": not args.all_stages,
    }
    train_dataset = SleepAnalysisDataset(train_subjects, **dataset_kwargs)
    val_dataset = SleepAnalysisDataset(val_subjects, **dataset_kwargs)

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "collate_fn": collate_sleep_analysis_exams,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    staging_model = SleepStagingNet().to(device)
    rswa_model = RSWADetectionNet().to(device)
    system = SleepStagingRSWASystem(staging_model, rswa_model).to(device)
    staging_loss_fn = StagingLoss()
    rswa_loss_fn = RSWALoss()
    staging_optimizer = torch.optim.AdamW(
        staging_model.parameters(), lr=args.lr_staging, weight_decay=args.weight_decay
    )
    rswa_optimizer = torch.optim.AdamW(
        rswa_model.parameters(), lr=args.lr_rswa, weight_decay=args.weight_decay
    )

    print(f"Dispositivo: {device}")
    print(f"Staging: {staging_model.n_params():,} parâmetros")
    print(f"RSWA: {rswa_model.n_params():,} parâmetros")

    history: list[dict[str, float | int]] = []
    for epoch in range(1, args.epochs + 1):
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
            tonic_targets = batch["tonic_labels"].to(device, non_blocking=True)
            phasic_targets = batch["phasic_labels"].to(device, non_blocking=True)
            stage_valid = batch["staging_valid"].to(device, non_blocking=True) & padding_mask
            rswa_valid = batch["rswa_valid"].to(device, non_blocking=True) & padding_mask

            # Os ramos são independentes: cada um recebe seu próprio backward e optimizer.step().
            if stage_valid.any():
                staging_optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
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
                    device_type="cuda",
                    dtype=torch.bfloat16,
                    enabled=(not args.no_amp and device.type == "cuda"),
                ):
                    rswa_outputs = rswa_model(emg, mask=padding_mask)
                    rswa_loss = rswa_loss_fn(
                        rswa_outputs, tonic_targets, phasic_targets, rswa_valid
                    )
                rswa_loss.backward()
                clip_grad_norm_(rswa_model.parameters(), args.grad_clip)
                rswa_optimizer.step()
                rswa_loss_sum += float(rswa_loss.detach().cpu())
                rswa_batches += 1

        val_metrics = evaluate_joint(
            system,
            val_loader,
            staging_loss_fn,
            rswa_loss_fn,
            device,
            amp=not args.no_amp,
            threshold=args.threshold,
        )
        row = {
            "epoch": epoch,
            "train_staging_loss": stage_loss_sum / max(stage_batches, 1),
            "train_rswa_loss": rswa_loss_sum / max(rswa_batches, 1),
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(row)
        print(
            f"ep={epoch:03d} "
            f"train_stg_loss={row['train_staging_loss']:.4f} "
            f"train_rswa_loss={row['train_rswa_loss']:.4f} | "
            f"val_stg_f1={val_metrics.get('staging_f1_macro', float('nan')):.4f} "
            f"val_rswa_f1={val_metrics.get('rswa_rswa_f1_macro', float('nan')):.4f}"
        )

        # Salva checkpoints separados para facilitar uso e avaliação posterior.
        save_checkpoint(
            args.checkpoint_dir / "staging_last.pt",
            model=staging_model,
            optimizer=staging_optimizer,
            epoch=epoch,
            metrics=val_metrics,
            extra={"task": "staging", "trained_with": "joint"},
        )
        save_checkpoint(
            args.checkpoint_dir / "rswa_last.pt",
            model=rswa_model,
            optimizer=rswa_optimizer,
            epoch=epoch,
            metrics=val_metrics,
            extra={"task": "rswa", "trained_with": "joint"},
        )
        write_history(args.checkpoint_dir / "history.json", history)


if __name__ == "__main__":
    main()
