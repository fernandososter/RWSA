from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
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
    RSWALoss,
    StagingLoss,
    evaluate_joint,
    load_checkpoint,
    resolve_device,
    run_rswa_epoch,
    run_staging_epoch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Avalia checkpoints de staging, RSWA ou ambos.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--task", choices=("staging", "rswa", "joint"), required=True)
    parser.add_argument("--checkpoint", type=Path, help="Checkpoint para staging ou RSWA.")
    parser.add_argument("--staging-checkpoint", type=Path)
    parser.add_argument("--rswa-checkpoint", type=Path)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--all-stages", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    subjects = load_subject_directory(args.data_dir)
    dataset = SleepAnalysisDataset(
        subjects,
        min_confidence=args.min_confidence,
        rem_mask_only=not args.all_stages,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_sleep_analysis_exams,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    if args.task == "staging":
        if args.checkpoint is None:
            raise ValueError("Informe --checkpoint para task=staging.")
        model = SleepStagingNet().to(device)
        load_checkpoint(args.checkpoint, model, device)
        metrics = run_staging_epoch(model, loader, StagingLoss(), device, amp=not args.no_amp)
    elif args.task == "rswa":
        if args.checkpoint is None:
            raise ValueError("Informe --checkpoint para task=rswa.")
        model = RSWADetectionNet().to(device)
        load_checkpoint(args.checkpoint, model, device)
        metrics = run_rswa_epoch(
            model, loader, RSWALoss(), device,
            amp=not args.no_amp, threshold=args.threshold,
        )
    else:
        if args.staging_checkpoint is None or args.rswa_checkpoint is None:
            raise ValueError("Informe --staging-checkpoint e --rswa-checkpoint para task=joint.")
        staging_model = SleepStagingNet().to(device)
        rswa_model = RSWADetectionNet().to(device)
        load_checkpoint(args.staging_checkpoint, staging_model, device)
        load_checkpoint(args.rswa_checkpoint, rswa_model, device)
        model = SleepStagingRSWASystem(staging_model, rswa_model).to(device)
        metrics = evaluate_joint(
            model, loader, StagingLoss(), RSWALoss(), device,
            amp=not args.no_amp, threshold=args.threshold,
        )

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
