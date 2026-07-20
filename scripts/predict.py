from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
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
from sleep_rswa.training import load_checkpoint, resolve_device

STAGE_NAMES = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gera predições sincronizadas por miniépoca de 3 segundos.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--staging-checkpoint", type=Path, required=True)
    parser.add_argument("--rswa-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("predictions.csv"))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    dataset = SleepAnalysisDataset(load_subject_directory(args.data_dir))
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_sleep_analysis_exams,
        pin_memory=device.type == "cuda", persistent_workers=args.num_workers > 0,
    )

    staging_model = SleepStagingNet().to(device)
    rswa_model = RSWADetectionNet().to(device)
    load_checkpoint(args.staging_checkpoint, staging_model, device)
    load_checkpoint(args.rswa_checkpoint, rswa_model, device)
    model = SleepStagingRSWASystem(staging_model, rswa_model).eval().to(device)

    rows: list[dict[str, object]] = []
    autocast_enabled = not args.no_amp and device.type == "cuda"
    with torch.no_grad():
        for batch in loader:
            signals = batch["signals"].to(device)
            emg = batch["emg_center"].to(device)
            mask = batch["padding_mask"].to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
                outputs = model(signals, emg, mask=mask)
            stage_prob = torch.softmax(outputs["staging_logits"], dim=-1)
            stage_pred = stage_prob.argmax(dim=-1)
            tonic_prob = torch.sigmoid(outputs["tonic_logits"])
            phasic_prob = torch.sigmoid(outputs["phasic_logits"])

            for i, subject_id in enumerate(batch["subject_ids"]):
                length = int(batch["lengths"][i])
                for epoch in range(length):
                    stage_id = int(stage_pred[i, epoch].cpu())
                    rows.append({
                        "subject_id": subject_id,
                        "mini_epoch": epoch,
                        "start_sec": epoch * 3,
                        "stage_id": stage_id,
                        "stage": STAGE_NAMES[stage_id],
                        "stage_confidence": float(stage_prob[i, epoch, stage_id].cpu()),
                        "tonic_probability": float(tonic_prob[i, epoch].cpu()),
                        "tonic_pred": int(tonic_prob[i, epoch] >= args.threshold),
                        "phasic_probability": float(phasic_prob[i, epoch].cpu()),
                        "phasic_pred": int(phasic_prob[i, epoch] >= args.threshold),
                    })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"Predições salvas em {args.output} ({len(rows):,} miniépocas).")


if __name__ == "__main__":
    main()
