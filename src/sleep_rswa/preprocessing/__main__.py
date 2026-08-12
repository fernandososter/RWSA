from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from . import PathConfig, run_preprocessing, run_preprocessing_parallel
from .config import ECG_GATE_DEFAULT, ECG_GATE_WINDOW_S


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pre-processa EDF + hipnograma + anotacoes RSWA em tensores .pt.",
    )
    parser.add_argument("--edf-dir", type=Path, default=PathConfig.EDF_DIR)
    parser.add_argument("--out-dir", type=Path, default=PathConfig.TENSOR_DIR)
    parser.add_argument("--mat-dir", type=Path, default=PathConfig.MAT_DIR)
    parser.add_argument("--rswa-dir", type=Path, default=PathConfig.RSWA_DIR)
    parser.add_argument("--rswa-source", choices=("auto", "csv", "aasm"), default="auto")
    parser.add_argument("--auto-label-model-path", type=Path, default=None)
    parser.add_argument("--auto-label-device", default="cpu")
    parser.add_argument("--auto-label-cnn-threshold", type=float, default=None)
    parser.add_argument("--auto-label-cnn-min-epochs", type=int, default=1)
    parser.add_argument("--auto-label-k-on", type=float, default=3.0)
    parser.add_argument("--auto-label-k-off", type=float, default=1.5)
    parser.add_argument("--auto-label-k-off-hold-s", type=float, default=0.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--tonic-min-coverage", type=float, default=0.5)
    parser.add_argument("--phasic-min-coverage", type=float, default=0.0)
    parser.add_argument("--any-min-coverage", type=float, default=0.0)
    parser.add_argument(
        "--aasm-atonia-pct", type=float, default=None,
        help=(
            "Percentil do envelope RMS de EMG em REM usado como nivel de "
            "atonia p/ rswa_source=aasm (default: usa rem_baseline_uv, "
            "percentil 10 -- ver AVISO DE CALIBRACAO em aasm_rule.py antes "
            "de usar em producao)."
        ),
    )
    parser.add_argument(
        "--no-ecg-gate", dest="ecg_gate", action="store_false",
        default=ECG_GATE_DEFAULT,
        help=(
            "Desativa o gating de artefato cardiaco no EMG via deteccao de "
            "picos-R no canal de ECG (default: ativado -- ver ecg_gating.py "
            "e docs/relatorio_impacto_regra_aasm.md Secao 12)."
        ),
    )
    parser.add_argument(
        "--ecg-gate-window-s", type=float, default=ECG_GATE_WINDOW_S,
        help="Semi-largura (segundos) da janela de gating em torno de cada pico-R.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    kwargs = {
        "mat_dir": args.mat_dir,
        "rswa_dir": args.rswa_dir,
        "rswa_source": args.rswa_source,
        "auto_label_model_path": args.auto_label_model_path,
        "auto_label_device": args.auto_label_device,
        "auto_label_cnn_threshold": args.auto_label_cnn_threshold,
        "auto_label_cnn_min_epochs": args.auto_label_cnn_min_epochs,
        "auto_label_k_on": args.auto_label_k_on,
        "auto_label_k_off": args.auto_label_k_off,
        "auto_label_k_off_hold_s": args.auto_label_k_off_hold_s,
        "tonic_min_coverage": args.tonic_min_coverage,
        "phasic_min_coverage": args.phasic_min_coverage,
        "any_min_coverage": args.any_min_coverage,
        "aasm_atonia_pct": args.aasm_atonia_pct,
        "ecg_gate": args.ecg_gate,
        "ecg_gate_window_s": args.ecg_gate_window_s,
    }

    if args.parallel:
        run_preprocessing_parallel(
            edf_dir=args.edf_dir,
            out_dir=args.out_dir,
            overwrite=args.overwrite,
            verbose=not args.quiet,
            max_workers=args.max_workers,
            **kwargs,
        )
    else:
        run_preprocessing(
            edf_dir=args.edf_dir,
            out_dir=args.out_dir,
            overwrite=args.overwrite,
            verbose=not args.quiet,
            **kwargs,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
