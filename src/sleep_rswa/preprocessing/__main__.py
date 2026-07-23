from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from . import PathConfig, run_preprocessing, run_preprocessing_parallel


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pre-processa EDF + hipnograma + anotacoes RSWA em tensores .pt.",
    )
    parser.add_argument("--edf-dir", type=Path, default=PathConfig.EDF_DIR)
    parser.add_argument("--out-dir", type=Path, default=PathConfig.TENSOR_DIR)
    parser.add_argument("--mat-dir", type=Path, default=PathConfig.MAT_DIR)
    parser.add_argument("--rswa-dir", type=Path, default=PathConfig.RSWA_DIR)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--tonic-min-coverage", type=float, default=0.5)
    parser.add_argument("--phasic-min-coverage", type=float, default=0.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    kwargs = {
        "mat_dir": args.mat_dir,
        "rswa_dir": args.rswa_dir,
        "tonic_min_coverage": args.tonic_min_coverage,
        "phasic_min_coverage": args.phasic_min_coverage,
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
