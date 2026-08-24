"""
Analisa um diretório de sujeitos (.pt) e resume o dataset em nível global e
por sujeito.

Saídas principais:
  - resumo global no terminal
  - CSV opcional com uma linha por sujeito
  - JSON opcional com o agregado e as linhas por sujeito

Exemplo:
    python scripts/analise_dataset.py \
        --data-dir view/validacao \
        --out-csv runs/dataset_summary.csv \
        --out-json runs/dataset_summary.json
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
SRC = PROJ / "src"
for candidate in (PROJ, SRC):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from sleep_rswa.config import RSWAConfig, SignalConfig  # noqa: E402
from sleep_rswa.constants import SLEEP_STAGE_NAMES  # noqa: E402
from sleep_rswa.data import SubjectData, load_subject_directory  # noqa: E402
from sleep_rswa.distribution import StageDistribution  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="Diretório com arquivos .pt.")
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help="Mesma regra de validade RSWA usada no treino.",
    )
    parser.add_argument(
        "--all-stages",
        action="store_true",
        help="Se ativo, a validade RSWA não é restrita ao REM.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Número de threads no carregamento dos .pt.",
    )
    parser.add_argument("--out-csv", type=Path, default=None, help="CSV de saída por sujeito.")
    parser.add_argument("--out-json", type=Path, default=None, help="JSON com resumo global e linhas por sujeito.")
    return parser.parse_args()


def _round_or_none(value: float | None, digits: int = 4) -> float | None:
    if value is None:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return round(float(value), digits)


def _get_head_labels(subject: SubjectData, rswa_cfg: RSWAConfig) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rswa_labels = subject.rswa_labels.long()
    if subject.tonic_labels is not None and subject.phasic_labels is not None:
        tonic = subject.tonic_labels.float().clone()
        phasic = subject.phasic_labels.float().clone()
    else:
        tonic = rswa_labels.eq(rswa_cfg.tonic_label).float()
        phasic = rswa_labels.eq(rswa_cfg.phasic_label).float()

    if subject.any_labels is not None:
        any_labels = subject.any_labels.float().clone()
    else:
        any_labels = rswa_labels.eq(rswa_cfg.any_label).float()
    return tonic, phasic, any_labels


def summarize_subject(
    subject: SubjectData,
    *,
    min_confidence: float,
    rem_mask_only: bool,
    rswa_cfg: RSWAConfig,
    sig_cfg: SignalConfig,
) -> dict[str, Any]:
    stages = subject.sleep_stages.long()
    confidence = subject.rswa_conf.float()
    valid = confidence > float(min_confidence)
    if rem_mask_only:
        valid = valid & stages.eq(rswa_cfg.rem_stage)

    tonic, phasic, any_labels = _get_head_labels(subject, rswa_cfg)
    movement = (tonic > 0.5) | (phasic > 0.5) | (any_labels > 0.5)

    stage_dist = StageDistribution()
    stage_dist.update(stages)
    stage_dict = stage_dist.as_dict()

    total = int(stages.numel())
    gaps = int(stages.eq(-1).sum().item())
    rem_total = int(stages.eq(rswa_cfg.rem_stage).sum().item())
    valid_total = int(valid.sum().item())

    tonic_pos = int((tonic[valid] > 0.5).sum().item())
    phasic_pos = int((phasic[valid] > 0.5).sum().item())
    any_pos = int((any_labels[valid] > 0.5).sum().item())
    movement_pos = int(movement[valid].sum().item())

    total_seconds = total * sig_cfg.epoch_sec
    valid_seconds = valid_total * sig_cfg.epoch_sec
    rem_seconds = rem_total * sig_cfg.epoch_sec

    row: dict[str, Any] = {
        "subject_id": subject.subject_id,
        "total_mini_epochs": total,
        "gap_mini_epochs": gaps,
        "rem_mini_epochs": rem_total,
        "valid_rswa_mini_epochs": valid_total,
        "total_hours": total_seconds / 3600.0,
        "rem_hours": rem_seconds / 3600.0,
        "valid_rswa_hours": valid_seconds / 3600.0,
        "tonic_positive": tonic_pos,
        "phasic_positive": phasic_pos,
        "any_positive": any_pos,
        "movement_positive": movement_pos,
        "pct_valid_of_total": (100.0 * valid_total / total) if total else 0.0,
        "pct_tonic_of_valid": (100.0 * tonic_pos / valid_total) if valid_total else 0.0,
        "pct_phasic_of_valid": (100.0 * phasic_pos / valid_total) if valid_total else 0.0,
        "pct_any_of_valid": (100.0 * any_pos / valid_total) if valid_total else 0.0,
        "pct_movement_of_valid": (100.0 * movement_pos / valid_total) if valid_total else 0.0,
        "rem_baseline_uv": _round_or_none(subject.rem_baseline_uv),
        "atonia_baseline_uv": _round_or_none(subject.atonia_baseline_uv),
        "baseline_relative_reference_ratio": _round_or_none(subject.baseline_relative_reference_ratio),
    }

    for stage_name, stats in stage_dict.items():
        row[f"stage_{stage_name}_count"] = int(stats["count"])
        row[f"stage_{stage_name}_pct"] = float(stats["percentage"])

    return row


def aggregate_rows(rows: list[dict[str, Any]], *, sig_cfg: SignalConfig) -> dict[str, Any]:
    totals: dict[str, Any] = {
        "subjects": len(rows),
        "total_mini_epochs": sum(int(r["total_mini_epochs"]) for r in rows),
        "gap_mini_epochs": sum(int(r["gap_mini_epochs"]) for r in rows),
        "rem_mini_epochs": sum(int(r["rem_mini_epochs"]) for r in rows),
        "valid_rswa_mini_epochs": sum(int(r["valid_rswa_mini_epochs"]) for r in rows),
        "tonic_positive": sum(int(r["tonic_positive"]) for r in rows),
        "phasic_positive": sum(int(r["phasic_positive"]) for r in rows),
        "any_positive": sum(int(r["any_positive"]) for r in rows),
        "movement_positive": sum(int(r["movement_positive"]) for r in rows),
    }
    totals["total_hours"] = totals["total_mini_epochs"] * sig_cfg.epoch_sec / 3600.0
    totals["rem_hours"] = totals["rem_mini_epochs"] * sig_cfg.epoch_sec / 3600.0
    totals["valid_rswa_hours"] = totals["valid_rswa_mini_epochs"] * sig_cfg.epoch_sec / 3600.0
    totals["pct_valid_of_total"] = (
        100.0 * totals["valid_rswa_mini_epochs"] / totals["total_mini_epochs"]
        if totals["total_mini_epochs"]
        else 0.0
    )
    for key in ("tonic", "phasic", "any", "movement"):
        positives = totals[f"{key}_positive"]
        valid = totals["valid_rswa_mini_epochs"]
        totals[f"pct_{key}_of_valid"] = (100.0 * positives / valid) if valid else 0.0

    stage_summary: dict[str, dict[str, float | int]] = {}
    stage_total = sum(int(r[f"stage_{name}_count"]) for r in rows for name in SLEEP_STAGE_NAMES.values())
    # stage_total acima soma repetidamente; o total correto é a soma de qualquer classe uma vez por linha
    stage_total = sum(int(r[f"stage_W_count"]) + int(r[f"stage_N1_count"]) + int(r[f"stage_N2_count"])
                      + int(r[f"stage_N3_count"]) + int(r[f"stage_REM_count"]) for r in rows)
    for name in SLEEP_STAGE_NAMES.values():
        count = sum(int(r[f"stage_{name}_count"]) for r in rows)
        pct = (100.0 * count / stage_total) if stage_total else 0.0
        stage_summary[name] = {"count": count, "percentage": pct}
    totals["stage_distribution"] = stage_summary

    for field in ("rem_baseline_uv", "atonia_baseline_uv", "baseline_relative_reference_ratio"):
        values = [float(r[field]) for r in rows if r[field] is not None]
        if values:
            values_t = torch.tensor(values, dtype=torch.float64)
            totals[f"{field}_mean"] = round(float(values_t.mean().item()), 4)
            totals[f"{field}_median"] = round(float(values_t.median().item()), 4)
            totals[f"{field}_count"] = len(values)
        else:
            totals[f"{field}_mean"] = None
            totals[f"{field}_median"] = None
            totals[f"{field}_count"] = 0

    return totals


def print_summary(summary: dict[str, Any]) -> None:
    print("=" * 80)
    print("Resumo global do dataset")
    print("=" * 80)
    print(f"Sujeitos               : {summary['subjects']}")
    print(f"Mini-épocas totais     : {summary['total_mini_epochs']:,}")
    print(f"Tempo total (h)        : {summary['total_hours']:.2f}")
    print(f"Mini-épocas REM        : {summary['rem_mini_epochs']:,} ({summary['rem_hours']:.2f} h)")
    print(
        f"RSWA avaliáveis        : {summary['valid_rswa_mini_epochs']:,} "
        f"({summary['valid_rswa_hours']:.2f} h; {summary['pct_valid_of_total']:.2f}% do total)"
    )
    print(f"Gaps/estágio -1        : {summary['gap_mini_epochs']:,}")
    print()
    print("Estágios do sono")
    for stage_name, stats in summary["stage_distribution"].items():
        print(f"  {stage_name:4s}: {stats['count']:,} ({stats['percentage']:.2f}%)")
    print()
    print("Eventos RSWA (na máscara de validade)")
    for key in ("tonic", "phasic", "any", "movement"):
        print(
            f"  {key:8s}: {summary[f'{key}_positive']:,} "
            f"({summary[f'pct_{key}_of_valid']:.2f}% das avaliáveis)"
        )
    print()
    print("Baselines")
    for key, label in (
        ("rem_baseline_uv", "REM baseline (uV)"),
        ("atonia_baseline_uv", "Atonia baseline (uV)"),
        ("baseline_relative_reference_ratio", "Reference ratio"),
    ):
        count = summary[f"{key}_count"]
        if count:
            print(
                f"  {label:22s}: n={count} | "
                f"media={summary[f'{key}_mean']} | mediana={summary[f'{key}_median']}"
            )
        else:
            print(f"  {label:22s}: indisponível")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, summary: dict[str, Any], rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "arguments": {
            "data_dir": str(args.data_dir),
            "min_confidence": float(args.min_confidence),
            "all_stages": bool(args.all_stages),
            "max_workers": args.max_workers,
        },
        "summary": summary,
        "subjects": rows,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    sig_cfg = SignalConfig()
    rswa_cfg = RSWAConfig()
    rem_mask_only = not args.all_stages

    subjects = load_subject_directory(args.data_dir, max_workers=args.max_workers)
    rows = [
        summarize_subject(
            subject,
            min_confidence=args.min_confidence,
            rem_mask_only=rem_mask_only,
            rswa_cfg=rswa_cfg,
            sig_cfg=sig_cfg,
        )
        for subject in subjects
    ]
    rows.sort(key=lambda row: row["subject_id"])
    summary = aggregate_rows(rows, sig_cfg=sig_cfg)

    print_summary(summary)
    if args.out_csv is not None:
        write_csv(args.out_csv, rows)
        print(f"\nCSV salvo em: {args.out_csv}")
    if args.out_json is not None:
        write_json(args.out_json, summary, rows, args)
        print(f"JSON salvo em: {args.out_json}")


if __name__ == "__main__":
    main()
