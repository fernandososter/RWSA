"""
Aplica rotulos revisados (CSV) nos tensores .pt de treino.

Le classifier/labels/<exame>_revisado.csv (mesmo formato gravado pelo /api/save
do view/: subject_id, onset_s, duration_s, type, score — so linhas type in
{tonic, phasic}, ja que o app so exporta eventos confirmados) e escreve
tonic_labels/phasic_labels no .pt correspondente em classifier/data/<exame>.pt.

Fonte dos CSVs: classifier/labels/ (copias que voce arquiva como "originais").
Este script NAO le nada de view/revisado/ — so consulta classifier/labels/exam_config.json
para obter o annot_start (metadado de offset de tempo, nao os CSVs revisados).
Essa copia deve ser atualizada (copiada de view/exam_config.json) pelo usuario
a cada retreino, junto com os CSVs revisados — assim classifier/ fica
autocontido, sem ler nada fora de classifier/labels/ e classifier/data/.

Por que o offset importa: o onset_s de um CSV pode estar em tempo do EDF
(quando o offset annot_start ja era conhecido no momento do save) ou em tempo
do .pt (quando nao era — a app grava um aviso "time_ref: pt" nesse caso). Este
script reconverte para tempo do .pt fazendo pt_onset = onset_s - annot_start,
usando o annot_start de cada exame em classifier/labels/exam_config.json.

USO (nao e executado automaticamente por nada no projeto):
    python classifier/apply_labels.py                  # aplica todos os CSVs em classifier/labels/
    python classifier/apply_labels.py --exam rbd6       # so um exame (repita --exam p/ varios)
    python classifier/apply_labels.py --dry-run         # so mostra o resumo, nao grava nada
    python classifier/apply_labels.py --time-ref pt     # forca tratar onset_s ja como tempo do .pt

Seguranca: antes de sobrescrever um .pt, faz backup do original (uma unica vez,
nao sobrescreve backup existente) em classifier/data_backup/<exame>.pt.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from classifier.movement_clf.dataio import EPOCH_SEC

DATA_DIR = HERE / "data"
LABELS_DIR = HERE / "labels"
BACKUP_DIR = HERE / "data_backup"
# Copia de view/exam_config.json que o usuario mantem atualizada dentro de
# classifier/labels/ a cada retreino — mantem classifier/ autocontido, sem
# ler nada de fora de classifier/labels/ e classifier/data/.
DEFAULT_CONFIG = LABELS_DIR / "exam_config.json"


def load_offsets(config_path: Path) -> dict:
    """{exame: annot_start} a partir do exam_config.json da app de revisao.
    So le o offset (numero); nao toca em nenhum CSV de view/revisado/."""
    if not config_path.exists():
        return {}
    cfg = json.load(open(config_path))
    return {k: v.get("annot_start") for k, v in cfg.items()}


def read_label_csv(path: Path) -> list[dict]:
    rows = []
    with open(path, newline="") as f:
        for d in csv.DictReader(f):
            if d["type"] not in ("tonic", "phasic"):
                continue
            rows.append({
                "onset_s": float(d["onset_s"]),
                "duration_s": float(d["duration_s"]),
                "type": d["type"],
            })
    return rows


def apply_one(exam: str, offsets: dict, time_ref: str, dry_run: bool) -> dict:
    pt_path = DATA_DIR / f"{exam}.pt"
    csv_path = LABELS_DIR / f"{exam}_revisado.csv"
    if not pt_path.exists():
        raise FileNotFoundError(f"{pt_path} nao encontrado")
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} nao encontrado")

    obj = torch.load(pt_path, map_location="cpu", weights_only=False)
    T = obj["signals"].shape[0]

    if time_ref == "pt":
        a0 = 0.0
    elif time_ref == "edf":
        if offsets.get(exam) is None:
            raise ValueError(f"{exam}: sem annot_start em {DEFAULT_CONFIG.name} e --time-ref edf foi forcado")
        a0 = float(offsets[exam])
    else:  # auto
        a0 = float(offsets[exam]) if offsets.get(exam) is not None else 0.0
        if offsets.get(exam) is None:
            print(f"  aviso: {exam} sem annot_start conhecido -> assumindo onset_s ja em tempo do .pt")

    rows = read_label_csv(csv_path)
    tonic = np.zeros(T, dtype=np.float32)
    phasic = np.zeros(T, dtype=np.float32)
    n_warn = 0
    for r in rows:
        pt_onset = r["onset_s"] - a0
        i0f = pt_onset / EPOCH_SEC
        i1f = (pt_onset + r["duration_s"]) / EPOCH_SEC
        i0, i1 = round(i0f), round(i1f)
        if abs(i0f - i0) > 1e-3 or abs(i1f - i1) > 1e-3:
            n_warn += 1  # onset/duration nao caem em fronteira de mini-epoca (3s) apos o offset
        i0c, i1c = max(0, i0), min(T, i1)
        if i0c >= i1c:
            n_warn += 1  # evento cai totalmente fora do exame (offset errado?)
            continue
        arr = tonic if r["type"] == "tonic" else phasic
        arr[i0c:i1c] = 1.0

    movement = (tonic > 0.5) | (phasic > 0.5)
    summary = {
        "exam": exam, "T": T, "n_events": len(rows),
        "n_tonic_epochs": int(tonic.sum()), "n_phasic_epochs": int(phasic.sum()),
        "n_movement_epochs": int(movement.sum()),
        "prevalence_pct": round(100 * float(movement.mean()), 2),
        "annot_start_used": a0, "n_alignment_warnings": n_warn,
    }

    if not dry_run:
        BACKUP_DIR.mkdir(exist_ok=True)
        backup_path = BACKUP_DIR / f"{exam}.pt"
        if not backup_path.exists():
            shutil.copy2(pt_path, backup_path)
        obj["tonic_labels"] = torch.from_numpy(tonic)
        obj["phasic_labels"] = torch.from_numpy(phasic)
        torch.save(obj, pt_path)

    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exam", action="append",
                     help="processar so este(s) exame(s) (repita p/ varios); default: todos os CSVs em classifier/labels/")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                     help="json com annot_start por exame (default: classifier/labels/exam_config.json)")
    ap.add_argument("--time-ref", choices=["auto", "edf", "pt"], default="auto",
                     help="auto (default): usa annot_start se conhecido, senao assume onset_s ja em tempo do .pt. "
                          "edf: exige annot_start p/ todos os exames processados. "
                          "pt: ignora offsets, onset_s ja e tratado como tempo do .pt.")
    ap.add_argument("--dry-run", action="store_true", help="so mostra o resumo, nao grava nada")
    args = ap.parse_args()

    offsets = load_offsets(args.config)
    exams = args.exam or sorted(p.stem.replace("_revisado", "") for p in LABELS_DIR.glob("*_revisado.csv"))
    if not exams:
        print(f"nenhum CSV encontrado em {LABELS_DIR}")
        return

    print(f"{'[DRY RUN] ' if args.dry_run else ''}aplicando rotulos em {len(exams)} exame(s): {exams}\n")
    for exam in exams:
        try:
            s = apply_one(exam, offsets, args.time_ref, args.dry_run)
            tag = "" if s["n_alignment_warnings"] == 0 else f"  <-- {s['n_alignment_warnings']} evento(s) desalinhado(s), confira o offset"
            print(f"  {exam}: {s['n_events']} eventos -> {s['n_movement_epochs']}/{s['T']} mini-epocas "
                  f"({s['prevalence_pct']}%) | annot_start={s['annot_start_used']}{tag}")
        except Exception as e:
            print(f"  {exam}: ERRO - {e}")

    if not args.dry_run:
        print(f"\nbackup dos .pt originais (1a vez) em: {BACKUP_DIR}")
    print("\ndry-run concluido, nada foi gravado." if args.dry_run else "\npronto.")


if __name__ == "__main__":
    main()
