"""
Backfill de rem_baseline_uv / rem_baseline_n_epochs nos .pt existentes.

NAO reprocessa nenhum EDF: os .pt ja tem `signals` (bruto) e `sleep_stages`,
suficiente para o calculo (ver src/sleep_rswa/preprocessing/rem_baseline.py).
Le cada .pt, calcula o basal, regrava SOMENTE se o campo ainda nao existir
(idempotente -- rodar de novo nao muda nada em arquivos ja atualizados).

Demais chaves do .pt (signals, sleep_stages, tonic_labels, etc.) sao
preservadas exatamente como estavam (torch.equal apos reload, verificado por
--verify) -- este script NAO altera rotulos nem sinal, so adiciona 2 campos.

Uso:
    python scripts/backfill_rem_baseline.py [--dir DIR] [--dry-run] [--verify]

Por padrao processa classifier/data/*.pt (diretorio canonico, 60 exames) e,
em seguida, testes/data_real/*.pt (copia usada pelos testes deterministicos,
subconjunto de 41 dos mesmos exames) -- ambos precisam do campo porque cada
um e um arquivo fisico independente no disco (nao um link).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sleep_rswa.preprocessing.rem_baseline import compute_rem_baseline  # noqa: E402

DEFAULT_DIRS = [
    PROJECT_ROOT / "classifier" / "data",
    PROJECT_ROOT / "testes" / "data_real",
]


def backfill_dir(data_dir: Path, dry_run: bool = False, verify: bool = False) -> list[dict]:
    rows = []
    pt_files = sorted(data_dir.glob("*.pt"))
    for pt_path in pt_files:
        obj = torch.load(pt_path, map_location="cpu", weights_only=False)

        already_has = "rem_baseline_uv" in obj
        if already_has:
            rows.append({
                "file": pt_path.name, "dir": str(data_dir), "status": "skip_existing",
                "rem_baseline_uv": float(obj["rem_baseline_uv"]),
                "rem_baseline_n_epochs": int(obj["rem_baseline_n_epochs"]),
            })
            continue

        signals = obj["signals"].numpy()
        stages = obj["sleep_stages"].numpy()
        result = compute_rem_baseline(signals, stages)

        if verify:
            before_keys = {k: v for k, v in obj.items()}

        obj["rem_baseline_uv"] = result["rem_baseline_uv"]
        obj["rem_baseline_n_epochs"] = result["rem_baseline_n_epochs"]

        if not dry_run:
            torch.save(obj, pt_path)
            if verify:
                reloaded = torch.load(pt_path, map_location="cpu", weights_only=False)
                for k, v in before_keys.items():
                    rv = reloaded[k]
                    if torch.is_tensor(v):
                        assert torch.equal(v, rv), f"{pt_path.name}: campo {k} mudou!"
                    else:
                        assert v == rv, f"{pt_path.name}: campo {k} mudou!"

        rows.append({
            "file": pt_path.name, "dir": str(data_dir),
            "status": "dry_run" if dry_run else "updated",
            "rem_baseline_uv": result["rem_baseline_uv"],
            "rem_baseline_n_epochs": result["rem_baseline_n_epochs"],
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=None,
                     help="Se omitido, processa classifier/data e testes/data_real")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verify", action="store_true",
                     help="Recarrega apos salvar e confere que os demais campos nao mudaram")
    ap.add_argument("--out-csv", type=Path,
                     default=PROJECT_ROOT / "out_rem_baseline" / "rem_baseline_backfill.csv")
    args = ap.parse_args()

    dirs = [args.dir] if args.dir is not None else DEFAULT_DIRS
    all_rows = []
    for d in dirs:
        if not d.exists():
            print(f"[SKIP] {d} nao existe")
            continue
        rows = backfill_dir(d, dry_run=args.dry_run, verify=args.verify)
        all_rows.extend(rows)
        n_updated = sum(1 for r in rows if r["status"] == "updated")
        n_skipped = sum(1 for r in rows if r["status"] == "skip_existing")
        n_dry = sum(1 for r in rows if r["status"] == "dry_run")
        print(f"[{d}] {len(rows)} arquivos | updated={n_updated} "
              f"skip_existing={n_skipped} dry_run={n_dry}")

    if all_rows:
        import csv
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.out_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)
        print(f"\nRelatorio: {args.out_csv}")


if __name__ == "__main__":
    main()
