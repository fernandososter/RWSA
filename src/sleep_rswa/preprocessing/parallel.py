"""
Pre-processamento em lote paralelo (ProcessPoolExecutor).

Convertido do notebook Parser_Exames (celulas 20-21).
"""
from __future__ import annotations

import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List

from .channels import list_raw_edfs
from .preprocess import _save_result, preprocess_exam


def _preprocess_one_worker(args):
    edf_path, out_dir, overwrite, verbose_worker, kwargs = args
    edf_path = Path(edf_path)
    out_dir = Path(out_dir)
    sid = edf_path.stem
    out_path = out_dir / f"{sid}.pt"

    if out_path.exists() and not overwrite:
        return {"sid": sid, "status": "skipped", "T": None, "REM": None, "error": None}

    try:
        result = preprocess_exam(edf_path, verbose=verbose_worker, **kwargs)
        if result is None:
            return {"sid": sid, "status": "failed", "T": None, "REM": None,
                    "error": "preprocess_exam retornou None"}

        _save_result(result, out_path)
        T = result["signals"].shape[0]
        rem = int((result["sleep_stages"] == 4).sum())
        return {"sid": sid, "status": "ok", "T": T, "REM": rem, "error": None}
    except Exception:
        return {"sid": sid, "status": "failed", "T": None, "REM": None,
                "error": traceback.format_exc()}


def run_preprocessing_parallel(
    edf_dir: Path,
    out_dir: Path = None,
    overwrite: bool = False,
    verbose: bool = True,
    max_workers: int = 4,
    **kwargs,
) -> List[str]:
    """
    Pre-processa todos os EDFs brutos em paralelo. kwargs extras sao repassados
    a preprocess_exam (mat_dir, rswa_dir, tonic_min_coverage, phasic_min_coverage).
    """
    edf_dir = Path(edf_dir)
    out_dir = Path(out_dir) if out_dir is not None else edf_dir.parent / "tensors"
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_edfs = list_raw_edfs(edf_dir)
    if not raw_edfs:
        print("[preprocess] Nenhum EDF bruto encontrado. Verifique o diretorio.")
        return []

    print(f"[preprocess] {len(raw_edfs)} EDFs | paralelo max_workers={max_workers}")
    tasks = [(edf_path, out_dir, overwrite, False, kwargs) for edf_path in raw_edfs]

    processed, skipped, failed = [], [], []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_preprocess_one_worker, t): t[0] for t in tasks}
        for i, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            sid, status = result["sid"], result["status"]
            if status == "ok":
                processed.append(sid)
                if verbose:
                    print(f"[{i}/{len(raw_edfs)}] OK      {sid} | "
                          f"T={result['T']} | REM={result['REM']}")
            elif status == "skipped":
                processed.append(sid)
                skipped.append(sid)
                if verbose:
                    print(f"[{i}/{len(raw_edfs)}] SKIP    {sid}")
            else:
                failed.append((sid, result["error"]))
                if verbose:
                    print(f"[{i}/{len(raw_edfs)}] FALHOU  {sid}")
                    if result["error"]:
                        print(result["error"].splitlines()[-1])

    print(f"\n[preprocess] Concluido: {len(processed)} OK/SKIP | "
          f"{len(skipped)} pulados | {len(failed)} falharam | {len(raw_edfs)} total")
    if failed:
        print("Falharam:")
        for sid, err in failed[:10]:
            print(f"  - {sid}: {err.splitlines()[-1] if err else 'erro desconhecido'}")
    return processed
