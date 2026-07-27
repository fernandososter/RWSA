"""
Calcula annot_start (offset de tempo, em segundos do EDF) de cada exame e grava
view/offsets.json  ->  {exam: annot_start}.

annot_start e o instante do EDF onde comeca a primeira epoca estagiada; e o mesmo
valor usado no crop do preprocessamento. A mini-epoca m do .pt corresponde a
    tempo_edf(m) = annot_start + m * 3s
Portanto o CSV revisado sai no tempo do EDF:
    onset_edf = annot_start + onset_pt

Precisa do hipnograma <exam>.mat (start_time + hyp) e do meas_date do EDF.
O meas_date pode vir do proprio .edf (lido so o cabecalho) OU ser informado
diretamente como HH:MM:SS (evita transferir o EDF inteiro).

Uso:
    # a partir do EDF (le so cabecalho):
    python view/compute_offsets.py --mat rbd1.mat --edf rbd1.edf --name rbd1
    # ou informando o inicio do EDF:
    python view/compute_offsets.py --mat rbd1.mat --meas-date 22:47:30 --name rbd1
    # varios de uma vez (pasta com <name>.mat e <name>.edf):
    python view/compute_offsets.py --dir /caminho/exames
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat

HERE = Path(__file__).resolve().parent
OFFSETS = HERE / "offsets.json"

# estagios validos (mesma convencao do pipeline; 7=Movement e descartado)
_VALID_STAGE_CODES = {0, 1, 2, 3, 4, 5}


def _edf_meas_date_seconds(edf_path: Path) -> float:
    """Segundos-do-dia do inicio do EDF (le so o cabecalho via mne)."""
    import mne
    raw = mne.io.read_raw_edf(str(edf_path), preload=False, verbose="ERROR")
    md = raw.info["meas_date"]
    if md is None:
        raise ValueError(f"{edf_path}: EDF sem meas_date no cabecalho.")
    return md.hour * 3600 + md.minute * 60 + md.second + md.microsecond / 1e6


def _hhmmss_to_seconds(s: str) -> float:
    parts = [float(x) for x in s.strip().split(":")]
    while len(parts) < 3:
        parts.insert(0, 0.0)
    h, m, sec = parts[-3], parts[-2], parts[-1]
    return h * 3600 + m * 60 + sec


def compute_annot_start(mat_path: Path, edf_meas_seconds: float) -> float:
    data = loadmat(mat_path, simplify_cells=True)
    if "hyp" not in data or "start_time" not in data:
        raise KeyError(f"{mat_path}: faltam 'hyp' e/ou 'start_time'.")
    hyp = np.asarray(data["hyp"], dtype=float)
    st = data["start_time"]
    stages = hyp[:, 0].astype(int)
    rel = hyp[:, 1].astype(float)

    annot_start_sec = float(st["h"]) * 3600 + float(st["m"]) * 60 + float(st["s"])
    offset = annot_start_sec - edf_meas_seconds
    if offset < -12 * 3600:
        offset += 24 * 3600
    elif offset > 12 * 3600:
        offset -= 24 * 3600

    aligned = rel + offset
    keep = np.isin(stages, list(_VALID_STAGE_CODES)) & (stages != 7)
    if not keep.any():
        raise ValueError(f"{mat_path}: nenhum estagio valido no hipnograma.")
    return float(aligned[keep].min())


def _load_offsets() -> dict:
    if OFFSETS.exists():
        return json.loads(OFFSETS.read_text())
    return {}


def _save_offsets(d: dict):
    OFFSETS.write_text(json.dumps(d, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mat")
    ap.add_argument("--edf")
    ap.add_argument("--meas-date", help="inicio do EDF, HH:MM:SS (alternativa ao --edf)")
    ap.add_argument("--name", help="nome do exame (default: stem do .mat)")
    ap.add_argument("--dir", help="pasta com <name>.mat e <name>.edf para processar em lote")
    args = ap.parse_args()

    offs = _load_offsets()

    if args.dir:
        d = Path(args.dir)
        mats = sorted(d.glob("*.mat"))
        for mp in mats:
            name = mp.stem
            edf = d / f"{name}.edf"
            if not edf.exists():
                print(f"[SKIP] {name}: sem {edf.name}")
                continue
            try:
                a = compute_annot_start(mp, _edf_meas_date_seconds(edf))
                offs[name] = round(a, 3)
                print(f"[OK] {name}: annot_start = {a:.3f}s ({a/3600:.3f}h)")
            except Exception as e:
                print(f"[ERRO] {name}: {e}")
        _save_offsets(offs)
        print(f"-> {OFFSETS}")
        return

    if not args.mat:
        ap.error("informe --mat (+ --edf ou --meas-date) ou --dir")
    mat = Path(args.mat)
    name = args.name or mat.stem
    if args.edf:
        meas = _edf_meas_date_seconds(Path(args.edf))
    elif args.meas_date:
        meas = _hhmmss_to_seconds(args.meas_date)
    else:
        ap.error("informe --edf OU --meas-date")
    a = compute_annot_start(mat, meas)
    offs[name] = round(a, 3)
    _save_offsets(offs)
    print(f"[OK] {name}: annot_start = {a:.3f}s ({a/3600:.3f}h)  -> {OFFSETS}")


if __name__ == "__main__":
    main()
