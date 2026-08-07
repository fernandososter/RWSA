"""
Amostragem de eventos rotulados automaticamente (CNN+limiar-duplo) para nova
revisao humana -- fonte de avaliacao independente do treino.

Este script NAO revisa nada sozinho: ele so SELECIONA um subconjunto de
exames/eventos ja escritos no .pt pelo pipeline automatico (classifier/
auto_label.py) e exporta em formato IDENTICO aos *_revisado.csv existentes
(subject_id, onset_s, duration_s, type, score), para o usuario editar
diretamente (apagar falsos positivos, corrigir onset/duration/type, adicionar
eventos que o pipeline perdeu). O CSV revisado resultante se torna a UNICA
fonte de verdade para a avaliacao final do modelo (distinta do treino) --
o deliverable desta etapa e a ferramenta de amostragem/exportacao, a revisao
em si fica com o usuario.

Estrategia de amostragem (estratificada por sujeito e por cabeca):
  - Le os eventos direto dos campos canonicos do .pt (tonic_labels/
    phasic_labels/any_labels + *_cov, escritos por auto_label.py),
    reconstruindo eventos contiguos por cabeca via events_from_binary.
  - Para cada sujeito, sorteia ate --n-tonic eventos tonicos, --n-phasic
    fasicos e --n-any "any" (com replacement=False; sujeitos com menos
    eventos do que o pedido cedem todos os que tem). Tonico e sub-
    representado no pipeline -- por isso o default pede TODOS os eventos
    tonicos de cada sujeito (--n-tonic -1 = sem limite) em vez de amostrar.
  - Adicionalmente sorteia --n-negative janelas SEM nenhum evento (todas as
    3 cabecas = 0) por sujeito, para o revisor tambem confirmar
    verdadeiros-negativos (essencial para medir especificidade/precisao).
  - Cobre --n-subjects sujeitos (default: todos), sorteados sem reposicao
    (seed fixa p/ reprodutibilidade).

Saida: um CSV por sujeito em --out-dir, no MESMO formato de
classifier/labels/*_revisado.csv (subject_id,onset_s,duration_s,type,score),
mas com o sufixo "_amostra_revisao.csv" para nao colidir com os CSVs humanos
originais. O `score` exportado e o score medio de cobertura do pipeline
automatico (nao um score humano) -- serve so de contexto para o revisor.

Uso:
    python classifier/sample_for_review.py                         # todos os sujeitos
    python classifier/sample_for_review.py --n-subjects 15 --seed 0
    python classifier/sample_for_review.py --exam rbd1 --n-phasic 20
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from classifier.movement_clf.dataio import events_from_binary, EPOCH_SEC  # noqa: E402

DATA_DIR = HERE / "data"
DEFAULT_OUT_DIR = HERE / "labels" / "amostra_revisao"

HEAD_TO_LABELFIELD = {
    "tonic": ("tonic_labels", "tonic_cov"),
    "phasic": ("phasic_labels", "phasic_cov"),
    "any": ("any_labels", "any_cov"),
}


def load_pt(pt_path: Path) -> dict:
    return torch.load(pt_path, map_location="cpu", weights_only=False)


def events_for_head(obj: dict, head: str, subject_id: str) -> list[dict]:
    """Reconstroi eventos contiguos (onset_s/duration_s) para uma cabeca a
    partir dos campos binarios do .pt, usando a cobertura media como score
    (mesmo algoritmo de fusao de classifier/movement_clf/dataio.py::
    events_from_binary, generalizado para qualquer cabeca).
    """
    lab_key, cov_key = HEAD_TO_LABELFIELD[head]
    if lab_key not in obj:
        return []
    lab = obj[lab_key]
    lab = lab.numpy() if isinstance(lab, torch.Tensor) else np.asarray(lab)
    cov = obj.get(cov_key)
    cov = cov.numpy() if isinstance(cov, torch.Tensor) else (np.asarray(cov) if cov is not None else None)
    events = events_from_binary(lab.astype(bool), scores=cov, subject_id=subject_id, etype=head)
    return events


def negative_windows(obj: dict, subject_id: str) -> list[dict]:
    """Mini-epocas isoladas onde NENHUMA das 3 cabecas e positiva e a
    mini-epoca foi escorada (rswa_conf==1). Exportadas como "eventos" de
    1 mini-epoca (3s) com type='negative', para o revisor confirmar que de
    fato nao ha movimento ali.
    """
    T = int(obj["signals"].shape[0]) if "signals" in obj else len(obj.get("tonic_labels", []))
    tonic = obj.get("tonic_labels")
    phasic = obj.get("phasic_labels")
    anyl = obj.get("any_labels")
    conf = obj.get("rswa_conf")

    def _np(x, default):
        if x is None:
            return np.full(T, default, dtype=np.float32)
        return x.numpy() if isinstance(x, torch.Tensor) else np.asarray(x)

    tonic = _np(tonic, 0.0)
    phasic = _np(phasic, 0.0)
    anyl = _np(anyl, 0.0)
    conf = _np(conf, 1.0)

    neg_mask = (tonic < 0.5) & (phasic < 0.5) & (anyl < 0.5) & (conf > 0.5)
    idx = np.where(neg_mask)[0]
    out = []
    for m in idx:
        out.append({
            "subject_id": subject_id,
            "onset_s": round(float(m * EPOCH_SEC), 3),
            "duration_s": round(float(EPOCH_SEC), 3),
            "type": "negative",
            "score": 0.0,
        })
    return out


def sample_subject(pt_path: Path, rng: np.random.Generator,
                    n_tonic: int, n_phasic: int, n_any: int, n_negative: int) -> list[dict]:
    obj = load_pt(pt_path)
    subject_id = pt_path.stem
    rows = []
    for head, n_want in (("tonic", n_tonic), ("phasic", n_phasic), ("any", n_any)):
        evs = events_for_head(obj, head, subject_id)
        if n_want is not None and n_want >= 0 and len(evs) > n_want:
            sel_idx = rng.choice(len(evs), size=n_want, replace=False)
            evs = [evs[i] for i in sorted(sel_idx)]
        rows.extend(evs)
    if n_negative and n_negative > 0:
        negs = negative_windows(obj, subject_id)
        if len(negs) > n_negative:
            sel_idx = rng.choice(len(negs), size=n_negative, replace=False)
            negs = [negs[i] for i in sorted(sel_idx)]
        rows.extend(negs)
    rows.sort(key=lambda r: r["onset_s"])
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=str(DATA_DIR), help="diretorio dos .pt ja auto-rotulados")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="diretorio de saida dos CSVs de amostra")
    ap.add_argument("--n-subjects", type=int, default=-1, help="quantos sujeitos amostrar (-1 = todos)")
    ap.add_argument("--n-tonic", type=int, default=-1,
                     help="eventos tonicos por sujeito (-1 = todos, default: tonico e raro, nao filtra)")
    ap.add_argument("--n-phasic", type=int, default=15, help="eventos fasicos por sujeito")
    ap.add_argument("--n-any", type=int, default=10, help="eventos 'any' por sujeito")
    ap.add_argument("--n-negative", type=int, default=10, help="janelas negativas (sem evento) por sujeito")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exam", default=None, help="amostrar so este sujeito (stem, sem .pt)")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.exam:
        pt_paths = [data_dir / f"{args.exam}.pt"]
    else:
        pt_paths = sorted(data_dir.glob("*.pt"))
        if args.n_subjects is not None and args.n_subjects >= 0 and len(pt_paths) > args.n_subjects:
            sel_idx = rng.choice(len(pt_paths), size=args.n_subjects, replace=False)
            pt_paths = [pt_paths[i] for i in sorted(sel_idx)]

    print(f"Amostrando {len(pt_paths)} sujeito(s) de {data_dir} -> {out_dir}")
    print(f"n_tonic={args.n_tonic} n_phasic={args.n_phasic} n_any={args.n_any} "
          f"n_negative={args.n_negative} seed={args.seed}\n")

    total_rows = 0
    summary = []
    for pt_path in pt_paths:
        if not pt_path.exists():
            print(f"  {pt_path.name}: NAO ENCONTRADO", file=sys.stderr)
            continue
        rows = sample_subject(pt_path, rng, args.n_tonic, args.n_phasic, args.n_any, args.n_negative)
        out_csv = out_dir / f"{pt_path.stem}_amostra_revisao.csv"
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["subject_id", "onset_s", "duration_s", "type", "score"])
            w.writeheader()
            w.writerows(rows)
        by_type = {}
        for r in rows:
            by_type[r["type"]] = by_type.get(r["type"], 0) + 1
        print(f"  {pt_path.stem:12s}: {len(rows):4d} linhas -> {out_csv.name}  ({by_type})")
        summary.append({"subject_id": pt_path.stem, "n_rows": len(rows), **by_type})
        total_rows += len(rows)

    print(f"\nTotal: {total_rows} linhas exportadas em {len(pt_paths)} arquivo(s), em {out_dir}")
    print("Proximo passo: revisar manualmente cada *_amostra_revisao.csv (editar/apagar/adicionar linhas)")
    print("e usar o resultado como ground truth em scripts/evaluate_auto_labels.py.")


if __name__ == "__main__":
    main()
