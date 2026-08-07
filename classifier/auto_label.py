"""
Rotulagem automatica dos exames de treino: CNN (detector de movimento) +
limiar duplo (histerese) -> escreve DIRETO nos campos canonicos do .pt
(tonic_labels/phasic_labels/any_labels + *_cov + rswa_conf), substituindo
qualquer rotulo anterior (humano ou nao).

Fluxo (passada unica, por exame):
  1. Le o .pt, extrai o EMG (canal 4) achatado em serie continua (mesma
     reconstrucao que testes/src/limiar/evaluate.py::load_exam_emg usa).
  2. Roda a CNN ja treinada (classifier/outputs/movement_cnn_final.pt) sobre
     o exame inteiro (score por mini-epoca, MovementCNN + build_tensors) e
     funde mini-epocas positivas adjacentes (score >= limiar da CNN) em
     janelas candidatas (onset_s, duration_s) -- mesma logica de
     classifier/predict_movements.py::predict_to_csv.
  3. Para cada janela candidata, recorta o EMG EXATAMENTE a essa janela (SEM
     margem -- mesma convencao ja validada em
     testes/src/limiar/evaluate.py::detect_events_in_interval, que tambem
     calcula a baseline local so dentro do recorte) e roda o limiar duplo
     (double threshold / histerese, testes/src/limiar/threshold_rule.py)
     dentro desse recorte.
  4. Se o limiar duplo nao confirma amplitude suficiente dentro da janela
     (nenhum segmento passa MIN_AMPLITUDE_RATIO), a janela e DESCARTADA por
     completo -- nao gera rotulo de tipo nenhum. Risco conhecido e aceito:
     perda silenciosa de eventos reais, sobretudo tonicos (ver relatorio de
     limitacoes).
  5. Eventos confirmados sao classificados por duracao E amplitude em
     fasico/any/tonico (mesmos cortes de threshold_rule.py) e convertidos em
     cobertura fracionaria por mini-epoca (mesmo algoritmo de
     rasterize_rswa_annotations), escritos direto nos campos do .pt.

Escopo: classifier/data/*.pt -- os 60 exames reais que scripts/train_rswa.py
de fato usa (--data-dir). testes/data_real/ e testes/data/ NAO sao tocados:
sao fixtures do teste unitario do limiar (ground truth via CSV revisado ou
sintetico conhecido), fora do escopo de treino do RSWADetectionNet.

Todos os 60 exames sao sobrescritos, inclusive os que ja tinham CSV revisado
por humano -- os 59 CSVs em classifier/labels/*_revisado.csv NAO sao tocados
e continuam a fonte primaria (podem re-gerar os campos humanos a qualquer
momento via rasterize_rswa_annotations). A avaliacao contra ground truth
humano fica provisoria (amostra_revisao.py) at'e existir uma nova amostra
revisada -- ver scripts/evaluate_auto_labels.py.

Uso:
    python classifier/auto_label.py --dry-run              # relatorio, nao grava
    python classifier/auto_label.py                          # aplica e grava
    python classifier/auto_label.py --exam rbd1 --dry-run    # um exame so
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

LIMIAR_DIR = PROJ / "testes" / "src" / "limiar"
if str(LIMIAR_DIR) not in sys.path:
    sys.path.insert(0, str(LIMIAR_DIR))

from classifier.movement_clf.dataio import (  # noqa: E402
    EMG_CHANNEL_INDEX, EPOCH_SEC, FS, load_exam, zscore_emg, events_from_binary,
)
from classifier.movement_clf.dataset import build_tensors  # noqa: E402
from classifier.movement_clf.model import MovementCNN  # noqa: E402
from classifier.movement_clf.engine import resolve_device  # noqa: E402
from threshold_rule import (  # noqa: E402
    K_SINGLE, K_ON, K_OFF, K_OFF_HOLD_S, MIN_AMPLITUDE_RATIO,
    PHASIC_LO_S, PHASIC_HI_S, TONIC_MIN_DUR_S, detect_events,
)

DATA_DIR = HERE / "data"
DEFAULT_MODEL = HERE / "outputs" / "movement_cnn_final.pt"
BACKUP_DIR = HERE / "data_backup_auto_label"
LABEL_SOURCE = "auto_cnn_limiar_duplo_v1"

# Cobertura minima por mini-epoca para contar como rotulo positivo daquele
# tipo (mesma convencao de rasterize_rswa_annotations): tonico exige maioria
# da janela coberta (evento sustentado); fasico/any contam com qualquer
# presenca (eventos breves nao devem exigir cobertura alta de uma janela de
# 3s). Ver relatorio de limitacoes para a discussao desse corte em "any".
TONIC_MIN_COVERAGE = 0.5
PHASIC_MIN_COVERAGE = 0.0
ANY_MIN_COVERAGE = 0.0


def load_emg_flat(pt_path: Path) -> tuple[np.ndarray, dict]:
    """Le o .pt e devolve (emg_flat[N] float64, obj) sem alterar o dict original."""
    obj = torch.load(pt_path, map_location="cpu", weights_only=False)
    emg_flat = obj["signals"][:, EMG_CHANNEL_INDEX, :].numpy().astype("float64").reshape(-1)
    return emg_flat, obj


@torch.no_grad()
def score_exam_cnn(model, exam, window_epochs: int, device: str = "cpu",
                    batch_size: int = 512) -> np.ndarray:
    from torch.utils.data import DataLoader, TensorDataset
    X, y = build_tensors([exam], window_epochs=window_epochs)
    loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=False)
    scores = []
    for x, _ in loader:
        scores.append(torch.sigmoid(model(x.to(device))).cpu().numpy())
    return np.concatenate(scores)


def load_cnn(model_path: Path, device: str = "cpu"):
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    window = ckpt.get("window_epochs", 5)
    model = MovementCNN(window_epochs=window)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model, ckpt


def clip_window_to_exam(start_s: float, end_s: float, n_samples: int, fs: int = FS) -> tuple[int, int]:
    i0 = max(0, int(round(start_s * fs)))
    i1 = min(n_samples, int(round(end_s * fs)))
    return i0, i1


def detect_in_window(emg_flat: np.ndarray, start_s: float, end_s: float) -> list[dict]:
    """Roda o limiar duplo dentro da janela candidata da CNN, SEM margem.

    Baseline e limiares calculados so dentro do recorte (mesma convencao de
    testes/src/limiar/evaluate.py::detect_events_in_interval). Devolve lista
    de eventos (onset_s/duration_s/type/score) JA no referencial do .pt
    inteiro (nao do recorte).
    """
    i0, i1 = clip_window_to_exam(start_s, end_s, len(emg_flat), fs=FS)
    if i0 >= i1:
        return []
    local_events = detect_events(
        emg_flat[i0:i1], method="double", fs=FS, apply_merge_gaps=True,
        k_single=K_SINGLE, k_on=K_ON, k_off=K_OFF, off_hold_s=K_OFF_HOLD_S,
    )
    clipped_start_s = i0 / FS
    out = []
    for ev in local_events:
        out.append({
            "onset_s": clipped_start_s + ev.onset_s,
            "duration_s": ev.duration_s,
            "type": ev.type,
            "score": ev.score,
        })
    return out


def events_to_pt_labels(events: list[dict], T: int, epoch_sec: float = EPOCH_SEC) -> dict[str, np.ndarray]:
    """Converte eventos classificados (onset_s/duration_s/type no referencial
    do .pt) em cobertura fracionaria + rotulo binario por mini-epoca, para as
    3 cabecas (tonic/phasic/any). Mesmo algoritmo de coverage-by-overlap de
    src/sleep_rswa/preprocessing/rswa_labels.py::rasterize_rswa_annotations,
    generalizado para 3 tipos e recebendo eventos ja no tempo do .pt (sem
    precisar de annot_start, pois nao ha CSV/EDF envolvido aqui).
    """
    tonic_cov = np.zeros(T, dtype=np.float64)
    phasic_cov = np.zeros(T, dtype=np.float64)
    any_cov = np.zeros(T, dtype=np.float64)

    for ev in events:
        start = max(0.0, ev["onset_s"])
        end = start + ev["duration_s"]
        if end <= 0:
            continue
        first = max(0, int(start // epoch_sec))
        last = min(T - 1, int((end - 1e-9) // epoch_sec))
        cov = {"tonic": tonic_cov, "phasic": phasic_cov, "any": any_cov}.get(ev["type"])
        if cov is None:
            continue
        for m in range(first, last + 1):
            m0, m1 = m * epoch_sec, (m + 1) * epoch_sec
            frac = max(0.0, min(end, m1) - max(start, m0)) / epoch_sec
            cov[m] = min(1.0, cov[m] + frac)

    tonic_lab = (tonic_cov >= TONIC_MIN_COVERAGE).astype(np.float32)
    phasic_lab = (phasic_cov > PHASIC_MIN_COVERAGE).astype(np.float32)
    any_lab = (any_cov > ANY_MIN_COVERAGE).astype(np.float32)

    return {
        "tonic_labels": tonic_lab,
        "phasic_labels": phasic_lab,
        "any_labels": any_lab,
        "tonic_cov": tonic_cov.astype(np.float32),
        "phasic_cov": phasic_cov.astype(np.float32),
        "any_cov": any_cov.astype(np.float32),
    }


def auto_label_one(pt_path: Path, cnn_model, cnn_ckpt, cnn_threshold: float,
                    cnn_min_epochs: int, device: str = "cpu") -> dict:
    """Roda o pipeline completo (CNN -> limiar duplo -> rotulos) para 1 exame.

    Devolve um dict com o novo estado dos campos do .pt + estatisticas de QC.
    NAO grava nada em disco (ver apply_to_pt / main).
    """
    emg_flat, obj = load_emg_flat(pt_path)
    exam = load_exam(pt_path, require_labels=False)
    window = cnn_ckpt.get("window_epochs", 5)

    cnn_scores = score_exam_cnn(cnn_model, exam, window, device=device)
    cnn_mask = cnn_scores >= cnn_threshold
    candidates = events_from_binary(cnn_mask, scores=cnn_scores, subject_id=exam.subject_id)
    min_dur = cnn_min_epochs * EPOCH_SEC
    candidates = [c for c in candidates if c["duration_s"] >= min_dur - 1e-6]

    confirmed_events = []
    n_discarded = 0
    for cand in candidates:
        evs = detect_in_window(emg_flat, cand["onset_s"], cand["onset_s"] + cand["duration_s"])
        if not evs:
            n_discarded += 1
            continue
        confirmed_events.extend(evs)

    T = int(obj["signals"].shape[0])
    labels = events_to_pt_labels(confirmed_events, T)

    stages = obj.get("sleep_stages")
    stages_np = stages.numpy() if isinstance(stages, torch.Tensor) else np.asarray(stages)
    rswa_conf = (stages_np != -1).astype(np.float32)

    n_tonic = int(labels["tonic_labels"].sum())
    n_phasic = int(labels["phasic_labels"].sum())
    n_any = int(labels["any_labels"].sum())
    stats = {
        "subject_id": exam.subject_id,
        "n_epochs": T,
        "hours": round(T * EPOCH_SEC / 3600.0, 2),
        "n_cnn_candidates": len(candidates),
        "n_confirmed_events": len(confirmed_events),
        "n_discarded_windows": n_discarded,
        "n_tonic_epochs": n_tonic,
        "n_phasic_epochs": n_phasic,
        "n_any_epochs": n_any,
        "pct_tonic": round(100.0 * n_tonic / T, 3) if T else 0.0,
        "pct_phasic": round(100.0 * n_phasic / T, 3) if T else 0.0,
        "pct_any": round(100.0 * n_any / T, 3) if T else 0.0,
    }
    return {
        "obj": obj, "labels": labels, "rswa_conf": rswa_conf,
        "stats": stats, "confirmed_events": confirmed_events,
    }


def apply_to_pt(pt_path: Path, result: dict, dry_run: bool) -> None:
    obj = result["obj"]
    obj["tonic_labels"] = torch.from_numpy(result["labels"]["tonic_labels"])
    obj["phasic_labels"] = torch.from_numpy(result["labels"]["phasic_labels"])
    obj["any_labels"] = torch.from_numpy(result["labels"]["any_labels"])
    obj["tonic_cov"] = torch.from_numpy(result["labels"]["tonic_cov"])
    obj["phasic_cov"] = torch.from_numpy(result["labels"]["phasic_cov"])
    obj["any_cov"] = torch.from_numpy(result["labels"]["any_cov"])
    obj["rswa_conf"] = torch.from_numpy(result["rswa_conf"])
    tonic_lab = result["labels"]["tonic_labels"]
    phasic_lab = result["labels"]["phasic_labels"]
    rswa_int = (phasic_lab.astype(np.int64) * 1) + (tonic_lab.astype(np.int64) * 2)
    obj["rswa_labels"] = torch.from_numpy(rswa_int)
    obj["label_source"] = LABEL_SOURCE
    if not dry_run:
        torch.save(obj, pt_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=str(DATA_DIR), help="diretorio dos .pt (default: classifier/data)")
    ap.add_argument("--model", default=str(DEFAULT_MODEL), help="checkpoint da CNN")
    ap.add_argument("--cnn-threshold", type=float, default=None,
                     help="limiar da CNN (default: o gravado no checkpoint)")
    ap.add_argument("--cnn-min-epochs", type=int, default=1,
                     help="duracao minima (em mini-epocas) de uma janela candidata da CNN")
    ap.add_argument("--device", default="auto", help="auto (default), cpu, cuda ou cuda:N")
    ap.add_argument("--exam", default=None, help="rotular so este exame (stem, sem .pt)")
    ap.add_argument("--dry-run", action="store_true", help="nao grava nada, so mostra o relatorio")
    ap.add_argument("--no-backup", action="store_true",
                     help="pula o backup automatico antes da 1a gravacao (backup principal ja existe em backups/)")
    args = ap.parse_args()

    device = resolve_device(args.device)
    data_dir = Path(args.data_dir)
    cnn_model, cnn_ckpt = load_cnn(Path(args.model), device=device)
    cnn_threshold = args.cnn_threshold if args.cnn_threshold is not None else cnn_ckpt.get("threshold", 0.5)

    if args.exam:
        pt_paths = [data_dir / f"{args.exam}.pt"]
    else:
        pt_paths = sorted(data_dir.glob("*.pt"))
    if not pt_paths:
        print(f"Nenhum .pt encontrado em {data_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"CNN: {args.model} (limiar={cnn_threshold:.3f}, device={device})")
    print(f"Limiar duplo: k_on={K_ON} k_off={K_OFF} min_amplitude_ratio={MIN_AMPLITUDE_RATIO}")
    print(f"Classificacao: fasico [{PHASIC_LO_S},{PHASIC_HI_S}]s | any ({PHASIC_HI_S},{TONIC_MIN_DUR_S})s | tonico >={TONIC_MIN_DUR_S}s")
    print(f"Exames: {len(pt_paths)} em {data_dir}")
    print("dry-run -- nada sera gravado\n" if args.dry_run else "APLICANDO -- os .pt serao sobrescritos\n")

    if not args.dry_run and not args.no_backup and not BACKUP_DIR.exists():
        import shutil
        BACKUP_DIR.mkdir(parents=True)
        for p in pt_paths:
            shutil.copy2(p, BACKUP_DIR / p.name)
        print(f"backup adicional (pre-escrita desta rodada) em: {BACKUP_DIR}\n")

    rows = []
    for pt_path in pt_paths:
        if not pt_path.exists():
            print(f"  {pt_path.name}: NAO ENCONTRADO", file=sys.stderr)
            continue
        try:
            result = auto_label_one(pt_path, cnn_model, cnn_ckpt, cnn_threshold,
                                      args.cnn_min_epochs, device=device)
        except Exception as e:
            print(f"  {pt_path.stem}: ERRO - {e}", file=sys.stderr)
            continue
        s = result["stats"]
        print(f"  {s['subject_id']:12s} {s['hours']:5.1f}h | candidatos_cnn={s['n_cnn_candidates']:4d} "
              f"confirmados={s['n_confirmed_events']:4d} descartados={s['n_discarded_windows']:4d} | "
              f"tonic={s['pct_tonic']:5.2f}% phasic={s['pct_phasic']:5.2f}% any={s['pct_any']:5.2f}%")
        apply_to_pt(pt_path, result, dry_run=args.dry_run)
        rows.append(s)

    if rows:
        import csv as _csv
        report_path = HERE / ("auto_label_report_dryrun.csv" if args.dry_run else "auto_label_report.csv")
        with open(report_path, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nrelatorio por exame: {report_path}")

    print("\ndry-run concluido, nada foi gravado." if args.dry_run else "\npronto -- .pt sobrescritos.")


if __name__ == "__main__":
    main()
