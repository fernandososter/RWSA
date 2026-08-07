"""
Avaliação por sujeito do RSWADetectionNet (3 cabeças: tônico/fásico/any).

Carrega um checkpoint treinado (best.pt de um fold, ou vários --checkpoint
para ensemble por média de probabilidade) e um diretório de sujeitos (.pt),
roda o modelo em cada mini-época válida (REM + confiança > min-confidence,
mesma máscara usada no treino) e calcula F1/precisão/recall/kappa POR
SUJEITO e POR CABEÇA, além do agregado (todas as mini-épocas juntas).

IMPORTANTE (provisório): por padrão este script avalia contra os rótulos
armazenados no próprio .pt de --data-dir. Se esse .pt foi rotulado pelo
pipeline automático (classifier/auto_label.py, label_source=
auto_cnn_limiar_duplo_v1), o "ground truth" usado aqui é o rótulo do MESMO
pipeline que também gerou dados de treino -- não é uma avaliação
independente. Assim que a amostra de revisão humana (classifier/
sample_for_review.py -> classifier/labels/amostra_revisao/*_amostra_revisao.csv,
editada manualmente) existir, rode este script com --ground-truth-dir
apontando para os CSVs revisados: nesse modo o script rasteriza cada CSV
revisado (mesmo formato de *_revisado.csv / *_amostra_revisao.csv:
subject_id,onset_s,duration_s,type,score) em rótulos por mini-época,
restringe a avaliação ÀS MINI-ÉPOCAS EFETIVAMENTE AMOSTRADAS (cada linha
do CSV, incluindo linhas type=negative, marca sua janela como revisada) e
avalia contra ELES em vez dos rótulos do .pt. Esse é o modo de avaliação real;
o modo padrão (rótulo do .pt) serve apenas para verificar que o pipeline
de treino/avaliação funciona ponta-a-ponta enquanto a amostra revisada
não está pronta.

Saída:
  --out-csv (default classifier/eval_per_subject_report.csv): uma linha por
    (subject_id, head) com n_mini_epochs, n_positive, tp/fp/fn/tn,
    precision, recall, f1, kappa, threshold. Mais linhas de agregado
    (subject_id="ALL") por cabeça.
  --out-summary (default classifier/eval_per_subject_summary.md): relatório
    Markdown legível com tabelas por cabeça e destaque de sujeitos com
    poucas mini-épocas positivas (métricas instáveis).

Uso:
    python scripts/evaluate_per_subject.py \\
        --checkpoint runs/rswa/.../fold_0/checkpoints/best.pt \\
        --data-dir classifier/data \\
        --threshold 0.5

    # Ensemble de múltiplos folds (média de probabilidade):
    python scripts/evaluate_per_subject.py \\
        --checkpoint runs/rswa/.../fold_0/checkpoints/best.pt \\
        --checkpoint runs/rswa/.../fold_1/checkpoints/best.pt \\
        --data-dir classifier/data --tonic-threshold 0.4

    # Avaliação real, contra amostra revisada por humano (fonte de avaliação):
    python scripts/evaluate_per_subject.py \\
        --checkpoint runs/rswa/.../fold_0/checkpoints/best.pt \\
        --data-dir classifier/data \\
        --ground-truth-dir classifier/labels/amostra_revisao \\
        --ground-truth-suffix _amostra_revisao.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from sleep_rswa import ModelConfig, RSWADetectionNet, SleepAnalysisDataset, collate_sleep_analysis_exams  # noqa: E402
from sleep_rswa.data import load_subject_directory  # noqa: E402
from sleep_rswa.training import load_checkpoint, resolve_device  # noqa: E402

_HEADS = ("tonic", "phasic", "any")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", action="append", required=True, dest="checkpoints",
                     help="Caminho para um checkpoint (best.pt). Repita para ensemble por média de probabilidade.")
    ap.add_argument("--data-dir", type=Path, required=True, help="Diretório com os .pt dos sujeitos a avaliar.")
    ap.add_argument("--min-confidence", type=float, default=0.0)
    ap.add_argument("--all-stages", action="store_true", help="Avalia todas as fases (não só REM).")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--tonic-threshold", type=float, default=None)
    ap.add_argument("--phasic-threshold", type=float, default=None)
    ap.add_argument("--any-threshold", type=float, default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--d-model", type=int, default=None,
                     help="Deve casar com o D_MODEL usado no treino do checkpoint (ver runs/.../run.json). "
                          "Se omitido, usa o default de ModelConfig (256; ignora env D_MODEL para evitar "
                          "mismatch silencioso -- SEMPRE confira o d_model salvo em run.json/summary.json do treino).")
    ap.add_argument("--dropout", type=float, default=0.35, help="Só afeta treino; irrelevante em eval/inferência (model.eval()).")
    ap.add_argument("--min-positive-for-stable", type=int, default=5,
                     help="Sujeitos com menos positivos que isto (por cabeça) são marcados como instáveis no relatório.")
    ap.add_argument("--ground-truth-dir", type=Path, default=None,
                     help="Se dado, usa CSVs revisados por humano deste diretório como ground truth (avaliação real), em vez dos rótulos do .pt.")
    ap.add_argument("--ground-truth-suffix", default="_amostra_revisao.csv",
                     help="Sufixo do arquivo de ground truth por sujeito: {subject_id}{sufixo}. Default: amostra de revisão (classifier/sample_for_review.py).")
    ap.add_argument("--out-csv", type=Path, default=PROJ / "classifier" / "eval_per_subject_report.csv")
    ap.add_argument("--out-summary", type=Path, default=PROJ / "classifier" / "eval_per_subject_summary.md")
    return ap.parse_args()


def _resolve_thresholds(args) -> dict[str, float]:
    return {
        "tonic": args.tonic_threshold if args.tonic_threshold is not None else args.threshold,
        "phasic": args.phasic_threshold if args.phasic_threshold is not None else args.threshold,
        "any": args.any_threshold if args.any_threshold is not None else args.threshold,
    }


def _load_ground_truth_labels(subject, gt_dir: Path, suffix: str):
    """Lê um CSV de amostra revisada por humano (formato de
    classifier/sample_for_review.py: subject_id,onset_s,duration_s,type,score,
    type em {tonic,phasic,any,negative}, onset_s no referencial do .pt --
    mini-época 0 = início do tensor, mesma convenção de events_from_binary)
    e rasteriza em rótulos por mini-época NO REFERENCIAL DO .pt.

    Diferente do .pt inteiro, este CSV é uma AMOSTRA (não cobertura total):
    só as mini-épocas explicitamente cobertas por uma linha (evento de
    qualquer tipo, incluindo 'negative') são consideradas revisadas. Por
    isso retorna também ``valid`` -- a máscara das mini-épocas efetivamente
    amostradas/revisadas -- que DEVE substituir a máscara de validade REM
    do dataset ao avaliar em modo --ground-truth-dir (senão mini-épocas
    nunca revisadas por humano seriam contadas como negativo verdadeiro).

    Retorna dict com chaves 'tonic'/'phasic'/'any' (np.ndarray[n_epochs]
    int64) e 'valid' (np.ndarray[n_epochs] bool), ou None se o CSV não
    existir para este sujeito (sujeito fica de fora da avaliação real).
    """
    from classifier.movement_clf.dataio import EPOCH_SEC

    csv_path = gt_dir / f"{subject.subject_id}{suffix}"
    if not csv_path.exists():
        return None
    n_epochs = subject.n_epochs
    labels = {h: np.zeros(n_epochs, dtype=np.int64) for h in _HEADS}
    valid = np.zeros(n_epochs, dtype=bool)

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            try:
                onset = float(row["onset_s"])
                dur = float(row["duration_s"])
            except (KeyError, TypeError, ValueError):
                continue
            etype = str(row.get("type", "")).strip().lower()
            m0 = int(round(onset / EPOCH_SEC))
            m1 = int(round((onset + dur) / EPOCH_SEC))
            m0 = max(0, m0)
            m1 = min(n_epochs, m1)
            if m1 <= m0:
                continue
            valid[m0:m1] = True
            if etype in labels:
                labels[etype][m0:m1] = 1
            # etype == "negative": ja marcado valid=True, labels ficam 0.

    labels["valid"] = valid
    return labels


def collect_predictions_for_subject(models, subject, thresholds, device, amp, min_confidence, rem_mask_only,
                                     gt_override=None):
    """Roda o(s) modelo(s) em UM sujeito e retorna, por cabeça, expected/prob/pred.

    Sem ``gt_override``: restringe às mini-épocas válidas do dataset
    (``rswa_valid`` -- REM + confiança, mesma máscara do treino).
    Com ``gt_override`` (dict head->array + 'valid'->bool array, de uma
    amostra revisada por humano): restringe às mini-épocas efetivamente
    amostradas/revisadas (``gt_override['valid']``), IGNORANDO a máscara
    REM do dataset -- a amostra de revisão já é a fonte de verdade sobre
    quais mini-épocas foram avaliadas.
    """
    ds = SleepAnalysisDataset([subject], min_confidence=min_confidence, rem_mask_only=rem_mask_only)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_sleep_analysis_exams)
    batch = next(iter(loader))
    emg = batch["emg_center"].to(device)
    padding_mask = batch["padding_mask"].to(device)
    if gt_override is not None:
        valid_np = gt_override["valid"] & padding_mask[0].detach().cpu().numpy().astype(bool)
        idx = np.nonzero(valid_np)[0]
    else:
        valid_mask = (batch["rswa_valid"].to(device) & padding_mask)[0]
        idx = torch.nonzero(valid_mask, as_tuple=False).flatten().detach().cpu().numpy()

    probs_sum = {h: None for h in _HEADS}
    for model in models:
        model.eval()
        with torch.no_grad():
            ctx = torch.autocast(device_type=device.type, enabled=amp) if device.type == "cuda" else _nullctx()
            with ctx:
                outputs = model(emg, mask=padding_mask)
        for h in _HEADS:
            p = torch.sigmoid(outputs[f"{h}_logits"].float()).detach().cpu().numpy()[0]
            probs_sum[h] = p if probs_sum[h] is None else probs_sum[h] + p

    n_models = len(models)
    result = {}
    for h in _HEADS:
        prob = probs_sum[h] / n_models
        if gt_override is not None:
            expected_full = gt_override[h]
        else:
            key = f"{h}_labels"
            arr = getattr(subject, key)
            if arr is None:
                arr = subject.rswa_labels  # fallback mono-rotulo (retrocompat), improvável de ser usado aqui
            expected_full = arr.numpy().astype(np.int64) if isinstance(arr, torch.Tensor) else np.asarray(arr, dtype=np.int64)
        expected = expected_full[idx]
        p = prob[idx]
        pred = (p >= thresholds[h]).astype(np.int64)
        result[h] = {"expected": expected, "probability": p, "prediction": pred}
    return result, len(idx)


class _nullctx:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


def _binary_stats(expected: np.ndarray, pred: np.ndarray, threshold: float) -> dict[str, Any]:
    n = len(expected)
    n_pos = int(expected.sum())
    tp = int(((pred == 1) & (expected == 1)).sum())
    fp = int(((pred == 1) & (expected == 0)).sum())
    fn = int(((pred == 0) & (expected == 1)).sum())
    tn = int(((pred == 0) & (expected == 0)).sum())
    if n_pos == 0 and pred.sum() == 0:
        precision = recall = f1 = kappa = float("nan")
    else:
        precision = float(precision_score(expected, pred, zero_division=0))
        recall = float(recall_score(expected, pred, zero_division=0))
        f1 = float(f1_score(expected, pred, zero_division=0))
        try:
            kappa = float(cohen_kappa_score(expected, pred))
        except Exception:
            kappa = float("nan")
    return {
        "n_mini_epochs": n, "n_positive": n_pos, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1, "kappa": kappa, "threshold": threshold,
    }


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    thresholds = _resolve_thresholds(args)
    rem_mask_only = not args.all_stages

    model_cfg = ModelConfig(d_model=args.d_model, dropout=args.dropout) if args.d_model is not None else ModelConfig(dropout=args.dropout)
    print(f"Carregando {len(args.checkpoints)} checkpoint(s) em {device} (d_model={model_cfg.d_model})...")
    models = []
    for ckpt_path in args.checkpoints:
        model = RSWADetectionNet(config=model_cfg).to(device)
        try:
            payload = load_checkpoint(ckpt_path, model, device)
        except RuntimeError as e:
            raise RuntimeError(
                f"Falha ao carregar {ckpt_path}: mismatch de arquitetura. Se este checkpoint foi treinado "
                f"com D_MODEL diferente do default (256), passe --d-model <valor> (confira em "
                f"runs/.../run.json ou no log de treino). Erro original: {e}"
            ) from e
        models.append(model)
        meta = {k: payload.get(k) for k in ("epoch", "metrics") if k in payload}
        print(f"  {ckpt_path}: epoch={meta.get('epoch')} metrics={meta.get('metrics')}")

    print(f"Carregando sujeitos de {args.data_dir}...")
    subjects = load_subject_directory(args.data_dir)
    print(f"  {len(subjects)} sujeito(s) carregado(s).")

    gt_dir = args.ground_truth_dir
    if gt_dir is not None:
        print(f"MODO AVALIAÇÃO REAL: ground truth de {gt_dir} (sufixo '{args.ground_truth_suffix}')")

    rows: list[dict[str, Any]] = []
    agg_by_head: dict[str, list[np.ndarray]] = {h: [] for h in _HEADS}
    agg_pred_by_head: dict[str, list[np.ndarray]] = {h: [] for h in _HEADS}
    n_skipped = 0

    for subject in subjects:
        gt_override = None
        if gt_dir is not None:
            gt_override = _load_ground_truth_labels(subject, gt_dir, args.ground_truth_suffix)
            if gt_override is None:
                n_skipped += 1
                print(f"  {subject.subject_id}: sem CSV de ground truth em {gt_dir}, pulando (não faz parte da amostra revisada).")
                continue
        try:
            preds, n_valid = collect_predictions_for_subject(
                models, subject, thresholds, device, amp=not args.no_amp,
                min_confidence=args.min_confidence, rem_mask_only=rem_mask_only,
                gt_override=gt_override,
            )
        except StopIteration:
            n_skipped += 1
            print(f"  {subject.subject_id}: nenhuma mini-época válida (REM/confiança), pulando.")
            continue
        if n_valid == 0:
            n_skipped += 1
            print(f"  {subject.subject_id}: 0 mini-épocas válidas, pulando.")
            continue

        for h in _HEADS:
            expected = preds[h]["expected"]
            pred = preds[h]["prediction"]
            stats = _binary_stats(expected, pred, thresholds[h])
            stats["subject_id"] = subject.subject_id
            stats["head"] = h
            stats["unstable"] = stats["n_positive"] < args.min_positive_for_stable
            rows.append(stats)
            agg_by_head[h].append(expected)
            agg_pred_by_head[h].append(pred)

    for h in _HEADS:
        if not agg_by_head[h]:
            continue
        expected_all = np.concatenate(agg_by_head[h])
        pred_all = np.concatenate(agg_pred_by_head[h])
        stats = _binary_stats(expected_all, pred_all, thresholds[h])
        stats["subject_id"] = "ALL"
        stats["head"] = h
        stats["unstable"] = False
        rows.append(stats)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["subject_id", "head", "n_mini_epochs", "n_positive", "tp", "fp", "fn", "tn",
                  "precision", "recall", "f1", "kappa", "threshold", "unstable"]
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fieldnames})
    print(f"\nRelatório CSV salvo em {args.out_csv} ({len(rows)} linhas).")

    _write_markdown_summary(args, rows, thresholds, gt_dir, n_skipped, len(subjects))
    print(f"Resumo Markdown salvo em {args.out_summary}.")


def _write_markdown_summary(args, rows, thresholds, gt_dir, n_skipped, n_subjects_total) -> None:
    lines = []
    lines.append("# Avaliação por sujeito -- RSWADetectionNet (tônico/fásico/any)\n")
    if gt_dir is not None:
        lines.append(f"**Modo:** avaliação REAL contra ground truth revisado por humano em `{gt_dir}` "
                      f"(sufixo `{args.ground_truth_suffix}`).\n")
    else:
        lines.append("**Modo:** PROVISÓRIO -- avaliado contra os rótulos do próprio `.pt` "
                      "(mesmo pipeline automático que gerou os dados de treino; NÃO é avaliação "
                      "independente). Substitua por `--ground-truth-dir` apontando para a amostra "
                      "revisada por humano assim que ela existir.\n")
    lines.append(f"Sujeitos carregados: {n_subjects_total} | avaliados: {n_subjects_total - n_skipped} | "
                  f"pulados (sem ground truth/sem mini-épocas válidas): {n_skipped}\n")
    lines.append(f"Limiares: tônico={thresholds['tonic']} fásico={thresholds['phasic']} any={thresholds['any']}\n")

    for h in _HEADS:
        head_rows = [r for r in rows if r["head"] == h and r["subject_id"] != "ALL"]
        agg_row = next((r for r in rows if r["head"] == h and r["subject_id"] == "ALL"), None)
        lines.append(f"\n## Cabeça: {h}\n")
        if agg_row:
            lines.append(f"**Agregado (todos os sujeitos, todas as mini-épocas):** "
                          f"F1={agg_row['f1']:.3f} | Precisão={agg_row['precision']:.3f} | "
                          f"Recall={agg_row['recall']:.3f} | Kappa={agg_row['kappa']:.3f} | "
                          f"n_positive={agg_row['n_positive']}/{agg_row['n_mini_epochs']}\n")
        n_unstable = sum(1 for r in head_rows if r["unstable"])
        lines.append(f"Sujeitos com < {args.min_positive_for_stable} positivos (métricas instáveis): {n_unstable}/{len(head_rows)}\n")
        lines.append("\n| subject_id | n_mini_epochs | n_positive | precision | recall | f1 | kappa | instável |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for r in sorted(head_rows, key=lambda r: (r["unstable"], -(r["n_positive"] or 0))):
            f1_str = f"{r['f1']:.3f}" if r['f1'] == r['f1'] else "n/a"
            prec_str = f"{r['precision']:.3f}" if r['precision'] == r['precision'] else "n/a"
            rec_str = f"{r['recall']:.3f}" if r['recall'] == r['recall'] else "n/a"
            kappa_str = f"{r['kappa']:.3f}" if r['kappa'] == r['kappa'] else "n/a"
            flag = "⚠" if r["unstable"] else ""
            lines.append(f"| {r['subject_id']} | {r['n_mini_epochs']} | {r['n_positive']} | "
                         f"{prec_str} | {rec_str} | {f1_str} | {kappa_str} | {flag} |")

    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
