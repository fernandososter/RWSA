"""
Avaliacao dos resultados LOSO: metricas por mini-epoca e por evento,
escolha do limiar de operacao (alta cobertura p/ pre-anotacao) e figuras.

Le  outputs/loso_predictions.npz
Salva:
  outputs/metrics_epoch.csv       metricas por mini-epoca, por sujeito + pooled
  outputs/metrics_event.csv       metricas por evento (recall, FA/h) por sujeito
  outputs/operating_point.json    limiar escolhido + metricas nesse ponto
  outputs/pr_curves.png           curvas PR por sujeito
  outputs/threshold_sweep.png     recall de evento e FA/h vs limiar (pooled)
Uso: python classifier/evaluate_loso.py [--target-event-recall 0.90]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from classifier.movement_clf.metrics import epoch_metrics, event_metrics, pr_auc, best_f1_threshold

OUT = HERE / "outputs"


def load_preds():
    z = np.load(OUT / "loso_predictions.npz", allow_pickle=True)
    subjects = [str(s) for s in z["subjects"]]
    data = {}
    for sid in subjects:
        data[sid] = dict(
            y=z[f"{sid}__y"], score=z[f"{sid}__score"],
            stages=z[f"{sid}__stages"], hours=float(z[f"{sid}__hours"][0]),
        )
    return subjects, data


def choose_threshold(data, target_event_recall=0.90, grid=None):
    """Menor limiar (mais coberto) que atinge o recall de EVENTO alvo (pooled)."""
    if grid is None:
        grid = np.round(np.linspace(0.05, 0.95, 19), 3)
    best = None
    sweep = []
    for t in grid:
        hits = tot = fa = 0
        hrs = 0.0
        for sid, d in data.items():
            em = event_metrics(d["y"], d["score"], t, d["hours"])
            hits += em["event_recall"] * em["n_true_events"] if not np.isnan(em["event_recall"]) else 0
            tot += em["n_true_events"]
            fa += em["false_alarms"]
            hrs += d["hours"]
        rec = hits / tot if tot else float("nan")
        fah = fa / hrs if hrs else float("nan")
        sweep.append((float(t), float(rec), float(fah)))
        if rec >= target_event_recall:
            best = float(t)  # continua subindo; guarda o maior t que ainda cumpre
    # menor FA/h entre os que cumprem o alvo -> maior limiar que satisfaz
    ok = [s for s in sweep if s[1] >= target_event_recall]
    thr = max(o[0] for o in ok) if ok else min(sweep, key=lambda s: -s[1])[0]
    return thr, sweep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-event-recall", type=float, default=0.90)
    args = ap.parse_args()

    subjects, data = load_preds()

    thr, sweep = choose_threshold(data, args.target_event_recall)
    print(f"limiar de operacao escolhido: {thr:.3f} (alvo recall de evento >= {args.target_event_recall})")

    # metricas por mini-epoca
    import csv
    pooled_y = np.concatenate([data[s]["y"] for s in subjects])
    pooled_score = np.concatenate([data[s]["score"] for s in subjects])
    rows_ep = []
    for sid in subjects:
        d = data[sid]
        m = epoch_metrics(d["y"], d["score"], thr)
        m["subject_id"] = sid
        rows_ep.append(m)
    mp = epoch_metrics(pooled_y, pooled_score, thr); mp["subject_id"] = "POOLED"
    rows_ep.append(mp)
    with open(OUT / "metrics_epoch.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["subject_id", "threshold", "precision", "recall",
                                          "f1", "pr_auc", "n_pos", "n_pred_pos"])
        w.writeheader()
        for r in rows_ep:
            w.writerow({k: r[k] for k in w.fieldnames})

    # metricas por evento
    rows_ev = []
    for sid in subjects:
        d = data[sid]
        em = event_metrics(d["y"], d["score"], thr, d["hours"])
        em["subject_id"] = sid
        em["pr_auc"] = round(pr_auc(d["y"], d["score"]), 4)
        rows_ev.append(em)
    with open(OUT / "metrics_event.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["subject_id", "pr_auc", "n_true_events", "n_pred_events",
                                          "event_recall", "false_alarms", "false_alarms_per_hour"])
        w.writeheader()
        for r in rows_ev:
            w.writerow({k: r.get(k) for k in w.fieldnames})

    # operating point json
    op = dict(threshold=thr, target_event_recall=args.target_event_recall,
              pooled_epoch=mp, per_subject_event=rows_ev)
    json.dump(op, open(OUT / "operating_point.json", "w"), indent=2)

    print("\n== por mini-epoca ==")
    for r in rows_ep:
        print(f"  {r['subject_id']:>7}: PR-AUC={r['pr_auc']:.3f} F1={r['f1']:.3f} "
              f"P={r['precision']:.3f} R={r['recall']:.3f}")
    print("== por evento ==")
    for r in rows_ev:
        print(f"  {r['subject_id']:>7}: recall={r['event_recall']:.3f} FA/h={r['false_alarms_per_hour']:.2f} "
              f"(true={r['n_true_events']})")

    # ---- figuras ----
    _figs(subjects, data, sweep, thr)


def _figs(subjects, data, sweep, thr):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve

    colors = {"rbd1": "#4a90c2", "rbd2": "#2a9d8f", "rbd3": "#e0a458", "rbd5": "#d1495b"}

    fig, ax = plt.subplots(figsize=(5, 4.2))
    for sid in subjects:
        d = data[sid]
        if d["y"].sum() == 0:
            continue
        p, r, _ = precision_recall_curve(d["y"].astype(int), d["score"])
        ap = pr_auc(d["y"], d["score"])
        prev = d["y"].mean()
        ax.plot(r, p, lw=1.4, color=colors.get(sid, "#666"), label=f"{sid} (AP={ap:.2f})")
        ax.axhline(prev, color=colors.get(sid, "#666"), lw=0.5, ls=":", alpha=0.5)
    ax.set_xlabel("recall (mini-epoca)"); ax.set_ylabel("precision")
    ax.set_title("Curvas Precision-Recall por sujeito (LOSO)", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(OUT / "pr_curves.png", dpi=200, bbox_inches="tight")

    sw = np.array(sweep)
    fig2, ax1 = plt.subplots(figsize=(5.6, 4.0))
    ax1.plot(sw[:, 0], sw[:, 1], "-o", color="#2a9d8f", ms=3, label="recall de evento")
    ax1.set_xlabel("limiar"); ax1.set_ylabel("recall de evento", color="#2a9d8f")
    ax1.tick_params(axis="y", labelcolor="#2a9d8f")
    ax1.axvline(thr, color="#666", ls="--", lw=1.0)
    ax1.text(thr + 0.01, 0.2, f"operacao\n{thr:.2f}", fontsize=7, color="#333")
    ax2 = ax1.twinx()
    ax2.plot(sw[:, 0], sw[:, 2], "-s", color="#d1495b", ms=3, label="FA/h")
    ax2.set_ylabel("falsos alarmes por hora", color="#d1495b")
    ax2.tick_params(axis="y", labelcolor="#d1495b")
    ax1.set_title("Limiar × cobertura vs falsos alarmes (pooled)", loc="left", fontsize=10)
    fig2.tight_layout()
    fig2.savefig(OUT / "threshold_sweep.png", dpi=200, bbox_inches="tight")
    print(f"\nfiguras salvas em {OUT}")


if __name__ == "__main__":
    main()
