"""
Revisor de movimento — aplicacao local para inspecionar as deteccoes do
modelo sobre o EMG e confirmar cada evento como tonico/fasico ou apagar.

- Le o canal EMG direto do .pt (o mesmo sinal que o modelo viu; alinhamento exato).
- Roda o detector (classifier/outputs/movement_cnn_final.pt) para marcar onde ha
  movimento, funde mini-epocas adjacentes em eventos.
- Serve uma pagina web (index.html) onde voce revisa evento a evento:
  botoes Tonico / Fasico / Apagar, e salva um CSV revisado no seu formato.

Backend usa apenas a biblioteca padrao (http.server). Precisa do venv do projeto
(torch/numpy) para ler o .pt e rodar o modelo.

Uso:
    python view/app.py [--data DIR] [--model CKPT] [--port 8000]
Depois abra http://localhost:8000 no navegador.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

# view/ pode importar de classifier/ (o contrato de isolamento e classifier<->src/sleep_rswa)
from classifier.movement_clf.dataio import load_exam, zscore_emg, events_from_binary, EPOCH_SEC, FS
from classifier.movement_clf.dataset import build_tensors
from classifier.movement_clf.model import MovementCNN

# ---- estado global (configurado no main) ----
CFG = {
    "data_dir": PROJ / "classifier" / "data",
    "model_path": PROJ / "classifier" / "outputs" / "movement_cnn_final.pt",
    "out_dir": HERE / "revisado",
}
_MODEL = {"net": None, "window": 5, "threshold": 0.2}
_CACHE = {}  # exam_name -> dict(emg, stages, scores, mask, events, hours)


MAT_DIR = HERE / "mat"
CFG_PATH = HERE / "exam_config.json"


def _load_config():
    """Config por exame: {exame: {meas_date, hipno_start, annot_start}}."""
    if CFG_PATH.exists():
        try:
            return json.loads(CFG_PATH.read_text())
        except Exception:
            return {}
    return {}


def _save_config(cfg):
    CFG_PATH.write_text(json.dumps(cfg, ensure_ascii=False, indent=2))


def _hms_to_sec(s):
    """'HH:MM:SS' (ou 'HH:MM') -> segundos desde a meia-noite. None se invalido."""
    if s is None:
        return None
    parts = str(s).strip().replace(",", ":").split(":")
    try:
        parts = [float(p) for p in parts if p != ""]
    except ValueError:
        return None
    if not parts:
        return None
    h = parts[0]
    m = parts[1] if len(parts) > 1 else 0.0
    sec = parts[2] if len(parts) > 2 else 0.0
    return h * 3600.0 + m * 60.0 + sec


def _sec_to_hms(x):
    if x is None:
        return None
    x = int(round(float(x))) % 86400
    return f"{x // 3600:02d}:{(x % 3600) // 60:02d}:{x % 60:02d}"


def _mat_hipno_start_sec(exam_name):
    """Hora de inicio do hipnograma (seg desde meia-noite) lida de view/mat/hyp_<exame>.mat."""
    p = MAT_DIR / f"hyp_{exam_name}.mat"
    if not p.exists():
        return None
    try:
        from scipy.io import loadmat
        d = loadmat(str(p), simplify_cells=True)
    except Exception:
        return None
    stt = d.get("start_time")
    if isinstance(stt, dict) and "h" in stt:
        return float(stt["h"]) * 3600.0 + float(stt["m"]) * 60.0 + float(stt["s"])
    ts = d.get("timestart")
    return float(ts) if ts is not None else None


def _compute_annot_start(exam_name, meas_sec):
    """annot_start = (inicio_hipnograma - meas_date_EDF), com correcao de meia-noite."""
    hs = _mat_hipno_start_sec(exam_name)
    if hs is None or meas_sec is None:
        return None
    off = hs - meas_sec
    if off < 0:
        off += 86400.0  # gravacao cruza a meia-noite
    return round(off, 3)


def _load_model():
    if _MODEL["net"] is not None:
        return
    ckpt = torch.load(CFG["model_path"], map_location="cpu", weights_only=False)
    window = ckpt.get("window_epochs", 5)
    net = MovementCNN(window_epochs=window)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    _MODEL["net"] = net
    _MODEL["window"] = window
    _MODEL["threshold"] = float(ckpt.get("threshold", 0.2))


@torch.no_grad()
def _score_exam(exam):
    from torch.utils.data import DataLoader, TensorDataset
    X, y = build_tensors([exam], window_epochs=_MODEL["window"])
    loader = DataLoader(TensorDataset(X, y), batch_size=512, shuffle=False)
    out = []
    for xb, _ in loader:
        out.append(torch.sigmoid(_MODEL["net"](xb)).cpu().numpy())
    return np.concatenate(out)


def _prepare(exam_name):
    if exam_name in _CACHE:
        return _CACHE[exam_name]
    _load_model()
    pt = CFG["data_dir"] / f"{exam_name}.pt"
    exam = load_exam(pt, require_labels=False)
    scores = _score_exam(exam)
    thr = _MODEL["threshold"]
    mask = scores >= thr
    events = events_from_binary(mask, scores=scores, subject_id=exam.subject_id, etype="movement")
    # rotulos existentes no .pt (se houver) para pre-sugerir tonico/fasico
    obj = torch.load(pt, map_location="cpu", weights_only=False)
    tonic = obj.get("tonic_labels")
    phasic = obj.get("phasic_labels")
    tonic = (tonic.numpy() if isinstance(tonic, torch.Tensor) else np.asarray(tonic)) if tonic is not None else None
    phasic = (phasic.numpy() if isinstance(phasic, torch.Tensor) else np.asarray(phasic)) if phasic is not None else None
    cfg = _load_config().get(exam_name, {})
    hipno_start = _mat_hipno_start_sec(exam_name)
    st = {
        "emg": zscore_emg(exam.emg).astype(np.float32),  # [T,300]
        "stages": exam.stages.astype(int),
        "scores": scores.astype(float),
        "mask": mask,
        "events": events,
        "hours": exam.hours,
        "subject_id": exam.subject_id,
        "n_epochs": exam.n_epochs,
        "threshold": thr,
        "tonic": tonic, "phasic": phasic,
        "annot_start": cfg.get("annot_start"),   # None ate informar meas_date
        "hipno_start": hipno_start,              # inicio do hipnograma (seg desde meia-noite)
        "meas_date": cfg.get("meas_date"),       # inicio do EDF (HH:MM:SS) informado pelo usuario
        "has_mat": hipno_start is not None,
    }
    _CACHE[exam_name] = st
    return st


STAGE_NAMES = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "REM", -1: "?"}


def _events_payload(st):
    """Lista de eventos com sugestao inicial tonico/fasico a partir dos rotulos do .pt."""
    out = []
    a0 = st.get("annot_start")
    for i, ev in enumerate(st["events"]):
        e0 = int(round(ev["onset_s"] / EPOCH_SEC))
        e1 = int(round((ev["onset_s"] + ev["duration_s"]) / EPOCH_SEC))
        e1 = max(e1, e0 + 1)
        stages = st["stages"][e0:e1]
        # estagio dominante do evento
        vals, cnts = np.unique(stages, return_counts=True)
        dom = int(vals[np.argmax(cnts)]) if len(vals) else -1
        suggestion = "movement"
        if st["tonic"] is not None and st["phasic"] is not None:
            t = float(st["tonic"][e0:e1].max()) if e1 > e0 else 0.0
            p = float(st["phasic"][e0:e1].max()) if e1 > e0 else 0.0
            if t > 0.5:
                suggestion = "tonic"
            elif p > 0.5:
                suggestion = "phasic"
        onset_edf = round(float(ev["onset_s"]) + a0, 1) if a0 is not None else None
        out.append({
            "id": i,
            "onset_s": round(float(ev["onset_s"]), 1),      # tempo relativo ao .pt
            "onset_edf": onset_edf,                          # tempo do EDF (None se sem offset)
            "duration_s": round(float(ev["duration_s"]), 1),
            "score": round(float(ev["score"]), 3),
            "epoch_start": e0, "epoch_end": e1,
            "stage": STAGE_NAMES.get(dom, "?"),
            "suggestion": suggestion,
        })
    return out


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass  # silencioso

    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode("utf-8")
        elif isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        try:
            if u.path in ("/", "/index.html"):
                html = (HERE / "index.html").read_text(encoding="utf-8")
                return self._send(200, html, "text/html; charset=utf-8")

            if u.path == "/api/exams":
                exams = sorted(p.stem for p in CFG["data_dir"].glob("*.pt"))
                return self._send(200, {"exams": exams})

            if u.path == "/api/exam":
                name = q["name"][0]
                st = _prepare(name)
                return self._send(200, {
                    "subject_id": st["subject_id"],
                    "n_epochs": st["n_epochs"],
                    "hours": round(st["hours"], 2),
                    "threshold": st["threshold"],
                    "fs": FS, "epoch_sec": EPOCH_SEC,
                    "n_events": len(st["events"]),
                    "annot_start": st.get("annot_start"),
                    "has_offset": st.get("annot_start") is not None,
                    "has_mat": st.get("has_mat", False),
                    "hipno_start": _sec_to_hms(st.get("hipno_start")),
                    "meas_date": st.get("meas_date"),
                    "events": _events_payload(st),
                })

            if u.path == "/api/signal":
                name = q["name"][0]
                t0 = float(q.get("t0", ["0"])[0])
                t1 = float(q.get("t1", ["30"])[0])
                st = _prepare(name)
                emg = st["emg"].reshape(-1)  # [T*300] em ordem temporal
                fs = FS
                i0 = max(0, int(t0 * fs)); i1 = min(len(emg), int(t1 * fs))
                seg = emg[i0:i1]
                # downsample p/ no maximo ~4000 pontos (plot leve)
                maxpts = 4000
                if len(seg) > maxpts:
                    step = int(np.ceil(len(seg) / maxpts))
                    seg = seg[::step]
                else:
                    step = 1
                # estagios por mini-epoca na janela
                e0 = int(t0 // EPOCH_SEC); e1 = int(np.ceil(t1 / EPOCH_SEC))
                stages = [STAGE_NAMES.get(int(s), "?") for s in st["stages"][e0:e1]]
                movemask = [bool(x) for x in st["mask"][e0:e1]]
                return self._send(200, {
                    "t0": t0, "t1": t1, "fs_eff": fs / step,
                    "samples": seg.round(3).tolist(),
                    "epoch_start": e0,
                    "stages": stages, "movemask": movemask,
                })
        except Exception as e:
            import traceback
            return self._send(500, {"error": str(e), "trace": traceback.format_exc()})
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        u = urlparse(self.path)
        n = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(n) if n else b"{}"
        try:
            payload = json.loads(raw or b"{}")
            if u.path == "/api/config":
                # Grava meas_date do EDF e recalcula annot_start para o exame.
                name = payload["exam"]
                meas_date = payload.get("meas_date")  # 'HH:MM:SS' ou None p/ limpar
                meas_sec = _hms_to_sec(meas_date)
                annot_start = _compute_annot_start(name, meas_sec) if meas_sec is not None else None
                hipno_start = _mat_hipno_start_sec(name)
                cfg = _load_config()
                if meas_date:
                    cfg[name] = {"meas_date": meas_date, "annot_start": annot_start,
                                 "hipno_start": _sec_to_hms(hipno_start)}
                else:
                    cfg.pop(name, None)
                _save_config(cfg)
                # atualiza cache em memoria (sem re-rodar o modelo)
                if name in _CACHE:
                    _CACHE[name]["annot_start"] = annot_start
                    _CACHE[name]["meas_date"] = meas_date
                return self._send(200, {
                    "exam": name, "meas_date": meas_date,
                    "hipno_start": _sec_to_hms(hipno_start),
                    "annot_start": annot_start,
                    "has_offset": annot_start is not None,
                    "has_mat": hipno_start is not None,
                })

            if u.path == "/api/save":
                name = payload["exam"]
                decisions = payload["decisions"]  # [{onset_s,duration_s,label,score}]
                st = _prepare(name)
                a0 = st.get("annot_start")
                CFG["out_dir"].mkdir(parents=True, exist_ok=True)
                out = CFG["out_dir"] / f"{name}_revisado.csv"
                kept = [d for d in decisions if d["label"] in ("tonic", "phasic")]
                # CSV no tempo do EDF: onset_edf = annot_start + onset_pt.
                # Sem offset conhecido, salva no tempo do .pt e avisa.
                for d in kept:
                    d["onset_out"] = round(float(d["onset_s"]) + a0, 3) if a0 is not None else round(float(d["onset_s"]), 3)
                with open(out, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=["subject_id", "onset_s", "duration_s", "type", "score"])
                    w.writeheader()
                    for d in sorted(kept, key=lambda x: x["onset_out"]):
                        w.writerow({"subject_id": name, "onset_s": d["onset_out"],
                                    "duration_s": round(float(d["duration_s"]), 3), "type": d["label"],
                                    "score": d["score"]})
                return self._send(200, {"saved": str(out), "n_kept": len(kept),
                                        "n_deleted": len(decisions) - len(kept),
                                        "time_ref": "edf" if a0 is not None else "pt",
                                        "annot_start": a0})
        except Exception as e:
            import traceback
            return self._send(500, {"error": str(e), "trace": traceback.format_exc()})
        return self._send(404, {"error": "not found"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(CFG["data_dir"]))
    ap.add_argument("--model", default=str(CFG["model_path"]))
    ap.add_argument("--out", default=str(CFG["out_dir"]))
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    CFG["data_dir"] = Path(args.data)
    CFG["model_path"] = Path(args.model)
    CFG["out_dir"] = Path(args.out)
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Revisor de movimento em http://localhost:{args.port}  (Ctrl+C para parar)")
    print(f"  dados : {CFG['data_dir']}")
    print(f"  modelo: {CFG['model_path']}")
    print(f"  saida : {CFG['out_dir']}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nencerrado.")


if __name__ == "__main__":
    main()
