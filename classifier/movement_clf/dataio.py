"""
Leitura de exames .pt para o detector de movimento (modulo isolado).

Nao importa nada de src/sleep_rswa. Depende apenas do *formato* do .pt:
    dict com:
      signals       Tensor[T, 5, 300]  (3 EEG + EOG + EMG mento; fs=100Hz, 3s/epoca)
      sleep_stages  Tensor[T]          (0=W,1=N1,2=N2,3=N3,4=REM,-1=unscored)
      tonic_labels  Tensor[T] float    (opcional; presente nos exames de treino)
      phasic_labels Tensor[T] float    (opcional; presente nos exames de treino)
    channel_names[4] deve ser o EMG do mento.

Alvo: MOVIMENTO binario por mini-epoca = (tonic OR phasic), a NOITE TODA
(sem mascara de REM), conforme decidido com o usuario.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

FS = 100
EPOCH_SEC = 3.0
SAMPLES_PER_EPOCH = 300
EMG_CHANNEL_INDEX = 4  # canal do EMG mento no tensor signals

STAGE_NAMES = {-1: "unscored", 0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}


@dataclass
class Exam:
    subject_id: str
    emg: np.ndarray          # [T, 300] float32  (EMG bruto, 1 canal)
    stages: np.ndarray       # [T] int64
    movement: np.ndarray     # [T] float32 {0,1}  alvo binario
    path: str

    @property
    def n_epochs(self) -> int:
        return int(self.emg.shape[0])

    @property
    def hours(self) -> float:
        return self.n_epochs * EPOCH_SEC / 3600.0


def _to_np(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def load_exam(path: str | Path, require_labels: bool = True) -> Exam:
    """Carrega um .pt e devolve um Exam com EMG e rotulo de movimento.

    Se require_labels=False (inferencia em exame novo), movimento vem tudo zero.
    """
    path = Path(path)
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict):
        raise TypeError(f"{path}: esperado dict, recebido {type(obj)!r}")

    signals = _to_np(obj["signals"]).astype(np.float32)   # [T,C,N]
    if signals.ndim != 3:
        raise ValueError(f"{path}: signals deve ser [T,C,N], recebeu {signals.shape}")
    T, C, N = signals.shape
    if C <= EMG_CHANNEL_INDEX:
        raise ValueError(f"{path}: signals tem {C} canais; EMG esperado no indice {EMG_CHANNEL_INDEX}")
    emg = signals[:, EMG_CHANNEL_INDEX, :SAMPLES_PER_EPOCH].copy()

    stages = _to_np(obj.get("sleep_stages", np.full(T, -1))).astype(np.int64)

    tonic = obj.get("tonic_labels")
    phasic = obj.get("phasic_labels")
    if tonic is None and phasic is None:
        if require_labels:
            raise ValueError(
                f"{path}: sem tonic_labels/phasic_labels. Exame nao anotado "
                f"nao pode ser usado no treino."
            )
        movement = np.zeros(T, dtype=np.float32)
    else:
        tonic = _to_np(tonic) if tonic is not None else np.zeros(T)
        phasic = _to_np(phasic) if phasic is not None else np.zeros(T)
        movement = ((tonic > 0.5) | (phasic > 0.5)).astype(np.float32)

    subject_id = str(obj.get("subject_id", path.stem))
    return Exam(subject_id=subject_id, emg=emg, stages=stages, movement=movement, path=str(path))


def load_dir(directory: str | Path, require_labels: bool = True) -> list[Exam]:
    paths = sorted(Path(directory).glob("*.pt"))
    if not paths:
        raise FileNotFoundError(f"Nenhum .pt em {directory}")
    return [load_exam(p, require_labels=require_labels) for p in paths]


def zscore_emg(emg: np.ndarray) -> np.ndarray:
    """Z-score por exame (estatisticas globais do EMG do exame)."""
    mu = emg.mean()
    sd = emg.std()
    sd = sd if sd > 1e-8 else 1.0
    return (emg - mu) / sd


def qc_table(exams: list[Exam]) -> list[dict]:
    """Estatisticas de QC por exame."""
    rows = []
    for ex in exams:
        st = ex.stages
        row = {
            "subject_id": ex.subject_id,
            "hours": round(ex.hours, 2),
            "n_epochs": ex.n_epochs,
            "n_movement": int(ex.movement.sum()),
            "prevalence_pct": round(100 * float(ex.movement.mean()), 2),
        }
        for k in (0, 1, 2, 3, 4, -1):
            row[f"pct_{STAGE_NAMES[k]}"] = round(100 * float((st == k).mean()), 1)
        rem = st == 4
        mov = ex.movement > 0.5
        row["mov_in_REM"] = int((mov & rem).sum())
        row["mov_out_REM"] = int((mov & ~rem).sum())
        rows.append(row)
    return rows


def events_from_binary(mask: np.ndarray, scores: np.ndarray | None = None,
                       subject_id: str = "", etype: str = "movement") -> list[dict]:
    """Funde mini-epocas positivas adjacentes em eventos (onset_s, duration_s).

    mask: [T] bool/0-1. scores: [T] float opcional (score medio do evento).
    Onsets no referencial do tensor (mini-epoca m -> [m*3, (m+1)*3) s).
    """
    mask = np.asarray(mask).astype(bool)
    T = len(mask)
    events = []
    i = 0
    while i < T:
        if mask[i]:
            j = i
            while j + 1 < T and mask[j + 1]:
                j += 1
            onset = i * EPOCH_SEC
            duration = (j - i + 1) * EPOCH_SEC
            ev = {
                "subject_id": subject_id,
                "onset_s": round(float(onset), 3),
                "duration_s": round(float(duration), 3),
                "type": etype,
            }
            if scores is not None:
                ev["score"] = round(float(np.mean(scores[i:j + 1])), 4)
            events.append(ev)
            i = j + 1
        else:
            i += 1
    return events
