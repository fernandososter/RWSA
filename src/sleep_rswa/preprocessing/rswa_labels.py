"""
Rasterizacao de eventos RSWA (intervalo -> rotulo por mini-epoca de 3 s).

Chamar dentro de preprocess_exam, LOGO APOS a expansao de estagios
(np.repeat(stages_30s, n_mini_per_epoch)) e ANTES do dict de retorno.

IMPORTANTE sobre rswa_conf:
    No loader (data.py), a mascara da loss e `valid_rswa = rswa_conf > min_confidence`.
    Portanto rswa_conf indica VALIDADE (a mini-epoca foi escorada e e usavel),
    NAO a cobertura do evento. Mini-epocas negativas (REM sem evento) DEVEM ter
    conf=1.0 para entrarem na loss como negativos — senao o modelo so ve positivos.
    Usamos conf=1.0 onde stage != -1 (escorada) e 0.0 nos gaps.
"""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np


def rasterize_rswa_annotations(
    csv_path,
    subject_id: str,
    stages_mini: np.ndarray,          # (T,) int64 — estagios ja expandidos p/ 3s
    annot_start: float,
    *,
    epoch_sec: float = 3.0,
    tonic_min_coverage: float = 0.5,
    phasic_min_coverage: float = 0.0,   # >0 => qualquer presenca
    tonic_priority: bool = True,        # se tonico e fasico caem na mesma mini-epoca
) -> dict[str, np.ndarray]:
    """
    Converte eventos (onset_s, duration_s, type) do CSV em rotulos por mini-epoca.

    Onsets no CSV estao no referencial do EDF BRUTO. O sinal foi cropado em
    `annot_start`; a mini-epoca m cobre [annot_start+m*dt, annot_start+(m+1)*dt).

    Retorna arrays de comprimento T = len(stages_mini):
        tonic_labels, phasic_labels : (T,) float32  {0,1}  (multi-rotulo, p/ 2 cabecas BCE)
        rswa_labels                 : (T,) int64     {0=nada,1=fasico,2=tonico}
                                      (mono-rotulo, compativel com data.py atual;
                                       co-ocorrencia resolvida por tonic_priority)
        rswa_conf                   : (T,) float32   {0,1}  VALIDADE (1=escorada)
        tonic_cov, phasic_cov       : (T,) float32   fracao 0..1 (diagnostico / soft target)
    """
    csv_path = Path(csv_path) if csv_path is not None else None
    T = int(len(stages_mini))
    tonic_cov  = np.zeros(T, dtype=np.float64)
    phasic_cov = np.zeros(T, dtype=np.float64)

    if csv_path is not None and csv_path.exists():
        with csv_path.open("r", encoding="utf-8-sig", newline="") as fh:
            for row in csv.DictReader(fh):
                if str(row.get("subject_id", "")).strip() != subject_id:
                    continue
                etype = str(row.get("type", "")).strip().lower()
                try:
                    onset = float(row["onset_s"]); dur = float(row["duration_s"])
                except (TypeError, ValueError, KeyError):
                    continue
                start = onset - annot_start
                end   = start + dur
                if end <= 0 or dur <= 0:
                    continue
                start = max(0.0, start)
                first = max(0, int(start // epoch_sec))
                last  = min(T - 1, int((end - 1e-9) // epoch_sec))
                for m in range(first, last + 1):
                    m0, m1 = m * epoch_sec, (m + 1) * epoch_sec
                    frac = max(0.0, min(end, m1) - max(start, m0)) / epoch_sec
                    if etype == "tonic":
                        tonic_cov[m]  = min(1.0, tonic_cov[m]  + frac)
                    elif etype == "phasic":
                        phasic_cov[m] = min(1.0, phasic_cov[m] + frac)

    tonic_lab  = (tonic_cov  >= tonic_min_coverage).astype(np.float32)
    phasic_lab = (phasic_cov >  phasic_min_coverage).astype(np.float32)

    # rotulo inteiro informativo: 0=nada, 1=fasico, 2=tonico, 3=ambos.
    # As cabecas do modelo usam tonic_labels/phasic_labels diretamente (multi-
    # rotulo); rswa_int e so p/ inspecao e p/ compat com .pt mono-rotulo antigos.
    rswa_int = (phasic_lab.astype(np.int64) * 1) + (tonic_lab.astype(np.int64) * 2)
    _ = tonic_priority  # mantido por compat de assinatura; co-ocorrencia agora preservada

    # VALIDADE: 1.0 onde a mini-epoca foi escorada (stage != -1), senao 0.0
    rswa_conf = (stages_mini != -1).astype(np.float32)

    return {
        "tonic_labels":  tonic_lab,
        "phasic_labels": phasic_lab,
        "rswa_labels":   rswa_int,
        "rswa_conf":     rswa_conf,
        "tonic_cov":     tonic_cov.astype(np.float32),
        "phasic_cov":    phasic_cov.astype(np.float32),
    }
