"""
preprocess.py — Pre-processamento de exames PSG (EDF -> tensores PyTorch).

Convertido do notebook Parser_Exames (celula 12), com DUAS mudancas em relacao
ao notebook:

  1. Ordem de canais corrigida (ver sleep_rswa/preprocessing/config.py):
     [3 EEG, EOG, EMG] para casar com src/sleep_rswa/config.py
     (staging=(0,1,2,3), emg_channel_index=4).

  2. Rasterizacao de RSWA integrada: apos expandir os estagios para mini-epocas
     de 3 s, os eventos do CSV (<subject>_rswa.csv) sao convertidos em rotulos
     por mini-epoca (tonic_labels, phasic_labels, rswa_labels, rswa_conf) e
     gravados no .pt. Era o elo faltante — o notebook carregava os eventos nas
     annotations do MNE mas nunca os transformava em rotulos.

Etapas
──────
1. Carrega EDF + hipnograma (.mat) alinhado + CSV de RSWA (se houver)
2. Resolve canais (ausentes -> zeros + channel_mask=False)
3. Constroi stage_map, cropa o raw em [annot_start, annot_end]
4. Filtra por tipo (EMG/EEG/EOG) + notch, reamostra para 100 Hz
5. Epocas de 30s -> stages por epoca
6. Zero-fill dos canais ausentes -> matriz (n_epochs, N_CHANNELS, 3000)
7. Sub-segmenta 30s -> mini-epocas de 3s (300 amostras)
8. Expande stages 30s -> mini-epocas (np.repeat)
9. Rasteriza eventos RSWA -> rotulos por mini-epoca (alinhados por annot_start)

Formato salvo (torch.save)
──────────────────────────
{
  "signals":       Tensor (T, N_CHANNELS, 300)  float32
  "sleep_stages":  Tensor (T,)                  int64   (-1 = gap)
  "channel_mask":  Tensor (N_CHANNELS,)         bool
  "channel_names": list[str | None]
  "tonic_labels":  Tensor (T,)  float32  {0,1}
  "phasic_labels": Tensor (T,)  float32  {0,1}
  "rswa_labels":   Tensor (T,)  int64    {0,1,2,3}
  "rswa_conf":     Tensor (T,)  float32  {0,1}  (validade p/ mascara da loss)
}
"""
from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .config import (
    FILTER_PARAMS,
    FS_TARGET,
    EPOCH_SEC,
    N_CHANNELS,
    NOTCH_FILTER,
    PathConfig,
    PSGConfig,
)
from .channels import find_mat_file, list_raw_edfs, resolve_channels
from .hypnogram import _parse_stage_value, load_aligned_hyp_annotations
from .annotations import (
    count_annotations_by_description,
    find_annotations_csv_file,
    load_subject_annotations_from_csv,
)
from .rswa_labels import rasterize_rswa_annotations


def preprocess_exam(
    edf_path: Path,
    fs_target: int = FS_TARGET,
    epoch_sec: int = EPOCH_SEC,
    verbose: bool = True,
    *,
    mat_dir: Optional[Path] = None,
    rswa_dir: Optional[Path] = None,
    tonic_min_coverage: float = 0.5,
    phasic_min_coverage: float = 0.0,
) -> Optional[Dict]:
    """
    Pre-processa um unico exame EDF. Retorna dict (ver formato no topo do modulo)
    ou None em falha irrecuperavel.
    """
    import mne
    mne.set_log_level("WARNING")

    edf_path = Path(edf_path)
    subject_id = edf_path.stem

    mat_dir = Path(mat_dir) if mat_dir is not None else PathConfig.MAT_DIR
    rswa_dir = Path(rswa_dir) if rswa_dir is not None else PathConfig.RSWA_DIR

    # ── 1. Carrega EDF + hipnograma + CSV de RSWA ─────────────────────────
    rswa_csv_path = None
    try:
        scores_path = find_mat_file(subject_id, mat_dir)
        if scores_path is None:
            print(f"  [ERRO] {subject_id}: hipnograma nao encontrado em {mat_dir}")
            return None
        print(f"[SCORE FILE]: {scores_path}")

        raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose="ERROR")

        annotations, hyp, alignment = load_aligned_hyp_annotations(
            raw=raw, mat_path=scores_path, include_movement=False,
        )

        rswa_csv_path = find_annotations_csv_file(
            subject_id=subject_id, csv_dir=rswa_dir,
        )
        if rswa_csv_path is not None:
            print(f"[CSV ANNOTATIONS] Encontrou {rswa_csv_path}")
            new = load_subject_annotations_from_csv(
                csv_path=rswa_csv_path,
                subject_id=subject_id,
                orig_time=annotations.orig_time,
            )
            if len(new):
                raw.set_annotations(annotations + new, emit_warning=False)
                print(f"[CSV ANNOTATIONS] {len(new)} anotacoes RSWA adicionadas "
                      f"para {subject_id}")
            else:
                raw.set_annotations(annotations, emit_warning=False)
                print(f"[CSV ANNOTATIONS] nenhuma anotacao RSWA para {subject_id}")
        else:
            raw.set_annotations(annotations, emit_warning=False)

        print(f"[ANNOTATION TOTAL] {len(raw.annotations)}")
        print(f"[ANNOTATION COUNTS] {count_annotations_by_description(raw.annotations)}")
    except Exception as error:
        print(f"[ERRO ANOTACAO] {subject_id} - {error}")
        return None

    # ── 2. Resolve canais ─────────────────────────────────────────────────
    matched_chs, ch_mask = resolve_channels(raw.ch_names)
    if sum(ch_mask) == 0:
        if verbose:
            print(f"  [SKIP] {subject_id}: nenhum canal reconhecido.")
        return None

    # Forca a ordem dos canais presentes para corresponder a CHANNEL_DEFS.
    present_chs_ordered = [ch for ch in matched_chs if ch is not None]
    raw.pick(present_chs_ordered)
    if verbose:
        print(f"  [DEBUG] Apos pick, ordem: {raw.ch_names}")

    # ── 3. stage_map + crop na janela de staging ──────────────────────────
    raw_stage_map: dict[float, int] = {
        ann["onset"]: s
        for ann in raw.annotations
        if (s := _parse_stage_value(ann["description"])) is not None
    }

    if not raw_stage_map:
        if verbose:
            print(f"  [SKIP] {subject_id}: nenhum estagio de sono reconhecido.")
        return None

    meas_date = raw.info["meas_date"]
    exam_end = raw.times[-1]
    annot_start = min(raw_stage_map)
    annot_end = max(raw_stage_map) + 30.0

    def _fmt(offset_s):
        if meas_date is not None:
            t = meas_date + timedelta(seconds=float(offset_s))
            return t.strftime("%H:%M:%S")
        return f"+{offset_s:.1f}s"

    if verbose:
        print(f"  [TEMPO] {subject_id}: annot_start={_fmt(annot_start)} "
              f"annot_end={_fmt(annot_end)} exam_end={_fmt(exam_end)}")

    raw.crop(tmin=annot_start, tmax=min(annot_end, exam_end))

    # Reindexa chaves para serem relativas ao novo tmin=0 (indice de epoca 30s).
    stage_map: dict[int, int] = {
        int((onset - annot_start) // 30.0): stage
        for onset, stage in raw_stage_map.items()
    }

    # ── 4. Filtra cada canal presente (in-place) ──────────────────────────
    for defn, ch_name, present in zip(PSGConfig.CHANNEL_DEFS, matched_chs, ch_mask):
        if not present:
            continue
        fp = FILTER_PARAMS[defn["filter"]]
        raw.filter(fp["l_freq"], fp["h_freq"], picks=[ch_name],
                   fir_design="firwin", verbose=False)
        raw.notch_filter(NOTCH_FILTER, picks=[ch_name],
                         method="spectrum_fit", verbose=False)

    # ── 5. Reamostra ──────────────────────────────────────────────────────
    if int(raw.info["sfreq"]) != fs_target:
        raw.resample(fs_target, verbose=False)

    # ── 6. Epocas de 30s ──────────────────────────────────────────────────
    epochs_30s = mne.make_fixed_length_epochs(
        raw, duration=30.0, preload=True, verbose=False,
    )
    if len(epochs_30s) == 0:
        if verbose:
            print(f"  [SKIP] {subject_id}: sinal muito curto para uma epoca de 30s")
        return None

    stages_30s = np.array(
        [stage_map.get(i, -1) for i in range(len(epochs_30s))], dtype=np.int64,
    )

    # ── 7. Reconstroi matriz completa de canais com zero-fill ─────────────
    epoch_data = epochs_30s.get_data().astype(np.float32)
    n_epochs_30s = epoch_data.shape[0]
    n_samples_30s = epoch_data.shape[2]

    full_data = np.zeros((n_epochs_30s, N_CHANNELS, n_samples_30s), dtype=np.float32)
    present_idx = 0
    for i, present in enumerate(ch_mask):
        if present:
            full_data[:, i, :] = epoch_data[:, present_idx, :]
            present_idx += 1

    # ── 8. Sub-segmenta 30s -> mini-epocas de epoch_sec ───────────────────
    n_mini_per_epoch = 30 // epoch_sec          # 10
    n_samples_mini = fs_target * epoch_sec      # 300
    n_channels_out = full_data.shape[1]

    signals = (
        full_data
        .reshape(n_epochs_30s, n_channels_out, n_mini_per_epoch, n_samples_mini)
        .transpose(0, 2, 1, 3)
        .reshape(-1, n_channels_out, n_samples_mini)
    )
    T = len(signals)

    # ── 9. Expande stages 30s -> mini-epocas e alinha comprimentos ────────
    stages_mini = np.repeat(stages_30s, n_mini_per_epoch)
    T_final = min(T, len(stages_mini))
    signals = signals[:T_final]
    stages_mini = stages_mini[:T_final]

    if (stages_mini != -1).sum() == 0:
        if verbose:
            print(f"  [SKIP] {subject_id}: nenhuma mini-epoca com estagio valido.")
        return None

    # ── 10. Rasteriza eventos RSWA -> rotulos por mini-epoca ──────────────
    # Onsets do CSV estao no referencial do EDF bruto; subtraimos annot_start
    # (mesmo crop dos estagios) para alinhar a grade de mini-epocas.
    rswa = rasterize_rswa_annotations(
        csv_path=rswa_csv_path,
        subject_id=subject_id,
        stages_mini=stages_mini,
        annot_start=annot_start,
        epoch_sec=float(epoch_sec),
        tonic_min_coverage=tonic_min_coverage,
        phasic_min_coverage=phasic_min_coverage,
    )

    if verbose:
        n_rem = int((stages_mini == 4).sum())
        n_gap = int((stages_mini == -1).sum())
        n_tonic = int(rswa["tonic_labels"].sum())
        n_phasic = int(rswa["phasic_labels"].sum())
        print(f" [ALINHAMENTO] {len(signals)} mini-epocas | REM={n_rem} | "
              f"gap={n_gap} | tonic+={n_tonic} | phasic+={n_phasic}")

    return {
        "subject_id":    subject_id,
        "signals":       signals.astype(np.float32),
        "sleep_stages":  stages_mini.astype(np.int64),
        "channel_mask":  np.array(ch_mask, dtype=bool),
        "channel_names": matched_chs,
        "tonic_labels":  rswa["tonic_labels"],
        "phasic_labels": rswa["phasic_labels"],
        "rswa_labels":   rswa["rswa_labels"],
        "rswa_conf":     rswa["rswa_conf"],
        "tonic_cov":     rswa["tonic_cov"],
        "phasic_cov":    rswa["phasic_cov"],
        "fs":            fs_target,
    }


def _save_result(result: Dict, out_path: Path) -> None:
    """Grava o dict de preprocess_exam como .pt (inclui rotulos de RSWA)."""
    import torch

    torch.save({
        "signals":       torch.from_numpy(result["signals"]),
        "sleep_stages":  torch.from_numpy(result["sleep_stages"]),
        "channel_mask":  torch.from_numpy(result["channel_mask"]),
        "channel_names": result["channel_names"],
        "tonic_labels":  torch.from_numpy(result["tonic_labels"]),
        "phasic_labels": torch.from_numpy(result["phasic_labels"]),
        "rswa_labels":   torch.from_numpy(result["rswa_labels"]),
        "rswa_conf":     torch.from_numpy(result["rswa_conf"]),
    }, out_path)


def run_preprocessing(
    edf_dir: Path,
    out_dir: Optional[Path] = None,
    overwrite: bool = False,
    verbose: bool = True,
    **kwargs,
) -> List[str]:
    """
    Pre-processa todos os EDFs brutos de edf_dir (serial) e salva .pt em out_dir.
    kwargs extras sao repassados a preprocess_exam (mat_dir, rswa_dir, limiares).
    """
    edf_dir = Path(edf_dir)
    out_dir = Path(out_dir) if out_dir is not None else edf_dir.parent / "tensors"
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_edfs = list_raw_edfs(edf_dir)
    if not raw_edfs:
        print("[preprocess] Nenhum EDF bruto encontrado. Verifique o diretorio.")
        return []

    processed, failed = [], []
    for i, edf_path in enumerate(raw_edfs):
        sid = edf_path.stem
        out_path = out_dir / f"{sid}.pt"

        if out_path.exists() and not overwrite:
            if verbose:
                print(f"[{i+1}/{len(raw_edfs)}] {sid} — ja existe, pulando")
            processed.append(sid)
            continue

        if verbose:
            print("--------------------------------------------------------")
            print(f"[{i+1}/{len(raw_edfs)}] {sid}...")

        result = preprocess_exam(edf_path, verbose=verbose, **kwargs)
        if result is None:
            if verbose:
                print("FALHOU")
            failed.append(sid)
            continue

        _save_result(result, out_path)
        processed.append(sid)

    print(f"\n[preprocess] Concluido: {len(processed)} OK | "
          f"{len(failed)} falharam | {len(raw_edfs)} total")
    if failed:
        print(f"  Falharam: {failed[:10]}{'...' if len(failed) > 10 else ''}")
    return processed
