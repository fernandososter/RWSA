"""
preprocess.py — Pre-processamento de exames PSG (EDF -> tensores PyTorch).

Convertido do notebook Parser_Exames (celula 12), com DUAS mudancas em relacao
ao notebook:

  1. Ordem de canais corrigida (ver sleep_rswa/preprocessing/config.py):
     [3 EEG, EOG, EMG] para casar com src/sleep_rswa/config.py
     (staging=(0,1,2,3), emg_channel_index=4).

  2. Rasterizacao de RSWA integrada: apos expandir os estagios para mini-epocas
     de 3 s, os eventos do CSV (<subject>_rswa.csv), a rota automatica
     (CNN de movimento + limiar duplo, rswa_source="auto") OU a regra textual
     da AASM (2023a) (rswa_source="aasm", ver aasm_rule.py -- EXPERIMENTAL,
     NAO usar em treinamento sem recalibrar aasm_atonia_pct primeiro, ver
     docs/relatorio_impacto_regra_aasm.md secao 4/5) sao convertidos em
     rotulos por mini-epoca (tonic_labels, phasic_labels, rswa_labels,
     rswa_conf) e gravados no .pt.

  3. Basal de EMG na fase REM (rem_baseline_uv): percentil 10 do envelope RMS
     do EMG dentro das mini-epocas REM, em microvolts BRUTOS (sem
     normalizacao -- ver rem_baseline.py). Adicionado em 2026-08-06 para uso
     futuro por detectores de eventos tonico/any/fasico; por ora e SO
     gravado no .pt, nenhum detector foi alterado para consumi-lo.

Etapas
──────
1. Carrega EDF + hipnograma (.mat) alinhado + CSV de RSWA (se rswa_source=csv)
2. Resolve canais (ausentes -> zeros + channel_mask=False)
3. Constroi stage_map, cropa o raw em [annot_start, annot_end]
4. Filtra por tipo (EMG/EEG/EOG) + notch, reamostra para 100 Hz
5. Epocas de 30s -> stages por epoca
6. Zero-fill dos canais ausentes -> matriz (n_epochs, N_CHANNELS, 3000)
7. Sub-segmenta 30s -> mini-epocas de 3s (300 amostras)
8. Expande stages 30s -> mini-epocas (np.repeat)
9. Rasteriza eventos RSWA -> rotulos por mini-epoca (alinhados por annot_start)
10. Calcula rem_baseline_uv (percentil 10 do envelope RMS do EMG em REM)

Formato salvo (torch.save)
──────────────────────────
{
  "signals":       Tensor (T, N_CHANNELS, 300)  float32
  "sleep_stages":  Tensor (T,)                  int64   (-1 = gap)
  "channel_mask":  Tensor (N_CHANNELS,)         bool
  "channel_names": list[str | None]
  "tonic_labels":  Tensor (T,)  float32  {0,1}
  "phasic_labels": Tensor (T,)  float32  {0,1}
  "any_labels":    Tensor (T,)  float32  {0,1}  (evento com duracao ambigua,
                                                  entre limiar fasico e minimo tonico)
  "rswa_labels":   Tensor (T,)  int64    {0,1,2,3}  (NAO inclui "any")
  "rswa_conf":     Tensor (T,)  float32  {0,1}  (validade p/ mascara da loss)
  "tonic_cov":     Tensor (T,)  float32  0..1   (fracao de cobertura, diagnostico)
  "phasic_cov":    Tensor (T,)  float32  0..1
  "any_cov":       Tensor (T,)  float32  0..1
  "label_metadata": dict        (metadados da origem dos labels; inclui
                                 parametros do auto-label quando rswa_source=auto)
  "rem_baseline_uv":       float  (uV brutos; NaN se exame sem mini-epoca REM)
  "rem_baseline_n_epochs": int    (quantas mini-epocas REM entraram no calculo)
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
    ECG_GATE_DEFAULT,
    ECG_GATE_WINDOW_PRE_S,
    ECG_GATE_WINDOW_POST_S,
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
from .auto_rswa import DEFAULT_AUTO_LABEL_MODEL, auto_label_rswa_from_signals
from .rswa_labels import rasterize_rswa_annotations
from .rem_baseline import compute_rem_baseline
from .aasm_rule import (
    label_exam_with_aasm_rule,
    AASM_CALIBRATION_WARNING,
    MIN_AMPLITUDE_RATIO as AASM_MIN_AMPLITUDE_RATIO,
)
from .ecg_gating import apply_ecg_gating_to_raw


def preprocess_exam(
    edf_path: Path,
    fs_target: int = FS_TARGET,
    epoch_sec: int = EPOCH_SEC,
    verbose: bool = True,
    *,
    mat_dir: Optional[Path] = None,
    rswa_dir: Optional[Path] = None,
    rswa_source: str = "auto",
    auto_label_model_path: str | Path = DEFAULT_AUTO_LABEL_MODEL,
    auto_label_device: str = "cpu",
    auto_label_cnn_threshold: float | None = None,
    auto_label_cnn_min_epochs: int = 1,
    auto_label_k_on: float = 3.0,
    auto_label_k_off: float = 1.5,
    auto_label_k_off_hold_s: float = 0.0,
    tonic_min_coverage: float = 0.5,
    phasic_min_coverage: float = 0.0,
    any_min_coverage: float = 0.0,
    aasm_atonia_pct: float | None = None,
    aasm_min_amplitude_ratio: float = AASM_MIN_AMPLITUDE_RATIO,
    ecg_gate: bool = ECG_GATE_DEFAULT,
    ecg_gate_window_pre_s: float = ECG_GATE_WINDOW_PRE_S,
    ecg_gate_window_post_s: float = ECG_GATE_WINDOW_POST_S,
) -> Optional[Dict]:
    """
    Pre-processa um unico exame EDF. Retorna dict (ver formato no topo do modulo)
    ou None em falha irrecuperavel.
    """
    import mne
    mne.set_log_level("WARNING")

    edf_path = Path(edf_path)
    subject_id = edf_path.stem
    rswa_source = rswa_source.strip().lower()
    if rswa_source not in {"csv", "auto", "aasm"}:
        raise ValueError(f"rswa_source invalido: {rswa_source!r}. Use 'csv', 'auto' ou 'aasm'.")

    mat_dir = Path(mat_dir) if mat_dir is not None else PathConfig.MAT_DIR
    rswa_dir = Path(rswa_dir) if rswa_dir is not None else PathConfig.RSWA_DIR
    if auto_label_model_path is None:
        auto_label_model_path = DEFAULT_AUTO_LABEL_MODEL

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

        if rswa_source == "csv":
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
                print(f"[CSV ANNOTATIONS] arquivo nao encontrado para {subject_id}")
        else:
            raw.set_annotations(annotations, emit_warning=False)
            print(f"[RSWA SOURCE] {rswa_source} — CSV nao sera usado")

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
    # Se ecg_gate=True, inclui tambem o canal de ECG (se presente no EDF) no
    # pick -- ele e usado apenas para deteccao de picos-R (etapa 3.5) e
    # removido antes da matriz final de sinais, nunca chega ao .pt.
    present_chs_ordered = [ch for ch in matched_chs if ch is not None]
    ecg_ch_raw_name = None
    if ecg_gate:
        from .channels import find_channel
        ecg_ch_raw_name = find_channel(raw.ch_names, PSGConfig.ECG_CANDIDATES)
    pick_list = present_chs_ordered + (
        [ecg_ch_raw_name] if ecg_ch_raw_name is not None else []
    )
    raw.pick(pick_list)
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

    # ── 3.5. Gating de artefato cardiaco no EMG (opcional, via ECG) ───────
    # Deve rodar ANTES do filtro passa-banda de EMG (etapa 4) para que o
    # sinal gated (e nao o original com espiculas de QRS) seja o que entra
    # no filtro/reamostragem/epocamento. Deteccao de picos-R usa o ECG
    # ainda bruto (sem filtro), o que preserva melhor a morfologia do QRS.
    # Ver ecg_gating.py e docs/relatorio_impacto_regra_aasm.md Secao 12.
    emg_defn_idx = next(
        (i for i, d in enumerate(PSGConfig.CHANNEL_DEFS) if d["name"] == "emg"), None
    )
    emg_ch_name = matched_chs[emg_defn_idx] if emg_defn_idx is not None else None
    ecg_gate_diag = {
        "ecg_gate_enabled": bool(ecg_gate),
        "ecg_gate_applied": False,
        "ecg_channel_found": None,
        "n_r_peaks": 0,
        "n_gated_samples": 0,
        "frac_gated": 0.0,
        "window_pre_s": float(ecg_gate_window_pre_s),
        "window_post_s": float(ecg_gate_window_post_s),
        "reason_skipped": "ecg_gate_disabled" if not ecg_gate else None,
    }
    if ecg_gate:
        ecg_gate_diag_result = apply_ecg_gating_to_raw(
            raw,
            emg_ch_name=emg_ch_name,
            ecg_candidates=PSGConfig.ECG_CANDIDATES,
            window_pre_s=ecg_gate_window_pre_s,
            window_post_s=ecg_gate_window_post_s,
            verbose=verbose,
        )
        ecg_gate_diag.update(ecg_gate_diag_result)
        ecg_gate_diag["ecg_gate_enabled"] = True

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

    # ── 10. Basal de EMG na fase REM (percentil 10, uV brutos) ────────────
    # Ver docstring de rem_baseline.py: nao usa tonic_labels/phasic_labels
    # (funciona igual em exames revisados e nao revisados); nao altera
    # nenhum detector -- so grava o valor no .pt para uso futuro. Movido
    # para ANTES da geracao de rotulos porque rswa_source="aasm" precisa
    # deste valor como referencia de amplitude (nivel de atonia REM).
    rem_baseline = compute_rem_baseline(signals, stages_mini)
    if verbose:
        print(f" [REM BASELINE] {rem_baseline['rem_baseline_uv']:.2f} uV "
              f"(n_mini_epocas_rem={rem_baseline['rem_baseline_n_epochs']})")

    # ── 11. Gera rotulos RSWA -> mini-epocas ───────────────────────────────
    if rswa_source == "csv":
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
            any_min_coverage=any_min_coverage,
        )
    elif rswa_source == "aasm":
        # ATENCAO -- ver AASM_CALIBRATION_WARNING (aasm_rule.py) e
        # docs/relatorio_impacto_regra_aasm.md secao 4: com o percentil de
        # atonia atual (10.0, herdado de rem_baseline_uv) o criterio de
        # amplitude satura phasic/any. NAO usar em treinamento sem antes
        # recalibrar `aasm_atonia_pct` e revalidar contra CSVs revisados.
        if verbose:
            print(f" [AASM RSWA][AVISO] {AASM_CALIBRATION_WARNING}")
        rswa = label_exam_with_aasm_rule(
            signals,
            stages_mini,
            rem_baseline["rem_baseline_uv"],
            rem_baseline["rem_baseline_n_epochs"],
            atonia_pct=aasm_atonia_pct,
            min_amplitude_ratio=aasm_min_amplitude_ratio,
        )
    else:
        rswa = auto_label_rswa_from_signals(
            signals,
            stages_mini,
            model_path=auto_label_model_path,
            device=auto_label_device,
            cnn_threshold=auto_label_cnn_threshold,
            cnn_min_epochs=auto_label_cnn_min_epochs,
            k_on=auto_label_k_on,
            k_off=auto_label_k_off,
            k_off_hold_s=auto_label_k_off_hold_s,
            tonic_min_coverage=tonic_min_coverage,
            phasic_min_coverage=phasic_min_coverage,
            any_min_coverage=any_min_coverage,
        )

    if verbose:
        n_rem = int((stages_mini == 4).sum())
        n_gap = int((stages_mini == -1).sum())
        n_tonic = int(rswa["tonic_labels"].sum())
        n_phasic = int(rswa["phasic_labels"].sum())
        print(f" [ALINHAMENTO] {len(signals)} mini-epocas | REM={n_rem} | "
              f"gap={n_gap} | tonic+={n_tonic} | phasic+={n_phasic}")
        if rswa_source == "auto":
            print(
                f" [AUTO RSWA] candidatos_cnn={rswa['n_cnn_candidates']} "
                f"confirmados={rswa['n_confirmed_events']} "
                f"descartados={rswa['n_discarded_windows']} "
                f"limiar_cnn={rswa['cnn_threshold']:.3f} "
                f"k_on={rswa['k_on']:.3f} "
                f"k_off={rswa['k_off']:.3f} "
                f"k_off_hold_s={rswa['k_off_hold_s']:.3f}"
            )
        elif rswa_source == "aasm":
            print(
                f" [AASM RSWA] n_rem_macro_epochs={rswa['n_rem_macro_epochs']} "
                f"n_tonic_macro_epochs={rswa['n_tonic_macro_epochs']} "
                f"n_phasic_macro_epochs={rswa['n_phasic_macro_epochs']} "
                f"atonia_source={rswa['atonia_source']} "
                f"atonia_baseline_uv={rswa['atonia_baseline_uv']:.3f} "
                f"atonia_pct_used={rswa['atonia_pct_used']:.1f}"
            )

    label_source = rswa.get("label_source", "csv_rswa_annotations_v1")
    label_metadata = {
        "rswa_source": rswa_source,
        "label_source": label_source,
        "coverage_thresholds": {
            "tonic_min_coverage": float(tonic_min_coverage),
            "phasic_min_coverage": float(phasic_min_coverage),
            "any_min_coverage": float(any_min_coverage),
        },
        "ecg_gate": ecg_gate_diag,
    }
    if rswa_source == "csv":
        label_metadata["csv_annotations_path"] = str(rswa_csv_path) if rswa_csv_path is not None else None
    elif rswa_source == "aasm":
        label_metadata["aasm_rule"] = {
            "atonia_baseline_uv": float(rswa["atonia_baseline_uv"]),
            "atonia_source": rswa["atonia_source"],
            "atonia_pct_used": float(rswa["atonia_pct_used"]),
            "n_rem_macro_epochs": int(rswa["n_rem_macro_epochs"]),
            "n_tonic_macro_epochs": int(rswa["n_tonic_macro_epochs"]),
            "n_phasic_macro_epochs": int(rswa["n_phasic_macro_epochs"]),
            "min_amplitude_ratio_used": float(rswa["min_amplitude_ratio_used"]),
            "tonic_support_schema": "tonic_coverage_s / 30.0 repetido nas 10 mini-epocas REM da macro-epoca",
            "calibration_warning": AASM_CALIBRATION_WARNING,
        }
    else:
        label_metadata["auto_label"] = {
            "model_path": str(Path(auto_label_model_path)),
            "device": str(auto_label_device),
            "cnn_threshold": float(rswa["cnn_threshold"]),
            "cnn_min_epochs": int(auto_label_cnn_min_epochs),
            "k_on": float(rswa["k_on"]),
            "k_off": float(rswa["k_off"]),
            "k_off_hold_s": float(rswa["k_off_hold_s"]),
            "n_cnn_candidates": int(rswa["n_cnn_candidates"]),
            "n_confirmed_events": int(rswa["n_confirmed_events"]),
            "n_discarded_windows": int(rswa["n_discarded_windows"]),
        }

    return {
        "subject_id":    subject_id,
        "signals":       signals.astype(np.float32),
        "sleep_stages":  stages_mini.astype(np.int64),
        "channel_mask":  np.array(ch_mask, dtype=bool),
        "channel_names": matched_chs,
        "tonic_labels":  rswa["tonic_labels"],
        "phasic_labels": rswa["phasic_labels"],
        "any_labels":    rswa["any_labels"],
        "tonic_support": rswa.get("tonic_support"),
        "rswa_labels":   rswa["rswa_labels"],
        "rswa_conf":     rswa["rswa_conf"],
        "tonic_cov":     rswa["tonic_cov"],
        "phasic_cov":    rswa["phasic_cov"],
        "any_cov":       rswa["any_cov"],
        "fs":            fs_target,
        "label_source":  label_source,
        "label_metadata": label_metadata,
        "rem_baseline_uv":       rem_baseline["rem_baseline_uv"],
        "rem_baseline_n_epochs": rem_baseline["rem_baseline_n_epochs"],
    }


def _save_result(result: Dict, out_path: Path) -> None:
    """Grava o dict de preprocess_exam como .pt (inclui rotulos de RSWA)."""
    import torch

    payload = {
        "signals":       torch.from_numpy(result["signals"]),
        "sleep_stages":  torch.from_numpy(result["sleep_stages"]),
        "channel_mask":  torch.from_numpy(result["channel_mask"]),
        "channel_names": result["channel_names"],
        "tonic_labels":  torch.from_numpy(result["tonic_labels"]),
        "phasic_labels": torch.from_numpy(result["phasic_labels"]),
        "any_labels":    torch.from_numpy(result["any_labels"]),
        "rswa_labels":   torch.from_numpy(result["rswa_labels"]),
        "rswa_conf":     torch.from_numpy(result["rswa_conf"]),
        "tonic_cov":     torch.from_numpy(result["tonic_cov"]),
        "phasic_cov":    torch.from_numpy(result["phasic_cov"]),
        "any_cov":       torch.from_numpy(result["any_cov"]),
        "label_source":  result["label_source"],
        "label_metadata": result["label_metadata"],
        # Basal de EMG na fase REM (percentil 10, uV brutos) -- ver
        # rem_baseline.py. Escalares Python simples (nao Tensor) porque sao
        # um unico valor por exame, nao uma serie por mini-epoca; NaN quando
        # o exame nao tem nenhuma mini-epoca REM (n_rem_epochs == 0).
        "rem_baseline_uv":       result["rem_baseline_uv"],
        "rem_baseline_n_epochs": result["rem_baseline_n_epochs"],
    }
    if result.get("tonic_support") is not None:
        payload["tonic_support"] = torch.from_numpy(result["tonic_support"])
    torch.save(payload, out_path)


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
