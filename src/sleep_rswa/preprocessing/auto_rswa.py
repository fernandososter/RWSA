"""
Rotulagem RSWA automática no pré-processamento: CNN de movimento + limiar duplo.

Este módulo é autocontido dentro de ``sleep_rswa.preprocessing``: não importa
nenhuma lógica clínica de ``testes/``. A regra do limiar duplo/histerese foi
copiada e adaptada do experimento validado em ``testes/src/limiar``.

Fluxo:
  1. Extrai o EMG do tensor já pré-processado em mini-épocas de 3 s.
  2. Z-score global por exame e inferência com a CNN binária de movimento.
  3. Funde mini-épocas positivas adjacentes em janelas candidatas.
  4. Dentro de cada janela candidata, roda o limiar duplo no EMG bruto.
  5. Classifica os segmentos confirmados em phasic / any / tonic por duração
     e amplitude, e converte em cobertura/rótulo por mini-época.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn


FS = 100
EPOCH_SEC = 3.0
SAMPLES_PER_EPOCH = 300

BASELINE_WIN_S = 120.0
BASELINE_PCT = 10
MERGE_GAP_S = 1.0
K_ON = 3.0
K_OFF = 1.5
K_OFF_HOLD_S = 0.0

PHASIC_LO_S = 0.1
PHASIC_HI_S = 5.0
TONIC_MIN_DUR_S = 15.0
MIN_AMPLITUDE_RATIO = 2.0

TONIC_MIN_COVERAGE = 0.5
PHASIC_MIN_COVERAGE = 0.0
ANY_MIN_COVERAGE = 0.0

LABEL_SOURCE = "auto_cnn_limiar_duplo_v1"
DEFAULT_AUTO_LABEL_MODEL = (
    Path(__file__).resolve().parents[3] / "classifier" / "outputs" / "movement_cnn_final.pt"
)

_MODEL_CACHE: dict[tuple[str, str], tuple[nn.Module, dict[str, Any]]] = {}


class MultiKernelStem(nn.Module):
    def __init__(
        self,
        out_ch_each: int = 6,
        kernel_sizes: tuple[int, ...] = (7, 15, 31, 63),
        stride: int = 2,
    ) -> None:
        super().__init__()
        self.branches = nn.ModuleList(
            [
                nn.Conv1d(1, out_ch_each, kernel_size=k, stride=stride, padding=k // 2)
                for k in kernel_sizes
            ]
        )
        self.bn = nn.BatchNorm1d(out_ch_each * len(kernel_sizes))
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [branch(x) for branch in self.branches]
        x = torch.cat(feats, dim=1)
        return self.pool(self.act(self.bn(x)))


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int, k: int = 7, pool: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.conv = nn.Conv1d(cin, cout, kernel_size=k, padding=k // 2)
        self.bn = nn.BatchNorm1d(cout)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(pool)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.pool(self.act(self.bn(self.conv(x)))))


class MovementCNN(nn.Module):
    def __init__(
        self,
        window_epochs: int = 5,
        samples_per_epoch: int = 300,
        stem_ch_each: int = 6,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.window_epochs = window_epochs
        self.samples_per_epoch = samples_per_epoch
        self.input_len = window_epochs * samples_per_epoch

        self.stem = MultiKernelStem(out_ch_each=stem_ch_each, stride=2)
        stem_ch = stem_ch_each * 4
        self.block1 = ConvBlock(stem_ch, 32, k=7, pool=4, dropout=dropout)
        self.block2 = ConvBlock(32, 48, k=5, pool=4, dropout=dropout)
        self.block3 = ConvBlock(48, 64, k=3, pool=2, dropout=dropout)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.gap(x)
        return self.head(x).squeeze(-1)


@dataclass
class DetectedEvent:
    onset_s: float
    duration_s: float
    type: str
    score: float


def resolve_device(choice: str = "cpu") -> torch.device:
    normalized = (choice or "cpu").lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(normalized)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA foi solicitada para auto-rotulagem, mas não está disponível.")
    return device


def zscore_emg(emg_epochs: np.ndarray) -> np.ndarray:
    emg_epochs = np.asarray(emg_epochs, dtype=np.float32)
    mu = float(emg_epochs.mean())
    sd = float(emg_epochs.std())
    if sd <= 1e-8:
        sd = 1.0
    return (emg_epochs - mu) / sd


def build_emg_windows(emg_epochs: np.ndarray, window_epochs: int) -> torch.Tensor:
    if window_epochs % 2 != 1:
        raise ValueError("window_epochs deve ser ímpar.")
    half = window_epochs // 2
    pad = np.zeros((half, SAMPLES_PER_EPOCH), dtype=np.float32)
    padded = np.concatenate([pad, emg_epochs.astype(np.float32), pad], axis=0)
    n_epochs = emg_epochs.shape[0]
    indices = np.arange(n_epochs)[:, None] + np.arange(window_epochs)[None, :]
    windows = padded[indices].reshape(n_epochs, window_epochs * SAMPLES_PER_EPOCH)
    return torch.from_numpy(windows[:, None, :])


def events_from_binary(
    mask: np.ndarray,
    scores: np.ndarray | None = None,
    *,
    epoch_sec: float = EPOCH_SEC,
) -> list[dict[str, float]]:
    mask = np.asarray(mask, dtype=bool)
    events: list[dict[str, float]] = []
    i = 0
    while i < len(mask):
        if not mask[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(mask) and mask[j + 1]:
            j += 1
        event = {
            "onset_s": round(float(i * epoch_sec), 3),
            "duration_s": round(float((j - i + 1) * epoch_sec), 3),
        }
        if scores is not None:
            event["score"] = round(float(np.mean(scores[i : j + 1])), 4)
        events.append(event)
        i = j + 1
    return events


def rms_envelope(x: np.ndarray, win_sec: float = 0.1, fs: int = FS) -> np.ndarray:
    win = max(1, int(round(win_sec * fs)))
    x2 = x.astype(np.float64) ** 2
    kernel = np.ones(win) / win
    ms = np.convolve(x2, kernel, mode="same")
    return np.sqrt(ms)


def rolling_baseline(
    env: np.ndarray,
    win_sec: float = BASELINE_WIN_S,
    pct: float = BASELINE_PCT,
    fs: int = FS,
) -> np.ndarray:
    win = max(1, int(round(win_sec * fs)))
    n = len(env)
    half = win // 2
    step = max(1, win // 4)
    edges = list(range(0, n, step)) + [n]
    block_vals = []
    block_centers = []
    for start in edges[:-1]:
        end = min(n, start + step)
        lo = max(0, start - half)
        hi = min(n, end + half)
        block_vals.append(np.percentile(env[lo:hi], pct))
        block_centers.append((start + end) / 2)
    return np.interp(np.arange(n), block_centers, block_vals)


def merge_gaps(mask: np.ndarray, gap_samples: int) -> np.ndarray:
    mask = mask.copy()
    i = 0
    while i < len(mask):
        if mask[i]:
            i += 1
            continue
        j = i
        while j < len(mask) and not mask[j]:
            j += 1
        gap_len = j - i
        has_before = i > 0 and mask[i - 1]
        has_after = j < len(mask) and mask[j]
        if has_before and has_after and gap_len <= gap_samples:
            mask[i:j] = True
        i = j
    return mask


def segments_from_mask(mask: np.ndarray) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    i = 0
    while i < len(mask):
        if not mask[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(mask) and mask[j + 1]:
            j += 1
        segments.append((i, j + 1))
        i = j + 1
    return segments


def double_threshold_mask(
    env: np.ndarray,
    baseline: np.ndarray,
    *,
    k_on: float = K_ON,
    k_off: float = K_OFF,
    off_hold_s: float = K_OFF_HOLD_S,
    fs: int = FS,
) -> np.ndarray:
    n = len(env)
    mask = np.zeros(n, dtype=bool)
    off_hold_samples = max(0, int(round(off_hold_s * fs)))
    active = False
    off_count = 0
    off_start: int | None = None

    for i in range(n):
        on_threshold = k_on * baseline[i]
        off_threshold = k_off * baseline[i]

        if not active:
            if env[i] > on_threshold:
                active = True
                mask[i] = True
            continue

        if env[i] > off_threshold:
            if off_start is not None:
                mask[off_start:i] = True
                off_start = None
                off_count = 0
            mask[i] = True
            continue

        if off_hold_samples == 0:
            active = False
            off_count = 0
            off_start = None
            continue

        if off_start is None:
            off_start = i
        off_count += 1

        if off_count >= off_hold_samples:
            active = False
            off_count = 0
            off_start = None

    if active and off_start is not None:
        mask[off_start:n] = True

    return mask


def segments_to_events(
    segments: list[tuple[int, int]],
    env: np.ndarray,
    baseline: np.ndarray,
    *,
    fs: int = FS,
    phasic_lo_s: float = PHASIC_LO_S,
    phasic_hi_s: float = PHASIC_HI_S,
    tonic_min_dur_s: float = TONIC_MIN_DUR_S,
    min_amplitude_ratio: float = MIN_AMPLITUDE_RATIO,
) -> list[DetectedEvent]:
    events: list[DetectedEvent] = []
    for start, end in segments:
        duration_s = (end - start) / fs
        if duration_s < phasic_lo_s:
            continue
        score = float(np.max(env[start:end]) / np.mean(baseline[start:end]))
        if score < min_amplitude_ratio:
            continue
        if phasic_lo_s <= duration_s <= phasic_hi_s:
            event_type = "phasic"
        elif duration_s >= tonic_min_dur_s:
            event_type = "tonic"
        else:
            event_type = "any"
        events.append(
            DetectedEvent(
                onset_s=start / fs,
                duration_s=duration_s,
                type=event_type,
                score=round(score, 3),
            )
        )
    return events


def detect_events(
    emg_flat: np.ndarray,
    *,
    fs: int = FS,
    apply_merge_gaps: bool = True,
    k_on: float = K_ON,
    k_off: float = K_OFF,
    off_hold_s: float = K_OFF_HOLD_S,
) -> list[DetectedEvent]:
    env = rms_envelope(emg_flat, win_sec=0.1, fs=fs)
    baseline = rolling_baseline(env, win_sec=BASELINE_WIN_S, pct=BASELINE_PCT, fs=fs)
    mask = double_threshold_mask(env, baseline, k_on=k_on, k_off=k_off, off_hold_s=off_hold_s, fs=fs)
    if apply_merge_gaps:
        mask = merge_gaps(mask, gap_samples=int(round(MERGE_GAP_S * fs)))
    segments = segments_from_mask(mask)
    return segments_to_events(segments, env, baseline, fs=fs)


def detect_in_window(emg_flat: np.ndarray, start_s: float, end_s: float) -> list[dict[str, float | str]]:
    return detect_in_window_with_params(
        emg_flat,
        start_s,
        end_s,
        k_on=K_ON,
        k_off=K_OFF,
        off_hold_s=K_OFF_HOLD_S,
    )


def detect_in_window_with_params(
    emg_flat: np.ndarray,
    start_s: float,
    end_s: float,
    *,
    k_on: float,
    k_off: float,
    off_hold_s: float,
) -> list[dict[str, float | str]]:
    start = max(0, int(round(start_s * FS)))
    end = min(len(emg_flat), int(round(end_s * FS)))
    if start >= end:
        return []
    local_events = detect_events(
        emg_flat[start:end],
        fs=FS,
        apply_merge_gaps=True,
        k_on=k_on,
        k_off=k_off,
        off_hold_s=off_hold_s,
    )
    clipped_start_s = start / FS
    return [
        {
            "onset_s": clipped_start_s + event.onset_s,
            "duration_s": event.duration_s,
            "type": event.type,
            "score": event.score,
        }
        for event in local_events
    ]


def events_to_labels(
    events: list[dict[str, float | str]],
    n_epochs: int,
    *,
    epoch_sec: float = EPOCH_SEC,
    tonic_min_coverage: float = TONIC_MIN_COVERAGE,
    phasic_min_coverage: float = PHASIC_MIN_COVERAGE,
    any_min_coverage: float = ANY_MIN_COVERAGE,
) -> dict[str, np.ndarray]:
    tonic_cov = np.zeros(n_epochs, dtype=np.float64)
    phasic_cov = np.zeros(n_epochs, dtype=np.float64)
    any_cov = np.zeros(n_epochs, dtype=np.float64)

    for event in events:
        start = max(0.0, float(event["onset_s"]))
        end = start + float(event["duration_s"])
        if end <= 0:
            continue
        first = max(0, int(start // epoch_sec))
        last = min(n_epochs - 1, int((end - 1e-9) // epoch_sec))
        cov = {"tonic": tonic_cov, "phasic": phasic_cov, "any": any_cov}.get(str(event["type"]))
        if cov is None:
            continue
        for mini_epoch in range(first, last + 1):
            m0 = mini_epoch * epoch_sec
            m1 = (mini_epoch + 1) * epoch_sec
            frac = max(0.0, min(end, m1) - max(start, m0)) / epoch_sec
            cov[mini_epoch] = min(1.0, cov[mini_epoch] + frac)

    tonic_labels = (tonic_cov >= tonic_min_coverage).astype(np.float32)
    phasic_labels = (phasic_cov > phasic_min_coverage).astype(np.float32)
    any_labels = (any_cov > any_min_coverage).astype(np.float32)

    return {
        "tonic_labels": tonic_labels,
        "phasic_labels": phasic_labels,
        "any_labels": any_labels,
        "tonic_cov": tonic_cov.astype(np.float32),
        "phasic_cov": phasic_cov.astype(np.float32),
        "any_cov": any_cov.astype(np.float32),
    }


def _load_movement_cnn(model_path: str | Path, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    resolved_path = str(Path(model_path).resolve())
    cache_key = (resolved_path, str(device))
    cached = _MODEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    checkpoint = torch.load(resolved_path, map_location="cpu", weights_only=False)
    window_epochs = int(checkpoint.get("window_epochs", 5))
    model = MovementCNN(window_epochs=window_epochs)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    cached = (model, checkpoint)
    _MODEL_CACHE[cache_key] = cached
    return cached


@torch.no_grad()
def _predict_movement_scores(
    model: nn.Module,
    windows: torch.Tensor,
    device: torch.device,
    *,
    batch_size: int = 512,
) -> np.ndarray:
    scores: list[np.ndarray] = []
    for start in range(0, len(windows), batch_size):
        batch = windows[start : start + batch_size].to(device)
        logits = model(batch)
        scores.append(torch.sigmoid(logits).detach().cpu().numpy())
    return np.concatenate(scores, axis=0)


def auto_label_rswa_from_signals(
    signals: np.ndarray,
    sleep_stages: np.ndarray,
    *,
    model_path: str | Path = DEFAULT_AUTO_LABEL_MODEL,
    emg_channel_index: int = 4,
    device: str = "cpu",
    cnn_threshold: float | None = None,
    cnn_min_epochs: int = 1,
    batch_size: int = 512,
    k_on: float = K_ON,
    k_off: float = K_OFF,
    k_off_hold_s: float = K_OFF_HOLD_S,
    tonic_min_coverage: float = TONIC_MIN_COVERAGE,
    phasic_min_coverage: float = PHASIC_MIN_COVERAGE,
    any_min_coverage: float = ANY_MIN_COVERAGE,
) -> dict[str, Any]:
    signals = np.asarray(signals, dtype=np.float32)
    sleep_stages = np.asarray(sleep_stages, dtype=np.int64)
    if signals.ndim != 3:
        raise ValueError(f"signals deve ter shape [T,C,N], recebeu {signals.shape}")
    if signals.shape[1] <= emg_channel_index:
        raise ValueError(
            f"signals tem {signals.shape[1]} canais; EMG esperado no índice {emg_channel_index}."
        )
    if signals.shape[2] != SAMPLES_PER_EPOCH:
        raise ValueError(
            "A rota automática de RSWA espera mini-épocas de 3 s a 100 Hz "
            f"(N={SAMPLES_PER_EPOCH}), mas recebeu N={signals.shape[2]}."
        )

    device_obj = resolve_device(device)
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint da CNN de movimento não encontrado: {model_path}")

    model, checkpoint = _load_movement_cnn(model_path, device_obj)
    threshold = float(checkpoint.get("threshold", 0.5) if cnn_threshold is None else cnn_threshold)
    window_epochs = int(checkpoint.get("window_epochs", 5))

    emg_epochs = signals[:, emg_channel_index, :SAMPLES_PER_EPOCH].astype(np.float64, copy=False)
    emg_flat = emg_epochs.reshape(-1)
    emg_z = zscore_emg(emg_epochs)
    windows = build_emg_windows(emg_z, window_epochs=window_epochs)
    scores = _predict_movement_scores(model, windows, device_obj, batch_size=batch_size)

    cnn_mask = scores >= threshold
    candidates = events_from_binary(cnn_mask, scores=scores)
    min_duration = cnn_min_epochs * EPOCH_SEC
    candidates = [candidate for candidate in candidates if candidate["duration_s"] >= min_duration - 1e-6]

    confirmed_events: list[dict[str, float | str]] = []
    discarded_windows = 0
    for candidate in candidates:
        events = detect_in_window_with_params(
            emg_flat,
            start_s=float(candidate["onset_s"]),
            end_s=float(candidate["onset_s"]) + float(candidate["duration_s"]),
            k_on=k_on,
            k_off=k_off,
            off_hold_s=k_off_hold_s,
        )
        if not events:
            discarded_windows += 1
            continue
        confirmed_events.extend(events)

    labels = events_to_labels(
        confirmed_events,
        n_epochs=signals.shape[0],
        epoch_sec=EPOCH_SEC,
        tonic_min_coverage=tonic_min_coverage,
        phasic_min_coverage=phasic_min_coverage,
        any_min_coverage=any_min_coverage,
    )
    rswa_conf = (sleep_stages != -1).astype(np.float32)
    rswa_labels = (labels["phasic_labels"].astype(np.int64) * 1) + (
        labels["tonic_labels"].astype(np.int64) * 2
    )

    return {
        **labels,
        "rswa_labels": rswa_labels,
        "rswa_conf": rswa_conf,
        "confirmed_events": confirmed_events,
        "n_cnn_candidates": len(candidates),
        "n_confirmed_events": len(confirmed_events),
        "n_discarded_windows": discarded_windows,
        "cnn_threshold": threshold,
        "k_on": float(k_on),
        "k_off": float(k_off),
        "k_off_hold_s": float(k_off_hold_s),
        "label_source": LABEL_SOURCE,
    }
