from dataclasses import dataclass
import os

@dataclass(frozen=True)
class SignalConfig:
    fs: int = 100
    epoch_sec: int = 3
    samples_per_epoch: int = 300
    context_radius: int = 1
    n_channels: int = 5
    staging_channel_indices: tuple[int, ...] = (0, 1, 2, 3)

@dataclass(frozen=True)
class ModelConfig:
    d_model: int = int(os.getenv("D_MODEL", "256"))
    d_state: int = 16
    dropout: float = float(os.getenv("DROPOUT", "0.35"))
    staging_num_classes: int = 5
    cnn_layers: int = 4
    staging_mamba_layers: int = 1
    rswa_mamba_layers: int = 1
    eeg_in_channels: int = 3
    eog_in_channels: int = 1
    branch_filters: int = 64
    eeg_kernels: tuple[int, ...] = (30, 70, 150)
    eog_kernels: tuple[int, ...] = (50, 150, 250)
    emg_kernels: tuple[int, ...] = (50, 150, 300)
    rswa_emg_in_channels: int = 1
    rswa_emg_filters: int = 64
    rswa_stage_conditioning: bool = False
    rswa_stage_conditioning_detach: bool = True
    rswa_use_baseline_relative_channel: bool = False
    rswa_use_rms_relative_channel: bool = False

@dataclass(frozen=True)
class RSWAConfig:
    emg_channel_index: int = 4
    none_label: int = 0
    phasic_label: int = 1
    tonic_label: int = 2
    rem_stage: int = 4
    min_confidence: float = 0.0
    # Dois canais auxiliares baseline-relative (use_baseline_relative_channel=True):
    #   1. signed_relative = clip(EMG / baseline_uv, -signed_relative_clamp, +signed_relative_clamp)
    #   2. amplitude_relative = clip(log1p(|EMG| / baseline_uv) / log1p(reference_ratio), 0, amplitude_relative_clamp)
    #
    # Assim, ambos os canais ficam na MESMA familia fisiologica (relativos ao
    # basal de atonia) e o valor 1.0 no canal amplitude_relative corresponde
    # exatamente a `reference_ratio` vezes o basal:
    #   log(1 + reference_ratio) / log(1 + reference_ratio) = 1
    # No load do .pt, esse valor pode ser sobrescrito pelo mesmo limiar usado
    # no preprocessamento (ex.: min_amplitude_ratio_used no AASM ou k_on no
    # auto-label), mantendo treinamento e rotulagem alinhados.
    #
    # A baseline correta e' atonia_baseline_uv (a mesma que a regra AASM usa
    # para decidir tonico/fasico -- ver label_metadata.aasm_rule), NAO
    # rem_baseline_uv (baseline crua de baixo percentil, ~5-6x menor).
    baseline_relative_signed_clamp: float = 6.0
    baseline_relative_reference_ratio: float = 4.0
    baseline_relative_amplitude_clamp: float = 2.0
    baseline_relative_rms_win_sec: float = 0.1
    # Fallback quando o .pt nao tem label_metadata.aasm_rule.atonia_baseline_uv
    # (exames legados/antigos): aproxima atonia_baseline_uv como
    # rem_baseline_uv * baseline_relative_fallback_ratio. Razao medida
    # empiricamente em 4 exames de validacao com metadados completos
    # (atonia_baseline_uv / rem_baseline_uv = 4.84 a 6.40, media 5.59).
    baseline_relative_fallback_ratio: float = 5.6

@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = int(os.getenv("BATCH_SIZE", "1"))
    num_workers: int = int(os.getenv("NUM_WORKERS", "2"))
    lr_staging: float = 1e-4
    lr_rswa: float = 1e-4
    epochs_staging: int = 50
    epochs_rswa: int = 30
