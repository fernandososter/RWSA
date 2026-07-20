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
    cnn_layers: int = 4
    staging_mamba_layers: int = 1
    rswa_mamba_layers: int = 1
    eeg_in_channels: int = 3
    eog_in_channels: int = 1
    branch_filters: int = 64
    eeg_kernels: tuple[int, ...] = (30, 70, 150)
    eog_kernels: tuple[int, ...] = (50, 150, 250)
    emg_kernels: tuple[int, ...] = (10, 30, 70)
    rswa_emg_in_channels: int = 1
    rswa_emg_filters: int = 64

@dataclass(frozen=True)
class RSWAConfig:
    emg_channel_index: int = 4
    none_label: int = 0
    phasic_label: int = 1
    tonic_label: int = 2
    rem_stage: int = 4
    min_confidence: float = 0.0

@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = int(os.getenv("BATCH_SIZE", "1"))
    num_workers: int = int(os.getenv("NUM_WORKERS", "2"))
    lr_staging: float = 1e-4
    lr_rswa: float = 1e-4
    epochs_staging: int = 50
    epochs_rswa: int = 30
