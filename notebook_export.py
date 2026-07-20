
# %% [cell 4]
# IPYTHON: !pip uninstall -y mamba-ssm causal-conv1d transformers torch torchvision torchaudio

# IPYTHON: !pip install --no-cache-dir torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
  --index-url https://download.pytorch.org/whl/cu128

# IPYTHON: !pip install "transformers==4.48.3"



# %% [cell 5]
# IPYTHON: !pip uninstall -y causal-conv1d mamba-ssm

# IPYTHON: %env TORCH_CUDA_ARCH_LIST=12.0
# IPYTHON: %env CAUSAL_CONV1D_FORCE_BUILD=TRUE
# IPYTHON: %env MAMBA_FORCE_BUILD=TRUE

# IPYTHON: !pip install --no-cache-dir --no-binary :all: causal-conv1d==1.5.2 --no-build-isolation
# IPYTHON: !pip install --no-cache-dir --no-binary :all: mamba-ssm==2.2.5 --no-build-isolation
# %% [cell 7]
# IPYTHON: !pip uninstall -y mamba-ssm causal-conv1d transformers torch torchvision torchaudio

# IPYTHON: !pip install torch==2.3.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# IPYTHON: !pip install "transformers==4.48.3"
# IPYTHON: !pip install causal-conv1d==1.4.0 mamba-ssm==2.2.2 --no-build-isolation
# %% [cell 8]
import torch
from mamba_ssm import Mamba
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.get_device_name(0))
print(torch.cuda.get_device_capability(0))
print(torch.cuda.get_arch_list())

import torch
from mamba_ssm import Mamba

x = torch.randn(2, 128, 64, device="cuda")
m = Mamba(d_model=64).cuda()
print(m)

# %% [cell 10]
DEBUG=True
# %% [cell 13]
from google.colab import drive
drive.mount('/content/drive')
# %% [cell 15]
# IPYTHON: !cp "/content/drive/MyDrive/Cursos/RWSA/tensors_completo.zip" /content/
# IPYTHON: !mkdir -p /content/tensors && unzip /content/tensors_completo.zip -d /content/
# IPYTHON: !mkdir -p /content/checkpoints
# %% [cell 18]
# IPYTHON: %%writefile .env
BASE_DIR=/content
BASE_LOG_DIR=/content/drive/MyDrive/Cursos/RWSA
D_MODEL=256
DROPOUT=0.35

# Treino
BATCH_SIZE=128
NUM_WORKERS=8
CHUNK_T=-1
# %% [cell 19]
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch  # precisa vir depois
# %% [cell 21]
import matplotlib.pyplot as plt
import torch
import numpy as np
#from torchviz import make_dot
from pathlib import Path
import os
from dotenv import load_dotenv
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
import uuid
from datetime import datetime
import random
from time import perf_counter

# %% [cell 23]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)
# %% [cell 25]

load_dotenv("/content/.env")
# Mudar para as outras execucoes.
SEED=42
# trocar por V0_CNN_only / V1_BiLSTM / V2_MambaUni / V3_BiMamba_noSE / V4_BiMamba_SE
VARIANT_ID = "V4_BiMamba_SE"
# %% [cell 26]
def set_model_seed(seed: int, deterministic: bool = True):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)

FOLD_SPLIT_SEED = SEED    # reaproveita o SEED=42 já existente, nome mais explícito
SEEDS = [0, 1, 2]   # seeds de modelo usadas no loop de run_pipeline
# %% [cell 27]
# ── Sinal ──────────────────────────────────────────────────────────────────
class SignalConfig:
    FS         = 100
    EPOCH_SEC  = 3
    # 1500 para 5
    N_SAMPLES  = 900 # context_window = 3s * 100Hz = 300 amostras por época. * 5 contextx windows = 1500.


    CHANNELS = {
        "emg":  ["EMG chin", "Chin EMG", "EMG1-EMG2", "chin"],
        "eeg_f4":       ["EEG F4-M1", "F4-M1", "F4-A1", "EEG F4"],
        "eeg_c4":       ["EEG C4-M1", "C4-M1", "C4-A1", "EEG C4"],
        "eeg_o2":       ["EEG O2-M1", "O2-M1", "O2-A1", "EEG O2"],
        "eog":       ["EOG E2-M2", "E2-M1", "EOG R", "ROC-A1", "EOG2","EOG-R"],
    }

    N_CHANNELS =  4 # 7 - mantendo 7 para evitar erros de leituras dos canais dos arquivos
    CTX_RADIUS = 1 # número de mini-épocas de contexto em cada lado (total 2*CTX_RADIUS+1 mini-épocas por época)
# %% [cell 29]
class PathConfig:
    BASE_DIR   = Path(os.environ["BASE_DIR"])
    BASE_LOG_DIR   = Path(os.environ["BASE_LOG_DIR"])
    #TENSOR_DIR = BASE_DIR / "tensors"
    TENSOR_DIR = BASE_DIR / "tensors"
    CKPT_DIR   = BASE_LOG_DIR / "checkpoints"
    TLOG_DIR = BASE_LOG_DIR / "tlog/"

# %% [cell 31]

class PreprocessConfig:
    # ── Preprocessamento ───────────────────────────────────────────────────────
    EPOCH_DURATION_S = 3.0
    FFT_BINS         = 128


# %% [cell 33]

class ModelConfig:
    # ── Modelo ──────────────────────────────────────────────────────────────────
    '''
    D_MODEL é a dimensão do vetor que representa cada mini-época ao longo
    de todo o pipeline após a CNN. É o "idioma comum" que conecta a CNN ao Mamba.
    Na pratica, é o tamanho do vetor que sai da CNN e entra no mamba
    apos o avgPool e o Squeeze and Extract.


    D_MODEL = 32  → muito comprimido, perde informação relevante
    D_MODEL = 64  → adequado para T4, suficiente para o problema
    D_MODEL = 128 → melhor capacidade, requer A100 ou gradient checkpointing
    D_MODEL = 256 → estado da arte em modelos grandes, fora de escala aqui
   '''

    D_MODEL         = int(os.environ.get("D_MODEL", 64))

    # BRANCH_IN_CH    = [3, 1, 1]        # EEG, EMG , EOG
    BRANCH_IN_CH    = [3, 1]
    BRANCH_FILTERS  = [64, 64]
    #BRANCH_KERNELS  = (30, 70, 150)
    BRANCH_KERNELS = {
        "eeg": (30, 70, 150),   # ondas mais lentas e padrões de sono
        "emg": (10, 30, 70),    # bursts musculares mais curtos
        "eog": (50, 150, 250),  # movimentos oculares mais lentos
    }
    CNN_LAYERS      = 4
    N_MAMBA_STAGING = 1
    D_STATE         = 16
    DROPOUT         = float(os.environ.get("DROPOUT", 0.1))


    # Ramo RSWA
    RSWA_EMG_IN_CH = 1
    RSWA_EMG_FILTERS = 64
    RSWA_FEATURE_DIM = D_MODEL
# %% [cell 35]
class RSWAConfig:
    EMG_CHANNEL_INDEX = 3

    N_SAMPLES = 300
    N_CLASSES = 2

    HIDDEN_CHANNELS = (32, 64, 128)
    KERNEL_SIZES = (7, 15, 31)

    DROPOUT = 0.30

    # Rótulos atualmente presentes no notebook:
    NONE_LABEL = 0
    PHASIC_LABEL = 1
    TONIC_LABEL = 2

    REM_STAGE = 4

    # Por enquanto, somente posições realmente anotadas.
    MIN_CONFIDENCE = 0.0
# %% [cell 37]
class TrainingConfig:
    # ── Treino ──────────────────────────────────────────────────────────────────
    BATCH_SIZE          = int(os.environ.get("BATCH_SIZE", 1))
    NUM_WORKERS         = int(os.environ.get("NUM_WORKERS", 2))
    # ABLATION
    #K_FOLDS             = 2
    #EPOCHS_PRETRAIN     = 50 # melhor valor de convergencia
    K_FOLDS             = 5
    EPOCHS_PRETRAIN     = 50 # melhor valor de convergencia

    LR_PRETRAIN         = 1e-4

    PATIENCE_PRETRAIN   = 15 # melhor espera ate agora
    LR_BACKBONE         = 1e-5
    LR_RSWA             = 1e-4
    EPOCHS_FINETUNE     = 30
    PATIENCE_FINETUNE   = 8
    STAGE_WEIGHT        = 0.4
    RSWA_WEIGHT         = 0.6
    CHUNK_T             = int(os.environ.get("CHUNK_T", 512))  # epocas por chunk; use -1 para desativar (exame completo, requer GPU grande)
    N1_WEIGHT           = 0.5 # ajuste fino para reduzir superpredição de N1
    REM_WEIGHT          = float(os.environ.get("REM_WEIGHT", 1)) # ajuste fino para reduzir superpredição de REM
    N2_WEIGHT           = float(os.environ.get("N2_WEIGHT", 1)) # ajuste fino para aumentar subprevisão de N2

# %% [cell 39]
import csv
import json
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
# %% [cell 41]

def make_run_id(label: str = None) -> str:
    """
    Gera um identificador de execução legível, combinando timestamp e um
    sufixo curto aleatório (evita colisão se duas execuções começarem no
    mesmo segundo). Use o mesmo run_id para todos os folds de uma mesma
    chamada de run_pipeline.

    label: rótulo opcional e legível para a execução, ex: "smoke_20ex_5ep"
           — útil para identificar rapidamente no CSV qual rodada gerou
           cada linha, sem depender de inferir pelo timestamp ou por
           n_epochs_run como seria necessário sem esse campo.
    """
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    suffix = uuid.uuid4().hex[:6]
    if label:
        return f"{label}_{ts}_{suffix}"
    return f"run_{ts}_{suffix}"



class ExperimentLogger:
    """
    Logger central do ablation. Uma instância por execução
    (variant, fold, seed) — instanciar dentro do loop principal,
    antes de chamar pretrain_staging/evaluate_fold.
    """

    EPOCH_LOG_COLUMNS = [
        "variant_id", "fold", "seed", "phase", "epoch",
        "train_loss", "val_loss", "val_f1", "val_kappa", "val_bac",
        "lr", "timestamp",
    ]

    FOLD_SUMMARY_COLUMNS = [
        "variant_id", "fold", "seed",
        "f1_macro", "kappa", "bac",
        "f1_W", "f1_N1", "f1_N2", "f1_N3", "f1_REM",
        "n_params", "train_time_epoch", "inference_time_total", "inference_time_per_epoch",
        "best_epoch", "n_epochs_run", "n_mini_epochs_test",
        "timestamp",
    ]

    def __init__(self, log_dir: Path, variant_id: str, fold: int, seed: int,
                 run_id: str = None):
        self.log_dir = Path(log_dir)
        self.variant_id = variant_id
        self.fold = fold
        self.seed = seed
        self.run_id = run_id or make_run_id()

        self.log_dir.mkdir(parents=True, exist_ok=True)
        (self.log_dir / "predictions").mkdir(exist_ok=True)
        (self.log_dir / "resources").mkdir(exist_ok=True)

        self.epoch_log_path = self.log_dir / f"{variant_id}_seed_{seed}_epoch_log.csv"
        self.fold_summary_path = self.log_dir / f"{variant_id}_seed_{seed}_summary.csv"

        self._ensure_header(self.epoch_log_path, self.EPOCH_LOG_COLUMNS)
        self._ensure_header(self.fold_summary_path, self.FOLD_SUMMARY_COLUMNS)

        self._train_start_time = None

    @staticmethod
    def _ensure_header(path: Path, columns: list):
        """Cria o arquivo com header se ele ainda não existir."""
        if not path.exists():
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=columns)
                writer.writeheader()

    def _append_row(self, path: Path, columns: list, row: dict):
        # Garante que todas as colunas existem na linha (preenche com None se faltar)
        full_row = {col: row.get(col, None) for col in columns}
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writerow(full_row)

    # ----------------------------------------------------------------
    # Chamado a cada época de treino (dentro do loop em pretrain_staging)
    # ----------------------------------------------------------------
    def log_epoch(self, phase: str, epoch: int, train_loss: float,
                  val_loss: float, val_metrics: dict, lr: float):
        row = {
            "variant_id": self.variant_id,
            "fold": self.fold,
            "seed": self.seed,
            "phase": phase,            # "pretrain_staging" (ou outro nome de fase, se houver)
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_f1": val_metrics.get("f1", None),
            "val_kappa": val_metrics.get("kappa", None),
            "val_bac": val_metrics.get("bac", None),
            "lr": lr,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        self._append_row(self.epoch_log_path, self.EPOCH_LOG_COLUMNS, row)

    # ----------------------------------------------------------------
    # Marcadores de tempo de treino (chamar antes/depois do loop de épocas)
    # ----------------------------------------------------------------
    def start_train_timer(self):
        self._train_start_time = time.time()

    def stop_train_timer(self) -> float:
        elapsed = time.time() - self._train_start_time
        self._train_start_time = None
        return elapsed

    # ----------------------------------------------------------------
    # Chamado uma vez ao final do fold, com o resultado de teste
    # ----------------------------------------------------------------
    def log_fold_summary(self, test_metrics: dict, n_params: int,
                          train_time_total_sec: float, inference_time_total_sec: float,
                          n_mini_epochs_test: int, best_epoch: int, n_epochs_run: int,
                          f1_per_class: dict = None):
        """
        train_time_total_sec: tempo total de treino do fold (soma de todas as épocas).
        inference_time_total_sec: tempo total de inferência no conjunto de teste do fold.
        n_mini_epochs_test: número total de mini-épocas (3s) avaliadas no teste deste fold
                             — usado para derivar inference_time_per_epoch (por mini-época,
                             não por exame), permitindo comparação justa entre variantes
                             com diferentes janelas de contexto.
        n_epochs_run: número de épocas de TREINO de fato executadas (para early stopping) —
                      usado para derivar train_time_epoch (tempo médio por época de treino).
        """
        f1_per_class = f1_per_class or {}

        train_time_epoch = (
            train_time_total_sec / n_epochs_run if n_epochs_run else None
        )
        inference_time_per_epoch = (
            inference_time_total_sec / n_mini_epochs_test if n_mini_epochs_test else None
        )

        row = {
            "variant_id": self.variant_id,
            "fold": self.fold,
            "seed": self.seed,
            "f1_macro": test_metrics.get("f1", None),
            "kappa": test_metrics.get("kappa", None),
            "bac": test_metrics.get("bac", None),
            "f1_W": f1_per_class.get("W", None),
            "f1_N1": f1_per_class.get("N1", None),
            "f1_N2": f1_per_class.get("N2", None),
            "f1_N3": f1_per_class.get("N3", None),
            "f1_REM": f1_per_class.get("REM", None),
            "n_params": n_params,
            "train_time_epoch": train_time_epoch,
            "inference_time_total": inference_time_total_sec,
            "inference_time_per_epoch": inference_time_per_epoch,
            "best_epoch": best_epoch,
            "n_epochs_run": n_epochs_run,
            "n_mini_epochs_test": n_mini_epochs_test,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        self._append_row(self.fold_summary_path, self.FOLD_SUMMARY_COLUMNS, row)

    # ----------------------------------------------------------------
    # Predições por mini-época, por exame — necessário para Fase 3
    # ----------------------------------------------------------------
    def log_predictions(self, exam_ids: list, epoch_idxs: list,
                         y_true: np.ndarray, y_pred: np.ndarray,
                         y_prob: np.ndarray = None):
        """
        exam_ids, epoch_idxs: listas/arrays do mesmo tamanho que y_true/y_pred,
        identificando a qual exame e posição temporal cada linha pertence.
        y_prob, se fornecido: array (N, n_classes) com probabilidades softmax.
        """
        df = pd.DataFrame({
            "exam_id": exam_ids,
            "epoch_idx": epoch_idxs,
            "y_true": y_true,
            "y_pred": y_pred,
        })
        if y_prob is not None:
            class_names = ["W", "N1", "N2", "N3", "REM"]
            for i, name in enumerate(class_names):
                df[f"y_prob_{name}"] = y_prob[:, i]

        out_path = (
            self.log_dir / "predictions"
            / f"{self.variant_id}_fold{self.fold}_seed{self.seed}.parquet"
        )
        df.to_parquet(out_path, index=False, compression="snappy")

    # ----------------------------------------------------------------
    # Metadados de configuração da execução (hiperparâmetros, etc.)
    # Útil para reprodutibilidade e debug posterior.
    # ----------------------------------------------------------------
    def log_run_config(self, config: dict):
        out_path = (
            self.log_dir / "predictions"
            / f"{self.variant_id}_fold{self.fold}_seed{self.seed}_config.json"
        )
        with open(out_path, "w") as f:
            json.dump(config, f, indent=2, default=str)
# %% [cell 43]
LOG_DIR = PathConfig.TLOG_DIR / "ablation_logs"
RLOG_DIR = LOG_DIR / "resources"
print(LOG_DIR)
print(RLOG_DIR)
# %% [cell 44]
INTERVAL_SEC=60
# %% [cell 45]
import time
import threading
import torch

class CUDAMemoryMonitor:
    def __init__(self, interval_sec=5):
        self.interval_sec = interval_sec
        self.running = False
        self.records = []
        self.thread = None

    def _sample(self):
        while self.running:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                self.records.append({
                    "time": time.time(),
                    "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
                    "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
                    "max_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2,
                    "max_reserved_mb": torch.cuda.max_memory_reserved() / 1024**2,
                })
            time.sleep(self.interval_sec)

    def start(self):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self.running = True
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join()
        return self.records
# %% [cell 49]
@dataclass
class SubjectData:
    subject_id:   str
    signals:      torch.Tensor   # Sinais: (T, C, 300)
    sleep_stages: torch.Tensor   # Estagios: N1,N2,etc (T,)
    rswa_labels:  torch.Tensor   # (T,)  0=none 1=phasic 2=tonic
    rswa_conf:    torch.Tensor   # (T,)  0.0=sem anotacao
    n_epochs: int = field(init=False)

    def __post_init__(self): self.n_epochs = self.signals.shape[0] # define a quantidade de epocas do exame (de 3 segundos)

    def __repr__(self):
        return f"SubjectData({self.subject_id!r}, T={self.n_epochs}, REM={(self.sleep_stages==4).sum().item()})"


# %% [cell 51]

class SleepAnalysisDataset(Dataset):
    def __init__(self, subjects, min_confidence=0.0, rem_mask_only=True, global_mean=None, global_std=None):
        self.subjects, self.mc, self.rm = subjects, min_confidence, rem_mask_only
        self.gm, self.gs = global_mean, global_std
    def __len__(self): return len(self.subjects)






    def __getitem__(self, idx):
      s = self.subjects[idx]

      rswa_labels = s.rswa_labels.clone()
      rswa_conf = s.rswa_conf.clone()
      if DEBUG:
        print(f"[RWSA LABELS] rswa_labels.shape: {rswa_labels.shape}")
        print(f"[RWSA LABELS] rswa_conf.shape: {rswa_conf.shape}")

      # ---------------------------------------------------------
      # EMG central antes da montagem do contexto
      # ---------------------------------------------------------
      emg_center = s.signals[:,
            RSWAConfig.EMG_CHANNEL_INDEX: RSWAConfig.EMG_CHANNEL_INDEX + 1,
          :].clone()

      # Shape:
      # emg_center = (T, 1, 300)

      # ---------------------------------------------------------
      # Máscara indicando que existe anotação RSWA
      # ---------------------------------------------------------
      rswa_valid = rswa_conf > self.mc

      if self.rm:
          rem_mask = s.sleep_stages == RSWAConfig.REM_STAGE
          rswa_valid = rswa_valid & rem_mask

      # Não transformar regiões não anotadas em negativos.
      # O rótulo pode permanecer zero, mas rswa_valid=False fará
      # com que essas posições sejam ignoradas pela loss.
      rswa_labels[~rswa_valid] = 0

      # ---------------------------------------------------------
      # Criar alvos binários separados
      # ---------------------------------------------------------
      phasic_labels = (
          rswa_labels == RSWAConfig.PHASIC_LABEL
      ).float()

      tonic_labels = (
          rswa_labels == RSWAConfig.TONIC_LABEL
      ).float()

      # ---------------------------------------------------------
      # Normalização já usada pelo estagiamento
      # ---------------------------------------------------------


      signals = s.signals

      T, C, S = signals.shape

      sig_flat = signals.permute(1, 0, 2).reshape(C, -1)

      mean = sig_flat.mean(dim=1)
      std = sig_flat.std(dim=1).clamp(min=1e-8)

      signals = (
          signals - mean[None, :, None]
      ) / std[None, :, None]

      # ── Normalização por training ─────────

      #if self.gm is not None and self.gs is not None:
      #    signals = (signals - self.gm[None, :, None]) / self.gs[None, :, None]
      # ──────────────────────────────────────────────────────────────────



      # ---------------------------------------------------------
      # Montagem do contexto miniépocas
      # ---------------------------------------------------------
      ctx = SignalConfig.CTX_RADIUS

      pad = torch.zeros( ctx, C, S, dtype=signals.dtype )

      padded = torch.cat( [pad, signals, pad], dim=0 )

      signals_ctx = padded.unfold( 0, 2 * ctx + 1, 1 )
      if DEBUG:
        print(f"[CONTEXT WINDOW] SIZE: {2 * ctx + 1}")
        print(f"[CONTEXT WINDOW] signals_ctx.shape: {signals_ctx.shape}")

      signals_ctx = ( signals_ctx
          .permute(0, 1, 3, 2)
          .reshape(T, C, (2 * ctx + 1) * S)
      )

      # ---------------------------------------------------------
      # Validade do contexto para estagiamento
      # ---------------------------------------------------------
      labels = s.sleep_stages

      pad_lab = torch.full( (ctx,),  -1, dtype=labels.dtype )

      labels_padded = torch.cat( [pad_lab, labels, pad_lab], dim=0 )

      labels_ctx = labels_padded.unfold( 0, 2 * ctx + 1, 1 )

      valid_ctx = ~( labels_ctx == -1 ).any(dim=1)

      # ---------------------------------------------------------
      # DEBUG VALIDACAO DE LABALES DE EXAMES
      # ---------------------------------------------------------


      # DEBUG para verificar se epocas estao sendo cruzadas
      if DEBUG:
        labels = s.sleep_stages
        ctx = SignalConfig.CTX_RADIUS

        pad_lab = torch.full((ctx,), -1, dtype=labels.dtype)
        labels_padded = torch.cat([pad_lab, labels, pad_lab], dim=0)

        labels_ctx = labels_padded.unfold(0, 2*ctx+1, 1)

        bad_windows = (labels_ctx == -1).any(dim=1)

        edge_invalid = torch.zeros_like(bad_windows)
        edge_invalid[:ctx] = True
        edge_invalid[-ctx:] = True

        gap_invalid = bad_windows & ~edge_invalid


        if gap_invalid.sum().item() > 0:
          print(
              f"{s.subject_id} | "
              f"⚠️ T={T} | "
              f"edge_invalid={edge_invalid.sum().item()} | "
              f"gap_invalid={gap_invalid.sum().item()} | "
              f"total_invalid={bad_windows.sum().item()} | "
              f"n_labels_minus1={(labels == -1).sum().item()}"
          )
        # DEBUG

      return {
          "signals": signals_ctx,
          "emg_center": emg_center,

          "sleep_stages": s.sleep_stages,
          "valid_ctx": valid_ctx,

          "rswa_labels": rswa_labels,
          "phasic_labels": phasic_labels,
          "tonic_labels": tonic_labels,
          "rswa_valid": rswa_valid,
          "rswa_conf": rswa_conf,

          "subject_id": s.subject_id,
      }

# %% [cell 52]

def collate_sleep_analysis_exams(batch):
    """
    Agrupa exames de comprimentos diferentes em um único batch.

    O mesmo batch alimenta dois modelos independentes:
      - SleepStagingNet: signals, sleep_stages e staging_valid
      - RSWADetectionNet: emg_center, phasic_labels, tonic_labels e rswa_valid

    Todas as saídas preservam o mesmo índice temporal T, em miniépocas de 3 s.
    """
    lengths = [item["signals"].shape[0] for item in batch]
    t_max = max(lengths)
    batch_size = len(batch)

    _, n_channels, n_context_samples = batch[0]["signals"].shape
    _, n_emg_channels, n_emg_samples = batch[0]["emg_center"].shape

    signals = torch.zeros(
        batch_size, t_max, n_channels, n_context_samples,
        dtype=torch.float32,
    )
    emg_center = torch.zeros(
        batch_size, t_max, n_emg_channels, n_emg_samples,
        dtype=torch.float32,
    )

    sleep_stages = torch.full(
        (batch_size, t_max), -1, dtype=torch.long,
    )
    staging_valid = torch.zeros(
        batch_size, t_max, dtype=torch.bool,
    )
    padding_mask = torch.zeros(
        batch_size, t_max, dtype=torch.bool,
    )

    rswa_labels = torch.zeros(
        batch_size, t_max, dtype=torch.long,
    )
    phasic_labels = torch.zeros(
        batch_size, t_max, dtype=torch.float32,
    )
    tonic_labels = torch.zeros(
        batch_size, t_max, dtype=torch.float32,
    )
    rswa_valid = torch.zeros(
        batch_size, t_max, dtype=torch.bool,
    )
    rswa_conf = torch.zeros(
        batch_size, t_max, dtype=torch.float32,
    )

    subject_ids = []

    for index, (item, length) in enumerate(zip(batch, lengths)):
        signals[index, :length] = item["signals"]
        emg_center[index, :length] = item["emg_center"]

        sleep_stages[index, :length] = item["sleep_stages"]
        staging_valid[index, :length] = item["valid_ctx"]
        padding_mask[index, :length] = True

        rswa_labels[index, :length] = item["rswa_labels"]
        phasic_labels[index, :length] = item["phasic_labels"]
        tonic_labels[index, :length] = item["tonic_labels"]
        rswa_valid[index, :length] = item["rswa_valid"]
        rswa_conf[index, :length] = item["rswa_conf"]

        subject_ids.append(item["subject_id"])

    return {
        "signals": signals,
        "emg_center": emg_center,

        "sleep_stages": sleep_stages,
        "staging_valid": staging_valid,
        # Alias temporário para não quebrar o treino de staging existente.
        "valid_ctx": staging_valid,

        "padding_mask": padding_mask,
        # Alias temporário para não quebrar o código existente.
        "mask": padding_mask,

        "rswa_labels": rswa_labels,
        "phasic_labels": phasic_labels,
        "tonic_labels": tonic_labels,
        "rswa_valid": rswa_valid,
        "rswa_conf": rswa_conf,

        "lengths": torch.tensor(lengths, dtype=torch.long),
        "subject_ids": subject_ids,
    }

# %% [cell 55]
from __future__ import annotations


# ── Detecta mamba-ssm oficial ─────────────────────────────────────────────
try:
    from mamba_ssm import Mamba as MambaOfficial
    MAMBA_OFFICIAL = True
    print("mamba-ssm oficial disponível — usando CUDA kernels")
except ImportError:
    MAMBA_OFFICIAL = False
    print("mamba-ssm não disponível — usando implementação PyTorch puro")


# ── Parallel Scan ─────────────────────────────────────────────────────────
def parallel_scan(A: torch.Tensor, B: torch.Tensor,
                        chunk: int = 32) -> torch.Tensor:
    """
    Scan SSM exato: h_t = A_t * h_{t-1} + B_t
    Chunked para reduzir iterações Python de T para T/chunk.
    Matematicamente exato — erro=0 vs recorrência ground truth.

    A : (B, T, d_inner, d_state)
    B : (B, T, d_inner, d_state)
    """
    orig_dtype = A.dtype
    A = A.float()
    B = B.float()
    Bd, T, di, ds = A.shape

    out = torch.zeros_like(B)
    h   = torch.zeros(Bd, di, ds, dtype=torch.float32, device=A.device)

    for t0 in range(0, T, chunk):
        t1  = min(t0 + chunk, T)
        A_c = A[:, t0:t1]
        B_c = B[:, t0:t1]
        for i in range(t1 - t0):
            h            = A_c[:, i] * h + B_c[:, i]
            out[:, t0+i] = h

    return out.to(orig_dtype)


# %% [cell 57]

# ── MambaSSM — usado apenas quando MAMBA_OFFICIAL=False ──────────────────
class MambaSSM(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2,
                 dt_rank="auto", dt_min=0.001, dt_max=0.1, bias=False):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv  = d_conv
        self.expand  = expand
        self.d_inner = expand * d_model
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj  = nn.Linear(d_model, self.d_inner * 2, bias=bias)
        self.conv1d   = nn.Conv1d(self.d_inner, self.d_inner, d_conv,
                                  bias=True, padding=d_conv-1,
                                  groups=self.d_inner)
        self.x_proj   = nn.Linear(self.d_inner,
                                   self.dt_rank + d_state * 2, bias=False)
        self.dt_proj  = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        A = torch.arange(1, d_state+1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)
        self._init_dt_proj(dt_min, dt_max)

    def _init_dt_proj(self, dt_min, dt_max):
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_weight_decay = True

    def ssm(self, x):
        B, T, d_inner = x.shape
        A       = -torch.exp(self.A_log.float())
        xz      = self.x_proj(x)
        dt_raw, B_ssm, C = xz.split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt      = F.softplus(self.dt_proj(dt_raw))
        A_bar   = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        B_bar   = (dt.unsqueeze(-1) * B_ssm.unsqueeze(2) * x.unsqueeze(-1))
        h       = parallel_scan(A_bar, B_bar)
        y       = (h * C.unsqueeze(2)).sum(-1)
        return y + self.D * x

    def forward(self, x):
        B, T, _ = x.shape
        xz      = self.in_proj(x)
        x_s, z  = xz.chunk(2, dim=-1)
        x_s     = x_s.transpose(1, 2)
        x_s     = self.conv1d(x_s)[..., :T]
        x_s     = x_s.transpose(1, 2)
        x_s     = F.silu(x_s)
        y       = self.ssm(x_s)
        y       = y * F.silu(z)
        return self.out_proj(y)

# %% [cell 59]


# ── BidirMambaBlock — definido UMA única vez ──────────────────────────────
class BidirMambaBlock(nn.Module):
    """
    Bloco Mamba Bidirecional.
    Usa mamba-ssm oficial (CUDA kernels) se disponível,
    caso contrário usa MambaSSM em PyTorch puro.
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)

        if MAMBA_OFFICIAL:
            self.ssm_fwd = MambaOfficial(d_model=d_model, d_state=d_state,
                                          d_conv=d_conv, expand=expand)
            self.ssm_bwd = MambaOfficial(d_model=d_model, d_state=d_state,
                                          d_conv=d_conv, expand=expand)
        else:
            self.ssm_fwd = MambaSSM(d_model, d_state, d_conv, expand)
            self.ssm_bwd = MambaSSM(d_model, d_state, d_conv, expand)

        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x_norm   = self.norm(x)
        y_fwd    = self.ssm_fwd(x_norm)
        if MAMBA_OFFICIAL:
            # mamba-ssm oficial tem conv1d causal — inverter sequência
            # é matematicamente correto para bidirecionalidade
            # A conv1d causal sobre sequência invertida equivale a
            # conv1d anti-causal sobre sequência original
            y_bwd = self.ssm_bwd(x_norm.flip(dims=[1])).flip(dims=[1])
        else:
            y_bwd = self.ssm_bwd(x_norm.flip(dims=[1])).flip(dims=[1])

        return residual + self.drop(y_fwd + y_bwd)



# %% [cell 61]
# ── MambaStack ────────────────────────────────────────────────────────────
class MambaStack(nn.Module):
    """Pilha de N blocos BidirMambaBlock."""
    def __init__(self, d_model, n_layers=4, d_state=16, d_conv=4,
                 expand=2, dropout=0.1):
        super().__init__()
        self.blocks = nn.ModuleList([
            BidirMambaBlock(d_model, d_state, d_conv, expand, dropout)
            for _ in range(n_layers)
        ])
        self.norm_out = nn.LayerNorm(d_model)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.norm_out(x)

    def n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
# %% [cell 63]
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint # Moved import to top of cell


# mamba_pure.py — versão com mamba-ssm oficial
try:
    from mamba_ssm import Mamba as MambaOfficial
    MAMBA_OFFICIAL = True
    print("mamba-ssm oficial disponível — usando CUDA kernels")
except ImportError:
    MAMBA_OFFICIAL = False
    print("mamba-ssm não disponível — usando implementação PyTorch puro")

## Modelo: CNN → Mamba Staging → Mamba RSWA

'''
(B,T,C,300)
  └─ CNN (paralelo sobre T)       → (B,T,D_MODEL)
       └─ Mamba Staging           → stage_logits (B,T,5)
            └─ [feat ‖ probs] → Mamba RSWA → rswa_logits (B,T,3)
'''


## HELPERs
def make_group_norm(channels, max_groups=8):
    for g in [max_groups, 4, 2, 1]:
        if channels % g == 0:
            return nn.GroupNorm(g, channels)
    return nn.GroupNorm(1, channels)

# %% [cell 65]
class SEBlock(nn.Module):
    def __init__(self, n_ch, r=8):
        super().__init__()
        mid = max(1, n_ch // r)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(n_ch, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, n_ch, bias=False),
            nn.Sigmoid())
    def forward(self, x):
        return x * self.fc(x).unsqueeze(-1)


# %% [cell 67]
class RSWAFeatureEncoder(nn.Module):
    """
    Encoder local para o EMG de cada miniépoca de 3 segundos.

    Entrada:
        emg_center: (B, T, C_emg, 300)

    Saída:
        features: (B, T, D_MODEL)

    O encoder reutiliza o mesmo padrão da CNN do estagiamento:
        MultiKernelCNNBranch
        -> SEBlock
        -> projeção 1x1
        -> convolução depthwise
        -> pooling global
    """

    def __init__(
        self,
        in_ch: int = ModelConfig.RSWA_EMG_IN_CH,
        out_ch: int = ModelConfig.RSWA_EMG_FILTERS,
        feature_dim: int = ModelConfig.RSWA_FEATURE_DIM,
        use_se: bool = True,
    ):
        super().__init__()

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.feature_dim = feature_dim
        self.use_se = use_se

        # Mesma implementação multikernel já usada no SleepStagingNet.
        self.emg_branch = MultiKernelCNNBranch(
            in_ch=in_ch,
            out_ch=out_ch,
            kernels=ModelConfig.BRANCH_KERNELS["emg"],
            n_layers=ModelConfig.CNN_LAYERS,
            drop=ModelConfig.DROPOUT,
        )

        # Mesmo padrão de SE global usado no estagiamento.
        if use_se:
            self.se = SEBlock(
                n_ch=out_ch,
                r=8,
            )
        else:
            self.se = nn.Identity()

        # Projeta a saída multikernel para D_MODEL.
        self.branch_proj = nn.Sequential(
            nn.Conv1d(
                out_ch,
                feature_dim,
                kernel_size=1,
                bias=False,
            ),
            make_group_norm(feature_dim),
            nn.ReLU(inplace=True),
        )

        # Mesmo bloco espacial/depthwise utilizado no SleepStagingNet.
        self.spatial = nn.Sequential(
            nn.Conv1d(
                feature_dim,
                feature_dim,
                kernel_size=3,
                padding=1,
                groups=feature_dim,
                bias=False,
            ),
            nn.Conv1d(
                feature_dim,
                feature_dim,
                kernel_size=1,
                bias=False,
            ),
            make_group_norm(feature_dim),
            nn.ReLU(inplace=True),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(
                module.weight,
                nonlinearity="relu",
            )

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(
                module.weight
            )

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(
            module,
            (
                nn.BatchNorm1d,
                nn.GroupNorm,
                nn.LayerNorm,
            ),
        ):
            if module.weight is not None:
                nn.init.ones_(module.weight)

            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, emg_center):
        """
        Parameters
        ----------
        emg_center:
            Tensor com formato (B, T, C_emg, S).

            Para apenas EMG bruto:
                (B, T, 1, 300)

            Caso futuramente seja acrescentado EMG retificado:
                (B, T, 2, 300)

        Returns
        -------
        Tensor:
            Features com formato (B, T, feature_dim).
        """

        if emg_center.ndim != 4:
            raise ValueError(
                "RSWAFeatureEncoder esperava um tensor com formato "
                f"(B,T,C,S), mas recebeu {tuple(emg_center.shape)}."
            )

        B, T, C, S = emg_center.shape

        if C != self.in_ch:
            raise ValueError(
                f"RSWAFeatureEncoder foi configurado para {self.in_ch} "
                f"canal(is), mas recebeu {C}."
            )

        # Cada miniépoca é processada individualmente pela CNN.
        x = emg_center.reshape(B * T,C,S,)

        # MultiKernel CNN:
        # cada caminho utiliza um kernel diferente para capturar
        # atividades musculares em diferentes escalas temporais.
        x = self.emg_branch(x)

        # Ponderação dos mapas de características.
        x = self.se(x)

        # Projeção para o mesmo D_MODEL usado no restante da rede.
        x = self.branch_proj(x)

        # Refinamento espacial/temporal local.
        x = self.spatial(x)

        # (B*T, D_MODEL, comprimento)
        # -> (B*T, D_MODEL, 1)
        x = self.pool(x)

        # -> (B*T, D_MODEL)
        x = x.squeeze(-1)

        # Restaura as dimensões exame e tempo.
        x = x.reshape(B, T,  self.feature_dim, )

        return x
# %% [cell 69]
class CNNBranch(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, n_layers=2, drop=0.1):
        super().__init__()
        layers, ch = [], in_ch
        for _ in range(n_layers):
            layers += [
                nn.Conv1d(ch, out_ch, kernel, padding=kernel//2, bias=False),
                nn.GroupNorm(8, out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2, 2)
            ]
            ch = out_ch
        self.net  = nn.Sequential(*layers)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.net(x))
# %% [cell 71]

class MultiKernelCNNBranch(nn.Module):
    def __init__(self, in_ch, out_ch, kernels=(30, 70, 150), n_layers=2, drop=0.1):
        super().__init__()

        per_kernel = out_ch // len(kernels)
        remainder = out_ch - per_kernel * len(kernels)

        self.paths = nn.ModuleList()
        for i, k in enumerate(kernels):
            ch_out = per_kernel + (remainder if i == 0 else 0)

            layers = []
            ch = in_ch
            for _ in range(n_layers):
                layers += [
                    nn.Conv1d(ch, ch_out, k, padding=k // 2, bias=False),
                    make_group_norm(ch_out),
                    nn.ReLU(inplace=True),
                    nn.MaxPool1d(2, 2),
                ]
                ch = ch_out

            self.paths.append(nn.Sequential(*layers))

        self.drop = nn.Dropout(drop)

    def forward(self, x):
        feats = [path(x) for path in self.paths]
        return self.drop(torch.cat(feats, dim=1))
# %% [cell 73]
# ============================================================================
# SleepStagingNet — versão parametrizada de RSWANet para o ablation study
# ============================================================================
#
# Substitui RSWANet nas execuções do ablation. Diferenças:
#   1. Sem ramo RSWA (mamba_rswa, rswa_proj, rswa_head) — decisão confirmada:
#      o paper foi reposicionado para estagiamento puro, e o ramo RSWA não
#      é mais treinado nem avaliado nas execuções do ablation.
#   2. `temporal_backbone` parametrizável: "none" | "bilstm" | "mamba_uni" | "bimamba"
#      — implementa as variantes V0, V1, V2, V3/V4 do plano.
#   3. `use_se` parametrizável: liga/desliga o SE Block (variante V3 vs V4).
#
# Mapeamento para as variantes do plano:
#   V0  -> SleepStagingNet(temporal_backbone="none",     use_se=True)
#   V1  -> SleepStagingNet(temporal_backbone="bilstm",    use_se=True)
#   V2  -> SleepStagingNet(temporal_backbone="mamba_uni", use_se=True)
#   V3  -> SleepStagingNet(temporal_backbone="bimamba",   use_se=False)
#   V4  -> SleepStagingNet(temporal_backbone="bimamba",   use_se=True)   # proposta original

import torch
import torch.nn as nn


# ----------------------------------------------------------------------------
# UniMambaBlock / UniMambaStack — análogos a BidirMambaBlock/MambaStack,
# mas processando a sequência em uma única direção (apenas ssm_fwd).
# Necessário para a variante V2 (isolar o efeito da bidirecionalidade).
# ----------------------------------------------------------------------------
class UniMambaBlock(nn.Module):
    """
    Versão unidirecional de BidirMambaBlock. Mesma estrutura de
    normalização/dropout/residual, removendo o ramo backward (ssm_bwd).
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm_fwd = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        y_fwd = self.ssm_fwd(x)
        return residual + self.drop(y_fwd)


class UniMambaStack(nn.Module):
    def __init__(self, d_model, n_layers, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            UniMambaBlock(d_model, d_state, d_conv, expand, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# ----------------------------------------------------------------------------
# SleepStagingNet
# ----------------------------------------------------------------------------


class SleepStagingNet(nn.Module):
    """
    Fluxo (estagiamento puro, sem ramo RSWA):
      (B,T,C,300) → CNN [+ SE opcional] → (B,T,D_MODEL)
                  → backbone temporal (none/bilstm/mamba_uni/bimamba)
                  → stage_logits (B,T,5)
    """
    def __init__(self, temporal_backbone: str = "bimamba", use_se: bool = True):
        super().__init__()
        assert temporal_backbone in ("none", "bilstm", "mamba_uni", "bimamba"), \
            f"temporal_backbone inválido: {temporal_backbone}"

        self.temporal_backbone_kind = temporal_backbone
        self.use_se = use_se

        total_filters = sum(ModelConfig.BRANCH_FILTERS)
        # branch_names = ["eeg", "emg", "eog"]
        branch_names = ["eeg","eog"]

        # ── CNN (idêntico a RSWANet) ────────────────────────────────────────
        self.branches = nn.ModuleList([
            MultiKernelCNNBranch(
                in_ch=in_ch,
                out_ch=filters,
                kernels=ModelConfig.BRANCH_KERNELS[name],
                n_layers=ModelConfig.CNN_LAYERS,
                drop=ModelConfig.DROPOUT,
            )
            for name, in_ch, filters in zip(
                branch_names,
                ModelConfig.BRANCH_IN_CH,
                ModelConfig.BRANCH_FILTERS,
            )
        ])

        # ── SE Block (opcional — variante V3 desliga isso) ──────────────────
        if self.use_se:
            self.se_global = SEBlock(n_ch=total_filters, r=8)
        else:
            self.se_global = nn.Identity()   # passagem direta, sem ponderação de canal

        self.branch_proj = nn.Sequential(
            nn.Conv1d(total_filters, ModelConfig.D_MODEL, 1, bias=False),
            nn.GroupNorm(8, ModelConfig.D_MODEL),
            nn.ReLU(inplace=True)
        )
        self.spatial = nn.Sequential(
            nn.Conv1d(ModelConfig.D_MODEL, ModelConfig.D_MODEL, 3, padding=1, groups=ModelConfig.D_MODEL, bias=False),
            nn.Conv1d(ModelConfig.D_MODEL, ModelConfig.D_MODEL, 1, bias=False),
            nn.GroupNorm(8, ModelConfig.D_MODEL),
            nn.ReLU(inplace=True))
        self.pool = nn.AdaptiveAvgPool1d(1)

        # ── Backbone temporal (parametrizável — V0/V1/V2/V3/V4) ─────────────
        if temporal_backbone == "none":
            # V0: nenhuma modelagem de sequência longa — features da CNN
            # vão direto para o stage_head, mini-época a mini-época.
            self.temporal = nn.Identity()

        elif temporal_backbone == "bilstm":
            # V1: BiLSTM como alternativa recorrente clássica ao BiMamba.
            # hidden_size = D_MODEL//2 em cada direção, para que a saída
            # concatenada (bidirecional) retorne a D_MODEL, comparável em
            # dimensão à saída do MambaStack.
            self._bilstm = nn.LSTM(
                input_size=ModelConfig.D_MODEL,
                hidden_size=ModelConfig.D_MODEL // 2,
                num_layers=ModelConfig.N_MAMBA_STAGING,  # mesmo nº de "camadas" do Mamba, para custo comparável
                batch_first=True,
                bidirectional=True,
                dropout=ModelConfig.DROPOUT if ModelConfig.N_MAMBA_STAGING > 1 else 0.0,
            )
            self.temporal = self._bilstm_forward

        elif temporal_backbone == "mamba_uni":
            # V2: Mamba unidirecional — isola o efeito específico da
            # bidirecionalidade (mesmo nº de camadas/d_state que o BiMamba).
            self.temporal = UniMambaStack(
                ModelConfig.D_MODEL, ModelConfig.N_MAMBA_STAGING,
                ModelConfig.D_STATE, dropout=ModelConfig.DROPOUT,
            )

        elif temporal_backbone == "bimamba":
            # V3/V4: arquitetura proposta original.
            self.temporal = MambaStack(
                ModelConfig.D_MODEL, ModelConfig.N_MAMBA_STAGING,
                ModelConfig.D_STATE, dropout=ModelConfig.DROPOUT,
            )

        self.stage_head = nn.Sequential(
            nn.Linear(ModelConfig.D_MODEL, ModelConfig.D_MODEL // 2), nn.ReLU(inplace=True),
            nn.Dropout(ModelConfig.DROPOUT), nn.Linear(ModelConfig.D_MODEL // 2, 5))

        self.apply(self._init_weights)

    def _bilstm_forward(self, x):
        # wrapper para que self.temporal(x) tenha a mesma assinatura
        # (entrada e saída (B,T,D_MODEL)) independente do backbone escolhido
        out, _ = self._bilstm(x)
        return out

    def _init_weights(self, m):
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm, nn.LayerNorm)):
            nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def cnn_encode(self, signals):
        B, T, C, S = signals.shape
        x = signals.view(B * T, C, S)

        #print(f"X: {x}")
        x_eeg = x[:, 0:3, :]   # F4, C4, O2, C3
      #  x_emg = x[:, 3:4, :]   # chin + EMG retificado
        x_eog = x[:, 4:5, :]   # E1, E2

        if DEBUG:
          print(f"-----------")
          print(f"EEG size: {len(x_eeg)} {x_eeg}")
          print(f"-----------")
          print(f"EOG size: {len(x_eog)} {x_eog} ")
          print(f"-----------")


        feat = torch.cat([
            self.branches[0](x_eeg),
            #self.branches[1](x_emg),
            self.branches[1](x_eog)],
        dim=1)

        feat = self.se_global(feat)   # Identity() se use_se=False
        feat = self.pool(self.spatial(self.branch_proj(feat))).squeeze(-1)
        return feat.view(B, T, ModelConfig.D_MODEL)

    def forward(self, signals, mask=None):
        feat = self.cnn_encode(signals)
        feat = self.temporal(feat)
        stage_logits = self.stage_head(feat)
        return stage_logits   # sem rswa_logits — ramo RSWA removido

    def n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ----------------------------------------------------------------------------
# Fábrica por nome de variante — usada nos notebooks finos por variante
# ----------------------------------------------------------------------------
VARIANT_CONFIGS = {
    "V0_CNN_only":   {"temporal_backbone": "none",      "use_se": True},
    "V1_BiLSTM":     {"temporal_backbone": "bilstm",    "use_se": True},
    "V2_MambaUni":   {"temporal_backbone": "mamba_uni", "use_se": True},
    "V3_BiMamba_noSE": {"temporal_backbone": "bimamba", "use_se": False},
    "V4_BiMamba_SE": {"temporal_backbone": "bimamba",   "use_se": True},
}


def build_model(variant_id: str) -> "SleepStagingNet":
    if variant_id not in VARIANT_CONFIGS:
        raise ValueError(f"variant_id desconhecido: {variant_id}. "
                          f"Opções: {list(VARIANT_CONFIGS.keys())}")
    cfg = VARIANT_CONFIGS[variant_id]
    return SleepStagingNet(**cfg)
# %% [cell 74]

# ============================================================================
# Ramo independente para detecção de RSWA em miniépocas de 3 segundos
# ============================================================================

class RSWADetectionNet(nn.Module):
    """
    Rede independente para detecção de atividade muscular tônica e fásica.

    Fluxo:
        EMG central (B,T,C_emg,300)
        -> RSWAFeatureEncoder
        -> BiMamba específico para RSWA
        -> duas saídas binárias por miniépoca:
             tonic_logits  (B,T)
             phasic_logits (B,T)

    As duas saídas são independentes para permitir coexistência de atividade
    tônica e fásica na mesma miniépoca.
    """

    def __init__(self, use_se: bool = True):
        super().__init__()

        self.encoder = RSWAFeatureEncoder(
            in_ch=ModelConfig.RSWA_EMG_IN_CH,
            out_ch=ModelConfig.RSWA_EMG_FILTERS,
            feature_dim=ModelConfig.RSWA_FEATURE_DIM,
            use_se=use_se,
        )

        self.temporal = MambaStack(
            d_model=ModelConfig.RSWA_FEATURE_DIM,
            n_layers=ModelConfig.N_MAMBA_RSWA,
            d_state=ModelConfig.D_STATE,
            dropout=ModelConfig.DROPOUT,
        )

        hidden_dim = max(ModelConfig.RSWA_FEATURE_DIM // 2, 1)

        self.tonic_head = nn.Sequential(
            nn.Linear(ModelConfig.RSWA_FEATURE_DIM, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(ModelConfig.DROPOUT),
            nn.Linear(hidden_dim, 1),
        )

        self.phasic_head = nn.Sequential(
            nn.Linear(ModelConfig.RSWA_FEATURE_DIM, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(ModelConfig.DROPOUT),
            nn.Linear(hidden_dim, 1),
        )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, emg_center, mask=None):
        features = self.encoder(emg_center)
        features = self.temporal(features)

        tonic_logits = self.tonic_head(features).squeeze(-1)
        phasic_logits = self.phasic_head(features).squeeze(-1)

        return {
            "tonic_logits": tonic_logits,
            "phasic_logits": phasic_logits,
        }

    def n_params(self):
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )


class SleepStagingRSWASystem(nn.Module):
    """
    Contêiner de execução conjunta.

    Os dois modelos continuam independentes:
      - nenhum peso é compartilhado;
      - cada ramo possui CNN/SE/BiMamba/head próprios;
      - a combinação ocorre somente nos outputs.

    Este contêiner não define loss nem optimizer compartilhados.
    """

    def __init__(
        self,
        staging_model: SleepStagingNet,
        rswa_model: RSWADetectionNet,
    ):
        super().__init__()
        self.staging_model = staging_model
        self.rswa_model = rswa_model

    def forward(self, signals, emg_center, mask=None):
        staging_logits = self.staging_model(
            signals,
            mask=mask,
        )

        rswa_outputs = self.rswa_model(
            emg_center,
            mask=mask,
        )

        return {
            "staging_logits": staging_logits,
            **rswa_outputs,
        }

# %% [cell 76]
model = build_model(VARIANT_ID).to(DEVICE)
# make_dot(model(torch.randn(1, 3000, SignalConfig.N_CHANNELS, SignalConfig.N_SAMPLES).to(DEVICE)), params=dict(model.named_parameters())).render("model_schema", format="png")
print(f"Modelo inicializado com {model.n_params():,} parametros")
print(model)
# %% [cell 78]
subjects = load_all_subjects_parallel(PathConfig.TENSOR_DIR, rswa_dir=None, max_workers=TrainingConfig.NUM_WORKERS)
# %% [cell 79]

# ── Estatísticas do dataset ───────────────────────────────────────────────
stage_names = {-1: "Unknown", 0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}
all_stages  = torch.cat([s.sleep_stages for s in subjects])
all_rswa    = torch.cat([s.rswa_labels  for s in subjects])
total_epochs = len(all_stages)
print(f"\n{'='*50}")
print(f"  Dataset: {len(subjects)} exames")
print(f"  Total mini-épocas : {total_epochs:,}  ({total_epochs * SignalConfig.EPOCH_SEC / 3600:.1f}h de sinal)")
print(f"\n  Distribuição de estágios:")
for code, name in stage_names.items():
    n = (all_stages == code).sum().item()
    print(f"    {name:8s} ({code:2d}): {n:7,}  ({100*n/total_epochs:.1f}%)")
print(f"\n  RSWA (apenas REM):")
rem_mask = all_stages == 4
n_rem = rem_mask.sum().item()
for code, name in [(0,"None"), (1,"Phasic"), (2,"Tonic")]:
    n = ((all_rswa == code) & rem_mask).sum().item()
    pct = 100 * n / n_rem if n_rem > 0 else 0
    print(f"    {name:8s}: {n:7,}  ({pct:.1f}% do REM)")
n_annotated = sum(1 for s in subjects if s.rswa_conf.max().item() > 0)
print(f"\n  Exames com anotação RSWA: {n_annotated}/{len(subjects)}")
print(f"{'='*50}\n")
# %% [cell 81]
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import  xml.etree.ElementTree as ET
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, cohen_kappa_score, balanced_accuracy_score
import time
# %% [cell 83]

def run_pipeline(subjects, seeds=None):
    seeds = seeds if seeds is not None else SEEDS

    pipeline_start = perf_counter()

    print("[VALIDANDO SUBJECTS]")
    print(f"Total de sujeitos no dataset: {len(subjects)}")
    for s in subjects[:5]:
        print(f"  {s.subject_id}:{s.signals.shape}")

    splits  = group_kfold_splits(subjects, TrainingConfig.K_FOLDS, FOLD_SPLIT_SEED)
    results = []

    RUN_ID = make_run_id(VARIANT_ID)
    print(f"RUN_ID desta execução: {RUN_ID}")

    for split in splits:
        fi = split["fold"]


        # ── Estatísticas globais calculadas SÓ com o treino deste fold ──
        print(f"\n[] Fold {fi}: calculando estatísticas de normalização "
              f"somente sobre os {len(split['train'])} exames de treino...")
        FOLD_GLOBAL_MEAN, FOLD_GLOBAL_STD = compute_global_stats(split["train"])
        print(f"  GLOBAL_MEAN: {FOLD_GLOBAL_MEAN} + GLOBAL_STD: {FOLD_GLOBAL_STD}")
        # ───

        for seed_exec in seeds:
            set_model_seed(seed_exec, deterministic=True)

            logger = ExperimentLogger(
                LOG_DIR, variant_id=VARIANT_ID, fold=fi, seed=seed_exec,
                run_id=RUN_ID,
            )

            print("INICIALIZANDO MODELO")
            model = build_model(VARIANT_ID).to(DEVICE)

            print(f"\n{'#'*50}\nFOLD {fi+1}/{TrainingConfig.K_FOLDS}  SEED {seed_exec}")
            print(f" Epocas: Treino={TrainingConfig.EPOCHS_PRETRAIN}")
            print(f" LR: {TrainingConfig.LR_PRETRAIN}")
            print(f" Filtros: {ModelConfig.BRANCH_FILTERS}")

            model, hist, train_info = train_staging(
                model, split["train"], split["val"], fi,seed_exec,
                global_mean=FOLD_GLOBAL_MEAN, global_std=FOLD_GLOBAL_STD,
                logger=logger,
            )

            plot_training_history(hist, fi)

            r = evaluate_fold(
                model, split["test"],
                global_mean=FOLD_GLOBAL_MEAN, global_std=FOLD_GLOBAL_STD,
                logger=logger,
            )
            results.append(r)

            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

            logger.log_fold_summary(
                test_metrics=r["stage"],
                n_params=n_params,
                train_time_total_sec=train_info["train_time_sec"],
                inference_time_total_sec=r["inference_time_total_sec"],
                n_mini_epochs_test=r["n_mini_epochs_test"],
                best_epoch=train_info["best_epoch"],
                n_epochs_run=train_info["n_epochs_run"],
                f1_per_class=r["f1_per_class"],
            )

            sf = r["stage"]
            print(f"  TESTE — stage f1={sf.get('f1',0):.3f} k={sf.get('kappa',0):.3f}")

            del model; torch.cuda.empty_cache()

    report_results(results)
    pipeline_time = perf_counter() - pipeline_start

    print("\n")
    print("=" * 70)
    print(f"METRIC_PIPELINE total: {pipeline_time/3600:.2f} horas")
    print("=" * 70)

    return results
# %% [cell 84]


# PIPELINE DE TREINAMENTO
def plot_training_history(hist, fold_id):
    n_ep = len(hist.get("val_loss", []))
    if n_ep == 0:
        return

    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    fig.suptitle(f"Fold {fold_id + 1} — Histórico de Treino (Staging)")

    eps = range(1, n_ep + 1)
    ax.plot(eps, hist["train_loss"], label="train loss", linewidth=1.5)
    ax.plot(eps, hist["val_loss"],   label="val loss",   linewidth=1.5)
    ax2 = ax.twinx()
    ax2.plot(eps, hist["val_f1"], color="green", linestyle="--",
             label="val f1", linewidth=1.5)
    ax2.set_ylabel("val F1", color="green")
    ax2.tick_params(axis="y", labelcolor="green")
    ax.set_xlabel("Época"); ax.set_ylabel("Loss")
    ax.legend(loc="upper left")

    plt.tight_layout()
    plt.show()

'''
Metodo para calcular as estatisticas globais do dataset, para normalizacao dos sinais.
'''
def compute_global_stats(subjects):
    """Calcula mean/std global por canal sobre todo o dataset."""
    n_ch = subjects[0].signals.shape[1]
    sums    = torch.zeros(n_ch, dtype=torch.float64)
    sq_sums = torch.zeros(n_ch, dtype=torch.float64)
    n_total = 0

    print(f"Calculando estatísticas globais ({len(subjects)} exames)...")
    for s in subjects:
        sig = s.signals.double()
        T, C, S = sig.shape
        sums    += sig.sum(dim=(0, 2))
        sq_sums += (sig ** 2).sum(dim=(0, 2))
        n_total += T * S

    mean = sums / n_total
    var  = (sq_sums / n_total) - mean ** 2
    std  = var.clamp(min=1e-12).sqrt()

    ch_names = ['EEG_F4', 'EEG_C4', 'EEG_O2', 'EEG_C3',
                'EMG_raw', 'EOG_E1', 'EOG_E2']
    print(f"  Estatísticas por canal:")
    for i in range(n_ch):
        name = ch_names[i] if i < len(ch_names) else f"ch{i}"
        print(f"    {name:<10}: mean={mean[i].item():+.3e}  std={std[i].item():.3e}")

    return mean.float(), std.float()





def compute_metrics(preds, targets, n_cls):
    p, t = np.array(preds), np.array(targets)
    v = t >= 0
    p, t = p[v], t[v]
    if len(t) == 0:
        return {}
    return {
        "acc":   (p == t).mean(),
        "f1":    f1_score(t, p, average="macro",
                          labels=list(range(n_cls)),
                          zero_division=0),
        "kappa": cohen_kappa_score(t, p),
        "bac":   balanced_accuracy_score(t, p),
    }



def fmt(m):
    s,r=m.get("stage",{}),m.get("rswa",{})
    parts=[f"loss={m['loss']:.4f}"]
    if s: parts.append(f"stg_f1={s.get('f1',0):.3f} k={s.get('kappa',0):.3f}")
    if r and r.get("f1",0)>0: parts.append(f"rswa_f1={r.get('f1',0):.3f}")
    return " | ".join(parts)

def save_ckpt(model,opt,epoch,metrics,path):
    torch.save({"epoch":epoch,"model":model.state_dict(),"opt":opt.state_dict(),"metrics":metrics},path)

def load_ckpt(model, path, opt=None):
    c = torch.load(path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(c["model"])
    if opt: opt.load_state_dict(c["opt"])
    print(f"  Checkpoint: {path.name} (epoch {c['epoch']})")
    return c






def log_bn_stats(model, ep):
    branches_names = ['EEG', 'EMG', 'EOG']
    for i, name in enumerate(branches_names):
        bn = model.branches[i].net[1]
        mean_range = (bn.running_mean.min().item(), bn.running_mean.max().item())
        var_range  = (bn.running_var.min().item(),  bn.running_var.max().item())
        print(f"  BN_{name}: mean=[{mean_range[0]:+.3e}, {mean_range[1]:+.3e}]  "
              f"var=[{var_range[0]:.3e}, {var_range[1]:.3e}]")


def compute_f1_per_class(preds, targets, n_cls=5):
    from sklearn.metrics import f1_score
    p, t = np.array(preds), np.array(targets)
    v = t >= 0
    p, t = p[v], t[v]
    class_names = ["W", "N1", "N2", "N3", "REM"]
    scores = f1_score(t, p, average=None, labels=list(range(n_cls)), zero_division=0)
    return dict(zip(class_names, scores))

def plot_cm(trues,preds,labels,title):
    cm=confusion_matrix(trues,preds,labels=list(range(len(labels))))
    norm=cm.astype(float)/cm.sum(1,keepdims=True).clip(1)
    fig,ax=plt.subplots(figsize=(6,5))
    sns.heatmap(norm,annot=True,fmt=".2f",cmap="Blues",
                xticklabels=labels,yticklabels=labels,ax=ax)
    ax.set(xlabel="Predito",ylabel="Real",title=title)
    plt.tight_layout(); plt.show()

def analyze_transitions(stage_preds, stage_trues):
    import numpy as np
    from sklearn.metrics import f1_score

    preds = np.array(stage_preds)
    trues = np.array(stage_trues)

    transitions = np.zeros(len(trues), dtype=bool)
    transitions[1:] = trues[1:] != trues[:-1]

    cont_mask  = ~transitions & (trues >= 0)
    trans_mask =  transitions & (trues >= 0)

    acc_cont  = (preds[cont_mask]  == trues[cont_mask]).mean()  if cont_mask.sum()  > 0 else 0
    acc_trans = (preds[trans_mask] == trues[trans_mask]).mean() if trans_mask.sum() > 0 else 0

    f1_trans = f1_score(trues[trans_mask], preds[trans_mask],
                        average="macro", labels=list(range(5)),
                        zero_division=0) if trans_mask.sum() > 0 else 0

    # Baseline correto — preds_prev filtrado pelos mesmos índices
    preds_prev = trues[:-1]
    trans_idx  = np.where(transitions[1:])[0]   # ← índices relativos a trues[1:]
    baseline_trans = (preds_prev[trans_idx] == trues[1:][trans_idx]).mean() \
                     if len(trans_idx) > 0 else 0

    print(f"\n{'='*50}")
    print(f"ANÁLISE DE TRANSIÇÕES")
    print(f"{'='*50}")
    print(f"Total timesteps:          {len(trues)}")
    print(f"Transições:               {trans_mask.sum()} ({100*trans_mask.mean():.1f}%)")
    print(f"Continuidade:             {cont_mask.sum()}  ({100*cont_mask.mean():.1f}%)")
    print(f"\nModelo atual:")
    print(f"  Acc em continuidade:    {acc_cont:.3f}")
    print(f"  Acc em transições:      {acc_trans:.3f}")
    print(f"  F1  em transições:      {f1_trans:.3f}")
    print(f"  Ratio trans/cont:       {acc_trans/acc_cont:.3f}")
    print(f"\nBaseline (repete anterior):")
    print(f"  Acc em continuidade:    1.000")
    print(f"  Acc em transições:      {baseline_trans:.3f}")
    print(f"\nGanho sobre baseline:     {acc_trans - baseline_trans:+.3f}")


def report_results(fold_results):
    print("\n"+"="*50+"\nRESULTADOS FINAIS\n"+"="*50)
    print("\nSleep Staging:")
    for metric in ["acc","f1","kappa","bac"]:
        vals=[r["stage"].get(metric,0) for r in fold_results]
        print(f"  {metric:6s}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")

    all_sp=np.concatenate([r["stage_preds"] for r in fold_results])
    all_st=np.concatenate([r["stage_trues"] for r in fold_results])
    plot_cm(all_st,all_sp,["W","N1","N2","N3","REM"],"Staging — Todos os Folds")

    print("\nAnálise de transições:")
    analyze_transitions(all_sp, all_st)

    stage_pred_arr = np.array(all_sp)
    stage_true_arr = np.array(all_st)

    print(f"\nDiagnóstico de classes — Staging:")
    print(f"  Classes nos targets : {np.unique(stage_true_arr).tolist()}")
    print(f"  Classes nas predições: {np.unique(stage_pred_arr).tolist()}")
    print(f"  Distribuição targets : { {c: (stage_true_arr==c).sum() for c in range(5)} }")
    print(f"  Distribuição predições: { {c: (stage_pred_arr==c).sum() for c in range(5)} }")


# %% [cell 86]
from datetime import datetime
start = datetime.now()
print(start)
# %% [cell 87]
# ============================================================================
# Funções de treino/avaliação ajustadas para SleepStagingNet (sem ramo RSWA)
# ============================================================================
#
# Substituem MultiTaskLoss, run_epoch, pretrain_staging, finetune_rswa,
# evaluate_fold do notebook original. A mudança estrutural em todas elas é
# a mesma: o modelo agora retorna só `stage_logits` (não mais a tupla
# `(stage_logits, rswa_logits)`), então todo código que desempacotava dois
# valores de model(...) precisa ser ajustado para desempacotar um só.
#
# `finetune_rswa` deixa de ser chamado em run_pipeline — não há mais uma
# segunda fase de treino, já que não há ramo RSWA para ajustar.

import time
import torch
import torch.nn as nn
from collections import Counter
from torch.utils.data import DataLoader
from tqdm import tqdm


# ----------------------------------------------------------------------------
# StagingLoss — versão de MultiTaskLoss sem o termo RSWA.
# Equivalente a MultiTaskLoss(stage_weight=1.0, rswa_weight=0.0), mas sem
# carregar o cálculo morto de l_rswa (que antes era computado e multiplicado
# por zero) e sem exigir rl/rt/rswa_conf como argumentos.
# ----------------------------------------------------------------------------
class StagingLoss(nn.Module):
    def __init__(self, label_smoothing=0.05, stage_cls_w=None):
        super().__init__()
        # ignore_index=-1: mini-épocas sem anotação (gaps entre scorings ou
        # fora do período anotado) são mantidas no tensor com stage=-1 para
        # preservar a continuidade temporal do backbone. A loss não é
        # calculada nessas mini-épocas.
        self.stage_ce = nn.CrossEntropyLoss(
            weight=stage_cls_w, ignore_index=-1,
            label_smoothing=label_smoothing, reduction="none",
        )

    def forward(self, sl, st, mask):
        B, T, _ = sl.shape
        m = mask.float().view(B * T)
        sl_ = sl.view(B * T, -1)
        st_ = st.view(B * T)
        sv = (st_ >= 0).float() * m
        l_stage = (self.stage_ce(sl_, st_) * sv).sum() / sv.sum().clamp(1)
        return l_stage


# ----------------------------------------------------------------------------
# run_epoch — idêntico em estrutura (chunking, AMP, masking) ao original,
# só removendo tudo relativo a rl/rt_/rp/rswa.
# ----------------------------------------------------------------------------
def run_epoch(model, loader, criterion, optimizer=None, phase="train"):
    """
    Forward/backward com chunked processing ao longo de T.
    Mesma lógica de chunking do run_epoch original — ver docstring da
    versão anterior para detalhes de memória/contexto.
    """
    model.train() if phase == "train" else model.eval()
    total_loss, n_batches = 0., 0
    sp, st_ = [], []

    device_type = "cuda" if str(DEVICE).startswith("cuda") else "cpu"
    use_amp = device_type == "cuda"

    ctx = torch.enable_grad() if phase == "train" else torch.no_grad()
    with ctx:
        for batch in tqdm(loader, desc=f"{phase} batches", leave=False):
            sig_cpu = batch["signals"]
            mask_cpu = batch["mask"]
            st_cpu = batch["sleep_stages"]
            valid_cpu = batch["valid_ctx"]

            T = sig_cpu.shape[1]

            if phase == "train":
                optimizer.zero_grad()

            # ── Exame completo (CHUNK_T=-1) ───────────────────────────────
            if TrainingConfig.CHUNK_T == -1:
                sig = sig_cpu.to(DEVICE, non_blocking=True)
                mask = mask_cpu.to(DEVICE, non_blocking=True)
                st = st_cpu.to(DEVICE, non_blocking=True)

                with torch.autocast(device_type, dtype=torch.bfloat16, enabled=use_amp):
                    sl = model(sig, mask)                 # >>> alterado: só 1 retorno
                    #loss = criterion(sl, st, mask)         # >>> alterado: assinatura de StagingLoss
                    valid = valid_cpu.to(DEVICE, non_blocking=True)
                    loss_mask = mask & valid

                    # ignorando se for -1
                    if loss_mask.sum() == 0:
                        continue

                    loss = criterion(sl, st, loss_mask)


                if phase == "train":
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                loss_val = loss.item()
                sl = sl.detach().cpu()
                del sig, mask, st

            # ── Chunked ───────────────────────────────────────────────────
            else:
                batch_loss, n_chunks, all_sl = 0., 0, []
                mask_gpu = mask_cpu.to(DEVICE, non_blocking=True)

                for t0 in range(0, T, TrainingConfig.CHUNK_T):
                    t1 = min(t0 + TrainingConfig.CHUNK_T, T)
                    sig_c = sig_cpu[:, t0:t1].to(DEVICE, non_blocking=True)
                    st_c = st_cpu[:, t0:t1].to(DEVICE, non_blocking=True)
                    #mask_c = mask_gpu[:, t0:t1]
                    valid_c = valid_cpu[:, t0:t1].to(DEVICE, non_blocking=True)
                    mask_c = mask_gpu[:, t0:t1] & valid_c

                    # ignorando janelas com -1
                    if mask_c.sum() == 0:
                        continue

                    with torch.autocast(device_type, dtype=torch.bfloat16, enabled=use_amp):
                        mask_orig_c = mask_gpu[:, t0:t1]
                        loss_mask_c = mask_orig_c & valid_c
                        sl_c = model(sig_c, mask_orig_c)
                        loss_c = criterion(sl_c, st_c, loss_mask_c)


                        #sl_c = model(sig_c, mask_c)             # >>> alterado: só 1 retorno
                        #loss_c = criterion(sl_c, st_c, mask_c)  # >>> alterado

                    if phase == "train":
                        loss_c.backward()

                    batch_loss += loss_c.item()
                    n_chunks += 1
                    all_sl.append(sl_c.detach().cpu())
                    del sig_c, st_c, sl_c, loss_c

                if phase == "train":
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                loss_val = batch_loss / max(n_chunks, 1)

                if len(all_sl) == 0:
                  continue

                sl = torch.cat(all_sl, dim=1)   # (1, T, 5) — CPU
                del mask_gpu

            total_loss += loss_val
            n_batches += 1

            v = mask_cpu[0] & valid_cpu[0]
            if v.sum() > 0:
              sp.append(sl[0].argmax(-1)[v])
              st_.append(st_cpu[0][v])

            del sig_cpu, mask_cpu, st_cpu, sl



    # protegendo contra -1.
    if len(sp) == 0:
      return {
          "loss": total_loss / max(n_batches, 1),
          "stage": None,
      }

    sp = torch.cat(sp).numpy()
    st_ = torch.cat(st_).numpy()

    return {"loss": total_loss / max(n_batches, 1),
            "stage": compute_metrics(sp, st_, 5)}


def fmt(m):
    s = m.get("stage", {})
    parts = [f"loss={m['loss']:.4f}"]
    if s:
        parts.append(f"stg_f1={s.get('f1', 0):.3f} k={s.get('kappa', 0):.3f}")
    return " | ".join(parts)


# ----------------------------------------------------------------------------
# train_staging — substitui pretrain_staging. Mesmo corpo (pesos de classe,
# otimizador, scheduler, early stopping, logging), só sem a segunda fase
# de fine-tuning RSWA — esta já é a única e completa fase de treino.
# ----------------------------------------------------------------------------
def train_staging(model, train_subj, val_subj, fold_id=0, seed=None, global_mean=None, global_std=None,
                    logger=None):

    print(f"\n{'='*50}\nTREINO fold={fold_id}")
    print(f"  Train={len(train_subj)}  Val={len(val_subj)} exames")

    mem_monitor = CUDAMemoryMonitor(interval_sec=INTERVAL_SEC)

    fold_start = perf_counter()
    mem_monitor.start()

    tl = DataLoader(
        SleepAnalysisDataset(train_subj, global_mean=global_mean, global_std=global_std),
        batch_size=1, shuffle=True,
        num_workers=TrainingConfig.NUM_WORKERS, collate_fn=collate_sleep_analysis_exams,
        pin_memory=True, persistent_workers=True
    )
    vl = DataLoader(
        SleepAnalysisDataset(val_subj, global_mean=global_mean, global_std=global_std),
        batch_size=1, shuffle=False,
        num_workers=TrainingConfig.NUM_WORKERS, collate_fn=collate_sleep_analysis_exams,
        pin_memory=True, persistent_workers=True
    )

    # ── Pesos de classe calculados a partir do treino deste fold ──
    # (mesma lógica e mesma decisão documentada: pesos fixos entre todas
    # as variantes do ablation — ver documento de alterações técnicas,
    # item 4, Opção A.)
    all_stages = []
    for subj in train_subj:
        s = subj.sleep_stages
        all_stages.extend(s[s >= 0].tolist())
    counts = Counter(all_stages)
    total = sum(counts.values())
    weights = torch.tensor(
        [total / (5 * counts.get(c, 1)) for c in range(5)],
        dtype=torch.float32
    ).to(DEVICE)

    weights[1] *= TrainingConfig.N1_WEIGHT
    weights[4] *= TrainingConfig.REM_WEIGHT
    weights[2] *= TrainingConfig.N2_WEIGHT

    print(f"  Class weights: W={weights[0]:.2f} N1={weights[1]:.2f} "
          f"( N1 ADJUST RATE: {TrainingConfig.N1_WEIGHT} )"
          f"N2={weights[2]:.2f} N3={weights[3]:.2f} REM={weights[4]:.2f}")

    criterion = StagingLoss(stage_cls_w=weights).to(DEVICE)   # >>> alterado: StagingLoss em vez de MultiTaskLoss

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=TrainingConfig.LR_PRETRAIN,
        weight_decay=1e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    print(f"  Otimizador: AdamW com lr={TrainingConfig.LR_PRETRAIN} e weight_decay=1e-4")

    sch = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=TrainingConfig.EPOCHS_PRETRAIN, eta_min=1e-6
    )
    print(f"  Scheduler: CosineAnnealingLR com T_max={TrainingConfig.EPOCHS_PRETRAIN} e eta_min=1e-6")

    #ckpt = PathConfig.CKPT_DIR / f"staging_f{fold_id}.pt"
    ckpt = PathConfig.CKPT_DIR / f"{VARIANT_ID}_fold{fold_id}_seed{logger.seed}.pt"
    best, wait = 0., 0
    history = {"train_loss": [], "val_loss": [], "val_f1": []}

    if logger is not None:
        logger.start_train_timer()

    best_epoch = 0

    for ep in range(1, TrainingConfig.EPOCHS_PRETRAIN + 1):

        epoch_start = perf_counter()
        # train time
        train_start = perf_counter()
        print(f"epoch: {ep}")
        tm = run_epoch(model, tl, criterion, opt, "train")
        train_time = perf_counter() - train_start

        # validation time
        val_start = perf_counter()
        print(f"validation.. ")
        vm = run_epoch(model, vl, criterion, None, "val")
        val_time = perf_counter() - val_start

        epoch_time = perf_counter() - epoch_start
        print(
          f" ⚠️  METRIC_EPOCH ep{ep:03d} "
          f"train={train_time:7.1f}s "
          f"val={val_time:7.1f}s "
          f"total={epoch_time:7.1f}s "
          f"train[{fmt(tm)}] "
          f"val[{fmt(vm)}]"
      )

        vf1 = vm["stage"].get("f1", 0)
        sch.step()
        history["train_loss"].append(tm["loss"])
        history["val_loss"].append(vm["loss"])
        history["val_f1"].append(vf1)
        print(f"  ep{ep:3d}  train[{fmt(tm)}]  val[{fmt(vm)}]")

        if logger is not None:
            logger.log_epoch(
                phase="train_staging", epoch=ep,
                train_loss=tm["loss"], val_loss=vm["loss"],
                val_metrics=vm["stage"],
                lr=sch.get_last_lr()[0],
            )

        if vf1 > best:
            best, wait = vf1, 0
            best_epoch = ep
            save_ckpt(model, opt, ep, vm, ckpt)
        else:
            wait += 1
            if wait >= TrainingConfig.PATIENCE_PRETRAIN:
                print("  Early stop")
                break

    train_time_sec = logger.stop_train_timer() if logger is not None else None
    mem_records = mem_monitor.stop()

    fold_time = perf_counter() - fold_start

    print("=" * 70)
    print(f"METRIC_FOLD {fold_id} finalizado em {fold_time/60:.2f} minutos")
    print("=" * 70)

    df_mem = pd.DataFrame(mem_records)
    df_mem.to_csv(f"{RLOG_DIR}/cuda_memory_{VARIANT_ID}_seed{seed}_fold{fold_id}.csv", index=False)

    if mem_records:
      peak_alloc = max(r["max_allocated_mb"] for r in mem_records)
      peak_reserved = max(r["max_reserved_mb"] for r in mem_records)
      print(f"METRIC_CUDA peak allocated: {peak_alloc:.1f} MB")
      print(f"METRIC_CUDA peak reserved : {peak_reserved:.1f} MB")

    load_ckpt(model, ckpt)
    print(f"  Melhor val stage_f1={best:.4f}")


    return model, history, {"best_epoch": best_epoch, "train_time_sec": train_time_sec, "n_epochs_run": ep}


# ----------------------------------------------------------------------------
# evaluate_fold — mesma lógica de preservação de exam_id/epoch_idx/y_prob
# já validada, só sem o desempacotamento de rl/r_true/r_pred.
# ----------------------------------------------------------------------------
def evaluate_fold(model, test_subj, global_mean=None, global_std=None, logger=None):
    dl = DataLoader(
        SleepAnalysisDataset(test_subj, global_mean=global_mean, global_std=global_std),
        batch_size=1, shuffle=False,
        num_workers=TrainingConfig.NUM_WORKERS, collate_fn=collate_sleep_analysis_exams
    )
    model.eval()
    sp, st_ = [], []
    exam_ids_all, epoch_idxs_all, probs_all = [], [], []

    t0 = time.time()

    with torch.no_grad():
        for batch in dl:
            sig = batch["signals"].to(DEVICE); mask = batch["mask"].to(DEVICE)
            sl = model(sig, mask)                            # >>> alterado: só 1 retorno
            mnp = mask.cpu().numpy()[0]

            valid = batch["valid_ctx"].to(DEVICE)
            eval_mask = mask & valid
            mnp = eval_mask.cpu().numpy()[0]

            s_true = batch["sleep_stages"].numpy()[0][mnp]
            s_pred = sl[0].argmax(-1).cpu().numpy()[mnp]

            sp.extend(s_pred[s_true >= 0]); st_.extend(s_true[s_true >= 0])

            exam_id = batch["subject_ids"][0]
            valid_idx = np.where(mnp)[0]
            stage_valid_mask = s_true >= 0
            exam_ids_all.extend([exam_id] * stage_valid_mask.sum())
            epoch_idxs_all.extend(valid_idx[stage_valid_mask].tolist())

            probs = torch.softmax(sl[0], dim=-1).cpu().numpy()[mnp][stage_valid_mask]
            probs_all.append(probs)

    inference_time_sec = time.time() - t0

    if logger is not None:
        logger.log_predictions(
            exam_ids=exam_ids_all,
            epoch_idxs=epoch_idxs_all,
            y_true=np.array(st_),
            y_pred=np.array(sp),
            y_prob=np.concatenate(probs_all, axis=0),
        )

    stage_metrics = compute_metrics(sp, st_, 5)
    f1_per_class = compute_f1_per_class(sp, st_, n_cls=5)

    return {"stage": stage_metrics,
            "stage_preds": np.array(sp), "stage_trues": np.array(st_),
            "f1_per_class": f1_per_class,
            "inference_time_total_sec": inference_time_sec,
            "n_mini_epochs_test": len(st_)}


def compute_f1_per_class(preds, targets, n_cls=5):
    from sklearn.metrics import f1_score
    p, t = np.array(preds), np.array(targets)
    v = t >= 0
    p, t = p[v], t[v]
    class_names = ["W", "N1", "N2", "N3", "REM"]
    scores = f1_score(t, p, average=None, labels=list(range(n_cls)), zero_division=0)
    return dict(zip(class_names, scores))
# %% [cell 88]


start = datetime.now()

run_pipeline(subjects, seeds=[0])

elapsed = datetime.now() - start
total_seconds = int(elapsed.total_seconds())
h, rem = divmod(total_seconds, 3600)
m, s = divmod(rem, 60)
print(f"Início : {start:%d/%m/%Y %H:%M:%S}")
print(f"Fim    : {datetime.now():%d/%m/%Y %H:%M:%S}")
print(f"Duração: {h:02d}:{m:02d}:{s:02d}")

# %% [cell 89]
from datetime import datetime
end = datetime.now()
print(end)
# %% [cell 90]
from google.colab import runtime
runtime.unassign()