"""
Pacote de pre-processamento PSG: EDF + hipnograma (.mat) + RSWA (CSV) -> .pt.

Versao modular do notebook Parser_Exames. Diferencas em relacao ao notebook:
  - ordem de canais corrigida para casar com src/sleep_rswa/config.py;
  - rasterizacao de RSWA integrada (o .pt agora grava tonic/phasic/rswa labels).

Uso rapido
──────────
    from parser import run_preprocessing
    run_preprocessing(edf_dir=..., out_dir=..., mat_dir=..., rswa_dir=...)

    # ou, para um exame:
    from parser import preprocess_exam
    result = preprocess_exam(edf_path, mat_dir=..., rswa_dir=...)
"""
from .config import PathConfig, PSGConfig, FS_TARGET, EPOCH_SEC, N_CHANNELS
from .preprocess import preprocess_exam, run_preprocessing
from .parallel import run_preprocessing_parallel
from .rswa_labels import rasterize_rswa_annotations

__all__ = [
    "PathConfig",
    "PSGConfig",
    "FS_TARGET",
    "EPOCH_SEC",
    "N_CHANNELS",
    "preprocess_exam",
    "run_preprocessing",
    "run_preprocessing_parallel",
    "rasterize_rswa_annotations",
]
