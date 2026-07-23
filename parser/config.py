"""
Configuracao do parser PSG (EDF -> tensores .pt).

Convertido do notebook Parser_Exames. Uma CORRECAO importante foi aplicada
em relacao ao notebook: a ordem dos canais em CHANNEL_DEFS foi trocada para
[eeg_f4, eeg_c4, eeg_o2, eog, emg] de modo a casar com src/sleep_rswa/config.py:
    - staging_channel_indices = (0,1,2,3)  -> 3 EEG + EOG   (eeg_in=3, eog_in=1)
    - emg_channel_index       = 4          -> EMG (mento), usado pelo RSWA
No notebook original o EMG estava no indice 3 e o EOG no 4 (trocados), o que
alimentava o detector de RSWA com o canal do olho. NAO reverta sem tambem
ajustar src/config.py.
"""
from __future__ import annotations

from pathlib import Path


class PathConfig:
    """
    Diretorios de entrada/saida. Ajuste conforme o ambiente (Colab/local).
    Os defaults podem ser sobrescritos por variaveis de ambiente.
    """
    import os as _os

    EDF_DIR = Path(_os.environ.get(
        "EDF_DIR", "/content/drive/MyDrive/Cursos/RWSA/exames"))
    # hipnogramas .mat (hyp_<subject>.mat) e anotacoes RSWA (<subject>_rswa.csv)
    MAT_DIR = Path(_os.environ.get("MAT_DIR", str(EDF_DIR / "converted")))
    RSWA_DIR = Path(_os.environ.get("RSWA_DIR", str(EDF_DIR / "rswa_annotations")))
    TENSOR_DIR = Path(_os.environ.get("TENSOR_DIR", str(EDF_DIR.parent / "tensors")))


# Parametros de sinal
FS_TARGET = 100                 # Hz apos reamostragem
EPOCH_SEC = 3                   # segundos por mini-epoca
N_SAMPLES = FS_TARGET * EPOCH_SEC  # 300 amostras por mini-epoca

# Filtros por tipo de canal (ajustados apos conversa clinica)
FILTER_PARAMS = {
    "emg": {"l_freq": 10.0, "h_freq": 40.0},
    "eeg": {"l_freq": 0.1,  "h_freq": 35.0},
    "eog": {"l_freq": 0.1,  "h_freq": 35.0},
}
NOTCH_FILTER = 60.0             # Hz

# Colunas obrigatorias do CSV de RSWA
CSV_ANNOTATION_COLUMNS = {"subject_id", "onset_s", "duration_s", "type"}


class PSGConfig:
    # Mapeamento de labels de staging (case-insensitive apos .upper()).
    STAGE_MAP = {
        "W": 0, "WAKE": 0, "SLEEP-S0": 0, "S0": 0, "A": 0,
        "N1": 1, "SLEEP-S1": 1, "S1": 1,
        "N2": 2, "SLEEP-S2": 2, "S2": 2,
        "N3": 3, "N4": 3, "SLEEP-S3": 3, "SLEEP-S4": 3, "S3": 3, "S4": 3,
        "R": 4, "REM": 4, "SLEEP-REM": 4,
        "U": -1, "?": -1, "UNSCORED": -1, "M": -1, "MT": -1, "Artefato": -1,
    }

    # Definicao de canais (tolerante a canais ausentes).
    # ORDEM: 3 EEG, EOG, EMG  -> casa com src/config.py (ver docstring do modulo).
    #   "candidates": nomes alternativos aceitos (case-insensitive)
    #   "filter":     tipo de filtro a aplicar
    #   "required":   se False, canal ausente -> zeros + channel_mask[i]=False
    CHANNEL_DEFS = [
        {
            "name": "eeg_f4",
            "candidates": ["EEG F4-M1", "F4-M1", "EEG F4-A1", "F4-A1", "F4", "F4A1",
                           "EEG F4", "F4-A2", "EEG F4-A2", "F2-F4", "Fp2-F4"],
            "filter": "eeg", "required": False,
        },
        {
            "name": "eeg_c4",
            "candidates": ["EEG C4-M1", "C4-M1", "EEG C4-A1", "C4-A1", "C4", "C4A1",
                           "EEG C4", "EEG Fpz-Cz", "C4-A2", "F4-C4"],
            "filter": "eeg", "required": False,
        },
        {
            "name": "eeg_o2",
            "candidates": ["EEG O2-M1", "O2-M1", "EEG O2-A1", "O2-A1", "O2", "O2A1",
                           "EEG O2", "O2-A2", "P4-O2"],
            "filter": "eeg", "required": False,
        },
        {   # EOG (indice 3) — usado no staging junto dos 3 EEG
            "name": "eog",
            "candidates": ["EOG E2-M2", "E2-M2", "ROC-M1", "EOG-R", "EOG ROC-A1",
                           "ROC-A1", "ROC-A2", "E2-M1", "ROC", "EOG R", "EOG2",
                           "ROC-LOC", "ROC / A1",
                           # candidatos do olho esquerdo, aceitos como fallback
                           "EOG E1-M2", "E1-M2", "LOC-M2", "LOC-A2", "LOC", "EOG-L",
                           "EOG1", "LOC-ROC"],
            "filter": "eog", "required": False,
        },
        {   # EMG mento (indice 4) — canal usado pelo detector de RSWA
            "name": "emg",
            "candidates": ["EMG chin", "EMG Chin", "Chin EMG", "CHIN1", "chin", "CHIN",
                           "EMG", "EMG1-EMG2", "EMG-EMG", "EMG1", "EMG2", "Chin",
                           "submental", "Submentalis", "chin-0", "chin-1",
                           "EMG submental"],
            "filter": "emg", "required": False,
        },
    ]


N_CHANNELS = len(PSGConfig.CHANNEL_DEFS)  # 5
