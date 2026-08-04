from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from parser import PathConfig, run_preprocessing, run_preprocessing_parallel


def start(parallel_exec: bool = True) -> None:
    if parallel_exec:
        run_preprocessing_parallel(edf_dir=PathConfig.EDF_DIR)
    else:
        run_preprocessing(edf_dir=PathConfig.EDF_DIR)

    return None


if __name__ == "__main__":
    start(False)
