from sleep_rswa.preprocessing import (
    PathConfig,
    run_preprocessing,
    run_preprocessing_parallel,
)


def start(parallel_exec: bool = True) -> None:
    if parallel_exec:
        run_preprocessing_parallel(edf_dir=PathConfig.EDF_DIR)
    else:
        run_preprocessing(edf_dir=PathConfig.EDF_DIR)

    return None


if __name__ == "__main__":
    start(False)
