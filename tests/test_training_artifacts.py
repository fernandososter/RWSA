import csv
import time

import numpy as np
import torch

from sleep_rswa.training import (
    ResourceMonitor,
    save_rswa_predictions_csv,
    save_staging_predictions_csv,
)


def test_save_staging_predictions_csv(tmp_path):
    path = tmp_path / "staging.csv"
    result = {
        "subject_id": np.asarray(["s1", "s1"], dtype=object),
        "mini_epoch_index": np.asarray([0, 1], dtype=np.int64),
        "expected": np.asarray([0, 4], dtype=np.int64),
        "prediction": np.asarray([0, 2], dtype=np.int64),
        "probabilities": np.asarray(
            [[0.8, 0.1, 0.05, 0.03, 0.02], [0.1, 0.2, 0.4, 0.2, 0.1]],
            dtype=np.float32,
        ),
    }
    save_staging_predictions_csv(path, result, split="validation", fold=1, source="best_checkpoint")
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 2
    assert rows[0]["subject_id"] == "s1"
    assert rows[0]["expected"] == "0"
    assert rows[0]["prediction"] == "0"
    assert "prob_W" in rows[0]
    assert "prob_REM" in rows[0]


def test_save_rswa_predictions_csv(tmp_path):
    path = tmp_path / "rswa.csv"
    result = {
        "subject_id": np.asarray(["s1", "s1"], dtype=object),
        "mini_epoch_index": np.asarray([0, 1], dtype=np.int64),
        "tonic_expected": np.asarray([0, 1], dtype=np.int64),
        "tonic_prediction": np.asarray([0, 0], dtype=np.int64),
        "tonic_probability": np.asarray([0.1, 0.4], dtype=np.float32),
        "phasic_expected": np.asarray([1, 0], dtype=np.int64),
        "phasic_prediction": np.asarray([1, 1], dtype=np.int64),
        "phasic_probability": np.asarray([0.8, 0.7], dtype=np.float32),
        "any_expected": np.asarray([1, 1], dtype=np.int64),
        "any_prediction": np.asarray([1, 1], dtype=np.int64),
        "any_probability": np.asarray([0.9, 0.95], dtype=np.float32),
        "movement_expected": np.asarray([1, 1], dtype=np.int64),
        "movement_prediction": np.asarray([1, 1], dtype=np.int64),
        "movement_probability": np.asarray([0.9, 0.95], dtype=np.float32),
    }
    save_rswa_predictions_csv(path, result, split="test", fold=None, source="ensemble")
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 2
    assert rows[1]["tonic_expected"] == "1"
    assert rows[1]["phasic_prediction"] == "1"
    assert "movement_probability" in rows[1]


def test_resource_monitor_writes_csv(tmp_path):
    path = tmp_path / "resource_usage.csv"
    with ResourceMonitor(path, device=torch.device("cpu"), interval_sec=1.0):
        time.sleep(1.2)
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) >= 2
    assert "cpu_percent" in rows[0]
    assert "process_rss_mb" in rows[0]
