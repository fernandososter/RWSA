import importlib
import os
import sys
from pathlib import Path

import numpy as np
import pytest


def load_auto_rswa_module(tmp_path):
    os.environ["MNE_USE_NUMBA"] = "false"
    os.environ["_MNE_FAKE_HOME_DIR"] = str(tmp_path)
    os.environ["MNE_DONTWRITE_HOME"] = "true"
    src_dir = Path(__file__).resolve().parents[1] / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    module = importlib.import_module("sleep_rswa.preprocessing.auto_rswa")
    return importlib.reload(module)


def test_events_to_labels_maps_tonic_phasic_any(tmp_path):
    auto = load_auto_rswa_module(tmp_path)

    events = [
        {"onset_s": 0.0, "duration_s": 3.0, "type": "tonic", "score": 3.0},
        {"onset_s": 3.0, "duration_s": 3.0, "type": "phasic", "score": 3.0},
        {"onset_s": 6.0, "duration_s": 3.0, "type": "any", "score": 3.0},
    ]

    labels = auto.events_to_labels(events, n_epochs=4)

    assert np.array_equal(labels["tonic_labels"], np.array([1, 0, 0, 0], dtype=np.float32))
    assert np.array_equal(labels["phasic_labels"], np.array([0, 1, 0, 0], dtype=np.float32))
    assert np.array_equal(labels["any_labels"], np.array([0, 0, 1, 0], dtype=np.float32))


def test_auto_label_rswa_from_signals_uses_cnn_candidates_and_window_detector(tmp_path, monkeypatch):
    auto = load_auto_rswa_module(tmp_path)

    dummy_checkpoint = tmp_path / "movement_cnn_final.pt"
    dummy_checkpoint.write_bytes(b"fake")

    def fake_load_model(model_path, device):
        assert Path(model_path) == dummy_checkpoint
        return object(), {"window_epochs": 3, "threshold": 0.7}

    def fake_predict_scores(model, windows, device, batch_size=512):
        del model, windows, device, batch_size
        return np.array([0.1, 0.8, 0.8, 0.2], dtype=np.float32)

    def fake_detect_in_window(emg_flat, start_s, end_s, *, k_on, k_off, off_hold_s):
        del emg_flat
        assert start_s == pytest.approx(3.0)
        assert end_s == pytest.approx(9.0)
        assert k_on == pytest.approx(4.0)
        assert k_off == pytest.approx(2.0)
        assert off_hold_s == pytest.approx(0.25)
        return [
            {"onset_s": 3.0, "duration_s": 6.0, "type": "any", "score": 2.5},
        ]

    monkeypatch.setattr(auto, "_load_movement_cnn", fake_load_model)
    monkeypatch.setattr(auto, "_predict_movement_scores", fake_predict_scores)
    monkeypatch.setattr(auto, "detect_in_window_with_params", fake_detect_in_window)

    signals = np.zeros((4, 5, 300), dtype=np.float32)
    sleep_stages = np.array([4, 4, 4, -1], dtype=np.int64)

    result = auto.auto_label_rswa_from_signals(
        signals,
        sleep_stages,
        model_path=dummy_checkpoint,
        device="cpu",
        k_on=4.0,
        k_off=2.0,
        k_off_hold_s=0.25,
    )

    assert result["n_cnn_candidates"] == 1
    assert result["n_confirmed_events"] == 1
    assert result["n_discarded_windows"] == 0
    assert result["cnn_threshold"] == pytest.approx(0.7)
    assert result["k_on"] == pytest.approx(4.0)
    assert result["k_off"] == pytest.approx(2.0)
    assert result["k_off_hold_s"] == pytest.approx(0.25)
    assert np.array_equal(result["any_labels"], np.array([0, 1, 1, 0], dtype=np.float32))
    assert np.array_equal(result["tonic_labels"], np.zeros(4, dtype=np.float32))
    assert np.array_equal(result["phasic_labels"], np.zeros(4, dtype=np.float32))
    assert np.array_equal(result["rswa_labels"], np.zeros(4, dtype=np.int64))
    assert np.array_equal(result["rswa_conf"], np.array([1.0, 1.0, 1.0, 0.0], dtype=np.float32))
    assert result["label_source"] == "auto_cnn_limiar_duplo_v1"
