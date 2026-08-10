import importlib
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


def load_view_app_module():
    root_dir = Path(__file__).resolve().parents[1]
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))
    module = importlib.import_module("view.app")
    return importlib.reload(module)


def test_prepare_review_reads_label_metadata_and_events(tmp_path):
    app = load_view_app_module()
    app._REVISAO_CACHE.clear()
    app.CFG["data_dir"] = tmp_path
    app.CFG["revisao_out_dir"] = tmp_path / "revisao"

    exam_path = tmp_path / "exam_a.pt"
    torch.save(
        {
            "signals": torch.zeros((4, 5, 300), dtype=torch.float32),
            "sleep_stages": torch.tensor([4, 4, 4, -1], dtype=torch.int64),
            "tonic_labels": torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
            "phasic_labels": torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float32),
            "any_labels": torch.tensor([0.0, 0.0, 1.0, 0.0], dtype=torch.float32),
            "tonic_cov": torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
            "phasic_cov": torch.tensor([0.0, 0.75, 0.0, 0.0], dtype=torch.float32),
            "any_cov": torch.tensor([0.0, 0.0, 0.5, 0.0], dtype=torch.float32),
            "label_source": "auto_cnn_limiar_duplo_v1",
            "label_metadata": {
                "rswa_source": "auto",
                "label_source": "auto_cnn_limiar_duplo_v1",
                "auto_label": {
                    "cnn_threshold": 0.2,
                    "k_on": 3.0,
                    "k_off": 1.5,
                    "k_off_hold_s": 0.0,
                    "n_confirmed_events": 3,
                },
            },
        },
        exam_path,
    )

    st = app._prepare_review("exam_a")

    assert st["label_source"] == "auto_cnn_limiar_duplo_v1"
    assert st["label_metadata"]["rswa_source"] == "auto"
    assert st["label_metadata"]["auto_label"]["k_on"] == pytest.approx(3.0)
    assert st["label_metadata"]["auto_label"]["k_off"] == pytest.approx(1.5)
    assert st["event_counts"] == {"tonic": 1, "phasic": 1, "any": 1, "total": 3}

    assert [ev["type"] for ev in st["events"]] == ["tonic", "phasic", "any"]
    assert [ev["onset_s"] for ev in st["events"]] == [0.0, 3.0, 6.0]
    assert st["events"][0]["mean_coverage"] == pytest.approx(1.0)
    assert st["events"][1]["mean_coverage"] == pytest.approx(0.75)
    assert st["events"][2]["mean_coverage"] == pytest.approx(0.5)
