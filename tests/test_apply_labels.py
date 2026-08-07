"""
Testes de regressao para classifier/apply_labels.py -- cobre o bug em que
eventos type=any eram silenciosamente descartados na leitura do CSV
revisado (read_label_csv) e nunca escritos no .pt (apply_one).

Cobre tambem a semantica de precedencia decidida para any_labels: se o CSV
revisado nao contem nenhuma linha type=any, o any_labels existente no .pt
(tipicamente escrito por classifier/auto_label.py) deve ser PRESERVADO, nao
zerado por omissao; se o CSV contem >=1 linha type=any, ela passa a ser a
fonte de verdade e sobrescreve integralmente.
"""
from __future__ import annotations

import csv
import importlib
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


def load_apply_labels_module():
    proj_root = Path(__file__).resolve().parents[1]
    if str(proj_root) not in sys.path:
        sys.path.insert(0, str(proj_root))
    module = importlib.import_module("classifier.apply_labels")
    return importlib.reload(module)


def _write_pt(path: Path, T: int, any_labels=None, label_source="auto_cnn_limiar_duplo_v1"):
    obj = {
        "signals": torch.zeros(T, 1, 300),
        "sleep_stages": torch.zeros(T, dtype=torch.int64),
        "channel_mask": torch.ones(1, dtype=torch.bool),
        "channel_names": ["emg"],
        "tonic_labels": torch.zeros(T),
        "phasic_labels": torch.zeros(T),
        "rswa_labels": torch.zeros(T, dtype=torch.int64),
        "rswa_conf": torch.ones(T),
        "label_source": label_source,
    }
    if any_labels is not None:
        obj["any_labels"] = torch.tensor(any_labels, dtype=torch.float32)
    torch.save(obj, path)


def _write_csv(path: Path, rows: list[tuple]):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["subject_id", "onset_s", "duration_s", "type", "score"])
        for row in rows:
            writer.writerow(row)


def test_read_label_csv_recognizes_any_type(tmp_path):
    al = load_apply_labels_module()
    csv_path = tmp_path / "examA_revisado.csv"
    _write_csv(csv_path, [
        ("examA", 0.0, 3.0, "tonic", 1.0),
        ("examA", 3.0, 3.0, "phasic", 1.0),
        ("examA", 6.0, 3.0, "any", 1.0),
    ])
    rows = al.read_label_csv(csv_path)
    types = sorted(r["type"] for r in rows)
    assert types == ["any", "phasic", "tonic"]


def test_apply_one_preserves_existing_any_when_csv_has_no_any_row(tmp_path, monkeypatch):
    al = load_apply_labels_module()
    data_dir = tmp_path / "data"
    labels_dir = tmp_path / "labels"
    data_dir.mkdir()
    labels_dir.mkdir()
    monkeypatch.setattr(al, "DATA_DIR", data_dir)
    monkeypatch.setattr(al, "LABELS_DIR", labels_dir)
    monkeypatch.setattr(al, "BACKUP_DIR", tmp_path / "data_backup")

    T = 10
    existing_any = [0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
    _write_pt(data_dir / "examA.pt", T, any_labels=existing_any)
    _write_csv(labels_dir / "examA_revisado.csv", [("examA", 0.0, 3.0, "tonic", 1.0)])

    summary = al.apply_one("examA", offsets={}, time_ref="pt", dry_run=False)

    assert summary["any_source"] == "preserved_existing"
    obj = torch.load(data_dir / "examA.pt", map_location="cpu", weights_only=False)
    assert np.array_equal(obj["any_labels"].numpy(), np.array(existing_any, dtype=np.float32))
    assert np.array_equal(obj["tonic_labels"].numpy(), np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32))
    assert obj["label_source"] == "human_reviewed"


def test_apply_one_overwrites_any_when_csv_has_any_row(tmp_path, monkeypatch):
    al = load_apply_labels_module()
    data_dir = tmp_path / "data"
    labels_dir = tmp_path / "labels"
    data_dir.mkdir()
    labels_dir.mkdir()
    monkeypatch.setattr(al, "DATA_DIR", data_dir)
    monkeypatch.setattr(al, "LABELS_DIR", labels_dir)
    monkeypatch.setattr(al, "BACKUP_DIR", tmp_path / "data_backup")

    T = 10
    existing_any = [0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
    _write_pt(data_dir / "examA.pt", T, any_labels=existing_any)
    _write_csv(labels_dir / "examA_revisado.csv", [
        ("examA", 0.0, 3.0, "tonic", 1.0),
        ("examA", 15.0, 3.0, "any", 1.0),
    ])

    summary = al.apply_one("examA", offsets={}, time_ref="pt", dry_run=False)

    assert summary["any_source"] == "human_reviewed_csv"
    obj = torch.load(data_dir / "examA.pt", map_location="cpu", weights_only=False)
    expected_any = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0], dtype=np.float32)
    assert np.array_equal(obj["any_labels"].numpy(), expected_any)


def test_apply_one_defaults_any_to_zero_when_never_present(tmp_path, monkeypatch):
    """Exame que nunca passou por auto_label.py (sem any_labels no .pt) e cujo
    CSV revisado tambem nao tem linha any -> any_labels final e zeros, nao erro."""
    al = load_apply_labels_module()
    data_dir = tmp_path / "data"
    labels_dir = tmp_path / "labels"
    data_dir.mkdir()
    labels_dir.mkdir()
    monkeypatch.setattr(al, "DATA_DIR", data_dir)
    monkeypatch.setattr(al, "LABELS_DIR", labels_dir)
    monkeypatch.setattr(al, "BACKUP_DIR", tmp_path / "data_backup")

    T = 10
    _write_pt(data_dir / "examA.pt", T, any_labels=None)
    _write_csv(labels_dir / "examA_revisado.csv", [("examA", 0.0, 3.0, "tonic", 1.0)])

    summary = al.apply_one("examA", offsets={}, time_ref="pt", dry_run=False)

    assert summary["any_source"] == "preserved_existing"
    obj = torch.load(data_dir / "examA.pt", map_location="cpu", weights_only=False)
    assert np.array_equal(obj["any_labels"].numpy(), np.zeros(T, dtype=np.float32))
