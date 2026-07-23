import os
import sys
from datetime import datetime, timezone
import importlib
from pathlib import Path

import numpy as np
import pytest


def load_preprocessing_module(tmp_path, module_name: str):
    os.environ["MNE_USE_NUMBA"] = "false"
    os.environ["_MNE_FAKE_HOME_DIR"] = str(tmp_path)
    os.environ["MNE_DONTWRITE_HOME"] = "true"
    src_dir = Path(__file__).resolve().parents[1] / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    return importlib.import_module(module_name)


def load_parser_module(tmp_path):
    return load_preprocessing_module(tmp_path, "sleep_rswa.preprocessing.annotations")


def test_load_subject_annotations_from_csv_filters_and_sorts(tmp_path):
    parser_module = load_parser_module(tmp_path)
    csv_path = tmp_path / "annotations.csv"
    csv_path.write_text(
        "\n".join(
            [
                "subject_id,onset_s,duration_s,type",
                "rbd5,542.0,8.0,Tonic",
                "ctrl1,100.0,1.0,Ignored",
                "rbd5,310.5,2.0,Phasic",
            ]
        ),
        encoding="utf-8",
    )

    orig_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
    annotations = parser_module.load_subject_annotations_from_csv(
        csv_path=csv_path,
        subject_id="rbd5",
        orig_time=orig_time,
    )

    assert list(annotations.onset) == [310.5, 542.0]
    assert list(annotations.duration) == [2.0, 8.0]
    assert list(annotations.description) == ["Phasic", "Tonic"]
    assert annotations.orig_time == orig_time


def test_load_subject_annotations_from_csv_validates_header(tmp_path):
    parser_module = load_parser_module(tmp_path)
    csv_path = tmp_path / "annotations.csv"
    csv_path.write_text(
        "\n".join(
            [
                "subject_id,onset_s,type",
                "rbd5,310.5,Phasic",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="colunas ausentes"):
        parser_module.load_subject_annotations_from_csv(
            csv_path=csv_path,
            subject_id="rbd5",
            orig_time=None,
        )


def test_count_annotations_by_description(tmp_path):
    parser_module = load_parser_module(tmp_path)
    csv_path = tmp_path / "annotations.csv"
    csv_path.write_text(
        "\n".join(
            [
                "subject_id,onset_s,duration_s,type",
                "rbd5,100.0,2.0,Phasic",
                "rbd5,200.0,5.0,Tonic",
                "rbd5,300.0,1.5,Phasic",
            ]
        ),
        encoding="utf-8",
    )

    annotations = parser_module.load_subject_annotations_from_csv(
        csv_path=csv_path,
        subject_id="rbd5",
        orig_time=None,
    )

    counts = parser_module.count_annotations_by_description(annotations)

    assert counts == {"Phasic": 2, "Tonic": 1}


def test_find_annotations_csv_file_by_subject_id(tmp_path):
    parser_module = load_parser_module(tmp_path)
    csv_dir = tmp_path / "csv"
    csv_dir.mkdir()
    expected_path = csv_dir / "rbd5_rswa.csv"
    expected_path.write_text(
        "subject_id,onset_s,duration_s,type\n",
        encoding="utf-8",
    )

    found_path = parser_module.find_annotations_csv_file(
        subject_id="rbd5",
        csv_dir=csv_dir,
    )

    assert found_path == expected_path


def test_find_annotations_csv_file_returns_none_when_missing(tmp_path):
    parser_module = load_parser_module(tmp_path)
    csv_dir = tmp_path / "csv"
    csv_dir.mkdir()

    found_path = parser_module.find_annotations_csv_file(
        subject_id="rbd5",
        csv_dir=csv_dir,
    )

    assert found_path is None


def test_csv_annotations_can_merge_with_relative_edf_timebase(tmp_path):
    parser_module = load_parser_module(tmp_path)
    csv_path = tmp_path / "rbd5_rswa.csv"
    csv_path.write_text(
        "\n".join(
            [
                "subject_id,onset_s,duration_s,type",
                "rbd5,310.5,2.0,Phasic",
            ]
        ),
        encoding="utf-8",
    )

    old = parser_module.mne.Annotations(
        onset=[0.0],
        duration=[30.0],
        description=["Sleep stage R"],
        orig_time=None,
    )
    new = parser_module.load_subject_annotations_from_csv(
        csv_path=csv_path,
        subject_id="rbd5",
        orig_time=old.orig_time,
    )

    merged = old + new

    assert len(merged) == 2
    assert merged.orig_time is None


def test_rasterize_rswa_annotations_handles_missing_csv(tmp_path):
    rswa_module = load_preprocessing_module(
        tmp_path,
        "sleep_rswa.preprocessing.rswa_labels",
    )

    stages_mini = np.array([4, 4, -1, 2], dtype=np.int64)
    rswa = rswa_module.rasterize_rswa_annotations(
        csv_path=None,
        subject_id="brux2",
        stages_mini=stages_mini,
        annot_start=10.0,
    )

    assert np.array_equal(rswa["tonic_labels"], np.zeros(4, dtype=np.float32))
    assert np.array_equal(rswa["phasic_labels"], np.zeros(4, dtype=np.float32))
    assert np.array_equal(rswa["rswa_labels"], np.zeros(4, dtype=np.int64))
    assert np.array_equal(
        rswa["rswa_conf"],
        np.array([1.0, 1.0, 0.0, 1.0], dtype=np.float32),
    )
