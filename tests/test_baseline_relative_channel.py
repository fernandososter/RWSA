"""Testes dos canais baseline-relative (SleepAnalysisDataset._extract_emg).

Desde 2026-08-14, quando use_baseline_relative_channel=True, o ramo EMG
passa a usar DOIS canais relativos ao basal de atonia:
  1. EMG assinado / baseline (clipado)
  2. log1p(|EMG| / baseline) / log(log_base)

Assim, ambos ficam na mesma familia fisiologica e o valor 1.0 do segundo
canal corresponde exatamente a 4x o basal quando log_base=5.0.
"""
import torch
import math

import tempfile
from pathlib import Path

from sleep_rswa.config import RSWAConfig
from sleep_rswa.data import SleepAnalysisDataset, SubjectData, load_subject_file


def _make_subject(rem_baseline_uv=None, atonia_baseline_uv=None, n_epochs=20, emg_scale=1e-5):
    signals = torch.randn(n_epochs, 5, 300) * emg_scale
    stages = torch.zeros(n_epochs, dtype=torch.long)
    rswa = torch.zeros(n_epochs, dtype=torch.long)
    conf = torch.ones(n_epochs)
    return SubjectData(
        subject_id="dummy",
        signals=signals,
        sleep_stages=stages,
        rswa_labels=rswa,
        rswa_conf=conf,
        rem_baseline_uv=rem_baseline_uv,
        atonia_baseline_uv=atonia_baseline_uv,
    )


def test_baseline_relative_channel_disabled_returns_single_channel():
    ds = SleepAnalysisDataset([_make_subject()], use_baseline_relative_channel=False)
    emg = ds._extract_emg(_make_subject(rem_baseline_uv=0.05, atonia_baseline_uv=0.30))
    assert emg.shape[1] == 1


def test_baseline_relative_channel_uses_atonia_baseline_when_available():
    """Com label_metadata.aasm_rule presente, o canal deve normalizar por
    atonia_baseline_uv, NAO por rem_baseline_uv."""
    ds = SleepAnalysisDataset([_make_subject()], use_baseline_relative_channel=True)
    rem_baseline_uv = 0.05
    atonia_baseline_uv = 0.30  # 6x maior, como observado empiricamente
    subj = _make_subject(rem_baseline_uv=rem_baseline_uv, atonia_baseline_uv=atonia_baseline_uv)
    emg = ds._extract_emg(subj)
    assert emg.shape[1] == 2

    raw_emg = subj.signals[:, ds.rswa_config.emg_channel_index, :]
    expected_signed = (raw_emg / (atonia_baseline_uv / 1e6)).clamp(
        -ds.rswa_config.baseline_relative_signed_clamp,
        ds.rswa_config.baseline_relative_signed_clamp,
    )
    expected_amplitude = (
        torch.log1p(raw_emg.abs() / (atonia_baseline_uv / 1e6))
        / math.log(ds.rswa_config.baseline_relative_log_base)
    ).clamp(0.0, ds.rswa_config.baseline_relative_amplitude_clamp)
    torch.testing.assert_close(emg[:, 0, :], expected_signed)
    torch.testing.assert_close(emg[:, 1, :], expected_amplitude)

    # Confirma que NAO e' igual ao calculo antigo (normalizando por rem_baseline_uv).
    wrong = (
        torch.log1p(raw_emg.abs() / (rem_baseline_uv / 1e6))
        / math.log(ds.rswa_config.baseline_relative_log_base)
    ).clamp(0.0, ds.rswa_config.baseline_relative_amplitude_clamp)
    assert not torch.allclose(emg[:, 1, :], wrong)


def test_baseline_relative_channel_legacy_fallback_uses_ratio():
    """Exame legado sem label_metadata.aasm_rule (atonia_baseline_uv=None):
    aproxima a baseline real como rem_baseline_uv * fallback_ratio."""
    ds = SleepAnalysisDataset([_make_subject()], use_baseline_relative_channel=True)
    rem_baseline_uv = 0.05
    subj = _make_subject(rem_baseline_uv=rem_baseline_uv, atonia_baseline_uv=None)
    emg = ds._extract_emg(subj)

    raw_emg = subj.signals[:, ds.rswa_config.emg_channel_index, :]
    approx_baseline_uv = rem_baseline_uv * ds.rswa_config.baseline_relative_fallback_ratio
    expected_signed = (raw_emg / (approx_baseline_uv / 1e6)).clamp(
        -ds.rswa_config.baseline_relative_signed_clamp,
        ds.rswa_config.baseline_relative_signed_clamp,
    )
    expected_amplitude = (
        torch.log1p(raw_emg.abs() / (approx_baseline_uv / 1e6))
        / math.log(ds.rswa_config.baseline_relative_log_base)
    ).clamp(0.0, ds.rswa_config.baseline_relative_amplitude_clamp)
    torch.testing.assert_close(emg[:, 0, :], expected_signed)
    torch.testing.assert_close(emg[:, 1, :], expected_amplitude)


def test_baseline_relative_channel_returns_zero_when_no_baseline_available():
    ds = SleepAnalysisDataset([_make_subject()], use_baseline_relative_channel=True)
    subj_none = _make_subject(rem_baseline_uv=None, atonia_baseline_uv=None)
    emg_none = ds._extract_emg(subj_none)
    assert torch.all(emg_none[:, 0, :] == 0)
    assert torch.all(emg_none[:, 1, :] == 0)

    subj_zero = _make_subject(rem_baseline_uv=0.0, atonia_baseline_uv=None)
    emg_zero = ds._extract_emg(subj_zero)
    assert torch.all(emg_zero[:, 0, :] == 0)
    assert torch.all(emg_zero[:, 1, :] == 0)


def test_baseline_relative_clamps_are_respected():
    ds = SleepAnalysisDataset([_make_subject()], use_baseline_relative_channel=True)
    subj = _make_subject(rem_baseline_uv=0.01, atonia_baseline_uv=0.001)  # forca saturacao
    emg = ds._extract_emg(subj)
    cfg = RSWAConfig()
    assert float(emg[:, 0, :].abs().max()) <= cfg.baseline_relative_signed_clamp
    assert float(emg[:, 1, :].max()) <= cfg.baseline_relative_amplitude_clamp


def test_baseline_relative_amplitude_channel_maps_4x_baseline_to_one():
    ds = SleepAnalysisDataset([_make_subject()], use_baseline_relative_channel=True)
    baseline_uv = 2.0
    baseline_v = baseline_uv / 1e6
    signals = torch.zeros(1, 5, 300)
    signals[:, ds.rswa_config.emg_channel_index, :] = 4.0 * baseline_v
    subj = SubjectData(
        subject_id="dummy_4x",
        signals=signals,
        sleep_stages=torch.zeros(1, dtype=torch.long),
        rswa_labels=torch.zeros(1, dtype=torch.long),
        rswa_conf=torch.ones(1),
        rem_baseline_uv=baseline_uv / ds.rswa_config.baseline_relative_fallback_ratio,
        atonia_baseline_uv=baseline_uv,
    )
    emg = ds._extract_emg(subj)
    torch.testing.assert_close(emg[:, 1, :], torch.ones_like(emg[:, 1, :]))
