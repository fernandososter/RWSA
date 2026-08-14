"""Testes do canal auxiliar 'baseline-relative' (SleepAnalysisDataset._extract_emg).

Cobre o bug corrigido em 2026-08-14: o canal normalizava por rem_baseline_uv
(baseline crua de baixo percentil) em vez de atonia_baseline_uv (a baseline
REAL usada pela regra AASM para decidir tonico/fasico -- ~5-6x maior),
fazendo o canal saturar no clamp exatamente na faixa de amplitude que
precisa discriminar evento de nao-evento. Mesma classe do bug corrigido na
UI de revisao (limiar 2x hardcoded vs limiar real da regra).
"""
import torch

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

    # Reconstroi o canal esperado usando atonia_baseline_uv e compara.
    raw_emg = subj.signals[:, ds.rswa_config.emg_channel_index, :].abs()
    expected = (raw_emg / (atonia_baseline_uv / 1e6)).clamp(0.0, ds.rswa_config.baseline_relative_clamp)
    torch.testing.assert_close(emg[:, 1, :], expected)

    # Confirma que NAO e' igual ao calculo antigo (normalizando por rem_baseline_uv).
    wrong = (raw_emg / (rem_baseline_uv / 1e6)).clamp(0.0, 20.0)
    assert not torch.allclose(emg[:, 1, :], wrong)


def test_baseline_relative_channel_legacy_fallback_uses_ratio():
    """Exame legado sem label_metadata.aasm_rule (atonia_baseline_uv=None):
    aproxima a baseline real como rem_baseline_uv * fallback_ratio."""
    ds = SleepAnalysisDataset([_make_subject()], use_baseline_relative_channel=True)
    rem_baseline_uv = 0.05
    subj = _make_subject(rem_baseline_uv=rem_baseline_uv, atonia_baseline_uv=None)
    emg = ds._extract_emg(subj)

    raw_emg = subj.signals[:, ds.rswa_config.emg_channel_index, :].abs()
    approx_baseline_uv = rem_baseline_uv * ds.rswa_config.baseline_relative_fallback_ratio
    expected = (raw_emg / (approx_baseline_uv / 1e6)).clamp(0.0, ds.rswa_config.baseline_relative_clamp)
    torch.testing.assert_close(emg[:, 1, :], expected)


def test_baseline_relative_channel_returns_zero_when_no_baseline_available():
    ds = SleepAnalysisDataset([_make_subject()], use_baseline_relative_channel=True)
    subj_none = _make_subject(rem_baseline_uv=None, atonia_baseline_uv=None)
    emg_none = ds._extract_emg(subj_none)
    assert torch.all(emg_none[:, 1, :] == 0)

    subj_zero = _make_subject(rem_baseline_uv=0.0, atonia_baseline_uv=None)
    emg_zero = ds._extract_emg(subj_zero)
    assert torch.all(emg_zero[:, 1, :] == 0)


def test_baseline_relative_clamp_is_configurable_via_rswa_config():
    ds = SleepAnalysisDataset([_make_subject()], use_baseline_relative_channel=True)
    subj = _make_subject(rem_baseline_uv=0.01, atonia_baseline_uv=0.001)  # forca saturacao
    emg = ds._extract_emg(subj)
    assert float(emg[:, 1, :].max()) <= RSWAConfig().baseline_relative_clamp
