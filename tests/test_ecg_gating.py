"""
Testes deterministicos para src/sleep_rswa/preprocessing/ecg_gating.py.

Cobre os 3 blocos do modulo:
  1. detect_r_peaks       — deteccao de picos-R via neurokit2 (+ casos de borda)
  2. gate_emg_by_ecg       — interpolacao/gating do EMG nas janelas dos picos-R
  3. apply_ecg_gating_to_raw — integracao com mne.io.Raw (localizar canal ECG,
                               gatear EMG in-place, remover canal ECG)

Sinais sinteticos com resultado conhecido (sem depender de dados reais de
PhysioNet), para que os testes sejam rapidos e deterministicos.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("MNE_USE_NUMBA", "false")

src_dir = Path(__file__).resolve().parents[1] / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from sleep_rswa.preprocessing import ecg_gating as eg  # noqa: E402

FS = 100.0


# ─────────────────────────────────────────────────────────────────────────
# detect_r_peaks
# ─────────────────────────────────────────────────────────────────────────
class TestDetectRPeaks:
    def _simulate_ecg(self, duration_s=20, heart_rate=60, random_state=42):
        import neurokit2 as nk
        return nk.ecg_simulate(
            duration=duration_s, sampling_rate=int(FS),
            heart_rate=heart_rate, random_state=random_state,
        )

    def test_detects_approximately_correct_number_of_beats(self):
        duration_s, hr = 20, 60
        ecg = self._simulate_ecg(duration_s=duration_s, heart_rate=hr)
        r_peaks = eg.detect_r_peaks(ecg, FS)
        expected = duration_s * hr / 60.0
        assert abs(len(r_peaks) - expected) <= 2

    def test_r_peaks_are_sorted_and_evenly_spaced_at_60bpm(self):
        ecg = self._simulate_ecg(duration_s=20, heart_rate=60)
        r_peaks = eg.detect_r_peaks(ecg, FS)
        assert np.all(np.diff(r_peaks) > 0)  # estritamente crescente
        intervals_s = np.diff(r_peaks) / FS
        # 60 bpm -> ~1.0s entre batimentos; tolerancia generosa p/ variabilidade
        assert np.all(np.abs(intervals_s - 1.0) < 0.15)

    def test_returns_empty_array_for_empty_signal(self):
        r_peaks = eg.detect_r_peaks(np.array([]), FS)
        assert isinstance(r_peaks, np.ndarray)
        assert len(r_peaks) == 0

    def test_returns_empty_array_for_signal_shorter_than_half_second(self):
        r_peaks = eg.detect_r_peaks(np.zeros(10), FS)  # 0.1s a 100Hz
        assert len(r_peaks) == 0

    def test_returns_empty_array_for_flat_signal_no_beats(self):
        r_peaks = eg.detect_r_peaks(np.full(2000, 0.001), FS)
        assert len(r_peaks) == 0

    def test_all_peaks_within_valid_signal_range(self):
        ecg = self._simulate_ecg(duration_s=15, heart_rate=75)
        r_peaks = eg.detect_r_peaks(ecg, FS)
        assert len(r_peaks) > 0
        assert r_peaks.min() >= 0
        assert r_peaks.max() < len(ecg)


# ─────────────────────────────────────────────────────────────────────────
# gate_emg_by_ecg
# ─────────────────────────────────────────────────────────────────────────
class TestGateEmgByEcg:
    def _make_spiky_emg(self, n=1000, spike_uv=50e-6, seed=0):
        rng = np.random.RandomState(seed)
        emg = rng.normal(0, 1e-6, n)
        r_peaks = np.array([100, 250, 400, 550, 700, 850])
        for p in r_peaks:
            emg[p - 2:p + 3] += spike_uv
        return emg, r_peaks

    def test_removes_known_spike_amplitude_at_each_r_peak(self):
        emg, r_peaks = self._make_spiky_emg()
        emg_gated, n_gated = eg.gate_emg_by_ecg(
            emg, r_peaks, FS, window_pre_s=0.05, window_post_s=0.05,
        )
        half_w = int(round(0.05 * FS))
        for p in r_peaks:
            window_before = emg[p - half_w:p + half_w + 1]
            window_after = emg_gated[p - half_w:p + half_w + 1]
            assert window_before.max() > 40e-6  # espicula original presente
            assert window_after.max() < 5e-6    # espicula removida por interpolacao

    def test_gated_sample_count_matches_union_of_windows(self):
        emg, r_peaks = self._make_spiky_emg()
        half_w = int(round(0.05 * FS))
        emg_gated, n_gated = eg.gate_emg_by_ecg(
            emg, r_peaks, FS, window_pre_s=0.05, window_post_s=0.05,
        )
        expected = len(r_peaks) * (2 * half_w + 1)  # picos isolados, sem sobreposicao
        assert n_gated == expected

    def test_asymmetric_window_gates_more_samples_after_peak_than_before(self):
        # window_post_s > window_pre_s (default do modulo): a janela de gating
        # deve se estender mais para a direita do pico-R do que para a esquerda.
        n = 1000
        emg = np.zeros(n)
        r_peaks = np.array([500])
        window_pre_s, window_post_s = 0.05, 0.15
        emg_gated, n_gated = eg.gate_emg_by_ecg(
            emg, r_peaks, FS, window_pre_s=window_pre_s, window_post_s=window_post_s,
        )
        half_w_pre = int(round(window_pre_s * FS))   # 5 amostras
        half_w_post = int(round(window_post_s * FS))  # 15 amostras
        assert half_w_post > half_w_pre
        expected = half_w_pre + half_w_post + 1  # [500-5, 500+15]
        assert n_gated == expected
        # amostra imediatamente apos +half_w_pre (ainda dentro da janela assimetrica
        # do lado "post") deve ter sido alterada, confirmando a extensao maior
        emg_spiky = emg.copy()
        emg_spiky[500 + half_w_pre + 1] = 99.0
        emg_gated2, _ = eg.gate_emg_by_ecg(
            emg_spiky, r_peaks, FS, window_pre_s=window_pre_s, window_post_s=window_post_s,
        )
        assert emg_gated2[500 + half_w_pre + 1] != 99.0

    def test_overlapping_windows_are_merged_without_double_counting(self):
        n = 200
        emg = np.zeros(n)
        r_peaks = np.array([50, 55])  # janelas de 0.1s (10 amostras) se sobrepoem
        emg_gated, n_gated = eg.gate_emg_by_ecg(
            emg, r_peaks, FS, window_pre_s=0.05, window_post_s=0.05,
        )
        half_w = int(round(0.05 * FS))
        union_size = (min(n - 1, 55 + half_w) - max(0, 50 - half_w) + 1)
        assert n_gated == union_size

    def test_empty_r_peaks_returns_unmodified_signal(self):
        emg = np.random.RandomState(2).normal(0, 1, 500)
        emg_gated, n_gated = eg.gate_emg_by_ecg(emg, np.array([]), FS)
        assert n_gated == 0
        np.testing.assert_array_equal(emg_gated, emg)

    def test_returns_copy_not_view_original_unchanged(self):
        emg, r_peaks = self._make_spiky_emg()
        emg_original = emg.copy()
        eg.gate_emg_by_ecg(emg, r_peaks, FS, window_pre_s=0.05, window_post_s=0.05)
        np.testing.assert_array_equal(emg, emg_original)

    def test_interpolation_is_linear_between_window_edges(self):
        n = 100
        emg = np.zeros(n)
        emg[10] = 0.0
        emg[30] = 10.0
        r_peaks = np.array([20])  # janela [20-half_w, 20+half_w]
        half_w = int(round(0.05 * FS))  # 5 amostras -> janela [15,25]
        emg[15] = 3.0
        emg[25] = 7.0
        emg_gated, _ = eg.gate_emg_by_ecg(
            emg, r_peaks, FS, window_pre_s=0.05, window_post_s=0.05,
        )
        expected = np.linspace(3.0, 7.0, 25 - 15 + 1)
        np.testing.assert_allclose(emg_gated[15:26], expected)


# ─────────────────────────────────────────────────────────────────────────
# apply_ecg_gating_to_raw
# ─────────────────────────────────────────────────────────────────────────
class TestApplyEcgGatingToRaw:
    def _make_raw(self, include_ecg=True, include_emg=True, duration_s=20, seed=1):
        import mne
        import neurokit2 as nk

        n = int(FS * duration_s)
        rng = np.random.RandomState(seed)
        channels, names = [], []

        if include_emg:
            emg = rng.normal(0, 1e-6, n)
            if include_ecg:
                ecg = nk.ecg_simulate(
                    duration=duration_s, sampling_rate=int(FS),
                    heart_rate=60, random_state=seed,
                )
                r_true = eg.detect_r_peaks(ecg, FS)
                for p in r_true:
                    lo, hi = max(0, p - 2), min(n, p + 3)
                    emg[lo:hi] += 40e-6
            channels.append(emg)
            names.append("EMG1-EMG2")

        if include_ecg:
            if not include_emg:
                ecg = nk.ecg_simulate(
                    duration=duration_s, sampling_rate=int(FS),
                    heart_rate=60, random_state=seed,
                )
            channels.append(ecg)
            names.append("ECG1-ECG2")

        info = mne.create_info(names, FS, ch_types=["misc"] * len(names))
        raw = mne.io.RawArray(np.vstack(channels), info, verbose=False)
        return raw

    def test_applies_gating_when_both_channels_present(self):
        raw = self._make_raw(include_ecg=True, include_emg=True)
        diag = eg.apply_ecg_gating_to_raw(
            raw, "EMG1-EMG2", ["ECG1-ECG2"], window_pre_s=0.05, window_post_s=0.05,
        )
        assert diag["ecg_gate_applied"] is True
        assert diag["ecg_channel_found"] == "ECG1-ECG2"
        assert diag["n_r_peaks"] > 0
        assert diag["n_gated_samples"] > 0
        assert diag["reason_skipped"] is None

    def test_removes_ecg_channel_from_raw_after_gating(self):
        raw = self._make_raw(include_ecg=True, include_emg=True)
        eg.apply_ecg_gating_to_raw(
            raw, "EMG1-EMG2", ["ECG1-ECG2"], window_pre_s=0.05, window_post_s=0.05,
        )
        assert "ECG1-ECG2" not in raw.ch_names
        assert "EMG1-EMG2" in raw.ch_names

    def test_skips_when_emg_channel_absent(self):
        raw = self._make_raw(include_ecg=True, include_emg=False)
        diag = eg.apply_ecg_gating_to_raw(raw, None, ["ECG1-ECG2"])
        assert diag["ecg_gate_applied"] is False
        assert diag["reason_skipped"] == "emg_channel_absent"

    def test_skips_when_ecg_channel_not_found(self):
        raw = self._make_raw(include_ecg=False, include_emg=True)
        diag = eg.apply_ecg_gating_to_raw(raw, "EMG1-EMG2", ["ECG1-ECG2"])
        assert diag["ecg_gate_applied"] is False
        assert diag["reason_skipped"] == "ecg_channel_not_found"
        assert "EMG1-EMG2" in raw.ch_names  # EMG preservado intacto

    def test_skips_and_drops_channel_when_no_r_peaks_detected(self):
        import mne
        n = int(FS * 5)
        emg = np.random.RandomState(3).normal(0, 1e-6, n)
        flat_ecg = np.full(n, 0.001)
        info = mne.create_info(
            ["EMG1-EMG2", "ECG1-ECG2"], FS, ch_types=["misc", "misc"],
        )
        raw = mne.io.RawArray(np.vstack([emg, flat_ecg]), info, verbose=False)
        diag = eg.apply_ecg_gating_to_raw(raw, "EMG1-EMG2", ["ECG1-ECG2"])
        assert diag["ecg_gate_applied"] is False
        assert diag["reason_skipped"] == "no_r_peaks_detected"
        assert "ECG1-ECG2" not in raw.ch_names  # ECG ainda e removido

    def test_gated_emg_has_lower_peak_amplitude_than_original(self):
        raw = self._make_raw(include_ecg=True, include_emg=True)
        emg_before = raw.get_data(picks=["EMG1-EMG2"])[0].copy()
        eg.apply_ecg_gating_to_raw(
            raw, "EMG1-EMG2", ["ECG1-ECG2"], window_pre_s=0.05, window_post_s=0.05,
        )
        emg_after = raw.get_data(picks=["EMG1-EMG2"])[0]
        assert np.abs(emg_after).max() < np.abs(emg_before).max()

    def test_ecg_candidate_matching_is_case_insensitive(self):
        import mne
        n = int(FS * 5)
        emg = np.random.RandomState(4).normal(0, 1e-6, n)
        ecg = np.random.RandomState(5).normal(0, 1e-3, n)
        info = mne.create_info(
            ["emg1-emg2", "ecg1-ecg2"], FS, ch_types=["misc", "misc"],
        )
        raw = mne.io.RawArray(np.vstack([emg, ecg]), info, verbose=False)
        diag = eg.apply_ecg_gating_to_raw(raw, "emg1-emg2", ["ECG1-ECG2"])
        assert diag["ecg_channel_found"] == "ecg1-ecg2"
