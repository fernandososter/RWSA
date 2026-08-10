"""
Testes deterministicos dos 3 criterios da regra AASM (2023a) em
src/sleep_rswa/preprocessing/aasm_rule.py: tonico (soma de segmentos >5s
cobrindo >=50% da epoca de 30s), fasico (>=5 das 10 mini-epocas de 3s com
burst 0.1-5.0s) e any (superset -- qualquer atividade acima do limiar,
independente de duracao).

Todos os casos usam amplitude e duracao construidas a mao (sem ruido) para
que o resultado esperado seja exato e nao dependa de nenhum ajuste de
parametro alem dos definidos no proprio modulo.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sleep_rswa.preprocessing import aasm_rule as ar

FS = ar.FS  # 100 Hz
MACRO_SAMPLES = int(ar.MACRO_EPOCH_SEC * FS)  # 3000 (30s @ 100Hz)
MINI_SAMPLES = int(ar.EPOCH_SEC * FS)          # 300  (3s @ 100Hz)

BASELINE_UV = 10.0
THRESHOLD_UV = ar.MIN_AMPLITUDE_RATIO * BASELINE_UV  # 20 uV


def _flat_epoch(amplitude_uv: float = 1.0) -> np.ndarray:
    """Epoca de 30s inteira abaixo do limiar (amplitude=baseline)."""
    return np.full(MACRO_SAMPLES, amplitude_uv, dtype=np.float64)


def _put_segment(env: np.ndarray, start_s: float, dur_s: float, amplitude_uv: float) -> np.ndarray:
    i0 = int(round(start_s * FS))
    i1 = int(round((start_s + dur_s) * FS))
    env[i0:i1] = amplitude_uv
    return env


# ─────────────────────────────────────────────────────────────────────────
# (i) TONICO: soma de segmentos > 5s cobrindo >= 50% (15s) da epoca de 30s
# ─────────────────────────────────────────────────────────────────────────

class TestTonic:
    def test_two_segments_over_5s_summing_16s_triggers_tonic(self):
        env = _flat_epoch()
        _put_segment(env, start_s=0.0, dur_s=8.0, amplitude_uv=30.0)   # segmento 1: 8s (>5s)
        _put_segment(env, start_s=15.0, dur_s=8.0, amplitude_uv=30.0)  # segmento 2: 8s (>5s); soma=16s>=15s
        result = ar.classify_macro_epoch(env, THRESHOLD_UV)
        assert result["tonic"] is True
        assert result["tonic_coverage_s"] == pytest.approx(16.0, abs=1e-6)

    def test_single_segment_exactly_5s_does_not_count_towards_tonic(self):
        # AASM exige duracao > 5s por segmento (nao >=5s) -- um segmento de
        # exatamente 5s nao entra na soma tonica.
        env = _flat_epoch()
        _put_segment(env, start_s=0.0, dur_s=5.0, amplitude_uv=30.0)
        result = ar.classify_macro_epoch(env, THRESHOLD_UV)
        assert result["tonic"] is False
        assert result["tonic_coverage_s"] == pytest.approx(0.0, abs=1e-6)

    def test_segments_over_5s_summing_below_50pct_do_not_trigger_tonic(self):
        # dois segmentos de 6s (>5s) somam 12s < 15s (50% de 30s) -> tonic=False
        env = _flat_epoch()
        _put_segment(env, start_s=0.0, dur_s=6.0, amplitude_uv=30.0)
        _put_segment(env, start_s=20.0, dur_s=6.0, amplitude_uv=30.0)
        result = ar.classify_macro_epoch(env, THRESHOLD_UV)
        assert result["tonic"] is False
        assert result["tonic_coverage_s"] == pytest.approx(12.0, abs=1e-6)

    def test_below_amplitude_threshold_never_triggers_tonic(self):
        # segmento de 20s (bem acima de 5s e de 50% da epoca) mas com
        # amplitude ABAIXO do limiar (1.5x baseline < 2x exigido) -> tonic=False
        env = _flat_epoch()
        _put_segment(env, start_s=0.0, dur_s=20.0, amplitude_uv=1.5 * BASELINE_UV)
        result = ar.classify_macro_epoch(env, THRESHOLD_UV)
        assert result["tonic"] is False
        assert result["tonic_coverage_s"] == pytest.approx(0.0, abs=1e-6)


# ─────────────────────────────────────────────────────────────────────────
# (ii) FASICO: >=5 das 10 mini-epocas de 3s com burst 0.1-5.0s
# ─────────────────────────────────────────────────────────────────────────

class TestPhasic:
    def test_exactly_5_of_10_mini_epochs_with_burst_triggers_phasic(self):
        env = _flat_epoch()
        for mini_idx in range(5):  # mini-epocas 0,1,2,3,4 (de um total de 10)
            start_s = mini_idx * ar.EPOCH_SEC + 0.5
            _put_segment(env, start_s=start_s, dur_s=1.0, amplitude_uv=30.0)
        result = ar.classify_macro_epoch(env, THRESHOLD_UV)
        assert result["n_phasic_mini"] == 5
        assert result["phasic"] is True

    def test_only_4_of_10_mini_epochs_does_not_trigger_phasic(self):
        env = _flat_epoch()
        for mini_idx in range(4):
            start_s = mini_idx * ar.EPOCH_SEC + 0.5
            _put_segment(env, start_s=start_s, dur_s=1.0, amplitude_uv=30.0)
        result = ar.classify_macro_epoch(env, THRESHOLD_UV)
        assert result["n_phasic_mini"] == 4
        assert result["phasic"] is False

    def test_burst_duration_outside_0p1_5p0s_window_is_not_phasic(self):
        # burst de 8s (fora da faixa 0.1-5.0s) em 6 mini-epocas nao conta
        # como fasico, mesmo cobrindo mais de 5 mini-epocas.
        env = _flat_epoch()
        _put_segment(env, start_s=0.0, dur_s=8.0, amplitude_uv=30.0)
        result = ar.classify_macro_epoch(env, THRESHOLD_UV)
        assert result["n_phasic_mini"] == 0
        assert result["phasic"] is False

    def test_burst_shorter_than_0p1s_is_not_phasic(self):
        env = _flat_epoch()
        _put_segment(env, start_s=1.0, dur_s=0.05, amplitude_uv=30.0)  # 0.05s < 0.1s
        result = ar.classify_macro_epoch(env, THRESHOLD_UV)
        assert result["n_phasic_mini"] == 0
        assert result["phasic"] is False


# ─────────────────────────────────────────────────────────────────────────
# (iii) ANY: superset -- qualquer atividade acima do limiar, qualquer duracao
# ─────────────────────────────────────────────────────────────────────────

class TestAny:
    def test_segment_between_5_and_15s_triggers_any_but_not_tonic_nor_phasic(self):
        # 10s: >5s (nao fasico, fora de 0.1-5.0s) mas < 15s de cobertura
        # necessaria para tonico isolado -> nem tonic nem phasic, mas any=1
        # nas mini-epocas cobertas (categoria "any chin EMG activity").
        env = _flat_epoch()
        _put_segment(env, start_s=0.0, dur_s=10.0, amplitude_uv=30.0)
        result = ar.classify_macro_epoch(env, THRESHOLD_UV)
        assert result["tonic"] is False
        assert result["phasic"] is False
        # 10s cobre as mini-epocas 0,1,2,3 (0-3s,3-6s,6-9s,9-12s parcial)
        assert result["any_mini"][:4].all()
        assert not result["any_mini"][4:].any()

    def test_activity_below_amplitude_threshold_does_not_trigger_any(self):
        env = _flat_epoch()
        _put_segment(env, start_s=0.0, dur_s=10.0, amplitude_uv=1.5 * BASELINE_UV)  # < 2x baseline
        result = ar.classify_macro_epoch(env, THRESHOLD_UV)
        assert not result["any_mini"].any()

    def test_any_is_superset_of_tonic_epoch_after_rasterization(self):
        # No nivel de exame completo (apply_aasm_rule), qualquer epoca R
        # marcada tonic=1 ou phasic=1 DEVE ter any=1 em todas as mini-epocas
        # correspondentes -- este e o requisito de "superset" da AASM.
        n_mini = ar.MINI_PER_MACRO  # uma unica epoca de 30s
        signals = np.zeros((n_mini, 5, MINI_SAMPLES), dtype=np.float32)
        stages = np.full(n_mini, ar.REM_STAGE, dtype=np.int64)

        # injeta um evento tonico: dois segmentos >5s somando 16s, dentro
        # das mini-epocas 0-5 (dentro da epoca de 30s)
        emg_flat = np.full(n_mini * MINI_SAMPLES, 1e-6, dtype=np.float64)  # baseline em Volts
        seg_amp_v = THRESHOLD_UV * 1.5 * 1e-6  # bem acima do limiar, em Volts
        _put_segment(emg_flat, start_s=0.0, dur_s=8.0, amplitude_uv=seg_amp_v)
        _put_segment(emg_flat, start_s=15.0, dur_s=8.0, amplitude_uv=seg_amp_v)
        signals[:, 4, :] = emg_flat.reshape(n_mini, MINI_SAMPLES)

        result = ar.apply_aasm_rule(
            signals, stages,
            rem_baseline_uv=BASELINE_UV, rem_baseline_n_epochs=n_mini,
        )
        assert result["n_tonic_macro_epochs"] == 1
        assert (result["tonic_labels"] == 1.0).all()
        # requisito de superset: toda mini-epoca tonica tambem e any
        assert np.all(result["any_labels"][result["tonic_labels"] == 1.0] == 1.0)


# ─────────────────────────────────────────────────────────────────────────
# Integracao: apply_aasm_rule -- estagio, fallback NREM, forma do retorno
# ─────────────────────────────────────────────────────────────────────────

class TestApplyAasmRuleIntegration:
    def test_non_rem_epochs_are_never_labeled(self):
        n_mini = ar.MINI_PER_MACRO
        signals = np.zeros((n_mini, 5, MINI_SAMPLES), dtype=np.float32)
        stages = np.full(n_mini, 2, dtype=np.int64)  # N2, nao REM
        # mesmo com atividade de amplitude alta, epoca fora de R nao e rotulada
        signals[:, 4, :] = 30 * 1e-6

        result = ar.apply_aasm_rule(
            signals, stages,
            rem_baseline_uv=BASELINE_UV, rem_baseline_n_epochs=n_mini,
        )
        assert result["n_rem_macro_epochs"] == 0
        assert not result["tonic_labels"].any()
        assert not result["phasic_labels"].any()
        assert not result["any_labels"].any()

    def test_nrem_fallback_used_when_rem_baseline_invalid(self):
        # duas epocas de 30s: uma REM (sem atonia valida -> rem_baseline_uv
        # invalido, simulando RSWA tonica contaminando todo o REM) e uma
        # N2 (usada como fonte do fallback).
        n_mini = 2 * ar.MINI_PER_MACRO
        signals = np.zeros((n_mini, 5, MINI_SAMPLES), dtype=np.float32)
        stages = np.concatenate([
            np.full(ar.MINI_PER_MACRO, ar.REM_STAGE, dtype=np.int64),
            np.full(ar.MINI_PER_MACRO, 2, dtype=np.int64),  # N2
        ])
        signals[:, 4, :] = 5e-6  # 5 uV em todas as mini-epocas (REM e N2)

        result = ar.apply_aasm_rule(
            signals, stages,
            rem_baseline_uv=float("nan"), rem_baseline_n_epochs=0,
        )
        assert result["atonia_source"] == "nrem_fallback"
        assert np.isfinite(result["atonia_baseline_uv"])

    def test_rejects_non_multiple_of_macro_epoch_length(self):
        signals = np.zeros((7, 5, MINI_SAMPLES), dtype=np.float32)  # 7 nao e multiplo de 10
        stages = np.full(7, ar.REM_STAGE, dtype=np.int64)
        with pytest.raises(ValueError):
            ar.apply_aasm_rule(
                signals, stages,
                rem_baseline_uv=BASELINE_UV, rem_baseline_n_epochs=1,
            )
