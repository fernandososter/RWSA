"""
Testes deterministicos dos 3 criterios da regra AASM (2023a) em
src/sleep_rswa/preprocessing/aasm_rule.py: tonico (soma de segmentos >5s
cobrindo >=50% da epoca de 30s), fasico (>=5 das 10 mini-epocas de 3s com
burst 0.1-5.0s) e any (superset de tonic/phasic -- qualquer atividade
acima do limiar, exceto cruzamentos mais curtos que o piso de plausibilidade
ANY_MIN_SEG_S=0.1s, que existem para filtrar ruido de envelope RMS e nao
correspondem a burst muscular real; ver aasm_rule.ANY_MIN_SEG_S).

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

    def test_noise_segment_shorter_than_floor_does_not_trigger_any(self):
        # Segmento de 20ms (< ANY_MIN_SEG_S=0.1s): cruzamento de limiar por
        # ruido de envelope, nao um burst real -- nao deve marcar any_mini.
        env = _flat_epoch()
        _put_segment(env, start_s=1.5, dur_s=0.02, amplitude_uv=30.0)
        result = ar.classify_macro_epoch(env, THRESHOLD_UV)
        assert not result["any_mini"].any()
        assert result["tonic"] is False
        assert result["phasic"] is False

    def test_segment_exactly_at_any_floor_triggers_any(self):
        # Segmento de exatamente ANY_MIN_SEG_S=0.1s: no piso, deve contar.
        env = _flat_epoch()
        _put_segment(env, start_s=1.5, dur_s=ar.ANY_MIN_SEG_S, amplitude_uv=30.0)
        result = ar.classify_macro_epoch(env, THRESHOLD_UV)
        assert result["any_mini"][0].item() is True or bool(result["any_mini"][0])
        assert result["tonic"] is False
        assert result["phasic"] is False

    def test_multiple_subfloor_noise_blips_never_trigger_any(self):
        # Varios cruzamentos de limiar de 10ms espalhados pela epoca inteira
        # (mesmo padrao de ruido de envelope observado em dados reais) --
        # nenhum isoladamente atinge o piso, entao any_mini deve ficar
        # todo zero, mesmo com 10 blips distintos.
        env = _flat_epoch()
        for k in range(10):
            _put_segment(env, start_s=2.0 + 3.0 * k, dur_s=0.01, amplitude_uv=30.0)
        result = ar.classify_macro_epoch(env, THRESHOLD_UV)
        assert not result["any_mini"].any()

    def test_long_segment_above_floor_still_triggers_any_even_when_also_tonic(self):
        # Segmento longo (16s, tambem dispara tonic) deve continuar
        # contando para any -- o piso de duracao NAO deve, por engano,
        # introduzir um teto ou exclusividade entre any e tonic/phasic
        # (any continua superset; ver test_any_is_superset_of_tonic_epoch_after_rasterization).
        env = _flat_epoch()
        _put_segment(env, start_s=0.0, dur_s=16.0, amplitude_uv=30.0)
        result = ar.classify_macro_epoch(env, THRESHOLD_UV)
        assert result["tonic"] is True
        assert result["any_mini"][:5].all()  # 16s cobre mini-epocas 0-4 (0-15s) e parte da 5

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


# ─────────────────────────────────────────────────────────────────────────
# (iv) FUSAO DE GAP <= MERGE_GAP_S (250ms, convencao RBDtector/SINBAR):
# segmentos separados por um gap curto abaixo do limiar devem ser fundidos
# ANTES de calcular duracoes para tonic/phasic/any -- caso contrario,
# atividade continua real fragmentada por ruido de amostra unica nunca
# acumula duracao suficiente para disparar tonico. Ver MERGE_GAP_S e
# _merge_close_segments em aasm_rule.py.
# ─────────────────────────────────────────────────────────────────────────

class TestGapMerge:
    def test_merge_close_segments_unit_merges_short_gap(self):
        # dois segmentos separados por um gap de 5 amostras (<=10) devem
        # ser fundidos num unico segmento continuo.
        segs = [(0, 100), (105, 200)]
        merged = ar._merge_close_segments(segs, gap_samples=10)
        assert merged == [(0, 200)]

    def test_merge_close_segments_unit_keeps_far_segments_separate(self):
        segs = [(0, 100), (150, 200)]  # gap=50 > 10
        merged = ar._merge_close_segments(segs, gap_samples=10)
        assert merged == [(0, 100), (150, 200)]

    def test_merge_close_segments_unit_gap_zero_disables_merge(self):
        segs = [(0, 100), (100, 200)]  # gap=0 (adjacentes)
        # gap_samples=0 desativa a fusao completamente (retorna inalterado),
        # mesmo para segmentos adjacentes.
        merged = ar._merge_close_segments(segs, gap_samples=0)
        assert merged == segs

    def test_two_segments_separated_by_short_gap_merge_and_trigger_tonic(self):
        # dois segmentos de 8s cada, separados por um gap de 0.2s (<=0.25s
        # de MERGE_GAP_S), devem ser fundidos num unico segmento de
        # 8+0.2+8=16.2s > 5s -- disparando tonico sozinho (sem a fusao,
        # cada segmento isolado de 8s > 5s tambem somaria 16s>=15s, entao
        # usamos segmentos de exatamente 5s cada para so disparar tonico
        # SE fundidos).
        env = _flat_epoch()
        _put_segment(env, start_s=0.0, dur_s=5.0, amplitude_uv=30.0)   # segmento 1: 5.0s (nao conta isolado, precisa ser >5s)
        _put_segment(env, start_s=5.2, dur_s=10.0, amplitude_uv=30.0)  # gap de 0.2s; segmento 2: 10.0s (>5s, conta isolado)
        # Fundido: 0.0-15.2s = 15.2s continuo (>5s) -> tonic=True, coverage=15.2s
        # Nao fundido: so o segmento 2 (10.0s) conta -> coverage=10.0s < 15s -> tonic=False
        result_merged = ar.classify_macro_epoch(env, THRESHOLD_UV, merge_gap_s=0.25)
        result_unmerged = ar.classify_macro_epoch(env, THRESHOLD_UV, merge_gap_s=0.0)
        assert result_merged["tonic"] is True
        assert result_merged["tonic_coverage_s"] == pytest.approx(15.2, abs=1e-6)
        assert result_unmerged["tonic"] is False
        assert result_unmerged["tonic_coverage_s"] == pytest.approx(10.0, abs=1e-6)

    def test_gap_longer_than_merge_tolerance_is_not_merged(self):
        # mesmo cenario, mas gap de 0.3s (>0.25s) -- nao deve fundir.
        env = _flat_epoch()
        _put_segment(env, start_s=0.0, dur_s=5.0, amplitude_uv=30.0)
        _put_segment(env, start_s=5.3, dur_s=10.0, amplitude_uv=30.0)  # gap=0.3s > MERGE_GAP_S
        result = ar.classify_macro_epoch(env, THRESHOLD_UV, merge_gap_s=0.25)
        assert result["tonic"] is False
        assert result["tonic_coverage_s"] == pytest.approx(10.0, abs=1e-6)

    def test_fragmented_continuous_activity_merges_into_tonic_via_default(self):
        # simula o padrao real observado (rbd9): um trecho continuamente
        # elevado e fragmentado em muitos micro-segmentos por amostras
        # isoladas de ruido caindo abaixo do limiar por <=250ms. Usa
        # merge_gap_s=MERGE_GAP_S (default do modulo) explicitamente.
        env = _flat_epoch()
        # trecho de 20s (0-20s) com 20 blips de ruido de 50ms cada,
        # espacados a cada 1s, caindo abaixo do limiar -- fragmenta o
        # trecho em 20 segmentos curtos separados por gaps de 50ms (<250ms).
        _put_segment(env, start_s=0.0, dur_s=20.0, amplitude_uv=30.0)
        for k in range(1, 20):
            _put_segment(env, start_s=float(k) - 0.05, dur_s=0.05, amplitude_uv=1.0)
        raw_segs = ar._segments_above_threshold(env, THRESHOLD_UV)
        assert len(raw_segs) > 5  # confirma que o trecho de fato fragmentou

        result = ar.classify_macro_epoch(env, THRESHOLD_UV, merge_gap_s=ar.MERGE_GAP_S)
        assert result["tonic"] is True
        assert result["tonic_coverage_s"] == pytest.approx(20.0, abs=1e-3)

        result_no_merge = ar.classify_macro_epoch(env, THRESHOLD_UV, merge_gap_s=0.0)
        assert result_no_merge["tonic"] is False  # nenhum segmento fragmentado isolado passa de 5s

    def test_merge_gap_propagates_through_apply_aasm_rule(self):
        # confirma que o parametro merge_gap_s chega intacto de
        # apply_aasm_rule/label_exam_with_aasm_rule at classify_macro_epoch
        # (nao apenas na chamada direta usada nos testes acima).
        n_mini = ar.MINI_PER_MACRO
        signals = np.zeros((n_mini, 5, MINI_SAMPLES), dtype=np.float32)
        stages = np.full(n_mini, ar.REM_STAGE, dtype=np.int64)

        # 4.5s de atividade, gap de 0.2s, 10.5s de atividade -> so dispara
        # tonico se merge_gap_s>=0.2 (default MERGE_GAP_S=0.25). Segmento 1
        # fica deliberadamente abaixo de 5s (mesmo apos o alargamento de
        # borda introduzido pela suavizacao de rms_envelope, ~0.01s aqui)
        # para nao contar isoladamente pelo criterio tonico -- so a fusao
        # com o segmento 2 produz um trecho continuo >5s cobrindo >=15s.
        emg_flat = np.zeros(MACRO_SAMPLES, dtype=np.float64)
        emg_flat[:] = 1e-6  # 1 uV baseline (abaixo do limiar em V)
        _put_segment(emg_flat, start_s=0.0, dur_s=4.5, amplitude_uv=30e-6)
        _put_segment(emg_flat, start_s=4.7, dur_s=10.5, amplitude_uv=30e-6)
        for m in range(n_mini):
            signals[m, 4, :] = emg_flat[m * MINI_SAMPLES:(m + 1) * MINI_SAMPLES]

        result_merged = ar.apply_aasm_rule(
            signals, stages, rem_baseline_uv=BASELINE_UV, rem_baseline_n_epochs=n_mini,
            merge_gap_s=0.25,
        )
        result_unmerged = ar.apply_aasm_rule(
            signals, stages, rem_baseline_uv=BASELINE_UV, rem_baseline_n_epochs=n_mini,
            merge_gap_s=0.0,
        )
        assert result_merged["n_tonic_macro_epochs"] == 1
        assert result_unmerged["n_tonic_macro_epochs"] == 0
