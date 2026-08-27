from dataclasses import dataclass, field
from pathlib import Path
import math
import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from typing import Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from .distribution import StageDistribution

from .config import RSWAConfig, SignalConfig


@dataclass
class SubjectData:
    subject_id: str
    signals: torch.Tensor
    sleep_stages: torch.Tensor
    rswa_labels: torch.Tensor
    rswa_conf: torch.Tensor
    rem_baseline_uv: float | None = None
    # atonia_baseline_uv: baseline REAL usada pela regra AASM para decidir
    # tonico/fasico (label_metadata.aasm_rule.atonia_baseline_uv no .pt).
    # NAO confundir com rem_baseline_uv (baseline crua de baixo percentil,
    # tipicamente 5-6x menor). Quando use_baseline_relative_channel=True,
    # os dois canais do ramo EMG passam a ser derivados desta baseline.
    atonia_baseline_uv: float | None = None
    baseline_relative_reference_ratio: float | None = None
    emg_signals: torch.Tensor | None = None
    # Rotulos por cabeca (opcionais). Se ausentes, sao derivados de
    # rswa_labels no load (retrocompatibilidade com .pt mono-rotulo antigos e
    # suporte ao schema exclusivo novo: 0=nada, 1=fasico, 2=tonico, 3=any).
    tonic_labels: torch.Tensor | None = None
    phasic_labels: torch.Tensor | None = None
    # any_labels: categoria "any" do limiar duplo (amplitude confirmada, duracao
    # ambigua 5s-15s) -- cabeca NOVA, nao existe em nenhum .pt antigo (mono- ou
    # multi-rotulo). Ausente -> zeros (nenhum "any" conhecido) ate a rotulagem
    # automatica (CNN+limiar-duplo) escrever este campo.
    any_labels: torch.Tensor | None = None
    tonic_proto_labels: torch.Tensor | None = None
    phasic_proto_labels: torch.Tensor | None = None
    n_epochs: int = field(init=False)

    def __post_init__(self) -> None:
        self.n_epochs = int(self.signals.shape[0])
        if self.sleep_stages.shape[0] != self.n_epochs:
            raise ValueError(f"{self.subject_id}: signals e sleep_stages possuem comprimentos diferentes.")
        if self.emg_signals is not None and self.emg_signals.shape[0] != self.n_epochs:
            raise ValueError(f"{self.subject_id}: emg_signals possui comprimento incompatível.")
        if self.tonic_proto_labels is not None and self.tonic_proto_labels.shape[0] != self.n_epochs:
            raise ValueError(f"{self.subject_id}: tonic_proto_labels possui comprimento incompatível.")
        if self.phasic_proto_labels is not None and self.phasic_proto_labels.shape[0] != self.n_epochs:
            raise ValueError(f"{self.subject_id}: phasic_proto_labels possui comprimento incompatível.")


def load_subject_file(path: str | Path) -> SubjectData:
    path = Path(path)
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, SubjectData):
        return obj
    if not isinstance(obj, dict):
        raise TypeError(f"{path}: esperado dict ou SubjectData, recebido {type(obj)!r}")

    def pick(*keys: str):
        for key in keys:
            if key in obj:
                return obj[key]
        raise KeyError(f"{path}: nenhuma chave encontrada entre {keys}")

    signals = pick("signals", "x", "signal")
    stages = pick("sleep_stages", "stages", "y_stage")
    rswa = obj.get("rswa_labels", torch.zeros_like(stages))
    conf = obj.get("rswa_conf", torch.zeros_like(stages, dtype=torch.float32))
    emg = obj.get("emg_signals", obj.get("emg", obj.get("emg_center")))
    # Rotulos por cabeca, se o .pt os gravou (parser novo). Se ausentes
    # (.pt antigo mono-rotulo), ficam None e sao derivados de rswa_labels
    # no __getitem__.
    tonic = obj.get("tonic_labels")
    phasic = obj.get("phasic_labels")
    any_lab = obj.get("any_labels")
    tonic_proto = obj.get("tonic_proto_labels")
    phasic_proto = obj.get("phasic_proto_labels")
    label_metadata = obj.get("label_metadata")
    rswa_meta = label_metadata if isinstance(label_metadata, dict) else {}
    aasm_meta = label_metadata.get("aasm_rule") if isinstance(label_metadata, dict) else None
    auto_meta = label_metadata.get("auto_label") if isinstance(label_metadata, dict) else None
    atonia_baseline_uv = None
    if isinstance(aasm_meta, dict) and aasm_meta.get("atonia_baseline_uv") is not None:
        try:
            _val = float(aasm_meta["atonia_baseline_uv"])
            if _val == _val and _val > 0:  # exclui NaN/<=0 (atonia_source="unavailable")
                atonia_baseline_uv = _val
        except (TypeError, ValueError):
            atonia_baseline_uv = None
    baseline_relative_reference_ratio = None
    ref_candidates: tuple[object | None, ...] = ()
    if isinstance(aasm_meta, dict):
        ref_candidates = (
            aasm_meta.get("min_amplitude_ratio_used"),
            aasm_meta.get("reference_ratio_used"),
        )
    elif isinstance(auto_meta, dict):
        ref_candidates = (
            auto_meta.get("k_on"),
            auto_meta.get("min_amplitude_ratio_used"),
            auto_meta.get("reference_ratio_used"),
        )
    else:
        ref_candidates = (
            rswa_meta.get("baseline_relative_reference_ratio"),
            rswa_meta.get("reference_ratio_used"),
        )
    for candidate in ref_candidates:
        if candidate is None:
            continue
        try:
            parsed = float(candidate)
        except (TypeError, ValueError):
            continue
        if parsed == parsed and parsed > 0:
            baseline_relative_reference_ratio = parsed
            break
    return SubjectData(
        subject_id=str(obj.get("subject_id", path.stem)),
        signals=signals,
        sleep_stages=stages,
        rswa_labels=rswa,
        rswa_conf=conf,
        rem_baseline_uv=(
            float(obj["rem_baseline_uv"])
            if obj.get("rem_baseline_uv") is not None
            else None
        ),
        atonia_baseline_uv=atonia_baseline_uv,
        baseline_relative_reference_ratio=baseline_relative_reference_ratio,
        emg_signals=emg,
        tonic_labels=tonic,
        phasic_labels=phasic,
        any_labels=any_lab,
        tonic_proto_labels=tonic_proto,
        phasic_proto_labels=phasic_proto,
    )



def load_subject_directory(directory: str | Path, max_workers: int | None = None) -> list[SubjectData]:
    paths = sorted(Path(directory).glob("*.pt"))
    if not paths:
        raise FileNotFoundError(f"Nenhum arquivo .pt em {directory}")

    # torch.load e leitura de disco se beneficiam de paralelismo por threads.
    workers = max_workers or min(32, max(1, (os.cpu_count() or 1) * 2))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(
            tqdm(
                executor.map(load_subject_file, paths),
                total=len(paths),
                desc="Carregando sujeitos",
                unit="arquivo",
            )
        )


def _zscore_per_channel(signals: torch.Tensor) -> torch.Tensor:
    # signals: [T, C, N]
    flat = signals.permute(1, 0, 2).reshape(signals.shape[1], -1)
    mean = flat.mean(dim=1)
    std = flat.std(dim=1).clamp_min(1e-8)
    return (signals - mean[None, :, None]) / std[None, :, None]


def _rms_envelope_same(x: torch.Tensor, *, win_sec: float, fs: int) -> torch.Tensor:
    """Envelope RMS com janela deslizante e mesmo comprimento da entrada."""
    if x.ndim != 3:
        raise ValueError(f"x deve ter shape [T,C,N], recebeu {tuple(x.shape)}")
    win = max(1, int(round(float(win_sec) * int(fs))))
    if win <= 1:
        return x.abs()
    left = (win - 1) // 2
    right = win - 1 - left
    x2 = x.to(torch.float32).pow(2.0)
    padded = F.pad(x2, (left, right))
    kernel = torch.ones((x.shape[1], 1, win), dtype=x2.dtype, device=x2.device) / float(win)
    ms = F.conv1d(padded, kernel, groups=x.shape[1])
    return ms.clamp_min(0.0).sqrt()


class SleepAnalysisDataset(Dataset):
    def __init__(
        self,
        subjects: Sequence[SubjectData],
        min_confidence: float = 0.0,
        rem_mask_only: bool = True,
        *,
        rswa_target_mode: str = "final",
        use_baseline_relative_channel: bool = False,
        use_rms_relative_channel: bool = False,
        target_epoch_sec: int = 3,
        context_radius: int | None = None,
    ):
        self.source_signal_config = SignalConfig()
        self.target_epoch_sec = int(target_epoch_sec)
        if self.target_epoch_sec <= 0:
            raise ValueError("target_epoch_sec precisa ser positivo.")
        if self.target_epoch_sec % self.source_signal_config.epoch_sec != 0:
            raise ValueError(
                "target_epoch_sec precisa ser múltiplo de "
                f"{self.source_signal_config.epoch_sec}s para reutilizar os .pt atuais."
            )
        resolved_context_radius = (
            int(context_radius)
            if context_radius is not None
            else (1 if self.target_epoch_sec == self.source_signal_config.epoch_sec else 0)
        )
        if resolved_context_radius < 0:
            raise ValueError("context_radius não pode ser negativo.")
        self.min_confidence = min_confidence
        self.rem_mask_only = rem_mask_only
        self.rswa_target_mode = rswa_target_mode.strip().lower()
        if self.rswa_target_mode not in {"final", "aasm_proto"}:
            raise ValueError(f"rswa_target_mode invalido: {rswa_target_mode!r}")
        self.use_baseline_relative_channel = bool(use_baseline_relative_channel)
        self.use_rms_relative_channel = bool(use_rms_relative_channel)
        if self.use_rms_relative_channel and not self.use_baseline_relative_channel:
            raise ValueError(
                "use_rms_relative_channel=True exige use_baseline_relative_channel=True."
            )
        self.signal_config = SignalConfig(
            fs=self.source_signal_config.fs,
            epoch_sec=self.target_epoch_sec,
            samples_per_epoch=self.source_signal_config.fs * self.target_epoch_sec,
            context_radius=resolved_context_radius,
            n_channels=self.source_signal_config.n_channels,
            staging_channel_indices=self.source_signal_config.staging_channel_indices,
        )
        self.rswa_config = RSWAConfig()
        self.subjects = [
            self._aggregate_subject(subject)
            for subject in subjects
        ]

    def __len__(self) -> int:
        return len(self.subjects)

    def _aggregate_subject(self, subject: SubjectData) -> SubjectData:
        factor = self.target_epoch_sec // self.source_signal_config.epoch_sec
        if factor == 1:
            return subject

        if subject.n_epochs % factor != 0:
            raise ValueError(
                f"{subject.subject_id}: n_epochs={subject.n_epochs} não é múltiplo de "
                f"{factor} para agregação em {self.target_epoch_sec}s."
            )

        new_t = subject.n_epochs // factor

        def _reshape_time(tensor: torch.Tensor | None) -> torch.Tensor | None:
            if tensor is None:
                return None
            if tensor.shape[0] != subject.n_epochs:
                raise ValueError(
                    f"{subject.subject_id}: tensor temporal incompatível para agregação: "
                    f"{tuple(tensor.shape)}"
                )
            return tensor.reshape(new_t, factor, *tensor.shape[1:])

        def _agg_binary(tensor: torch.Tensor | None) -> torch.Tensor | None:
            shaped = _reshape_time(tensor)
            if shaped is None:
                return None
            return shaped.amax(dim=1)

        stage_blocks = _reshape_time(subject.sleep_stages.long())
        if stage_blocks is None:
            raise RuntimeError("sleep_stages não pode ser None.")
        same_stage = (stage_blocks == stage_blocks[:, :1]).all(dim=1)
        valid_stage = (stage_blocks >= 0).all(dim=1) & same_stage
        aggregated_stages = stage_blocks[:, 0].clone()
        aggregated_stages[~valid_stage] = -1

        conf_blocks = _reshape_time(subject.rswa_conf.float())
        if conf_blocks is None:
            raise RuntimeError("rswa_conf não pode ser None.")
        aggregated_conf = conf_blocks.amin(dim=1)

        aggregated_tonic = _agg_binary(subject.tonic_labels)
        aggregated_phasic = _agg_binary(subject.phasic_labels)
        aggregated_any = _agg_binary(subject.any_labels)
        aggregated_tonic_proto = _agg_binary(subject.tonic_proto_labels)
        aggregated_phasic_proto = _agg_binary(subject.phasic_proto_labels)

        signals = _reshape_time(subject.signals)
        if signals is None:
            raise RuntimeError("signals não pode ser None.")
        aggregated_signals = signals.reshape(
            new_t,
            factor,
            *subject.signals.shape[1:],
        ).transpose(1, 2).reshape(
            new_t,
            subject.signals.shape[1],
            factor * subject.signals.shape[2],
        )

        aggregated_emg = None
        if subject.emg_signals is not None:
            emg = subject.emg_signals
            if emg.ndim == 2:
                emg = emg.unsqueeze(1)
            emg_blocks = _reshape_time(emg)
            if emg_blocks is None:
                raise RuntimeError("emg_signals não pode ser None após checagem.")
            aggregated_emg = emg_blocks.reshape(
                new_t,
                factor,
                *emg.shape[1:],
            ).transpose(1, 2).reshape(
                new_t,
                emg.shape[1],
                factor * emg.shape[2],
            )

        rswa_blocks = _reshape_time(subject.rswa_labels.long())
        if rswa_blocks is None:
            raise RuntimeError("rswa_labels não pode ser None.")
        aggregated_rswa = torch.zeros(new_t, dtype=torch.long)
        if aggregated_any is not None:
            aggregated_rswa[aggregated_any > 0.5] = self.rswa_config.any_label
        if aggregated_phasic is not None:
            aggregated_rswa[aggregated_phasic > 0.5] = self.rswa_config.phasic_label
        if aggregated_tonic is not None:
            aggregated_rswa[aggregated_tonic > 0.5] = self.rswa_config.tonic_label
        if aggregated_tonic is None and aggregated_phasic is None and aggregated_any is None:
            aggregated_rswa = rswa_blocks.amax(dim=1)

        return SubjectData(
            subject_id=subject.subject_id,
            signals=aggregated_signals,
            sleep_stages=aggregated_stages,
            rswa_labels=aggregated_rswa,
            rswa_conf=aggregated_conf,
            rem_baseline_uv=subject.rem_baseline_uv,
            atonia_baseline_uv=subject.atonia_baseline_uv,
            baseline_relative_reference_ratio=subject.baseline_relative_reference_ratio,
            emg_signals=aggregated_emg,
            tonic_labels=aggregated_tonic,
            phasic_labels=aggregated_phasic,
            any_labels=aggregated_any,
            tonic_proto_labels=aggregated_tonic_proto,
            phasic_proto_labels=aggregated_phasic_proto,
        )

    def _extract_staging_signals(self, subject: SubjectData) -> torch.Tensor:
        signals = subject.signals.float().clone()
        if signals.ndim != 3:
            raise ValueError(
                f"{subject.subject_id}: signals deve ter shape [T,C,N], recebeu {tuple(signals.shape)}"
            )
        indices = self.signal_config.staging_channel_indices
        if max(indices) >= signals.shape[1]:
            raise ValueError(
                f"{subject.subject_id}: canais de staging {indices} não existem em signals "
                f"com {signals.shape[1]} canais."
            )
        return _zscore_per_channel(signals[:, indices, :])

    def _extract_emg(self, subject: SubjectData) -> torch.Tensor:
        if subject.emg_signals is not None:
            emg = subject.emg_signals.float().clone()
            if emg.ndim == 2:
                emg = emg.unsqueeze(1)
            elif emg.ndim == 3 and emg.shape[1] != 1:
                raise ValueError(
                    f"{subject.subject_id}: emg_signals deve ter shape [T,N] ou [T,1,N]."
                )
        else:
            signals = subject.signals.float()
            index = self.rswa_config.emg_channel_index
            if index >= signals.shape[1]:
                raise ValueError(
                    f"{subject.subject_id}: nenhum EMG separado foi encontrado e o índice EMG "
                    f"{index} não existe em signals com {signals.shape[1]} canais. "
                    "Salve a chave 'emg_signals'/'emg' ou ajuste RSWAConfig.emg_channel_index."
                )
            emg = signals[:, index:index + 1, :].clone()
        primary = _zscore_per_channel(emg)[:, :, : self.signal_config.samples_per_epoch]
        if not self.use_baseline_relative_channel:
            return primary
        # Baseline correta = atonia_baseline_uv (a MESMA baseline que a regra
        # AASM usa para decidir tonico/fasico -- ver label_metadata.aasm_rule).
        # Quando o canal auxiliar baseline-relative esta habilitado, os DOIS
        # canais do ramo EMG passam a ser relativos ao basal:
        #   1. EMG assinado / baseline
        #   2. log1p(|EMG| / baseline) / log1p(reference_ratio)
        # Opcionalmente, um terceiro canal carrega o envelope RMS de 100 ms
        # relativo ao mesmo basal, alinhando a entrada ao dominio da regra AASM.
        # Assim, o mesmo limiar de amplitude usado no preprocessamento pode
        # ser representado explicitamente no segundo canal (valor 1.0).
        baseline_uv = subject.atonia_baseline_uv
        reference_ratio = subject.baseline_relative_reference_ratio
        if baseline_uv is None:
            # Exame legado sem label_metadata.aasm_rule: aproxima a partir de
            # rem_baseline_uv usando a razao media medida empiricamente
            # (ver RSWAConfig.baseline_relative_fallback_ratio).
            rem_baseline_uv = subject.rem_baseline_uv
            if rem_baseline_uv is not None and rem_baseline_uv > 0:
                baseline_uv = float(rem_baseline_uv) * self.rswa_config.baseline_relative_fallback_ratio
        if reference_ratio is None or reference_ratio <= 0:
            reference_ratio = self.rswa_config.baseline_relative_reference_ratio
        if baseline_uv is None or baseline_uv <= 0:
            signed_relative = torch.zeros_like(primary)
            amplitude_relative = torch.zeros_like(primary)
            rms_relative = torch.zeros_like(primary)
        else:
            baseline_v = float(baseline_uv) / 1e6
            window = emg[:, :, : self.signal_config.samples_per_epoch]
            ratio = window / baseline_v
            signed_relative = ratio.clamp(
                -self.rswa_config.baseline_relative_signed_clamp,
                self.rswa_config.baseline_relative_signed_clamp,
            )
            amplitude_relative = (
                torch.log1p(ratio.abs())
                / math.log1p(float(reference_ratio))
            ).clamp(
                0.0,
                self.rswa_config.baseline_relative_amplitude_clamp,
            )
            rms_ratio = (
                _rms_envelope_same(
                    window,
                    win_sec=self.rswa_config.baseline_relative_rms_win_sec,
                    fs=self.signal_config.fs,
                ) / baseline_v
            )
            rms_relative = (
                torch.log1p(rms_ratio)
                / math.log1p(float(reference_ratio))
            ).clamp(
                0.0,
                self.rswa_config.baseline_relative_amplitude_clamp,
            )
        channels = [signed_relative, amplitude_relative]
        if self.use_rms_relative_channel:
            channels.append(rms_relative)
        return torch.cat(channels, dim=1)


    def stage_distribution(self) -> StageDistribution:
        distribution = StageDistribution()

        for subject in self.subjects:
            distribution.update(
                subject.sleep_stages,
            )

        return distribution

    def movement_distribution(self) -> dict[str, int | float]:
        """Contagem de movimento 'any' (tônico OU fásico) usando a MESMA máscara
        de validade do ``__getitem__`` (confiança > min_confidence e, se
        ``rem_mask_only``, apenas mini-épocas em REM).

        Retorna, sobre todas as mini-épocas dos sujeitos deste split:
          - ``total_mini_epochs``    total de mini-épocas
          - ``evaluable_mini_epochs`` mini-épocas válidas para RSWA (na máscara)
          - ``movement_positive``    mini-épocas válidas com movimento anotado
          - ``pct_movement_of_evaluable`` % de positivos entre as avaliáveis
          - ``pct_movement_of_total``     % de positivos sobre o total
          - ``pct_evaluable_of_total``    % de mini-épocas avaliáveis sobre o total
        """
        total = 0
        evaluable = 0
        positives = 0
        for subject in self.subjects:
            stages = subject.sleep_stages.long()
            conf = subject.rswa_conf.float()
            valid = conf > self.min_confidence
            if self.rem_mask_only:
                valid = valid & stages.eq(self.rswa_config.rem_stage)

            if subject.tonic_labels is not None and subject.phasic_labels is not None:
                movement = (subject.tonic_labels > 0.5) | (subject.phasic_labels > 0.5)
            else:
                rswa = subject.rswa_labels.long()
                movement = rswa.eq(self.rswa_config.tonic_label) | rswa.eq(
                    self.rswa_config.phasic_label
                )
                movement = movement | rswa.eq(self.rswa_config.any_label)
            if subject.any_labels is not None:
                movement = movement | (subject.any_labels > 0.5)

            movement_valid = movement & valid
            total += int(stages.numel())
            evaluable += int(valid.sum().item())
            positives += int(movement_valid.sum().item())

        return {
            "total_mini_epochs": total,
            "evaluable_mini_epochs": evaluable,
            "movement_positive": positives,
            "pct_movement_of_evaluable": (100.0 * positives / evaluable) if evaluable else 0.0,
            "pct_movement_of_total": (100.0 * positives / total) if total else 0.0,
            "pct_evaluable_of_total": (100.0 * evaluable / total) if total else 0.0,
        }

    def summary(self) -> dict[str, int]:
        return {
            "exams": len(self.subjects),
            "items": len(self),
        }
    

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        subject = self.subjects[idx]
        signals = self._extract_staging_signals(subject)
        emg = self._extract_emg(subject)
        t, c, n = signals.shape

        ctx = self.signal_config.context_radius
        if ctx == 0:
            context = signals
        else:
            pad = torch.zeros(ctx, c, n, dtype=signals.dtype)
            context = (
                torch.cat([pad, signals, pad], dim=0)
                .unfold(0, 2 * ctx + 1, 1)
                .permute(0, 1, 3, 2)
                .reshape(t, c, (2 * ctx + 1) * n)
            )

        labels = subject.sleep_stages.long()
        if ctx == 0:
            valid_ctx = labels != -1
        else:
            pad_lab = torch.full((ctx,), -1, dtype=labels.dtype)
            valid_ctx = ~(
                torch.cat([pad_lab, labels, pad_lab]).unfold(0, 2 * ctx + 1, 1) == -1
            ).any(dim=1)

        rswa_labels = subject.rswa_labels.long().clone()
        confidence = subject.rswa_conf.float().clone()
        valid_rswa = confidence > self.min_confidence
        if self.rem_mask_only:
            valid_rswa &= labels.eq(self.rswa_config.rem_stage)

        # Rotulos por cabeca. Se o .pt os traz explicitos, usa-os; senao,
        # deriva do inteiro rswa_labels (retrocompat com .pt mono-rotulo).
        if subject.tonic_labels is not None and subject.phasic_labels is not None:
            tonic_labels = subject.tonic_labels.float().clone()
            phasic_labels = subject.phasic_labels.float().clone()
        else:
            tonic_labels = rswa_labels.eq(self.rswa_config.tonic_label).float()
            phasic_labels = rswa_labels.eq(self.rswa_config.phasic_label).float()

        # any_labels: se o .pt nao a traz explicitamente, deriva de
        # rswa_labels=3 no schema exclusivo novo. Em .pt legados isso segue
        # zerado.
        if subject.any_labels is not None:
            any_labels = subject.any_labels.float().clone()
        else:
            any_labels = rswa_labels.eq(self.rswa_config.any_label).float()

        if self.rswa_target_mode == "aasm_proto":
            tonic_targets = (
                subject.tonic_proto_labels.float().clone()
                if subject.tonic_proto_labels is not None else tonic_labels.clone()
            )
            phasic_targets = (
                subject.phasic_proto_labels.float().clone()
                if subject.phasic_proto_labels is not None else phasic_labels.clone()
            )
        else:
            tonic_targets = tonic_labels.clone()
            phasic_targets = phasic_labels.clone()
        any_targets = any_labels.clone()

        # Zera rotulos fora da mascara de validade (cada cabeca independente).
        tonic_labels[~valid_rswa] = 0.0
        phasic_labels[~valid_rswa] = 0.0
        any_labels[~valid_rswa] = 0.0
        tonic_targets[~valid_rswa] = 0.0
        phasic_targets[~valid_rswa] = 0.0
        any_targets[~valid_rswa] = 0.0
        rswa_labels[~valid_rswa] = self.rswa_config.none_label

        # Alias historico "movement" = uniao das 3 cabecas (qualquer movimento
        # anotado, tonico OU fasico OU any). Mantido so para inspecao/QC e
        # compatibilidade com codigo antigo -- NAO e mais o alvo de treino
        # (o treino agora usa as 3 cabecas tonic/phasic/any independentes).
        movement_labels = ((tonic_labels > 0.5) | (phasic_labels > 0.5) | (any_labels > 0.5)).float()
        movement_labels[~valid_rswa] = 0.0

        return {
            "signals": context,
            "emg_center": emg,
            "sleep_stages": labels,
            "staging_valid": valid_ctx,
            "rswa_labels": rswa_labels,
            "phasic_labels": phasic_labels,
            "tonic_labels": tonic_labels,
            "any_labels": any_labels,
            "phasic_targets": phasic_targets,
            "tonic_targets": tonic_targets,
            "any_targets": any_targets,
            "movement_labels": movement_labels,
            "rswa_valid": valid_rswa,
            "rswa_conf": confidence,
            "subject_id": subject.subject_id,
        }


def collate_sleep_analysis_exams(batch):
    b = len(batch)
    lengths = [item["signals"].shape[0] for item in batch]
    tmax = max(lengths)
    _, c, n = batch[0]["signals"].shape
    _, ce, ne = batch[0]["emg_center"].shape
    out = {
        "signals": torch.zeros(b, tmax, c, n),
        "emg_center": torch.zeros(b, tmax, ce, ne),
        "sleep_stages": torch.full((b, tmax), -1, dtype=torch.long),
        "staging_valid": torch.zeros(b, tmax, dtype=torch.bool),
        "padding_mask": torch.zeros(b, tmax, dtype=torch.bool),
        "rswa_labels": torch.zeros(b, tmax, dtype=torch.long),
        "phasic_labels": torch.zeros(b, tmax),
        "tonic_labels": torch.zeros(b, tmax),
        "any_labels": torch.zeros(b, tmax),
        "phasic_targets": torch.zeros(b, tmax),
        "tonic_targets": torch.zeros(b, tmax),
        "any_targets": torch.zeros(b, tmax),
        "movement_labels": torch.zeros(b, tmax),
        "rswa_valid": torch.zeros(b, tmax, dtype=torch.bool),
        "rswa_conf": torch.zeros(b, tmax),
        "subject_ids": [],
        "lengths": torch.tensor(lengths),
    }
    for i, (item, length) in enumerate(zip(batch, lengths)):
        for key in (
            "signals",
            "emg_center",
            "sleep_stages",
            "staging_valid",
            "rswa_labels",
            "phasic_labels",
            "tonic_labels",
            "any_labels",
            "phasic_targets",
            "tonic_targets",
            "any_targets",
            "movement_labels",
            "rswa_valid",
            "rswa_conf",
        ):
            out[key][i, :length] = item[key]
        out["padding_mask"][i, :length] = True
        out["subject_ids"].append(item["subject_id"])
    out["mask"] = out["padding_mask"]
    out["valid_ctx"] = out["staging_valid"]
    return out
