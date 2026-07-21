from dataclasses import dataclass, field
from pathlib import Path
import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from typing import Sequence

import torch
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
    emg_signals: torch.Tensor | None = None
    n_epochs: int = field(init=False)

    def __post_init__(self) -> None:
        self.n_epochs = int(self.signals.shape[0])
        if self.sleep_stages.shape[0] != self.n_epochs:
            raise ValueError(f"{self.subject_id}: signals e sleep_stages possuem comprimentos diferentes.")
        if self.emg_signals is not None and self.emg_signals.shape[0] != self.n_epochs:
            raise ValueError(f"{self.subject_id}: emg_signals possui comprimento incompatível.")


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
    return SubjectData(
        subject_id=str(obj.get("subject_id", path.stem)),
        signals=signals,
        sleep_stages=stages,
        rswa_labels=rswa,
        rswa_conf=conf,
        emg_signals=emg,
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


class SleepAnalysisDataset(Dataset):
    def __init__(
        self,
        subjects: Sequence[SubjectData],
        min_confidence: float = 0.0,
        rem_mask_only: bool = True,
    ):
        self.subjects = list(subjects)
        self.min_confidence = min_confidence
        self.rem_mask_only = rem_mask_only
        self.signal_config = SignalConfig()
        self.rswa_config = RSWAConfig()

    def __len__(self) -> int:
        return len(self.subjects)

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
        return _zscore_per_channel(emg)[:, :, : self.signal_config.samples_per_epoch]


    def stage_distribution(self) -> StageDistribution:
        distribution = StageDistribution()

        for subject in self.subjects:
            distribution.update(
                subject.sleep_stages,
            )

        return distribution

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
        pad = torch.zeros(ctx, c, n, dtype=signals.dtype)
        context = (
            torch.cat([pad, signals, pad], dim=0)
            .unfold(0, 2 * ctx + 1, 1)
            .permute(0, 1, 3, 2)
            .reshape(t, c, (2 * ctx + 1) * n)
        )

        labels = subject.sleep_stages.long()
        pad_lab = torch.full((ctx,), -1, dtype=labels.dtype)
        valid_ctx = ~(
            torch.cat([pad_lab, labels, pad_lab]).unfold(0, 2 * ctx + 1, 1) == -1
        ).any(dim=1)

        rswa_labels = subject.rswa_labels.long().clone()
        confidence = subject.rswa_conf.float().clone()
        valid_rswa = confidence > self.min_confidence
        if self.rem_mask_only:
            valid_rswa &= labels.eq(self.rswa_config.rem_stage)
        rswa_labels[~valid_rswa] = self.rswa_config.none_label

        return {
            "signals": context,
            "emg_center": emg,
            "sleep_stages": labels,
            "staging_valid": valid_ctx,
            "rswa_labels": rswa_labels,
            "phasic_labels": rswa_labels.eq(self.rswa_config.phasic_label).float(),
            "tonic_labels": rswa_labels.eq(self.rswa_config.tonic_label).float(),
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
            "rswa_valid",
            "rswa_conf",
        ):
            out[key][i, :length] = item[key]
        out["padding_mask"][i, :length] = True
        out["subject_ids"].append(item["subject_id"])
    out["mask"] = out["padding_mask"]
    out["valid_ctx"] = out["staging_valid"]
    return out
