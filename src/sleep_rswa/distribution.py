from __future__ import annotations

from dataclasses import dataclass, field

import torch

from sleep_rswa.constants import (
    NUM_SLEEP_STAGES,
    SLEEP_STAGE_NAMES,
)


@dataclass
class StageDistribution:
    num_classes: int = NUM_SLEEP_STAGES
    counts: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        self.counts = torch.zeros(
            self.num_classes,
            dtype=torch.long,
        )

    def reset(self) -> None:
        self.counts.zero_()

    def update(
        self,
        targets: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> None:
        """
        targets:
            [B, T] ou qualquer tensor contendo classes.

        mask:
            [B, T], com True para posições válidas.
        """
        targets = targets.detach()

        if mask is not None:
            mask = mask.detach().bool()
            targets = targets[mask]
        else:
            targets = targets.reshape(-1)

        # Remove posições inválidas, por exemplo -1.
        valid = (
            (targets >= 0)
            & (targets < self.num_classes)
        )

        targets = targets[valid]

        if targets.numel() == 0:
            return

        batch_counts = torch.bincount(
            targets.to(
                device="cpu",
                dtype=torch.long,
            ),
            minlength=self.num_classes,
        )

        self.counts += batch_counts

    @property
    def total(self) -> int:
        return int(self.counts.sum().item())

    def as_dict(self) -> dict[str, dict[str, float | int]]:
        total = self.total

        result: dict[str, dict[str, float | int]] = {}

        for class_id in range(self.num_classes):
            name = SLEEP_STAGE_NAMES.get(
                class_id,
                str(class_id),
            )

            count = int(self.counts[class_id].item())

            percentage = (
                100.0 * count / total
                if total > 0
                else 0.0
            )

            result[name] = {
                "count": count,
                "percentage": percentage,
            }

        return result