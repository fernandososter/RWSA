from __future__ import annotations

import inspect

import torch
import torch.nn as nn

from ..config import ModelConfig
from .staging import SleepStagingNet
from .rswa import RSWADetectionNet


class SleepStagingRSWASystem(nn.Module):
    def __init__(self, staging_model=None, rswa_model=None):
        super().__init__()
        self.staging_model = staging_model or SleepStagingNet()
        self.rswa_model = rswa_model or RSWADetectionNet()
        self.cfg = getattr(
            self.staging_model,
            "cfg",
            getattr(self.rswa_model, "cfg", ModelConfig()),
        )
        self.use_stage_conditioning = bool(
            getattr(self.cfg, "rswa_stage_conditioning", True)
        )
        self.detach_stage_probs = bool(
            getattr(self.cfg, "rswa_stage_conditioning_detach", True)
        )
        self._rswa_accepts_stage_probs = (
            "stage_probs" in inspect.signature(self.rswa_model.forward).parameters
        )

    def forward(self, signals, emg_center, mask=None):
        staging_logits = self.staging_model(signals, mask)
        stage_probs = torch.softmax(staging_logits, dim=-1)
        stage_context = stage_probs.detach() if self.detach_stage_probs else stage_probs
        rswa_kwargs = {"mask": mask}
        if self.use_stage_conditioning and self._rswa_accepts_stage_probs:
            rswa_kwargs["stage_probs"] = stage_context
        return {
            "staging_logits": staging_logits,
            "stage_probs": stage_probs,
            **self.rswa_model(emg_center, **rswa_kwargs),
        }

    def n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
