import torch
import torch.nn as nn
from ..config import ModelConfig
from .common import SEBlock,make_group_norm
from .mamba import MambaStack
from ..config import SignalConfig


def _rswa_head(d_in: int, dropout: float) -> nn.Sequential:
    h = d_in // 2
    return nn.Sequential(
        nn.Linear(d_in, h),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(h, 1),
    )


class EMGSubwindowFeatureEncoder(nn.Module):
    """Encoder local sobre sub-janelas intramini-época do EMG."""

    def __init__(self, config: ModelConfig | None = None):
        super().__init__()
        cfg = config or ModelConfig()
        sig = SignalConfig(
            fs=cfg.signal_fs,
            epoch_sec=cfg.signal_epoch_sec,
            samples_per_epoch=cfg.signal_samples_per_epoch,
            context_radius=cfg.signal_context_radius,
        )
        self.cfg = cfg
        self.sig = sig
        self.subwindow_samples = int(round(sig.fs * (cfg.emg_subwindow_ms / 1000.0)))
        if self.subwindow_samples <= 0:
            raise ValueError("emg_subwindow_ms precisa gerar ao menos 1 amostra por sub-janela.")
        if sig.samples_per_epoch % self.subwindow_samples != 0:
            raise ValueError(
                "samples_per_epoch precisa ser múltiplo do tamanho da sub-janela. "
                f"Recebido: samples_per_epoch={sig.samples_per_epoch}, "
                f"subwindow_samples={self.subwindow_samples}."
            )
        self.n_subwindows = sig.samples_per_epoch // self.subwindow_samples
        self.base_feature_dim = 4 if cfg.rswa_emg_in_channels >= 2 else 3
        hidden = int(cfg.emg_subwindow_hidden)
        local_dim = int(cfg.emg_local_embedding_dim)
        self.temporal = nn.Sequential(
            nn.Conv1d(self.base_feature_dim, hidden, kernel_size=3, padding=1, bias=False),
            make_group_norm(hidden),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1, bias=False),
            make_group_norm(hidden),
            nn.ReLU(inplace=True),
        )
        self.proj = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(hidden * self.n_subwindows, local_dim, bias=False),
            nn.LayerNorm(local_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout),
        )
        self.last_shape_info: dict[str, tuple[int, ...]] = {}

    def _compute_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, t, c, n = x.shape
        if n != self.sig.samples_per_epoch:
            raise ValueError(
                "EMGSubwindowFeatureEncoder esperava mini-épocas com "
                f"{self.sig.samples_per_epoch} amostras, recebeu {n}."
            )
        x_main = x[:, :, 0, :].reshape(b, t, self.n_subwindows, self.subwindow_samples)
        rms = x_main.pow(2.0).mean(dim=-1).clamp_min(0.0).sqrt()
        mav = x_main.abs().mean(dim=-1)
        std = x_main.std(dim=-1, unbiased=False)
        features = [rms, mav, std]
        if c >= 2:
            amplitude_relative = x[:, :, 1, :].reshape(
                b, t, self.n_subwindows, self.subwindow_samples
            ).mean(dim=-1)
            features.append(amplitude_relative)
        feat = torch.stack(features, dim=-1)
        return feat, x_main

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, tuple[int, ...]]]:
        feat, subwindows = self._compute_features(x)
        b, t, k, d = feat.shape
        z = feat.reshape(b * t, k, d).permute(0, 2, 1)
        z = self.temporal(z)
        z = self.proj(z)
        z = z.reshape(b, t, -1)
        self.last_shape_info = {
            "emg_subwindows": tuple(subwindows.shape),
            "emg_subwindow_features": tuple(feat.shape),
            "emg_local_embedding": tuple(z.shape),
        }
        return z, dict(self.last_shape_info)

class RSWAFeatureEncoder(nn.Module):
    def __init__(self,config=None,use_se=True):
        super().__init__(); cfg=config or ModelConfig(); self.cfg=cfg
        per=cfg.rswa_emg_filters//len(cfg.emg_kernels); rem=cfg.rswa_emg_filters-per*len(cfg.emg_kernels)
        paths=[]
        for i,k in enumerate(cfg.emg_kernels):
            co=per+(rem if i==0 else 0)
            paths.append(
                nn.Sequential(
                    nn.Conv1d(cfg.rswa_emg_in_channels,co,k,padding=k//2,bias=False),
                    make_group_norm(co),
                    nn.ReLU(inplace=True),
                    nn.MaxPool1d(2,2),
                )
            )
        self.paths=nn.ModuleList(paths)
        self.drop=nn.Dropout(cfg.dropout)
        self.se=SEBlock(cfg.rswa_emg_filters) if use_se else nn.Identity()
        self.refine=nn.Sequential(
            nn.Conv1d(cfg.rswa_emg_filters,cfg.rswa_emg_filters,5,padding=2,bias=False),
            make_group_norm(cfg.rswa_emg_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2,2),
            nn.Conv1d(cfg.rswa_emg_filters,cfg.d_model,3,padding=1,bias=False),
            make_group_norm(cfg.d_model),
            nn.ReLU(inplace=True),
        )
        self.pool=nn.AdaptiveAvgPool1d(1)
        self.use_emg_subwindow_features = bool(cfg.use_emg_subwindow_features)
        self.local_encoder = (
            EMGSubwindowFeatureEncoder(cfg)
            if self.use_emg_subwindow_features
            else None
        )
        self.local_fusion = (
            nn.Sequential(
                nn.Linear(cfg.d_model + cfg.emg_local_embedding_dim, cfg.d_model, bias=False),
                nn.LayerNorm(cfg.d_model),
                nn.ReLU(inplace=True),
                nn.Dropout(cfg.dropout),
            )
            if self.use_emg_subwindow_features
            else None
        )
        self.last_shape_info: dict[str, tuple[int, ...]] = {}
    def forward(self,x):
        b,t,c,n=x.shape; z=x.reshape(b*t,c,n)
        ys=[path(z) for path in self.paths]; m=min(y.shape[-1] for y in ys)
        z=self.drop(torch.cat([y[...,:m] for y in ys],1))
        z=self.se(z)
        z=self.refine(z)
        z=self.pool(z).squeeze(-1)
        cnn_embedding = z.reshape(b,t,-1)
        if not self.use_emg_subwindow_features:
            self.last_shape_info = {
                "emg_raw": tuple(x.shape),
                "emg_cnn_embedding": tuple(cnn_embedding.shape),
                "fused_emg_embedding": tuple(cnn_embedding.shape),
            }
            return cnn_embedding
        if self.local_encoder is None or self.local_fusion is None:
            raise RuntimeError("Encoder local EMG não foi inicializado.")
        local_embedding, local_shapes = self.local_encoder(x)
        if local_embedding.shape[:2] != cnn_embedding.shape[:2]:
            raise ValueError(
                "local_embedding e cnn_embedding precisam alinhar em (B,T); "
                f"recebido {tuple(local_embedding.shape)} vs {tuple(cnn_embedding.shape)}."
            )
        fused = self.local_fusion(torch.cat([cnn_embedding, local_embedding], dim=-1))
        self.last_shape_info = {
            "emg_raw": tuple(x.shape),
            **local_shapes,
            "emg_cnn_embedding": tuple(cnn_embedding.shape),
            "fused_emg_embedding": tuple(fused.shape),
        }
        return fused

class RSWADetectionNet(nn.Module):
    def __init__(self, config=None, use_se=True, *, stage_conditioning: bool | None = None):
        super().__init__()
        cfg = config or ModelConfig()
        self.cfg = cfg
        self.encoder = RSWAFeatureEncoder(cfg, use_se)
        self.use_stage_conditioning = bool(
            cfg.rswa_stage_conditioning if stage_conditioning is None else stage_conditioning
        )
        self.stage_context_dim = int(cfg.staging_num_classes)
        self.stage_fusion = (
            nn.Sequential(
                nn.Linear(
                    cfg.d_model + self.stage_context_dim,
                    cfg.d_model,
                    bias=False,
                ),
                nn.LayerNorm(cfg.d_model),
                nn.ReLU(inplace=True),
                nn.Dropout(cfg.dropout),
            )
            if self.use_stage_conditioning
            else None
        )
        self.temporal = MambaStack(
            cfg.d_model,
            cfg.rswa_mamba_layers,
            cfg.d_state,
            cfg.dropout,
        )
        h = cfg.d_model // 2
        # Tres cabecas independentes por mini-epoca (multi-rotulo, BCE cada):
        #   tonic_head  : evento tonico confirmado (duracao >= 15s, score >= 2.0)
        #   phasic_head : evento fasico confirmado (0.1s <= duracao <= 5s, score >= 2.0)
        #   any_head    : evento com amplitude confirmada mas duracao ambigua
        #                 (5s < duracao < 15s) -- categoria "any" do limiar duplo,
        #                 NAO e um "qualquer movimento" (isso seria a uniao das 3
        #                 cabecas em pos-processamento, nao uma cabeca propria).
        # Substitui a antiga cabeca unica movement_head (commit ec5d505, revertido).
        self.tonic_head = _rswa_head(cfg.d_model, cfg.dropout)
        self.phasic_head = _rswa_head(cfg.d_model, cfg.dropout)
        self.any_head = _rswa_head(cfg.d_model, cfg.dropout)
    def forward(self, emg_center, mask=None, stage_probs=None):
        z = self.encoder(emg_center)
        if self.stage_fusion is not None and stage_probs is not None:
            if stage_probs.shape[:2] != z.shape[:2]:
                raise ValueError(
                    "stage_probs deve alinhar com (B,T) das features RSWA; "
                    f"recebeu {tuple(stage_probs.shape)} para features "
                    f"{tuple(z.shape)}."
                )
            if stage_probs.shape[-1] != self.stage_context_dim:
                raise ValueError(
                    "stage_probs deve ter "
                    f"{self.stage_context_dim} classes de estagio; recebeu "
                    f"{stage_probs.shape[-1]}."
                )
            z = self.stage_fusion(
                torch.cat([z, stage_probs.to(z.dtype)], dim=-1)
            )
        z = self.temporal(z, mask)
        return {
            "tonic_logits":self.tonic_head(z).squeeze(-1),
            "phasic_logits":self.phasic_head(z).squeeze(-1),
            "any_logits":self.any_head(z).squeeze(-1),
        }
    def n_params(self): return sum(p.numel() for p in self.parameters() if p.requires_grad)
