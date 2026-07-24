"""
CNN simples 1D para detectar movimento em uma mini-epoca de EMG do mento.

Modulo isolado: nao importa nada de src/sleep_rswa.

Entrada : janela de W mini-epocas de EMG bruto (z-scored por exame),
          concatenadas -> [B, 1, W*300].
Saida   : 1 logit (movimento na mini-epoca CENTRAL da janela).

Arquitetura: bloco de entrada multikernel (kernels curtos e longos capturam
picos rapidos e bursts sustentados de EMG), seguido de blocos conv+pool,
pooling global e uma cabeca linear. Pequena de proposito (poucos milhares de
parametros) para nao sobreajustar com 4 sujeitos.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MultiKernelStem(nn.Module):
    """Primeira camada: convolucoes paralelas com kernels de tamanhos diferentes.

    Usa stride para reduzir a resolucao ja na entrada (EMG a 100Hz nao precisa de
    resolucao plena para padroes de movimento) — corta muito o custo em CPU.
    """

    def __init__(self, out_ch_each: int = 6, kernel_sizes=(7, 15, 31, 63), stride: int = 2):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Conv1d(1, out_ch_each, kernel_size=k, stride=stride, padding=k // 2)
            for k in kernel_sizes
        ])
        self.bn = nn.BatchNorm1d(out_ch_each * len(kernel_sizes))
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(2)  # reduz mais 2x antes dos blocos pesados

    def forward(self, x):
        feats = [b(x) for b in self.branches]
        x = torch.cat(feats, dim=1)
        return self.pool(self.act(self.bn(x)))


class ConvBlock(nn.Module):
    def __init__(self, cin, cout, k=7, pool=4, dropout=0.1):
        super().__init__()
        self.conv = nn.Conv1d(cin, cout, kernel_size=k, padding=k // 2)
        self.bn = nn.BatchNorm1d(cout)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(pool)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.pool(self.act(self.bn(self.conv(x)))))


class MovementCNN(nn.Module):
    """CNN binaria para movimento. ~poucos milhares de parametros."""

    def __init__(self, window_epochs: int = 5, samples_per_epoch: int = 300,
                 stem_ch_each: int = 6, dropout: float = 0.1):
        super().__init__()
        self.window_epochs = window_epochs
        self.samples_per_epoch = samples_per_epoch
        self.input_len = window_epochs * samples_per_epoch

        self.stem = MultiKernelStem(out_ch_each=stem_ch_each, stride=2)
        stem_ch = stem_ch_each * 4
        self.block1 = ConvBlock(stem_ch, 32, k=7, pool=4, dropout=dropout)
        self.block2 = ConvBlock(32, 48, k=5, pool=4, dropout=dropout)
        self.block3 = ConvBlock(48, 64, k=3, pool=2, dropout=dropout)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        # x: [B, 1, L]
        x = self.stem(x)
        x = self.block1(x); x = self.block2(x); x = self.block3(x)
        x = self.gap(x)
        return self.head(x).squeeze(-1)  # [B] logits


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class FocalLoss(nn.Module):
    """Focal loss binaria com alpha (peso da classe positiva) para desbalanceamento."""

    def __init__(self, alpha: float = 0.85, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits, targets):
        bce = self.bce(logits, targets)
        p = torch.sigmoid(logits)
        pt = torch.where(targets > 0.5, p, 1 - p)
        alpha_t = torch.where(targets > 0.5, self.alpha, 1 - self.alpha)
        loss = alpha_t * (1 - pt) ** self.gamma * bce
        return loss.mean()
