"""
Motor de treino isolado para o detector de movimento.

Treina a MovementCNN com focal loss, early stopping por PR-AUC de validacao.
Roda em CPU. Nao importa nada de src/sleep_rswa.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from .model import MovementCNN, FocalLoss, count_params
from .metrics import pr_auc


@dataclass
class TrainConfig:
    window_epochs: int = 5
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 40
    patience: int = 6
    focal_alpha: float = 0.6
    focal_gamma: float = 2.0
    neg_per_pos: int = 5
    dropout: float = 0.1
    seed: int = 0


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def predict_scores(model, loader, device="cpu") -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        x = x.to(device)
        logit = model(x)
        ps.append(torch.sigmoid(logit).cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ps)


def train_one(train_loader, val_loader, cfg: TrainConfig, device="cpu", verbose=True):
    """Treina um modelo; devolve (model, history, best_state)."""
    set_seed(cfg.seed)
    model = MovementCNN(window_epochs=cfg.window_epochs, dropout=cfg.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = FocalLoss(alpha=cfg.focal_alpha, gamma=cfg.focal_gamma)

    best_auc, best_state, best_epoch = -1.0, None, -1
    history = []
    for epoch in range(cfg.max_epochs):
        model.train()
        tot, n = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logit = model(x)
            loss = loss_fn(logit, y)
            loss.backward()
            opt.step()
            tot += loss.item() * len(y); n += len(y)
        yv, pv = predict_scores(model, val_loader, device)
        auc = pr_auc(yv, pv)
        history.append({"epoch": epoch, "train_loss": tot / max(n, 1), "val_pr_auc": auc})
        if verbose:
            print(f"  epoch {epoch:2d}  train_loss={tot/max(n,1):.4f}  val_PR_AUC={auc:.4f}")
        if auc > best_auc:
            best_auc, best_epoch = auc, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= cfg.patience:
            if verbose:
                print(f"  early stop @ epoch {epoch} (melhor PR-AUC={best_auc:.4f} @ {best_epoch})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, best_state, best_auc


def model_size(cfg: TrainConfig) -> int:
    return count_params(MovementCNN(window_epochs=cfg.window_epochs, dropout=cfg.dropout))
