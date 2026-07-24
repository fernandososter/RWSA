"""
Dataset/DataLoader isolado para o detector de movimento.

Constroi janelas de EMG (z-scored por exame) centradas em cada mini-epoca.
Nao importa nada de src/sleep_rswa.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from .dataio import Exam, zscore_emg, SAMPLES_PER_EPOCH


class MovementWindowDataset(Dataset):
    """Uma amostra = janela de `window_epochs` mini-epocas -> rotulo da central.

    Cada exame e z-scored globalmente. Bordas preenchidas com zero (sinal
    z-scored -> zero ~ media do exame).
    """

    def __init__(self, exams: list[Exam], window_epochs: int = 5):
        assert window_epochs % 2 == 1, "window_epochs deve ser impar"
        self.window_epochs = window_epochs
        self.half = window_epochs // 2
        self.spe = SAMPLES_PER_EPOCH

        self._emg = []       # lista de [T, 300] z-scored por exame
        self._labels = []    # lista de [T]
        self.index = []      # (exam_idx, epoch_idx)
        self.subject_ids = []
        for ei, ex in enumerate(exams):
            z = zscore_emg(ex.emg).astype(np.float32)
            self._emg.append(z)
            self._labels.append(ex.movement.astype(np.float32))
            self.subject_ids.append(ex.subject_id)
            for m in range(ex.n_epochs):
                self.index.append((ei, m))

    def __len__(self):
        return len(self.index)

    def _window(self, ei: int, m: int) -> np.ndarray:
        emg = self._emg[ei]              # [T,300]
        T = emg.shape[0]
        lo, hi = m - self.half, m + self.half
        parts = []
        for k in range(lo, hi + 1):
            if 0 <= k < T:
                parts.append(emg[k])
            else:
                parts.append(np.zeros(self.spe, dtype=np.float32))
        return np.concatenate(parts)     # [window_epochs*300]

    def __getitem__(self, idx):
        ei, m = self.index[idx]
        x = self._window(ei, m)[None, :]         # [1, L]
        y = self._labels[ei][m]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32)

    def labels_array(self) -> np.ndarray:
        return np.array([self._labels[ei][m] for ei, m in self.index], dtype=np.float32)


def build_tensors(exams, window_epochs=5):
    """Pre-computa TODAS as janelas de forma vetorizada -> (X[N,1,L], y[N]).

    Muito mais rapido que montar janela-a-janela em Python no __getitem__.
    Cada exame e z-scored globalmente; bordas preenchidas com zero.
    """
    import numpy as np
    half = window_epochs // 2
    spe = SAMPLES_PER_EPOCH
    Xs, ys = [], []
    for ex in exams:
        z = zscore_emg(ex.emg).astype(np.float32)     # [T,300]
        T = z.shape[0]
        pad = np.zeros((half, spe), dtype=np.float32)
        zp = np.concatenate([pad, z, pad], axis=0)    # [T+2*half, 300]
        # janela m = linhas [m, m+window_epochs) de zp -> flatten
        idx = np.arange(T)[:, None] + np.arange(window_epochs)[None, :]   # [T, window]
        win = zp[idx]                                  # [T, window, 300]
        win = win.reshape(T, window_epochs * spe)      # [T, L]
        Xs.append(win)
        ys.append(ex.movement.astype(np.float32))
    X = np.concatenate(Xs, axis=0)[:, None, :]         # [N,1,L]
    y = np.concatenate(ys, axis=0)                     # [N]
    return torch.from_numpy(X), torch.from_numpy(y)


def subsample_negatives(X, y, neg_per_pos=5, seed=0):
    """Mantem todos positivos e amostra negativos ate neg_per_pos:1. So no treino."""
    import numpy as np
    rng = np.random.default_rng(seed)
    yn = y.numpy()
    pos = np.where(yn > 0.5)[0]
    neg = np.where(yn <= 0.5)[0]
    n_keep = min(len(neg), int(len(pos) * neg_per_pos))
    neg_keep = rng.choice(neg, size=n_keep, replace=False)
    idx = np.concatenate([pos, neg_keep])
    rng.shuffle(idx)
    idx = torch.from_numpy(idx)
    return X[idx], y[idx]


def make_loaders(train_exams, val_exams, window_epochs=5, batch_size=256,
                 num_workers=0, neg_per_pos=5, seed=0):
    from torch.utils.data import DataLoader, TensorDataset
    Xtr, ytr = build_tensors(train_exams, window_epochs)
    if neg_per_pos is not None:
        Xtr, ytr = subsample_negatives(Xtr, ytr, neg_per_pos=neg_per_pos, seed=seed)
    Xva, yva = build_tensors(val_exams, window_epochs)
    tr = TensorDataset(Xtr, ytr)
    va = TensorDataset(Xva, yva)
    tl = DataLoader(tr, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=False)
    vl = DataLoader(va, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    return tr, va, tl, vl
