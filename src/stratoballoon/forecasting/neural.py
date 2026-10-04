"""Neural rungs of the ladder: an LSTM and a temporal CNN.

They exist to answer one question: with pooled multi-column data (hundreds of
thousands of windows), does sequence modelling beat ridge?
Both predict the change from persistence, are trained on the training split
only, use early stopping on the validation split, and are scored by the same
ladder code as every other model. They are kept only if they beat ridge
outside the bootstrap interval.

Inputs are the same feature vector as the linear models, reshaped back into a
(time, channel) sequence; the static features (time of day, season, position)
are repeated along the sequence.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from stratoballoon.forecasting.models import Forecaster, safe_std


def _split_features(names: list[str]):
    """Indices of sequence features grouped by lag (oldest first) and static ones."""
    lags = sorted({int(n.split("_lag")[1]) for n in names if "_lag" in n}, reverse=True)
    seq = [[i for i, n in enumerate(names) if n.endswith(f"_lag{k}")] for k in lags]
    static = [i for i, n in enumerate(names) if "_lag" not in n]
    return seq, static


class _SeqNet(nn.Module):
    def __init__(self, kind: str, n_in: int, n_out: int, hidden: int, steps: int):
        super().__init__()
        self.kind = kind
        if kind == "lstm":
            self.body = nn.LSTM(n_in, hidden, batch_first=True)
        else:   # causal temporal CNN: dilated 1-D convolutions over the window
            # Each layer with dilation d shortens the sequence by d, so only as
            # many doubling dilations are stacked as the window can absorb:
            # 8 steps takes 1, 2, 4; a 4-step window (24 h at 6-hourly) takes 1, 2.
            layers, d, left, ch = [], 1, steps, n_in
            while left - d >= 1:
                layers += [nn.Conv1d(ch, hidden, 2, dilation=d), nn.GELU()]
                left, ch, d = left - d, hidden, d * 2
            if not layers:      # a single-step window: a 1x1 convolution
                layers = [nn.Conv1d(n_in, hidden, 1), nn.GELU()]
            self.body = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, n_out))

    def forward(self, x):                          # x (B, T, F)
        if self.kind == "lstm":
            h, _ = self.body(x)
            z = h[:, -1]
        else:
            z = self.body(x.transpose(1, 2))[:, :, -1]
        return self.head(z)


class _NeuralForecaster(Forecaster):
    kind = "lstm"

    def __init__(self, hidden: int = 64, epochs: int = 30, batch: int = 512, lr: float = 2e-3,
                 patience: int = 4, max_train: int = 400_000, seed: int = 0):
        self.hidden, self.epochs, self.batch, self.lr = hidden, epochs, batch, lr
        self.patience, self.max_train, self.seed = patience, max_train, seed

    def _seq(self, X):
        Xs = (X - self.mu) / self.sd
        seq = np.stack([Xs[:, idx] for idx in self.seq_idx], 1)               # (N, T, f)
        stat = np.repeat(Xs[:, self.static_idx][:, None], seq.shape[1], 1)
        return torch.from_numpy(np.concatenate([seq, stat], -1).astype("float32"))

    def fit(self, train, val, A, h_steps):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.seq_idx, self.static_idx = _split_features(train.feature_names)
        self.mu, self.sd = train.X.mean(0), safe_std(train.X)
        self.shape = train.Y.shape[1:]
        self.y_sd = train.Y.reshape(len(train.Y), -1).std(0) + 1e-6
        idx = rng.choice(len(train.X), min(self.max_train, len(train.X)), replace=False)
        Xt, Yt = self._seq(train.X[idx]), torch.from_numpy(
            (train.Y[idx].reshape(len(idx), -1) / self.y_sd).astype("float32"))
        vi = rng.choice(len(val.X), min(100_000, len(val.X)), replace=False)
        Xv, Yv = self._seq(val.X[vi]), torch.from_numpy(
            (val.Y[vi].reshape(len(vi), -1) / self.y_sd).astype("float32"))
        self.net = _SeqNet(self.kind, Xt.shape[-1], Yt.shape[-1], self.hidden, Xt.shape[1])
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-4)
        best, bad, state, self.history = np.inf, 0, None, []
        for ep in range(self.epochs):
            self.net.train()
            perm = torch.randperm(len(Xt))
            for i in range(0, len(Xt), self.batch):
                b = perm[i:i + self.batch]
                opt.zero_grad()
                loss = nn.functional.mse_loss(self.net(Xt[b]), Yt[b])
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
            self.net.eval()
            with torch.no_grad():
                vl = float(nn.functional.mse_loss(self.net(Xv), Yv))
            self.history.append(vl)
            if vl < best - 1e-5:
                best, bad = vl, 0
                state = {k: v.clone() for k, v in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        self.net.load_state_dict(state)
        self.best_epoch = int(np.argmin(self.history)) + 1
        return self

    def predict(self, s, A, h_steps):
        self.net.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(s.X), 20_000):
                out.append(self.net(self._seq(s.X[i:i + 20_000])).numpy())
        return (np.concatenate(out) * self.y_sd).reshape(-1, *self.shape).astype("float32")


class LSTM(_NeuralForecaster):
    name = "lstm"
    kind = "lstm"


class TCN(_NeuralForecaster):
    name = "tcn"
    kind = "tcn"
