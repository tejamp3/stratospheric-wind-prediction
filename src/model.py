"""LSTM forecaster and the persistence baseline it must beat."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C


class WindLSTM(nn.Module):
    """Stacked LSTM that maps a 24 h history to multi-horizon (u, v) forecasts.

    An LSTM suits this problem because stratospheric wind at a fixed point is a
    smooth, strongly autocorrelated series with a slowly rotating background
    flow: the recurrent state can carry the current regime (e.g. easterly phase
    of the QBO) while the gates suppress hour-to-hour noise. The head predicts
    all horizons at once so the shared encoder is trained on every lead time.

    In `residual` mode the head predicts a *correction to persistence* rather
    than the wind itself: the last observed (u, v) is added back to the output.
    Because the wind is so strongly autocorrelated, persistence already gets most
    of the answer, so learning only the correction is a far easier target and the
    model starts from a strong forecast instead of having to rediscover it. This
    is a framing change, not an architecture change - the network is identical.

    The skip connection relies on the first two feature channels being the same
    quantities as the two targets, standardised identically. `build_dataset` fits
    both scalers on the same train slice of the same columns, so the statistics
    match exactly; `assert_residual_safe` checks it rather than trusting it.
    """

    def __init__(self, n_features: int, n_horizons: int = len(C.HORIZONS),
                 n_targets: int = 2, hidden: int = C.HIDDEN,
                 num_layers: int = C.NUM_LAYERS, dropout: float = C.DROPOUT,
                 residual: bool = False):
        super().__init__()
        self.n_horizons, self.n_targets = n_horizons, n_targets
        self.residual = residual
        self.lstm = nn.LSTM(
            input_size=n_features, hidden_size=hidden, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, n_horizons * n_targets)
        if residual:
            # Start life as exact persistence, then learn away from it.
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F) -> use the final hidden state as the sequence summary.
        out, _ = self.lstm(x)
        h = self.drop(out[:, -1, :])
        y = self.head(h).view(-1, self.n_horizons, self.n_targets)
        if self.residual:
            # Last observed (u, v), broadcast across horizons: persistence.
            y = y + x[:, -1, :self.n_targets].unsqueeze(1)
        return y

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def assert_residual_safe(scaler_x, scaler_y, n_targets: int = 2) -> None:
    """Verify the assumption the residual skip connection depends on.

    The skip adds the last observed feature channels straight onto the output,
    which is only the identity-in-physical-units if the first n_targets feature
    channels are standardised with exactly the same mean and scale as the
    targets. That holds by construction in build_dataset, but a future change to
    the feature order or the scaler would break it silently and produce a model
    that trains happily while forecasting nonsense.
    """
    import numpy as _np
    if not (_np.allclose(scaler_x.mean[:n_targets], scaler_y.mean, rtol=1e-6)
            and _np.allclose(scaler_x.std[:n_targets], scaler_y.std, rtol=1e-6)):
        raise ValueError(
            "residual mode requires the first "
            f"{n_targets} features to share the targets' normalisation; "
            f"got feature mean/std {scaler_x.mean[:n_targets]}/"
            f"{scaler_x.std[:n_targets]} vs target {scaler_y.mean}/{scaler_y.std}")


def persistence_forecast(last_uv: np.ndarray,
                         n_horizons: int = len(C.HORIZONS)) -> np.ndarray:
    """Baseline: the wind holds its last observed value at every horizon.

    Persistence is the honest benchmark in the stratosphere. Flow there is
    smooth and slowly varying, so "no change" is already a strong forecast, and
    operational meteorology scores short-range models against exactly this.

    last_uv: (N, 2) the final observed (u50, v50) of each window, in m/s.
    Returns (N, n_horizons, 2) in m/s.
    """
    return np.repeat(last_uv[:, None, :], n_horizons, axis=1)


def count_size_mb(model: nn.Module) -> float:
    return sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6


class RidgeBaseline:
    """Linear ridge regression on the flattened input window.

    This exists to answer the question an interviewer should ask: does this
    problem actually need a recurrent network? A ridge fit on the same flattened
    (steps x features) input, trained on the same split, is the cheapest
    non-trivial learner available. If the LSTM cannot beat it, the LSTM is not
    earning its complexity.
    """

    def __init__(self, alpha: float = 1.0):
        from sklearn.linear_model import Ridge
        self.model = Ridge(alpha=alpha)
        self.shape: tuple[int, int] | None = None

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "RidgeBaseline":
        self.shape = Y.shape[1:]
        self.model.fit(X.reshape(len(X), -1), Y.reshape(len(Y), -1))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        out = self.model.predict(X.reshape(len(X), -1))
        return out.reshape(len(X), *self.shape).astype("float32")
