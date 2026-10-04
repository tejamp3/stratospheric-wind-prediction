"""The forecasting ladder, simplest first.

Every model predicts the change from persistence, shape (N, H, L, 2). A model
earns its place only by beating the rung below it outside the bootstrap
confidence interval; the evaluation script reports that, not this module.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stratoballoon.atmosphere import Atmosphere
from stratoballoon.forecasting.features import Samples


class Forecaster:
    name = "base"

    def fit(self, train: Samples, val: Samples, A: Atmosphere, h_steps: list[int]):
        return self

    def predict(self, s: Samples, A: Atmosphere, h_steps: list[int]) -> np.ndarray:
        raise NotImplementedError


class Persistence(Forecaster):
    """The wind stays as it is now."""
    name = "persistence"

    def predict(self, s, A, h_steps):
        return np.zeros_like(s.Y)


def _check_grid(model, A: Atmosphere):
    """Per-cell statistics only make sense on the grid they were fitted on."""
    if model.grid != (len(A.lats), len(A.lons)):
        raise ValueError(f"{model.name} was fitted on a {model.grid} grid and cannot "
                         f"forecast on {(len(A.lats), len(A.lons))}")


def _cell_stats(A: Atmosphere, t_idx: np.ndarray, key_fn) -> dict:
    """Mean (u, v) per (cell, level, key) over the given times."""
    when = pd.DatetimeIndex(A.times[t_idx])
    keys = key_fn(when)
    uv = np.stack([A.u[t_idx], A.v[t_idx]], axis=-1)      # (n, L, Y, X, 2)
    out = {}
    for k in np.unique(keys):
        out[k] = uv[keys == k].mean(0)                     # (L, Y, X, 2)
    return out


class TidePersistence(Forecaster):
    """Persistence corrected for the mean daily cycle at each column.

    The wind here has a 24 h tide large enough to make plain persistence better
    at 24 h than at 12 h. This removes that artefact: forecast = now +
    (mean anomaly at the target hour - mean anomaly at the issue hour).
    """
    name = "tide_persistence"

    def fit(self, train, val, A, h_steps):
        t_idx = np.unique(train.t_idx)
        stats = _cell_stats(A, t_idx, lambda w: w.hour.to_numpy())
        mean = np.mean(list(stats.values()), axis=0)
        self.anom = {h: s - mean for h, s in stats.items()}
        self.ny, self.nx = A.u.shape[2], A.u.shape[3]
        self.grid = (self.ny, self.nx)
        return self

    def predict(self, s, A, h_steps):
        _check_grid(self, A)
        out = np.zeros_like(s.Y)
        yi, xi = np.divmod(s.cell, self.nx)
        step = A.step_hours
        issue_hour = pd.DatetimeIndex(A.times[s.t_idx]).hour.to_numpy()
        for k, h in enumerate(h_steps):
            tgt_hour = (issue_hour + int(h * step)) % 24
            for hr in np.unique(issue_hour):
                for th in np.unique(tgt_hour[issue_hour == hr]):
                    m = (issue_hour == hr) & (tgt_hour == th)
                    a_t = self.anom.get(th)
                    a_i = self.anom.get(hr)
                    if a_t is None or a_i is None:
                        continue
                    out[m, k] = (a_t[:, yi[m], xi[m]] - a_i[:, yi[m], xi[m]]).transpose(1, 0, 2)
        return out


class Climatology(Forecaster):
    """Mean wind for the target calendar month and hour at each column."""
    name = "climatology"

    def fit(self, train, val, A, h_steps):
        t_idx = np.unique(train.t_idx)
        self.stats = _cell_stats(A, t_idx, lambda w: (w.month * 100 + w.hour).to_numpy())
        self.fallback = np.mean(list(self.stats.values()), axis=0)
        self.nx = A.u.shape[3]
        self.grid = (len(A.lats), len(A.lons))
        return self

    def predict(self, s, A, h_steps):
        _check_grid(self, A)
        out = np.empty_like(s.Y)
        yi, xi = np.divmod(s.cell, self.nx)
        step = A.step_hours
        issue = pd.DatetimeIndex(A.times[s.t_idx])
        for k, h in enumerate(h_steps):
            tgt = issue + pd.Timedelta(hours=h * step)
            key = (tgt.month * 100 + tgt.hour).to_numpy()
            for kk in np.unique(key):
                m = key == kk
                c = self.stats.get(kk, self.fallback)
                out[m, k] = c[:, yi[m], xi[m]].transpose(1, 0, 2) - s.current[m]
        return out


class MovingAverage(Forecaster):
    """Forecast the mean of the input window (a smoothed persistence)."""
    name = "moving_average"

    def fit(self, train, val, A, h_steps):
        names = train.feature_names
        L = train.current.shape[1]
        self.u_cols = [[i for i, n in enumerate(names) if n.startswith(f"u{int(A.levels[l])}_")]
                       for l in range(L)]
        self.v_cols = [[i for i, n in enumerate(names) if n.startswith(f"v{int(A.levels[l])}_")]
                       for l in range(L)]
        return self

    def predict(self, s, A, h_steps):
        mean = np.stack([np.stack([s.X[:, uc].mean(1), s.X[:, vc].mean(1)], axis=-1)
                         for uc, vc in zip(self.u_cols, self.v_cols)], axis=1)
        return np.repeat((mean - s.current)[:, None], len(h_steps), axis=1)


def safe_std(X: np.ndarray) -> np.ndarray:
    """Per-feature standard deviation, with constant features given scale 1.

    Dividing by a near-zero spread turns any later departure from the training
    value into an enormous standardised input. That is not hypothetical: a
    model trained only on 00 and 12 UTC forecasts sees sin(hour) as constant,
    and at 06 UTC the same feature came out at a million standard deviations
    and produced winds of a million m/s. A constant feature carries no
    information, so it is centred and left unscaled.
    """
    sd = X.std(0)
    return np.where(sd < 1e-4, 1.0, sd)


class _Sklearn(Forecaster):
    """Shared plumbing: standardise inputs on train, flatten targets."""

    def _xs(self, X):
        return (X - self.mu) / self.sd

    def _fit_scaler(self, X):
        self.mu, self.sd = X.mean(0), safe_std(X)


class Linear(_Sklearn):
    name = "linear"

    def fit(self, train, val, A, h_steps):
        from sklearn.linear_model import LinearRegression
        self._fit_scaler(train.X)
        self.shape = train.Y.shape[1:]
        self.m = LinearRegression().fit(self._xs(train.X), train.Y.reshape(len(train.Y), -1))
        return self

    def predict(self, s, A, h_steps):
        return self.m.predict(self._xs(s.X)).reshape(-1, *self.shape).astype("float32")


class Ridge(_Sklearn):
    """Ridge with the penalty chosen on validation loss."""
    name = "ridge"
    alphas = (0.1, 1.0, 10.0, 100.0, 1000.0)

    def fit(self, train, val, A, h_steps):
        from sklearn.linear_model import Ridge as SkRidge
        self._fit_scaler(train.X)
        self.shape = train.Y.shape[1:]
        Yt = train.Y.reshape(len(train.Y), -1)
        Xv = self._xs(val.X)
        best = None
        for a in self.alphas:
            m = SkRidge(alpha=a).fit(self._xs(train.X), Yt)
            err = float(np.mean((m.predict(Xv) - val.Y.reshape(len(val.Y), -1)) ** 2))
            if best is None or err < best[0]:
                best = (err, a, m)
        self.val_mse, self.alpha, self.m = best
        return self

    def predict(self, s, A, h_steps):
        return self.m.predict(self._xs(s.X)).reshape(-1, *self.shape).astype("float32")


class GradientBoosting(_Sklearn):
    """One histogram gradient-boosted tree model per output.

    Trees capture regime effects (monsoon, QBO phase) that a linear model
    cannot. They are trained on a random subsample to keep the cost bounded,
    with early stopping on an internal validation fraction.
    """
    name = "gradient_boosting"

    def __init__(self, max_rows: int = 150_000, max_iter: int = 300, seed: int = 0):
        self.max_rows, self.max_iter, self.seed = max_rows, max_iter, seed

    def fit(self, train, val, A, h_steps):
        from sklearn.ensemble import HistGradientBoostingRegressor
        rng = np.random.default_rng(self.seed)
        idx = rng.choice(len(train.X), min(self.max_rows, len(train.X)), replace=False)
        X = train.X[idx]
        Y = train.Y[idx].reshape(len(idx), -1)
        self.shape = train.Y.shape[1:]
        self.models = [HistGradientBoostingRegressor(
            max_iter=self.max_iter, learning_rate=0.1, max_leaf_nodes=31,
            early_stopping=True, validation_fraction=0.1, random_state=self.seed
        ).fit(X, Y[:, j]) for j in range(Y.shape[1])]
        return self

    def predict(self, s, A, h_steps):
        out = np.stack([m.predict(s.X) for m in self.models], axis=1)
        return out.reshape(-1, *self.shape).astype("float32")


def ladder() -> list[type[Forecaster]]:
    """Every rung, simplest first. Neural rungs are imported lazily (torch)."""
    from stratoballoon.forecasting.neural import LSTM, TCN
    return [Persistence, TidePersistence, Climatology, MovingAverage, Linear, Ridge,
            GradientBoosting, LSTM, TCN]
