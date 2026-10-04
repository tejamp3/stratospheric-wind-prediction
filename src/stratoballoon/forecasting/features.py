"""Training samples for wind forecasting, pooled over many grid columns.

Learning from one column gives only a few thousand training windows. Pooling columns
multiplies the data by the number of cells and lets one model forecast any point
in the domain, which is what trajectory prediction needs.

Leakage rules, enforced here rather than trusted:

* a sample is (issue time, column); its inputs use only times <= issue time;
* every label lies inside the same split as its issue time (`issue_indices`
  stops max(horizon) short of the split end);
* splits are separated by an embargo so the end of one split's labels cannot
  sit next to the start of the next split's inputs.

Targets are the change from persistence: label = wind(t + h) - wind(t).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from stratoballoon.atmosphere import Atmosphere


@dataclass
class Split:
    name: str
    start: pd.Timestamp
    end: pd.Timestamp       # inclusive


def rolling_splits(test_year: int, first_year: int, embargo_days: int = 3) -> list[Split]:
    """Train on every year before the validation year, validate on the year
    before the test year, test on the test year, with an embargo at each seam."""
    e = pd.Timedelta(days=embargo_days)
    val0, test0 = pd.Timestamp(test_year - 1, 1, 1), pd.Timestamp(test_year, 1, 1)
    return [Split("train", pd.Timestamp(first_year, 1, 1), val0 - e),
            Split("val", val0 + e, test0 - e),
            Split("test", test0 + e, pd.Timestamp(test_year, 12, 31, 23))]


def issue_indices(A: Atmosphere, split: Split, n_hist: int, max_h_steps: int,
                  stride: int = 1) -> np.ndarray:
    """Time indices that can issue a forecast with all inputs and labels in the split."""
    t = pd.DatetimeIndex(A.times)
    inside = np.where((t >= split.start) & (t <= split.end))[0]
    if len(inside) == 0:
        return inside
    lo, hi = inside[0] + n_hist - 1, inside[-1] - max_h_steps
    return np.arange(lo, hi + 1, stride)


@dataclass
class Samples:
    X: np.ndarray        # (N, F) float32 inputs
    Y: np.ndarray        # (N, H, L, 2) change from persistence, m/s
    current: np.ndarray  # (N, L, 2) wind at issue time (the persistence forecast)
    t_idx: np.ndarray    # (N,) issue time index into the Atmosphere
    cell: np.ndarray     # (N,) flat cell index y * nx + x
    feature_names: list[str]

    @property
    def truth(self) -> np.ndarray:
        """Absolute wind at each horizon, (N, H, L, 2)."""
        return self.Y + self.current[:, None]

    def subset(self, mask) -> "Samples":
        return Samples(self.X[mask], self.Y[mask], self.current[mask], self.t_idx[mask],
                       self.cell[mask], self.feature_names)


def cell_grid(A: Atmosphere, stride: int = 1, margin: int = 0):
    """Flat indices of the grid cells used for training (every `stride`-th)."""
    ys = np.arange(margin, len(A.lats) - margin, stride)
    xs = np.arange(margin, len(A.lons) - margin, stride)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    return yy.ravel(), xx.ravel()


def build_samples(A: Atmosphere, t_idx: np.ndarray, yi: np.ndarray, xi: np.ndarray,
                  n_hist: int, h_steps: list[int]) -> Samples:
    """Sample every (issue time, cell) pair. Time-major ordering."""
    nx = len(A.lons)
    u = A.u[:, :, yi, xi]                    # (T, L, C)
    v = A.v[:, :, yi, xi]
    tt = A.t[:, :, yi, xi]
    L, C = u.shape[1], u.shape[2]
    feats, names = [], []
    for k in range(n_hist - 1, -1, -1):      # oldest first
        ti = t_idx - k
        for arr, nm in ((u, "u"), (v, "v"), (tt, "t")):
            for lev in range(L):
                feats.append(arr[ti, lev, :])                    # (n_t, C)
                names.append(f"{nm}{int(A.levels[lev])}_lag{k}")
    when = pd.DatetimeIndex(A.times[t_idx])
    hour = 2 * np.pi * when.hour.to_numpy() / 24
    doy = 2 * np.pi * when.dayofyear.to_numpy() / 365.25
    for vals, nm in ((np.sin(hour), "hour_sin"), (np.cos(hour), "hour_cos"),
                     (np.sin(doy), "doy_sin"), (np.cos(doy), "doy_cos")):
        feats.append(np.repeat(vals[:, None], C, axis=1))
        names.append(nm)
    feats.append(np.repeat(A.lats[yi][None, :], len(t_idx), axis=0))
    names.append("lat")
    feats.append(np.repeat(A.lons[xi][None, :], len(t_idx), axis=0))
    names.append("lon")
    X = np.stack([f.reshape(-1) for f in feats], axis=1).astype("float32")

    cur = np.stack([u[t_idx], v[t_idx]], axis=-1)                    # (n_t, L, C, 2)
    fut = np.stack([np.stack([u[t_idx + h], v[t_idx + h]], axis=-1) for h in h_steps],
                   axis=1)                                            # (n_t, H, L, C, 2)
    Y = fut - cur[:, None]
    Y = np.moveaxis(Y, 3, 1).reshape(-1, len(h_steps), L, 2)          # time-major, then cell
    cur = np.moveaxis(cur, 2, 1).reshape(-1, L, 2)
    t_rep = np.repeat(t_idx, C)
    cells = np.tile(yi * nx + xi, len(t_idx))
    return Samples(X, Y.astype("float32"), cur.astype("float32"), t_rep, cells, names)
