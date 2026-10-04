"""Forecast uncertainty: calibrated intervals and sampled wind scenarios.

MODEL PREDICTION. The controller must not treat a forecast as exact. This
module turns validation-period forecast errors into

* **conformal prediction regions**: a radius around the forecast wind vector
  that contained the truth in a chosen fraction of validation cases, per
  horizon and level. Split conformal makes no assumption about the error
  distribution, but its guarantee is *marginal*: coverage on average over
  the conditions in the calibration data, not in every weather regime. With
  time-correlated samples the guarantee is approximate, which is why coverage
  is re-measured on a later year rather than assumed.
* **component intervals**: separate lower/upper bounds for u and v, from
  signed residual quantiles, so a biased model gets asymmetric intervals.
* **scenarios**: whole residual vectors (every horizon and level at once)
  drawn from the validation period and added to the forecast. Drawing them
  jointly keeps the real correlation of errors across lead time and altitude.
  Within a scenario the same error is applied across the domain, which
  assumes spatially coherent error over a few hundred km; that is stated as an
  assumption, not tested.
"""
from __future__ import annotations

import numpy as np


class VectorConformal:
    def fit(self, val_residual: np.ndarray) -> "VectorConformal":
        """val_residual: truth - forecast, shape (N, H, L, 2)."""
        self.resid = val_residual.astype("float32")
        self.norm = np.sort(np.linalg.norm(val_residual, axis=-1), axis=0)   # (N, H, L)
        self.n = len(val_residual)
        return self

    def radius(self, coverage: float) -> np.ndarray:
        """Radius (m/s) of the prediction disc, per (H, L)."""
        k = min(self.n - 1, int(np.ceil((self.n + 1) * coverage)) - 1)
        return self.norm[k]

    def component_bounds(self, coverage: float) -> tuple[np.ndarray, np.ndarray]:
        """(lo, hi) offsets for u and v, each (H, L, 2): forecast + lo .. forecast + hi."""
        a = (1 - coverage) / 2
        return (np.quantile(self.resid, a, axis=0), np.quantile(self.resid, 1 - a, axis=0))

    def evaluate(self, test_residual: np.ndarray,
                 levels=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95)) -> list[dict]:
        """Empirical coverage and size of the regions on later data."""
        norm = np.linalg.norm(test_residual, axis=-1)
        rows = []
        for c in levels:
            r = self.radius(c)
            lo, hi = self.component_bounds(c)
            inside_c = ((test_residual >= lo) & (test_residual <= hi)).all(-1)
            for k in range(r.shape[0]):
                for l in range(r.shape[1]):
                    rows.append({"nominal": c, "h_idx": k, "l_idx": l,
                                 "disc_coverage": float((norm[:, k, l] <= r[k, l]).mean()),
                                 "disc_radius_ms": float(r[k, l]),
                                 "box_coverage": float(inside_c[:, k, l].mean()),
                                 "box_width_u_ms": float(hi[k, l, 0] - lo[k, l, 0]),
                                 "box_width_v_ms": float(hi[k, l, 1] - lo[k, l, 1])})
        return rows

    def sample(self, k: int, rng: np.random.Generator) -> np.ndarray:
        """k residual scenarios, (k, H, L, 2), drawn jointly across H and L."""
        return self.resid[rng.integers(0, self.n, k)]


class RegimeConformal:
    """Conformal regions calibrated separately per wind regime (Mondrian conformal).

    Split conformal promises coverage on average, and the average hides
    regimes: errors are larger when the wind is strong. Here the calibration
    errors are grouped by the wind speed *at issue time* (known when the
    forecast is made, so this is not peeking), and each group gets its own
    radius. Coverage then holds within each regime, at the cost of fewer
    calibration samples per radius.
    """

    def fit(self, val_residual: np.ndarray, current_speed: np.ndarray,
            n_bins: int = 3) -> "RegimeConformal":
        """current_speed: (N, L) wind speed at issue time, per level."""
        self.edges = np.quantile(current_speed, np.linspace(0, 1, n_bins + 1)[1:-1], axis=0)
        bins = self._bin(current_speed)                                     # (N, L)
        self.parts = []
        for b in range(n_bins):
            r = np.where((bins == b)[:, None, :, None], val_residual, np.nan)
            self.parts.append(r)
        self.n_bins = n_bins
        return self

    def _bin(self, speed):
        return (speed[..., None] > self.edges.T[None]).sum(-1)              # (N, L)

    def radius(self, coverage: float, current_speed: np.ndarray) -> np.ndarray:
        """Radius per sample, (N, H, L)."""
        bins = self._bin(current_speed)
        out = np.empty((len(current_speed), self.parts[0].shape[1], current_speed.shape[1]))
        for b, r in enumerate(self.parts):
            norm = np.linalg.norm(r, axis=-1)                                # (Nv, H, L), NaN outside
            n = np.sum(~np.isnan(norm), axis=0)
            q = np.minimum(1.0, np.ceil((n + 1) * coverage) / np.maximum(n, 1))
            rad = np.stack([[np.nanquantile(norm[:, h, l], q[h, l])
                             for l in range(norm.shape[2])] for h in range(norm.shape[1])])
            for l in range(current_speed.shape[1]):
                m = bins[:, l] == b
                out[m, :, l] = rad[None, :, l]
        return out


def adaptive_coverage(test_norm: np.ndarray, t_idx: np.ndarray, cal_norm_sorted: np.ndarray,
                      target: float = 0.9, gamma: float = 0.005, lag: int = 1) -> np.ndarray:
    """Adaptive conformal inference (Gibbs and Candes, 2021), run through a test period.

    Split conformal assumes the test period behaves like the calibration
    period; when the next year is windier, coverage falls short and stays
    short. ACI adjusts the working miscoverage level after every verified
    forecast: alpha <- alpha + gamma * (target miss rate - observed miss rate).
    A forecast for lead h can only be verified h later, so updates use the
    outcome from `lag` issue times ago, never the current one.

    test_norm: (N,) error size of each test forecast; t_idx: (N,) its issue
    time; cal_norm_sorted: sorted calibration error sizes. Returns (N,) covered.
    """
    alpha_target = 1 - target
    alpha = alpha_target
    times = np.unique(t_idx)
    pos = np.searchsorted(times, t_idx)
    misses = np.zeros(len(times))
    covered = np.zeros(len(test_norm), bool)
    n = len(cal_norm_sorted)
    for i in range(len(times)):
        if i - lag >= 0:
            alpha = alpha + gamma * (alpha_target - misses[i - lag])
        q = np.clip(1 - alpha, 0, 1)
        radius = cal_norm_sorted[min(n - 1, max(0, int(np.ceil(q * (n + 1))) - 1))] \
            if q < 1 else np.inf
        m = pos == i
        covered[m] = test_norm[m] <= radius
        misses[i] = 1 - covered[m].mean()
    return covered


def crps_ensemble(members: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """CRPS for a scalar from an ensemble, (M, N) and (N,) -> (N,)."""
    m = members.shape[0]
    t1 = np.abs(members - truth[None]).mean(0)
    s = np.sort(members, axis=0)
    i = np.arange(1, m + 1)[:, None]
    t2 = ((2 * i - m - 1) * s).sum(0) / (m * m)
    return t1 - t2
