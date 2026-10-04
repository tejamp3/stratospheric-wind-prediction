"""Forecast fields the controller and trajectory predictor can query anywhere.

MODEL PREDICTION. A forecast is issued every cycle (like an NWP run) for every
grid cell, at lead times 0 (the analysis at issue time) and each model horizon.
Queries interpolate linearly in lead time, bilinearly in space and linearly in
height using the heights known at issue time. Beyond the last horizon the
forecast is held at its last value, and callers should treat that as stale.

`TruthProvider` answers the same queries from the real atmosphere at the
actual valid time; it is the "perfect forecast" upper bound and must never be
given to a controller except as that labelled reference.
"""
from __future__ import annotations

import numpy as np

from stratoballoon.atmosphere import Atmosphere
from stratoballoon.forecasting import features as F


MAX_PLAUSIBLE_WIND = 150.0   # m/s


class FieldForecast:
    def __init__(self, A: Atmosphere, model, issue_idx: np.ndarray, n_hist: int,
                 horizons_h: list[int], name: str, batch: int = 64):
        self.A, self.name = A, name
        self.issue_idx = np.asarray(issue_idx)
        self.lead_h = np.array([0.0] + list(horizons_h))
        step = A.step_hours
        h_steps = [int(h // step) for h in horizons_h]
        yi, xi = F.cell_grid(A, 1)
        ny, nx, L = len(A.lats), len(A.lons), len(A.levels)
        # float16 storage: ~0.01 m/s resolution, half the memory of a year of
        # wide-domain fields; values are cast back to float32 when queried.
        field = np.empty((len(issue_idx), len(self.lead_h), L, ny, nx, 2), "float16")
        for b in range(0, len(issue_idx), batch):
            ti = self.issue_idx[b:b + batch]
            s = _samples_without_labels(A, ti, yi, xi, n_hist, h_steps)
            pred = model.predict(s, A, h_steps) + s.current[:, None]       # (n*C, H, L, 2)
            full = np.concatenate([s.current[:, None], pred], axis=1)      # lead 0 first
            # Stratospheric winds stay well under 150 m/s. A forecast beyond
            # that is a broken model, and it must stop here, not reach a balloon.
            worst = float(np.nanmax(np.abs(full))) if np.isfinite(full).all() else float("inf")
            if worst > MAX_PLAUSIBLE_WIND:
                when = str(A.times[ti[int(np.argmax(np.abs(np.nan_to_num(
                    full, nan=np.inf, posinf=np.inf, neginf=np.inf)).reshape(len(ti), -1).max(1)))]])
                raise ValueError(f"{name} forecast issued {when[:13]} reaches {worst:.3g} m/s; "
                                 "refusing to build a forecast field from it")
            field[b:b + batch] = full.reshape(len(ti), ny, nx, len(self.lead_h), L, 2
                                              ).transpose(0, 3, 4, 1, 2, 5)
        self.field = field
        self.height = A.h[self.issue_idx]                                  # (I, L, Y, X)
        self.issue_hours = A.hours[self.issue_idx]

    def latest_issue(self, hours_now: float | np.ndarray) -> np.ndarray:
        """Index of the most recent forecast issued at or before `hours_now`."""
        return np.clip(np.searchsorted(self.issue_hours, hours_now, side="right") - 1,
                       0, len(self.issue_hours) - 1)

    def wind(self, issue_k, valid_hours, lat, lon, alt, offset=None):
        """u, v forecast for each query. `offset` (N, n_lead, L, 2) adds a scenario."""
        A = self.A
        issue_k, valid_hours, lat, lon, alt = (np.atleast_1d(np.asarray(a, float))
                                               for a in (issue_k, valid_hours, lat, lon, alt))
        issue_k = issue_k.astype(int)
        lead = np.clip(valid_hours - self.issue_hours[issue_k], 0, self.lead_h[-1])
        j = np.clip(np.searchsorted(self.lead_h, lead, side="right") - 1, 0, len(self.lead_h) - 2)
        wl = ((lead - self.lead_h[j]) / (self.lead_h[j + 1] - self.lead_h[j]))[:, None, None]
        yi = np.clip(np.interp(lat, A.lats, np.arange(len(A.lats))), 0, len(A.lats) - 1)
        xi = np.clip(np.interp(lon, A.lons, np.arange(len(A.lons))), 0, len(A.lons) - 1)
        y0 = np.minimum(yi.astype(int), len(A.lats) - 2)
        x0 = np.minimum(xi.astype(int), len(A.lons) - 2)
        wy, wx = (yi - y0)[:, None], (xi - x0)[:, None]
        f, hgt = self.field, self.height
        prof = 0.0
        hp = 0.0
        for dy, wy_ in ((0, 1 - wy), (1, wy)):
            for dx, wx_ in ((0, 1 - wx), (1, wx)):
                w = (wy_ * wx_)
                a = f[issue_k, j, :, y0 + dy, x0 + dx].astype("float32")   # (N, L, 2)
                b = f[issue_k, j + 1, :, y0 + dy, x0 + dx].astype("float32")
                prof = prof + w[..., None] * (a * (1 - wl) + b * wl)
                hp = hp + w * hgt[issue_k, :, y0 + dy, x0 + dx]
        if offset is not None:
            oa = offset[np.arange(len(j)), j]
            ob = offset[np.arange(len(j)), j + 1]
            prof = prof + oa * (1 - wl) + ob * wl
        return _vertical(prof, hp, alt)


class TruthProvider:
    """The real atmosphere at the valid time: the perfect-forecast reference."""
    name = "perfect"

    def __init__(self, A: Atmosphere):
        self.A = A
        self.issue_hours = A.hours

    def latest_issue(self, hours_now):
        # The truth is always current: report the present time step so that
        # forecast-age bookkeeping reads zero.
        return np.clip(np.searchsorted(self.issue_hours, np.atleast_1d(hours_now), side="right") - 1,
                       0, len(self.issue_hours) - 1)

    def wind(self, issue_k, valid_hours, lat, lon, alt, offset=None):
        s = self.A.sample(valid_hours, lat, lon, alt)
        return s["u"], s["v"]


def _vertical(prof: np.ndarray, h: np.ndarray, alt: np.ndarray):
    """Linear interpolation in height of (N, L, 2) profiles at heights (N, L)."""
    L = h.shape[1]
    k = np.clip((h < alt[:, None]).sum(1) - 1, 0, L - 2)
    r = np.arange(len(alt))
    w = np.clip((alt - h[r, k]) / (h[r, k + 1] - h[r, k]), 0, 1)[:, None]
    uv = prof[r, k] * (1 - w) + prof[r, k + 1] * w
    return uv[:, 0], uv[:, 1]


def _samples_without_labels(A, t_idx, yi, xi, n_hist, h_steps) -> F.Samples:
    """Inputs for issue times whose labels may lie beyond the record.

    build_samples needs labels; for forecasting at the end of the record they do
    not exist, so labels are filled from the last available time and never read.
    """
    T = len(A.times)
    safe = np.minimum(t_idx, T - 1 - max(h_steps))
    s = F.build_samples(A, safe, yi, xi, n_hist, h_steps)
    if np.array_equal(safe, t_idx):
        return s
    return F.build_samples(_Shifted(A, max(h_steps)), t_idx, yi, xi, n_hist, h_steps)


class _Shifted:
    """Pads the time axis with copies of the last step so indexing cannot overflow."""

    def __init__(self, A, pad):
        self.u = np.concatenate([A.u, np.repeat(A.u[-1:], pad, 0)])
        self.v = np.concatenate([A.v, np.repeat(A.v[-1:], pad, 0)])
        self.t = np.concatenate([A.t, np.repeat(A.t[-1:], pad, 0)])
        step = A.times[-1] - A.times[-2]
        self.times = np.concatenate([A.times, A.times[-1] + step * np.arange(1, pad + 1)])
        self.levels, self.lats, self.lons = A.levels, A.lats, A.lons
