"""Fault injection and what the controller is allowed to see.

SIMULATION. A `Faults` object describes, per mission, what goes wrong. The
`ControllerView` sits between the forecast and the controller and is the only
way the controller reaches forecast data, so faults that should hide or corrupt
information do it in one place:

* comms loss: no new forecast reaches the balloon, so it keeps using the last
  one it received, which grows stale;
* forecast degradation: the forecast's error is multiplied (scale > 1 means a
  worse forecast than the one the model actually produced);
* bias correction (not a fault, a feature): the Kalman filter's running
  estimate of the forecast error is added near-term and decays with lead time.

Truth-side faults (GPS outage, barometer bias, pump failure, battery fade, an
unforecast wind change, a wrong initial position) are applied by `fly`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Faults:
    """Per-mission fault windows; times are hours after mission start."""
    n: int
    gps_out: np.ndarray = None          # (n, 2) start, end; NaN = none
    comms_out: np.ndarray = None        # (n, 2)
    pump_fail_h: np.ndarray = None      # (n,) time pump dies; inf = never
    battery_factor: np.ndarray = None   # (n,) capacity multiplier
    baro_bias_m: np.ndarray = None      # (n,)
    forecast_error_scale: np.ndarray = None   # (n,)
    gust: np.ndarray = None             # (n, 4) start, end, du, dv (m/s) unforecast
    init_pos_err_km: np.ndarray = None  # (n,) controller's initial position error
    label: str = "nominal"

    def __post_init__(self):
        n = self.n
        nan2 = np.full((n, 2), np.nan)
        self.gps_out = nan2.copy() if self.gps_out is None else self.gps_out
        self.comms_out = nan2.copy() if self.comms_out is None else self.comms_out
        self.pump_fail_h = np.full(n, np.inf) if self.pump_fail_h is None else self.pump_fail_h
        self.battery_factor = np.ones(n) if self.battery_factor is None else self.battery_factor
        self.baro_bias_m = np.zeros(n) if self.baro_bias_m is None else self.baro_bias_m
        self.forecast_error_scale = (np.ones(n) if self.forecast_error_scale is None
                                     else self.forecast_error_scale)
        self.gust = np.full((n, 4), np.nan) if self.gust is None else self.gust
        self.init_pos_err_km = (np.zeros(n) if self.init_pos_err_km is None
                                else self.init_pos_err_km)

    @staticmethod
    def _in(win, t):
        return (t >= win[:, 0]) & (t < win[:, 1])

    def gps_ok(self, t):
        return ~self._in(self.gps_out, t)

    def comms_ok(self, t):
        return ~self._in(self.comms_out, t)

    def gust_uv(self, t):
        on = self._in(self.gust[:, :2], t)
        return np.where(on, self.gust[:, 2], 0.0), np.where(on, self.gust[:, 3], 0.0)


def fault_suite(n: int, duration_h: float, seed: int) -> list[Faults]:
    """The standard robustness suite: one nominal set and one per fault type.

    Fault timing and size are random per mission but seeded, so every
    controller meets exactly the same faults.
    """
    rng = np.random.default_rng(seed)

    def window(lo_len, hi_len):
        length = rng.uniform(lo_len, hi_len, n)
        start = rng.uniform(0, max(1.0, duration_h - length.max()), n)
        return np.stack([start, start + length], 1)
    return [
        Faults(n, label="nominal"),
        Faults(n, gps_out=window(6, 24), label="gps outage 6-24 h"),
        Faults(n, comms_out=window(12, 48), label="comms loss 12-48 h"),
        Faults(n, forecast_error_scale=np.full(n, 2.0), label="forecast error x2"),
        Faults(n, gust=np.concatenate([window(6, 12), rng.normal(0, 8, (n, 2))], 1),
               label="unforecast wind change"),
        Faults(n, pump_fail_h=rng.uniform(0, duration_h / 2, n), label="pump failure"),
        Faults(n, battery_factor=np.full(n, 0.2), label="battery at 20%"),
        Faults(n, baro_bias_m=rng.choice([-1, 1], n) * 300.0, label="barometer bias 300 m"),
        # A wrong initial state only matters until the first GPS fix, so it is
        # paired with a GPS cold start: no fix for the first 6 hours.
        Faults(n, init_pos_err_km=np.full(n, 50.0), gps_out=np.tile([0.0, 6.0], (n, 1)),
               label="initial position 50 km off, no GPS for 6 h"),
    ]


class ControllerView:
    """The forecast as the controller experiences it."""

    def __init__(self, provider, truth_atm, faults: Faults, start_hours: np.ndarray,
                 bias_tau_h: float = 12.0):
        self.p, self.A, self.f, self.t0 = provider, truth_atm, faults, start_hours
        self.issue_hours = provider.issue_hours
        self.last_issue = None
        self.bias = None              # (n, 2) from the filter, if used
        self.bias_tau_h = bias_tau_h
        self.now = None

    def tick(self, now: np.ndarray):
        """Receive whatever forecast the link allows at time `now`."""
        newest = self.p.latest_issue(now)
        ok = self.f.comms_ok(now - self.t0)
        if self.last_issue is None:
            self.last_issue = newest.copy()
        self.last_issue = np.where(ok, newest, self.last_issue)
        self.now = now

    def latest_issue(self, hours):
        return self.last_issue if np.ndim(hours) else self.last_issue[:1]

    def forecast_age_h(self):
        return self.now - self.issue_hours[self.last_issue]

    def wind(self, issue_k, valid_hours, lat, lon, alt, offset=None):
        u, v = self.p.wind(issue_k, valid_hours, lat, lon, alt, offset)
        n = len(self.f.forecast_error_scale)
        rep = max(1, len(u) // n)
        mission = np.arange(len(u)) // rep
        scale = self.f.forecast_error_scale[mission]
        if np.any(scale != 1):
            s = self.A.sample(valid_hours, lat, lon, alt)
            u = s["u"] + scale * (u - s["u"])
            v = s["v"] + scale * (v - s["v"])
        if self.bias is not None:
            lead = np.clip(valid_hours - self.now[mission], 0, None)
            decay = np.exp(-lead / self.bias_tau_h)
            u = u + self.bias[mission, 0] * decay
            v = v + self.bias[mission, 1] * decay
        return u, v
