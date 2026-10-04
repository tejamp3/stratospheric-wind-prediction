"""Altitude controllers for station keeping.

CONTROL. A balloon cannot fly towards its target; it can only choose an
altitude and let the wind at that altitude carry it. Every controller here
outputs a target altitude, re-decided every few hours, and the ballonet loop in
`dynamics` tries to reach it.

Controllers, simplest first (each must beat the one above it to be kept):

* `Hold`: never changes altitude. The do-nothing baseline.
* `Greedy`: every decision, picks the altitude whose forecast wind brings the
  balloon closest to the target over the next few hours, with hysteresis.
* `MPC`: model predictive control. Searches altitude plans over the next day
  (first move to any grid altitude, then one notch up, down or hold per step),
  scores each by forecast distance to the target plus energy and switching
  costs, executes only the first move, and re-plans at the next decision with
  fresh observations. With `scenarios > 0` it re-scores the best plans of that
  search against several sampled wind futures with a tail-risk term
  (uncertainty-aware MPC); with 0 it trusts the point forecast.

The planning model is deliberately simpler than the simulator: altitude moves
towards the commanded value at fixed climb/descent rates, and horizontal motion
follows the forecast wind. That mismatch is realistic; a controller never has
the true dynamics.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from stratoballoon.atmosphere import G0, R_D, distance_km, move


@dataclass
class Band:
    """Altitude band the controller may command, and its discrete grid."""
    alt_min: float
    alt_max: float
    n_grid: int = 5

    @property
    def grid(self) -> np.ndarray:
        return np.linspace(self.alt_min, self.alt_max, self.n_grid)

    def nearest(self, alt) -> np.ndarray:
        return np.clip(np.rint((alt - self.alt_min) / (self.alt_max - self.alt_min)
                               * (self.n_grid - 1)), 0, self.n_grid - 1).astype(int)


@dataclass
class Rates:
    """Planning-model climb and descent rates (m/s) and descent energy (Wh/m)."""
    up: float
    down: float
    wh_per_m_down: float

    @classmethod
    def from_params(cls, p, T: float = 220.0, rho: float = 0.05) -> "Rates":
        h_rho = R_D * T / G0
        m = p.structure_kg + p.gas_kg + 0.5 * p.ballonet_max_kg
        kg_per_m = m / h_rho
        wh_per_kg = p.superpressure_pa / rho / p.pump_efficiency / 3600
        return cls(up=h_rho * p.vent_kg_s / m, down=h_rho * p.pump_kg_s / m,
                   wh_per_m_down=kg_per_m * wh_per_kg)


@dataclass
class Observation:
    hours: np.ndarray          # per mission, hours since atmosphere start
    lat: np.ndarray
    lon: np.ndarray
    alt: np.ndarray
    battery_wh: np.ndarray


class Controller:
    name = "base"

    def reset(self, n: int, alt0: np.ndarray):
        self.target = alt0.copy()

    def decide(self, obs: Observation, mission, provider) -> np.ndarray:
        raise NotImplementedError


class Hold(Controller):
    name = "hold"

    def decide(self, obs, mission, provider):
        return self.target


def _kinematic_alt(alt0, targets, step_h, sub_h, rates):
    """Altitude at each sub-step for a sequence of targets. (M,), (M, S) -> (M, S*n)."""
    n_sub = int(round(step_h / sub_h))
    alt = alt0.copy()
    out = []
    for s in range(targets.shape[1]):
        for _ in range(n_sub):
            d = targets[:, s] - alt
            lim = np.where(d > 0, rates.up, rates.down) * sub_h * 3600
            alt = alt + np.clip(d, -lim, lim)
            out.append(alt.copy())
    return np.stack(out, axis=1)


def rollout(provider, issue_k, t_hours, lat, lon, alt_path, sub_h, offset=None):
    """Integrate positions along altitude paths with the forecast wind.

    All inputs are flat over (mission x plan x scenario). Returns distance-ready
    (lat, lon) tracks, each (M, n_steps).
    """
    lats, lons = [], []
    la, lo = lat.copy(), lon.copy()
    for j in range(alt_path.shape[1]):
        u, v = provider.wind(issue_k, t_hours + j * sub_h, la, lo, alt_path[:, j], offset)
        la, lo = move(la, lo, u, v, sub_h * 3600)
        lats.append(la)
        lons.append(lo)
    return np.stack(lats, 1), np.stack(lons, 1)


class MPC(Controller):
    """Receding-horizon search over up / hold / down sequences."""

    def __init__(self, band: Band, rates: Rates, step_h: float = 6.0, n_steps: int = 4,
                 sub_h: float = 1.0, energy_km_per_wh: float = 0.05,
                 switch_km: float = 5.0, scenarios: int = 0, sampler=None,
                 risk_weight: float = 0.5, risk_quantile: float = 0.8, seed: int = 0,
                 shortlist: int = 15, name: str | None = None):
        self.band, self.rates = band, rates
        self.step_h, self.n_steps, self.sub_h = step_h, n_steps, sub_h
        self.energy_km_per_wh, self.switch_km = energy_km_per_wh, switch_km
        self.scenarios, self.sampler = scenarios, sampler
        self.risk_weight, self.risk_quantile = risk_weight, risk_quantile
        self.shortlist = shortlist
        self.rng = np.random.default_rng(seed)
        # Plans: the first step may go to any grid altitude (as greedy can);
        # later steps move one notch up, down or hold. 5 x 3^3 = 135 plans.
        self.first = np.arange(band.n_grid)
        rest = list(itertools.product((-1, 0, 1), repeat=n_steps - 1))
        moves = np.array([(f, *r) for f in self.first for r in rest])         # (P, S)
        # grid index at each plan step: absolute first step, then cumulative moves
        self.plans = np.clip(np.cumsum(moves, 1), 0, band.n_grid - 1)
        self.name = name or ("mpc_robust" if scenarios else "mpc")

    def _cost(self, obs, mission, provider, idx: np.ndarray, K: int) -> np.ndarray:
        """Cost of each plan (n, P) given per-mission plan indices idx (n, P, S)."""
        n, P = idx.shape[:2]
        g0 = self.band.nearest(obs.alt)
        targets = self.band.grid[idx]
        flat_t = np.repeat(targets.reshape(n * P, -1), K, axis=0)              # (n*P*K, S)
        rep = P * K
        alt0 = np.repeat(obs.alt, rep)
        path = _kinematic_alt(alt0, flat_t, self.step_h, self.sub_h, self.rates)
        issue = np.repeat(provider.latest_issue(obs.hours), rep)
        offset = None
        if K > 1:
            draws = self.sampler(n * K, self.rng)                             # (n*K, H, L, 2)
            draws = draws.reshape(n, 1, K, *draws.shape[1:])
            offset = np.broadcast_to(draws, (n, P, K, *draws.shape[3:])).reshape(
                n * P * K, *draws.shape[3:])
        la, lo = rollout(provider, issue, np.repeat(obs.hours, rep), np.repeat(obs.lat, rep),
                         np.repeat(obs.lon, rep), path, self.sub_h, offset)
        d = distance_km(la, lo, np.repeat(mission.target_lat, rep)[:, None],
                        np.repeat(mission.target_lon, rep)[:, None]).mean(1).reshape(n, P, K)
        descent = np.clip(-np.diff(np.concatenate([alt0[:, None], path], 1), axis=1), 0,
                          None).sum(1).reshape(n, P, K)[..., 0]
        switches = (np.diff(np.concatenate([g0[:, None, None].repeat(P, 1), idx], 2), axis=2)
                    != 0).sum(2)
        if K > 1:
            tail = np.quantile(d, self.risk_quantile, axis=2)
            dist_cost = (1 - self.risk_weight) * d.mean(2) + self.risk_weight * tail
        else:
            dist_cost = d[..., 0]
        cost = (dist_cost + self.energy_km_per_wh * descent * self.rates.wh_per_m_down
                + self.switch_km * switches)
        # with an empty battery the pump cannot run: forbid any first move down
        down_first = self.band.grid[idx[:, :, 0]] < obs.alt[:, None] - 1
        return np.where((obs.battery_wh[:, None] <= 0) & down_first, np.inf, cost)

    def decide(self, obs, mission, provider):
        n = len(obs.lat)
        idx = np.broadcast_to(self.plans[None], (n, *self.plans.shape))           # (n, P, S)
        cost = self._cost(obs, mission, provider, idx, 1)
        if self.scenarios:
            # Two stages: rank every plan on the point forecast, then re-rank
            # the best few against the sampled scenarios. Scoring all plans
            # under every scenario costs about 4x more for the same choice set.
            top = np.argsort(cost, axis=1)[:, :self.shortlist]
            idx = np.take_along_axis(idx, top[..., None], axis=1)
            cost = self._cost(obs, mission, provider, idx, self.scenarios)
        best = np.argmin(cost, axis=1)
        self.target = self.band.grid[idx[np.arange(n), best, 0]]
        return self.target


class Greedy(MPC):
    """One step of look-ahead, choosing any grid altitude, with hysteresis."""

    def __init__(self, band, rates, step_h: float = 6.0, hysteresis_km: float = 10.0, **kw):
        super().__init__(band, rates, step_h=step_h, n_steps=1, name="greedy", **kw)
        self.hysteresis_km = hysteresis_km

    def decide(self, obs, mission, provider):
        n, G = len(obs.lat), self.band.n_grid
        targets = np.tile(self.band.grid, (n, 1))                              # (n, G)
        alt0 = np.repeat(obs.alt, G)
        path = _kinematic_alt(alt0, targets.reshape(-1, 1), self.step_h, self.sub_h,
                              self.rates)
        issue = np.repeat(provider.latest_issue(obs.hours), G)
        la, lo = rollout(provider, issue, np.repeat(obs.hours, G), np.repeat(obs.lat, G),
                         np.repeat(obs.lon, G), path, self.sub_h)
        d = distance_km(la, lo, np.repeat(mission.target_lat, G)[:, None],
                        np.repeat(mission.target_lon, G)[:, None]).mean(1).reshape(n, G)
        cur = self.band.nearest(self.target)
        best = np.argmin(d, axis=1)
        keep = d[np.arange(n), cur] - d[np.arange(n), best] < self.hysteresis_km
        choice = np.where(keep, cur, best)
        self.target = self.band.grid[choice]
        return self.target
