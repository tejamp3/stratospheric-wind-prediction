"""Mission planning: before launch, what altitude strategy, and how likely is success?

MODEL PREDICTION + SIMULATION. Given a launch point and time, a target, a
duration, the commandable altitude band, the energy budget and an acceptable
risk, the planner flies the uncertainty-aware MPC closed-loop through many
sampled wind futures (forecast plus validation-period errors) and reports:

* the recommended altitude strategy (the plan the controller follows in the
  median scenario);
* the predicted trajectory as a fan of scenarios;
* the probability of success, defined as ending within the target radius, and
  the fraction of time inside it;
* the expected arrival time for transit missions (first entry into the
  radius);
* risk: the probability of leaving the data domain, of draining the battery,
  and the 90th-percentile worst distance.

The "futures" are the forecast with sampled errors, not the real atmosphere,
so the success probability is only as good as the uncertainty calibration;
`experiments/uncertainty_eval.py` measures that calibration.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from stratoballoon.atmosphere import distance_km
from stratoballoon.control import MPC
from stratoballoon.simulation import Missions, fly


@dataclass
class MissionRequest:
    launch_time: str
    launch_lat: float
    launch_lon: float
    target_lat: float
    target_lon: float
    duration_h: float = 72.0
    radius_km: float = 50.0
    alt0: float | None = None
    max_risk: float = 0.2            # acceptable probability of failure


class ScenarioAtmosphere:
    """A forecast-plus-error 'atmosphere' the planner can fly balloons through.

    Each mission row is one scenario; its wind is the forecast issued at
    launch plus that scenario's sampled error. Temperature and density come
    from the forecast's issue-time analysis via the real atmosphere object, which
    affects only the buoyancy bookkeeping.
    """

    def __init__(self, A, provider, issue_k: int, offsets: np.ndarray):
        self.A, self.p, self.k, self.off = A, provider, issue_k, offsets
        self.times, self.lats, self.lons = A.times, A.lats, A.lons

    def sample(self, hours, lat, lon, alt):
        n = len(np.atleast_1d(lat))
        s = self.A.sample(np.full(n, self.p.issue_hours[self.k]), lat, lon, alt)
        u, v = self.p.wind(np.full(n, self.k), np.atleast_1d(hours) * np.ones(n), lat, lon,
                           alt, self.off[:n])
        s["u"], s["v"] = u, v
        return s

    def contains(self, lat, lon):
        return self.A.contains(lat, lon)

    @property
    def hours(self):
        return self.A.hours


def plan(ctx, req: MissionRequest, provider, sampler, n_scenarios: int = 64,
         seed: int = 0) -> dict:
    A = ctx.A
    t0 = A.hours[A.time_index(req.launch_time)]
    k = int(provider.latest_issue(np.array([t0]))[0])
    rng = np.random.default_rng(seed)
    offs = sampler(n_scenarios, rng)
    alt0 = req.alt0 if req.alt0 is not None else float(np.mean(ctx.band.grid))
    m = Missions(np.full(n_scenarios, t0), np.full(n_scenarios, req.target_lat),
                 np.full(n_scenarios, req.target_lon), np.full(n_scenarios, alt0),
                 req.duration_h, seed, launch_lat=np.full(n_scenarios, req.launch_lat),
                 launch_lon=np.full(n_scenarios, req.launch_lon))
    world = ScenarioAtmosphere(A, provider, k, offs)
    ctrl = MPC(ctx.band, ctx.rates, scenarios=8, sampler=sampler, seed=seed)
    tr = fly(world, m, ctrl, provider, ctx.params, seed=seed)
    d = tr["dist_km"]
    inside = d <= req.radius_km
    first = np.where(inside.any(1), inside.argmax(1) * 0.5, np.nan)
    p_success = float((d[:, -1] <= req.radius_km).mean())
    med = int(np.argsort(d.mean(1))[n_scenarios // 2])
    return {
        "p_success_end_within_radius": p_success,
        "mean_time_within_radius": float(inside.mean()),
        "expected_arrival_h": float(np.nanmedian(first)) if np.isfinite(first).any() else None,
        "p_arrive": float(np.isfinite(first).mean()),
        "p_leave_domain": float((~tr["in_domain"].all(1)).mean()),
        "p_battery_empty": float((tr["battery_wh"].min(1) <= 0).mean()),
        "worst_distance_p90_km": float(np.percentile(d.max(1), 90)),
        "recommended_altitude_m": tr["target_alt"][med],
        "go": bool(1 - p_success <= req.max_risk),
        "tracks": (tr["lat"], tr["lon"]),
        "altitude": tr["alt"],
        "median_scenario": med,
    }
