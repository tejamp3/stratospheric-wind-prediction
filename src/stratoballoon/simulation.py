"""Closed-loop mission simulation: atmosphere -> balloon -> sensors -> controller.

SIMULATION. The truth the balloon flies through is ERA5 (real data); the
controller only ever sees noisy observations and a forecast. Missions are
vectorised: N independent balloons step together, each with its own start time,
station and initial state, so every controller can be flown on the identical set
of scenarios.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from stratoballoon.atmosphere import EARTH_R, Atmosphere, distance_km
from stratoballoon.control import Controller, Observation
from stratoballoon.estimation import WindBiasKF
from stratoballoon.scenarios import ControllerView, Faults
from stratoballoon.dynamics import (BalloonParams, BalloonState, ballonet_for_altitude,
                                    solar_elevation_sin, step)


@dataclass
class Missions:
    start_hours: np.ndarray      # per mission, hours since atmosphere start
    target_lat: np.ndarray
    target_lon: np.ndarray
    alt0: np.ndarray
    duration_h: float
    seed: int = 0
    # Where each balloon starts; defaults to the target (station keeping).
    launch_lat: np.ndarray | None = None
    launch_lon: np.ndarray | None = None

    def __len__(self):
        return len(self.target_lat)


@dataclass
class SensorNoise:
    """Standard deviations of what the controller observes. GPS noise in metres."""
    gps_m: float = 10.0
    baro_alt_m: float = 30.0


def sample_missions(A: Atmosphere, n: int, start: str, end: str, duration_h: float,
                    alt_min: float, alt_max: float, seed: int,
                    lat_range=(10.0, 30.0), lon_range=(66.0, 94.0),
                    issue_every_h: int = 6) -> Missions:
    """Draw `n` missions with independent start times, stations and altitudes.

    Stations are drawn from the domain interior so a balloon does not start at
    the data edge. Start times fall on forecast cycles.
    """
    rng = np.random.default_rng(seed)
    t = pd.DatetimeIndex(A.times)
    ok = np.where((t >= pd.Timestamp(start)) & (t <= pd.Timestamp(end) - pd.Timedelta(hours=duration_h))
                  & (t.hour % issue_every_h == 0))[0]
    starts = np.sort(rng.choice(ok, n, replace=len(ok) < n))
    lat = rng.uniform(*lat_range, n)
    lon = rng.uniform(*lon_range, n)
    alt = rng.uniform(alt_min, alt_max, n)
    return Missions(A.hours[starts], lat, lon, alt, duration_h, seed)


def fly(A: Atmosphere, mission: Missions, controller: Controller, provider,
        params: BalloonParams, dt_s: float = 120.0, decide_every_h: float = 3.0,
        noise: SensorNoise = SensorNoise(), record_every_h: float = 0.5,
        seed: int = 0, faults: Faults | None = None, estimator: bool = False,
        bias_correction: bool = False, gps_every_s: float = 600.0) -> dict[str, np.ndarray]:
    """Fly one batch of missions closed-loop and return time series.

    Without `estimator` the controller sees the latest GPS fix with noise (and,
    during a GPS outage, the last fix it had). With it, a Kalman filter fuses
    GPS and barometer and dead-reckons through outages; `bias_correction` then
    also feeds the filter's forecast-error estimate into the forecast.
    """
    rng = np.random.default_rng(seed)
    n = len(mission)
    f = faults or Faults(n)
    t0 = mission.start_hours.astype(float)
    lat0 = mission.target_lat if mission.launch_lat is None else mission.launch_lat
    lon0 = mission.target_lon if mission.launch_lon is None else mission.launch_lon
    s = BalloonState(lat=np.array(lat0, float), lon=np.array(lon0, float),
                     alt=mission.alt0.copy(), w=np.zeros(n), ballonet_kg=np.zeros(n),
                     battery_wh=params.battery_wh * f.battery_factor,
                     energy_pump_wh=np.zeros(n))
    # start each balloon in equilibrium at its initial altitude
    atm = A.sample(t0, s.lat, s.lon, s.alt)
    s.ballonet_kg = ballonet_for_altitude(params, atm["rho"], atm["T"], s.alt, s.alt)
    controller.reset(n, s.alt)
    view = ControllerView(provider, A, f, t0)
    epoch_hours = (A.times[0] - np.datetime64("1970-01-01T00:00")) / np.timedelta64(1, "h")

    # what the vehicle believes at the start (possibly wrong)
    ang = rng.uniform(0, 2 * np.pi, n)
    err_m = f.init_pos_err_km * 1000
    b_lat = s.lat + np.degrees(err_m * np.cos(ang) / EARTH_R)
    b_lon = s.lon + np.degrees(err_m * np.sin(ang) / (EARTH_R * np.cos(np.radians(s.lat))))
    kf = WindBiasKF(mission.target_lat, mission.target_lon, b_lat, b_lon, s.alt) if estimator else None
    if kf is not None and np.any(err_m > 0):
        kf.P[:, 0, 0] += err_m ** 2
        kf.P[:, 1, 1] += err_m ** 2
    last_fix = np.stack([b_lat, b_lon], 1)

    n_steps = int(mission.duration_h * 3600 / dt_s)
    every_dec = int(round(decide_every_h * 3600 / dt_s))
    every_rec = int(round(record_every_h * 3600 / dt_s))
    every_gps = max(1, int(round(gps_every_s / dt_s)))
    rec = {k: [] for k in ("dist_km", "alt", "target_alt", "battery_wh", "energy_pump_wh",
                           "in_domain", "lat", "lon", "est_err_km", "forecast_age_h",
                           "telemetry_backlog")}
    target = s.alt.copy()
    backlog = np.zeros(n)
    # A balloon that leaves the data domain has no real wind to fly through;
    # it is frozen where it left and counted as a failure from then on.
    inside = A.contains(s.lat, s.lon)
    fc_u = fc_v = np.zeros(n)
    for i in range(n_steps + 1):
        now = t0 + i * dt_s / 3600
        el = now - t0
        gps_ok = f.gps_ok(el)
        baro = s.alt + f.baro_bias_m + rng.normal(0, noise.baro_alt_m, n)
        if i % every_gps == 0:
            g_lat = s.lat + rng.normal(0, noise.gps_m, n) / 111_000
            g_lon = s.lon + rng.normal(0, noise.gps_m, n) / (111_000 * np.cos(np.radians(s.lat)))
            last_fix = np.where(gps_ok[:, None], np.stack([g_lat, g_lon], 1), last_fix)
            if kf is not None:
                kf.update_gps(g_lat, g_lon, s.alt + rng.normal(0, noise.gps_m * 1.5, n), gps_ok)
                kf.update_baro(baro)
        # store-and-forward telemetry: records queue while the link is down
        backlog = np.where(f.comms_ok(el), 0.0, backlog + dt_s / 60)
        if i % every_dec == 0 and i < n_steps:
            view.tick(now)
            if kf is not None:
                e_lat, e_lon = kf.latlon
                alt_obs = kf.x[:, 2]
                view.bias = np.stack(kf.bias, 1) if bias_correction else None
            else:
                e_lat, e_lon = last_fix[:, 0], last_fix[:, 1]
                alt_obs = baro
            obs = Observation(hours=now, lat=e_lat, lon=e_lon, alt=alt_obs,
                              battery_wh=s.battery_wh.copy())
            target = controller.decide(obs, mission, view)
        if i % every_rec == 0:
            est = kf.latlon if kf is not None else (last_fix[:, 0], last_fix[:, 1])
            rec["dist_km"].append(distance_km(s.lat, s.lon, mission.target_lat, mission.target_lon))
            rec["alt"].append(s.alt.copy())
            rec["target_alt"].append(np.asarray(target, float).copy())
            rec["battery_wh"].append(s.battery_wh.copy())
            rec["energy_pump_wh"].append(s.energy_pump_wh.copy())
            rec["in_domain"].append(inside.copy())
            rec["lat"].append(s.lat.copy())
            rec["lon"].append(s.lon.copy())
            rec["est_err_km"].append(distance_km(s.lat, s.lon, *est))
            rec["forecast_age_h"].append(view.forecast_age_h() if view.now is not None
                                         else np.zeros(n))
            rec["telemetry_backlog"].append(backlog.copy())
        if i == n_steps:
            break
        atm = A.sample(now, s.lat, s.lon, s.alt)
        gu, gv = f.gust_uv(el)
        atm["u"], atm["v"] = atm["u"] + gu, atm["v"] + gv
        if kf is not None:
            # the filter propagates with the forecast at its own estimate of position
            if i % every_gps == 0:
                k_lat, k_lon = kf.latlon
                iss = view.latest_issue(now)
                saved, view.bias = view.bias, None     # the filter carries its own bias state
                fc_u, fc_v = view.wind(iss, now, k_lat, k_lon, kf.x[:, 2])
                view.bias = saved
            kf.predict(fc_u, fc_v, dt_s)
        sun = solar_elevation_sin(epoch_hours + now, s.lat, s.lon)
        new = step(params, s, atm, np.asarray(target, float), dt_s, sun,
                   pump_ok=el < f.pump_fail_h, battery_cap_wh=params.battery_wh * f.battery_factor)
        new.lat = np.where(inside, new.lat, s.lat)
        new.lon = np.where(inside, new.lon, s.lon)
        s = new
        inside = inside & A.contains(s.lat, s.lon)
    return {k: np.stack(v, 1) for k, v in rec.items()}


def mission_metrics(tr: dict[str, np.ndarray], radius_km: float = 50.0) -> pd.DataFrame:
    """Per-mission outcomes. Time outside the data domain counts as failure."""
    d = tr["dist_km"]
    inside = (d <= radius_km) & tr["in_domain"]
    changes = (np.abs(np.diff(tr["target_alt"], axis=1)) > 1).sum(1)
    return pd.DataFrame({
        "twr50": inside.mean(1),
        "mean_dist_km": d.mean(1),
        "final_dist_km": d[:, -1],
        "max_dist_km": d.max(1),
        "pump_energy_wh": tr["energy_pump_wh"][:, -1],
        "min_battery_wh": tr["battery_wh"].min(1),
        "target_changes": changes,
        "left_domain": ~tr["in_domain"].all(1),
        "est_err_mean_km": tr["est_err_km"].mean(1),
        "est_err_max_km": tr["est_err_km"].max(1),
        "max_forecast_age_h": tr["forecast_age_h"].max(1),
        "max_telemetry_backlog_min": tr["telemetry_backlog"].max(1),
    })
