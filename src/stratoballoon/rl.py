"""Reinforcement learning for altitude control, on the same simulator.

CONTROL. Formulated so the comparison with the rule-based and MPC controllers
is fair: identical atmosphere, dynamics, forecast provider and decision
interval (3 h), and the policy only sees what the other controllers see.

State (observation), all scaled to roughly unit range
  target offset east/north (1,000 km units) and distance
  altitude within the band, battery fraction, mission time used
  forecast age (stale forecasts after comms loss)
  forecast wind (u, v) at every band altitude at the balloon's position, now
  and 6 h ahead, and the forecast's calibrated 6 h error radius per altitude

Action
  which of the band's grid altitudes to command (Discrete(n_grid)); holding is
  choosing the current one

Reward, per 3 h decision
  + fraction of the interval within 50 km of the target    (station keeping)
  + 0.2 x distance closed, per 100 km, clipped to +-1       (progress)
  - 0.002 per Wh of pump energy                             (energy)
  - 0.05 per change of commanded altitude                   (actuator wear)
  - 1 on leaving the data domain, 0.5 if the battery empties (safety)
  Reaching the target is not rewarded on its own: holding near it is.

Training uses missions from the validation year only; evaluation uses the
test year through the same Monte-Carlo harness as every other controller.
"""
from __future__ import annotations

import numpy as np

from stratoballoon.atmosphere import distance_km
from stratoballoon.control import Controller
from stratoballoon.dynamics import ballonet_for_altitude, solar_elevation_sin, step
from stratoballoon.dynamics import BalloonState
from stratoballoon.estimation import to_enu

DECIDE_H = 3.0


def observation(hours, lat, lon, alt, battery_frac, elapsed_frac, target_lat, target_lon,
                provider, band, radius6) -> np.ndarray:
    n, G = len(lat), band.n_grid
    e, nn = to_enu(lat, lon, target_lat, target_lon)
    d = np.hypot(e, nn)
    issue = provider.latest_issue(hours)
    age = hours - provider.issue_hours[issue]
    feats = [e / 1e6, nn / 1e6, d / 1e6,
             (alt - band.alt_min) / (band.alt_max - band.alt_min) * 2 - 1,
             battery_frac, elapsed_frac, np.clip(age / 24, 0, 3)]
    gl = np.tile(band.grid, n)
    rep = lambda a: np.repeat(a, G)  # noqa: E731
    for lead in (0.0, 6.0):
        u, v = provider.wind(rep(issue), rep(hours) + lead, rep(lat), rep(lon), gl)
        feats += list((u.reshape(n, G) / 20).T) + list((v.reshape(n, G) / 20).T)
    feats += list(np.repeat(radius6[None], n, 0).T / 10)
    return np.stack(feats, 1).astype("float32")


def _radius6(conformal, A, band) -> np.ndarray:
    """90% conformal radius at 6 h for each band altitude (nearest data level)."""
    r = conformal.radius(0.9)[0]                       # (L,)
    hmean = A.h.mean(axis=(0, 2, 3))
    return np.array([r[int(np.argmin(np.abs(hmean - g)))] for g in band.grid])


def make_vec_env(A, provider, conformal, params, band, start_idx, n_envs=64,
                 duration_h=72.0, dt_s=120.0, seed=0):
    """A Stable-Baselines3 VecEnv flying `n_envs` balloons in lockstep."""
    from gymnasium import spaces
    from stable_baselines3.common.vec_env import VecEnv

    radius6 = _radius6(conformal, A, band)
    n_obs = 7 + 4 * band.n_grid + band.n_grid
    epoch = (A.times[0] - np.datetime64("1970-01-01T00:00")) / np.timedelta64(1, "h")

    class StationKeepingVecEnv(VecEnv):
        def __init__(self):
            super().__init__(n_envs, spaces.Box(-np.inf, np.inf, (n_obs,), np.float32),
                             spaces.Discrete(band.n_grid))
            self.rng = np.random.default_rng(seed)
            self.actions = None

        # ------------------------------------------------------------ helpers
        def _new(self, mask):
            k = int(mask.sum())
            if k == 0:
                return
            self.t0[mask] = A.hours[self.rng.choice(start_idx, k)]
            self.tlat[mask] = self.rng.uniform(10, 30, k)
            self.tlon[mask] = self.rng.uniform(60, 110, k)
            alt = self.rng.uniform(band.alt_min, band.alt_max, k)
            atm = A.sample(self.t0[mask], self.tlat[mask], self.tlon[mask], alt)
            s = self.s
            s.lat[mask], s.lon[mask], s.alt[mask] = self.tlat[mask], self.tlon[mask], alt
            s.ballonet_kg[mask] = ballonet_for_altitude(params, atm["rho"], atm["T"], alt, alt)
            s.battery_wh[mask] = params.battery_wh
            s.energy_pump_wh[mask] = 0
            self.el[mask] = 0
            self.cmd[mask] = band.nearest(alt)

        def _obs(self):
            s = self.s
            return observation(self.t0 + self.el, s.lat, s.lon, s.alt,
                               s.battery_wh / params.battery_wh, self.el / duration_h,
                               self.tlat, self.tlon, provider, band, radius6)

        # ------------------------------------------------------------ VecEnv API
        def reset(self):
            z = np.zeros(n_envs)
            self.s = BalloonState(z.copy(), z.copy(), z.copy(), z.copy(), z.copy(), z.copy(),
                                  z.copy())
            self.t0, self.tlat, self.tlon = z.copy(), z.copy(), z.copy()
            self.el, self.cmd = z.copy(), np.zeros(n_envs, int)
            self._new(np.ones(n_envs, bool))
            return self._obs()

        def step_async(self, actions):
            self.actions = np.asarray(actions, int)

        def step_wait(self):
            s = self.s
            target = band.grid[self.actions]
            changed = self.actions != self.cmd
            self.cmd = self.actions.copy()
            d0 = distance_km(s.lat, s.lon, self.tlat, self.tlon)
            e0 = s.energy_pump_wh.copy()
            inside = np.zeros(n_envs)
            alive = A.contains(s.lat, s.lon)
            n_sub = int(DECIDE_H * 3600 / dt_s)
            for _ in range(n_sub):
                now = self.t0 + self.el
                atm = A.sample(now, s.lat, s.lon, s.alt)
                sun = solar_elevation_sin(epoch + now, s.lat, s.lon)
                new = step(params, s, atm, target, dt_s, sun)
                new.lat = np.where(alive, new.lat, s.lat)
                new.lon = np.where(alive, new.lon, s.lon)
                s = new
                alive = alive & A.contains(s.lat, s.lon)
                self.el = self.el + dt_s / 3600
                inside += (distance_km(s.lat, s.lon, self.tlat, self.tlon) <= 50) & alive
            self.s = s
            d1 = distance_km(s.lat, s.lon, self.tlat, self.tlon)
            r = (inside / n_sub + 0.2 * np.clip((d0 - d1) / 100, -1, 1)
                 - 0.002 * (s.energy_pump_wh - e0) - 0.05 * changed
                 - 1.0 * ~alive - 0.5 * (s.battery_wh <= 0))
            done = (self.el >= duration_h - 1e-6) | ~alive
            infos = [{} for _ in range(n_envs)]
            obs = self._obs()
            if done.any():
                for i in np.where(done)[0]:
                    infos[i]["terminal_observation"] = obs[i]
                    infos[i]["TimeLimit.truncated"] = bool(alive[i])
                self._new(done)
                obs = self._obs()
            return obs, r.astype("float32"), done, infos

        def close(self):
            pass

        def get_attr(self, attr_name, indices=None):
            return [getattr(self, attr_name)] * n_envs

        def set_attr(self, attr_name, value, indices=None):
            setattr(self, attr_name, value)

        def env_method(self, method_name, *args, indices=None, **kwargs):
            return [getattr(self, method_name)(*args, **kwargs)] * n_envs

        def env_is_wrapped(self, wrapper_class, indices=None):
            return [False] * n_envs

        def seed(self, seed=None):
            self.rng = np.random.default_rng(seed)
            return [seed] * n_envs

    return StationKeepingVecEnv()


class RLController(Controller):
    """A trained policy behind the common controller interface."""
    name = "rl-ppo"

    def __init__(self, model, band, params, conformal, A, duration_h=72.0):
        self.model, self.band, self.params = model, band, params
        self.radius6 = _radius6(conformal, A, band)
        self.duration_h = duration_h

    def reset(self, n, alt0):
        super().reset(n, alt0)
        self.t_start = None

    def decide(self, obs, mission, provider):
        if self.t_start is None:
            self.t_start = obs.hours.copy()
        el = (obs.hours - self.t_start) / self.duration_h
        o = observation(obs.hours, obs.lat, obs.lon, obs.alt,
                        obs.battery_wh / self.params.battery_wh, el, mission.target_lat,
                        mission.target_lon, provider, self.band, self.radius6)
        a, _ = self.model.predict(o, deterministic=True)
        self.target = self.band.grid[np.asarray(a, int)]
        return self.target
