"""Onboard telemetry anomaly detection.

SIMULATION (telemetry) + CONTROL (detectors). There is no public flight
telemetry for this vehicle, so telemetry is synthesised from a small model
with the physical couplings that make naive detection hard: the sun heats the
gas, which raises envelope super-pressure and lifts the balloon a little; it
charges the battery; it warms the payload. A detector that does not know about
the day/night cycle sees every sunrise as an anomaly.

Channels (1-minute rate)
  altitude_m, vspeed_ms, envelope_dp_pa, battery_v, solar_w, payload_temp_c,
  gps_residual_m (GPS minus the navigation filter's prediction), rssi_db

Injected faults (labelled windows)
  altimeter drift, altimeter bias step, GPS position jump, envelope leak,
  battery degradation, thermal excursion, sudden altitude loss, comms failure

Detectors, compared at the same false-alarm rate on nominal data
  * threshold: fixed limits per channel from nominal data
  * residual + CUSUM: subtract what a day/night model expects each channel to
    read (the digital-twin idea), then accumulate standardised residuals
  * Isolation Forest on windowed features
  * autoencoder (a small MLP trained to reconstruct nominal windows)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

WARMUP = 180   # minutes of each record a detector uses to settle; never scored

CHANNELS = ["altitude_m", "vspeed_ms", "envelope_dp_pa", "battery_v", "solar_w",
            "payload_temp_c", "gps_residual_m", "rssi_db"]
FAULTS = ["altimeter drift", "altimeter bias", "gps jump", "envelope leak",
          "battery degradation", "thermal excursion", "altitude loss", "comms failure"]


@dataclass
class Telemetry:
    data: np.ndarray        # (n_missions, T, C)
    sun: np.ndarray         # (n_missions, T) sin of solar elevation, known onboard
    label: np.ndarray       # (n_missions, T) fault index or -1
    events: pd.DataFrame    # mission, fault, start, end (minutes)


def simulate(n: int, days: float, rng: np.random.Generator, inject: bool = True,
             fault_rate_per_day: float = 0.5) -> Telemetry:
    T = int(days * 1440)
    t = np.arange(T) / 60.0                                       # hours
    lon = rng.uniform(66, 94, n)[:, None]
    solar_time = (t[None] + rng.uniform(0, 24, n)[:, None] + lon / 15) % 24
    sun = np.clip(np.cos(np.radians(15 * (solar_time - 12))) * 0.95, -1, 1)
    day = np.clip(sun, 0, None)
    # slow thermal lag of gas and payload behind the sun (about 1 h)
    k = np.exp(-1 / 60)
    lag = np.zeros_like(day)
    for i in range(1, T):
        lag[:, i] = k * lag[:, i - 1] + (1 - k) * day[:, i]
    alt0 = rng.uniform(21_000, 25_000, n)[:, None]
    altitude = alt0 + 150 * lag + np.cumsum(rng.normal(0, 0.6, (n, T)), 1)
    dp = 200 + 140 * lag + rng.normal(0, 2, (n, T))
    solar = 300 * day + rng.normal(0, 3, (n, T))
    energy = np.cumsum((solar - 60 - 15 * (rng.random((n, T)) < 0.02)) / 60, 1)
    soc = np.clip(0.6 + energy / 3000, 0.05, 1.0)
    battery = 22 + 3.5 * soc + rng.normal(0, 0.02, (n, T))
    temp = -12 + 20 * lag + rng.normal(0, 0.3, (n, T))
    gps_res = np.abs(rng.normal(0, 10, (n, T)))
    rssi = -95 + 4 * np.sin(2 * np.pi * t / 6)[None] + rng.normal(0, 1.5, (n, T))
    data = np.stack([altitude, np.gradient(altitude, axis=1) / 60, dp, battery, solar, temp,
                     gps_res, rssi], -1)
    label = np.full((n, T), -1)
    events = []
    if inject:
        for m in range(n):
            n_ev = rng.poisson(fault_rate_per_day * days)
            for _ in range(n_ev):
                f = int(rng.integers(len(FAULTS)))
                dur = int(rng.uniform(2, 10) * 60)
                s = int(rng.integers(WARMUP, max(WARMUP + 1, T - dur)))
                e = s + dur
                if (label[m, s:e] >= 0).any():
                    continue
                _apply(data[m], f, s, e, rng)
                label[m, s:e] = f
                events.append({"mission": m, "fault": FAULTS[f], "start": s, "end": e})
    return Telemetry(data, sun, label, pd.DataFrame(events,
                                                    columns=["mission", "fault", "start", "end"]))


def _apply(d, f, s, e, rng):
    r = np.arange(e - s)
    if f == 0:   # altimeter drift: reported altitude walks away
        d[s:e, 0] += r * rng.uniform(0.5, 1.5)
    elif f == 1:   # altimeter bias step
        d[s:e, 0] += rng.choice([-1, 1]) * rng.uniform(80, 200)
    elif f == 2:   # GPS jump (multipath or spoofing)
        d[s:e, 6] += rng.uniform(300, 2000)
    elif f == 3:   # envelope leak: super-pressure bleeds off
        d[s:e, 2] -= r * rng.uniform(0.2, 0.6)
    elif f == 4:   # battery degradation: voltage sags under load
        d[s:e, 3] -= np.minimum(r / 120, 1) * rng.uniform(0.15, 0.5)
    elif f == 5:   # thermal excursion
        d[s:e, 5] += np.minimum(r / 30, 1) * rng.uniform(8, 20)
    elif f == 6:   # sudden altitude loss (true motion: altitude and vertical speed)
        drop = np.minimum(r / 60, 1) * rng.uniform(400, 1200)
        d[s:e, 0] -= drop
        d[s:e, 1] -= np.gradient(drop) / 60
    elif f == 7:   # comms failure: link margin collapses
        d[s:e, 7] -= rng.uniform(15, 30)


# ------------------------------------------------------------------- features
WINDOW = 30   # minutes


def window_features(tel: Telemetry) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per (mission, window): mean, slope and spread of every channel, plus sun.

    Returns X (n_win, F), window end index (n_win,), mission (n_win,).
    """
    n, T, C = tel.data.shape
    ends = np.arange(WINDOW, T + 1, WINDOW // 2)
    feats, mission, idx = [], [], []
    tt = np.arange(WINDOW) - (WINDOW - 1) / 2
    for e in ends:
        w = tel.data[:, e - WINDOW:e]
        mean = w.mean(1)
        slope = (w * tt[None, :, None]).sum(1) / (tt ** 2).sum()
        sd = w.std(1)
        sun = tel.sun[:, e - WINDOW:e].mean(1, keepdims=True)
        feats.append(np.concatenate([mean, slope, sd, sun], 1))
        mission.append(np.arange(n))
        idx.append(np.full(n, e))
    return np.concatenate(feats), np.concatenate(idx), np.concatenate(mission)


class Expected:
    """Day/night model of each channel: what a healthy vehicle should read.

    A linear model per channel on lagged sun terms, fitted on nominal data. Its
    residuals remove the diurnal cycle, which is the only reason the
    model-based detectors can use tight thresholds.
    """

    def _design(self, tel: Telemetry):
        sun = tel.sun
        day = np.clip(sun, 0, None)
        k = np.exp(-1 / 60)
        lag = np.zeros_like(day)
        for i in range(1, day.shape[1]):
            lag[:, i] = k * lag[:, i - 1] + (1 - k) * day[:, i]
        # battery state of charge follows the integrated energy budget: measured
        # solar input minus the nominal load
        energy = np.cumsum((tel.data[..., CHANNELS.index("solar_w")] - 60) / 60, 1) / 3000
        return np.stack([np.ones_like(sun), day, lag, energy], -1)

    def fit(self, tel: Telemetry):
        X = self._design(tel).reshape(-1, 4)
        Y = tel.data.reshape(-1, tel.data.shape[-1])
        self.coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
        r = self.residual(tel).reshape(-1, tel.data.shape[-1])
        # robust spread of the healthy residual (median absolute deviation)
        self.sd = 1.4826 * np.median(np.abs(r - np.median(r, 0)), 0) + 1e-9
        return self

    def residual(self, tel: Telemetry) -> np.ndarray:
        pred = self._design(tel) @ self.coef
        r = tel.data - pred
        # remove each mission's slowly varying baseline: 3 h trailing mean
        base = _trailing_mean(r, 180)
        return r - base


def _trailing_mean(x, n):
    c = np.cumsum(np.pad(x, ((0, 0), (n, 0), (0, 0)), mode="edge"), axis=1)
    return (c[:, n:] - c[:, :-n]) / n


class ThresholdDetector:
    name = "threshold"

    def fit(self, tel):
        d = tel.data.reshape(-1, tel.data.shape[-1])
        self.lo, self.hi = np.percentile(d, 0.05, 0), np.percentile(d, 99.95, 0)
        return self

    def score(self, tel):
        d = tel.data
        z = np.maximum((self.lo - d) / (self.hi - self.lo), (d - self.hi) / (self.hi - self.lo))
        return z.max(-1)                                          # (n, T)


class CusumDetector:
    """Two-sided CUSUM on prewhitened, day/night-corrected residuals.

    Healthy residuals of slow channels (altitude, battery) are strongly
    autocorrelated, and CUSUM assumes independent inputs: run on the raw
    residual it accumulates ordinary wander into alarms. So each channel's
    residual is first reduced to its one-step prediction error under an AR(1)
    model fitted on healthy data, and the CUSUM runs on that.
    """
    name = "residual + CUSUM"

    def __init__(self, drift: float = 0.5, cap: float = 30.0):
        # The accumulators are capped so that, once a fault clears, the score
        # falls back within an hour instead of alarming for the rest of the day.
        self.k, self.cap = drift, cap

    def _innov(self, tel):
        r = self.model.residual(tel)
        return r[:, 1:] - self.phi * r[:, :-1]

    def fit(self, tel):
        self.model = Expected().fit(tel)
        r = self.model.residual(tel)
        a, b = r[:, 1:].reshape(-1, r.shape[-1]), r[:, :-1].reshape(-1, r.shape[-1])
        self.phi = np.clip((a * b).sum(0) / (b * b).sum(0), 0, 0.999)
        e = self._innov(tel).reshape(-1, r.shape[-1])
        self.sd = 1.4826 * np.median(np.abs(e - np.median(e, 0)), 0) + 1e-9
        self.mid = np.median(e, 0)
        return self

    def score(self, tel):
        z = np.clip((self._innov(tel) - self.mid) / self.sd, -20, 20)
        n, T1, C = z.shape
        up = np.zeros((n, C))
        dn = np.zeros((n, C))
        out = np.zeros((n, T1 + 1))
        for i in range(WARMUP, T1):
            up = np.clip(up + z[:, i] - self.k, 0, self.cap)
            dn = np.clip(dn - z[:, i] - self.k, 0, self.cap)
            out[:, i + 1] = np.maximum(up, dn).max(-1)
        return out


class _WindowDetector:
    """Shared plumbing for detectors that score 30-minute windows."""

    def fit(self, tel):
        X, _, _ = window_features(tel)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        self._fit((X - self.mu) / self.sd)
        return self

    def score(self, tel):
        X, ends, mission = window_features(tel)
        s = self._score((X - self.mu) / self.sd)
        n, T = tel.data.shape[:2]
        out = np.zeros((n, T))
        for e in np.unique(ends):
            m = ends == e
            out[mission[m], e - WINDOW // 2:e] = s[m][:, None]
        return out


class IsolationForestDetector(_WindowDetector):
    name = "isolation forest"

    def _fit(self, X):
        from sklearn.ensemble import IsolationForest
        self.m = IsolationForest(n_estimators=200, random_state=0).fit(X)

    def _score(self, X):
        return -self.m.score_samples(X)


class AutoencoderDetector(_WindowDetector):
    name = "autoencoder"

    def _fit(self, X):
        from sklearn.neural_network import MLPRegressor
        self.m = MLPRegressor(hidden_layer_sizes=(16, 6, 16), max_iter=300, random_state=0,
                              early_stopping=True).fit(X, X)

    def _score(self, X):
        return ((self.m.predict(X) - X) ** 2).mean(1)


DETECTORS = [ThresholdDetector, CusumDetector, IsolationForestDetector, AutoencoderDetector]


def evaluate(score: np.ndarray, tel: Telemetry, threshold: float, minutes_per_day=1440) -> dict:
    """Event recall, latency, false alarms per day and alarm precision."""
    alarm = score > threshold
    alarm[:, :WARMUP] = False
    n, T = alarm.shape
    rec, lat = [], []
    for ev in tel.events.itertuples():
        hit = np.where(alarm[ev.mission, ev.start:ev.end + 30])[0]
        rec.append(len(hit) > 0)
        if len(hit):
            lat.append(hit[0])
    # alarm onsets outside any fault window (+30 min grace) are false alarms
    inside = tel.label >= 0
    for ev in tel.events.itertuples():
        inside[ev.mission, ev.end:min(T, ev.end + 30)] = True
    onset = alarm & ~np.pad(alarm, ((0, 0), (1, 0)))[:, :-1]
    false = (onset & ~inside).sum()
    true_onsets = (onset & inside).sum()
    healthy = ~inside
    return {"recall": float(np.mean(rec)) if rec else np.nan,
            "healthy_time_in_alarm": float(alarm[healthy].mean()) if healthy.any() else np.nan,
            "median_latency_min": float(np.median(lat)) if lat else np.nan,
            "false_alarms_per_day": float(false / (n * T / minutes_per_day)),
            "precision": float(true_onsets / max(1, true_onsets + false))}


def threshold_for_rate(score_nominal: np.ndarray, per_day: float,
                       max_alarm_fraction: float = 0.01) -> float:
    """Lowest threshold whose false alarms on healthy telemetry stay in budget.

    Two conditions, because counting alarm onsets alone can be gamed: an
    alarm stuck permanently on has a single onset. So the alarmed fraction of
    healthy time is capped too.
    """
    score_nominal = score_nominal[:, WARMUP:]
    n, T = score_nominal.shape
    for q in np.concatenate([np.linspace(0.90, 0.999, 100), np.linspace(0.999, 1, 200)]):
        thr = float(np.quantile(score_nominal, q))
        a = score_nominal > thr
        onsets = (a & ~np.pad(a, ((0, 0), (1, 0)))[:, :-1]).sum() / (n * T / 1440)
        if onsets <= per_day and a.mean() <= max_alarm_fraction:
            return thr
    return float(np.max(score_nominal))
