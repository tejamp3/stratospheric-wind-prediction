"""State estimation: where is the balloon, and how wrong is the forecast right now?

CONTROL (onboard). A Kalman filter on a local east/north/up frame centred on the
station.

State  x = [E, N, U, w, bE, bN]
  E, N, U  position (m), U is altitude
  w        vertical speed (m/s)
  bE, bN   forecast wind error at the balloon's altitude (m/s): truth minus
           forecast, modelled as a slow random walk

Process  E += (forecast_u + bE) dt,  N += (forecast_v + bN) dt,  U += w dt
Sensors  GPS position (E, N, U) when available; barometric altitude always.

Why a filter at all, when GPS is accurate to metres?
  1. GPS can drop out. Then the filter dead-reckons with the forecast plus
     its own estimate of the forecast error, instead of freezing at the last
     fix.
  2. The bias states are a running measurement of how wrong the forecast is
     at the balloon's position. That is information the controller can use.

Why a linear Kalman filter and not an EKF or UKF? With the forecast wind as a
known input and pressure converted to altitude before the filter, every
equation above is linear, so the extended and unscented variants would
compute the same thing at more cost. The flat-earth frame is an approximation
that loses accuracy beyond about 1,000 km from the station.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from stratoballoon.atmosphere import EARTH_R


@dataclass
class KFNoise:
    gps_h_m: float = 10.0
    gps_v_m: float = 15.0
    baro_m: float = 30.0
    accel_ms2: float = 0.002     # vertical-speed random walk
    bias_ms_per_sqrt_h: float = 0.5
    pos_process_m: float = 5.0


def to_enu(lat, lon, lat0, lon0):
    e = np.radians(lon - lon0) * EARTH_R * np.cos(np.radians(lat0))
    n = np.radians(lat - lat0) * EARTH_R
    return e, n


def from_enu(e, n, lat0, lon0):
    lat = lat0 + np.degrees(n / EARTH_R)
    lon = lon0 + np.degrees(e / (EARTH_R * np.cos(np.radians(lat0))))
    return lat, lon


class WindBiasKF:
    """Vectorised over missions: x (n, 6), P (n, 6, 6)."""
    H_GPS = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]], float)
    H_BARO = np.array([[0, 0, 1, 0, 0, 0]], float)

    def __init__(self, lat0, lon0, lat, lon, alt, noise: KFNoise = KFNoise()):
        self.lat0, self.lon0, self.q = np.asarray(lat0), np.asarray(lon0), noise
        n = len(lat)
        e, nn = to_enu(lat, lon, self.lat0, self.lon0)
        self.x = np.zeros((n, 6))
        self.x[:, 0], self.x[:, 1], self.x[:, 2] = e, nn, alt
        self.P = np.tile(np.diag([50.0, 50.0, 50.0, 1.0, 3.0, 3.0]) ** 2, (n, 1, 1))

    def predict(self, fu, fv, dt):
        F = np.eye(6)
        F[0, 4] = F[1, 5] = F[2, 3] = dt
        x = self.x @ F.T
        x[:, 0] += fu * dt
        x[:, 1] += fv * dt
        q = self.q
        Q = np.diag([(q.pos_process_m) ** 2, (q.pos_process_m) ** 2, 1.0,
                     (q.accel_ms2 * dt) ** 2,
                     q.bias_ms_per_sqrt_h ** 2 * dt / 3600, q.bias_ms_per_sqrt_h ** 2 * dt / 3600])
        self.x = x
        self.P = F @ self.P @ F.T + Q

    def _update(self, z, H, R, mask):
        if not mask.any():
            return np.zeros(len(z))
        x, P = self.x[mask], self.P[mask]
        y = z[mask] - x @ H.T
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        self.x[mask] = x + np.einsum("nij,nj->ni", K, y)
        self.P[mask] = (np.eye(6) - K @ H) @ P
        nis = np.full(len(z), np.nan)
        nis[mask] = np.einsum("ni,nij,nj->n", y, np.linalg.inv(S), y)
        return nis

    def update_gps(self, lat, lon, alt, available):
        e, n = to_enu(lat, lon, self.lat0, self.lon0)
        q = self.q
        R = np.diag([q.gps_h_m ** 2, q.gps_h_m ** 2, q.gps_v_m ** 2])
        return self._update(np.stack([e, n, alt], 1), self.H_GPS, R, np.asarray(available))

    def update_baro(self, alt):
        R = np.array([[self.q.baro_m ** 2]])
        return self._update(alt[:, None], self.H_BARO, R, np.ones(len(alt), bool))

    @property
    def latlon(self):
        return from_enu(self.x[:, 0], self.x[:, 1], self.lat0, self.lon0)

    @property
    def bias(self):
        return self.x[:, 4], self.x[:, 5]

    @property
    def pos_sigma_m(self):
        return np.sqrt(self.P[:, 0, 0] + self.P[:, 1, 1])
