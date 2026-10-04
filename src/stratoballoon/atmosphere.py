"""The atmosphere as a gridded field the balloon flies through.

REAL DATA. Everything in this module is ERA5 reanalysis, reshaped and
interpolated; nothing here is a prediction.

Layout used everywhere downstream: arrays of shape (time, level, lat, lon) with
levels ordered from highest pressure to lowest (so altitude increases with the
level index) and latitude ascending.

Altitude: when geopotential is present it is converted to geopotential height
(z / g0), which is within about 0.5% of geometric height at these altitudes.
Without it, heights are built hypsometrically from the
temperature profile, anchored at the lowest level's standard-atmosphere height;
that anchor is the main approximation and is recorded on the object.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

R_D = 287.05          # J kg-1 K-1, dry air
G0 = 9.80665          # m s-2
EARTH_R = 6_371_000.0 # m


def standard_height(p_hpa: float) -> float:
    """US Standard Atmosphere 1976 height (m) for a pressure, 11-32 km layers."""
    p = p_hpa * 100
    if p >= 5474.89:                       # 11-20 km, isothermal 216.65 K
        return 11_000 + R_D * 216.65 / G0 * np.log(22632.1 / p)
    # 20-32 km, lapse rate +1 K/km from 216.65 K at 20 km
    t0, lapse = 216.65, 0.001
    return 20_000 + t0 / lapse * ((p / 5474.89) ** (-R_D * lapse / G0) - 1)


def open_era5(paths: list[Path]) -> xr.Dataset:
    """Open monthly ERA5 files as one dataset in the canonical layout."""
    ds = xr.open_mfdataset(sorted(paths), combine="by_coords",
                           drop_variables=["expver", "number"])
    ds = ds.rename({"valid_time": "time", "pressure_level": "level",
                    "latitude": "lat", "longitude": "lon"})
    ds = ds.sortby("time").sortby("lat").sortby("level", ascending=False)
    _, keep = np.unique(ds.time.values, return_index=True)
    return ds.isel(time=np.sort(keep))


@dataclass
class Atmosphere:
    """In-memory gridded atmosphere with vectorised interpolation."""
    times: np.ndarray        # (T,) datetime64[ns]
    levels: np.ndarray       # (L,) hPa, decreasing
    lats: np.ndarray         # (Y,) ascending
    lons: np.ndarray         # (X,) ascending
    u: np.ndarray            # (T, L, Y, X) m/s, eastward
    v: np.ndarray            # (T, L, Y, X) m/s, northward
    t: np.ndarray            # (T, L, Y, X) K
    h: np.ndarray            # (T, L, Y, X) m, geopotential height
    height_source: str = "geopotential"

    # ---------------------------------------------------------- construction
    @classmethod
    def from_dataset(cls, ds: xr.Dataset, start=None, end=None, stride: int = 1) -> "Atmosphere":
        if start is not None or end is not None:
            ds = ds.sel(time=slice(start, end))
        if stride > 1:
            ds = ds.isel(lat=slice(None, None, stride), lon=slice(None, None, stride))
        steps = np.diff(ds.time.values) / np.timedelta64(1, "h")
        if len(steps) and not np.all(steps == steps[0]):
            # A missing month would otherwise be stitched over silently, and every
            # window spanning it would mix times days apart.
            gaps = [(str(ds.time.values[i])[:13], float(steps[i]))
                    for i in np.where(steps != np.median(steps))[0][:5]]
            raise ValueError(f"irregular time axis (gaps after, hours): {gaps}; "
                             "re-run the downloader to fill missing months")
        u = ds["u"].values.astype("float32")
        v = ds["v"].values.astype("float32")
        t = ds["t"].values.astype("float32")
        levels = ds.level.values.astype(float)
        if "z" in ds:
            h = (ds["z"].values / G0).astype("float32")
            src = "geopotential"
        else:
            h = hypsometric_heights(t, levels)
            src = "hypsometric from T, anchored at standard-atmosphere height"
        return cls(ds.time.values, levels, ds.lat.values.astype(float),
                   ds.lon.values.astype(float), u, v, t, h, src)

    @classmethod
    def from_files(cls, paths: list[Path], start=None, end=None, stride: int = 1,
                   bbox: tuple[float, float, float, float] | None = None) -> "Atmosphere":
        """Load a time window, optionally a lat/lon box (S, N, W, E) and every
        `stride`-th grid point.

        Selection happens before loading, so a 12-year, wide-domain record does
        not have to fit in memory to train on a thinned grid or simulate one year.
        """
        ds = open_era5(paths)
        if start is not None or end is not None:
            ds = ds.sel(time=slice(start, end))
        if bbox is not None:
            s_, n_, w_, e_ = bbox
            ds = ds.sel(lat=slice(s_, n_), lon=slice(w_, e_))
        if stride > 1:
            ds = ds.isel(lat=slice(None, None, stride), lon=slice(None, None, stride))
        return cls.from_dataset(ds.load())

    # ------------------------------------------------------------ properties
    @property
    def hours(self) -> np.ndarray:
        """Time axis as float hours since the first sample."""
        return (self.times - self.times[0]) / np.timedelta64(1, "h")

    @property
    def step_hours(self) -> float:
        return float(np.median(np.diff(self.hours)))

    def time_index(self, when) -> int:
        return int(np.searchsorted(self.times, np.datetime64(pd.Timestamp(when))))

    def contains(self, lat, lon) -> np.ndarray:
        return ((lat >= self.lats[0]) & (lat <= self.lats[-1])
                & (lon >= self.lons[0]) & (lon <= self.lons[-1]))

    # --------------------------------------------------------- interpolation
    def _horizontal(self, ti, wt, lat, lon):
        """Bilinear in space, linear in time; returns column profiles (N, L)."""
        yi = np.clip(np.interp(lat, self.lats, np.arange(len(self.lats))), 0, len(self.lats) - 1)
        xi = np.clip(np.interp(lon, self.lons, np.arange(len(self.lons))), 0, len(self.lons) - 1)
        y0 = np.minimum(yi.astype(int), len(self.lats) - 2)
        x0 = np.minimum(xi.astype(int), len(self.lons) - 2)
        wy, wx = (yi - y0)[:, None], (xi - x0)[:, None]
        t0 = np.minimum(ti, len(self.times) - 2)
        wt = wt[:, None]

        def col(a):
            out = 0.0
            for dt_, wt_ in ((0, 1 - wt), (1, wt)):
                for dy, wy_ in ((0, 1 - wy), (1, wy)):
                    for dx, wx_ in ((0, 1 - wx), (1, wx)):
                        out = out + wt_ * wy_ * wx_ * a[t0 + dt_, :, y0 + dy, x0 + dx]
            return out
        return col(self.u), col(self.v), col(self.t), col(self.h)

    def time_weights(self, hours_since_start: np.ndarray):
        th = self.hours
        f = np.clip(np.interp(hours_since_start, th, np.arange(len(th))), 0, len(th) - 1 - 1e-9)
        ti = f.astype(int)
        return ti, f - ti

    def sample(self, hours_since_start, lat, lon, alt_m) -> dict[str, np.ndarray]:
        """Wind, temperature, pressure and density at arbitrary points.

        Vertical interpolation is linear in height for u, v and T, and
        log-linear for pressure. Points above or below the data are clamped
        to the top or bottom level and flagged.
        """
        hrs = np.atleast_1d(np.asarray(hours_since_start, float))
        lat = np.atleast_1d(np.asarray(lat, float))
        lon = np.atleast_1d(np.asarray(lon, float))
        alt = np.atleast_1d(np.asarray(alt_m, float))
        hrs, lat, lon, alt = np.broadcast_arrays(hrs, lat, lon, alt)
        ti, wt = self.time_weights(hrs.ravel())
        u, v, t, h = self._horizontal(ti, wt, lat.ravel(), lon.ravel())
        a = alt.ravel()
        n_lev = h.shape[1]
        k = np.clip((h < a[:, None]).sum(1) - 1, 0, n_lev - 2)
        rows = np.arange(len(a))
        h0, h1 = h[rows, k], h[rows, k + 1]
        w = np.clip((a - h0) / (h1 - h0), 0.0, 1.0)

        def lin(x):
            return x[rows, k] * (1 - w) + x[rows, k + 1] * w
        logp = np.log(self.levels)
        p_hpa = np.exp(logp[k] * (1 - w) + logp[k + 1] * w)
        T = lin(t)
        out = {"u": lin(u), "v": lin(v), "T": T, "p_hpa": p_hpa,
               "rho": p_hpa * 100 / (R_D * T),
               "out_of_column": (a < h[:, 0]) | (a > h[:, -1])}
        return {k_: v_.reshape(alt.shape) for k_, v_ in out.items()}

    def altitude_of_level(self, level_hpa: float) -> float:
        """Domain- and time-mean height of a pressure level, for config defaults."""
        i = int(np.argmin(np.abs(self.levels - level_hpa)))
        return float(self.h[:, i].mean())


def hypsometric_heights(t: np.ndarray, levels_hpa: np.ndarray) -> np.ndarray:
    """Heights of each pressure level from the temperature profile.

    dz = (R_d / g0) * mean(T) * ln(p_lower / p_upper) between adjacent levels,
    starting from the standard-atmosphere height of the highest-pressure level.
    """
    h = np.empty_like(t, dtype="float32")
    h[:, 0] = standard_height(levels_hpa[0])
    for k in range(1, len(levels_hpa)):
        tm = 0.5 * (t[:, k - 1] + t[:, k])
        h[:, k] = h[:, k - 1] + R_D / G0 * tm * np.log(levels_hpa[k - 1] / levels_hpa[k])
    return h


def move(lat, lon, u, v, seconds):
    """Advance positions by a velocity on a sphere (small-step approximation)."""
    dlat = v * seconds / EARTH_R
    dlon = u * seconds / (EARTH_R * np.cos(np.radians(lat)))
    return lat + np.degrees(dlat), lon + np.degrees(dlon)


def distance_km(lat1, lon1, lat2, lon2):
    """Great-circle distance (haversine)."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_R / 1000 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def bearing_deg(lat1, lon1, lat2, lon2):
    """Initial bearing from point 1 to point 2, degrees clockwise from north."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(lon2 - lon1)
    x = np.sin(dl) * np.cos(p2)
    y = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl)
    return np.degrees(np.arctan2(x, y)) % 360
