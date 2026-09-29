"""Loading, cleaning, windowing and splitting of the ERA5 stratospheric dataset.

The forecasting problem is framed as a single-station time-series task: predict
the 50 hPa wind vector over the airship's float point 6/12/24 h ahead, from a
24 h history of (u, v, T) at 50/30/10 hPa. The two upper levels are included
because stratospheric wind shear is the main driver of short-term change at the
float level.

Everything is indexed in timesteps of config.STEP_HOURS (3 h), so a 24 h history
is 8 steps and the 6/12/24 h horizons are 2/4/8 steps ahead.
"""
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

log = logging.getLogger("data")

# Base features: (u, v, T) at each level. u50 and v50 must stay first, in that
# order, because the residual skip connection adds those two channels straight
# onto the model output.
FEATURES = [f"{v}{lev}" for lev in C.LEVELS_HPA for v in ("u", "v", "t")]

# Clock features, appended so the base ordering is untouched. The EDA shows a
# clean 24 h tide in the meridional wind worth about 81% of its standard
# deviation, and a model given no clock cannot represent it. Encoding the hour
# as a sine/cosine pair rather than a raw number avoids the discontinuity at
# midnight, which is the same reason the targets are (u, v) rather than a bearing.
TIME_FEATURES = ["hour_sin", "hour_cos"]
FEATURES_WITH_TIME = FEATURES + TIME_FEATURES

TARGETS = ["u50", "v50"]


# --------------------------------------------------------------------- loading
def open_raw(paths: list[Path] | None = None) -> xr.Dataset:
    """Open all monthly files as one time-sorted dataset."""
    paths = sorted(paths or C.DATA_RAW.glob("era5_strat_*.nc"))
    if not paths:
        raise FileNotFoundError(
            f"No ERA5 files in {C.DATA_RAW}. Run: python src/download_era5.py"
        )
    ds = xr.open_mfdataset(
        paths, combine="by_coords", engine="netcdf4",
        # expver/number are scalar-ish metadata that break naive concatenation
        drop_variables=["expver", "number"],
    )
    ds = ds.sortby("valid_time")
    # Drop duplicated timestamps (months can overlap at boundaries after retries)
    _, keep = np.unique(ds.valid_time.values, return_index=True)
    if len(keep) != ds.valid_time.size:
        log.warning("Dropping %d duplicate timestamps", ds.valid_time.size - len(keep))
        ds = ds.isel(valid_time=np.sort(keep))
    return ds


def station_frame(ds: xr.Dataset,
                  lat: float = C.TARGET_LAT,
                  lon: float = C.TARGET_LON) -> pd.DataFrame:
    """Extract the (u, v, T) column above one point as a tidy hourly DataFrame."""
    col = ds.sel(latitude=lat, longitude=lon, method="nearest")
    out = {}
    for lev in C.LEVELS_HPA:
        at = col.sel(pressure_level=lev)
        for var in ("u", "v", "t"):
            out[f"{var}{lev}"] = at[var].values
    df = pd.DataFrame(out, index=pd.DatetimeIndex(ds.valid_time.values, name="time"))
    return df.astype("float32")


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Add wind speed and meteorological direction for each level."""
    df = df.copy()
    for lev in C.LEVELS_HPA:
        u, v = df[f"u{lev}"], df[f"v{lev}"]
        df[f"speed{lev}"] = np.hypot(u, v)
        # Meteorological convention: direction the wind blows FROM, degrees CW from N.
        df[f"dir{lev}"] = (np.degrees(np.arctan2(-u, -v)) % 360).astype("float32")
    df["shear_10_50"] = df["speed10"] - df["speed50"]
    hour = df.index.hour.to_numpy()
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24).astype("float32")
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24).astype("float32")
    return df


# -------------------------------------------------------------------- cleaning
def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reindex to a gapless STEP_HOURS axis, then forward-fill + interpolate.

    Returns the cleaned frame and a boolean mask of values that were imputed,
    so the EDA can show exactly where data was missing. Reindexing first is what
    turns an absent timestamp into a visible NaN - without it a missing hour
    would silently vanish from the series and corrupt every window that spans it.
    """
    full = pd.date_range(df.index.min(), df.index.max(), freq=f"{C.STEP_HOURS}h")
    df = df.reindex(full)
    df.index.name = "time"
    missing = df[FEATURES].isna()
    # Forward fill handles single dropouts (persistence is a good stratospheric
    # prior over one step); interpolation then closes any interior run, and bfill
    # covers a gap at the very start.
    df[FEATURES] = (df[FEATURES].ffill(limit=1)
                                .interpolate(method="time", limit_direction="both")
                                .bfill())
    return df, missing


# ------------------------------------------------------------ split + normalise
@dataclass
class Scaler:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def inverse(self, x: np.ndarray) -> np.ndarray:
        return x * self.std + self.mean

    def save(self, path: Path) -> None:
        np.savez(path, mean=self.mean, std=self.std)

    @classmethod
    def load(cls, path: Path) -> "Scaler":
        z = np.load(path)
        return cls(z["mean"], z["std"])


def time_splits(n: int, fractions=C.SPLITS) -> tuple[slice, slice, slice]:
    """Chronological split indices - never random, to avoid leaking the future."""
    a = int(n * fractions[0])
    b = int(n * (fractions[0] + fractions[1]))
    return slice(0, a), slice(a, b), slice(b, n)


def make_windows(x: np.ndarray, y: np.ndarray,
                 input_steps: int = C.INPUT_STEPS,
                 horizon_steps: list[int] = C.HORIZON_STEPS
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Sliding windows, all indices in timesteps.

    x: (T, F) normalised features.  y: (T, 2) normalised (u50, v50) targets.
    Returns X (N, input_steps, F) and Y (N, n_horizons, 2).

    Window i spans x[i : i+input_steps]; the last observed step is
    t = i+input_steps-1, and the labels are y[t + h] for each horizon h. The
    count stops max(h) short of the end so every label exists.
    """
    hmax = max(horizon_steps)
    n = len(x) - input_steps - hmax + 1
    if n <= 0:
        raise ValueError(
            f"Series too short ({len(x)} steps) for {input_steps}-step window "
            f"+ {hmax}-step horizon")
    idx = np.arange(n)
    X = np.stack([x[i:i + input_steps] for i in idx]).astype("float32")
    last = idx + input_steps - 1
    Y = np.stack([y[last + h] for h in horizon_steps], axis=1).astype("float32")
    return X, Y


def build_dataset(df: pd.DataFrame | None = None, verbose: bool = True,
                  input_steps: int = C.INPUT_STEPS,
                  feature_names: list[str] | None = None):
    """End-to-end: raw files -> normalised windowed tensors + metadata.

    Normalisation statistics are fit on the TRAIN slice only. `input_steps` and
    `feature_names` are exposed so the history length and the feature set can be
    treated as hyperparameters; anything that loads a trained model must pass the
    values that model was trained with, which is why both are recorded in the
    checkpoint.
    """
    if df is None:
        df, _ = clean(add_derived(station_frame(open_raw())))
    feature_names = list(feature_names or FEATURES)

    feats = df[feature_names].to_numpy("float32")
    targs = df[TARGETS].to_numpy("float32")
    tr, va, te = time_splits(len(df))

    scaler_x = Scaler(feats[tr].mean(0), feats[tr].std(0) + 1e-8)
    scaler_y = Scaler(targs[tr].mean(0), targs[tr].std(0) + 1e-8)
    fx, fy = scaler_x.transform(feats), scaler_y.transform(targs)

    out = {}
    for name, sl in (("train", tr), ("val", va), ("test", te)):
        # Windows are built inside each contiguous slice, so no window ever
        # straddles a split boundary.
        X, Y = make_windows(fx[sl], fy[sl], input_steps=input_steps)
        # Timestamp of the last observed step in each window, for plotting.
        t0 = df.index[sl][input_steps - 1: input_steps - 1 + len(X)]
        out[name] = {"X": X, "Y": Y, "time": t0}
        if verbose:
            log.info("%-5s X=%s Y=%s  %s -> %s", name, X.shape, Y.shape,
                     t0[0].date(), t0[-1].date())

    return out, scaler_x, scaler_y, df
