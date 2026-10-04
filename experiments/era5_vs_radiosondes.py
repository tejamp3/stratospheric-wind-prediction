"""Experiment: how close is ERA5 to real observations at balloon altitudes?

ERA5 is both the "truth" the simulator flies through and the data the
forecasters learn from, so its own error matters. This compares ERA5 winds with
radiosonde soundings (weather balloons, NOAA's IGRA2 archive) at Indian
stations, at the mandatory pressure levels that ERA5 also has.

Caveats, stated in the output too: a radiosonde drifts tens of kilometres by
20 km altitude, and it samples a point while ERA5 represents a ~100 km grid box,
so part of the difference is representativeness, not ERA5 error. ERA5 also
assimilates many of these soundings, so the comparison is not independent; the
differences are a lower bound on ERA5's error where there are no soundings.

Usage:  python experiments/era5_vs_radiosondes.py --config configs/experiment.yaml
"""
from __future__ import annotations

import argparse
import io
import logging
import urllib.request
import zipfile

import numpy as np
import pandas as pd

from stratoballoon import viz
from stratoballoon.atmosphere import Atmosphere
from stratoballoon.config import ROOT
from stratoballoon.experiment import Context
from stratoballoon.runlog import Run

log = logging.getLogger("radiosondes")
URL = ("https://www.ncei.noaa.gov/data/integrated-global-radiosonde-archive/access/data-por/"
       "{}-data.txt.zip")
STATIONS = {
    "INM00042182": "New Delhi", "INM00042339": "Jodhpur", "INM00042410": "Guwahati",
    "INM00042647": "Ahmedabad", "INM00043003": "Mumbai", "INM00043128": "Hyderabad",
    "INM00043185": "Machilipatnam", "INM00043371": "Thiruvananthapuram",
}


def download(station: str) -> bytes:
    path = ROOT / "data" / "obs" / "igra2" / f"{station}-data.txt.zip"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        log.info("downloading %s", station)
        with urllib.request.urlopen(URL.format(station), timeout=300) as r:
            path.write_bytes(r.read())
    return path.read_bytes()


def parse(raw: bytes, first_year: int, last_year: int, levels_pa: set[int]) -> pd.DataFrame:
    """Winds at the requested pressure levels from IGRA2's fixed-width format."""
    rows = []
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        text = io.TextIOWrapper(z.open(z.namelist()[0]), encoding="ascii", errors="ignore")
        keep, when, lat, lon = False, None, None, None
        for line in text:
            if line.startswith("#"):
                year, month, day, hour = (int(line[13:17]), int(line[18:20]),
                                          int(line[21:23]), int(line[24:26]))
                keep = first_year <= year <= last_year and hour != 99
                if keep:
                    when = pd.Timestamp(year, month, day, hour)
                    lat, lon = int(line[55:62]) / 1e4, int(line[63:71]) / 1e4
                continue
            if not keep:
                continue
            p = int(line[9:15])
            if p not in levels_pa:
                continue
            wdir, wspd = int(line[40:45]), int(line[46:51])
            if wdir < 0 or wspd < 0:
                continue
            spd = wspd / 10.0
            rad = np.radians(wdir)
            rows.append({"time": when, "lat": lat, "lon": lon, "level_hpa": p // 100,
                         "u": -spd * np.sin(rad), "v": -spd * np.cos(rad)})
    return pd.DataFrame(rows)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ctx = Context(ap.parse_args().config)
    out = ctx.out / "radiosondes"
    run = Run(out, ctx.cfg, ctx.paths)
    # India only, recent years: the full wide-domain record would not fit in memory
    A = Atmosphere.from_files(ctx.paths, start="2019-01-01", bbox=(5, 35, 65, 100))
    years = pd.DatetimeIndex(A.times).year
    y0, y1 = int(years.min()), int(years.max())
    levels = {int(l) * 100 for l in A.levels}
    step = A.step_hours

    rows = []
    for st, name in STATIONS.items():
        obs = parse(download(st), y0, y1, levels)
        if obs.empty:
            continue
        # soundings are nominally 00 and 12 UTC; launch is ~45 min early, so
        # round to the nearest ERA5 time step
        t = obs.time.dt.round(f"{int(step)}h")
        hrs = ((t - pd.Timestamp(A.times[0])) / pd.Timedelta(hours=1)).to_numpy()
        ok = (hrs >= 0) & (hrs <= A.hours[-1]) & A.contains(obs.lat.to_numpy(), obs.lon.to_numpy())
        obs, hrs = obs[ok], hrs[ok]
        ti = np.searchsorted(A.hours, hrs)
        yi = np.interp(obs.lat, A.lats, np.arange(len(A.lats)))
        xi = np.interp(obs.lon, A.lons, np.arange(len(A.lons)))
        y0_, x0_ = np.floor(yi).astype(int), np.floor(xi).astype(int)
        wy, wx = yi - y0_, xi - x0_
        y1_, x1_ = np.minimum(y0_ + 1, len(A.lats) - 1), np.minimum(x0_ + 1, len(A.lons) - 1)
        li = np.array([int(np.argmin(np.abs(A.levels - l))) for l in obs.level_hpa])

        def bil(a):
            return ((1 - wy) * (1 - wx) * a[ti, li, y0_, x0_] + wy * (1 - wx) * a[ti, li, y1_, x0_]
                    + (1 - wy) * wx * a[ti, li, y0_, x1_] + wy * wx * a[ti, li, y1_, x1_])
        obs = obs.assign(u_era5=bil(A.u), v_era5=bil(A.v), station=name)
        rows.append(obs)
        log.info("%-20s %6d level-observations", name, len(obs))
    df = pd.concat(rows)
    df["du"], df["dv"] = df.u_era5 - df.u, df.v_era5 - df.v
    summ = df.groupby("level_hpa").apply(lambda g: pd.Series({
        "n": len(g),
        "vector_rms_diff_ms": float(np.sqrt(np.mean(g.du ** 2 + g.dv ** 2))),
        "bias_u_ms": float(g.du.mean()), "bias_v_ms": float(g.dv.mean()),
        "obs_mean_speed_ms": float(np.hypot(g.u, g.v).mean())}), include_groups=False).reset_index()
    by_station = df.groupby(["station", "level_hpa"]).apply(lambda g: float(
        np.sqrt(np.mean(g.du ** 2 + g.dv ** 2))), include_groups=False).unstack()
    # forecast error of the selected model at 6 h, against ERA5, for comparison
    m = pd.read_csv(ctx.forecast_dir / "metrics.csv")
    best = ctx.best_model()
    f6 = m[(m.model == best) & (m.horizon_h == 6)].groupby("level_hpa").vector_rmse.mean()
    summ["forecast_6h_vector_rmse_ms"] = summ.level_hpa.map(f6)
    summ.to_csv(out / "summary.csv", index=False)
    by_station.to_csv(out / "by_station.csv")
    print(summ.round(2).to_string(index=False))
    print(by_station.round(2))

    import matplotlib.pyplot as plt
    viz.apply_style()
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(summ))
    ax.bar(x - 0.2, summ.vector_rms_diff_ms, width=0.4, color=viz.C2,
           label="ERA5 vs radiosonde (RMS difference)")
    ax.bar(x + 0.2, summ.forecast_6h_vector_rmse_ms, width=0.4, color=viz.C1,
           label=f"{best} 6 h forecast vs ERA5 (RMSE)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(l)} hPa" for l in summ.level_hpa])
    ax.set_ylabel("Vector wind error (m/s)")
    ax.set_title("How far is the 'truth' from observations, next to the forecast error?")
    ax.legend(loc="upper left")
    viz.finish(fig, out / "era5_vs_radiosondes.png",
               "REAL DATA: IGRA2 radiosondes vs ERA5 at 8 Indian stations. Includes "
               "representativeness error; ERA5 assimilates these soundings.")
    run.finish(stations=list(STATIONS.values()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
