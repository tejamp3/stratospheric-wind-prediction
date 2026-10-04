"""Case study: can we reproduce where Red Balloon's VISTA flight landed?

REAL DATA (sparse) + MODEL. VISTA launched from Vijayawada on 27 May 2026,
was reported at 12.2 km over Guntur district at 10:05 IST, reached nearly 25 km,
flew for 7 h 30 min (it was designed for 24 h) and was recovered in Raichur
district, Karnataka. No track, altitude profile or landing coordinates have
been published.

So this does not fit a trajectory. It asks three things:

1. Across every flight profile consistent with the public facts, do ERA5 winds
   carry the balloon from Vijayawada to Raichur district?
2. What did radiosondes at the four nearest stations actually measure at
   balloon altitude that week, and how does ERA5 compare?
3. Which single change closes the gap: the measured winds, a lower float
   altitude, or a longer flight?

It needs winds from the ground up, so it downloads the flight window at every
pressure level (one small CDS request), and radiosonde records from NOAA IGRA2.

Usage:  python experiments/vista_case_study.py [--profiles 2000]
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import urllib.request
import zipfile

import numpy as np
import pandas as pd

from stratoballoon import viz
from stratoballoon.atmosphere import Atmosphere, distance_km, move
from stratoballoon.config import ROOT
from stratoballoon.runlog import Run

log = logging.getLogger("vista")
LEVELS = [1000, 975, 950, 925, 900, 875, 850, 825, 800, 775, 750, 700, 650, 600, 550, 500, 450,
          400, 350, 300, 250, 225, 200, 175, 150, 125, 100, 70, 50, 30, 20]
LAUNCH = (16.51, 80.63)            # Indira Gandhi Stadium, Vijayawada
RAICHUR = (16.20, 77.36)           # Raichur town; the district spans roughly +-0.8 deg
IST = pd.Timedelta(hours=5, minutes=30)
IGRA = ("https://www.ncei.noaa.gov/data/integrated-global-radiosonde-archive/access/data-por/"
        "{}-data.txt.zip")
STATIONS = {"INM00043185": ("Machilipatnam", 16.20, 81.15), "INM00043128": ("Hyderabad", 17.45, 78.47),
            "INM00043150": ("Visakhapatnam", 17.68, 83.30), "INM00043295": ("Bengaluru", 12.97, 77.58)}
SONDE_LEVELS = [70, 50, 30, 20]    # hPa: about 18.5, 20.5, 24 and 26.5 km
STRATOSPHERE_M = 17_000


def fetch_era5(path):
    import cdsapi
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    cdsapi.Client(quiet=True, progress=False).retrieve("reanalysis-era5-pressure-levels", {
        "product_type": ["reanalysis"], "variable": ["u_component_of_wind",
                                                     "v_component_of_wind", "temperature",
                                                     "geopotential"],
        "pressure_level": [str(p) for p in LEVELS], "year": ["2026"], "month": ["05"],
        # the flight spans about 02:30-10:30 UTC; 00-12 UTC keeps the request
        # under the CDS cost limit (12 x days x times x levels x variables)
        "day": ["27"], "time": [f"{h:02d}:00" for h in range(13)],
        "area": [22, 74, 12, 84], "grid": [0.25, 0.25], "data_format": "netcdf",
        "download_format": "unarchived"}, str(path))


def sondes(first_day=24, last_day=30) -> pd.DataFrame:
    """Stratospheric winds measured at the four nearest stations that week."""
    rows = []
    for st, (name, _, _) in STATIONS.items():
        path = ROOT / "data" / "obs" / "igra2" / f"{st}-data.txt.zip"
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(urllib.request.urlopen(IGRA.format(st), timeout=300).read())
        z = zipfile.ZipFile(io.BytesIO(path.read_bytes()))
        keep, day = False, None
        for line in io.TextIOWrapper(z.open(z.namelist()[0]), encoding="ascii", errors="ignore"):
            if line.startswith("#"):
                y, m, day = int(line[13:17]), int(line[18:20]), int(line[21:23])
                keep = (y, m) == (2026, 5) and first_day <= day <= last_day
                continue
            if not keep:
                continue
            p, wd, ws = int(line[9:15]), int(line[40:45]), int(line[46:51])
            if wd < 0 or ws < 0 or p <= 0:
                continue
            level = min(SONDE_LEVELS, key=lambda l: abs(np.log(p / 100 / l)))
            if abs(np.log(p / 100 / level)) > 0.2:          # within about 20% of the level
                continue
            rad = np.radians(wd)
            rows.append({"station": name, "day": day, "level_hpa": level,
                         "u": -ws / 10 * np.sin(rad), "v": -ws / 10 * np.cos(rad)})
    return pd.DataFrame(rows)


def era5_at_stations(A: Atmosphere) -> pd.DataFrame:
    """ERA5 on the flight day at the same stations and levels (mean of 00-12 UTC)."""
    rows = []
    for name, lat, lon in STATIONS.values():
        yi, xi = int(np.argmin(np.abs(A.lats - lat))), int(np.argmin(np.abs(A.lons - lon)))
        for lev in SONDE_LEVELS:
            li = int(np.argmin(np.abs(A.levels - lev)))
            rows.append({"station": name, "level_hpa": lev, "u": float(A.u[:, li, yi, xi].mean()),
                         "height_m": float(A.h[:, li, yi, xi].mean())})
    return pd.DataFrame(rows)


def fly(A, n, seed, float_lo=22_000, float_hi=26_500, total_h=7.5, correction=None):
    """Fly n flight profiles consistent with the public facts.

    Ranges are assumptions: launch 08:00-09:45 IST, ascent 3-7 m/s, descent
    5-15 m/s, the given float altitude range and time aloft. `correction` is an
    (heights, du) pair added to ERA5's eastward wind above the tropopause.
    """
    rng = np.random.default_rng(seed)
    launch = pd.Timestamp("2026-05-27 08:00") + pd.to_timedelta(rng.uniform(0, 105, n), "min")
    ascent, descent = rng.uniform(3, 7, n), rng.uniform(5, 15, n)
    float_alt = rng.uniform(float_lo, float_hi, n)
    total_s = total_h * 3600
    t_up, t_down = float_alt / ascent, float_alt / descent
    t_float = total_s - t_up - t_down
    t0 = (((launch - IST) - pd.Timestamp(A.times[0])) / pd.Timedelta(hours=1)).to_numpy()
    lat, lon = np.full(n, LAUNCH[0]), np.full(n, LAUNCH[1])
    track = []
    for i in range(int(total_s / 60)):
        el = i * 60.0
        alt = np.where(el < t_up, ascent * el,
                       np.where(el < t_up + t_float, float_alt,
                                np.maximum(0, float_alt - descent * (el - t_up - t_float))))
        # ERA5 was downloaded to 12 UTC; a longer flight reuses the last hour
        s = A.sample(np.clip(t0 + el / 3600, 0, A.hours[-1]), lat, lon, np.maximum(alt, 150))
        u = s["u"]
        if correction is not None:
            u = u + np.where(alt > STRATOSPHERE_M, np.interp(alt, *correction), 0.0)
        lat, lon = move(lat, lon, u, s["v"], 60.0)
        if i % 10 == 0:
            track.append((lat.copy(), lon.copy()))
    t_12 = launch + pd.to_timedelta(12_200 / ascent, "s")
    # consistent with the report: passing 12.2 km within 30 minutes of 10:05 IST
    ok = (t_float > 0) & (np.abs((t_12 - pd.Timestamp("2026-05-27 10:05"))
                                 / pd.Timedelta(minutes=1)) <= 30)
    return {"lat": lat, "lon": lon, "ok": np.asarray(ok), "track": track,
            "float_alt": float_alt, "ascent": ascent, "descent": descent}


def summarise(r):
    ok = r["ok"]
    d = distance_km(r["lat"], r["lon"], *RAICHUR)[ok]
    flown = distance_km(*LAUNCH, r["lat"], r["lon"])[ok]
    return {"consistent_profiles": int(ok.sum()), "median_km_flown": float(np.median(flown)),
            "median_km_from_raichur_town": float(np.median(d)),
            "share_within_80km": float((d <= 80).mean())}


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = ROOT / "results" / "vista"
    path = ROOT / "data" / "raw_case" / "era5_vista_20260527_00-12utc_all_levels.nc"
    fetch_era5(path)
    run = Run(out, vars(a), [path])
    A = Atmosphere.from_files([path])

    # ---- 2. what the radiosondes measured, against ERA5
    obs = sondes()
    era = era5_at_stations(A)
    cmp_ = (obs.groupby("level_hpa").agg(sonde_u=("u", "mean"), n_obs=("u", "size"),
                                         n_soundings=("day", "nunique"))
            .join(era.groupby("level_hpa").agg(era5_u=("u", "mean"), height_m=("height_m", "mean")))
            .reset_index().sort_values("level_hpa", ascending=False))
    cmp_["difference"] = cmp_.sonde_u - cmp_.era5_u
    cmp_.to_csv(out / "sondes_vs_era5.csv", index=False)
    print(cmp_.round(1).to_string(index=False))
    correction = (cmp_.height_m.to_numpy(), cmp_.difference.to_numpy())

    # ---- 1 and 3. the reconstruction, then one change at a time
    scenarios = [
        ("ERA5 winds, float 22-26.5 km, 7.5 h aloft (baseline)", {}),
        ("float lower, 18-22 km", {"float_lo": 18_000, "float_hi": 22_000}),
        ("ERA5 corrected to the week's radiosonde means", {"correction": correction}),
        ("corrected winds and float 18-24 km", {"correction": correction, "float_lo": 18_000,
                                               "float_hi": 24_000}),
        ("ERA5 winds, 9.5 h aloft", {"total_h": 9.5}),
    ]
    rows, runs = [], {}
    for label, kw in scenarios:
        runs[label] = fly(A, a.profiles, a.seed, **kw)
        rows.append({"scenario": label, **summarise(runs[label])})
    sens = pd.DataFrame(rows)
    sens.to_csv(out / "sensitivity.csv", index=False)
    print(sens.round(2).to_string(index=False))

    base = runs[scenarios[0][0]]
    pd.DataFrame({"landing_lat": base["lat"], "landing_lon": base["lon"],
                  "km_from_raichur": distance_km(base["lat"], base["lon"], *RAICHUR),
                  "consistent_with_report": base["ok"], "ascent_ms": base["ascent"],
                  "float_alt_m": base["float_alt"], "descent_ms": base["descent"]}
                 ).to_csv(out / "profiles.csv", index=False)
    g_lat, g_lon = base["lat"][base["ok"]], base["lon"][base["ok"]]
    s0 = sens.iloc[0]
    summary = {
        "profiles": int(a.profiles), "consistent_profiles": int(s0.consistent_profiles),
        "median_km_from_raichur_town": float(s0.median_km_from_raichur_town),
        "share_within_80km_of_raichur_town": float(s0.share_within_80km),
        "median_landing": [float(np.median(g_lat)), float(np.median(g_lon))],
        "median_distance_flown_km": float(s0.median_km_flown),
        "straight_line_km": float(distance_km(*LAUNCH, *RAICHUR)),
        "needed_mean_speed_ms": float(distance_km(*LAUNCH, *RAICHUR) / 7.5 / 3.6),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1))

    figure(runs, scenarios, out)
    run.finish(summary=summary)
    return 0


def figure(runs, scenarios, out):
    import matplotlib.pyplot as plt
    viz.apply_style()
    fig, ax = plt.subplots(figsize=(9, 6))
    base = runs[scenarios[0][0]]
    corr = runs[scenarios[2][0]]
    tr_lat = np.stack([t[0] for t in base["track"]], 1)
    tr_lon = np.stack([t[1] for t in base["track"]], 1)
    for i in np.where(base["ok"])[0][:120]:
        ax.plot(tr_lon[i], tr_lat[i], color=viz.C1, alpha=0.07, lw=1)
    ax.scatter(base["lon"][base["ok"]], base["lat"][base["ok"]], s=6, color=viz.C1, alpha=0.45,
               label="landings with ERA5 winds as they are")
    ax.scatter(corr["lon"][corr["ok"]], corr["lat"][corr["ok"]], s=6, color=viz.C3, alpha=0.45,
               label="landings with ERA5 corrected to the radiosondes")
    ax.plot(*LAUNCH[::-1], marker="o", ms=9, color=viz.INK, label="launch: Vijayawada")
    ax.plot(*RAICHUR[::-1], marker="*", ms=16, color=viz.C2, label="reported landing district: Raichur")
    ax.add_patch(plt.Circle(RAICHUR[::-1], 0.75, fill=False, ls="--", color=viz.C2))
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect(1 / np.cos(np.radians(16.3)), adjustable="datalim")
    ax.set_title("VISTA, 27 May 2026: where do the winds take it?")
    ax.legend(loc="lower left", fontsize=8)
    viz.finish(fig, out / "vista.png",
               "REAL DATA: ERA5 winds, IGRA2 radiosondes, public launch and landing facts. "
               "MODEL: assumed ascent, float and descent ranges. Circle: rough extent of "
               "Raichur district.")


if __name__ == "__main__":
    raise SystemExit(main())
