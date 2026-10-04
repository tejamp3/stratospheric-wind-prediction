"""Experiment: when and where is station keeping physically possible at all?

Flies MPC with a *perfect* forecast (the upper bound no real forecast can
beat) and the do-nothing Hold baseline from a grid of stations in every month
of the test year. Where even perfect knowledge fails, no forecaster or
controller will help; that is a launch-window question, not a modelling one.

Usage:  python experiments/feasibility_map.py --config configs/experiment.yaml
"""
from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

from stratoballoon import viz
from stratoballoon.control import MPC, Hold
from stratoballoon.experiment import Context
from stratoballoon.forecasting.provider import TruthProvider
from stratoballoon.runlog import Run
from stratoballoon.simulation import Missions, fly, mission_metrics

log = logging.getLogger("feasibility")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--per_cell", type=int, default=6, help="missions per station and month")
    a = ap.parse_args()
    ctx = Context(a.config)
    out = ctx.out / "feasibility"
    run = Run(out, ctx.cfg, ctx.paths)
    A, band, rng = ctx.A, ctx.band, np.random.default_rng(ctx.seed)
    dur = float(ctx.cfg["mission_hours"])
    lats, lons = np.arange(10, 31, 5.0), np.arange(70, 91, 5.0)
    t = pd.DatetimeIndex(A.times)
    rows = []
    for month in range(1, 13):
        ok = np.where((t.year == ctx.test_year) & (t.month == month) & (t.hour % 6 == 0)
                      & (t <= ctx.test_end - pd.Timedelta(hours=dur)))[0]
        if len(ok) == 0:
            continue
        st, la, lo, al = [], [], [], []
        for y in lats:
            for x in lons:
                st += list(rng.choice(ok, a.per_cell))
                la += [y] * a.per_cell
                lo += [x] * a.per_cell
                al += list(rng.uniform(band.alt_min, band.alt_max, a.per_cell))
        m = Missions(A.hours[np.array(st)], np.array(la), np.array(lo), np.array(al), dur,
                     ctx.seed)
        truth = TruthProvider(A)
        for name, c in (("hold", Hold()), ("mpc / perfect forecast", MPC(band, ctx.rates))):
            r = mission_metrics(fly(A, m, c, truth, ctx.params, seed=ctx.seed))
            r["month"], r["lat"], r["lon"], r["controller"] = month, m.target_lat, m.target_lon, name
            rows.append(r)
        log.info("month %2d done", month)
    res = pd.concat(rows)
    res.to_csv(out / "missions.csv", index=False)
    by_month = res.pivot_table(index="month", columns="controller", values="twr50")
    by_month.to_csv(out / "twr50_by_month.csv")
    print(by_month.round(3))
    figure(res, out, band)
    run.finish()
    return 0


def figure(res, out, band):
    import matplotlib.pyplot as plt
    viz.apply_style()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.3, 1]})
    for i, (c, g) in enumerate(res.groupby("controller", sort=False)):
        mm = g.groupby("month").twr50.mean()
        a1.plot(mm.index, mm.values * 100, marker="o", color=[viz.C2, viz.C1][i], label=c)
    a1.set_xticks(range(1, 13))
    a1.set_xticklabels(list("JFMAMJJASOND"))
    a1.set_ylabel("Time within 50 km (%)")
    a1.set_title("Upper bound on station keeping by month")
    a1.legend(loc="upper left")
    g = res[res.controller == "mpc / perfect forecast"].groupby(["lat", "lon"]).twr50.mean()
    piv = g.unstack()
    im = a2.imshow(piv.values * 100, origin="lower", cmap=viz.seq_cmap(), aspect="auto",
                   extent=[piv.columns.min() - 2.5, piv.columns.max() + 2.5,
                           piv.index.min() - 2.5, piv.index.max() + 2.5])
    fig.colorbar(im, ax=a2, label="Time within 50 km (%)")
    a2.set_xlabel("Station longitude")
    a2.set_ylabel("Station latitude")
    a2.set_title("Perfect-forecast MPC, all months")
    a2.grid(False)
    viz.finish(fig, out / "feasibility.png",
               viz.source_note("sim", f"Altitude band {band.alt_min / 1000:.1f}-"
                                      f"{band.alt_max / 1000:.1f} km; ERA5 winds"))


if __name__ == "__main__":
    raise SystemExit(main())
