"""Experiment: what breaks the system, and does the state estimator help?

Flies the same seeded missions through every fault in the suite (GPS outage,
comms loss, doubled forecast error, an unforecast wind change, pump failure,
weak battery, barometer bias, wrong initial position), for several controller
configurations. Every mission is kept. Reports station keeping, energy, the
estimator's position error, forecast staleness and telemetry backlog per fault.

Usage:  python experiments/robustness.py --config configs/experiment.yaml
"""
from __future__ import annotations

import argparse
import logging
import time

import numpy as np
import pandas as pd

from stratoballoon import viz
from stratoballoon.control import MPC, Hold
from stratoballoon.experiment import Context
from stratoballoon.runlog import Run
from stratoballoon.scenarios import fault_suite
from stratoballoon.simulation import fly, mission_metrics, sample_missions

log = logging.getLogger("robustness")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--missions", type=int, default=300)
    args = ap.parse_args()
    ctx = Context(args.config)
    out = ctx.out / "robustness"
    run = Run(out, ctx.cfg, ctx.paths)
    A, band, rates, params = ctx.A, ctx.band, ctx.rates, ctx.params
    dur = float(ctx.cfg["mission_hours"])
    missions = sample_missions(A, args.missions, str(ctx.test_start.date()),
                               str(ctx.test_end.date()), dur, band.alt_min, band.alt_max,
                               ctx.seed + 1)
    best = ctx.best_model()
    prov = ctx.provider(best)
    configs = [
        ("hold", lambda: Hold(), dict()),
        (f"mpc / {best}, raw GPS", lambda: MPC(band, rates), dict()),
        (f"mpc / {best}, Kalman filter", lambda: MPC(band, rates), dict(estimator=True)),
        (f"mpc / {best}, filter + bias correction", lambda: MPC(band, rates),
         dict(estimator=True, bias_correction=True)),
    ]
    rows = []
    for faults in fault_suite(len(missions), dur, ctx.seed):
        for label, make, kw in configs:
            t0 = time.time()
            m = mission_metrics(fly(A, missions, make(), prov, params, faults=faults,
                                    seed=ctx.seed, **kw))
            m["fault"], m["config"] = faults.label, label
            m["mission"] = np.arange(len(m))
            rows.append(m)
            log.info("%-30s %-40s %4.0fs TWR50 %.3f est err %.1f km", faults.label, label,
                     time.time() - t0, m.twr50.mean(), m.est_err_mean_km.mean())
    res = pd.concat(rows)
    res.to_csv(out / "missions.csv", index=False)
    summ = res.groupby(["fault", "config"], sort=False).agg(
        twr50=("twr50", "mean"), mean_dist_km=("mean_dist_km", "mean"),
        pump_energy_wh=("pump_energy_wh", "mean"),
        battery_empty=("min_battery_wh", lambda x: float((x <= 0).mean())),
        est_err_mean_km=("est_err_mean_km", "mean"), est_err_max_km=("est_err_max_km", "mean"),
        max_forecast_age_h=("max_forecast_age_h", "mean"),
        max_backlog_min=("max_telemetry_backlog_min", "mean")).reset_index()
    summ.to_csv(out / "summary.csv", index=False)
    pd.set_option("display.width", 220)
    print(summ.round(3).to_string(index=False))
    figure(summ, out, len(missions), dur)
    run.finish(best_model=best, n_missions=len(missions))
    return 0


def figure(summ, out, n, dur):
    import matplotlib.pyplot as plt
    viz.apply_style()
    faults = list(summ.fault.unique())
    cfgs = list(summ.config.unique())
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.2))
    y = np.arange(len(faults))
    w = 0.8 / len(cfgs)
    for i, c in enumerate(cfgs):
        g = summ[summ.config == c].set_index("fault").loc[faults]
        a1.barh(y + i * w, g.twr50 * 100, height=w * 0.95, color=viz.SERIES[i % 5], label=c)
        a2.barh(y + i * w, g.est_err_max_km, height=w * 0.95, color=viz.SERIES[i % 5])
    for a in (a1, a2):
        a.set_yticks(y + w * (len(cfgs) - 1) / 2)
        a.set_yticklabels(faults)
        a.invert_yaxis()
        a.grid(axis="y", visible=False)
    a2.set_yticklabels([])
    a1.set_xlabel("Time within 50 km of station (%)")
    a1.set_title("Station keeping under each fault")
    a1.legend(loc="lower right", fontsize=8)
    a2.set_xscale("log")
    a2.set_xlabel("Worst position-estimate error in the mission (km, log)")
    a2.set_title("What the vehicle believes vs where it is")
    viz.finish(fig, out / "robustness.png",
               viz.source_note("sim", f"{n} identical missions per fault and configuration, "
                                      f"{dur:.0f} h each"))


if __name__ == "__main__":
    raise SystemExit(main())
