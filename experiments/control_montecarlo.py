"""Experiment: does altitude control keep a balloon on station, and which kind?

Every controller flies the identical set of seeded missions (start time, station,
initial altitude) through the real atmosphere, seeing only noisy sensors and a
forecast. Every mission is reported; nothing is filtered. Differences between
controllers are paired, with bootstrap confidence intervals over missions
clustered by start week (missions in the same week share weather).

Usage:  python experiments/control_montecarlo.py --config configs/experiment.yaml
"""
from __future__ import annotations

import argparse
import logging
import time

import numpy as np
import pandas as pd

from stratoballoon import viz
from stratoballoon.control import MPC, Greedy, Hold
from stratoballoon.evaluation import paired_ci
from stratoballoon.experiment import Context
from stratoballoon.runlog import Run
from stratoballoon.simulation import fly, mission_metrics, sample_missions

log = logging.getLogger("control")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--missions", type=int, help="override the mission count")
    args = ap.parse_args()
    ctx = Context(args.config)
    out = ctx.out / "control"
    run = Run(out, ctx.cfg, ctx.paths)
    A, band, rates, params = ctx.A, ctx.band, ctx.rates, ctx.params
    n = args.missions or int(ctx.cfg["missions"])
    dur = float(ctx.cfg["mission_hours"])
    missions = sample_missions(A, n, str(ctx.test_start.date()), str(ctx.test_end.date()),
                               dur, band.alt_min, band.alt_max, ctx.seed)
    best = ctx.best_model()
    log.info("band %.0f-%.0f m, balloon V=%.0f m3, ballonet %.0f kg; best forecaster: %s",
             band.alt_min, band.alt_max, params.volume_m3, params.ballonet_max_kg, best)

    providers = {name: ctx.provider(name) for name in ("perfect", "persistence", best)}
    K = int(ctx.cfg["scenarios"])
    runs = [
        ("hold", lambda: Hold(), "persistence"),
        (f"greedy / {best}", lambda: Greedy(band, rates), best),
        ("mpc / persistence", lambda: MPC(band, rates), "persistence"),
        (f"mpc / {best}", lambda: MPC(band, rates), best),
        (f"mpc-robust / {best}", lambda: MPC(band, rates, scenarios=K,
                                             sampler=ctx.sampler(best), seed=ctx.seed), best),
        ("mpc / perfect forecast", lambda: MPC(band, rates), "perfect"),
    ]
    policy = ctx.out / "rl" / "ppo_policy.zip"
    if policy.exists():
        # trained by experiments/train_rl.py on the validation year only
        from stable_baselines3 import PPO
        from stratoballoon.rl import RLController
        ppo = PPO.load(policy, device="cpu")
        runs.append((f"rl-ppo / {best}", lambda: RLController(
            ppo, band, params, ctx.conformal(best), A, dur), best))
    results, traces = [], {}
    for label, make, src in runs:
        t0 = time.time()
        tr = fly(A, missions, make(), providers[src], params, seed=ctx.seed)
        m = mission_metrics(tr)
        m.insert(0, "mission", np.arange(n))
        m.insert(0, "run", label)
        results.append(m)
        traces[label] = tr
        log.info("%-24s %5.0fs  TWR50 %.3f  mean dist %.0f km  pump %.0f Wh", label,
                 time.time() - t0, m.twr50.mean(), m.mean_dist_km.mean(), m.pump_energy_wh.mean())
    res = pd.concat(results)
    start = pd.DatetimeIndex(A.times[0] + (missions.start_hours * 3600).astype("timedelta64[s]"))
    week = start.isocalendar().week.to_numpy()
    season = start.month.map({12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
                              6: "JJAS", 7: "JJAS", 8: "JJAS", 9: "JJAS", 10: "ON",
                              11: "ON"}).to_numpy()
    res["season"] = np.tile(season, len(runs))
    res["start_week"] = np.tile(week, len(runs))
    res.to_csv(out / "missions.csv", index=False)

    summary = []
    base = res[res.run == "hold"].set_index("mission")
    ref = res[res.run == f"mpc / {best}"].set_index("mission")
    for label, g in res.groupby("run", sort=False):
        g = g.set_index("mission")
        row = {"run": label, "twr50": g.twr50.mean(), "mean_dist_km": g.mean_dist_km.mean(),
               "median_dist_km": g.mean_dist_km.median(),
               "pump_energy_wh": g.pump_energy_wh.mean(), "target_changes": g.target_changes.mean(),
               "left_domain": g.left_domain.mean(), "battery_empty": (g.min_battery_wh <= 0).mean()}
        for name, other in (("vs_hold", base), ("vs_mpc_best", ref)):
            mu, lo, hi = paired_ci(g.twr50, other.twr50, week)
            row[f"twr50_{name}"], row[f"twr50_{name}_lo"], row[f"twr50_{name}_hi"] = mu, lo, hi
            mu, lo, hi = paired_ci(g.mean_dist_km, other.mean_dist_km, week)
            row[f"dist_{name}"], row[f"dist_{name}_lo"], row[f"dist_{name}_hi"] = mu, lo, hi
        summary.append(row)
    summary = pd.DataFrame(summary)
    summary.to_csv(out / "summary.csv", index=False)
    by_season = res.pivot_table(index="run", columns="season", values="twr50", sort=False)
    by_season.to_csv(out / "twr50_by_season.csv")
    pd.set_option("display.width", 200)
    print("\n", summary[["run", "twr50", "mean_dist_km", "pump_energy_wh", "target_changes",
                         "twr50_vs_hold", "twr50_vs_hold_lo", "twr50_vs_hold_hi",
                         "dist_vs_mpc_best", "dist_vs_mpc_best_lo", "dist_vs_mpc_best_hi"]
                        ].round(3).to_string(index=False))
    print("\nTWR50 by season\n", by_season.round(3))

    figures(summary, res, traces, missions, ctx, best, out)
    run.finish(best_model=best, n_missions=n, balloon=params.__dict__,
               band=[band.alt_min, band.alt_max])
    return 0


def figures(summary, res, traces, missions, ctx, best, out):
    import matplotlib.pyplot as plt
    viz.apply_style()
    note = viz.source_note("sim", f"{len(missions)} identical missions per controller, "
                                  f"{ctx.cfg['mission_hours']} h each")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.2, 1]})
    y = np.arange(len(summary))
    err = np.vstack([summary.twr50_vs_hold - summary.twr50_vs_hold_lo,
                     summary.twr50_vs_hold_hi - summary.twr50_vs_hold])
    base = summary.loc[summary.run == "hold", "twr50"].iloc[0]
    a1.barh(y, summary.twr50 * 100, color=viz.C1, edgecolor=viz.SURFACE, linewidth=2, height=.6)
    a1.errorbar((base + summary.twr50_vs_hold) * 100, y, xerr=err * 100, fmt="none",
                ecolor=viz.INK, capsize=3, lw=1.2)
    right = np.maximum(summary.twr50, base + summary.twr50_vs_hold_hi) * 100
    for yi, v, r in zip(y, summary.twr50, right):
        a1.text(r, yi, f"  {v:.1%}", va="center", fontsize=9)     # clear of the error bar
    a1.set_xlim(0, right.max() * 1.12)
    a1.set_yticks(y)
    a1.set_yticklabels(summary.run)
    a1.invert_yaxis()
    a1.set_xlabel("Time within 50 km of station (%), 95% CI of the gain over hold")
    a1.set_title("Station keeping by controller")
    a1.grid(axis="y", visible=False)
    for i, (label, g) in enumerate(res.groupby("run", sort=False)):
        d = np.sort(g.mean_dist_km.to_numpy())
        a2.plot(d, np.linspace(0, 1, len(d)), color=viz.SERIES[i % 5] if "perfect" not in label
                else viz.MUTED, label=label)
    a2.set_xscale("log")
    a2.set_xlabel("Mean distance from station over the mission (km)")
    a2.set_ylabel("Fraction of missions")
    a2.set_title("Every mission, not just the good ones")
    a2.legend(loc="lower right", fontsize=8)
    viz.finish(fig, out / "controller_comparison.png", note)

    # one mission in detail: track, altitude and command, battery
    label = f"mpc / {best}"
    tr, th = traces[label], traces["hold"]
    # The median mission here is a failure for every controller, which shows
    # nothing about how control works, so the example is the 90th-percentile
    # mission by time on station, and is labelled as that.
    on_station = (tr["dist_km"] <= 50).mean(1)
    i = int(np.argsort(on_station)[int(0.9 * len(on_station))])
    hrs = np.arange(tr["alt"].shape[1]) * 0.5
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for t, c, lab in ((th, viz.C2, "hold"), (tr, viz.C1, label)):
        axes[0].plot(t["lon"][i], t["lat"][i], color=c, label=lab)
    axes[0].plot(missions.target_lon[i], missions.target_lat[i], marker="*", ms=14, color=viz.INK)
    axes[0].set_title("Track (a 90th-percentile mission, not typical)")
    axes[0].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")
    axes[0].legend(loc="best")
    axes[1].plot(hrs, tr["alt"][i] / 1000, color=viz.C1, label="altitude")
    axes[1].step(hrs, tr["target_alt"][i] / 1000, where="post", color=viz.INK2, lw=1.2,
                 ls="--", label="commanded")
    axes[1].set_xlabel("Hours")
    axes[1].set_ylabel("km")
    axes[1].set_title("Controller actions")
    axes[1].legend(loc="best")
    axes[2].plot(hrs, tr["dist_km"][i], color=viz.C1, label=label)
    axes[2].plot(hrs, th["dist_km"][i], color=viz.C2, label="hold")
    ax2 = axes[2].twinx()
    ax2.plot(hrs, tr["battery_wh"][i], color=viz.C3, lw=1.2)
    ax2.set_ylabel("Battery (Wh)", color=viz.C3)
    axes[2].axhline(50, color=viz.MUTED, ls=":", lw=1)
    axes[2].set_xlabel("Hours")
    axes[2].set_ylabel("Distance from station (km)")
    axes[2].set_title("Station-keeping error and energy")
    axes[2].legend(loc="upper left")
    viz.finish(fig, out / "example_mission.png", note)


if __name__ == "__main__":
    raise SystemExit(main())
