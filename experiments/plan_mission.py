"""Plan one mission: altitude strategy, trajectory fan, success probability, risk.

Usage:
    python experiments/plan_mission.py --config configs/experiment.yaml \\
        --mission configs/mission_example.yaml
"""
from __future__ import annotations

import argparse
import json
import logging

import numpy as np

from stratoballoon import viz
from stratoballoon.config import load_yaml
from stratoballoon.experiment import Context
from stratoballoon.planner import MissionRequest, plan
from stratoballoon.runlog import Run


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--mission", required=True)
    ap.add_argument("--scenarios", type=int, default=64)
    a = ap.parse_args()
    ctx = Context(a.config)
    req = MissionRequest(**load_yaml(a.mission)["mission"])
    out = ctx.out / "planner"
    run = Run(out, {"experiment": ctx.cfg, "mission": req.__dict__}, ctx.paths)
    best = ctx.best_model()
    t = np.datetime64(req.launch_time)
    issue = ctx.issue_indices(start=t - np.timedelta64(2, "D"), end=t + np.timedelta64(
        int(req.duration_h) + 24, "h"))
    if len(issue) == 0:
        raise SystemExit(f"launch time {req.launch_time} is outside the data this config loads "
                         f"({str(ctx.A.times[0])[:10]} to {str(ctx.A.times[-1])[:10]})")
    prov = ctx.provider(best, issue)
    res = plan(ctx, req, prov, ctx.sampler(best), a.scenarios, ctx.seed)
    report = {k: v for k, v in res.items() if k not in ("tracks", "altitude",
                                                        "recommended_altitude_m")}
    hrs = np.arange(len(res["recommended_altitude_m"])) * 0.5
    report["recommended_altitude_km_every_6h"] = [
        round(float(x) / 1000, 2) for x in res["recommended_altitude_m"][::12]]
    report["forecast_model"] = best
    print(json.dumps(report, indent=1))
    (out / "plan.json").write_text(json.dumps(report, indent=1))
    figure(res, req, hrs, out)
    run.finish()
    return 0


def figure(res, req, hrs, out):
    import matplotlib.pyplot as plt
    viz.apply_style()
    la, lo = res["tracks"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    for i in range(len(la)):
        a1.plot(lo[i], la[i], color=viz.C1, alpha=0.15, lw=1)
    m = res["median_scenario"]
    a1.plot(lo[m], la[m], color=viz.C1, lw=2.4, label="median scenario")
    a1.plot(req.launch_lon, req.launch_lat, marker="o", ms=9, color=viz.INK, label="launch")
    a1.plot(req.target_lon, req.target_lat, marker="*", ms=15, color=viz.C2, label="target")
    a1.set_xlabel("Longitude")
    a1.set_ylabel("Latitude")
    a1.set_title(f"Planned trajectory fan; P(success) = {res['p_success_end_within_radius']:.0%}")
    a1.legend(loc="best")
    alt = res["altitude"]
    a2.fill_between(hrs, np.percentile(alt, 10, 0) / 1000, np.percentile(alt, 90, 0) / 1000,
                    color=viz.C1, alpha=0.2, label="10-90% of scenarios")
    a2.step(hrs, res["recommended_altitude_m"] / 1000, where="post", color=viz.INK,
            label="recommended (median scenario)")
    a2.set_xlabel("Hours after launch")
    a2.set_ylabel("Altitude (km)")
    a2.set_title("Recommended altitude strategy")
    a2.legend(loc="best")
    viz.finish(fig, out / "plan.png",
               viz.source_note("sim", "Futures are the forecast plus sampled forecast errors"))


if __name__ == "__main__":
    raise SystemExit(main())
