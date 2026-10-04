"""Experiment: what does each onboard component cost to run?

Measures single-balloon latency (one CPU thread), peak Python memory and
serialised size for every component that could run onboard: the forecaster for
one column, the Kalman filter update, the MPC decision (deterministic and
uncertainty-aware), and the anomaly detectors. Also sizes the forecast field
the ground would need to uplink.

Numbers are measured on THIS development machine (x86 desktop), not on flight
hardware. They bound relative costs; absolute figures on an ARM flight computer
must be measured there. The report says so in every row.

Usage:  python experiments/edge_profile.py --config configs/experiment.yaml
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse  # noqa: E402
import logging  # noqa: E402
import pickle  # noqa: E402
import platform  # noqa: E402
import time  # noqa: E402
import tracemalloc  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from stratoballoon import anomaly as AD  # noqa: E402
from stratoballoon.control import MPC, Observation  # noqa: E402
from stratoballoon.estimation import WindBiasKF  # noqa: E402
from stratoballoon.experiment import Context  # noqa: E402
from stratoballoon.forecasting import features as F  # noqa: E402
from stratoballoon.runlog import Run  # noqa: E402
from stratoballoon.simulation import Missions  # noqa: E402


def timed(fn, n=50, warm=5):
    for _ in range(warm):
        fn()
    tracemalloc.start()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e3)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return float(np.median(ts)), float(np.percentile(ts, 95)), peak / 1e6


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ctx = Context(ap.parse_args().config)
    out = ctx.out / "edge"
    run = Run(out, ctx.cfg, ctx.paths)
    A = ctx.A
    rows = []
    machine = f"{platform.processor() or platform.machine()} (development PC, 1 thread)"

    def add(component, where, fn, size_bytes=None, n=50, note=""):
        p50, p95, mem = timed(fn, n)
        rows.append({"component": component, "proposed_location": where, "p50_ms": p50,
                     "p95_ms": p95, "peak_python_mb": mem,
                     "size_kb": None if size_bytes is None else size_bytes / 1024,
                     "measured_on": machine, "note": note})

    # forecasters for a single column
    yi, xi = np.array([15]), np.array([20])
    ti = np.array([len(A.times) - 200])
    s = F.build_samples(A, ti, yi, xi, ctx.n_hist, [int(h // A.step_hours) for h in ctx.horizons])
    for name in dict.fromkeys(("ridge", "gradient_boosting", ctx.best_model())):
        try:
            m = ctx.model(name)
        except FileNotFoundError:
            continue
        add(f"forecast one column ({name})", "ground (or onboard fallback)",
            lambda m=m: m.predict(s, A, [1]), len(pickle.dumps(m)))

    # Kalman filter: one predict + GPS + baro update
    kf = WindBiasKF(np.array([20.0]), np.array([80.0]), np.array([20.0]), np.array([80.0]),
                    np.array([22_000.0]))

    def kf_step():
        kf.predict(np.array([5.0]), np.array([1.0]), 600.0)
        kf.update_gps(np.array([20.0]), np.array([80.0]), np.array([22_000.0]), np.array([True]))
        kf.update_baro(np.array([22_000.0]))
    add("Kalman filter step", "onboard", kf_step, kf.x.nbytes + kf.P.nbytes)

    # MPC decision for one balloon against a small forecast field
    issue = ctx.issue_indices()[:20]
    prov = ctx.provider(ctx.best_model(), issue)
    hrs = np.array([A.hours[issue[5]]])
    obs = Observation(hrs, np.array([20.0]), np.array([80.0]), np.array([22_000.0]),
                      np.array([2000.0]))
    mis = Missions(hrs, np.array([20.0]), np.array([80.0]), np.array([22_000.0]), 72)
    det = MPC(ctx.band, ctx.rates)
    det.reset(1, np.array([22_000.0]))
    add(f"MPC decision, point forecast ({len(det.plans)} plans)", "onboard",
        lambda: det.decide(obs, mis, prov), n=20)
    rob = MPC(ctx.band, ctx.rates, scenarios=8, sampler=ctx.sampler(ctx.best_model()))
    rob.reset(1, np.array([22_000.0]))
    add(f"MPC decision, + best {rob.shortlist} plans x {rob.scenarios} scenarios", "onboard",
        lambda: rob.decide(obs, mis, prov), n=10)

    # anomaly detectors: per-minute update cost for one balloon
    rng = np.random.default_rng(0)
    tel = AD.simulate(4, 1, rng, inject=False)
    one = AD.Telemetry(tel.data[:1], tel.sun[:1], tel.label[:1], tel.events)
    for D in (AD.CusumDetector, AD.AutoencoderDetector, AD.IsolationForestDetector):
        d = D().fit(tel)
        add(f"anomaly: {d.name} (one day of telemetry)", "onboard",
            lambda d=d: d.score(one), len(pickle.dumps(d)), n=5,
            note="cost of scoring 1,440 minutes")

    # forecast field to uplink for one balloon: 1,000 km box, all levels and leads
    L, leads = len(A.levels), len(ctx.horizons) + 1
    cells = 21 * 21
    uplink = cells * L * leads * 2 * 2      # float16 u and v
    rows.append({"component": "forecast uplink per cycle (21 x 21 cells, float16)",
                 "proposed_location": "ground -> balloon", "p50_ms": None, "p95_ms": None,
                 "peak_python_mb": None, "size_kb": uplink / 1024, "measured_on": "computed",
                 "note": f"{L} levels x {leads} lead times; every 6 h"})
    df = pd.DataFrame(rows)
    df.to_csv(out / "edge_profile.csv", index=False)
    pd.set_option("display.width", 200)
    print(df.drop(columns=["measured_on"]).round(3).to_string(index=False))
    run.finish(machine=machine)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
