"""Experiment: which telemetry anomaly detector is worth flying?

Trains each detector on fault-free telemetry, sets every detector's threshold to
the same false-alarm budget on separate fault-free telemetry, then measures
recall and detection latency per fault type on telemetry with injected faults.

Usage:  python experiments/anomaly_eval.py [--missions 60 --days 5]
"""
from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

from stratoballoon import anomaly as AD
from stratoballoon import viz
from stratoballoon.config import ROOT
from stratoballoon.runlog import Run

log = logging.getLogger("anomaly")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--missions", type=int, default=60)
    ap.add_argument("--days", type=float, default=5)
    ap.add_argument("--false_alarms_per_day", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/anomaly")
    a = ap.parse_args()
    out = ROOT / a.out
    run = Run(out, vars(a), [])
    rng = np.random.default_rng(a.seed)
    train = AD.simulate(a.missions, a.days, rng, inject=False)
    calib = AD.simulate(a.missions, a.days, rng, inject=False)
    test = AD.simulate(a.missions, a.days, rng, inject=True, fault_rate_per_day=0.6)
    log.info("%d injected faults in test telemetry", len(test.events))

    rows, per_fault = [], []
    for D in AD.DETECTORS:
        det = D().fit(train)
        thr = AD.threshold_for_rate(det.score(calib), a.false_alarms_per_day)
        s = det.score(test)
        r = AD.evaluate(s, test, thr)
        rows.append({"detector": det.name, **r})
        for f in AD.FAULTS:
            sub = AD.Telemetry(test.data, test.sun, test.label,
                               test.events[test.events.fault == f])
            if len(sub.events):
                rf = AD.evaluate(s, sub, thr)
                per_fault.append({"detector": det.name, "fault": f, "n": len(sub.events),
                                  "recall": rf["recall"],
                                  "median_latency_min": rf["median_latency_min"]})
        log.info("%-18s recall %.2f  latency %.0f min  false alarms/day %.2f", det.name,
                 r["recall"], r["median_latency_min"], r["false_alarms_per_day"])
    summ, pf = pd.DataFrame(rows), pd.DataFrame(per_fault)
    summ.to_csv(out / "summary.csv", index=False)
    pf.to_csv(out / "per_fault.csv", index=False)
    print(summ.round(3).to_string(index=False))
    print(pf.pivot(index="fault", columns="detector", values="recall").round(2))
    figure(pf, summ, out, a)
    run.finish(n_events=len(test.events))
    return 0


def figure(pf, summ, out, a):
    import matplotlib.pyplot as plt
    viz.apply_style()
    piv = pf.pivot(index="fault", columns="detector", values="recall")
    piv = piv[[d for d in summ.detector if d in piv.columns]]
    fig, ax = plt.subplots(figsize=(11, 4.8))
    y = np.arange(len(piv))
    w = 0.8 / piv.shape[1]
    for i, d in enumerate(piv.columns):
        ax.barh(y + i * w, piv[d] * 100, height=w * 0.95, color=viz.SERIES[i % 5], label=d)
    ax.set_yticks(y + w * (piv.shape[1] - 1) / 2)
    ax.set_yticklabels(piv.index)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Faults detected (%)")
    ax.set_title(f"Recall per fault at an equal budget of {a.false_alarms_per_day:g} false "
                 "alarms per mission-day")
    ax.legend(loc="lower right", fontsize=8)
    viz.finish(fig, out / "anomaly_recall.png",
               viz.source_note("sim", "Synthetic telemetry with injected faults"))


if __name__ == "__main__":
    raise SystemExit(main())
