"""Experiment: are the forecast uncertainty estimates honest?

Calibrates conformal regions on the validation year and measures, on the test
year, how often the truth falls inside them (coverage) and how large they are
(sharpness). Coverage is also broken down by season and by wind strength,
because split conformal only promises coverage on average. Finally scores the
scenario ensemble with CRPS against the point forecast's absolute error.

Usage:  python experiments/uncertainty_eval.py --config configs/experiment.yaml
"""
from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

from stratoballoon import viz
from stratoballoon.experiment import Context
from stratoballoon.runlog import Run
from stratoballoon.uncertainty import crps_ensemble

log = logging.getLogger("uncertainty")
SEASONS = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM", 6: "JJAS",
           7: "JJAS", 8: "JJAS", 9: "JJAS", 10: "ON", 11: "ON"}


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ctx = Context(ap.parse_args().config)
    out = ctx.out / "uncertainty"
    run = Run(out, ctx.cfg, ctx.paths)
    fd, y = ctx.forecast_dir, ctx.test_year
    truth = np.load(fd / f"test_truth_{y}.npy")
    t_idx = np.load(fd / f"test_tidx_{y}.npy")
    # t_idx indexes the ladder's full record, which is longer than the window
    # this experiment loads, so issue times come from the full time axis.
    from stratoballoon.atmosphere import open_era5
    full_times = open_era5(ctx.paths).time.values
    issue_time = pd.DatetimeIndex(full_times[t_idx])
    step_h = float((full_times[1] - full_times[0]) / np.timedelta64(1, "h"))
    levels = [int(x) for x in ctx.A.levels]

    rows = []
    for m in ctx.forecast_models:
        p = fd / f"test_pred_{m}_{y}.npy"
        if not p.exists():
            log.warning("no test predictions for %s; skipped", m)
            continue
        resid = truth - np.load(p)
        for r in ctx.conformal(m).evaluate(resid):
            r.update(model=m, horizon_h=ctx.horizons[r.pop("h_idx")],
                     level_hpa=levels[r.pop("l_idx")])
            rows.append(r)
    cov = pd.DataFrame(rows)
    cov.to_csv(out / "coverage.csv", index=False)
    print("\nDisc coverage at nominal 90%, mean over levels\n",
          cov[cov.nominal == 0.9].pivot_table(index="model", columns="horizon_h",
                                              values="disc_coverage").round(3))

    # ---- conditional coverage of the chosen model at 90%
    best = ctx.best_model()
    resid = truth - np.load(fd / f"test_pred_{best}_{y}.npy")
    conf = ctx.conformal(best)
    r90 = conf.radius(0.9)
    inside = np.linalg.norm(resid, axis=-1) <= r90[None]
    month = issue_time.month
    season = np.array([SEASONS[m] for m in month])
    speed = np.linalg.norm(truth, axis=-1)
    tercile = np.digitize(speed, np.quantile(speed, [1 / 3, 2 / 3]))
    cond = []
    for k, h in enumerate(ctx.horizons):
        for s in ("DJF", "MAM", "JJAS", "ON"):
            cond.append({"horizon_h": h, "group": "season", "value": s,
                         "coverage": float(inside[season == s, k].mean())})
        for t, lab in enumerate(("weak", "moderate", "strong")):
            cond.append({"horizon_h": h, "group": "wind", "value": lab,
                         "coverage": float(inside[:, k][tercile[:, k] == t].mean())})
    cond = pd.DataFrame(cond)
    cond["method"] = "marginal"

    # ---- regime-conditional (Mondrian) conformal, grouped by wind at issue time
    vc, tc = fd / f"val_current_{y}.npy", fd / f"test_current_{y}.npy"
    if vc.exists() and tc.exists():
        from stratoballoon.uncertainty import RegimeConformal
        val_res = np.load(fd / f"val_residual_{best}_{y}.npy")
        v_speed = np.linalg.norm(np.load(vc), axis=-1)
        t_speed = np.linalg.norm(np.load(tc), axis=-1)
        rc = RegimeConformal().fit(val_res, v_speed)
        r_reg = rc.radius(0.9, t_speed)                                   # (N, H, L)
        inside_r = np.linalg.norm(resid, axis=-1) <= r_reg
        issue_terc = np.digitize(t_speed, np.quantile(t_speed, [1 / 3, 2 / 3]))   # (N, L)

        # adaptive conformal: recalibrate online as forecasts verify, with the
        # verification delay of each lead time respected
        from stratoballoon.uncertainty import adaptive_coverage
        ut = np.unique(t_idx)
        stride_h = float(np.median(np.diff(ut))) * step_h
        test_norm = np.linalg.norm(resid, axis=-1)
        inside_a = np.zeros_like(inside)
        for k, h in enumerate(ctx.horizons):
            lag = int(np.ceil(h / stride_h))
            for l in range(inside.shape[2]):
                inside_a[:, k, l] = adaptive_coverage(test_norm[:, k, l], t_idx,
                                                      conf.norm[:, k, l], 0.9, lag=lag)
        rows_r = []
        for k, h in enumerate(ctx.horizons):
            for meth, ins, rad in (("marginal", inside, np.broadcast_to(r90[None], inside.shape)),
                                   ("regime", inside_r, r_reg),
                                   ("adaptive", inside_a, np.full(inside.shape, np.nan))):
                rows_r.append({"horizon_h": h, "method": meth, "group": "all",
                               "coverage": float(ins[:, k].mean()),
                               "mean_radius_ms": float(rad[:, k].mean())})
                for t, lab in enumerate(("weak now", "moderate now", "strong now")):
                    m_ = issue_terc == t
                    rows_r.append({"horizon_h": h, "method": meth, "group": lab,
                                   "coverage": float(ins[:, k][m_].mean()),
                                   "mean_radius_ms": float(rad[:, k][m_].mean())})
        reg = pd.DataFrame(rows_r)
        reg.to_csv(out / "regime_conformal.csv", index=False)
        print("\nMarginal vs regime-conditional conformal, 90% nominal, by wind at issue time\n",
              reg.pivot_table(index=["method", "group"], columns="horizon_h",
                              values="coverage").round(3))
    cond.to_csv(out / "conditional_coverage.csv", index=False)

    # ---- ensemble CRPS vs point MAE, per component
    rng = np.random.default_rng(ctx.seed)
    pick = rng.choice(len(truth), min(20_000, len(truth)), replace=False)
    pred = np.load(fd / f"test_pred_{best}_{y}.npy")[pick]
    mem = pred[None] + conf.sample(50, rng)[:, None]                        # (M, n, H, L, 2)
    crps_rows = []
    for k, h in enumerate(ctx.horizons):
        for l, lev in enumerate(levels):
            for c, comp in enumerate("uv"):
                crps = crps_ensemble(mem[:, :, k, l, c], truth[pick, k, l, c]).mean()
                mae = np.abs(pred[:, k, l, c] - truth[pick, k, l, c]).mean()
                crps_rows.append({"horizon_h": h, "level_hpa": lev, "component": comp,
                                  "crps": float(crps), "point_mae": float(mae)})
    pd.DataFrame(crps_rows).to_csv(out / "crps.csv", index=False)

    figures(cov, cond, best, out)
    run.finish(best_model=best)
    return 0


def figures(cov, cond, best, out):
    import matplotlib.pyplot as plt
    viz.apply_style()
    models = list(cov.model.unique())
    fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 4.0), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, models):
        g = cov[cov.model == m].groupby(["horizon_h", "nominal"]).disc_coverage.mean().reset_index()
        for i, (h, gh) in enumerate(g.groupby("horizon_h")):
            ax.plot(gh.nominal, gh.disc_coverage, marker="o", color=viz.SERIES[i % 5],
                    label=f"{h} h")
        ax.plot([0.45, 1], [0.45, 1], color=viz.MUTED, ls="--", lw=1.2)
        ax.set_title(m.replace("_", " "))
        ax.set_xlabel("Nominal coverage")
    axes[0].set_ylabel("Observed coverage on the test year")
    axes[-1].legend(title="lead", loc="lower right")
    fig.suptitle("Reliability of conformal wind regions: on the diagonal is honest",
                 x=0.005, ha="left", fontsize=11, fontweight="semibold")
    viz.finish(fig, out / "reliability.png", viz.source_note("model"))

    fig, ax = plt.subplots(figsize=(9, 4))
    g = cond[cond.horizon_h == cond.horizon_h.min()]
    ax.bar(g.value, g.coverage, color=[viz.C1] * 4 + [viz.C3] * 3, edgecolor=viz.SURFACE,
           linewidth=2)
    ax.axhline(0.9, color=viz.INK, ls="--", lw=1.2)
    for x, v in zip(g.value, g.coverage):
        ax.text(x, v, f"{v:.0%}", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Coverage of the 90% region")
    ax.set_title(f"Coverage is guaranteed on average, not in every regime ({best}, "
                 f"{g.horizon_h.iloc[0]} h lead)")
    viz.finish(fig, out / "conditional_coverage.png", viz.source_note("model"))


if __name__ == "__main__":
    raise SystemExit(main())
