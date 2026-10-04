"""Experiment: how well can we predict where a drifting balloon will be?

From many start times and places in the test year, a balloon holding a fixed
altitude is flown through the real atmosphere (the truth track) and, separately,
through each forecast (the predicted track), with no forecast updates after
launch. Position error is measured at 6-72 h. The scenario ensemble gives an
uncertainty cone, whose coverage is checked, and a probability of still being
within 200 km of the launch point, scored with the Brier score.

Usage:  python experiments/trajectory_eval.py --config configs/experiment.yaml
"""
from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

from stratoballoon import viz
from stratoballoon.atmosphere import distance_km, move
from stratoballoon.control import rollout
from stratoballoon.experiment import Context
from stratoballoon.runlog import Run

log = logging.getLogger("trajectory")
SUB_H = 0.5
LEADS = [6, 12, 24, 48, 72]


def truth_track(A, hours0, lat, lon, alt, n_steps):
    """Track through ERA5, plus whether it is still inside the data at each step."""
    lats, lons, ins = [], [], []
    la, lo = lat.copy(), lon.copy()
    inside = A.contains(la, lo)
    for j in range(n_steps):
        s = A.sample(hours0 + j * SUB_H, la, lo, alt)
        la, lo = move(la, lo, s["u"], s["v"], SUB_H * 3600)
        inside = inside & A.contains(la, lo)
        lats.append(la)
        lons.append(lo)
        ins.append(inside.copy())
    return np.stack(lats, 1), np.stack(lons, 1), np.stack(ins, 1)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ctx = Context(ap.parse_args().config)
    out = ctx.out / "trajectory"
    run = Run(out, ctx.cfg, ctx.paths)
    A, rng = ctx.A, np.random.default_rng(ctx.seed)
    n = int(ctx.cfg["trajectory_starts"])
    n_steps = int(max(LEADS) / SUB_H)

    issues_all = ctx.issue_indices(end=ctx.test_end - pd.Timedelta(hours=max(LEADS)))
    issues_all = issues_all[A.times[issues_all] >= np.datetime64(ctx.test_start)]
    start_idx = np.sort(rng.choice(issues_all, n))
    uniq = np.unique(start_idx)
    lat0, lon0 = rng.uniform(10, 30, n), rng.uniform(66, 94, n)
    alt0 = rng.choice(ctx.band.grid, n)
    hours0 = A.hours[start_idx]
    t_la, t_lo, t_in = truth_track(A, hours0, lat0, lon0, alt0, n_steps)
    lead_cols = [int(L / SUB_H) - 1 for L in LEADS]

    best = ctx.best_model()
    sources = ["persistence"] + [m for m in ctx.forecast_models
                                 if m not in ("persistence",)]
    rows, tracks = [], {"truth": (t_la, t_lo), "truth_inside": t_in}
    for src in ["perfect"] + sources:
        try:
            prov = ctx.provider(src, uniq)
        except FileNotFoundError:
            log.warning("no fitted %s; skipped", src)
            continue
        k = np.searchsorted(uniq, start_idx) if src != "perfect" else np.zeros(n, int)
        path = np.repeat(alt0[:, None], n_steps, 1)
        p_la, p_lo = rollout(prov, k, hours0, lat0, lon0, path, SUB_H)
        tracks[src] = (p_la, p_lo)
        err = distance_km(p_la, p_lo, t_la, t_lo)
        # Score only while both tracks are inside the data domain: outside it
        # there is no wind to compare against.
        valid = t_in & A.contains(p_la, p_lo)
        for L, c in zip(LEADS, lead_cols):
            e = err[valid[:, c], c]
            rows.append({"source": src, "lead_h": L, "n_scored": int(len(e)),
                         "median_km": float(np.median(e)) if len(e) else np.nan,
                         "mean_km": float(e.mean()) if len(e) else np.nan,
                         "p90_km": float(np.percentile(e, 90)) if len(e) else np.nan})
        log.info("%-18s 24 h median error %.0f km", src, np.median(err[:, lead_cols[2]]))
    errs = pd.DataFrame(rows)
    errs.to_csv(out / "position_error.csv", index=False)

    # ---- ensemble from the best model
    M = int(ctx.cfg["trajectory_members"])
    prov = ctx.provider(best, uniq)
    k = np.repeat(np.searchsorted(uniq, start_idx), M)
    offs = ctx.sampler(best)(n * M, rng)
    path = np.repeat(alt0[:, None], n_steps, 1).repeat(M, 0)
    e_la, e_lo = rollout(prov, k, hours0.repeat(M), lat0.repeat(M), lon0.repeat(M), path,
                         SUB_H, offs)
    e_la, e_lo = e_la.reshape(n, M, -1), e_lo.reshape(n, M, -1)
    cone, brier = [], []
    for L, c in zip(LEADS, lead_cols):
        ok = t_in[:, c]
        m_la, m_lo = e_la[:, :, c].mean(1), e_lo[:, :, c].mean(1)
        spread = distance_km(e_la[:, :, c], e_lo[:, :, c], m_la[:, None], m_lo[:, None])
        miss = distance_km(t_la[:, c], t_lo[:, c], m_la, m_lo)
        for q in (0.5, 0.8, 0.95):
            r = np.quantile(spread, q, axis=1)
            cone.append({"lead_h": L, "nominal": q, "n_scored": int(ok.sum()),
                         "coverage": float((miss <= r)[ok].mean()),
                         "median_radius_km": float(np.median(r[ok]))})
        # event: still within 200 km of launch
        # Leaving the domain is far more than 200 km from launch here, so the
        # event is well defined even for tracks that left the data.
        obs = distance_km(t_la[:, c], t_lo[:, c], lat0, lon0) <= 200
        p_ens = (distance_km(e_la[:, :, c], e_lo[:, :, c], lat0[:, None], lon0[:, None])
                 <= 200).mean(1)
        p_det = (distance_km(*[a[:, c] for a in tracks[best]], lat0, lon0) <= 200).astype(float)
        clim = obs.mean()
        brier.append({"lead_h": L, "event_rate": float(clim),
                      "brier_ensemble": float(np.mean((p_ens - obs) ** 2)),
                      "brier_deterministic": float(np.mean((p_det - obs) ** 2)),
                      "brier_climatology": float(np.mean((clim - obs) ** 2))})
    pd.DataFrame(cone).to_csv(out / "cone_coverage.csv", index=False)
    pd.DataFrame(brier).to_csv(out / "region_probability.csv", index=False)
    print("\nMedian position error (km)\n",
          errs.pivot(index="source", columns="lead_h", values="median_km").round(0))
    print("\nCone coverage\n", pd.DataFrame(cone).pivot(index="nominal", columns="lead_h",
                                                       values="coverage").round(2))
    print("\nBrier scores\n", pd.DataFrame(brier).round(3))

    figures(ctx, errs, tracks, (e_la, e_lo), start_idx, lat0, lon0, alt0, best, out)
    run.finish(best_model=best, n_starts=n)
    return 0


def figures(ctx, errs, tracks, ens, start_idx, lat0, lon0, alt0, best, out):
    import matplotlib.pyplot as plt
    viz.apply_style()
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    order = [s for s in ("perfect", "persistence", "climatology", "ridge", "gradient_boosting", "lstm", "tcn")
             if s in errs.source.unique()]
    for i, s in enumerate(order):
        g = errs[errs.source == s]
        ax.plot(g.lead_h, g.median_km, marker="o", color=viz.SERIES[i % 5] if s != "perfect"
                else viz.MUTED, label=s.replace("_", " "))
    ax.set_xticks(LEADS)
    ax.set_xlabel("Hours after launch (no forecast updates)")
    ax.set_ylabel("Median position error (km)")
    ax.set_title("Trajectory prediction error grows with lead time")
    ax.legend(loc="upper left")
    viz.finish(fig, out / "position_error.png",
               viz.source_note("model", "Fixed-altitude drift, truth flown through ERA5"))

    # one example with its cone and the wind it flew through
    e_la, e_lo = ens
    t_la, t_lo = tracks["truth"]
    stayed = np.where(tracks["truth_inside"][:, -1])[0]
    d_end = distance_km(t_la[stayed, -1], t_lo[stayed, -1], lat0[stayed], lon0[stayed])
    i = int(stayed[np.argsort(d_end)[len(stayed) // 2]])
    A = ctx.A
    ti = start_idx[i]
    fig, ax = plt.subplots(figsize=(8.4, 6.2))
    for m in range(e_la.shape[1]):
        ax.plot(e_lo[i, m], e_la[i, m], color=viz.C1, alpha=0.18, lw=1)
    ax.plot(t_lo[i], t_la[i], color=viz.INK, lw=2.4, label="actual (flown through ERA5)")
    ax.plot(tracks[best][1][i], tracks[best][0][i], color=viz.C1, lw=2.2,
            label=f"predicted ({best})")
    if "persistence" in tracks:
        ax.plot(tracks["persistence"][1][i], tracks["persistence"][0][i], color=viz.C2, lw=1.8,
                label="predicted (persistence)")
    lo_min, lo_max = ax.get_xlim()
    la_min, la_max = ax.get_ylim()
    ys = (A.lats >= la_min - 1) & (A.lats <= la_max + 1)
    xs = (A.lons >= lo_min - 1) & (A.lons <= lo_max + 1)
    lev = int(np.argmin(np.abs(A.h[ti, :, 15, 20] - alt0[i])))
    X, Y = np.meshgrid(A.lons[xs], A.lats[ys])
    ax.quiver(X, Y, A.u[ti, lev][np.ix_(ys, xs)], A.v[ti, lev][np.ix_(ys, xs)],
              color=viz.MUTED, alpha=0.6, width=0.0025)
    ax.plot(lon0[i], lat0[i], marker="*", ms=14, color=viz.INK)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"72 h drift at {alt0[i] / 1000:.1f} km from "
                 f"{str(A.times[ti])[:13]}: thin lines are ensemble members")
    ax.legend(loc="best")
    ax.set_aspect(1 / np.cos(np.radians(np.mean(t_la[i]))), adjustable="datalim")
    viz.finish(fig, out / "example_cone.png",
               viz.source_note("model", "Arrows: ERA5 wind at launch, nearest level"))


if __name__ == "__main__":
    raise SystemExit(main())
