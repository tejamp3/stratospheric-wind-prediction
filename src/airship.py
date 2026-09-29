"""Turn forecast error into station-keeping performance for a stratospheric airship.

The physical argument is short. A buoyant platform at ~20 km holds station by
flying against the ambient wind. If the controller cancels the wind it *expects*,
whatever is left over is the forecast error, and that residual pushes the vehicle
off station. So a forecast error of e m/s sustained over H hours displaces the
airship by roughly e * H * 3.6 km. Better forecasts convert directly into
kilometres of station-keeping, which is the number an operator cares about.

Two views are produced:

1. **Error-to-drift distribution.** Every test window becomes one station-keeping
   trial, giving the distribution of drift at each lead time, for the LSTM and
   for persistence.
2. **Trajectory simulation.** A controller re-plans every 6 h and the resulting
   2-D track is integrated, comparing the LSTM, persistence, and perfect
   knowledge over a multi-day stretch.

A third scenario covers altitude selection: because 50/30/10 hPa often carry very
different winds, choosing the calmest available level is itself a control lever,
and one worth quantifying.

Usage:  python src/airship.py [--tag base] [--max-airspeed 12]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
import data as D
import metrics as M
import viz
from evaluate import features_of, input_steps_of, load_model, physical, predict

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("airship")

KM_PER_MS_PER_H = 3.6          # 1 m/s for 1 h = 3.6 km
DEFAULT_MAX_AIRSPEED = 12.0    # m/s, representative of a solar HAPS at 20 km


# ------------------------------------------------------- 1. error-to-drift view
def drift_table(pred: np.ndarray, true: np.ndarray, base: np.ndarray) -> pd.DataFrame:
    """Drift statistics per lead time, in km, for the model and for persistence.

    Two different quantities matter here and they behave differently. The
    per-window drift radius comes from the *magnitude* of the error and describes
    a single station-keeping attempt. Cumulative drift over many consecutive
    windows is instead governed by the *mean* error vector, because a bias adds
    up every step while random error partly cancels. A model can therefore look
    good on RMSE and still walk the vehicle off station, so the bias columns are
    reported alongside.
    """
    rows = []
    for k, h in enumerate(C.HORIZONS):
        r = {"horizon_h": h}
        for name, series in (("model", pred), ("persist", base)):
            du = series[:, k, 0] - true[:, k, 0]
            dv = series[:, k, 1] - true[:, k, 1]
            err = np.hypot(du, dv)
            d = err * h * KM_PER_MS_PER_H
            r[f"{name}_drift_mean_km"] = float(d.mean())
            r[f"{name}_drift_p50_km"] = float(np.percentile(d, 50))
            r[f"{name}_drift_p90_km"] = float(np.percentile(d, 90))
            r[f"{name}_drift_p95_km"] = float(np.percentile(d, 95))
            # Bias is what accumulates; report it as a drift rate in km/day.
            r[f"{name}_bias_u_ms"] = float(du.mean())
            r[f"{name}_bias_v_ms"] = float(dv.mean())
            r[f"{name}_bias_drift_km_per_day"] = float(
                np.hypot(du.mean(), dv.mean()) * 24 * KM_PER_MS_PER_H)
        r["drift_reduction_pct"] = float(
            (r["persist_drift_p90_km"] - r["model_drift_p90_km"])
            / r["persist_drift_p90_km"] * 100)
        rows.append(r)
    return pd.DataFrame(rows)


def hold_radius(pred, true, horizon_idx: int, horizon_h: int,
                pct: float = 90.0) -> float:
    """The radius that contains `pct` of station-keeping outcomes at this lead."""
    err = np.hypot(pred[:, horizon_idx, 0] - true[:, horizon_idx, 0],
                   pred[:, horizon_idx, 1] - true[:, horizon_idx, 1])
    return float(np.percentile(err * horizon_h * KM_PER_MS_PER_H, pct))


# -------------------------------------------------------- 2. trajectory sim
def simulate_track(wind_u: np.ndarray, wind_v: np.ndarray,
                   plan_u: np.ndarray, plan_v: np.ndarray,
                   max_airspeed: float,
                   step_h: int = C.STEP_HOURS) -> np.ndarray:
    """Integrate 2-D displacement under a counter-wind controller.

    The controller wants to fly at -(planned wind). Its airspeed is capped, so
    the commanded velocity is clipped to max_airspeed; the vehicle's ground
    velocity is then the true wind plus that command.

    Returns an (N+1, 2) array of east/north displacement in km.
    """
    cmd = np.stack([-plan_u, -plan_v], axis=1)
    mag = np.hypot(cmd[:, 0], cmd[:, 1])
    over = mag > max_airspeed
    cmd[over] *= (max_airspeed / mag[over])[:, None]

    ground = np.stack([wind_u, wind_v], axis=1) + cmd
    steps = ground * step_h * KM_PER_MS_PER_H
    return np.vstack([[0.0, 0.0], np.cumsum(steps, axis=0)])


def fig_drift_distribution(table: pd.DataFrame, pred, true, base):
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))

    x = np.arange(len(C.HORIZONS))
    w = 0.36
    a1.bar(x - w / 2, table.model_drift_p90_km, width=w * .94, color=viz.PREDICTED,
           edgecolor=viz.SURFACE, linewidth=2, label="LSTM forecast")
    a1.bar(x + w / 2, table.persist_drift_p90_km, width=w * .94, color=viz.BASELINE,
           edgecolor=viz.SURFACE, linewidth=2, label="Persistence")
    for i, (m, p) in enumerate(zip(table.model_drift_p90_km, table.persist_drift_p90_km)):
        a1.text(i - w / 2, m, f"{m:.0f}", ha="center", va="bottom", fontsize=9,
                color=viz.INK)
        a1.text(i + w / 2, p, f"{p:.0f}", ha="center", va="bottom", fontsize=9,
                color=viz.INK)
    a1.set_xticks(x)
    a1.set_xticklabels([f"{h} h" for h in C.HORIZONS])
    a1.set_ylabel("Drift radius containing 90% of cases (km)")
    a1.set_title("Station-keeping error by lead time")
    a1.legend(loc="upper left")

    for k, h in enumerate(C.HORIZONS):
        err = np.hypot(pred[:, k, 0] - true[:, k, 0], pred[:, k, 1] - true[:, k, 1])
        d = np.sort(err * h * KM_PER_MS_PER_H)
        a2.plot(d, np.linspace(0, 100, len(d)), color=viz.SERIES[k], label=f"{h} h lead")
    a2.axhline(90, color=viz.MUTED, ls="--", lw=1.2)
    a2.text(0.99, 90, "90% ", transform=a2.get_yaxis_transform(), ha="right",
            va="bottom", fontsize=8.5, color=viz.MUTED)
    a2.set_xlabel("Drift from station (km)")
    a2.set_ylabel("Cases within this radius (%)")
    a2.set_title("How often the airship stays inside a given radius")
    a2.legend(loc="lower right")
    viz.finish(fig, C.FIGURES / "airship_drift_distribution.png", viz.SOURCE_NOTE)


def fig_trajectories(tracks: dict[str, np.ndarray], days: float, max_airspeed: float):
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.2, 5.0))

    order = [("Perfect knowledge", viz.MUTED), ("LSTM forecast", viz.PREDICTED),
             ("Persistence", viz.BASELINE)]
    for name, color in order:
        if name not in tracks:
            continue
        t = tracks[name]
        a1.plot(t[:, 0], t[:, 1], color=color, lw=2, label=name)
        a1.plot(t[-1, 0], t[-1, 1], marker="o", ms=7, color=color,
                markeredgecolor=viz.SURFACE, markeredgewidth=1.6)
    a1.plot(0, 0, marker="*", ms=16, color=viz.INK, markeredgecolor=viz.SURFACE,
            markeredgewidth=1.5, zorder=5)
    a1.annotate("station", (0, 0), textcoords="offset points", xytext=(9, -4),
                fontsize=8.5, color=viz.INK)
    a1.set_xlabel("East displacement (km)")
    a1.set_ylabel("North displacement (km)")
    a1.set_title(f"Track over {days:.0f} days, airspeed capped at {max_airspeed:g} m/s")
    a1.set_aspect("equal", adjustable="datalim")
    a1.legend(loc="best")

    hours = None
    for name, color in order:
        if name not in tracks:
            continue
        t = tracks[name]
        r = np.hypot(t[:, 0], t[:, 1])
        hours = np.arange(len(r)) * C.STEP_HOURS
        a2.plot(hours, r, color=color, label=name)
    a2.set_xlabel("Hours since start")
    a2.set_ylabel("Distance from station (km)")
    a2.set_title("Distance from station over time")
    a2.legend(loc="upper left")
    viz.finish(fig, C.FIGURES / "airship_trajectories.png", viz.SOURCE_NOTE)


def fig_altitude_choice(df: pd.DataFrame, max_airspeed: float):
    """Altitude as a control lever: the calmest of three levels beats any one level."""
    import matplotlib.pyplot as plt
    spd = pd.DataFrame({lev: df[f"speed{lev}"] for lev in C.LEVELS_HPA})
    best = spd.min(axis=1)
    frac = {**{f"{lev} hPa": float((spd[lev] <= max_airspeed).mean() * 100)
               for lev in C.LEVELS_HPA},
            "best of three": float((best <= max_airspeed).mean() * 100)}

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.4))
    names = list(frac)
    colors = [viz.LEVEL_COLOR[l] for l in C.LEVELS_HPA] + [viz.C4]
    a1.bar(names, [frac[n] for n in names], color=colors, edgecolor=viz.SURFACE,
           linewidth=2, width=.6)
    for n, c in zip(names, colors):
        a1.text(n, frac[n], f"{frac[n]:.0f}%", ha="center", va="bottom", fontsize=9,
                color=viz.INK)
    a1.set_ylabel(f"Time wind is within {max_airspeed:g} m/s (%)")
    a1.set_title("Station-keeping is feasible far more often if altitude can be chosen")
    a1.set_ylim(0, 105)

    for lev in C.LEVELS_HPA:
        m = df.groupby(df.index.month)[f"speed{lev}"].mean()
        a2.plot(m.index, m.values, color=viz.LEVEL_COLOR[lev], marker="o",
                label=f"{lev} hPa")
    a2.axhline(max_airspeed, color=viz.INK, ls="--", lw=1.4)
    a2.text(0.99, max_airspeed, f"airspeed limit {max_airspeed:g} m/s ",
            transform=a2.get_yaxis_transform(), ha="right", va="bottom",
            fontsize=8.5, color=viz.INK)
    a2.set_xticks(range(1, 13))
    a2.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    a2.set_ylabel("Mean wind speed (m/s)")
    a2.set_title("Monthly mean wind by level against the airspeed limit")
    a2.legend(ncol=3, loc="upper right")
    viz.finish(fig, C.FIGURES / "airship_altitude_choice.png", viz.SOURCE_NOTE)
    return frac


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="base")
    ap.add_argument("--max-airspeed", type=float, default=DEFAULT_MAX_AIRSPEED)
    ap.add_argument("--sim-days", type=float, default=10.0)
    args = ap.parse_args()
    viz.apply_style()

    model, ck = load_model(args.tag)
    ds, sx, sy, df = D.build_dataset(verbose=False,
                                     input_steps=input_steps_of(ck),
                                     feature_names=features_of(ck))
    pred, true, base = physical(ds["test"], predict(model, ds["test"]["X"]), sx, sy)

    table = drift_table(pred, true, base)
    table.to_csv(C.METRICS / "airship_drift.csv", index=False)
    print("\n" + table.round(2).to_string(index=False) + "\n")

    fig_drift_distribution(table, pred, true, base)

    # ---- trajectory simulation over a continuous stretch of the test period
    nsteps = int(args.sim_days * 24 / C.STEP_HOURS)
    nsteps = min(nsteps, len(pred) - 1)
    k6 = C.HORIZONS.index(6)

    # Which stretch to simulate is itself a decision worth making explicitly.
    # Starting at an arbitrary point lands in the monsoon easterly jet, where the
    # wind is roughly twice the vehicle's airspeed: nothing can hold station
    # there, so all three controllers drift together and the figure says nothing
    # about forecast quality. A real operator picks a deployment window. So the
    # simulation starts at the calmest contiguous stretch, and the infeasible
    # case is reported separately as the feasibility fraction.
    spd_all = M.speed(true[:, k6, 0], true[:, k6, 1])
    roll = pd.Series(spd_all).rolling(nsteps).mean()
    start = int(roll.idxmin()) - nsteps + 1 if roll.notna().any() else 0
    start = max(0, min(start, len(pred) - nsteps - 1))
    sel = slice(start, start + nsteps)
    window_mean = float(spd_all[sel].mean())
    log.info("simulating the calmest %.0f-day window in the test period: "
             "starts %s, mean wind %.1f m/s against a %.0f m/s airspeed",
             nsteps * C.STEP_HOURS / 24, str(ds["test"]["time"][start])[:10],
             window_mean, args.max_airspeed)

    # The controller plans against the 6 h forecast and the vehicle experiences
    # the wind that actually verified at that time.
    wu, wv = true[sel, k6, 0], true[sel, k6, 1]
    tracks = {
        "Perfect knowledge": simulate_track(wu, wv, wu, wv, args.max_airspeed),
        "LSTM forecast": simulate_track(wu, wv, pred[sel, k6, 0],
                                        pred[sel, k6, 1], args.max_airspeed),
        "Persistence": simulate_track(wu, wv, base[sel, k6, 0],
                                      base[sel, k6, 1], args.max_airspeed),
    }
    fig_trajectories(tracks, nsteps * C.STEP_HOURS / 24, args.max_airspeed)

    frac = fig_altitude_choice(df, args.max_airspeed)

    final = {name: float(np.hypot(*t[-1])) for name, t in tracks.items()}
    worst = {name: float(np.hypot(t[:, 0], t[:, 1]).max()) for name, t in tracks.items()}
    r90 = {str(h): hold_radius(pred, true, k, h) for k, h in enumerate(C.HORIZONS)}

    summary = {
        "max_airspeed_ms": args.max_airspeed,
        "sim_days": nsteps * C.STEP_HOURS / 24,
        "sim_window_start": str(ds["test"]["time"][start]),
        "sim_window_mean_wind_ms": round(window_mean, 2),
        "sim_window_note": ("the calmest contiguous window in the test period; "
                            "an arbitrary window lands in the monsoon jet where "
                            "no forecast can hold station"),
        "hold_radius_90pct_km": r90,
        "final_offset_km": final,
        "max_offset_km": worst,
        "feasible_time_pct_by_level": frac,
        "per_horizon_drift": table.to_dict("records"),
        "headline": (
            f"Using the 6 h forecast, the airship stays within "
            f"{r90['6']:.0f} km of station in 90% of cases, against "
            f"{np.percentile(np.hypot(base[:, 0, 0] - true[:, 0, 0], base[:, 0, 1] - true[:, 0, 1]) * 6 * KM_PER_MS_PER_H, 90):.0f} km "
            "on persistence alone."),
    }
    (C.METRICS / "airship_summary.json").write_text(json.dumps(summary, indent=1))
    log.info(summary["headline"])
    for name, v in worst.items():
        log.info("%-18s max offset %7.1f km, final %7.1f km", name, v, final[name])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
