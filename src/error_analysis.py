"""When does the model fail, and what does failure correlate with?

The metric tables say how large the error is on average. This asks the more
useful operational question: given the current state of the atmosphere, should
the next forecast be trusted? Each test window is paired with the conditions at
its last observed step, and the forecast error is regressed against them.

Usage:  python src/error_analysis.py [--tag base]
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
log = logging.getLogger("errors")

# Human-readable names for the candidate drivers, in the order they are shown.
DRIVERS = {
    "speed50": "Wind speed at 50 hPa",
    "shear_10_50": "Shear, 10 minus 50 hPa",
    "abs_shear": "Shear magnitude",
    "tendency": "Recent rate of change",
    "t50": "Temperature at 50 hPa",
    "dir_variability": "Direction variability in window",
}


def build_conditions(df: pd.DataFrame, times: pd.DatetimeIndex,
                     win: int = C.INPUT_STEPS) -> pd.DataFrame:
    """Atmospheric state at the moment each forecast is issued.

    `win` must be the input window the model was actually trained with, so that
    "recent rate of change" means the same span the model saw.
    """
    cond = pd.DataFrame(index=times)
    cond["speed50"] = df.loc[times, "speed50"].to_numpy()
    cond["shear_10_50"] = df.loc[times, "shear_10_50"].to_numpy()
    cond["abs_shear"] = cond["shear_10_50"].abs()
    cond["t50"] = df.loc[times, "t50"].to_numpy()

    # How fast the wind vector was already moving over the input window. This is
    # the direct test of the "rapid changes" hypothesis.
    u, v = df["u50"], df["v50"]
    tend = np.hypot(u.diff(win), v.diff(win)) / (win * C.STEP_HOURS)
    cond["tendency"] = tend.loc[times].to_numpy()

    # Spread of direction across the input window: a proxy for an unsettled flow.
    dirs = df["dir50"]
    rad = np.radians(dirs)
    # Circular standard deviation, so the 360/0 boundary does not inflate it.
    csd = (pd.Series(np.cos(rad), index=df.index).rolling(win).mean() ** 2
           + pd.Series(np.sin(rad), index=df.index).rolling(win).mean() ** 2)
    cond["dir_variability"] = np.degrees(
        np.sqrt(np.maximum(-2 * np.log(np.sqrt(csd.clip(1e-9, 1))), 0))
    ).loc[times].to_numpy()
    return cond


def fig_correlations(cond: pd.DataFrame, errors: dict[int, np.ndarray]):
    """Heatmap of Spearman correlation between conditions and error."""
    import matplotlib.pyplot as plt
    cols = [c for c in DRIVERS if c in cond.columns]
    mat = np.zeros((len(cols), len(errors)))
    for j, (h, err) in enumerate(errors.items()):
        for i, c in enumerate(cols):
            ok = np.isfinite(cond[c].to_numpy()) & np.isfinite(err)
            mat[i, j] = (pd.Series(cond[c].to_numpy()[ok]).corr(
                pd.Series(err[ok]), method="spearman") if ok.sum() > 10 else np.nan)

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    lim = float(np.nanmax(np.abs(mat))) or 1.0
    im = ax.imshow(mat, cmap=viz.div_cmap(), vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(len(errors)))
    ax.set_xticklabels([f"{h} h" for h in errors])
    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels([DRIVERS[c] for c in cols])
    for i in range(len(cols)):
        for j in range(len(errors)):
            if np.isfinite(mat[i, j]):
                # Label every cell: the sign is the whole point, and the colour
                # alone must not be the only encoding.
                ax.text(j, i, f"{mat[i, j]:+.2f}", ha="center", va="center",
                        fontsize=9,
                        color=viz.SURFACE if abs(mat[i, j]) > lim * .6 else viz.INK)
    ax.set_title("Spearman correlation of forecast error with conditions at issue time")
    ax.grid(visible=False)
    cb = fig.colorbar(im, ax=ax, fraction=.035, pad=.03)
    cb.set_label("Correlation")
    cb.outline.set_visible(False)
    viz.finish(fig, C.FIGURES / "error_drivers_correlation.png", viz.SOURCE_NOTE)
    return pd.DataFrame(mat, index=cols, columns=[f"{h}h" for h in errors])


def fig_error_vs_driver(cond: pd.DataFrame, err: np.ndarray, horizon: int):
    """Binned mean error against the two drivers that matter most operationally."""
    import matplotlib.pyplot as plt
    picks = [("tendency", "Recent rate of change (m/s per h)"),
             ("abs_shear", "Shear magnitude, 10 vs 50 hPa (m/s)"),
             ("speed50", "Wind speed at issue time (m/s)")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, (col, label) in zip(axes, picks):
        x = cond[col].to_numpy()
        ok = np.isfinite(x) & np.isfinite(err)
        xs, es = x[ok], err[ok]
        if len(xs) < 40:
            ax.text(.5, .5, "not enough data", transform=ax.transAxes,
                    ha="center", color=viz.MUTED)
            continue
        # Deciles, so each point rests on the same number of cases.
        edges = np.percentile(xs, np.linspace(0, 100, 11))
        edges = np.unique(edges)
        idx = np.clip(np.digitize(xs, edges[1:-1]), 0, len(edges) - 2)
        centres = [xs[idx == b].mean() for b in range(len(edges) - 1)]
        means = [es[idx == b].mean() for b in range(len(edges) - 1)]
        ax.scatter(xs, es, s=6, color=viz.C1, alpha=.13, edgecolors="none",
                   rasterized=True)
        ax.plot(centres, means, color=viz.C2, marker="o", lw=2.2,
                label="Mean by decile")
        ax.set_xlabel(label)
        ax.legend(loc="upper left")
    axes[0].set_ylabel(f"Vector error at {horizon} h (m/s)")
    fig.suptitle(f"What drives {horizon} h forecast error", x=0.005, ha="left",
                 fontsize=11, fontweight="semibold")
    viz.finish(fig, C.FIGURES / f"error_drivers_{horizon}h.png", viz.SOURCE_NOTE)


def fig_worst_cases(cond: pd.DataFrame, err: np.ndarray, horizon: int):
    """How the worst decile of forecasts differs from the rest."""
    import matplotlib.pyplot as plt
    thresh = np.nanpercentile(err, 90)
    worst = err >= thresh
    cols = [c for c in DRIVERS if c in cond.columns]
    fig, ax = plt.subplots(figsize=(8.8, 4.3))
    x = np.arange(len(cols))
    w = .38
    rest_m, worst_m, rest_s, worst_s = [], [], [], []
    for c in cols:
        v = cond[c].to_numpy()
        rest_m.append(np.nanmean(v[~worst]))
        worst_m.append(np.nanmean(v[worst]))
        rest_s.append(np.nanstd(v[~worst]))
        worst_s.append(np.nanstd(v[worst]))
    # Standardise so variables with different units share one axis.
    scale = np.array([s if s > 0 else 1 for s in rest_s])
    rest_z = np.zeros(len(cols))
    worst_z = (np.array(worst_m) - np.array(rest_m)) / scale
    ax.bar(x - w / 2, rest_z, width=w * .94, color=viz.C1, edgecolor=viz.SURFACE,
           linewidth=2, label="Best 90% of forecasts")
    ax.bar(x + w / 2, worst_z, width=w * .94, color=viz.C2, edgecolor=viz.SURFACE,
           linewidth=2, label="Worst 10% of forecasts")
    for xi, v in zip(x, worst_z):
        ax.text(xi + w / 2, v, f"{v:+.2f}", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=8.5, color=viz.INK)
    ax.axhline(0, color=viz.MUTED, lw=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels([DRIVERS[c] for c in cols], rotation=18, ha="right")
    ax.set_ylabel("Difference from the rest (standard deviations)")
    ax.set_title(f"Conditions during the worst 10% of {horizon} h forecasts")
    ax.legend(loc="upper left")
    viz.finish(fig, C.FIGURES / f"error_worst_cases_{horizon}h.png", viz.SOURCE_NOTE)
    return {c: float(z) for c, z in zip(cols, worst_z)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="base")
    args = ap.parse_args()
    viz.apply_style()

    model, ck = load_model(args.tag)
    ds, sx, sy, df = D.build_dataset(verbose=False,
                                     input_steps=input_steps_of(ck),
                                     feature_names=features_of(ck))
    pred, true, base = physical(ds["test"], predict(model, ds["test"]["X"]), sx, sy)
    times = pd.DatetimeIndex(ds["test"]["time"])

    cond = build_conditions(df, times, win=input_steps_of(ck))
    errors = {h: np.hypot(pred[:, k, 0] - true[:, k, 0],
                          pred[:, k, 1] - true[:, k, 1])
              for k, h in enumerate(C.HORIZONS)}

    corr = fig_correlations(cond, errors)
    corr.round(3).to_csv(C.METRICS / "error_drivers_correlation.csv")
    print("\nSpearman correlation, error vs conditions at issue time:")
    print(corr.round(3).to_string() + "\n")

    k6 = C.HORIZONS.index(6)
    fig_error_vs_driver(cond, errors[6], 6)
    worst = fig_worst_cases(cond, errors[6], 6)

    out = {
        "tag": args.tag,
        "n_test_windows": int(len(times)),
        "correlation_spearman": json.loads(corr.round(4).to_json()),
        "worst_decile_z_shift_6h": {k: round(v, 3) for k, v in worst.items()},
        "worst_decile_threshold_ms_6h": float(np.nanpercentile(errors[6], 90)),
    }
    (C.METRICS / "error_analysis.json").write_text(json.dumps(out, indent=1))

    top = corr["6h"].abs().sort_values(ascending=False)
    log.info("strongest driver of 6 h error: %s (rho = %+.3f)",
             DRIVERS[top.index[0]], corr.loc[top.index[0], "6h"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
