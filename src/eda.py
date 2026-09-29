"""Exploratory analysis of the stratospheric wind record.

Produces the figures and the statistics table that justify the modelling
choices, and writes a data-quality report.

Usage:  python src/eda.py
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
import data as D
import viz

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("eda")

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


# --------------------------------------------------------------------- figures
def fig_series(df: pd.DataFrame):
    """The whole record: speed, the two components, and direction."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 1, figsize=(11.5, 10.4), sharex=True,
                             gridspec_kw={"hspace": 0.16})

    ax = axes[0]
    for lev in C.LEVELS_HPA:
        ax.plot(df.index, df[f"speed{lev}"], color=viz.LEVEL_COLOR[lev],
                lw=0.7, label=f"{lev} hPa")
    ax.set_ylabel("Speed (m/s)")
    ax.set_title(f"Stratospheric wind above {C.TARGET_LAT:.0f}N {C.TARGET_LON:.0f}E, "
                 f"{df.index[0]:%b %Y} to {df.index[-1]:%b %Y}")
    # Legend anchored right so it shares the title's line without colliding.
    ax.legend(ncol=3, loc="lower right", bbox_to_anchor=(1, 1.0))

    for ax, comp, name in ((axes[1], "u", "Zonal wind u (m/s, + = eastward)"),
                          (axes[2], "v", "Meridional wind v (m/s, + = northward)")):
        for lev in C.LEVELS_HPA:
            ax.plot(df.index, df[f"{comp}{lev}"], color=viz.LEVEL_COLOR[lev],
                    lw=0.7, label=f"{lev} hPa")
        ax.axhline(0, color=viz.MUTED, lw=1, ls="--")
        ax.set_ylabel(name)
        ax.legend(ncol=3, loc="upper right")

    ax = axes[3]
    ax.plot(df.index, df[f"dir{C.PRIMARY_LEVEL}"], lw=0, marker=".", ms=1.6,
            color=viz.LEVEL_COLOR[C.PRIMARY_LEVEL],
            label=f"{C.PRIMARY_LEVEL} hPa direction")
    ax.set_ylim(0, 360)
    ax.set_yticks([0, 90, 180, 270, 360])
    ax.set_yticklabels(["N", "E", "S", "W", "N"])
    ax.set_ylabel("Direction (from)")
    ax.legend(loc="upper right")
    fig.autofmt_xdate()
    viz.finish(fig, C.FIGURES / "eda_timeseries_full.png", viz.SOURCE_NOTE)


def fig_monthly(df: pd.DataFrame):
    """Seasonal cycle: monthly mean and spread of speed and of the zonal wind."""
    import matplotlib.pyplot as plt
    g = df.groupby(df.index.month)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))

    for lev in C.LEVELS_HPA:
        m = g[f"speed{lev}"].mean()
        s = g[f"speed{lev}"].std()
        a1.plot(m.index, m.values, color=viz.LEVEL_COLOR[lev], marker="o",
                label=f"{lev} hPa")
        a1.fill_between(m.index, m - s, m + s, color=viz.LEVEL_COLOR[lev], alpha=.13,
                        linewidth=0)
    a1.set_xticks(range(1, 13))
    a1.set_xticklabels(MONTHS)
    a1.set_ylabel("Wind speed (m/s)")
    a1.set_title("Monthly mean speed, shaded +/- 1 sd")
    a1.legend(ncol=3, loc="upper right")

    for lev in C.LEVELS_HPA:
        m = g[f"u{lev}"].mean()
        a2.plot(m.index, m.values, color=viz.LEVEL_COLOR[lev], marker="o",
                label=f"{lev} hPa")
    a2.axhline(0, color=viz.MUTED, lw=1.2, ls="--")
    # Anchored left: at this data range the zero line sits near the top of the
    # panel, where a right-aligned note collides with the legend.
    a2.text(0.01, 0, " easterly below / westerly above",
            transform=a2.get_yaxis_transform(),
            ha="left", va="bottom", fontsize=8.5, color=viz.MUTED)
    a2.set_xticks(range(1, 13))
    a2.set_xticklabels(MONTHS)
    a2.set_ylabel("Zonal wind u (m/s)")
    a2.set_title("Monthly mean zonal wind: the sign is the regime")
    a2.legend(ncol=3, loc="upper right")
    viz.finish(fig, C.FIGURES / "eda_seasonal_cycle.png", viz.SOURCE_NOTE)


def fig_shear(df: pd.DataFrame):
    """Wind shear between the float level and the levels above it."""
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.4),
                                 gridspec_kw={"width_ratios": [2, 1]})
    a1.plot(df.index, df["shear_10_50"], color=viz.C3, lw=0.7)
    a1.axhline(0, color=viz.MUTED, lw=1.2, ls="--")
    a1.set_ylabel("Speed difference (m/s)")
    a1.set_title("Wind shear, 10 hPa minus 50 hPa")
    viz.annotate(a1, "positive = faster aloft")

    monthly = df.groupby(df.index.month)["shear_10_50"]
    a2.bar(monthly.mean().index, monthly.mean().values, color=viz.C3,
           edgecolor=viz.SURFACE, linewidth=2)
    a2.axhline(0, color=viz.MUTED, lw=1.2, ls="--")
    a2.set_xticks(range(1, 13))
    a2.set_xticklabels(MONTHS, fontsize=8)
    a2.set_ylabel("Mean shear (m/s)")
    a2.set_title("Monthly mean shear")
    fig.autofmt_xdate()
    viz.finish(fig, C.FIGURES / "eda_wind_shear.png", viz.SOURCE_NOTE)


def fig_roses(df: pd.DataFrame):
    """Wind roses: direction frequency, coloured by speed band, one per level."""
    import matplotlib.pyplot as plt
    nsec = 16
    width = 2 * np.pi / nsec
    sectors = np.linspace(0, 360, nsec + 1)
    bands = [(0, 10), (10, 20), (20, 30), (30, 45), (45, np.inf)]
    band_colors = viz.SEQ_BLUE[2::2][:len(bands)]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.0),
                             subplot_kw={"projection": "polar"})
    for ax, lev in zip(np.atleast_1d(axes), C.LEVELS_HPA):
        d, s = df[f"dir{lev}"].to_numpy(), df[f"speed{lev}"].to_numpy()
        bottom = np.zeros(nsec)
        for (lo, hi), col in zip(bands, band_colors):
            inband = (s >= lo) & (s < hi)
            frac = np.array([
                ((d >= sectors[i]) & (d < sectors[i + 1]) & inband).sum()
                for i in range(nsec)]) / len(d) * 100
            ax.bar(np.radians(sectors[:-1] + 360 / nsec / 2), frac, width=width * .92,
                   bottom=bottom, color=col, edgecolor=viz.SURFACE, linewidth=1.2,
                   label=f"{lo:g}-{hi:g} m/s" if np.isfinite(hi) else f"{lo:g}+ m/s")
            bottom += frac
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_xticks(np.radians(np.arange(0, 360, 45)))
        ax.set_xticklabels(["N", "NE", "E", "SE", "S", "SW", "W", "NW"])
        ax.set_title(f"{lev} hPa", pad=16)
        ax.tick_params(labelsize=8)
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=len(bands), loc="lower center",
               bbox_to_anchor=(0.5, -0.04), frameon=False, fontsize=9)
    fig.suptitle("Wind rose: frequency of the direction the wind comes FROM (% of hours)",
                 x=0.005, ha="left", fontsize=11, fontweight="semibold")
    viz.finish(fig, C.FIGURES / "eda_wind_roses.png", viz.SOURCE_NOTE)


def fig_missing(missing: pd.DataFrame, df_clean: pd.DataFrame):
    """Where the record had gaps, before imputation."""
    import matplotlib.pyplot as plt
    total = int(missing.to_numpy().sum())
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11.5, 5.2),
                                 gridspec_kw={"height_ratios": [1, 1.5]})

    per_col = missing.sum()
    a1.bar(per_col.index, per_col.values, color=viz.C2, edgecolor=viz.SURFACE,
           linewidth=2)
    a1.set_ylabel("Missing steps")
    a1.set_title(f"Missing values before imputation: {total} cells "
                 f"of {missing.size} ({total / max(missing.size, 1) * 100:.3f}%)")
    a1.tick_params(axis="x", labelrotation=0)

    # A day-by-variable map makes a clustered outage obvious; an evenly sprinkled
    # one means isolated steps.
    daily = missing.any(axis=1).astype(int).resample("1D").sum()
    a2.fill_between(daily.index, daily.values, color=viz.C2, linewidth=0)
    a2.set_ylabel("Steps missing per day")
    a2.set_xlabel("")
    if total == 0:
        a2.text(0.5, 0.5, "No gaps: the hourly axis is complete",
                transform=a2.transAxes, ha="center", va="center",
                fontsize=11, color=viz.MUTED)
    fig.autofmt_xdate()
    viz.finish(fig, C.FIGURES / "eda_missing_data.png", viz.SOURCE_NOTE)


def fig_spatial(ds):
    """Mean 50 hPa flow over the domain, and the seasonal reversal."""
    import matplotlib.pyplot as plt
    lev = ds.sel(pressure_level=C.PRIMARY_LEVEL)
    lon, lat = ds.longitude.values, ds.latitude.values
    months = pd.DatetimeIndex(ds.valid_time.values).month

    panels = [("January (winter)", months == 1), ("July (monsoon)", months == 7)]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True)
    u_all = lev.u.values
    v_all = lev.v.values
    vmax = float(np.nanpercentile(np.hypot(u_all, v_all), 99))

    for ax, (title, sel) in zip(axes, panels):
        if not sel.any():
            ax.text(0.5, 0.5, "no data for this month", transform=ax.transAxes,
                    ha="center", va="center", color=viz.MUTED)
            continue
        u = np.nanmean(u_all[sel], axis=0)
        v = np.nanmean(v_all[sel], axis=0)
        spd = np.hypot(u, v)
        im = ax.pcolormesh(lon, lat, spd, cmap=viz.seq_cmap(), vmin=0, vmax=vmax,
                           shading="auto")
        st = 3
        ax.quiver(lon[::st], lat[::st], u[::st, ::st], v[::st, ::st],
                  color=viz.INK, alpha=.72, width=.0035, scale=420)
        ax.plot(C.TARGET_LON, C.TARGET_LAT, marker="*", ms=15, color=viz.C2,
                markeredgecolor=viz.SURFACE, markeredgewidth=1.5, zorder=5)
        ax.annotate("station", (C.TARGET_LON, C.TARGET_LAT),
                    textcoords="offset points", xytext=(10, -4),
                    fontsize=8.5, color=viz.INK)
        ax.set_title(title)
        ax.set_xlabel("Longitude (E)")
        ax.grid(visible=False)
    axes[0].set_ylabel("Latitude (N)")
    cb = fig.colorbar(im, ax=axes, fraction=.03, pad=.02)
    cb.set_label("Mean wind speed (m/s)")
    cb.outline.set_visible(False)
    fig.suptitle(f"Mean {C.PRIMARY_LEVEL} hPa flow: the stratospheric wind reverses "
                 "between seasons", x=0.005, ha="left", fontsize=11,
                 fontweight="semibold")
    viz.finish(fig, C.FIGURES / "eda_spatial_mean_flow.png", viz.SOURCE_NOTE)


def fig_autocorr(df: pd.DataFrame):
    """How far ahead persistence stays useful - the case for the horizons chosen."""
    import matplotlib.pyplot as plt
    max_lag_h = 24 * 10
    lags = np.arange(1, max_lag_h // C.STEP_HOURS + 1)
    fig, ax = plt.subplots(figsize=(9, 4.3))
    for lev, comp in ((C.PRIMARY_LEVEL, "u"), (C.PRIMARY_LEVEL, "v")):
        s = df[f"{comp}{lev}"]
        ac = [s.autocorr(lag=int(k)) for k in lags]
        ax.plot(lags * C.STEP_HOURS, ac, label=f"{comp} at {lev} hPa",
                color=viz.C1 if comp == "u" else viz.C2)
    for h in C.HORIZONS:
        ax.axvline(h, color=viz.MUTED, ls="--", lw=1)
        ax.text(h, 1.0, f" {h}h", fontsize=8, color=viz.MUTED, va="top")
    ax.axhline(0, color=viz.MUTED, lw=1)
    ax.set_xlabel("Lag (hours)")
    ax.set_ylabel("Autocorrelation")
    ax.set_title("Two different regimes: a persistent zonal flow and a tidal "
                 "meridional one")
    ax.legend(loc="center right")
    viz.annotate(ax, "u decays barely at all over 10 days; v rings at a clean "
                     "24 h period", loc="lower left")
    viz.finish(fig, C.FIGURES / "eda_autocorrelation.png", viz.SOURCE_NOTE)


def fig_diurnal(df: pd.DataFrame):
    """The diurnal tide in the meridional wind, and why it matters for the model.

    The autocorrelation of v rings at exactly 24 h. That is the atmospheric
    diurnal tide, and it is large: the swing across the day is comparable to v's
    own standard deviation. A model given no clock cannot represent it, which is
    the argument for encoding the hour of day as a feature.
    """
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.3))

    hours = sorted(df.index.hour.unique())
    for lev in C.LEVELS_HPA:
        g = df.groupby(df.index.hour)[f"v{lev}"]
        m, sem = g.mean(), g.std() / np.sqrt(g.count())
        a1.plot(m.index, m.values, color=viz.LEVEL_COLOR[lev], marker="o",
                label=f"{lev} hPa")
        a1.fill_between(m.index, m - 1.96 * sem, m + 1.96 * sem,
                        color=viz.LEVEL_COLOR[lev], alpha=.15, linewidth=0)
    a1.axhline(0, color=viz.MUTED, lw=1.2, ls="--")
    a1.set_xticks(hours)
    a1.set_xlabel("Hour of day (UTC)")
    a1.set_ylabel("Mean meridional wind v (m/s)")
    a1.set_title("Diurnal tide in v, shaded 95% interval on the mean")
    a1.legend(ncol=3, loc="upper right")

    # The same cut for u, to show the tide is specific to the meridional wind.
    amps = {}
    for comp in ("u", "v"):
        g = df.groupby(df.index.hour)[f"{comp}{C.PRIMARY_LEVEL}"].mean()
        amps[comp] = float(g.max() - g.min())
    sd = {c: float(df[f"{c}{C.PRIMARY_LEVEL}"].std()) for c in ("u", "v")}

    # Absolute swing would mislead: u varies far more overall, so the only fair
    # comparison is the swing as a share of each component's own variability.
    labels = ["u (zonal)", "v (meridional)"]
    share = [amps["u"] / sd["u"] * 100, amps["v"] / sd["v"] * 100]
    a2.bar(labels, share, color=[viz.C1, viz.C2],
           edgecolor=viz.SURFACE, linewidth=2, width=.55)
    for i, (c, pct) in enumerate(zip(("u", "v"), share)):
        a2.text(i, pct, f"{pct:.0f}%\n({amps[c]:.2f} of {sd[c]:.2f} m/s sd)",
                ha="center", va="bottom", fontsize=9, color=viz.INK)
    a2.set_ylim(0, max(share) * 1.45)
    a2.set_ylabel("Diurnal swing as % of the component's sd")
    a2.set_title("The tide dominates v but is a minor part of u")
    viz.finish(fig, C.FIGURES / "eda_diurnal_cycle.png", viz.SOURCE_NOTE)


# ----------------------------------------------------------------- stats tables
def stats_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = ([f"speed{l}" for l in C.LEVELS_HPA]
            + [f"u{l}" for l in C.LEVELS_HPA]
            + [f"v{l}" for l in C.LEVELS_HPA]
            + [f"t{l}" for l in C.LEVELS_HPA] + ["shear_10_50"])
    t = df[cols].describe().T[["mean", "std", "min", "25%", "50%", "75%", "max"]]
    t.insert(0, "n", df[cols].notna().sum().values)
    t["units"] = ["m/s" if c.startswith(("speed", "u", "v", "shear")) else "K"
                  for c in cols]
    return t.round(3)


def main() -> int:
    viz.apply_style()
    ds = D.open_raw()
    log.info("raw: %d timesteps, %s to %s", ds.valid_time.size,
             str(ds.valid_time.values[0])[:13], str(ds.valid_time.values[-1])[:13])

    raw_df = D.add_derived(D.station_frame(ds))
    df, missing = D.clean(raw_df)
    log.info("station series: %d steps, %d features", len(df), len(D.FEATURES))

    fig_series(df)
    fig_monthly(df)
    fig_shear(df)
    fig_roses(df)
    fig_missing(missing, df)
    fig_autocorr(df)
    fig_diurnal(df)
    try:
        fig_spatial(ds)
    except Exception as exc:  # noqa: BLE001 - a map is nice-to-have, not the point
        log.warning("spatial figure skipped: %s", exc)

    tbl = stats_table(df)
    tbl.to_csv(C.METRICS / "eda_statistics.csv")
    print("\n" + tbl.to_string() + "\n")

    n_expected = len(pd.date_range(df.index[0], df.index[-1], freq=f"{C.STEP_HOURS}h"))
    report = {
        "period": [str(df.index[0]), str(df.index[-1])],
        "step_hours": C.STEP_HOURS,
        "n_steps": int(len(df)),
        "n_expected_steps": int(n_expected),
        "completeness_pct": round(len(raw_df) / n_expected * 100, 4),
        "n_imputed_cells": int(missing.to_numpy().sum()),
        "grid": {"area_N_W_S_E": C.AREA, "resolution_deg": C.GRID,
                 "n_lat": int(ds.latitude.size), "n_lon": int(ds.longitude.size)},
        "levels_hPa": C.LEVELS_HPA,
        "station": {"lat": C.TARGET_LAT, "lon": C.TARGET_LON},
        "features": D.FEATURES,
        "statistics": json.loads(tbl.to_json(orient="index")),
    }
    (C.METRICS / "eda_report.json").write_text(json.dumps(report, indent=1))
    log.info("figures -> %s ; tables -> %s", C.FIGURES, C.METRICS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
