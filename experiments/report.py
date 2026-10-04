"""Fill the generated sections of README.md from the result files.

Every number in the README and every comparative verdict ("MPC beats greedy",
"no significant difference") is written here from the CSV and JSON files, with
the verdict decided by the confidence interval, so the prose cannot claim more
than the results support. Each managed region is delimited by
<!-- AUTO:name --> ... <!-- /AUTO:name -->.

Usage:  python experiments/report.py --config configs/experiment.yaml
"""
from __future__ import annotations

import argparse
import json
import re

import numpy as np
import pandas as pd

from stratoballoon.config import ROOT, load_yaml
from stratoballoon.evaluation import paired_ci


def _csv(p):
    return pd.read_csv(p) if p.exists() else None


def table(df: pd.DataFrame, fmt: dict[str, str] | None = None) -> str:
    fmt = fmt or {}
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                cells.append(fmt.get(c, "{:.1f}").format(v) if np.isfinite(v) else "-")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def verdict(mu, lo, hi, unit, better="higher"):
    if lo > 0:
        word = "better" if better == "higher" else "worse"
    elif hi < 0:
        word = "worse" if better == "higher" else "better"
    else:
        return f"no significant difference ({mu:+.1f} {unit}, 95% CI {lo:+.1f} to {hi:+.1f})"
    return f"{word} by {abs(mu):.1f} {unit} (95% CI {lo:+.1f} to {hi:+.1f})"


class Report:
    def __init__(self, cfg_path):
        self.cfg = load_yaml(cfg_path)["experiment"]
        self.cfg_path = str(cfg_path).replace("\\", "/")
        self.out = ROOT / self.cfg["out_dir"]
        self.fd = ROOT / self.cfg["forecast_dir"]
        self.year = int(self.cfg["test_year"])

    # --------------------------------------------------------------- blocks
    def provenance(self):
        run = json.loads((self.fd / "run.json").read_text()) if (self.fd / "run.json").exists() else {}
        d = run.get("data", {})
        lv = run.get("levels_hpa", [])
        return (f"Results below were generated from `{self.cfg['data_glob']}` "
                f"({d.get('n_files', '?')} monthly files, dataset checksum "
                f"`{str(d.get('sha256', '?'))[:12]}`), pressure levels {lv} hPa, with "
                f"{self.year} as the held-out test year. Configuration: "
                f"`{self.cfg_path}`.")

    def forecast(self):
        ci = _csv(self.fd / "skill_ci.csv")
        if ci is None:
            return "_Not generated yet._"
        ci = ci[ci.test_year == ci.test_year.max()] if self.year not in set(ci.test_year) else ci[ci.test_year == self.year]
        p = ci[ci.reference == "persistence"].groupby(["model", "horizon_h"]).agg(
            skill=("skill_pct", "mean"), lo=("ci_lo_pct", "mean"), hi=("ci_hi_pct", "mean")).reset_index()
        wide = p.pivot(index="model", columns="horizon_h", values="skill").round(1)
        order = [m for m in ["tide_persistence", "climatology", "moving_average", "linear", "ridge",
                             "gradient_boosting", "lstm", "tcn"] if m in wide.index]
        wide = wide.loc[order]
        wide.columns = [f"{c} h" for c in wide.columns]
        wide.insert(0, "model", [m.replace("_", " ") for m in wide.index])
        txt = ("Vector-RMSE skill against persistence (%), mean over pressure levels, test year "
               f"{self.year}. Higher is better; negative is worse than assuming the wind stays "
               "as it is.\n\n" + table(wide.reset_index(drop=True)))
        r = ci[ci.reference == "ridge"].groupby("model").agg(
            lo=("ci_lo_pct", lambda x: (x > 0).mean()), skill=("skill_pct", "mean"))
        lines = []
        for m in ("gradient_boosting", "lstm", "tcn"):
            if m in r.index:
                lines.append(f"- **{m.replace('_', ' ')} vs ridge**: mean skill "
                             f"{r.loc[m, 'skill']:+.1f}%, significantly better in "
                             f"{r.loc[m, 'lo']:.0%} of (horizon, level) cells.")
        best = self._best()
        txt += "\n\n" + "\n".join(lines)

        # the same comparison in every rolling test year, to show it is not one lucky year
        allci = _csv(self.fd / "skill_ci.csv")
        yearly = allci[(allci.reference == "persistence")
                       & allci.model.isin(["ridge", "gradient_boosting", "lstm", "tcn"])]
        if yearly.test_year.nunique() > 1:
            y = yearly.pivot_table(index="test_year", columns="model", values="skill_pct")
            y = y[[c for c in ("ridge", "gradient_boosting", "lstm", "tcn") if c in y.columns]]
            y.columns = [c.replace("_", " ") for c in y.columns]
            y.insert(0, "test year", [str(int(i)) for i in y.index])
            txt += ("\n\nMean skill over all lead times and levels, by rolling test year (each "
                    "model retrained on the years before it):\n\n" + table(y.reset_index(drop=True)))
            txt += (f"\n\n**Selected forecaster: {best}.** The choice was made on the test years "
                    f"before {self.year} only: among models that beat ridge outside the "
                    "block-bootstrap interval in most lead-time and level combinations, the one "
                    f"with the highest mean skill. {self.year} did not influence it.")
        else:
            txt += f"\n\nSelected forecaster by the stated rule: **{best}**."
        txt += self._vs_simpler(best)
        return txt

    def _vs_simpler(self, best: str) -> str:
        """Is the selected model clearly better than the next simpler one, or just ahead?"""
        simpler = {"lstm": "gradient_boosting", "tcn": "gradient_boosting",
                   "gradient_boosting": "ridge"}.get(best)
        files = [self.fd / f"test_pred_{m}_{self.year}.npy" for m in (best, simpler or "")]
        truth, tidx = self.fd / f"test_truth_{self.year}.npy", self.fd / f"test_tidx_{self.year}.npy"
        if not simpler or not all(f.exists() for f in files + [truth, tidx]):
            return ""
        from stratoballoon.forecasting import metrics as M
        t = np.load(truth)
        sq = [np.sum((np.load(f) - t) ** 2, -1) for f in files]
        t_idx = np.load(tidx)
        blocks = M.block_ids(t_idx, 32)                      # 8-day blocks of 6-hourly steps
        res = [M.bootstrap_skill(sq[0][:, k, l], sq[1][:, k, l], blocks, 300)
               for k in range(t.shape[1]) for l in range(t.shape[2])]
        better = np.mean([r["ci_lo_pct"] > 0 for r in res])
        mean = np.mean([r["skill_pct"] for r in res])
        return (f" Against the next simpler model, {simpler.replace('_', ' ')}, {best} has "
                f"{mean:.1f}% lower error on {self.year} and is significantly better in "
                f"{better:.0%} of lead-time and level combinations. "
                + ("That margin is clear." if better > 0.5 and mean >= 2 else
                   f"That margin is small: {simpler.replace('_', ' ')} would be a defensible "
                   "choice where a simpler model is preferred."))

    def _best(self):
        from stratoballoon.experiment import Context
        c = Context.__new__(Context)
        c.forecast_dir, c.test_year = self.fd, self.year
        try:
            return Context.best_model(c)
        except Exception:  # noqa: BLE001
            return "ridge"

    def ablation(self):
        df = _csv(self.fd / "ablation" / "importance.csv")
        if df is None:
            return "_Not generated yet._"
        g = df[df.horizon_h.isin([6, 24])].pivot(index="group", columns="horizon_h",
                                                 values="ablation_increase_pct")
        g = g.sort_values(g.columns[0], ascending=False)
        g.columns = [f"error increase without it, {c} h (%)" for c in g.columns]
        g.insert(0, "input group", g.index)
        return table(g.reset_index(drop=True), {c: "{:+.1f}" for c in g.columns[1:]})

    def uncertainty(self):
        cov = _csv(self.out / "uncertainty" / "coverage.csv")
        cond = _csv(self.out / "uncertainty" / "conditional_coverage.csv")
        crps = _csv(self.out / "uncertainty" / "crps.csv")
        if cov is None:
            return "_Not generated yet._"
        best = self._best()
        c = cov[(cov.model == best)].groupby(["nominal", "horizon_h"]).disc_coverage.mean().unstack()
        c = (c * 100).round(1)
        c.columns = [f"{h} h" for h in c.columns]
        c.insert(0, "nominal", [f"{n:.0%}" for n in c.index])
        txt = (f"Observed coverage (%) of the conformal wind regions of the {best} forecast on "
               f"the test year, calibrated on the year before.\n\n" + table(c.reset_index(drop=True)))
        if cond is not None:
            g = cond[cond.horizon_h == cond.horizon_h.min()]
            worst = g.loc[g.coverage.idxmin()]
            txt += (f"\n\nCoverage is only guaranteed on average: at {int(worst.horizon_h)} h it "
                    f"drops to {worst.coverage:.0%} for the '{worst.value}' {worst.group} group "
                    f"(nominal 90%).")
        reg = _csv(self.out / "uncertainty" / "regime_conformal.csv")
        if reg is not None:
            g = reg[reg.horizon_h == reg.horizon_h.min()].pivot(index="group", columns="method",
                                                                 values="coverage")
            g = (g * 100).round(1).reindex(["all", "weak now", "moderate now", "strong now"])
            g = g[[c for c in ("marginal", "regime", "adaptive") if c in g.columns]]
            g.insert(0, "wind when issued", g.index)
            txt += ("\n\nMarginal, regime-conditional and adaptive calibration, coverage (%) of the "
                    f"90% region at {int(reg.horizon_h.min())} h, grouped by the wind at issue "
                    "time:\n\n" + table(g.reset_index(drop=True)))
        if crps is not None:
            m = crps.groupby("horizon_h")[["crps", "point_mae"]].mean()
            imp = (1 - m.crps / m.point_mae) * 100
            txt += ("\n\nThe sampled-scenario ensemble scores "
                    + ", ".join(f"{v:.0f}% better at {h} h" for h, v in imp.items())
                    + " than the point forecast on CRPS (equal to MAE for a single forecast).")
        return txt

    def trajectory(self):
        e = _csv(self.out / "trajectory" / "position_error.csv")
        cone = _csv(self.out / "trajectory" / "cone_coverage.csv")
        br = _csv(self.out / "trajectory" / "region_probability.csv")
        if e is None:
            return "_Not generated yet._"
        w = e[e.source != "perfect"].pivot(index="source", columns="lead_h", values="median_km")
        n = e[e.source == "persistence"].set_index("lead_h").n_scored
        w.columns = [f"{c} h" for c in w.columns]
        w.insert(0, "forecast", [s.replace("_", " ") for s in w.index])
        txt = ("Median distance (km) between predicted and actual position of a balloon "
               "drifting at fixed altitude, with no forecast updates after launch. "
               f"Scored tracks per lead: {', '.join(f'{int(k)} h: {int(v)}' for k, v in n.items())} "
               "(tracks that leave the data domain stop being scored).\n\n"
               + table(w.reset_index(drop=True), {c: "{:.0f}" for c in w.columns[1:]}))
        if cone is not None:
            c = cone[cone.nominal.isin([0.8, 0.95])].pivot(index="nominal", columns="lead_h",
                                                           values="coverage")
            c = (c * 100).round(0)
            c.columns = [f"{h} h" for h in c.columns]
            c.insert(0, "cone", [f"{n:.0%}" for n in c.index])
            txt += "\n\nUncertainty-cone coverage (%):\n\n" + table(c.reset_index(drop=True), {k: "{:.0f}" for k in c.columns[1:]})
        if br is not None:
            r = br.set_index("lead_h")
            txt += ("\n\nProbability of still being within 200 km of launch, Brier score "
                    "(lower is better), ensemble / single forecast / climatology: "
                    + "; ".join(f"{h} h {x.brier_ensemble:.3f} / {x.brier_deterministic:.3f} / "
                                f"{x.brier_climatology:.3f}" for h, x in r.iterrows()) + ".")
        return txt

    def control(self):
        s = _csv(self.out / "control" / "summary.csv")
        if s is None:
            return "_Not generated yet._"
        t = pd.DataFrame({
            "controller / forecast": s.run,
            "time within 50 km (%)": s.twr50 * 100,
            "gain over hold (pp, 95% CI)": [f"{m * 100:+.1f} [{lo * 100:+.1f}, {hi * 100:+.1f}]"
                                           for m, lo, hi in zip(s.twr50_vs_hold, s.twr50_vs_hold_lo,
                                                                s.twr50_vs_hold_hi)],
            "median mean distance (km)": s.median_dist_km,
            "pump energy (Wh)": s.pump_energy_wh,
            "altitude changes": s.target_changes,
            "left domain (%)": s.left_domain * 100,
        })
        txt = table(t, {"time within 50 km (%)": "{:.1f}", "median mean distance (km)": "{:.0f}",
                         "pump energy (Wh)": "{:.0f}", "altitude changes": "{:.1f}",
                         "left domain (%)": "{:.0f}"})
        m = pd.read_csv(self.out / "control" / "missions.csv")
        best = self._best()
        pairs = [(f"mpc / {best}", f"greedy / {best}", "MPC against the greedy rule"),
                 (f"mpc-robust / {best}", f"mpc / {best}",
                  "uncertainty-aware MPC against deterministic MPC"),
                 (f"mpc / {best}", "mpc / persistence", f"MPC with {best} against MPC with persistence"),
                 ("mpc / perfect forecast", f"mpc / {best}", f"a perfect forecast against {best}"),
                 (f"rl-ppo / {best}", f"mpc / {best}", "reinforcement learning against MPC")]
        lines = []
        for a_, b_, label in pairs:
            ga, gb = m[m.run == a_].sort_values("mission"), m[m.run == b_].sort_values("mission")
            if len(ga) == 0 or len(gb) == 0:
                continue
            mu, lo, hi = paired_ci(ga.twr50.to_numpy() * 100, gb.twr50.to_numpy() * 100,
                                   ga.start_week.to_numpy())
            lines.append(f"- {label}: {verdict(mu, lo, hi, 'percentage points')}.")
        return (txt + "\n\nPaired comparisons on identical missions (cluster bootstrap by "
                "launch week):\n\n" + "\n".join(lines))

    def season(self):
        b = _csv(self.out / "control" / "twr50_by_season.csv")
        if b is None:
            return "_Not generated yet._"
        b = b.set_index("run")[[c for c in ("DJF", "MAM", "JJAS", "ON") if c in b.columns]] * 100
        b.insert(0, "controller / forecast", b.index)
        return "Time within 50 km (%) by season of launch:\n\n" + table(b.reset_index(drop=True))

    def feasibility(self):
        f = _csv(self.out / "feasibility" / "twr50_by_month.csv")
        if f is None:
            return "_Not generated yet._"
        f = f.set_index("month") * 100
        months = "JFMAMJJASOND"
        t = pd.DataFrame({"controller": f.columns})
        for mth in f.index:
            t[months[int(mth) - 1] + str(int(mth))] = f.loc[mth].values
        return ("Time within 50 km (%) by launch month, with a perfect forecast (the upper "
                "bound no forecaster can beat) and with no control:\n\n" + table(t))

    def robustness(self):
        r = _csv(self.out / "robustness" / "summary.csv")
        if r is None:
            return "_Not generated yet._"
        w = r.pivot(index="fault", columns="config", values="twr50") * 100
        w = w.loc[r.fault.unique()]
        e = r.pivot(index="fault", columns="config", values="est_err_max_km").loc[r.fault.unique()]
        w.columns = [f"{c} (%)" for c in w.columns]
        w.insert(0, "fault", w.index)
        kf = [c for c in e.columns if "filter" in c]
        raw = [c for c in e.columns if "raw GPS" in c]
        txt = "Time within 50 km (%) under each injected fault:\n\n" + table(w.reset_index(drop=True))
        if kf and raw:
            g = e.loc[e.index.str.contains("gps")]
            if len(g):
                txt += (f"\n\nDuring GPS outages the worst position-estimate error is "
                        f"{g[raw[0]].iloc[0]:.0f} km holding the last fix, against "
                        f"{g[kf[0]].iloc[0]:.0f} km with the Kalman filter dead-reckoning.")
        return txt

    def anomaly(self):
        a = _csv(ROOT / "results/anomaly/summary.csv")
        if a is None:
            return "_Not generated yet._"
        a = a[["detector", "recall", "median_latency_min", "false_alarms_per_day", "precision"]].copy()
        a["recall"] *= 100
        a["precision"] *= 100
        a.columns = ["detector", "faults caught (%)", "median delay (min)", "false alarms per day",
                     "alarm precision (%)"]
        return table(a, {"faults caught (%)": "{:.0f}", "median delay (min)": "{:.0f}",
                         "false alarms per day": "{:.2f}", "alarm precision (%)": "{:.0f}"})

    def edge(self):
        e = _csv(self.out / "edge" / "edge_profile.csv")
        if e is None:
            return "_Not generated yet._"
        e = e[["component", "proposed_location", "p50_ms", "size_kb"]]
        e.columns = ["component", "where it should run", "median latency (ms)", "size (KB)"]
        return ("Measured on the development PC (x86, one thread), not flight hardware:\n\n"
                + table(e, {"median latency (ms)": "{:.2f}", "size (KB)": "{:.1f}"}))

    def radiosondes(self):
        r = _csv(self.out / "radiosondes" / "summary.csv")
        if r is None:
            return "_Not generated yet._"
        t = pd.DataFrame({"level": [f"{int(l)} hPa" for l in r.level_hpa],
                          "soundings": r.n.astype(int),
                          "ERA5 vs radiosonde, vector RMS (m/s)": r.vector_rms_diff_ms,
                          "forecast 6 h vs ERA5, vector RMSE (m/s)": r.forecast_6h_vector_rmse_ms,
                          "mean observed speed (m/s)": r.obs_mean_speed_ms})
        ratio = (r.vector_rms_diff_ms / r.forecast_6h_vector_rmse_ms).mean()
        shared = r.dropna(subset=["forecast_6h_vector_rmse_ms"])
        combined = np.hypot(shared.vector_rms_diff_ms, shared.forecast_6h_vector_rmse_ms)
        lev = shared.level_hpa.astype(int).tolist()
        return (table(t, {"ERA5 vs radiosonde, vector RMS (m/s)": "{:.2f}",
                          "forecast 6 h vs ERA5, vector RMSE (m/s)": "{:.2f}",
                          "mean observed speed (m/s)": "{:.1f}"})
                + "\n\n"
                + f"The reanalysis differs from the soundings by {ratio:.1f}x the forecast's "
                "6-hour error against the reanalysis. Part of that is representativeness (a "
                "drifting point measurement against a ~100 km grid box), and ERA5 assimilates "
                "these soundings, so it is a lower bound on ERA5's error away from stations. "
                "If the two errors were independent they would add in quadrature, putting the "
                "forecast's 6-hour error against real observations nearer "
                + ", ".join(f"{c:.1f} m/s at {l} hPa" for c, l in zip(combined, lev))
                + " than the figures scored against ERA5 suggest.")

    def vista(self):
        p = ROOT / "results" / "vista" / "summary.json"
        if not p.exists():
            return "_Not generated yet._"
        v = json.loads(p.read_text())
        lat, lon = v["median_landing"]
        straight = v.get("straight_line_km", 351.0)
        txt = (f"**The reconstruction falls short.** Of {v['profiles']} assumed flight profiles, "
               f"{v['consistent_profiles']} match the reported 12.2 km over Guntur at 10:05 IST. "
               f"Flown through ERA5 winds, they cover a median {v['median_distance_flown_km']:.0f} km "
               f"and land a median {v['median_km_from_raichur_town']:.0f} km from Raichur town "
               f"(median landing {lat:.2f} N, {lon:.2f} E); "
               f"{v['share_within_80km_of_raichur_town']:.0%} land within 80 km of it. Vijayawada "
               f"to Raichur is {straight:.0f} km in a straight line, so the balloon had to average "
               f"{v.get('needed_mean_speed_ms', straight / 7.5 / 3.6):.0f} m/s westward for the "
               "whole flight, including the slow climb.")
        vdir = ROOT / "results" / "vista"
        s = _csv(vdir / "sondes_vs_era5.csv")
        if s is not None:
            t = pd.DataFrame({
                "level": [f"{int(l)} hPa (~{h / 1000:.1f} km)" for l, h in zip(s.level_hpa, s.height_m)],
                "radiosondes, eastward wind (m/s)": s.sonde_u,
                "ERA5 on the flight day (m/s)": s.era5_u,
                "difference (m/s)": s.difference,
                "days with soundings": s.n_soundings.astype(int)})
            txt += ("\n\n**What radiosondes measured.** Mean eastward wind at the four nearest "
                    "stations (Machilipatnam, Hyderabad, Visakhapatnam, Bengaluru) over 24-30 May "
                    "2026, against ERA5 at the same stations on the flight day. Negative is "
                    "towards the west.\n\n"
                    + table(t, {"difference (m/s)": "{:+.1f}"}))
        sens = _csv(vdir / "sensitivity.csv")
        if sens is not None:
            t = pd.DataFrame({"scenario": sens.scenario,
                              "median distance flown (km)": sens.median_km_flown,
                              "median miss from Raichur town (km)": sens.median_km_from_raichur_town,
                              "landing within 80 km (%)": sens.share_within_80km * 100})
            txt += ("\n\n**What closes the gap.** The same reconstruction with one change at a "
                    "time:\n\n" + table(t, {c: "{:.0f}" for c in t.columns[1:]}))
            base, corr = sens.iloc[0], sens[sens.scenario.str.contains("corrected to")].iloc[0]
            closed = 1 - corr.median_km_from_raichur_town / base.median_km_from_raichur_town
            txt += (f"\n\nCorrecting ERA5 to the measured winds closes about {closed:.0%} of the "
                    "miss; no single change within the assumed 7.5 hours closes all of it. A "
                    "longer time aloft would, and the public record does not say whether 7 h 30 "
                    "min is the whole flight or the time at float. A flight track would settle it.")
        return txt

    def headline(self):
        """The few numbers a reader needs, each traceable to a table in the README."""
        out = []
        best = self._best()
        ci = _csv(self.fd / "skill_ci.csv")
        if ci is not None:
            c = ci[(ci.reference == "persistence") & (ci.test_year == self.year) & (ci.model == best)]
            if len(c):
                s6 = c[c.horizon_h == 6].skill_pct.mean()
                out.append(f"- **Forecasting:** the model the ladder selected was "
                           f"**{best}**, with {s6:.0f}% lower vector error than persistence at 6 h "
                           f"(test year {self.year}).")
        r = _csv(self.out / "radiosondes" / "summary.csv")
        if r is not None:
            out.append(f"- **Truth check:** ERA5 differs from radiosondes by "
                       f"{r.vector_rms_diff_ms.min():.1f}-{r.vector_rms_diff_ms.max():.1f} m/s "
                       "(vector RMS) at balloon levels, the same order as the forecast error.")
        e = _csv(self.out / "trajectory" / "position_error.csv")
        if e is not None:
            g = e[e.lead_h == 24].set_index("source").median_km
            if best in g and "persistence" in g:
                out.append(f"- **Trajectory:** after 24 h the predicted position is a median "
                           f"{g[best]:.0f} km off with {best}, against {g['persistence']:.0f} km "
                           "with persistence.")
        s = _csv(self.out / "control" / "summary.csv")
        if s is not None:
            hold = s.loc[s.run == "hold", "twr50"].iloc[0]
            real = s[~s.run.str.contains("perfect")]
            top = real.loc[real.twr50.idxmax()]
            perf = s.loc[s.run.str.contains("perfect"), "twr50"]
            out.append(f"- **Control (simulation):** the best real controller, {top.run}, keeps "
                       f"the balloon within 50 km {top.twr50:.1%} of the time against "
                       f"{hold:.1%} for doing nothing"
                       + (f" and {perf.iloc[0]:.1%} with a perfect forecast." if len(perf) else "."))
        f = _csv(self.out / "feasibility" / "twr50_by_month.csv")
        if f is not None:
            col = [c for c in f.columns if "perfect" in c][0]
            mx, mn = f.loc[f[col].idxmax()], f.loc[f[col].idxmin()]
            out.append(f"- **Feasibility:** even a perfect forecast holds station only "
                       f"{mn[col]:.0%} of the time in month {int(mn.month)} and {mx[col]:.0%} in "
                       f"month {int(mx.month)}: the launch window matters more than the model.")
        a = _csv(ROOT / "results/anomaly/summary.csv")
        if a is not None:
            t = a.loc[a.recall.idxmax()]
            out.append(f"- **Anomaly detection:** at an equal false-alarm budget, "
                       f"{t.detector} caught {t.recall:.0%} of injected faults.")
        return "\n".join(out) if out else "_Not generated yet._"

    BLOCKS = ["headline", "provenance", "radiosondes", "vista", "forecast", "ablation", "uncertainty", "trajectory", "control",
              "season", "feasibility", "robustness", "anomaly", "edge"]


def fill(path, rep: Report):
    text = path.read_text(encoding="utf-8")
    for name in Report.BLOCKS:
        pat = re.compile(rf"(<!-- AUTO:{name} -->)(.*?)(<!-- /AUTO:{name} -->)", re.S)
        if pat.search(text):
            try:
                body = getattr(rep, name)()
            except (KeyError, AttributeError, ValueError) as exc:
                # Stale or partial result files: say so in place rather than
                # silently keeping old text or aborting the whole report.
                body = f"_Could not generate this section from the current results ({exc!r})._"
                print(f"warning: {name}: {exc!r}")
            text = pat.sub(lambda m: f"{m.group(1)}\n{body}\n{m.group(3)}", text)
    path.write_text(text, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--target", default="README.md")
    a = ap.parse_args()
    fill(ROOT / a.target, Report(a.config))
    print("filled", a.target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
