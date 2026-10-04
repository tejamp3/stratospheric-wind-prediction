"""Experiment: which inputs does the forecast actually use?

Two complementary views on the ridge forecaster (the ladder's selected model):

* ablation: retrain without one input group and measure how much worse the
  test-year forecast gets. This answers "do we need this input at all?"
* permutation importance: shuffle one group in the test inputs of the full
  model and measure the damage. This answers "how much does the trained model
  lean on it?" Correlated groups can hide each other here, which is why
  ablation is reported alongside.

Groups: each pressure level, each variable (u, v, T), the older half of the
history, time of day, season, and position.

Usage:  python experiments/forecast_ablation.py --config configs/forecast.yaml
"""
from __future__ import annotations

import argparse
import glob
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from stratoballoon import viz
from stratoballoon.atmosphere import Atmosphere
from stratoballoon.config import ROOT, load_yaml
from stratoballoon.forecasting import features as F
from stratoballoon.forecasting.models import Ridge
from stratoballoon.runlog import Run

log = logging.getLogger("ablation")


def groups(names: list[str], levels) -> dict[str, list[int]]:
    g = {}
    for lev in levels:
        g[f"{int(lev)} hPa level"] = [i for i, n in enumerate(names)
                                       if n[1:].startswith(f"{int(lev)}_")]
    for v, lab in (("u", "eastward wind u"), ("v", "northward wind v"), ("t", "temperature")):
        g[lab] = [i for i, n in enumerate(names) if n.startswith(v) and "_lag" in n]
    lags = sorted({int(n.split("_lag")[1]) for n in names if "_lag" in n})
    old = lags[len(lags) // 2:]
    g["older half of history"] = [i for i, n in enumerate(names)
                                  if "_lag" in n and int(n.split("_lag")[1]) in old]
    g["time of day"] = [names.index("hour_sin"), names.index("hour_cos")]
    g["season"] = [names.index("doy_sin"), names.index("doy_cos")]
    g["position"] = [names.index("lat"), names.index("lon")]
    return g


def rmse(p, y):
    return float(np.sqrt(np.mean(np.sum((p - y) ** 2, -1))))


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = load_yaml(ap.parse_args().config)["experiment"]
    paths = [Path(p) for p in sorted(glob.glob(str(ROOT / cfg["data_glob"])))]
    out = ROOT / cfg["out_dir"] / "ablation"
    run = Run(out, cfg, paths)
    A = Atmosphere.from_files(paths, stride=int(cfg.get("load_stride", 1)))
    step = A.step_hours
    n_hist = int(cfg["history_hours"] // step)
    h_steps = [int(h // step) for h in cfg["horizons_hours"]]
    yi, xi = F.cell_grid(A, cfg["cell_stride"])
    test_year = cfg["test_years"][-1]
    sp = {s.name: s for s in F.rolling_splits(test_year, cfg["first_year"], cfg["embargo_days"])}
    stride = max(1, int(cfg["issue_stride_hours"] // step))
    sam = {k: F.build_samples(A, F.issue_indices(A, v, n_hist, max(h_steps), stride), yi, xi,
                              n_hist, h_steps) for k, v in sp.items()}
    names = sam["train"].feature_names
    full = Ridge().fit(sam["train"], sam["val"], A, h_steps)
    base_pred = full.predict(sam["test"], A, h_steps)
    Y = sam["test"].Y
    rng = np.random.default_rng(0)
    rows = []
    for gname, cols in groups(names, A.levels).items():
        if not cols:
            continue
        keep = [i for i in range(len(names)) if i not in cols]
        sub = {k: F.Samples(s.X[:, keep], s.Y, s.current, s.t_idx, s.cell,
                            [names[i] for i in keep]) for k, s in sam.items()}
        abl = Ridge().fit(sub["train"], sub["val"], A, h_steps).predict(sub["test"], A, h_steps)
        Xp = sam["test"].X.copy()
        Xp[:, cols] = Xp[rng.permutation(len(Xp))][:, cols]
        perm = full.predict(F.Samples(Xp, Y, sam["test"].current, sam["test"].t_idx,
                                      sam["test"].cell, names), A, h_steps)
        for k, h in enumerate(cfg["horizons_hours"]):
            b = rmse(base_pred[:, k], Y[:, k])
            rows.append({"group": gname, "horizon_h": h, "base_rmse": b,
                         "ablation_increase_pct": 100 * (rmse(abl[:, k], Y[:, k]) - b) / b,
                         "permutation_increase_pct": 100 * (rmse(perm[:, k], Y[:, k]) - b) / b})
        log.info("%-24s done", gname)
    df = pd.DataFrame(rows)
    df.to_csv(out / "importance.csv", index=False)
    print(df.pivot(index="group", columns="horizon_h", values="ablation_increase_pct").round(2))
    figure(df, out, test_year)
    run.finish(test_year=test_year)
    return 0


def figure(df, out, test_year):
    import matplotlib.pyplot as plt
    viz.apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, col, title in ((axes[0], "ablation_increase_pct", "Retrained without the group"),
                           (axes[1], "permutation_increase_pct", "Group shuffled in the full model")):
        piv = df.pivot(index="group", columns="horizon_h", values=col)
        piv = piv.loc[piv.mean(1).sort_values().index]
        y = np.arange(len(piv))
        w = 0.8 / piv.shape[1]
        for i, h in enumerate(piv.columns):
            ax.barh(y + i * w, piv[h], height=w * 0.95, color=viz.SERIES[i % 5], label=f"{h} h")
        ax.set_yticks(y + w * (piv.shape[1] - 1) / 2)
        ax.set_yticklabels(piv.index)
        ax.axvline(0, color=viz.INK, lw=1)
        ax.set_xlabel("Increase in vector RMSE (%)")
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
    axes[1].legend(title="lead", loc="lower right", fontsize=8)
    viz.finish(fig, out / "importance.png",
               viz.source_note("model", f"Ridge forecaster, test year {test_year}"))


if __name__ == "__main__":
    raise SystemExit(main())
