"""Experiment: which is the simplest forecaster that is good enough?

Fits every rung of the ladder on rolling-origin splits and scores each on the
test year at every horizon and level, with block-bootstrap confidence intervals
on skill against persistence. Also saves validation residuals (for conformal
calibration) and the fitted models used downstream.

Usage:  python experiments/forecast_ladder.py --config configs/forecast.yaml
"""
from __future__ import annotations

import argparse
import glob
import logging
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from stratoballoon.atmosphere import Atmosphere
from stratoballoon.config import ROOT, load_yaml
from stratoballoon.forecasting import features as F
from stratoballoon.forecasting import metrics as M
from stratoballoon.forecasting.models import ladder
from stratoballoon.runlog import Run

log = logging.getLogger("ladder")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--models", nargs="*", help="override the model list")
    args = ap.parse_args()
    cfg = load_yaml(args.config)["experiment"]
    if args.models:
        cfg["models"] = args.models
    paths = [Path(p) for p in sorted(glob.glob(str(ROOT / cfg["data_glob"])))]
    out = ROOT / cfg["out_dir"]
    run = Run(out, cfg, paths)

    A = Atmosphere.from_files(paths, stride=int(cfg.get("load_stride", 1)))
    step = A.step_hours
    n_hist = int(cfg["history_hours"] // step)
    h_steps = [int(h // step) for h in cfg["horizons_hours"]]
    stride = max(1, int(cfg["issue_stride_hours"] // step))
    yi, xi = F.cell_grid(A, cfg["cell_stride"])
    by_name = {m.name: m for m in ladder()}
    log.info("atmosphere %s, %d levels, %.0f h step; %d training cells; height from %s",
             A.u.shape, len(A.levels), step, len(yi), A.height_source)

    rows, ci_rows = [], []
    for test_year in cfg["test_years"]:
        splits = {s.name: s for s in F.rolling_splits(test_year, cfg["first_year"],
                                                      cfg["embargo_days"])}
        sam = {}
        for name, sp in splits.items():
            # Train every `stride`; evaluate on every cell of the grid stride too.
            ti = F.issue_indices(A, sp, n_hist, max(h_steps), stride)
            sam[name] = F.build_samples(A, ti, yi, xi, n_hist, h_steps)
            log.info("%d %s: %s -> %s, %d samples", test_year, name, sp.start.date(),
                     sp.end.date(), len(sam[name].X))
        truth = sam["test"].truth
        blocks = M.block_ids(sam["test"].t_idx, int(cfg["bootstrap_block_days"] * 24 / step))
        preds = {}
        for mname in cfg["models"]:
            t0 = time.time()
            model = by_name[mname]().fit(sam["train"], sam["val"], A, h_steps)
            fit_s = time.time() - t0
            p_test = model.predict(sam["test"], A, h_steps) + sam["test"].current[:, None]
            p_val = model.predict(sam["val"], A, h_steps) + sam["val"].current[:, None]
            preds[mname] = p_test
            np.save(out / f"val_residual_{mname}_{test_year}.npy",
                    (sam["val"].truth - p_val).astype("float32"))
            with open(out / f"model_{mname}_{test_year}.pkl", "wb") as f:
                pickle.dump(model, f)
            log.info("%-18s fit %.0fs", mname, fit_s)
            for k, h in enumerate(cfg["horizons_hours"]):
                for l, lev in enumerate(A.levels):
                    sc = M.scores(p_test[:, k, l], truth[:, k, l])
                    rows.append({"test_year": test_year, "model": mname, "horizon_h": h,
                                 "level_hpa": int(lev), **sc, "fit_s": round(fit_s, 1)})
        # Skill with CIs against persistence, and against ridge for the models above it.
        sq = {m: np.sum((p - truth) ** 2, axis=-1) for m, p in preds.items()}  # (N,H,L)
        for ref in ("persistence", "ridge"):
            if ref not in sq:
                continue
            for mname in cfg["models"]:
                if mname == ref:
                    continue
                for k, h in enumerate(cfg["horizons_hours"]):
                    for l, lev in enumerate(A.levels):
                        r = M.bootstrap_skill(sq[mname][:, k, l], sq[ref][:, k, l], blocks,
                                              cfg["n_boot"])
                        ci_rows.append({"test_year": test_year, "model": mname,
                                        "reference": ref, "horizon_h": h,
                                        "level_hpa": int(lev), **r})
        np.save(out / f"test_truth_{test_year}.npy", truth.astype("float32"))
        for m, p in preds.items():
            np.save(out / f"test_pred_{m}_{test_year}.npy", p.astype("float32"))
        np.save(out / f"test_tidx_{test_year}.npy", sam["test"].t_idx)
        # wind at issue time, known when the forecast is made: used to condition
        # the uncertainty on the current regime
        np.save(out / f"val_current_{test_year}.npy", sam["val"].current.astype("float32"))
        np.save(out / f"test_current_{test_year}.npy", sam["test"].current.astype("float32"))

    pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
    ci = pd.DataFrame(ci_rows)
    ci.to_csv(out / "skill_ci.csv", index=False)
    view = ci[ci.reference == "persistence"].pivot_table(
        index=["model"], columns=["horizon_h"], values="skill_pct", aggfunc="mean").round(1)
    print("\nVector-RMSE skill vs persistence (%), mean over levels and test years\n", view)
    run.finish(levels_hpa=[int(x) for x in A.levels], height_source=A.height_source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
