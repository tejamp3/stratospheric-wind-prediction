"""Evaluate a trained forecaster against persistence and write figures + tables.

Usage:  python src/evaluate.py [--tag base]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
import data as D
import metrics as M
import viz
from model import WindLSTM, persistence_forecast, RidgeBaseline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("eval")


def input_steps_of(ck: dict) -> int:
    """How long a history the checkpoint was trained on, in timesteps.

    Older checkpoints predate the configurable window and only record hours, so
    fall back to deriving it rather than silently using the current config.
    """
    cfg = ck["config"]
    if "input_steps" in cfg:
        return int(cfg["input_steps"])
    return max(1, int(cfg.get("input_hours", C.INPUT_HOURS)) // C.STEP_HOURS)


def features_of(ck: dict) -> list[str]:
    """The exact feature list the checkpoint was trained on.

    Recorded at training time so a model trained with clock features can never
    be fed a nine-column window, or the reverse, which would misalign every
    channel rather than failing loudly.
    """
    return list(ck["config"]["features"])


def load_model(tag: str, device: str = "cpu") -> tuple[WindLSTM, dict]:
    ck = torch.load(C.MODELS / f"lstm_{tag}.pt", map_location=device, weights_only=False)
    cfg = ck["config"]
    m = WindLSTM(n_features=cfg["n_features"], hidden=cfg["hidden"],
                 num_layers=cfg["num_layers"], dropout=cfg["dropout"],
                 n_horizons=len(cfg["horizons"]),
                 residual=cfg.get("residual", False))
    m.load_state_dict(ck["state_dict"])
    m.eval().to(device)
    return m, ck


@torch.no_grad()
def predict(model, X: np.ndarray, batch: int = 256, device: str = "cpu") -> np.ndarray:
    out = []
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(X[i:i + batch]).to(device)
        out.append(model(xb).cpu().numpy())
    return np.concatenate(out)


def physical(ds_split: dict, pred_norm: np.ndarray, sx: D.Scaler, sy: D.Scaler):
    """Convert normalised model output, labels and last-observed wind to m/s."""
    shape = pred_norm.shape
    pred = sy.inverse(pred_norm.reshape(-1, 2)).reshape(shape)
    true = sy.inverse(ds_split["Y"].reshape(-1, 2)).reshape(ds_split["Y"].shape)
    # Last observed step of each window, back in physical units -> persistence.
    last = sx.inverse(ds_split["X"][:, -1, :])[:, :2]
    base = persistence_forecast(last, n_horizons=shape[1])
    return pred, true, base


def fit_ridge(ds, sy) -> np.ndarray:
    """Train the linear baseline on the same split and return test forecasts."""
    r = RidgeBaseline().fit(ds["train"]["X"], ds["train"]["Y"])
    out = r.predict(ds["test"]["X"])
    return sy.inverse(out.reshape(-1, 2)).reshape(out.shape)


# ----------------------------------------------------------------- metric table
def metric_table(pred, true, base, ridge=None) -> pd.DataFrame:
    keep = ("speed_mae", "speed_rmse", "vector_rmse", "maae_deg",
            "dir_acc_15deg", "dir_acc_30deg")
    rows = []
    for k, h in enumerate(C.HORIZONS):
        mm = M.evaluate(pred[:, k, :], true[:, k, :])
        bb = M.evaluate(base[:, k, :], true[:, k, :])
        row = {
            "horizon_h": h,
            **{f"model_{x}": mm[x] for x in keep},
            **{f"persist_{x}": bb[x] for x in keep},
            "skill_speed_rmse_pct": M.skill_score(mm["speed_rmse"], bb["speed_rmse"]),
            "skill_vector_rmse_pct": M.skill_score(mm["vector_rmse"], bb["vector_rmse"]),
            "n": mm["n_total"],
        }
        if ridge is not None:
            rr = M.evaluate(ridge[:, k, :], true[:, k, :])
            row["ridge_speed_rmse"] = rr["speed_rmse"]
            row["ridge_dir_acc_30deg"] = rr["dir_acc_30deg"]
            row["skill_ridge_speed_rmse_pct"] = M.skill_score(
                rr["speed_rmse"], bb["speed_rmse"])
            row["lstm_vs_ridge_pct"] = M.skill_score(
                mm["speed_rmse"], rr["speed_rmse"])
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------- figures
def fig_timeseries(times, pred, true, base, weeks: int = 3):
    import matplotlib.pyplot as plt
    n = min(weeks * 7 * (24 // C.STEP_HOURS), len(times))
    sl = slice(0, n)
    # The speed panel carries the argument; direction is supporting detail, so it
    # gets less height. A left-aligned title and a legend on the same line need
    # the legend anchored right, or the two collide.
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.4), sharex=True,
                             gridspec_kw={"height_ratios": [1.9, 1],
                                          "hspace": 0.16})
    ps = M.speed(pred[sl, 0, 0], pred[sl, 0, 1])
    ts_ = M.speed(true[sl, 0, 0], true[sl, 0, 1])
    bs = M.speed(base[sl, 0, 0], base[sl, 0, 1])
    pdir = M.direction(pred[sl, 0, 0], pred[sl, 0, 1])
    tdir = M.direction(true[sl, 0, 0], true[sl, 0, 1])

    ax = axes[0]
    ax.plot(times[sl], ts_, color=viz.ACTUAL, label="Actual", lw=2.2)
    ax.plot(times[sl], ps, color=viz.PREDICTED, label="LSTM 6 h forecast")
    ax.plot(times[sl], bs, color=viz.BASELINE, label="Persistence 6 h", lw=1.5, alpha=.85)
    ax.set_ylabel("Wind speed (m/s)")
    ax.set_title(f"50 hPa wind at {C.TARGET_LAT:.0f}N {C.TARGET_LON:.0f}E - "
                 "6 h forecast vs observed")
    ax.legend(ncol=3, loc="lower right", bbox_to_anchor=(1, 1.0))

    ax = axes[1]
    ax.plot(times[sl], tdir, color=viz.ACTUAL, lw=0, marker="o", ms=3.2, label="Actual")
    ax.plot(times[sl], pdir, color=viz.PREDICTED, lw=0, marker="o", ms=3.2,
            label="LSTM 6 h forecast")
    ax.set_ylabel("Direction (from)")
    ax.set_ylim(0, 360)
    ax.set_yticks([0, 90, 180, 270, 360])
    ax.set_yticklabels(["N", "E", "S", "W", "N"])
    ax.legend(ncol=2, loc="lower right", bbox_to_anchor=(1, 1.0))
    # Percentiles of a bearing are meaningless because the scale wraps, so the
    # note reports the thing that actually explains the scatter: bearing becomes
    # ill-conditioned when the wind drops, which is why the directional metrics
    # are scored only above 2 m/s.
    calm = float((ts_ < 2.0).mean() * 100)
    viz.annotate(ax, f"bearing scatters where wind is weak; "
                     f"{calm:.0f}% of this window is below 2 m/s",
                 loc="lower left")
    fig.autofmt_xdate()
    viz.finish(fig, C.FIGURES / "eval_timeseries_6h.png", viz.SOURCE_NOTE)


def fig_error_box(pred, true, base):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    data, labels, colors = [], [], []
    for k, h in enumerate(C.HORIZONS):
        me = np.abs(M.speed(pred[:, k, 0], pred[:, k, 1])
                    - M.speed(true[:, k, 0], true[:, k, 1]))
        be = np.abs(M.speed(base[:, k, 0], base[:, k, 1])
                    - M.speed(true[:, k, 0], true[:, k, 1]))
        data += [me, be]
        labels += [f"{h} h\nLSTM", f"{h} h\npersist"]
        colors += [viz.PREDICTED, viz.BASELINE]
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=.6,
                    showfliers=False, medianprops=dict(color=viz.INK, lw=1.6),
                    whiskerprops=dict(color=viz.MUTED),
                    capprops=dict(color=viz.MUTED))
    for patch, c in zip(bp["boxes"], colors):
        # 2px surface ring so adjacent fills never touch.
        patch.set(facecolor=c, edgecolor=viz.SURFACE, linewidth=2, alpha=.92)
    ax.set_ylabel("Absolute speed error (m/s)")
    ax.set_title("Error grows with lead time; the LSTM holds its margin over persistence")
    viz.annotate(ax, "box: IQR   line: median   whiskers: 1.5x IQR")
    viz.finish(fig, C.FIGURES / "eval_error_by_horizon.png", viz.SOURCE_NOTE)


def fig_scatter(pred, true):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(C.HORIZONS), figsize=(12, 4.3),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (k, h) in zip(axes, enumerate(C.HORIZONS)):
        ps = M.speed(pred[:, k, 0], pred[:, k, 1])
        ts_ = M.speed(true[:, k, 0], true[:, k, 1])
        ax.scatter(ts_, ps, s=9, color=viz.PREDICTED, alpha=.30,
                   edgecolors="none", rasterized=True)
        lim = [0, float(max(ts_.max(), ps.max())) * 1.05]
        ax.plot(lim, lim, color=viz.MUTED, lw=1.2, ls="--", zorder=0)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect("equal")
        r = float(np.corrcoef(ts_, ps)[0, 1])
        ax.set_title(f"{h} h lead")
        viz.annotate(ax, f"r = {r:.3f}\nRMSE = {M.rmse(ps, ts_):.2f} m/s")
        ax.set_xlabel("Observed speed (m/s)")
    axes[0].set_ylabel("Forecast speed (m/s)")
    fig.suptitle("Forecast vs observed 50 hPa wind speed, by lead time",
                 x=0.005, ha="left", fontsize=11, fontweight="semibold")
    viz.finish(fig, C.FIGURES / "eval_scatter_speed.png", viz.SOURCE_NOTE)


def fig_error_by_direction(pred, true):
    """Polar bars: mean speed error in each observed-direction sector."""
    import matplotlib.pyplot as plt
    nb = 12
    edges = np.linspace(0, 360, nb + 1)
    fig, axes = plt.subplots(1, len(C.HORIZONS), figsize=(12.5, 4.8),
                             subplot_kw={"projection": "polar"})
    axes = np.atleast_1d(axes)
    for ax, (k, h) in zip(axes, enumerate(C.HORIZONS)):
        td = M.direction(true[:, k, 0], true[:, k, 1])
        err = np.abs(M.speed(pred[:, k, 0], pred[:, k, 1])
                     - M.speed(true[:, k, 0], true[:, k, 1]))
        means, thetas = [], []
        for i in range(nb):
            sel = (td >= edges[i]) & (td < edges[i + 1])
            if not sel.any():
                # A polar bar with a NaN height raises inside matplotlib's path
                # transform, so empty sectors are skipped rather than drawn.
                continue
            means.append(float(err[sel].mean()))
            thetas.append(np.radians(edges[i] + 180 / nb))
        if not means:
            ax.text(0, 0, "no data", ha="center", va="center", color=viz.MUTED)
        else:
            ax.bar(thetas, means, width=np.radians(26), color=viz.PREDICTED,
                   edgecolor=viz.SURFACE, linewidth=2, alpha=.92)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_xticks(np.radians(np.arange(0, 360, 45)))
        ax.set_xticklabels(["N", "NE", "E", "SE", "S", "SW", "W", "NW"])
        ax.set_title(f"{h} h lead", pad=16)
        ax.tick_params(labelsize=8)
    fig.suptitle("Mean absolute speed error by observed wind direction (m/s)",
                 x=0.005, ha="left", fontsize=11, fontweight="semibold")
    viz.finish(fig, C.FIGURES / "eval_error_by_direction.png", viz.SOURCE_NOTE)


def fig_error_by_month(times, pred, true, base):
    import matplotlib.pyplot as plt
    months = pd.DatetimeIndex(times).month
    present = sorted(set(months))
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for series, color, label in ((pred, viz.PREDICTED, "LSTM 6 h"),
                                (base, viz.BASELINE, "Persistence 6 h")):
        vals = [M.rmse(M.speed(series[months == m, 0, 0], series[months == m, 0, 1]),
                       M.speed(true[months == m, 0, 0], true[months == m, 0, 1]))
                for m in present]
        ax.plot(present, vals, color=color, marker="o", label=label)
    ax.set_xticks(present)
    ax.set_xticklabels([pd.Timestamp(2022, m, 1).strftime("%b") for m in present])
    ax.set_ylabel("Speed RMSE (m/s)")
    ax.set_title("Forecast error by calendar month of the test period")
    ax.legend(loc="upper left")
    viz.finish(fig, C.FIGURES / "eval_error_by_month.png", viz.SOURCE_NOTE)


def fig_skill(table: pd.DataFrame):
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))
    a1.plot(table.horizon_h, table.model_speed_rmse, color=viz.PREDICTED,
            marker="o", label="LSTM")
    a1.plot(table.horizon_h, table.persist_speed_rmse, color=viz.BASELINE,
            marker="o", label="Persistence")
    if "ridge_speed_rmse" in table:
        a1.plot(table.horizon_h, table.ridge_speed_rmse, color=viz.C3,
                marker="o", label="Ridge regression")
    a1.set_xticks(C.HORIZONS)
    a1.set_xlabel("Lead time (h)")
    a1.set_ylabel("Speed RMSE (m/s)")
    a1.set_title("Error vs lead time")
    a1.legend(loc="upper left")

    xs = table.horizon_h.astype(str)
    a2.bar(xs, table.skill_speed_rmse_pct, color=viz.PREDICTED,
           edgecolor=viz.SURFACE, linewidth=2, width=.55)
    a2.axhline(20, color=viz.BASELINE, ls="--", lw=1.4)
    a2.text(0.99, 20, "20% target ", color=viz.BASELINE, fontsize=8.5,
            va="bottom", ha="right", transform=a2.get_yaxis_transform())
    for x, y in zip(xs, table.skill_speed_rmse_pct):
        a2.text(x, y, f"{y:.0f}%", ha="center",
                va="bottom" if y >= 0 else "top", fontsize=9, color=viz.INK)
    a2.set_xlabel("Lead time (h)")
    a2.set_ylabel("RMSE reduction vs persistence (%)")
    a2.set_title("Skill score (higher is better)")
    viz.finish(fig, C.FIGURES / "eval_skill_vs_horizon.png", viz.SOURCE_NOTE)


def fig_training_curves(tag: str):
    import matplotlib.pyplot as plt
    p = C.METRICS / f"history_{tag}.json"
    if not p.exists():
        return
    h = pd.DataFrame(json.loads(p.read_text())["history"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.1))
    a1.plot(h.epoch, h.train_loss, color=viz.PREDICTED, label="Train")
    a1.plot(h.epoch, h.val_loss, color=viz.BASELINE, label="Validation")
    best = int(h.val_loss.idxmin())
    a1.axvline(h.epoch[best], color=viz.MUTED, ls="--", lw=1.2)
    a1.text(h.epoch[best], float(h.val_loss.max()), f"  best epoch {h.epoch[best]}",
            fontsize=8.5, color=viz.MUTED, va="top")
    a1.set_xlabel("Epoch")
    a1.set_ylabel("MSE loss (normalised units)")
    a1.set_title("Training and validation loss")
    a1.legend(loc="upper right")

    a2.plot(h.epoch, h.val_mae, color=viz.PREDICTED, label="Val MAE")
    a2.plot(h.epoch, h.val_rmse, color=viz.BASELINE, label="Val RMSE")
    a2.set_xlabel("Epoch")
    a2.set_ylabel("Normalised error")
    a2.set_title("Validation MAE and RMSE")
    a2.legend(loc="upper right")
    viz.finish(fig, C.FIGURES / "train_curves.png")


# ------------------------------------------------------------- interpretability
def occlusion_importance(model, X: np.ndarray, n: int = 600) -> np.ndarray:
    """How much each input timestep matters, by neutralising it.

    Replacing one step with 0 (which is the train-set mean, because inputs are
    standardised) and measuring the change in output is a direct, assumption-free
    read on which parts of the 24 h history the LSTM actually uses.
    """
    Xs = X[:n]
    base = predict(model, Xs)
    out = []
    for t in range(Xs.shape[1]):
        Xm = Xs.copy()
        Xm[:, t, :] = 0.0
        out.append(float(np.abs(predict(model, Xm) - base).mean()))
    return np.array(out)


def fig_importance(imp: np.ndarray):
    import matplotlib.pyplot as plt
    lags = -(np.arange(len(imp))[::-1]) * C.STEP_HOURS
    fig, ax = plt.subplots(figsize=(8.6, 4.0))
    ax.bar([f"{l}" if l else "0" for l in lags], imp, color=viz.PREDICTED,
           edgecolor=viz.SURFACE, linewidth=2, width=.62)
    ax.set_xlabel("Input timestep (hours relative to forecast time)")
    ax.set_ylabel("Mean absolute change in forecast")
    ax.set_title("Which part of the 24 h history the model relies on")
    viz.annotate(ax, "each step zeroed in turn; inputs are standardised, so 0 = train mean")
    viz.finish(fig, C.FIGURES / "interpretability_timestep_importance.png")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="base")
    args = ap.parse_args()
    viz.apply_style()

    model, ck = load_model(args.tag)
    steps = input_steps_of(ck)
    ds, sx, sy, df = D.build_dataset(input_steps=steps,
                                     feature_names=features_of(ck))
    log.info("loaded %s (epoch %d, val_loss %.5f, %d params, %d-step input)",
             args.tag, ck["epoch"], ck["val_loss"], model.n_params, steps)

    pred_n = predict(model, ds["test"]["X"])
    pred, true, base = physical(ds["test"], pred_n, sx, sy)
    times = ds["test"]["time"]
    ridge = fit_ridge(ds, sy)

    table = metric_table(pred, true, base, ridge)
    csv = C.METRICS / f"test_metrics_{args.tag}.csv"
    table.to_csv(csv, index=False)
    log.info("metrics -> %s", csv.name)
    print("\n" + table.round(3).to_string(index=False) + "\n")

    fig_timeseries(times, pred, true, base)
    fig_error_box(pred, true, base)
    fig_scatter(pred, true)
    fig_error_by_direction(pred, true)
    fig_error_by_month(times, pred, true, base)
    fig_skill(table)
    fig_training_curves(args.tag)
    fig_importance(occlusion_importance(model, ds["test"]["X"]))

    row6 = table.loc[table.horizon_h == 6].iloc[0]
    summary = {
        "tag": args.tag,
        "test_period": [str(times[0]), str(times[-1])],
        "n_test_windows": int(len(times)),
        "per_horizon": table.to_dict("records"),
        "success_criteria": {
            "skill_vs_persistence_ge_20pct": {
                str(h): bool(v >= 20) for h, v in
                zip(table.horizon_h, table.skill_speed_rmse_pct)},
            "dir_acc_30deg_6h_ge_70pct": bool(row6.model_dir_acc_30deg >= 70),
            "dir_acc_15deg_6h": float(row6.model_dir_acc_15deg),
        },
    }
    (C.METRICS / f"test_summary_{args.tag}.json").write_text(json.dumps(summary, indent=1))
    log.info("figures -> %s", C.FIGURES)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
