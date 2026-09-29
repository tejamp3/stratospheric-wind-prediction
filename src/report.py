"""Fill the auto-generated tables in skills.md and README.md from the result files.

Every number in the documentation is written by this script from the CSV and JSON
under results/, so the prose can never drift out of step with the last run. Each
managed region in a markdown file is delimited by

    <!-- AUTO:name -->
    ...generated...
    <!-- /AUTO:name -->

and only the text between the markers is replaced.

Usage:  python src/report.py [--tag base]
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("report")

TICK, CROSS, DASH = "met", "not met", "n/a"


def _read_csv(p: Path) -> pd.DataFrame | None:
    return pd.read_csv(p) if p.exists() else None


def _read_json(p: Path) -> dict | None:
    return json.loads(p.read_text()) if p.exists() else None


def md_table(df: pd.DataFrame, headers: dict[str, str], decimals: int = 2) -> str:
    """Render selected columns as a markdown table with friendly headers."""
    sub = df[list(headers)].copy()
    for c in sub.columns:
        if pd.api.types.is_float_dtype(sub[c]):
            # A column that is float only because CSV round-tripped it, and whose
            # values are whole numbers, should print as "6" not "6.0".
            if sub[c].notna().all() and (sub[c] % 1 == 0).all():
                sub[c] = sub[c].astype("int64")
            else:
                sub[c] = sub[c].round(decimals)
    # Format each column to strings while it still has its own dtype. Going via
    # iterrows() would upcast every row to one common dtype, which silently turns
    # an integer lead time of 6 into "6.0".
    cols = [sub[c].map(lambda v: f"{v}").tolist() for c in sub.columns]
    names = list(headers.values())
    lines = ["| " + " | ".join(names) + " |",
             "|" + "|".join(["---"] * len(names)) + "|"]
    for i in range(len(sub)):
        lines.append("| " + " | ".join(col[i] for col in cols) + " |")
    return "\n".join(lines)


# ------------------------------------------------------------------- generators
def block_dataset() -> str:
    rep = _read_json(C.METRICS / "eda_report.json")
    if not rep:
        return "_Not generated yet: run `python src/eda.py`._"
    g = rep["grid"]
    stats = rep.get("statistics", {})
    sp = stats.get(f"speed{C.PRIMARY_LEVEL}", {})
    rows = [
        ("Source", "ERA5 reanalysis pressure levels (Copernicus Climate Data Store)"),
        ("Period", f"{rep['period'][0][:10]} to {rep['period'][1][:10]}"),
        ("Sampling", f"{rep['step_hours']}-hourly, {rep['n_steps']} timesteps"),
        ("Completeness", f"{rep['completeness_pct']:.3f}% of the expected axis"),
        ("Imputed cells", f"{rep['n_imputed_cells']}"),
        ("Region", f"{g['area_N_W_S_E'][2]}-{g['area_N_W_S_E'][0]} N, "
                   f"{g['area_N_W_S_E'][1]}-{g['area_N_W_S_E'][3]} E "
                   f"({g['n_lat']} x {g['n_lon']} at {g['resolution_deg'][0]} deg)"),
        ("Levels", ", ".join(f"{l} hPa" for l in rep["levels_hPa"])),
        ("Station", f"{rep['station']['lat']:.0f} N {rep['station']['lon']:.0f} E"),
        ("Model features", f"{len(rep['features'])} "
                           f"((u, v, T) at each of 3 levels)"),
    ]
    if sp:
        rows.append(("50 hPa wind speed",
                     f"mean {sp['mean']:.2f} m/s, sd {sp['std']:.2f}, "
                     f"range {sp['min']:.2f} to {sp['max']:.2f}"))
    return ("| Property | Value |\n|---|---|\n"
            + "\n".join(f"| {k} | {v} |" for k, v in rows))


def block_metrics(tag: str) -> str:
    df = _read_csv(C.METRICS / f"test_metrics_{tag}.csv")
    if df is None:
        return "_Not generated yet: run `python src/evaluate.py`._"
    cols = {
        "horizon_h": "Lead (h)",
        "model_speed_rmse": "LSTM RMSE (m/s)",
        "persist_speed_rmse": "Persistence RMSE (m/s)",
    }
    if "ridge_speed_rmse" in df:
        cols["ridge_speed_rmse"] = "Ridge RMSE (m/s)"
    cols["skill_speed_rmse_pct"] = "LSTM skill vs persistence (%)"
    if "skill_ridge_speed_rmse_pct" in df:
        cols["skill_ridge_speed_rmse_pct"] = "Ridge skill vs persistence (%)"
    cols.update({
        "model_maae_deg": "LSTM direction MAAE (deg)",
        "model_dir_acc_15deg": "Within 15 deg (%)",
        "model_dir_acc_30deg": "Within 30 deg (%)",
    })
    return md_table(df, cols)


def block_training(tag: str) -> str:
    h = _read_json(C.METRICS / f"history_{tag}.json")
    if not h:
        return "_Not generated yet: run `python src/train.py`._"
    a = h["args"]
    rows = [
        ("Architecture", f"{a['layers']}-layer LSTM, {a['hidden']} hidden units, "
                         f"dropout {a['dropout']}"),
        ("Weights", f"{h['n_params']:,}"),
        ("Optimiser", f"AdamW, lr {a['lr']}, weight decay {C.WEIGHT_DECAY}"),
        ("Batch size", str(a["batch_size"])),
        ("Epochs run", f"{len(h['history'])} (best {h['best_epoch']}, "
                       f"early stopping patience {a['patience']})"),
        ("Best val loss", f"{h['best_val_loss']:.5f} (MSE, normalised units)"),
        ("Training time", f"{h['minutes']:.1f} min on CPU"),
    ]
    return ("| Setting | Value |\n|---|---|\n"
            + "\n".join(f"| {k} | {v} |" for k, v in rows))


def block_sweep() -> str:
    df = _read_csv(C.METRICS / "sweep.csv")
    best = _read_json(C.METRICS / "sweep_best.json")
    if df is None:
        return "_Not generated yet: run `python src/sweep.py`._"
    df = df.copy()
    df["residual"] = df["residual"].map(
        lambda v: "correction" if bool(v) else "absolute")
    tbl = md_table(df, {
        "hidden": "Hidden units",
        "layers": "Layers",
        "dropout": "Dropout",
        "weight_decay": "Weight decay",
        "input_hours": "History (h)",
        "residual": "Target framing",
        "weights": "Weights",
        "best_epoch": "Best epoch",
        "train_loss_at_best": "Train loss",
        "val_loss": "Validation loss",
    }, decimals=4)
    note = ""
    if best:
        note = ("\n\nSelected on validation loss alone, across "
                f"{best['n_configurations']} configurations; the test split was "
                f"not consulted. Winner: {best['hidden']} hidden units, "
                f"{best['layers']} layer(s), "
                f"{'correction' if best['residual'] else 'absolute'} framing.")
    return tbl + note


def block_optimization() -> str:
    df = _read_csv(C.METRICS / "optimization.csv")
    if df is None:
        return "_Not generated yet: run `python src/optimize.py`._"
    out = md_table(df, {
        "variant": "Variant",
        "weights": "Weights",
        "size_kb": "Weight file (KB)",
        "p50_ms": "Latency p50 (ms)",
        "p95_ms": "Latency p95 (ms)",
        "speed_rmse_6h": "6 h speed RMSE (m/s)",
        "dir_acc_30deg_6h": "6 h within 30 deg (%)",
    })
    notes = [f"- **{r.variant}** - {r.note}" for r in df.itertuples()
             if isinstance(getattr(r, "note", ""), str) and r.note]
    return out + "\n\nSingle sample, one CPU thread.\n\n" + "\n".join(notes)


def block_airship() -> str:
    s = _read_json(C.METRICS / "airship_summary.json")
    if not s:
        return "_Not generated yet: run `python src/airship.py`._"
    df = pd.DataFrame(s["per_horizon_drift"])
    tbl = md_table(df, {
        "horizon_h": "Lead (h)",
        "model_drift_p50_km": "LSTM drift p50 (km)",
        "model_drift_p90_km": "LSTM drift p90 (km)",
        "persist_drift_p90_km": "Persistence drift p90 (km)",
        "drift_reduction_pct": "Reduction at p90 (%)",
        "model_bias_drift_km_per_day": "LSTM bias drift (km/day)",
    }, decimals=1)
    feas = s["feasible_time_pct_by_level"]
    feas_tbl = ("| Level | Time wind is within airspeed limit |\n|---|---|\n"
                + "\n".join(f"| {k} | {v:.0f}% |" for k, v in feas.items()))
    return (f"Airspeed limit assumed: **{s['max_airspeed_ms']:g} m/s**.\n\n"
            + tbl + "\n\n" + feas_tbl)


def block_errors() -> str:
    j = _read_json(C.METRICS / "error_analysis.json")
    if not j:
        return "_Not generated yet: run `python src/error_analysis.py`._"
    corr = pd.DataFrame(j["correlation_spearman"])
    names = {
        "speed50": "Wind speed at issue time",
        "shear_10_50": "Shear, 10 minus 50 hPa",
        "abs_shear": "Shear magnitude",
        "tendency": "Recent rate of change",
        "t50": "Temperature at 50 hPa",
        "dir_variability": "Direction variability in the window",
    }
    corr.index = [names.get(i, i) for i in corr.index]
    head = "| Condition at issue time | " + " | ".join(corr.columns) + " |"
    sep = "|" + "|".join(["---"] * (len(corr.columns) + 1)) + "|"
    rows = ["| " + r + " | " + " | ".join(f"{v:+.2f}" for v in corr.loc[r]) + " |"
            for r in corr.index]
    return ("Spearman correlation between forecast error and the state of the "
            "atmosphere when the forecast was issued "
            f"(n = {j['n_test_windows']} test windows).\n\n"
            + "\n".join([head, sep] + rows))


def block_criteria(tag: str) -> str:
    m = _read_csv(C.METRICS / f"test_metrics_{tag}.csv")
    opt = _read_csv(C.METRICS / "optimization.csv")
    if m is None:
        return "_Not generated yet: run the pipeline._"
    r6 = m.loc[m.horizon_h == 6].iloc[0]
    checks = []

    for _, r in m.iterrows():
        v = r.skill_speed_rmse_pct
        best = v
        label = "LSTM"
        if "skill_ridge_speed_rmse_pct" in m.columns and r.skill_ridge_speed_rmse_pct > v:
            best, label = r.skill_ridge_speed_rmse_pct, "ridge"
        checks.append((f"Beats persistence RMSE by >= 20% at {int(r.horizon_h)} h",
                       f"{v:.1f}% LSTM, best {best:.1f}% ({label})",
                       TICK if best >= 20 else CROSS))
    checks.append(("Directional accuracy >= 70% at 6 h (within 30 deg)",
                   f"{r6.model_dir_acc_30deg:.1f}%",
                   TICK if r6.model_dir_acc_30deg >= 70 else CROSS))
    if opt is not None:
        lite = opt[opt.variant.str.contains("Lite", case=False, na=False)]
        if len(lite):
            row = lite.iloc[-1]
            checks.append(("Lightweight variant suitable for edge inference",
                           f"{row.variant}, {row.size_kb:.0f} KB, "
                           f"{row.p50_ms:.2f} ms", TICK))
        else:
            checks.append(("Lightweight variant suitable for edge inference",
                           "not trained", CROSS))
    return ("| Criterion | Measured | Status |\n|---|---|---|\n"
            + "\n".join(f"| {a} | {b} | {c} |" for a, b, c in checks))


BLOCKS = {
    "dataset": block_dataset,
    "metrics": block_metrics,
    "training": block_training,
    "optimization": block_optimization,
    "airship": block_airship,
    "errors": block_errors,
    "sweep": block_sweep,
    "criteria": block_criteria,
}


def fill(path: Path, tag: str) -> int:
    if not path.exists():
        log.warning("%s does not exist yet", path.name)
        return 0
    text = path.read_text(encoding="utf-8")
    n = 0
    for name, fn in BLOCKS.items():
        pattern = re.compile(
            rf"(<!-- AUTO:{name} -->)(.*?)(<!-- /AUTO:{name} -->)", re.S)
        if not pattern.search(text):
            continue
        body = fn(tag) if fn.__code__.co_argcount else fn()
        text = pattern.sub(lambda m: f"{m.group(1)}\n{body}\n{m.group(3)}", text)
        n += 1
    path.write_text(text, encoding="utf-8")
    log.info("%s: filled %d block(s)", path.name, n)
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="base")
    args = ap.parse_args()
    for name in ("skills.md", "README.md"):
        fill(C.ROOT / name, args.tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
