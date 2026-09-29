"""Small hyperparameter sweep, selected on validation loss.

Training one model here takes seconds on CPU, so there is no excuse for picking
the architecture by assertion. This runs a grid, ranks it by the best validation
loss each configuration reached, and writes the table that `skills.md` reports.

The test split is never consulted. Selection is on validation only; the winner is
scored on test exactly once, afterwards, by `evaluate.py`.

Usage:
    python src/sweep.py                 # the default grid
    python src/sweep.py --quick         # a smaller grid
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("sweep")

PYTHON = sys.executable
TRAIN = str(Path(__file__).resolve().parent / "train.py")

# The grid targets the diagnosed failure mode. A first fit at the brief's
# 2 x 128 default reached train loss 0.29 against validation 0.74: it is
# memorising a few thousand windows rather than generalising, which is exactly
# why a ridge baseline was competitive. So the axes swept are the ones that
# control capacity and regularisation - width, depth, dropout and weight decay.
GRID = {
    "hidden": [16, 32, 64, 128],
    "layers": [1, 2],
    "dropout": [0.2, 0.4],
    "weight_decay": [1e-5, 1e-3],
    "residual": [True],
    "input_hours": [24],
}
# A second, smaller grid for the framing and history-length questions, held
# separate so the main grid stays legible.
ABLATION = {
    "hidden": [32],
    "layers": [1, 2],
    "dropout": [0.2],
    "weight_decay": [1e-3],
    "residual": [True, False],
    # The brief specifies a 24 h history; the autocorrelation runs to days, so
    # more context may help most at the 24 h lead where skill is weakest.
    "input_hours": [24, 72, 120],
}
QUICK = {
    "hidden": [32, 128],
    "layers": [2],
    "dropout": [0.2],
    "weight_decay": [1e-5],
    "residual": [True, False],
    "input_hours": [24],
}


def tag_for(cfg: dict) -> str:
    return (f"sw_h{cfg['hidden']}_l{cfg['layers']}"
            f"_d{str(cfg['dropout']).replace('.', '')}"
            f"_wd{cfg['weight_decay']:g}".replace("-", "") +
            f"_w{cfg['input_hours']}"
            f"_{'res' if cfg['residual'] else 'abs'}")


def run_one(cfg: dict, epochs: int) -> dict | None:
    tag = tag_for(cfg)
    cmd = [PYTHON, TRAIN, "--tag", tag, "--epochs", str(epochs),
           "--hidden", str(cfg["hidden"]), "--layers", str(cfg["layers"]),
           "--dropout", str(cfg["dropout"]),
           "--weight-decay", str(cfg["weight_decay"]),
           "--input-hours", str(cfg["input_hours"])]
    if cfg["residual"]:
        cmd.append("--residual")

    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        log.error("%s failed:\n%s", tag, proc.stderr[-600:])
        return None

    hist = json.loads((C.METRICS / f"history_{tag}.json").read_text())
    return {
        "tag": tag,
        **{k: cfg[k] for k in ("hidden", "layers", "dropout", "weight_decay",
                               "residual", "input_hours")},
        "n_train": hist.get("dataset", {}).get("n_train_windows", -1),
        "n_files": hist.get("dataset", {}).get("n_raw_files", -1),
        "weights": hist["n_params"],
        "best_epoch": hist["best_epoch"],
        "epochs_run": len(hist["history"]),
        "train_loss_at_best": next(
            h["train_loss"] for h in hist["history"]
            if h["epoch"] == hist["best_epoch"]),
        "val_loss": hist["best_val_loss"],
        "val_mae": min(h["val_mae"] for h in hist["history"]),
        "minutes": round((time.time() - t0) / 60, 2),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--ablation", action="store_true",
                    help="sweep target framing and history length instead")
    ap.add_argument("--epochs", type=int, default=C.MAX_EPOCHS)
    args = ap.parse_args()

    grid = QUICK if args.quick else (ABLATION if args.ablation else GRID)
    combos = [dict(zip(grid, v)) for v in itertools.product(*grid.values())]
    log.info("sweeping %d configurations", len(combos))

    rows = []
    for i, cfg in enumerate(combos, 1):
        r = run_one(cfg, args.epochs)
        if r:
            rows.append(r)
            log.info("[%2d/%d] %-28s val_loss %.5f  (%d weights, %.1f min)",
                     i, len(combos), r["tag"], r["val_loss"], r["weights"],
                     r["minutes"])

    if not rows:
        log.error("every configuration failed")
        return 1

    table = pd.DataFrame(rows).sort_values("val_loss").reset_index(drop=True)

    # A sweep is only a comparison if every configuration saw the same data. If
    # the dataset changed underneath the run - a download still landing files,
    # say - the ranking is meaningless, so say so loudly rather than report it.
    sizes = set(table["n_train"].tolist())
    if len(sizes) > 1:
        log.error("DATASET CHANGED DURING THE SWEEP: train-window counts %s. "
                  "Configurations saw different data, so this ranking is not a "
                  "controlled comparison. Re-run once the data is stable.",
                  sorted(sizes))
        table["INVALID_dataset_changed"] = True

    table.to_csv(C.METRICS / "sweep.csv", index=False)
    print("\n" + table.to_string(index=False) + "\n")

    best = table.iloc[0]
    if len(sizes) > 1:
        log.error("not promoting a winner from an invalid sweep")
        return 2
    log.info("best on validation: %s (val_loss %.5f, %d train windows)",
             best.tag, best.val_loss, best.n_train)
    (C.METRICS / "sweep_best.json").write_text(json.dumps({
        "best_tag": best.tag,
        "selected_on": "validation loss; the test split was not consulted",
        "hidden": int(best.hidden), "layers": int(best.layers),
        "dropout": float(best.dropout), "residual": bool(best.residual),
        "weight_decay": float(best.weight_decay),
        "input_hours": int(best.input_hours),
        "val_loss": float(best.val_loss), "weights": int(best.weights),
        "n_configurations": len(table),
        "n_train_windows": int(best.n_train),
        "n_raw_files": int(best.n_files),
    }, indent=1))
    print("To promote it:\n"
          f"  python src/train.py --tag base --hidden {int(best.hidden)} "
          f"--layers {int(best.layers)} --dropout {best.dropout} "
          f"--weight-decay {best.weight_decay:g} "
          f"--input-hours {int(best.input_hours)}"
          f"{' --residual' if best.residual else ''}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
