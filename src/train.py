"""Train the LSTM wind forecaster.

Usage:  python src/train.py [--epochs 100] [--hidden 128] [--tag base]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
import data as D
from model import WindLSTM, assert_residual_safe

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("train")


def loaders(ds: dict, batch_size: int):
    out = {}
    for split in ("train", "val", "test"):
        t = TensorDataset(torch.from_numpy(ds[split]["X"]),
                          torch.from_numpy(ds[split]["Y"]))
        # Shuffling windows within the train split is fine: the split itself is
        # chronological, so no future information crosses into training.
        out[split] = DataLoader(t, batch_size=batch_size,
                                shuffle=(split == "train"), drop_last=False)
    return out


@torch.no_grad()
def evaluate_epoch(model, loader, loss_fn, device) -> dict[str, float]:
    model.eval()
    tot = n = 0.0
    abs_sum = sq_sum = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        p = model(xb)
        tot += loss_fn(p, yb).item() * len(xb)
        abs_sum += (p - yb).abs().sum().item()
        sq_sum += ((p - yb) ** 2).sum().item()
        n += len(xb)
    cells = n * model.n_horizons * model.n_targets
    return {"loss": tot / n, "mae": abs_sum / cells,
            "rmse": float(np.sqrt(sq_sum / cells))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=C.MAX_EPOCHS)
    ap.add_argument("--hidden", type=int, default=C.HIDDEN)
    ap.add_argument("--layers", type=int, default=C.NUM_LAYERS)
    ap.add_argument("--dropout", type=float, default=C.DROPOUT)
    ap.add_argument("--lr", type=float, default=C.LR)
    ap.add_argument("--weight-decay", type=float, default=C.WEIGHT_DECAY,
                    help="L2 strength via AdamW decoupled weight decay")
    ap.add_argument("--batch-size", type=int, default=C.BATCH_SIZE)
    ap.add_argument("--patience", type=int, default=C.PATIENCE)
    ap.add_argument("--tag", default="base", help="name for checkpoint/history files")
    ap.add_argument("--input-hours", type=int, default=C.INPUT_HOURS,
                    help="length of the history window in hours")
    ap.add_argument("--time-features", action="store_true",
                    help="append hour-of-day sine/cosine to the inputs; the EDA "
                         "shows a 24 h tide worth ~81%% of v's standard deviation")
    ap.add_argument("--residual", action="store_true",
                    help="predict a correction to persistence instead of the "
                         "wind itself; usually much stronger on this series")
    args = ap.parse_args()

    torch.manual_seed(C.SEED)
    np.random.seed(C.SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("device=%s  torch=%s", device, torch.__version__)

    input_steps = max(1, args.input_hours // C.STEP_HOURS)
    features = D.FEATURES_WITH_TIME if args.time_features else D.FEATURES
    ds, sx, sy, df = D.build_dataset(input_steps=input_steps,
                                     feature_names=features)
    sx.save(C.DATA_PROC / "scaler_x.npz")
    sy.save(C.DATA_PROC / "scaler_y.npz")
    dl = loaders(ds, args.batch_size)

    if args.residual:
        assert_residual_safe(sx, sy)
    model = WindLSTM(n_features=len(features), hidden=args.hidden,
                     num_layers=args.layers, dropout=args.dropout,
                     residual=args.residual).to(device)
    log.info("model params=%d  residual=%s  input=%dh (%d steps)  features=%d",
             model.n_params, args.residual, args.input_hours, input_steps,
             len(features))

    # L2 regularisation enters through AdamW's decoupled weight decay.
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=4)
    loss_fn = nn.MSELoss()

    ckpt = C.MODELS / f"lstm_{args.tag}.pt"
    history: list[dict] = []
    best, best_epoch, bad = float("inf"), -1, 0
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        run = n = 0.0
        for xb, yb in dl["train"]:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            # Recurrent nets can spike; clipping keeps the run stable.
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            run += loss.item() * len(xb)
            n += len(xb)
        tr = {"loss": run / n}
        tr.update({k: v for k, v in evaluate_epoch(model, dl["train"], loss_fn, device).items()
                   if k != "loss"})
        va = evaluate_epoch(model, dl["val"], loss_fn, device)
        sched.step(va["loss"])

        history.append({"epoch": epoch, "lr": opt.param_groups[0]["lr"],
                        **{f"train_{k}": v for k, v in tr.items()},
                        **{f"val_{k}": v for k, v in va.items()}})
        log.info("epoch %3d  train_loss %.5f  val_loss %.5f  val_mae %.4f  val_rmse %.4f",
                 epoch, tr["loss"], va["loss"], va["mae"], va["rmse"])

        if va["loss"] < best - 1e-6:
            best, best_epoch, bad = va["loss"], epoch, 0
            torch.save({"state_dict": model.state_dict(),
                        "config": {"n_features": len(features), "hidden": args.hidden,
                                   "num_layers": args.layers, "dropout": args.dropout,
                                   "horizons": C.HORIZONS, "features": features,
                                   "targets": D.TARGETS,
                                   "input_hours": args.input_hours,
                                   "input_steps": input_steps,
                                   "residual": args.residual},
                        "epoch": epoch, "val_loss": best}, ckpt)
        else:
            bad += 1
            if bad >= args.patience:
                log.info("early stop at epoch %d (best %d, val_loss %.5f)",
                         epoch, best_epoch, best)
                break

    mins = (time.time() - t0) / 60
    log.info("done in %.1f min; best epoch %d val_loss %.5f -> %s",
             mins, best_epoch, best, ckpt.name)

    hist_path = C.METRICS / f"history_{args.tag}.json"
    hist_path.write_text(json.dumps(
        {"tag": args.tag, "best_epoch": best_epoch, "best_val_loss": best,
         "minutes": mins, "n_params": model.n_params,
         # Recording the dataset makes runs comparable. Two runs trained on
         # different amounts of data are not a controlled comparison, and
         # without this the difference is invisible in the results.
         "dataset": {"n_steps": int(len(df)),
                     "n_train_windows": int(len(ds["train"]["X"])),
                     "n_val_windows": int(len(ds["val"]["X"])),
                     "n_test_windows": int(len(ds["test"]["X"])),
                     "period": [str(df.index[0]), str(df.index[-1])],
                     "n_raw_files": len(list(C.DATA_RAW.glob("era5_strat_*.nc")))},
         "args": vars(args), "history": history}, indent=1))
    log.info("history -> %s", hist_path.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
