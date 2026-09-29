"""Inference pipeline: turn recent atmospheric data into a wind forecast.

Two ways to get uncertainty out of the model:

* **Calibrated residual intervals** (default). The spread of the validation
  residuals at each horizon is measured once and stored; a forecast then carries
  an interval that actually covered that fraction of validation cases. This is
  the interval to quote operationally, because it is verified rather than
  assumed.
* **Monte-Carlo dropout**. Dropout is left active and the model is run many
  times, giving the spread the model itself implies. Useful for spotting inputs
  the model finds unfamiliar, but it is not calibrated on its own.

Usage:
    python src/inference.py --calibrate            # fit intervals on validation
    python src/inference.py --demo                 # forecast from the latest data
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
import data as D
import metrics as M
from evaluate import features_of, input_steps_of, load_model, predict

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("inference")

CALIB_PATH = C.MODELS / "calibration.json"


# --------------------------------------------------------------------- results
@dataclass
class Forecast:
    issued_at: str
    horizon_h: int
    u: float
    v: float
    speed: float
    direction_deg: float
    speed_lo: float
    speed_hi: float
    direction_sd_deg: float
    interval_pct: float

    def line(self) -> str:
        return (f"  +{self.horizon_h:>2d} h  "
                f"{self.speed:5.1f} m/s [{self.speed_lo:4.1f}-{self.speed_hi:4.1f}]  "
                f"from {self.direction_deg:5.1f} deg (+/-{self.direction_sd_deg:.0f})")


class Forecaster:
    """Loads a trained model once and forecasts from raw feature windows."""

    def __init__(self, tag: str = "base", device: str = "cpu"):
        self.device = device
        self.model, self.ckpt = load_model(tag, device)
        self.features: list[str] = self.ckpt["config"]["features"]
        self.horizons: list[int] = self.ckpt["config"]["horizons"]
        self.input_steps: int = input_steps_of(self.ckpt)
        self.sx = D.Scaler.load(C.DATA_PROC / "scaler_x.npz")
        self.sy = D.Scaler.load(C.DATA_PROC / "scaler_y.npz")
        self.calib = json.loads(CALIB_PATH.read_text()) if CALIB_PATH.exists() else None

    # ------------------------------------------------------------- core predict
    def _window_from_frame(self, df: pd.DataFrame) -> np.ndarray:
        """Validate and normalise the most recent input_steps rows of a frame."""
        missing = [f for f in self.features if f not in df.columns]
        if missing:
            raise ValueError(f"input is missing required columns: {missing}")
        if len(df) < self.input_steps:
            raise ValueError(f"need {self.input_steps} timesteps, got {len(df)}")
        win = df[self.features].to_numpy("float32")[-self.input_steps:]
        if not np.isfinite(win).all():
            raise ValueError("input window contains NaN or inf; clean it first")
        return self.sx.transform(win)[None, ...].astype("float32")

    def raw_predict(self, xn: np.ndarray) -> np.ndarray:
        """(1, steps, F) normalised in -> (n_horizons, 2) m/s out."""
        out = predict(self.model, xn, device=self.device)[0]
        return self.sy.inverse(out)

    def mc_dropout(self, xn: np.ndarray, n: int = 100) -> tuple[np.ndarray, np.ndarray]:
        """Mean and per-horizon standard deviation over n stochastic passes."""
        self.model.train()          # re-enable dropout
        with torch.no_grad():
            xb = torch.from_numpy(np.repeat(xn, n, axis=0)).to(self.device)
            draws = self.model(xb).cpu().numpy()
        self.model.eval()
        draws = self.sy.inverse(draws.reshape(-1, 2)).reshape(n, len(self.horizons), 2)
        return draws.mean(0), draws.std(0)

    # ---------------------------------------------------------------- public API
    def forecast(self, df: pd.DataFrame, use_mc: bool = False,
                 mc_samples: int = 100) -> list[Forecast]:
        """Forecast from a DataFrame whose last rows are the most recent data."""
        xn = self._window_from_frame(df)
        point = self.raw_predict(xn)
        _, sd = self.mc_dropout(xn, mc_samples) if use_mc else (None, None)
        issued = str(df.index[-1])

        out = []
        for k, h in enumerate(self.horizons):
            u, v = float(point[k, 0]), float(point[k, 1])
            spd = float(M.speed(u, v))
            drc = float(M.direction(u, v))
            if use_mc:
                half = 1.96 * float(np.hypot(*sd[k]))
                lo, hi, pct = max(0.0, spd - half), spd + half, 95.0
                dsd = float(np.degrees(np.hypot(*sd[k]) / max(spd, 1e-3)))
            elif self.calib:
                cal = self.calib["per_horizon"][str(h)]
                lo, hi = max(0.0, spd + cal["speed_q_lo"]), spd + cal["speed_q_hi"]
                pct = self.calib["interval_pct"]
                dsd = cal["dir_abs_error_p50"]
            else:
                lo = hi = spd
                pct = 0.0
                dsd = float("nan")
            out.append(Forecast(issued, h, u, v, spd, drc, lo, hi, dsd, pct))
        return out


# ------------------------------------------------------------------- calibration
def calibrate(tag: str = "base", interval_pct: float = 90.0) -> dict:
    """Measure validation residual quantiles so intervals mean something.

    The interval is [forecast + q_lo, forecast + q_hi] on the residual
    "actual minus forecast", so it corrects any systematic bias as well as
    describing the spread. One consequence worth knowing: if the model
    consistently under-forecasts, both quantiles are positive and the interval
    sits entirely above the point forecast. That is the calibration doing its
    job, not an error - and a point forecast falling outside its own interval is
    a direct signal of bias worth fixing in the model.
    """
    fc = Forecaster(tag)
    ds, sx, sy, _ = D.build_dataset(verbose=False, input_steps=fc.input_steps,
                                    feature_names=fc.features)
    pred = sy.inverse(predict(fc.model, ds["val"]["X"]).reshape(-1, 2)
                      ).reshape(-1, len(fc.horizons), 2)
    true = sy.inverse(ds["val"]["Y"].reshape(-1, 2)).reshape(pred.shape)

    lo_q = (100 - interval_pct) / 2
    per = {}
    for k, h in enumerate(fc.horizons):
        ps = M.speed(pred[:, k, 0], pred[:, k, 1])
        ts = M.speed(true[:, k, 0], true[:, k, 1])
        resid = ts - ps                              # actual minus forecast
        ang = M.angular_error(M.direction(pred[:, k, 0], pred[:, k, 1]),
                              M.direction(true[:, k, 0], true[:, k, 1]))
        per[str(h)] = {
            "speed_q_lo": float(np.percentile(resid, lo_q)),
            "speed_q_hi": float(np.percentile(resid, 100 - lo_q)),
            "speed_resid_sd": float(resid.std()),
            "dir_abs_error_p50": float(np.percentile(ang, 50)),
            "dir_abs_error_p90": float(np.percentile(ang, 90)),
        }
    out = {"tag": tag, "interval_pct": interval_pct,
           "n_val_windows": int(len(pred)), "per_horizon": per}
    CALIB_PATH.write_text(json.dumps(out, indent=1))
    log.info("calibration -> %s", CALIB_PATH.name)

    # Verify the intervals on the held-out test split: coverage should land near
    # the nominal level. An interval nobody checked is not an interval.
    tp = sy.inverse(predict(fc.model, ds["test"]["X"]).reshape(-1, 2)
                    ).reshape(-1, len(fc.horizons), 2)
    tt = sy.inverse(ds["test"]["Y"].reshape(-1, 2)).reshape(tp.shape)
    cov = {}
    for k, h in enumerate(fc.horizons):
        ps = M.speed(tp[:, k, 0], tp[:, k, 1])
        ts = M.speed(tt[:, k, 0], tt[:, k, 1])
        c = per[str(h)]
        inside = (ts >= ps + c["speed_q_lo"]) & (ts <= ps + c["speed_q_hi"])
        cov[str(h)] = round(float(inside.mean() * 100), 2)
        log.info("horizon %2d h: nominal %.0f%% interval covers %.1f%% of test cases",
                 h, interval_pct, cov[str(h)])
    out["test_coverage_pct"] = cov
    CALIB_PATH.write_text(json.dumps(out, indent=1))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="base")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--demo", action="store_true",
                    help="forecast from the final window of the downloaded record")
    ap.add_argument("--mc", action="store_true", help="use Monte-Carlo dropout instead")
    ap.add_argument("--interval", type=float, default=90.0)
    args = ap.parse_args()

    if args.calibrate:
        calibrate(args.tag, args.interval)
    if args.demo or not args.calibrate:
        df, _ = D.clean(D.add_derived(D.station_frame(D.open_raw())))
        fc = Forecaster(args.tag)
        rows = fc.forecast(df, use_mc=args.mc)
        print(f"\n50 hPa wind forecast for {C.TARGET_LAT:.0f}N {C.TARGET_LON:.0f}E")
        print(f"issued from data up to {rows[0].issued_at} "
              f"({'MC dropout' if args.mc else f'{rows[0].interval_pct:.0f}% calibrated'} "
              "interval)")
        for r in rows:
            print(r.line())
        print()
        (C.RESULTS / "latest_forecast.json").write_text(
            json.dumps([asdict(r) for r in rows], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
