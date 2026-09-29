"""Forecast verification metrics for wind vectors.

All functions take (u, v) arrays in m/s and operate on the last axis pair, so
they work for any leading shape.
"""
from __future__ import annotations

import numpy as np


def speed(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.hypot(u, v)


def direction(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Meteorological direction (degrees the wind comes FROM, clockwise from N)."""
    return np.degrees(np.arctan2(-u, -v)) % 360


def angular_error(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest signed-magnitude separation between two bearings, in [0, 180]."""
    d = np.abs(a - b) % 360
    return np.minimum(d, 360 - d)


def mae(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.abs(x - y)))


def rmse(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.sqrt(np.mean((x - y) ** 2)))


def vector_rmse(pu, pv, tu, tv) -> float:
    """RMSE of the error vector magnitude - the quantity that matters for drift."""
    return float(np.sqrt(np.mean((pu - tu) ** 2 + (pv - tv) ** 2)))


def evaluate(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    """Full metric set for one forecast horizon.

    pred, true: (N, 2) arrays of (u, v) in m/s.
    """
    pu, pv = pred[:, 0], pred[:, 1]
    tu, tv = true[:, 0], true[:, 1]
    ps, ts = speed(pu, pv), speed(tu, tv)
    pd_, td = direction(pu, pv), direction(tu, tv)
    ang = angular_error(pd_, td)

    # Directional accuracy is only meaningful when there is a wind to have a
    # direction: below ~2 m/s the bearing is numerically unstable and
    # operationally irrelevant, so it is scored on the resolvable subset.
    resolvable = ts >= 2.0
    ang_r = ang[resolvable]

    return {
        "speed_mae": mae(ps, ts),
        "speed_rmse": rmse(ps, ts),
        "speed_bias": float(np.mean(ps - ts)),
        "u_rmse": rmse(pu, tu),
        "v_rmse": rmse(pv, tv),
        "vector_rmse": vector_rmse(pu, pv, tu, tv),
        "maae_deg": float(np.mean(ang_r)) if ang_r.size else float("nan"),
        "dir_acc_15deg": float(np.mean(ang_r <= 15) * 100) if ang_r.size else float("nan"),
        "dir_acc_30deg": float(np.mean(ang_r <= 30) * 100) if ang_r.size else float("nan"),
        "n_resolvable": int(resolvable.sum()),
        "n_total": int(len(ts)),
    }


def skill_score(model_rmse: float, baseline_rmse: float) -> float:
    """Fractional RMSE reduction versus the baseline.

    Positive is better. The task brief writes this as
    (model - baseline) / baseline, which is negative when the model wins; we
    report the sign-flipped version so "higher is better" holds throughout, and
    label it as a percentage improvement.
    """
    if baseline_rmse == 0:
        return float("nan")
    return float((baseline_rmse - model_rmse) / baseline_rmse * 100)
