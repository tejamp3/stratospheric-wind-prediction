"""Verification metrics for wind-vector forecasts, with honest uncertainty.

Samples pooled over many columns at the same time are not independent, and
consecutive times are strongly autocorrelated. Confidence intervals therefore
come from a block bootstrap that resamples whole multi-day blocks of issue
times, every column included, never individual samples.
"""
from __future__ import annotations

import numpy as np


def direction_deg(u, v):
    """Meteorological direction the wind comes FROM, degrees clockwise from north."""
    return np.degrees(np.arctan2(-u, -v)) % 360


def angular_error(a, b):
    d = np.abs(a - b) % 360
    return np.minimum(d, 360 - d)


def scores(pred: np.ndarray, true: np.ndarray, min_speed: float = 2.0) -> dict[str, float]:
    """Metric set for (N, 2) wind vectors."""
    err = pred - true
    ps, ts = np.hypot(*pred.T), np.hypot(*true.T)
    ok = ts >= min_speed
    ang = angular_error(direction_deg(*pred[ok].T), direction_deg(*true[ok].T))
    corr = np.mean([np.corrcoef(pred[:, i], true[:, i])[0, 1] for i in (0, 1)])
    return {
        "vector_rmse": float(np.sqrt(np.mean(np.sum(err ** 2, axis=1)))),
        "component_mae": float(np.mean(np.abs(err))),
        "bias_ms": float(np.hypot(*err.mean(0))),
        "bias_u": float(err[:, 0].mean()), "bias_v": float(err[:, 1].mean()),
        "correlation": float(corr),
        "speed_rmse": float(np.sqrt(np.mean((ps - ts) ** 2))),
        "direction_mae_deg": float(ang.mean()) if ang.size else float("nan"),
        "n": int(len(true)),
    }


def block_ids(t_idx: np.ndarray, block_steps: int) -> np.ndarray:
    return (t_idx - t_idx.min()) // block_steps


def bootstrap_skill(err_model: np.ndarray, err_ref: np.ndarray, blocks: np.ndarray,
                    n_boot: int = 1000, seed: int = 0) -> dict[str, float]:
    """Skill = 1 - RMSE_model / RMSE_ref with a block-bootstrap 95% CI.

    err_* are squared vector errors per sample (N,), paired on the same samples,
    so the interval is for the difference between the two models, not for
    either one alone.
    """
    rng = np.random.default_rng(seed)
    ub = np.unique(blocks)
    pos = np.searchsorted(ub, blocks)
    sm = np.bincount(pos, err_model, len(ub))
    sr = np.bincount(pos, err_ref, len(ub))
    cnt = np.bincount(pos, minlength=len(ub))

    def skill(w):
        return 1 - np.sqrt((w @ sm) / (w @ cnt)) / np.sqrt((w @ sr) / (w @ cnt))
    point = float(skill(np.ones(len(ub))))
    draws = []
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, len(ub), len(ub)), minlength=len(ub)).astype(float)
        draws.append(skill(w))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"skill_pct": 100 * point, "ci_lo_pct": 100 * float(lo),
            "ci_hi_pct": 100 * float(hi), "n_blocks": int(len(ub))}
