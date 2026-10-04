"""Statistics for comparing controllers on paired missions."""
from __future__ import annotations

import numpy as np


def paired_ci(a, b, clusters, n_boot: int = 2000, seed: int = 0):
    """Mean of (a - b) over paired missions, with a cluster-bootstrap 95% CI.

    Missions launched in the same week share weather, so they are resampled
    together (by cluster), not as independent draws.
    """
    rng = np.random.default_rng(seed)
    d = np.asarray(a, float) - np.asarray(b, float)
    cu = np.unique(clusters)
    pos = np.searchsorted(cu, clusters)
    s, c = np.bincount(pos, d, len(cu)), np.bincount(pos, minlength=len(cu))
    draws = []
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, len(cu), len(cu)), minlength=len(cu))
        draws.append((w @ s) / (w @ c))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)
