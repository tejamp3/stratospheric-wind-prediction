"""Shared setup for the downstream experiments.

One YAML config fixes the data, the test year, the altitude band, the balloon
and the forecast models, so the trajectory, uncertainty and control experiments
all use exactly the same objects.
"""
from __future__ import annotations

import glob
import pickle
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd

from stratoballoon.atmosphere import Atmosphere
from stratoballoon.config import ROOT, load_yaml
from stratoballoon.control import Band, Rates
from stratoballoon.dynamics import BalloonParams
from stratoballoon.forecasting.provider import FieldForecast, TruthProvider
from stratoballoon.uncertainty import VectorConformal


class Context:
    def __init__(self, config_path: str):
        self.cfg = load_yaml(config_path)["experiment"]
        c = self.cfg
        self.paths = [Path(p) for p in sorted(glob.glob(str(ROOT / c["data_glob"])))]
        self.out = ROOT / c["out_dir"]
        self.out.mkdir(parents=True, exist_ok=True)
        self.forecast_dir = ROOT / c["forecast_dir"]
        self.test_year = int(c["test_year"])
        self.horizons = list(c["horizons_hours"])
        self.seed = int(c.get("seed", 0))

    # ------------------------------------------------------------- atmosphere
    @cached_property
    def A(self) -> Atmosphere:
        return Atmosphere.from_files(self.paths, self.cfg.get("load_start"),
                                     self.cfg.get("load_end"))

    @property
    def n_hist(self) -> int:
        return int(self.cfg["history_hours"] // self.A.step_hours)

    @property
    def test_start(self) -> pd.Timestamp:
        return pd.Timestamp(self.test_year, 1, 4)

    @property
    def test_end(self) -> pd.Timestamp:
        return pd.Timestamp(self.test_year, 12, 31, 18)

    # ----------------------------------------------------------- vehicle
    @cached_property
    def band(self) -> Band:
        bottom, top = self.cfg["band_hpa"]
        m = float(self.cfg.get("band_margin_m", 200))
        return Band(self.A.altitude_of_level(bottom) + m, self.A.altitude_of_level(top) - m,
                    int(self.cfg.get("band_grid", 5)))

    @cached_property
    def params(self) -> BalloonParams:
        """Envelope and ballonet sized so the band is exactly reachable."""
        rng = np.random.default_rng(self.seed)
        t = pd.DatetimeIndex(self.A.times)
        idx = np.where((t >= self.test_start) & (t <= self.test_end))[0]
        n = 2000
        hrs = self.A.hours[rng.choice(idx, n)]
        lat, lon = rng.uniform(10, 30, n), rng.uniform(66, 94, n)
        top = self.A.sample(hrs, lat, lon, np.full(n, self.band.alt_max))["rho"].mean()
        bot = self.A.sample(hrs, lat, lon, np.full(n, self.band.alt_min))["rho"].mean()
        return BalloonParams.design(float(top), float(bot),
                                    structure_kg=float(self.cfg.get("structure_kg", 120)))

    @property
    def rates(self) -> Rates:
        return Rates.from_params(self.params)

    # ----------------------------------------------------------- forecasts
    def model(self, name: str):
        with open(self.forecast_dir / f"model_{name}_{self.test_year}.pkl", "rb") as f:
            return pickle.load(f)

    def conformal(self, name: str) -> VectorConformal:
        r = np.load(self.forecast_dir / f"val_residual_{name}_{self.test_year}.npy")
        return VectorConformal().fit(r)

    def sampler(self, name: str):
        """Scenario offsets (n, n_lead, L, 2): zero at lead 0, residuals after."""
        conf = self.conformal(name)

        def draw(n, rng):
            r = conf.sample(n, rng)
            return np.concatenate([np.zeros((n, 1, *r.shape[2:]), "float32"), r], 1)
        return draw

    def issue_indices(self, start=None, end=None, every_h: int = 6) -> np.ndarray:
        t = pd.DatetimeIndex(self.A.times)
        start = start or self.test_start - pd.Timedelta(days=1)
        end = end or self.test_end
        ok = (t >= start) & (t <= end) & (t.hour % every_h == 0)
        idx = np.where(ok)[0]
        return idx[idx >= self.n_hist - 1]

    def provider(self, name: str, issue_idx: np.ndarray | None = None):
        if name == "perfect":
            return TruthProvider(self.A)
        issue_idx = self.issue_indices() if issue_idx is None else issue_idx
        return FieldForecast(self.A, self.model(name), issue_idx, self.n_hist,
                             self.horizons, name)

    @property
    def forecast_models(self) -> list[str]:
        """Models the downstream experiments compare: the configured list plus
        whichever model the ladder selected, so the selected one is never missing."""
        models = list(self.cfg["forecast_models"])
        best = self.best_model()
        return models if best in models else models + [best]

    def best_model(self) -> str:
        """The simplest model the ladder could not beat, by its stated rule.

        A model more complex than ridge is chosen only if its skill over ridge
        has a bootstrap lower bound above zero in most (horizon, level) cells.

        The choice is made on the rolling test years *before* the final test
        year, so the hold-out year never influences which model is selected.
        Only when no earlier year was evaluated (the development config, which
        has a single test year) does it fall back to that year.
        """
        ci = pd.read_csv(self.forecast_dir / "skill_ci.csv")
        ci = ci[ci.reference == "ridge"]
        earlier = ci[ci.test_year < self.test_year]
        ci = earlier if len(earlier) else ci[ci.test_year == self.test_year]
        winners = []
        for m, g in ci.groupby("model"):
            if m in ("persistence", "tide_persistence", "climatology", "moving_average",
                     "linear"):
                continue
            if (g.ci_lo_pct > 0).mean() > 0.5:
                winners.append((g.skill_pct.mean(), m))
        return max(winners)[1] if winners else "ridge"
