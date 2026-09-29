"""Correctness tests for the pieces that are easy to get subtly wrong.

These use synthetic data with known answers, so they test the code rather than
the weather. Run with:  python -m pytest tests -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

import config as C          # noqa: E402
import data as D            # noqa: E402
import metrics as M         # noqa: E402
from model import WindLSTM, persistence_forecast  # noqa: E402


# --------------------------------------------------------------------- fixtures
@pytest.fixture
def frame() -> pd.DataFrame:
    """A gapless synthetic station series with the right columns."""
    n = 600
    idx = pd.date_range("2022-01-01", periods=n, freq=f"{C.STEP_HOURS}h")
    rng = np.random.default_rng(0)
    data = {f: rng.normal(0, 1, n).cumsum().astype("float32") for f in D.FEATURES}
    return pd.DataFrame(data, index=idx)


# ------------------------------------------------------------------- directions
def test_direction_convention():
    """Meteorological convention: direction is where the wind comes FROM."""
    # Wind blowing toward the east (u positive) arrives from the west = 270 deg.
    assert M.direction(np.array(10.0), np.array(0.0)) == pytest.approx(270.0)
    # Blowing toward the north (v positive) arrives from the south = 180 deg.
    assert M.direction(np.array(0.0), np.array(10.0)) == pytest.approx(180.0)
    assert M.direction(np.array(-10.0), np.array(0.0)) == pytest.approx(90.0)
    assert M.direction(np.array(0.0), np.array(-10.0)) == pytest.approx(0.0)


def test_angular_error_wraps():
    """350 and 10 degrees are 20 apart, not 340."""
    assert M.angular_error(np.array(350.0), np.array(10.0)) == pytest.approx(20.0)
    assert M.angular_error(np.array(10.0), np.array(350.0)) == pytest.approx(20.0)
    assert M.angular_error(np.array(0.0), np.array(180.0)) == pytest.approx(180.0)
    # Never exceeds 180.
    a = np.random.default_rng(1).uniform(0, 360, 500)
    b = np.random.default_rng(2).uniform(0, 360, 500)
    err = M.angular_error(a, b)
    assert err.min() >= 0 and err.max() <= 180


def test_skill_score_sign():
    """Higher is better: beating the baseline gives a positive score."""
    assert M.skill_score(8.0, 10.0) == pytest.approx(20.0)
    assert M.skill_score(10.0, 10.0) == pytest.approx(0.0)
    assert M.skill_score(12.0, 10.0) == pytest.approx(-20.0)


def test_perfect_forecast_scores_perfectly():
    rng = np.random.default_rng(3)
    truth = rng.normal(0, 10, (200, 2))
    r = M.evaluate(truth.copy(), truth)
    assert r["speed_rmse"] == pytest.approx(0.0, abs=1e-6)
    assert r["vector_rmse"] == pytest.approx(0.0, abs=1e-6)
    assert r["maae_deg"] == pytest.approx(0.0, abs=1e-6)
    assert r["dir_acc_30deg"] == pytest.approx(100.0)


def test_direction_metrics_skip_calm_cases():
    """Bearings are only scored where the wind is strong enough to have one."""
    truth = np.array([[0.1, 0.0], [20.0, 0.0], [30.0, 0.0]])
    r = M.evaluate(truth.copy(), truth)
    assert r["n_total"] == 3
    assert r["n_resolvable"] == 2      # the 0.1 m/s case is excluded


# ------------------------------------------------------------------- windowing
def test_window_alignment():
    """Label at horizon h must be the series value h steps past the window end."""
    n = 100
    x = np.arange(n, dtype="float32")[:, None].repeat(len(D.FEATURES), axis=1)
    y = np.stack([np.arange(n), np.arange(n) * 10], axis=1).astype("float32")
    X, Y = D.make_windows(x, y, input_steps=8, horizon_steps=[2, 4, 8])

    # Window 0 spans indices 0..7, so its last observed step is 7.
    assert X[0, -1, 0] == 7
    # Labels are at 7+2, 7+4, 7+8.
    assert Y[0, 0, 0] == 9
    assert Y[0, 1, 0] == 11
    assert Y[0, 2, 0] == 15
    # Second target channel is 10x the first, confirming channels are not swapped.
    assert Y[0, 0, 1] == 90
    # Count stops max(horizon) short so every label exists.
    assert len(X) == n - 8 - 8 + 1
    assert Y[-1, 2, 0] == n - 1


def test_windows_too_short_raise():
    x = np.zeros((5, len(D.FEATURES)), "float32")
    y = np.zeros((5, 2), "float32")
    with pytest.raises(ValueError):
        D.make_windows(x, y, input_steps=8, horizon_steps=[8])


# ------------------------------------------------------------ splits / leakage
def test_splits_are_chronological_and_disjoint(frame):
    out, sx, sy, df = D.build_dataset(frame, verbose=False)
    tr, va, te = out["train"]["time"], out["val"]["time"], out["test"]["time"]
    assert tr[-1] < va[0] < va[-1] < te[0]
    assert len(set(tr) & set(va)) == 0
    assert len(set(va) & set(te)) == 0


def test_split_fractions(frame):
    a, b, c = D.time_splits(1000, (0.7, 0.15, 0.15))
    assert (a.start, a.stop) == (0, 700)
    assert (b.start, b.stop) == (700, 850)
    assert (c.start, c.stop) == (850, 1000)


def test_scaler_is_fit_on_train_only(frame):
    """The scaler must reproduce train-slice statistics, not whole-record ones."""
    out, sx, sy, df = D.build_dataset(frame, verbose=False)
    feats = df[D.FEATURES].to_numpy("float32")
    tr, _, _ = D.time_splits(len(df))
    np.testing.assert_allclose(sx.mean, feats[tr].mean(0), rtol=1e-5)
    # And it should differ from the full-record mean on a trending series.
    assert not np.allclose(sx.mean, feats.mean(0), rtol=1e-3)


def test_scaler_round_trip(frame):
    out, sx, sy, df = D.build_dataset(frame, verbose=False)
    raw = df[D.FEATURES].to_numpy("float32")[:50]
    np.testing.assert_allclose(sx.inverse(sx.transform(raw)), raw, rtol=1e-4, atol=1e-3)


def test_train_split_is_standardised(frame):
    out, _, _, _ = D.build_dataset(frame, verbose=False)
    flat = out["train"]["X"].reshape(-1, len(D.FEATURES))
    assert np.abs(flat.mean(0)).max() < 0.35
    assert np.abs(flat.std(0) - 1).max() < 0.5


# ---------------------------------------------------------------- cleaning
def test_clean_fills_gaps_and_flags_them():
    idx = pd.date_range("2022-01-01", periods=40, freq=f"{C.STEP_HOURS}h")
    df = pd.DataFrame({f: np.arange(40, dtype="float32") for f in D.FEATURES},
                      index=idx)
    df.iloc[10:13] = np.nan
    clean, missing = D.clean(df)
    assert int(missing.to_numpy().sum()) == 3 * len(D.FEATURES)
    assert clean[D.FEATURES].isna().sum().sum() == 0


def test_clean_restores_a_dropped_timestamp():
    """A missing row must reappear on the axis rather than silently vanish."""
    idx = pd.date_range("2022-01-01", periods=20, freq=f"{C.STEP_HOURS}h")
    df = pd.DataFrame({f: np.ones(20, dtype="float32") for f in D.FEATURES}, index=idx)
    gapped = df.drop(df.index[5])
    clean, missing = D.clean(gapped)
    assert len(clean) == 20
    assert clean.index.equals(idx)
    assert int(missing.to_numpy().sum()) == len(D.FEATURES)


def test_derived_columns_match_components():
    idx = pd.date_range("2022-01-01", periods=10, freq=f"{C.STEP_HOURS}h")
    df = pd.DataFrame({f: np.full(10, 3.0, dtype="float32") for f in D.FEATURES},
                      index=idx)
    out = D.add_derived(df)
    # u = v = 3 gives speed 3*sqrt(2), coming from the south-west = 225 deg.
    assert out["speed50"].iloc[0] == pytest.approx(3 * np.sqrt(2), rel=1e-5)
    assert out["dir50"].iloc[0] == pytest.approx(225.0, rel=1e-4)
    assert out["shear_10_50"].iloc[0] == pytest.approx(0.0, abs=1e-5)


# -------------------------------------------------------------------- model
def test_model_output_shape():
    m = WindLSTM(n_features=len(D.FEATURES))
    import torch
    y = m(torch.zeros(5, C.INPUT_STEPS, len(D.FEATURES)))
    assert y.shape == (5, len(C.HORIZONS), 2)


def test_persistence_repeats_last_observation():
    last = np.array([[3.0, 4.0], [0.0, -1.0]])
    p = persistence_forecast(last, n_horizons=3)
    assert p.shape == (2, 3, 2)
    for k in range(3):
        np.testing.assert_allclose(p[:, k, :], last)


def test_persistence_on_constant_wind_is_exact():
    """If the wind never changes, persistence should be a perfect forecast."""
    last = np.tile(np.array([[5.0, -2.0]]), (50, 1))
    pred = persistence_forecast(last, n_horizons=1)
    r = M.evaluate(pred[:, 0, :], last)
    assert r["vector_rmse"] == pytest.approx(0.0, abs=1e-9)


# ------------------------------------------------------------ residual framing
def test_residual_model_starts_as_exact_persistence():
    """With a zero-initialised head, residual mode must reproduce persistence."""
    import torch
    m = WindLSTM(n_features=len(D.FEATURES), residual=True).eval()
    x = torch.randn(7, C.INPUT_STEPS, len(D.FEATURES))
    y = m(x)
    last = x[:, -1, :2].unsqueeze(1).expand_as(y)
    torch.testing.assert_close(y, last)


def test_assert_residual_safe_rejects_mismatched_scalers():
    from model import assert_residual_safe
    good_x = D.Scaler(np.array([1.0, 2.0, 9.0]), np.array([3.0, 4.0, 9.0]))
    good_y = D.Scaler(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
    assert_residual_safe(good_x, good_y)          # must not raise

    bad_y = D.Scaler(np.array([1.0, 5.0]), np.array([3.0, 4.0]))
    with pytest.raises(ValueError):
        assert_residual_safe(good_x, bad_y)


def test_build_dataset_keeps_residual_assumption(frame):
    """The scalers build_dataset produces must satisfy the residual precondition."""
    from model import assert_residual_safe
    _, sx, sy, _ = D.build_dataset(frame, verbose=False)
    assert_residual_safe(sx, sy)


# ------------------------------------------------------- figure edge cases
def test_polar_error_figure_survives_empty_direction_sectors(tmp_path, monkeypatch):
    """Sectors with no cases must be skipped, not drawn with a NaN height.

    A NaN bar height raises inside matplotlib's polar path transform, which took
    the whole evaluation run down rather than just the figure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import evaluate as E
    import viz
    viz.apply_style()
    monkeypatch.setattr(E.C, "FIGURES", tmp_path)

    n = 60
    rng = np.random.default_rng(0)
    # All wind from roughly one bearing, so most of the 12 sectors are empty.
    true = np.zeros((n, len(C.HORIZONS), 2), dtype="float32")
    true[:, :, 0] = 10.0
    true[:, :, 1] = rng.normal(0, 0.05, (n, len(C.HORIZONS)))
    pred = true + rng.normal(0, 0.4, true.shape).astype("float32")

    E.fig_error_by_direction(pred, true)          # must not raise
    assert (tmp_path / "eval_error_by_direction.png").exists()
