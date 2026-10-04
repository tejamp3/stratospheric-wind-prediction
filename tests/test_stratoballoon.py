"""Tests for the package. They run on a small synthetic atmosphere, so CI
needs no ERA5 download."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stratoballoon.atmosphere import (Atmosphere, distance_km, hypsometric_heights, move,
                                      standard_height)
from stratoballoon.control import Band, Greedy, Hold, MPC, Rates
from stratoballoon.dynamics import BalloonParams
from stratoballoon.forecasting import features as F
from stratoballoon.forecasting import metrics as Mx
from stratoballoon.forecasting.models import Persistence, Ridge
from stratoballoon.forecasting.provider import FieldForecast, TruthProvider
from stratoballoon.simulation import fly, mission_metrics, sample_missions
from stratoballoon.uncertainty import VectorConformal


def synthetic(n_t=3 * 365 * 4, levels=(50.0, 30.0, 20.0), seed=0,
              shear=True) -> Atmosphere:
    """Three years of 6-hourly fields with a seasonal cycle and opposing layers."""
    rng = np.random.default_rng(seed)
    times = pd.date_range("2021-01-01", periods=n_t, freq="6h").values
    lats, lons = np.arange(5.0, 36.0, 1.0), np.arange(60.0, 101.0, 1.0)
    L = len(levels)
    doy = pd.DatetimeIndex(times).dayofyear.to_numpy()
    season = np.sin(2 * np.pi * doy / 365.25)[:, None, None, None]
    sign = np.array([1.0, -1.0, 0.5])[:L] if shear else np.ones(L)
    u = 8 * season * sign[None, :, None, None] + rng.normal(0, 1, (n_t, L, len(lats), len(lons)))
    v = 3 * np.cos(2 * np.pi * doy / 365.25)[:, None, None, None] * sign[None, :, None, None] \
        + rng.normal(0, 1, (n_t, L, len(lats), len(lons)))
    t = np.broadcast_to(np.array([215.0, 225.0, 230.0])[:L][None, :, None, None],
                        u.shape).copy() + rng.normal(0, 0.5, u.shape)
    lv = np.array(levels)
    h = hypsometric_heights(t.astype("float32"), lv)
    return Atmosphere(times, lv, lats, lons, u.astype("float32"), v.astype("float32"),
                      t.astype("float32"), h, "synthetic")


@pytest.fixture(scope="module")
def atm():
    return synthetic()


# ------------------------------------------------------------------ atmosphere
def test_standard_heights_match_reference():
    assert abs(standard_height(100) - 16_180) < 30
    assert abs(standard_height(50) - 20_576) < 30
    assert abs(standard_height(30) - 23_849) < 60


def test_sample_reproduces_grid_values(atm):
    i, l, y, x = 40, 1, 10, 20
    s = atm.sample(atm.hours[i], atm.lats[y], atm.lons[x], atm.h[i, l, y, x])
    assert np.isclose(s["u"][0], atm.u[i, l, y, x], atol=1e-4)
    assert np.isclose(s["p_hpa"][0], atm.levels[l], rtol=1e-6)


def test_sample_interpolates_between_times(atm):
    i, y, x = 40, 10, 20
    mid = 0.5 * (atm.hours[i] + atm.hours[i + 1])
    s = atm.sample(mid, atm.lats[y], atm.lons[x], atm.h[i, 0, y, x])
    # heights barely change between steps, so u is close to the time mean
    assert abs(s["u"][0] - 0.5 * (atm.u[i, 0, y, x] + atm.u[i + 1, 0, y, x])) < 0.5


def test_move_and_distance_agree():
    lat, lon = move(np.array([20.0]), np.array([80.0]), np.array([10.0]), np.array([0.0]), 3600)
    assert abs(distance_km(20.0, 80.0, lat, lon)[0] - 36.0) < 0.1


# -------------------------------------------------------------------- features
def test_rolling_splits_are_ordered_with_embargo():
    tr, va, te = F.rolling_splits(2024, 2015, embargo_days=3)
    assert tr.end < va.start and va.end < te.start
    assert (va.start - tr.end) >= pd.Timedelta(days=6)


def test_labels_stay_inside_their_split(atm):
    step = atm.step_hours
    h_steps = [1, 2, 4]
    for sp in F.rolling_splits(2023, 2021):
        ti = F.issue_indices(atm, sp, n_hist=4, max_h_steps=max(h_steps))
        last_label = pd.Timestamp(atm.times[ti[-1] + max(h_steps)])
        first_input = pd.Timestamp(atm.times[ti[0] - 3])
        assert first_input >= sp.start and last_label <= sp.end, sp.name
        assert step == 6


def test_sample_targets_are_change_from_persistence(atm):
    yi, xi = F.cell_grid(atm, 10)
    ti = np.array([100, 101])
    s = F.build_samples(atm, ti, yi, xi, n_hist=4, h_steps=[2])
    c = 1
    assert np.allclose(s.truth[c, 0, :, 0], atm.u[ti[0] + 2, :, yi[c], xi[c]])
    assert np.allclose(s.current[c, :, 0], atm.u[ti[0], :, yi[c], xi[c]])


def test_persistence_scores_zero_skill_against_itself(atm):
    yi, xi = F.cell_grid(atm, 10)
    s = F.build_samples(atm, np.arange(10, 200), yi, xi, 4, [1])
    sq = np.sum(s.Y ** 2, -1)[:, 0, 0]
    r = Mx.bootstrap_skill(sq, sq, Mx.block_ids(s.t_idx, 16), n_boot=50)
    assert abs(r["skill_pct"]) < 1e-9


def test_ridge_beats_persistence_on_predictable_series(atm):
    yi, xi = F.cell_grid(atm, 6)
    sp = {s.name: s for s in F.rolling_splits(2023, 2021)}
    sam = {k: F.build_samples(atm, F.issue_indices(atm, v, 4, 8, 2), yi, xi, 4, [4, 8])
           for k, v in sp.items()}
    pred = Ridge().fit(sam["train"], sam["val"], atm, [4, 8]).predict(sam["test"], atm, [4, 8])
    err_r = np.mean((pred - sam["test"].Y) ** 2)
    err_p = np.mean(sam["test"].Y ** 2)
    assert err_r < err_p


# ------------------------------------------------------------------ uncertainty
def test_conformal_covers_exchangeable_data():
    rng = np.random.default_rng(1)
    c = VectorConformal().fit(rng.normal(0, 2, (4000, 2, 2, 2)))
    rows = c.evaluate(rng.normal(0, 2, (4000, 2, 2, 2)), levels=(0.8, 0.9))
    for r in rows:
        assert abs(r["disc_coverage"] - r["nominal"]) < 0.03


# ---------------------------------------------------------------- provider
def test_field_forecast_lead_zero_is_the_analysis(atm):
    issue = np.arange(10, 30)
    fc = FieldForecast(atm, Persistence(), issue, 4, [6, 12], "persistence")
    k, y, x = 5, 12, 7
    ti = issue[k]
    u, _ = fc.wind(np.array([k]), np.array([atm.hours[ti]]), np.array([atm.lats[y]]),
                   np.array([atm.lons[x]]), np.array([atm.h[ti, 1, y, x]]))
    # fields are stored at half precision (~0.01 m/s)
    assert np.isclose(u[0], atm.u[ti, 1, y, x], atol=0.01)
    # persistence: the same value at every lead
    u12, _ = fc.wind(np.array([k]), np.array([atm.hours[ti] + 12]), np.array([atm.lats[y]]),
                     np.array([atm.lons[x]]), np.array([atm.h[ti, 1, y, x]]))
    assert np.isclose(u12[0], u[0], atol=1e-4)


# -------------------------------------------------------------------- dynamics
def design(atm):
    hi = atm.sample(atm.hours[:20], np.full(20, 20.0), np.full(20, 80.0),
                    np.full(20, atm.h[:, 2].mean() - 300))
    lo = atm.sample(atm.hours[:20], np.full(20, 20.0), np.full(20, 80.0),
                    np.full(20, atm.h[:, 0].mean() + 300))
    band = Band(float(atm.h[:, 0].mean() + 300), float(atm.h[:, 2].mean() - 300), 5)
    return BalloonParams.design(float(hi["rho"].mean()), float(lo["rho"].mean())), band


def test_balloon_reaches_commanded_altitude(atm):
    p, band = design(atm)
    m = sample_missions(atm, 6, "2022-02-01", "2022-03-01", 24, band.alt_min, band.alt_min, 0)

    class GoUp(Hold):
        def decide(self, obs, mission, provider):
            return np.full(len(obs.lat), band.alt_max)
    tr = fly(atm, m, GoUp(), TruthProvider(atm), p, dt_s=120)
    assert np.all(np.abs(tr["alt"][:, -1] - band.alt_max) < 300)


def test_descending_costs_energy_and_ascending_does_not(atm):
    p, band = design(atm)
    up = sample_missions(atm, 4, "2022-02-01", "2022-03-01", 12, band.alt_min, band.alt_min, 0)
    down = sample_missions(atm, 4, "2022-02-01", "2022-03-01", 12, band.alt_max, band.alt_max, 0)

    class Go(Hold):
        def __init__(self, alt):
            self.alt = alt

        def decide(self, obs, mission, provider):
            return np.full(len(obs.lat), self.alt)
    e_up = fly(atm, up, Go(band.alt_max), TruthProvider(atm), p)["energy_pump_wh"][:, -1]
    e_dn = fly(atm, down, Go(band.alt_min), TruthProvider(atm), p)["energy_pump_wh"][:, -1]
    assert e_dn.mean() > 10 * max(e_up.mean(), 1e-3)


def test_no_pumping_with_an_empty_battery(atm):
    p, band = design(atm)
    p.battery_wh = 0.0
    p.solar_peak_w = 0.0
    m = sample_missions(atm, 3, "2022-02-01", "2022-03-01", 6, band.alt_max, band.alt_max, 0)

    class Down(Hold):
        def decide(self, obs, mission, provider):
            return np.full(len(obs.lat), band.alt_min)
    tr = fly(atm, m, Down(), TruthProvider(atm), p)
    assert np.all(tr["energy_pump_wh"][:, -1] == 0)
    assert np.all(tr["alt"][:, -1] > band.alt_max - 500)


# --------------------------------------------------------------------- control
def test_perfect_mpc_beats_hold_when_layers_oppose(atm):
    p, band = design(atm)
    rates = Rates.from_params(p)
    m = sample_missions(atm, 30, "2022-01-01", "2022-12-01", 48, band.alt_min, band.alt_max, 3)
    truth = TruthProvider(atm)
    hold = mission_metrics(fly(atm, m, Hold(), truth, p))
    mpc = mission_metrics(fly(atm, m, MPC(band, rates), truth, p))
    assert mpc.mean_dist_km.mean() < hold.mean_dist_km.mean()


def test_controllers_output_targets_inside_the_band(atm):
    p, band = design(atm)
    rates = Rates.from_params(p)
    m = sample_missions(atm, 8, "2022-01-01", "2022-06-01", 12, band.alt_min, band.alt_max, 4)
    for c in (Greedy(band, rates), MPC(band, rates)):
        tr = fly(atm, m, c, TruthProvider(atm), p)
        assert tr["target_alt"].min() >= band.alt_min - 1
        assert tr["target_alt"].max() <= band.alt_max + 1


# ------------------------------------------------- estimation and fault injection
from stratoballoon.estimation import WindBiasKF  # noqa: E402
from stratoballoon.scenarios import Faults, fault_suite  # noqa: E402


def test_kalman_filter_recovers_a_constant_forecast_bias():
    n, dt = 5, 600.0
    kf = WindBiasKF(np.full(n, 20.0), np.full(n, 80.0), np.full(n, 20.0), np.full(n, 80.0),
                    np.full(n, 21_000.0))
    lat, lon = np.full(n, 20.0), np.full(n, 80.0)
    true_u, fc_u = 10.0, 7.0                     # forecast is 3 m/s too slow
    for _ in range(72):
        lat, lon = move(lat, lon, np.full(n, true_u), np.zeros(n), dt)
        kf.predict(np.full(n, fc_u), np.zeros(n), dt)
        kf.update_gps(lat, lon, np.full(n, 21_000.0), np.ones(n, bool))
    assert np.allclose(kf.bias[0], 3.0, atol=0.3)


def test_filter_dead_reckons_better_than_the_last_fix_during_gps_loss(atm):
    p, band = design(atm)
    m = sample_missions(atm, 20, "2022-02-01", "2022-11-01", 24, band.alt_min, band.alt_max, 5)
    f = Faults(len(m), gps_out=np.tile([6.0, 24.0], (len(m), 1)))
    raw = fly(atm, m, Hold(), TruthProvider(atm), p, faults=f)
    kf = fly(atm, m, Hold(), TruthProvider(atm), p, faults=f, estimator=True)
    # worst error during the outage (GPS returns at the final step)
    assert kf["est_err_km"].max(1).mean() < 0.5 * raw["est_err_km"].max(1).mean()


def test_comms_loss_ages_the_forecast_and_queues_telemetry(atm):
    p, band = design(atm)
    m = sample_missions(atm, 6, "2022-02-01", "2022-11-01", 36, band.alt_min, band.alt_max, 6)
    fc = FieldForecast(atm, Persistence(), np.arange(0, len(atm.times), 1), 4, [6, 12, 24],
                       "persistence")
    f = Faults(len(m), comms_out=np.tile([3.0, 30.0], (len(m), 1)))
    tr = fly(atm, m, Hold(), fc, p, faults=f)
    met = mission_metrics(tr)
    assert met.max_forecast_age_h.min() >= 24
    assert met.max_telemetry_backlog_min.min() >= 26 * 60


def test_pump_failure_stops_all_pumping(atm):
    p, band = design(atm)
    m = sample_missions(atm, 4, "2022-02-01", "2022-11-01", 12, band.alt_max, band.alt_max, 7)

    class Down(Hold):
        def decide(self, obs, mission, provider):
            return np.full(len(obs.lat), band.alt_min)
    f = Faults(len(m), pump_fail_h=np.zeros(len(m)))
    tr = fly(atm, m, Down(), TruthProvider(atm), p, faults=f)
    assert np.all(tr["energy_pump_wh"][:, -1] == 0)


def test_fault_suite_is_reproducible():
    a, b = fault_suite(10, 72, seed=3), fault_suite(10, 72, seed=3)
    assert [x.label for x in a] == [x.label for x in b]
    assert np.allclose(a[1].gps_out, b[1].gps_out)


# --------------------------------------------- neural forecasters, anomaly, planner
from stratoballoon import anomaly as AD  # noqa: E402


def test_neural_forecasters_train_and_predict_the_right_shape(atm):
    from stratoballoon.forecasting.neural import LSTM, TCN
    yi, xi = F.cell_grid(atm, 10)
    sp = {s.name: s for s in F.rolling_splits(2023, 2021)}
    sam = {k: F.build_samples(atm, F.issue_indices(atm, v, 8, 4, 4), yi, xi, 8, [2, 4])
           for k, v in sp.items()}
    for M_ in (LSTM, TCN):
        m = M_(hidden=8, epochs=2, max_train=2000).fit(sam["train"], sam["val"], atm, [2, 4])
        assert m.predict(sam["test"], atm, [2, 4]).shape == sam["test"].Y.shape


def test_threshold_rule_rejects_an_alarm_stuck_on():
    # A score that is high everywhere has a single onset per mission; the rule
    # must still refuse it because it alarms all the time.
    score = np.full((4, 3000), 5.0)
    thr = AD.threshold_for_rate(score, per_day=0.5)
    assert (score[:, AD.WARMUP:] > thr).mean() == 0


def test_cusum_stays_quiet_on_healthy_telemetry():
    rng = np.random.default_rng(2)
    train = AD.simulate(10, 3, rng, inject=False)
    calib = AD.simulate(10, 3, rng, inject=False)
    det = AD.CusumDetector().fit(train)
    thr = AD.threshold_for_rate(det.score(calib), per_day=0.5)
    healthy = AD.simulate(10, 3, rng, inject=False)
    r = AD.evaluate(det.score(healthy), healthy, thr)
    assert r["false_alarms_per_day"] < 1.5


def test_cusum_catches_a_large_altimeter_step():
    rng = np.random.default_rng(3)
    det = AD.CusumDetector().fit(AD.simulate(10, 3, rng, inject=False))
    thr = AD.threshold_for_rate(det.score(AD.simulate(10, 3, rng, inject=False)), 0.5)
    tel = AD.simulate(4, 2, rng, inject=False)
    for m in range(4):
        AD._apply(tel.data[m], 1, 1000, 1400, np.random.default_rng(m))   # bias step
        tel.label[m, 1000:1400] = 1
    tel.events = pd.DataFrame({"mission": range(4), "fault": "altimeter bias",
                               "start": 1000, "end": 1400})
    assert AD.evaluate(det.score(tel), tel, thr)["recall"] == 1.0


def test_planner_reports_a_probability_and_a_strategy(atm):
    from stratoballoon.planner import MissionRequest, plan
    p, band = design(atm)

    class Ctx:
        pass
    ctx = Ctx()
    ctx.A, ctx.band, ctx.params, ctx.rates = atm, band, p, Rates.from_params(p)
    issue = np.arange(400, 520)
    fc = FieldForecast(atm, Persistence(), issue, 4, [6, 12, 24], "persistence")
    zero = lambda n, rng: np.zeros((n, 4, len(atm.levels), 2), "float32")  # noqa: E731
    req = MissionRequest(str(atm.times[420])[:16], 20.0, 80.0, 20.5, 80.5, duration_h=12)
    res = plan(ctx, req, fc, zero, n_scenarios=8)
    assert 0.0 <= res["p_success_end_within_radius"] <= 1.0
    assert len(res["recommended_altitude_m"]) == res["altitude"].shape[1]


def test_loader_refuses_a_time_axis_with_a_missing_month(atm):
    import xarray as xr
    keep = np.r_[0:200, 400:600]                      # drop 50 days in the middle
    ds = xr.Dataset({k: (("time", "level", "lat", "lon"), getattr(atm, k)[keep])
                     for k in ("u", "v", "t")},
                    coords={"time": atm.times[keep], "level": atm.levels,
                            "lat": atm.lats, "lon": atm.lons})
    with pytest.raises(ValueError, match="irregular time axis"):
        Atmosphere.from_dataset(ds)


def test_regime_conformal_fixes_coverage_in_the_strong_wind_regime():
    from stratoballoon.uncertainty import RegimeConformal
    rng = np.random.default_rng(0)

    def draw(n):
        speed = rng.uniform(0, 30, (n, 2))
        res = rng.normal(0, 1, (n, 3, 2, 2)) * (0.5 + speed[:, None, :, None] / 10)
        return speed, res
    sv, rv = draw(6000)
    st, rt = draw(6000)
    inside_reg = np.linalg.norm(rt, axis=-1) <= RegimeConformal().fit(rv, sv).radius(0.9, st)
    inside_marg = np.linalg.norm(rt, axis=-1) <= VectorConformal().fit(rv).radius(0.9)[None]
    strong = st > 20
    assert abs(inside_reg[:, 0][strong].mean() - 0.9) < 0.03
    assert inside_marg[:, 0][strong].mean() < 0.85


def test_adaptive_conformal_recovers_coverage_after_a_shift():
    from stratoballoon.uncertainty import adaptive_coverage
    rng = np.random.default_rng(1)
    cal = np.sort(np.abs(rng.normal(0, 1, 5000)))
    # the test period is 30% noisier than calibration, as a windier year would be
    t_idx = np.repeat(np.arange(2000), 20)
    test = np.abs(rng.normal(0, 1.3, len(t_idx)))
    fixed = (test <= cal[int(0.9 * len(cal))]).mean()
    adaptive = adaptive_coverage(test, t_idx, cal, 0.9, gamma=0.01, lag=2).mean()
    assert fixed < 0.82
    assert abs(adaptive - 0.9) < 0.02


@pytest.mark.parametrize("n_hist", [1, 4, 8])
def test_temporal_cnn_fits_any_history_length(atm, n_hist):
    # 24 h of history is 8 steps at 3-hourly but only 4 at 6-hourly
    from stratoballoon.forecasting.neural import TCN
    yi, xi = F.cell_grid(atm, 10)
    sp = {s.name: s for s in F.rolling_splits(2023, 2021)}
    sam = {k: F.build_samples(atm, F.issue_indices(atm, v, n_hist, 4, 8), yi, xi, n_hist, [2])
           for k, v in sp.items()}
    m = TCN(hidden=8, epochs=1, max_train=500).fit(sam["train"], sam["val"], atm, [2])
    assert m.predict(sam["test"], atm, [2]).shape == sam["test"].Y.shape


def test_constant_training_feature_cannot_blow_up_a_forecast(atm):
    # Train on 00/12 UTC only (sin(hour) is constant), then forecast from 06 UTC.
    from stratoballoon.forecasting.models import safe_std
    X = np.column_stack([np.random.default_rng(0).normal(0, 3, 500), np.full(500, 1e-16)])
    sd = safe_std(X)
    assert sd[1] == 1.0
    assert abs((1.0 - X[:, 1].mean()) / sd[1]) < 2          # sin(hour) = 1 at 06 UTC
    yi, xi = F.cell_grid(atm, 10)
    sp = {s.name: s for s in F.rolling_splits(2023, 2021)}
    sam = {k: F.build_samples(atm, F.issue_indices(atm, v, 4, 4, 2), yi, xi, 4, [2])
           for k, v in sp.items()}                           # stride 2 = 12-hourly issues
    model = Ridge().fit(sam["train"], sam["val"], atm, [2])
    every6h = F.build_samples(atm, F.issue_indices(atm, sp["test"], 4, 4, 1), yi, xi, 4, [2])
    assert np.abs(model.predict(every6h, atm, [2])).max() < 60


def test_field_forecast_refuses_impossible_winds(atm):
    class Broken(Persistence):
        def predict(self, s, A, h_steps):
            return np.full_like(s.Y, 1e6)
    with pytest.raises(ValueError, match="refusing to build"):
        FieldForecast(atm, Broken(), np.arange(10, 14), 4, [6, 12], "broken")
