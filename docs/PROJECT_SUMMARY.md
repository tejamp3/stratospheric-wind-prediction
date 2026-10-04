# Project summary

## What did I build?

I built a research simulation of an altitude-controlled super-pressure balloon
over South Asia. It is driven by twelve years of real ERA5 wind data and checked
against radiosonde observations. It forecasts the wind at every reachable
altitude with calibrated uncertainty, predicts where a drifting balloon will be,
and estimates the balloon's own state from noisy sensors with a Kalman filter.
Rule-based, model-predictive and reinforcement-learning controllers then choose
the altitude to steer by. Every component is compared against simpler baselines
on the same held-out year and the same seeded missions, with confidence
intervals. Faults are injected to see what breaks it.

## What makes it different from a normal ML project?

It closes the loop between atmospheric prediction, uncertainty estimation,
trajectory prediction, state estimation and autonomous control, and judges each
part by what it does for the mission rather than by its own error score. Where
a simpler method was as good, that is reported: the greedy altitude rule matched
or beat model-predictive control, and reinforcement learning did not beat
either. The first full run produced impossible forecasts; the cause was found,
fixed, and guarded against, and that is in the README too.

## Headline results

<!-- AUTO:headline -->
- **Forecasting:** the model the ladder selected was **lstm**, with 27% lower vector error than persistence at 6 h (test year 2025).
- **Truth check:** ERA5 differs from radiosondes by 3.7-4.3 m/s (vector RMS) at balloon levels, the same order as the forecast error.
- **Trajectory:** after 24 h the predicted position is a median 105 km off with lstm, against 180 km with persistence.
- **Control (simulation):** the best real controller, greedy / lstm, keeps the balloon within 50 km 11.1% of the time against 3.4% for doing nothing and 12.1% with a perfect forecast.
- **Feasibility:** even a perfect forecast holds station only 1% of the time in month 7 and 35% in month 3: the launch window matters more than the model.
- **Anomaly detection:** at an equal false-alarm budget, autoencoder caught 100% of injected faults.
<!-- /AUTO:headline -->

## One page

**Problem.** A super-pressure balloon has no engine; it steers only by changing
altitude to catch winds blowing in different directions. Holding position needs
a forecast of those winds, an honest idea of how wrong the forecast might be, an
estimate of where the balloon is, and a good altitude policy.

**Data.** ERA5 pressure levels 100-20 hPa (about 16.5-26.5 km) over 0-40°N,
40-130°E, 2015-2025, with a checksummed manifest; radiosondes from eight Indian
stations to measure how good ERA5 itself is.

**Methods.**
- *Forecasting:* a ladder from persistence through climatology, linear and
  ridge regression, and gradient-boosted trees to an LSTM and a temporal CNN,
  with a keep-or-drop rule stated in advance.
- *Uncertainty:* split-conformal regions, with a regime-conditional variant,
  plus error scenarios.
- *Trajectories:* ensemble trajectory prediction with an uncertainty cone.
- *Estimation:* a Kalman filter that also estimates the forecast's current
  error.
- *Control:* hold, greedy, MPC (deterministic and uncertainty-aware) and PPO
  reinforcement learning.
- *Supporting tools:* a mission planner, telemetry anomaly detection, and edge
  profiling.

**Evaluation.**
- Chronological splits with embargoes, and a final hold-out year used once.
- Block-bootstrap intervals on forecast skill.
- Paired, week-clustered comparisons on identical missions.
- A fault-injection suite, with every mission reported.

**What it is not.** A flight-certified simulator, or flight results. The
README's limitations section lists what is simplified and what is unmeasured.
