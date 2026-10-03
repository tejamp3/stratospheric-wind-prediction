# Engineering audit of the stratospheric wind project

Audit date: 2026-10-03. Scope: the repository as of commit `fe9bfbc` on `main`.

Every claim below marked **[measured]** comes from
[`audit_experiments.py`](audit_experiments.py), which reruns against the
committed checkpoints and the 36 monthly ERA5 files. Its raw output is in
[`audit_results.json`](audit_results.json). Claims without that tag come from
reading the code.

## Summary

The forecasting half of the project is careful work. The splits are
chronological, the scalers are fitted on the training data only, model selection
uses validation data only, and the ridge-beats-LSTM result is real and
statistically significant.

The application half does not hold up as a model of a stratospheric balloon. The
station-keeping simulation is open-loop: the controller never looks at its own
position. That one design choice produces the headline finding that "bias flips
the ranking", and the finding disappears when a basic position-feedback term is
added. The vehicle modelled is a powered airship, not a balloon. The altitude
levels are 3 to 7 km apart and include one at 31 km that no balloon of the kind
discussed could reach. The model forecasts a single grid column, while a free
balloon leaves that column within hours.

Two statements in the public README should be corrected regardless of what else
is built (section C).

---

## A. What is already scientifically sound

| Item | Evidence |
|---|---|
| Chronological train / validation / test split, no shuffling across time | `data.time_splits`; test `test_splits_are_chronological_and_disjoint` |
| Windows built inside each split, so no input or label crosses a split boundary | `data.build_dataset`; test `test_window_alignment` |
| Normalisation fitted on the training split only | test `test_scaler_is_fit_on_train_only` |
| Hyperparameters selected on validation loss only (32-config sweep) | `results/metrics/sweep_best.json` |
| Interval calibration fitted on validation, coverage verified on test | `models/calibration.json`: 92.1 / 87.9 / 90.9 % for nominal 90 % |
| Persistence is used as the reference baseline, which is the correct first baseline for a strongly autocorrelated series | `model.persistence_forecast` |
| A linear baseline (ridge) is reported next to the LSTM and wins | `results/metrics/test_metrics_base.csv` |
| Directions scored with angular wraparound, and only where wind ≥ 2 m/s | `metrics.angular_error`, `metrics.evaluate` |
| The residual-to-persistence framing asserts its precondition instead of assuming it | `model.assert_residual_safe` |
| Every README number is generated from result files | `src/report.py`; reproduced independently on 2026-10-02 |
| No missing timestamps or values in the 8,768-step record | [measured] |

## B. Assumptions that are unrealistic

1. **The vehicle is a powered airship, not a balloon.** `airship.simulate_track`
   lets the vehicle fly at up to 12 m/s in any direction. A super-pressure
   balloon has no horizontal thrust; it can only change altitude.
2. **The controller is open-loop.** At each step it cancels the *forecast* wind
   and never corrects for where it actually is. Any real station-keeping
   controller uses position feedback (GPS). See C2 for what this changes.
3. **Altitude levels are unrealistic for altitude control.** The data has only
   50, 30 and 10 hPa (about 20.5, 23.8 and 31 km). A super-pressure balloon with
   a ballonet changes altitude by a few kilometres at most; 31 km is out of
   reach from a 20 km float, and the 3–7 km gaps hide the closely spaced wind
   layers that altitude steering actually exploits. Red Balloon's VISTA flight
   was designed for about 25 km, which is between two of the downloaded levels.
4. **"Best of three levels" assumes instant, free altitude changes.** The 70 %
   feasibility figure counts the vehicle as being at whichever of 20.5, 23.8 or
   31 km is calmest at every 3-hour step, with no climb time, energy cost or
   limit on how often it switches.
5. **A single grid column is treated as "the wind".** [measured] A free balloon
   launched at 20°N 80°E at 50 hPa moves a median 587 km in 24 h (90th
   percentile 1,688 km), and 44 % of launches leave the 30° × 40° download
   domain within 72 h. A forecast at one fixed point cannot predict where the
   balloon goes after the first few hours.
6. **The drift metric assumes a constant error held for the whole lead time**
   (`error × hours × 3.6`), with no feedback and no dynamics.
7. **ERA5 is treated as truth.** It is a reanalysis, i.e. a model constrained by
   observations. In the tropical stratosphere, where observations are sparse, its
   own wind error is not negligible and has not been checked here against
   observations such as radiosondes.
8. **The forecast's input is reanalysis.** In operation the controller would
   receive a numerical weather prediction (NWP) forecast, issued in real time.
   Reanalysis lags real time by about five days and is computed with hindsight.
9. **No geopotential height was downloaded** (`raw variables: u, v, t`
   [measured]), so altitude in kilometres is approximated from pressure, not
   computed.

## C. Results: which are convincing

**C1. Ridge regression beats the LSTM. Convincing.** [measured] With a block
bootstrap (8-day blocks, 2,000 resamples), the LSTM's speed RMSE minus ridge's
has a 95 % confidence interval of +0.13 to +0.27 m/s at 6 h, +0.20 to +0.43 at
12 h and +0.07 to +0.31 at 24 h. None includes zero. The result also holds on
vector error, at 5 other grid points, and across 5 training seeds.

**C2. "Bias flips the ranking" for station keeping. Not convincing. It is an
artefact of the open-loop controller.** [measured] Re-running the same 158
five-day missions with a simple position-feedback term (fly back towards the
station at distance ÷ 6 h, still capped at 12 m/s):

| Controller | Held within 200 km, open loop (as published) | With position feedback |
|---|---|---|
| Perfect forecast | 53.2 % | 54.4 % |
| Persistence | 29.1 % | 49.4 % |
| LSTM | 9.5 % | 54.4 % |

With feedback, the LSTM matches the perfect forecast and beats persistence, and
the ranking reverses again. Bias only accumulates when nothing corrects it. What
remains true: a biased forecast is costly for an open-loop or infrequently
corrected system, and a free balloon, which cannot correct by thrust, is closer
to that case. The README currently presents C2 as a general finding; it should
be restated.

**C3. Interval coverage. Convincing for what it covers.** Coverage of 88–92 %
for nominal 90 % on a held-out period is a real check. But the intervals are on
speed only, not on the wind vector, and their width is the same in every
condition.

**C4. The 10-day trajectory (LSTM 214 km vs persistence 80 km). Anecdotal.**
The window was chosen as the calmest stretch *of the test period*, which means
the test data was used to choose it. It is one sample, and it is open-loop
(C2).

## D. Results that need stronger validation

| Result | Problem | Measured spread |
|---|---|---|
| LSTM skill 18.5 % at 6 h; "20 % target not met" | Single seed, single test season | [measured] 95 % CI 14.1–22.6 %; 5 seeds give 14.4–21.1 % |
| Skill reported on **speed** RMSE | Speed error ignores direction; what moves a balloon is the vector error | [measured] LSTM vector skill is 26.7 % at 6 h, against 18.5 % on speed |
| "Weight decay made no difference" | Best epoch is 3–6 in every run, at learning rate 1e-3: there is too little training for weight decay to act | [measured] best epochs 3, 5, 6, 4, 4 across seeds |
| LSTM bias of 42 km/day | Varies by seed and grows with lead time | [measured] 42–50 km/day at 6 h; 26–65 km/day at 24 h |
| Test period | One five-month stretch (Jul–Dec 2024), one monsoon-to-winter transition; validation is a different season (Feb–Jul 2024) | 21 independent 8-day blocks only |
| Edge latency 0.12–0.49 ms | Measured on a desktop x86 CPU, not on flight-class hardware; no RAM or energy measured | — |
| Persistence error is lower at 24 h than at 12 h | Caused by the 24-hour tide in the meridional wind: persistence at 24 h is "same time yesterday" | [measured] persistence vector RMSE 5.10 / 6.37 / 5.23 m/s at 6 / 12 / 24 h |

## E. Experiments that are missing

1. A **climatology** baseline and a **tide-aware persistence** baseline. [measured]
   Tide-aware persistence is 6.7 % better than plain persistence at 6 h;
   month-by-hour climatology from two training years is worse than persistence
   (bias 1.9 m/s, because the quasi-biennial oscillation shifts the mean between
   years).
2. **Tree ensembles** (random forest / gradient boosting) between ridge and the
   LSTM.
3. **Rolling-origin evaluation**: several test periods covering every season,
   not one.
4. **Longer record**: 3 years covers only about 1.3 cycles of the
   quasi-biennial oscillation.
5. **Horizons of 48 and 72 h.**
6. **Spatial generalisation.** [measured, first pass] Ridge trained at 20°N 80°E
   keeps 18–32 % vector skill at 6 h at four other points; the LSTM keeps
   16–27 %. Not yet tested: a model trained on many grid points.
7. **Feature ablation**: which levels and variables matter.
8. **Comparison against an NWP forecast**, the baseline an operator actually has.
9. **Validation of ERA5 against observations** (radiosondes) in this region.
10. **Any balloon physics**, trajectory prediction, state estimation or
    altitude control. None exists.

## F. Code that is fragile or hard to reproduce

| Issue | Where | Effect |
|---|---|---|
| Not a Python package; every module does `sys.path.insert` | all of `src/` | Imports break if run from another directory; cannot be pip-installed |
| Settings are module constants, not versioned config files | `src/config.py` | A run cannot record which configuration produced it |
| No data manifest or checksums | `data/raw` | Cannot prove two runs used identical data |
| Single seed; PyTorch determinism not enforced | `train.py` | Results vary by about ±3 points of skill between seeds |
| Scripts must run in a fixed order with no driver | README "Running it" | Easy to report figures from mixed runs |
| The sweep shells out to `train.py` and parses files | `src/sweep.py` | Slow and brittle |
| No continuous integration | — | Nothing runs the tests on push |
| The notebook test needs the 209 MB raw data | `tests/test_notebook.py` | Skipped on every fresh clone |
| Model, data and evaluation code are coupled through `evaluate.py` imports | `airship.py`, `optimize.py`, `inference.py` | Changing evaluation can break inference |

Strengths to keep: 24 focused unit tests on leakage-prone code, the
`report.py` pattern of generated documentation, pinned requirements, a resumable
downloader that respects the data provider's limits.

## G. What is useful for a real balloon system

- The ERA5 downloader and its documented constraints.
- The leakage-safe windowing and split code.
- The finding that a linear model is as good as the LSTM here, which argues for
  simple, verifiable onboard models.
- The calibrated-interval approach, once extended to the wind vector.
- The edge-profiling harness, once run on target-class hardware.
- The altitude-feasibility idea, once redone on closely spaced levels with
  climb limits.

## H. What is an academic demonstration

- The airship controller and everything derived from it (drift tables, scenario
  scan, the 10-day track).
- The "best of three levels" figure.
- FP16 and int8 variants of a 72,000-weight model: the model is already tiny,
  so shrinking it does not change what can run onboard.
- Monte-Carlo dropout on a one-layer LSTM with dropout only before the output
  layer.

## I. What an aerospace or AI interviewer would ask

1. "Your balloon has no engine. How does your airship simulation apply?"
2. "Why is your controller open-loop? What happens with GPS feedback?" (C2)
3. "A balloon moves 600 km a day. How does a single-point forecast predict its
   path?"
4. "Is ERA5 the truth? How good is it in the tropical stratosphere?"
5. "Ridge beats your LSTM. Why did you keep the LSTM?"
6. "Your test set is five months of one year. How do you know it generalises to
   other seasons or other years?"
7. "Why score speed error when the balloon is moved by the vector?"
8. "Why did persistence get *better* from 12 h to 24 h?" (the tide)
9. "What would you compare against in operations?" (NWP)
10. "How would a super-pressure balloon change altitude, and what does that
    cost in energy?"
11. "Where would your latency numbers come from on the real flight computer?"
12. "Your weight-decay result: how many epochs did the model train?"

## Corrections to make to the public README now

1. Finding 2 ("cumulative drift is driven by bias, and this flips the ranking")
   should be restated as holding for an open-loop controller, with the feedback
   result alongside.
2. The 10-day trajectory should be labelled as one illustrative window chosen
   from the test period.
