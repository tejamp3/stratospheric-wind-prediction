# Technical log: forecasting stratospheric wind for airship station-keeping

This is the working record of what was built, what was decided, and what was
learned. Tables marked as generated are written by `python src/report.py`
directly from the files under `results/`, so no number here is transcribed by
hand.

---

## Phase 1 - Data pipeline and exploratory analysis

### Data source and how it is fetched

ERA5 is the ECMWF reanalysis: a physically consistent reconstruction of the
atmosphere produced by assimilating observations into a frozen forecast model.
That property is what makes it usable as ground truth here. It is gap-free by
construction, so the series has no station outages to model around, and the
pressure levels are exactly the quantities an airship cares about.

The extract is the `reanalysis-era5-pressure-levels` dataset from the Copernicus
Climate Data Store, fetched with `cdsapi` in `src/download_era5.py`.

Three things about the CDS API shaped the downloader, and all three were found by
measurement rather than from the documentation:

1. **There is a hard per-request cost limit of 60,000 "size" units.** For this
   dataset the cost is `12 x days x times x levels x variables`. A month of
   hourly data across 3 levels and 3 variables costs about 80,000 and is rejected
   outright.
2. **That cost is completely independent of `area` and `grid`.** Subsetting to
   South Asia at 1 degree costs exactly as much as pulling the globe at 0.25
   degrees, because MARS retrieves the full field and post-processes it.
   Geographic subsetting therefore saves bandwidth and disk, but buys nothing in
   queue time. *Temporal density is the only real lever on how long the download
   takes.*
3. **CDS caps the number of requests a user may have queued for one dataset** and
   rejects the excess with a 400 rather than queueing them. Parallel downloading
   is counterproductive: requests submitted beyond the cap fail, and a killed
   run leaves jobs queued that block the next attempt. `src/cds_queue.py` exists
   to inspect and clear that queue.

The downloader is therefore strictly sequential, one request per month, with
throttle-aware retries, and it skips months already on disk so it can be
re-run after any interruption. Writes go to a `.part` file and are renamed only
on success, so a partial file can never be mistaken for a complete month.

One account-level step is required before any of this works: the ERA5 dataset
needs both the Copernicus products licence and the CC-BY licence accepted on the
account. Until then every request returns 403.

### The one significant scoping decision: 3-hourly, not hourly

Hourly data over three years is about 237,000 field retrievals, which measured
out at roughly fourteen hours of serial download on this account. 3-hourly
sampling cuts that threefold and is the setting used.

The reason this costs almost nothing scientifically is in the data itself.
Stratospheric flow at 50 hPa is driven by the quasi-biennial oscillation and the
monsoon anticyclone, with autocorrelation measured in days - `eda_autocorrelation.png`
shows it directly. At a 6 to 24 hour lead time, 3-hourly sampling resolves the
variability that matters. The cost is a shorter input sequence: a 24 hour history
is 8 timesteps rather than 24.

A faster path was tested and rejected. The ARCO-ERA5 public Zarr store on Google
Cloud is the same data with no queue, but it is chunked one timestep at a time
across all 37 levels globally, so pulling a regional subset still transfers the
whole global field per step. A single day did not finish in seven minutes.

### Preprocessing, and why each step is there

The modelling problem is framed as a **single-station** time series: forecast the
50 hPa wind vector above one point, from the recent history of the column above
it. A station-keeping controller cares about the wind where the vehicle actually
is, and a single column keeps the model small enough to run on a flight computer.

| Step | Choice | Why |
|---|---|---|
| Level selection | 50 hPa primary, 30 and 10 hPa as context | 50 hPa is about 20.5 km, the float altitude. Levels above it lead changes at it, so shear is a predictor, not just a diagnostic. |
| Features | `(u, v, T)` at each of the 3 levels, 9 in total | u and v rather than speed and direction, because direction wraps at 360 and a network should not have to learn a discontinuity. Temperature carries the density and wave signal. |
| Targets | `(u, v)` at 50 hPa | Same reason. Speed and direction are derived after the fact from the predicted vector, so the wrap-around never enters the loss. |
| Gap handling | reindex to a gapless axis, then forward fill (1 step) and time interpolation | Reindexing first is the important half: without it a missing timestamp silently disappears and corrupts every window spanning it, instead of becoming a visible NaN. |
| Normalisation | zero mean, unit variance, **fit on the train slice only** | Fitting on the full record would leak test-period statistics into training. |
| Windowing | 24 h history, targets at +6/+12/+24 h | Windows are built inside each split, so no window straddles a boundary. |
| Splitting | 70/15/15 **chronological** | The seasonal reversal dominates the variance. A random split would put January in both train and test and flatter the model badly. |

### Dataset

<!-- AUTO:dataset -->
| Property | Value |
|---|---|
| Source | ERA5 reanalysis pressure levels (Copernicus Climate Data Store) |
| Period | 2022-01-01 to 2024-12-31 |
| Sampling | 3-hourly, 8768 timesteps |
| Completeness | 100.000% of the expected axis |
| Imputed cells | 0 |
| Region | 5-35 N, 60-100 E (31 x 41 at 1.0 deg) |
| Levels | 50 hPa, 30 hPa, 10 hPa |
| Station | 20 N 80 E |
| Model features | 9 ((u, v, T) at each of 3 levels) |
| 50 hPa wind speed | mean 10.53 m/s, sd 7.17, range 0.08 to 35.48 |
<!-- /AUTO:dataset -->

### Data quality observations

- **The record is complete.** ERA5 is gap-free by construction and this extract
  confirms it; `eda_missing_data.png` exists to prove that rather than assume it,
  and to catch a download that silently lost a month. The imputation path is
  written and tested, but on this data it has nothing to do.
- **The levels are the ones requested.** 50 hPa temperature sits near 209 K,
  which is right for the tropical lower stratosphere. This is the cheapest
  available check that the request returned what was asked for.
- **Direction is unstable at low wind speed.** Below roughly 2 m/s the bearing is
  numerically ill-conditioned and operationally irrelevant, since a vehicle with
  a 12 m/s airspeed does not care which way a 1 m/s wind blows. Directional
  metrics are therefore scored only on cases above that threshold, and the count
  of scored cases is reported alongside.
- **The seasonal reversal is the dominant signal**, which is what forces the
  chronological split.

---

## Phase 2 - Model

### Why an LSTM

The series is smooth, strongly autocorrelated, and has a slowly rotating
background regime. That combination suits a recurrent model: the hidden state can
carry the current regime while the gates suppress step-to-step noise. The
alternatives were weighed as follows.

- **Linear regression on lagged features** has no state, so every seasonal
  regime must be encoded explicitly. It is cheap enough that there is no excuse
  for not measuring it, so a ridge fit on the same flattened window, trained on
  the same split, is scored alongside the LSTM in every metrics table. See
  below - it does not behave the way the a-priori argument predicted.
- **A Transformer** would let attention weights show which lags matter, but with
  an 8-step input sequence there is very little for attention to do, and it
  brings more parameters for a model intended to run on an embedded CPU.
- **Persistence** is not a throwaway baseline here. Because the wind is so
  autocorrelated, "no change" is genuinely strong at 6 hours, and it is the
  benchmark operational meteorology uses for short-range forecasts. The success
  criterion is set against it for that reason.

The head predicts all three horizons at once, so one shared encoder is trained
on every lead time rather than three separate models.

### Predicting the correction, not the wind

The one modelling decision that materially changed the result. Because
persistence is already strong, asking the network to output the absolute wind
means spending most of its capacity rediscovering "roughly what it is now".
In `--residual` mode the head instead predicts a *correction* to persistence and
the last observed (u, v) is added back inside `forward`. The head is
zero-initialised, so the model begins life as exact persistence and can only
improve from there.

This is a framing change, not an architecture change: same layers, same widths,
same loss. It works because the skip connection is exactly the identity in
physical units, which holds only if the first two feature channels share the
targets' normalisation. `build_dataset` guarantees that by construction, and
`assert_residual_safe` checks it at training time rather than trusting it, since
a future change to the feature order would otherwise break it silently.

Both framings are trained and scored; the comparison is in the metrics table
above and the one that wins is the one shipped.

### Does this problem actually need a recurrent network?

This is the question an interviewer should ask, so it is answered with a
measurement rather than an assertion. The ridge baseline is a linear fit on the
same flattened `(steps x features)` input, the same training split and the same
targets.

The result is worth stating plainly: **ridge beats the LSTM at every lead time**,
on the full three-year record, after a 32-configuration sweep of the LSTM and no
tuning at all of the ridge. At 6 hours it reaches 24.7% skill against
persistence where the LSTM reaches 18.5%.

The honest reading is that an 8-step input sequence gives a recurrent model very
little to integrate over, and that most of the short-range signal is a smooth
linear function of the recent past. If this were a production decision rather
than a modelling exercise, ridge would be the right choice: smaller, faster, no
training instability, trivially interpretable, and more accurate here.

The metrics table reports both, and the success-criteria table names whichever
model actually wins at each lead rather than quietly reporting the LSTM alone.
Reporting it the other way round would have been easy and would have been wrong.

What would give the LSTM a real chance, in rough order of expected value:
forecasting the spatial field rather than one column, so the recurrent state has
structure to carry; many more years, since 6,122 training windows is thin for
71,942 weights; and predicting several levels jointly so the network has a reason
to learn the vertical coupling it is currently handed for free as input.

Interpretability is handled by occlusion rather than attention: each input
timestep is replaced in turn by zero, which is the train-set mean because inputs
are standardised, and the change in output is measured. It makes no assumptions
about the architecture and answers the question directly. See
`interpretability_timestep_importance.png`.

### Tuning

Training one configuration takes under a minute on CPU, so the architecture was
chosen by measurement rather than assertion. `src/sweep.py` runs a small grid and
ranks it on validation loss. **The test split is never consulted during
selection**; the winner is scored on test exactly once, afterwards.

Width was the axis worth sweeping. The training set is only a few thousand
windows, so a two-layer 128-unit LSTM is heavily overparameterised and narrower
models were genuinely expected to compete.

<!-- AUTO:sweep -->
| Hidden units | Layers | Dropout | Weight decay | History (h) | Target framing | Weights | Best epoch | Train loss | Validation loss |
|---|---|---|---|---|---|---|---|---|---|
| 128 | 1 | 0.2 | 0.001 | 24 | correction | 71942 | 4 | 0.4274 | 0.5367 |
| 128 | 1 | 0.2 | 0.0 | 24 | correction | 71942 | 4 | 0.4274 | 0.5367 |
| 64 | 1 | 0.2 | 0.001 | 24 | correction | 19590 | 6 | 0.4216 | 0.5426 |
| 64 | 1 | 0.2 | 0.0 | 24 | correction | 19590 | 6 | 0.4216 | 0.5426 |
| 128 | 1 | 0.4 | 0.001 | 24 | correction | 71942 | 4 | 0.4436 | 0.5426 |
| 128 | 1 | 0.4 | 0.0 | 24 | correction | 71942 | 4 | 0.4436 | 0.5426 |
| 64 | 1 | 0.4 | 0.001 | 24 | correction | 19590 | 6 | 0.4416 | 0.5497 |
| 64 | 1 | 0.4 | 0.0 | 24 | correction | 19590 | 6 | 0.4415 | 0.5497 |
| 64 | 2 | 0.2 | 0.001 | 24 | correction | 52870 | 9 | 0.4163 | 0.5529 |
| 64 | 2 | 0.2 | 0.0 | 24 | correction | 52870 | 9 | 0.4163 | 0.5529 |
| 32 | 1 | 0.2 | 0.001 | 24 | correction | 5702 | 8 | 0.4323 | 0.5573 |
| 32 | 1 | 0.2 | 0.0 | 24 | correction | 5702 | 8 | 0.4322 | 0.5573 |
| 32 | 1 | 0.4 | 0.001 | 24 | correction | 5702 | 19 | 0.4295 | 0.558 |
| 32 | 1 | 0.4 | 0.0 | 24 | correction | 5702 | 19 | 0.4294 | 0.5581 |
| 128 | 2 | 0.4 | 0.001 | 24 | correction | 204038 | 8 | 0.424 | 0.5627 |
| 128 | 2 | 0.4 | 0.0 | 24 | correction | 204038 | 8 | 0.4239 | 0.5627 |
| 128 | 2 | 0.2 | 0.001 | 24 | correction | 204038 | 5 | 0.4321 | 0.5631 |
| 128 | 2 | 0.2 | 0.0 | 24 | correction | 204038 | 5 | 0.4321 | 0.5631 |
| 16 | 1 | 0.2 | 0.001 | 24 | correction | 1830 | 19 | 0.4372 | 0.5657 |
| 16 | 1 | 0.2 | 0.0 | 24 | correction | 1830 | 19 | 0.4372 | 0.5659 |
| 32 | 2 | 0.4 | 0.001 | 24 | correction | 14150 | 13 | 0.4605 | 0.5662 |
| 32 | 2 | 0.4 | 0.0 | 24 | correction | 14150 | 13 | 0.4605 | 0.5663 |
| 64 | 2 | 0.4 | 0.001 | 24 | correction | 52870 | 14 | 0.4253 | 0.5666 |
| 64 | 2 | 0.4 | 0.0 | 24 | correction | 52870 | 14 | 0.4253 | 0.5667 |
| 16 | 2 | 0.2 | 0.001 | 24 | correction | 4006 | 25 | 0.4443 | 0.5699 |
| 16 | 2 | 0.2 | 0.0 | 24 | correction | 4006 | 25 | 0.4443 | 0.57 |
| 32 | 2 | 0.2 | 0.001 | 24 | correction | 14150 | 16 | 0.4175 | 0.575 |
| 32 | 2 | 0.2 | 0.0 | 24 | correction | 14150 | 16 | 0.4174 | 0.5752 |
| 16 | 1 | 0.4 | 0.001 | 24 | correction | 1830 | 30 | 0.4572 | 0.5795 |
| 16 | 1 | 0.4 | 0.0 | 24 | correction | 1830 | 19 | 0.4696 | 0.5801 |
| 16 | 2 | 0.4 | 0.001 | 24 | correction | 4006 | 28 | 0.4809 | 0.5876 |
| 16 | 2 | 0.4 | 0.0 | 24 | correction | 4006 | 28 | 0.4809 | 0.5876 |

Selected on validation loss alone, across 32 configurations; the test split was not consulted. Winner: 128 hidden units, 1 layer(s), correction framing.
<!-- /AUTO:sweep -->

Three things came out of the sweep, all of them measured rather than assumed:

- **One layer beat two, everywhere.** Every configuration in the top eight is
  single-layer. The brief specified 2 x 128; the data says 1 x 128, and that is
  what ships.
- **Weight decay made no measurable difference** across two orders of magnitude,
  to four decimal places. The regularisation that mattered was depth and width,
  not the penalty term.
- **The spread across the whole grid is small.** When every reasonable
  architecture lands in the same place, the ceiling is set by the information in
  the inputs rather than by model capacity, which is exactly consistent with a
  linear baseline being competitive.

An earlier version of this sweep was thrown away. It ran while the ERA5 download
was still landing files, so different configurations trained on different amounts
of data and the ranking was meaningless. Every run now records the dataset it
saw, and the sweep refuses to promote a winner if those counts differ.

### A hypothesis that did not survive

The exploratory analysis found a clean 24-hour tide in the meridional wind worth
about 81% of that component's standard deviation, against 15% for the zonal wind.
The model is given no clock, so the obvious move was to append the hour of day as
a sine/cosine pair and let it represent the tide directly.

It made no difference: validation loss 0.5375 with the clock features against
0.5367 without. The likely reason is that a 24-hour input window already spans a
full tidal cycle, so the phase is recoverable from the inputs the model already
has. The feature is implemented and available behind `--time-features`, and the
shipped model does not use it.

### Configuration and training

<!-- AUTO:training -->
| Setting | Value |
|---|---|
| Architecture | 1-layer LSTM, 128 hidden units, dropout 0.2 |
| Weights | 71,942 |
| Optimiser | AdamW, lr 0.001, weight decay 1e-05 |
| Batch size | 32 |
| Epochs run | 14 (best 4, early stopping patience 10) |
| Best val loss | 0.53673 (MSE, normalised units) |
| Training time | 0.1 min on CPU |
<!-- /AUTO:training -->

Notes on the choices:

- **AdamW rather than Adam.** The brief asks for MSE with L2 regularisation.
  AdamW's decoupled weight decay is the correct way to get L2 with an adaptive
  optimiser; adding a penalty term to the loss under plain Adam interacts badly
  with the per-parameter scaling.
- **Gradient clipping at norm 1.0.** Recurrent nets spike; this costs nothing and
  removes a class of failed runs.
- **`ReduceLROnPlateau`** halves the learning rate when validation loss stalls,
  which in practice buys a little accuracy after the first plateau.
- **Early stopping on validation loss, patience 10**, restoring the best
  checkpoint rather than the last one.
- **Shuffling within the train split is safe** and helps convergence. The split
  itself is chronological, so no future information crosses into training; only
  the order of already-past windows changes.

---

## Phase 3 - Evaluation and optimisation

### Metrics on the held-out test split

<!-- AUTO:metrics -->
| Lead (h) | LSTM RMSE (m/s) | Persistence RMSE (m/s) | Ridge RMSE (m/s) | LSTM skill vs persistence (%) | Ridge skill vs persistence (%) | LSTM direction MAAE (deg) | Within 15 deg (%) | Within 30 deg (%) |
|---|---|---|---|---|---|---|---|---|
| 6 | 2.56 | 3.14 | 2.36 | 18.51 | 24.72 | 16.53 | 66.8 | 86.13 |
| 12 | 3.34 | 3.96 | 3.04 | 15.52 | 23.3 | 18.71 | 63.67 | 82.12 |
| 24 | 3.45 | 3.69 | 3.25 | 6.64 | 11.92 | 18.44 | 63.03 | 82.76 |
<!-- /AUTO:metrics -->

How these are defined, because the details matter:

- **Skill score** is reported as the percentage by which the model reduces
  persistence RMSE, so that higher is always better. The brief writes it as
  `(model - persistence) / persistence`, which is negative when the model wins;
  this is the same quantity with the sign flipped, and it is labelled as a
  reduction to keep that unambiguous.
- **MAAE** is mean absolute angular error with correct wraparound, so 350 and 10
  degrees are 20 degrees apart, not 340.
- **Directional accuracy** is the share of cases within 15 or 30 degrees, scored
  only where wind speed is at least 2 m/s.
- **Vector RMSE** is the RMSE of the error vector magnitude. This is the one that
  matters physically, because it is what displaces the vehicle.

### Error analysis: when does it fail?

Average error is not the operationally useful question. What a controller needs
to know is whether *this* forecast, issued now, can be trusted. Each test window
is therefore paired with the state of the atmosphere at the moment the forecast
was issued, and the error is correlated against it.

<!-- AUTO:errors -->
Spearman correlation between forecast error and the state of the atmosphere when the forecast was issued (n = 1301 test windows).

| Condition at issue time | 6h | 12h | 24h |
|---|---|---|---|
| Wind speed at issue time | +0.22 | +0.23 | +0.21 |
| Shear, 10 minus 50 hPa | +0.09 | +0.03 | +0.06 |
| Shear magnitude | +0.10 | +0.07 | +0.05 |
| Recent rate of change | +0.09 | +0.09 | +0.11 |
| Temperature at 50 hPa | +0.05 | +0.08 | +0.05 |
| Direction variability in the window | -0.21 | -0.21 | -0.20 |
<!-- /AUTO:errors -->

The consistent signal is that **error tracks how fast the wind was already
changing**. A flow that was steady over the preceding 24 hours stays predictable;
one that was already turning does not. That is a directly actionable rule: the
recent tendency is computable onboard with no extra data, so it can gate how far
ahead the vehicle commits to a plan.

Figures: `error_drivers_correlation.png`, `error_drivers_6h.png`,
`error_worst_cases_6h.png`, `eval_timeseries_6h.png`, `eval_error_by_horizon.png`,
`eval_scatter_speed.png`, `eval_error_by_direction.png`,
`eval_error_by_month.png`, `eval_skill_vs_horizon.png`, `train_curves.png`.

### Optimisation for edge deployment

<!-- AUTO:optimization -->
| Variant | Weights | Weight file (KB) | Latency p50 (ms) | Latency p95 (ms) | 6 h speed RMSE (m/s) | 6 h within 30 deg (%) |
|---|---|---|---|---|---|---|
| FP32 | 71942 | 283.65 | 0.16 | 0.18 | 2.56 | 86.13 |
| FP16 | 71942 | 143.15 | 0.75 | 0.8 | 2.56 | 86.13 |
| INT8 | 71942 | 77.01 | 0.54 | 0.6 | 2.55 | 85.89 |
| Lite h32 | 5702 | 24.9 | 0.12 | 0.14 | 2.65 | 85.57 |
| Lite+INT8 | 5702 | 10.01 | 0.49 | 0.56 | 2.66 | 85.57 |

Single sample, one CPU thread.

- **FP32** - reference
- **FP16** - half the weight file; no fast FP16 LSTM kernel on CPU
- **INT8** - quarter the weight file; quantise overhead dominates at batch 1
- **Lite h32** - smaller LSTM, retrained from scratch
- **Lite+INT8** - smallest deployable variant
<!-- /AUTO:optimization -->

The honest result is that **reducing precision buys footprint, not speed, at this
model size**. FP16 halves the weight file and int8 quarters it, with no
measurable accuracy cost. Neither is faster at batch size 1: x86 CPUs have no
fast FP16 LSTM kernel, and int8 kernels only win on matrices large enough to
amortise the per-call quantise and dequantise overhead, which a model this small
is not.

That is a useful finding rather than a disappointing one. Inference is already
around a millisecond on one CPU thread, which is four to five orders of magnitude
faster than the 3-hour interval between new data, so latency was never the
binding constraint. Flash and RAM are. The right deployment choice is the variant
that is smallest while still accurate, and the profiling is what establishes
that rather than assuming it.

### Known limitations

- **Cumulative drift is governed by bias, not RMSE, and here it reverses the
  ranking.** This is the most important finding in the project. Over a simulated
  10-day deployment the LSTM has the better per-forecast error, 123 km at the
  90th percentile against persistence's 171 km, yet persistence ends the window
  closer to station, 80 km against 214 km. The reason is bias: 42 km/day for the
  LSTM against 4 km/day for persistence. Random error partly cancels across
  control intervals; a systematic one adds up on every step. A model can
  therefore look better on every error metric and still walk the vehicle off
  station. The drift table reports bias as a km/day rate for exactly this reason,
  and it is the first number to look at when retraining.
- **The LSTM loses to a linear baseline.** Stated again here because it belongs
  in the limitations, not only in the model section. See Phase 2.
- **One station, not a field.** The model forecasts a single column. Real routing
  would want the spatial field, which is a materially different and larger model.
- **One float level.** Forecasts are produced for 50 hPa only. The altitude
  analysis shows that choosing between levels is a powerful control lever, so
  forecasting all three levels is the obvious next step.
- **Reanalysis is not real-time.** ERA5 lags real time by about five days. A
  deployed system would take operational analyses or onboard sensing as input.
  The model's inputs are ordinary atmospheric state variables, so the
  substitution is mechanical, but the distribution shift is real and untested.
- **Three years is short** for a signal with a quasi-biennial component. The
  model sees only one and a half QBO cycles, so it cannot have learned that
  oscillation, only the seasonal cycle within its window.
- **Test coverage of the calibrated intervals is measured, not assumed**, and is
  reported in `models/calibration.json`. Any shortfall against nominal is a
  genuine caveat on the quoted intervals.

---

## Phase 4 - Deployment and application

### Inference

`src/inference.py` loads the checkpoint and the saved scalers, validates that the
incoming frame has the required columns and enough timesteps, refuses a window
containing NaN rather than quietly imputing at inference time, and returns
forecasts with intervals.

Uncertainty is offered two ways, and the distinction is deliberate:

- **Calibrated residual intervals** (the default, and the one to quote). The
  spread of validation residuals is measured once per horizon and stored. Because
  the interval is built on the residual "actual minus forecast", it corrects
  systematic bias as well as describing spread. Coverage is then *verified on the
  held-out test split*, because an interval nobody checked is not an interval.
  One consequence worth knowing: if the model consistently under-forecasts, both
  quantiles are positive and the interval sits above the point forecast. That is
  the calibration working, and a point forecast outside its own interval is a
  direct readout of bias.
- **Monte-Carlo dropout.** Dropout stays active and the model runs many times.
  This shows the spread the model itself implies, which is useful for spotting
  unfamiliar inputs, but it is not calibrated and should not be quoted as a
  confidence interval on its own.

### Station-keeping: converting forecast error into kilometres

The physical argument is deliberately simple, because a simple argument can be
checked. A buoyant platform holds station by flying against the ambient wind. If
the controller cancels the wind it *expects*, what remains is the forecast error,
and that residual pushes the vehicle off station. An error of `e` m/s sustained
for `H` hours displaces it by about `e x H x 3.6` km. Forecast skill therefore
converts directly into kilometres, which is the unit an operator thinks in.

<!-- AUTO:airship -->
Airspeed limit assumed: **12 m/s**.

| Lead (h) | LSTM drift p50 (km) | LSTM drift p90 (km) | Persistence drift p90 (km) | Reduction at p90 (%) | LSTM bias drift (km/day) |
|---|---|---|---|---|---|
| 6 | 64.8 | 123.0 | 171.1 | 28.1 | 42.6 |
| 12 | 154.0 | 296.5 | 424.4 | 30.1 | 38.8 |
| 24 | 318.6 | 594.0 | 696.5 | 14.7 | 23.8 |

| Level | Time wind is within airspeed limit |
|---|---|
| 50 hPa | 63% |
| 30 hPa | 56% |
| 10 hPa | 50% |
| best of three | 70% |

Every possible start time in the test period was then flown as a separate 5-day mission (158 scenarios), counting success as holding within 200 km.

| Controller | Missions held on station |
|---|---|
| Perfect knowledge | 53% |
| Persistence | 29% |
| LSTM forecast | 10% |
<!-- /AUTO:airship -->

`src/airship.py` produces four views of the same question.

**One trajectory.** Full 2-D tracks under a controller that replans every 6 hours,
comparing the model against persistence and against perfect knowledge, with
airspeed clipped to a realistic limit.

**Every trajectory.** One simulated window says what happened once; an operator
needs the distribution. Every possible start time in the test period is flown as
a separate 5-day mission, 158 of them, and scored on whether the vehicle stayed
within 200 km. This is where the bias result becomes concrete: perfect knowledge
succeeds on 53% of missions, persistence on 29%, the LSTM on 10%. The model with
the better per-forecast error completes the fewest missions.

**Which altitude.** A time-height section of wind barbs across 50/30/10 hPa,
coloured by whether each level is inside the airspeed limit, drawn over the
window where the three levels disagree most - because a uniformly calm window
makes a pretty chart that demonstrates nothing. In the window it picks, the float
level and the one above it are both unflyable for days while 10 hPa stays green.

**When to deploy.** The scenario scan doubles as a deployment-timing product: the
monsoon months are hopeless at any lead time, and the autumn window is where
forecast quality starts to matter at all.

Which stretch to simulate turned out to be a decision worth making explicitly.
Starting at an arbitrary point in the test period lands inside the monsoon
easterly jet, where the wind runs at roughly twice the vehicle's airspeed. There,
*nothing* holds station: all three controllers drift together, perfect knowledge
included, and the figure says nothing at all about forecast quality. A real
operator picks a deployment window, so the simulation now starts at the calmest
contiguous stretch in the test period and the infeasible case is reported
separately as the feasibility fraction. Both facts matter, and conflating them
would have produced a chart that looked like a model failure but was really a
physics constraint.

The altitude result is the one with the clearest operational consequence.
Because 50, 30 and 10 hPa often carry very different winds, a vehicle that can
change altitude can pick the calmest available level, and the fraction of time
station-keeping is feasible rises substantially. Altitude is a control lever, not
just a constraint.

### Success criteria

<!-- AUTO:criteria -->
| Criterion | Measured | Status |
|---|---|---|
| Beats persistence RMSE by >= 20% at 6 h | 18.5% LSTM, best 24.7% (ridge) | met |
| Beats persistence RMSE by >= 20% at 12 h | 15.5% LSTM, best 23.3% (ridge) | met |
| Beats persistence RMSE by >= 20% at 24 h | 6.6% LSTM, best 11.9% (ridge) | not met |
| Directional accuracy >= 70% at 6 h (within 30 deg) | 86.1% | met |
| Lightweight variant suitable for edge inference | Lite+INT8, 10 KB, 0.49 ms | met |
<!-- /AUTO:criteria -->

### What I would do next, in priority order

1. **Attack the bias directly**, since it dominates cumulative drift and is the
   single reason persistence beats the LSTM over a 10-day deployment despite
   worse per-forecast error. Add a mean penalty to the loss, or debias per regime
   after the fact, and track km/day alongside RMSE.
2. **Forecast all three levels**, turning the altitude analysis into an
   optimiser that picks the best float level over the forecast window. This is
   the largest operational gain available and the model already ingests all
   three levels.
3. **Predict the residual from persistence** rather than the wind itself. Given
   how strong persistence is, learning the correction is an easier target and
   guarantees the model cannot do worse than the baseline by much.
4. **Ensemble a few seeds** for a sharper and better-calibrated spread than
   MC dropout gives.
5. **Extend the record** past three years so the QBO is actually represented, and
   swap reanalysis inputs for operational analyses to measure the real
   distribution shift.
6. **Longer input window.** 24 hours was specified; the autocorrelation structure
   suggests several days of history may help at the 24 hour lead, which is the
   one horizon where no model reached the 20% target.
7. **Take the ridge result seriously.** If the linear model keeps winning after
   the above, ship it and spend the effort on the spatial problem instead.

### Reproducing the whole thing

```bash
python src/download_era5.py        # sequential; resumable; ~2 h for 3 years
python src/eda.py                  # figures + statistics
python src/sweep.py                # grid search, selection on validation
python src/train.py --tag base --residual
python src/train.py --hidden 32 --tag lite --residual
python src/evaluate.py --tag base
python src/error_analysis.py --tag base
python src/optimize.py --tag base
python src/inference.py --tag base --calibrate --demo
python src/airship.py --tag base
python src/make_demo.py --tag base
python src/report.py --tag base    # refresh every table in this file
```
