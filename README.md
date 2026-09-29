# Stratospheric wind forecasting for airship station-keeping

Forecasting the wind at 20 km over South Asia, 6 to 24 hours ahead, and turning
that forecast into the number an airship operator actually needs: how far off
station the vehicle will drift.

A high-altitude airship floats near 20 km and has no way to hold position except
to fly against the ambient wind. Its airspeed is small, roughly 10 to 15 m/s, so
station-keeping is only possible when the wind is weak or well predicted. This
project trains an LSTM on ERA5 reanalysis data to forecast the 50 hPa wind vector,
scores it against the persistence baseline that operational meteorology uses, and
converts the residual error into kilometres of drift.

## Results

Held-out test split, chronologically after everything the model was trained on:

<!-- AUTO:metrics -->
| Lead (h) | LSTM RMSE (m/s) | Persistence RMSE (m/s) | Ridge RMSE (m/s) | LSTM skill vs persistence (%) | Ridge skill vs persistence (%) | LSTM direction MAAE (deg) | Within 15 deg (%) | Within 30 deg (%) |
|---|---|---|---|---|---|---|---|---|
| 6 | 2.56 | 3.14 | 2.36 | 18.51 | 24.72 | 16.53 | 66.8 | 86.13 |
| 12 | 3.34 | 3.96 | 3.04 | 15.52 | 23.3 | 18.71 | 63.67 | 82.12 |
| 24 | 3.45 | 3.69 | 3.25 | 6.64 | 11.92 | 18.44 | 63.03 | 82.76 |
<!-- /AUTO:metrics -->

Against the project's success criteria:

<!-- AUTO:criteria -->
| Criterion | Measured | Status |
|---|---|---|
| Beats persistence RMSE by >= 20% at 6 h | 18.5% LSTM, best 24.7% (ridge) | met |
| Beats persistence RMSE by >= 20% at 12 h | 15.5% LSTM, best 23.3% (ridge) | met |
| Beats persistence RMSE by >= 20% at 24 h | 6.6% LSTM, best 11.9% (ridge) | not met |
| Directional accuracy >= 70% at 6 h (within 30 deg) | 86.1% | met |
| Lightweight variant suitable for edge inference | Lite+INT8, 10 KB, 0.49 ms | met |
<!-- /AUTO:criteria -->

Station-keeping, which is what the forecast is for:

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
<!-- /AUTO:airship -->

## Four findings worth knowing

**This problem does not need a recurrent network.** A ridge regression on the same
flattened 24-hour window beats the LSTM at every lead time: 24.7% against 18.5%
skill at 6 hours. The baseline is reported next to the model in every table
rather than quietly omitted. An 8-step input sequence gives a recurrent model
very little to integrate over, and most of the short-range signal turns out to be
a smooth linear function of the recent past. If this were a production decision
rather than a modelling exercise, ridge would be the honest choice: smaller,
faster, no training instability, trivially interpretable.

**Cumulative drift is driven by bias, not RMSE, and this flips the ranking.** Over
a simulated 10-day deployment the LSTM has the better per-forecast error, holding
within 123 km at the 90th percentile against persistence's 171 km. Yet persistence
finishes closer to station, 80 km against 214 km, because the LSTM carries a
42 km/day bias while persistence carries 4 km/day. Random error partly cancels
across control intervals; a systematic one adds up on every single step. Any
station-keeping model should be scored on bias, not only on RMSE.

**Altitude is a control lever, not just a constraint.** The 50, 30 and 10 hPa
levels often carry very different winds. Station-keeping is feasible 63% of the
time at 50 hPa alone, but 70% if the vehicle can pick the calmest of the three.
See `results/figures/airship_altitude_choice.png`.

**Reducing precision buys footprint, not speed, at this model size.** Int8 cuts
the weight file to a quarter with no measurable accuracy cost, but it is *slower*
than FP32 at batch size 1, because the model is too small to amortise the
conversion overhead. The retrained 32-unit variant is the one that is both
smaller and faster: 25 KB against 284 KB and 0.12 ms against 0.16 ms, for 3.7%
more error. Inference was never the binding constraint against a 3-hour data
cadence; flash and RAM are, and profiling is what establishes which variant to
ship rather than assumption.

## Layout

```
src/
  config.py          every tunable in one place
  download_era5.py   sequential, resumable CDS downloader
  cds_queue.py       inspect or clear the CDS request queue
  data.py            loading, cleaning, windowing, chronological splits
  model.py           the LSTM and the persistence baseline
  metrics.py         wind verification metrics with correct angle wraparound
  train.py           training with early stopping
  sweep.py           small grid search, selected on validation only
  evaluate.py        metrics, figures, occlusion interpretability
  error_analysis.py  when the model fails, and what that correlates with
  optimize.py        FP16 / int8 variants, latency and footprint profiling
  inference.py       deployable forecaster with calibrated intervals
  airship.py         forecast error -> station-keeping drift
  make_demo.py       builds results/demo.html, a standalone interactive page
  viz.py             one validated palette and rc block for every figure
  report.py          writes the tables in README.md and skills.md from results/
notebooks/
  01_exploratory_analysis.ipynb
results/
  demo.html          interactive forecast viewer, opens offline with no deps
  figures/           every plot
  metrics/           every table, as CSV and JSON
models/              checkpoints, scalers, interval calibration
data/raw/            monthly ERA5 netCDF (git-ignored, re-downloadable)
skills.md            technical log: decisions, rationale, limitations
```

## Setup

Python 3.11 is the reference interpreter; 3.10 to 3.12 also work.

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # Windows
# .venv/bin/python -m pip install -r requirements.txt     # Linux / macOS
```

To download data you need a free Copernicus Climate Data Store account. Put your
key in `~/.cdsapirc`:

```
url: https://cds.climate.copernicus.eu/api
key: <your-key>
```

**Accept the licences first.** The ERA5 pressure-levels dataset requires both the
Copernicus products licence and CC-BY to be accepted on your account, or every
request returns 403. Either click through on the dataset page, or:

```python
from ecmwf.datastores import Client
c = Client(url="https://cds.climate.copernicus.eu/api", key="<your-key>")
c.accept_licence("licence-to-use-copernicus-products", revision=12)
c.accept_licence("cc-by", revision=1)
```

## Running it

```bash
python src/download_era5.py        # sequential and resumable; ~2 h for 3 years
python src/eda.py                  # figures and statistics
python src/sweep.py                # grid search, picks on validation
python src/train.py --tag base --residual
python src/train.py --hidden 32 --tag lite --residual   # the edge variant
python src/evaluate.py --tag base
python src/error_analysis.py --tag base
python src/optimize.py --tag base
python src/inference.py --tag base --calibrate  # fit and verify intervals
python src/airship.py --tag base
python src/make_demo.py --tag base
python src/report.py --tag base    # refresh the tables in this file
```

The downloader is safe to interrupt and re-run: it skips months already on disk.
If a run is killed, clear any jobs it left queued with
`python src/cds_queue.py --cancel`, or the next run will be rejected.

## Forecasting from new data

`Forecaster` takes a DataFrame indexed by time whose last rows are the most recent
observations. It needs the nine feature columns and at least 8 timesteps of
3-hourly history. It validates the input and refuses a window containing NaN
rather than quietly imputing at inference time.

```python
import sys; sys.path.insert(0, "src")
from inference import Forecaster

fc = Forecaster(tag="base")
print(fc.features)        # u50 v50 t50 u30 v30 t30 u10 v10 t10

for f in fc.forecast(df):                 # df: >= 8 rows, 3-hourly
    print(f.horizon_h, f.speed, f.direction_deg, f.speed_lo, f.speed_hi)
```

Intervals are calibrated on validation residuals and their coverage is verified on
the test split; the measured coverage is in `models/calibration.json`. Pass
`use_mc=True` for Monte-Carlo dropout spread instead, which is useful for spotting
unfamiliar inputs but is not calibrated and should not be quoted as a confidence
interval.

To reproduce the demo against the downloaded record:

```bash
python src/inference.py --tag base --demo
```

## The interactive viewer

`results/demo.html` is a single self-contained file: the data is inlined, there
are no dependencies and no network calls, so it opens straight from a clone. It
plots observed against LSTM, persistence and ridge forecasts over the whole test
period, with a crosshair readout that shows every series at once, a lead-time
selector, a clickable legend, and a table view so no number is reachable only by
hovering. Regenerate it with `python src/make_demo.py --tag base`.

## When to trust it, and when not to

- **Trust it at 6 hours** more than at 24. Skill against persistence falls as lead
  time grows, and the drift table shows how fast.
- **Watch the bias, not just the RMSE**, if you care about holding station over
  days rather than hours.
- **It forecasts one column at one level.** It is not a routing model, and it says
  nothing about 30 or 10 hPa beyond using them as inputs.
- **It is trained on reanalysis, which lags real time by about five days.** A
  deployed system would feed it operational analyses or onboard sensing. The
  inputs are ordinary state variables so the swap is mechanical, but the
  distribution shift is real and has not been measured here.
- **Three years is short** for a signal with a quasi-biennial component. The model
  has seen roughly one and a half QBO cycles, so it has learned the seasonal cycle
  and not that oscillation.
- **Even a perfect forecast cannot always hold station.** When the wind exceeds
  the vehicle's airspeed, no forecast helps; the simulation shows this explicitly.

`skills.md` has the full technical log: what was decided, why, and what was
measured rather than assumed.

## References

**Data.** Hersbach, H. et al. (2020), *The ERA5 global reanalysis*, Quarterly
Journal of the Royal Meteorological Society 146, 1999-2049.
doi:10.1002/qj.3803. Dataset: Copernicus Climate Change Service (C3S) Climate Data
Store, *ERA5 hourly data on pressure levels from 1940 to present*.
doi:10.24381/cds.bd0915c6. Contains modified Copernicus Climate Change Service
information; neither the European Commission nor ECMWF is responsible for any use
of it.

**Method.** Hochreiter, S. and Schmidhuber, J. (1997), *Long short-term memory*,
Neural Computation 9(8), 1735-1780. Gal, Y. and Ghahramani, Z. (2016), *Dropout as
a Bayesian approximation*, ICML, for the Monte-Carlo dropout variant. Loshchilov,
I. and Hutter, F. (2019), *Decoupled weight decay regularization*, ICLR, for
AdamW.

**Atmosphere.** Baldwin, M. P. et al. (2001), *The quasi-biennial oscillation*,
Reviews of Geophysics 39(2), 179-229, on the dominant mode of tropical
stratospheric wind variability. Randel, W. J. and Park, M. (2006), on the Asian
monsoon anticyclone in the upper troposphere and lower stratosphere.

## Licence note

ERA5 data is redistributed under CC-BY-4.0 terms; raw files are git-ignored and
should be re-downloaded from the CDS rather than copied from a repository.
