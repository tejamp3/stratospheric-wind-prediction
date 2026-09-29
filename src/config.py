"""Central configuration for the stratospheric wind forecasting project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
METRICS = RESULTS / "metrics"

for _d in (DATA_RAW, DATA_PROC, MODELS, RESULTS, FIGURES, METRICS):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- ERA5 request
DATASET = "reanalysis-era5-pressure-levels"
VARIABLES = ["u_component_of_wind", "v_component_of_wind", "temperature"]
# 50 hPa ~ 20.5 km, 30 hPa ~ 23.8 km, 10 hPa ~ 31 km.
# 50 hPa is the primary airship float level; 30/10 hPa give the shear context.
PRESSURE_LEVELS = ["50", "30", "10"]
LEVELS_HPA = [50, 30, 10]
PRIMARY_LEVEL = 50

# India / South Asia. CDS area order is [North, West, South, East].
AREA = [35, 60, 5, 100]
GRID = [1.0, 1.0]  # stratospheric flow is large-scale; 1 deg is ample and 16x smaller

YEARS = [2022, 2023, 2024]

# 3-hourly sampling. CDS bills a request by fields (days x times x levels x
# vars) and that cost is independent of `area`/`grid` - MARS pulls the global
# field either way - so temporal density is the only real lever on retrieval
# time. Hourly over 3 years is a ~14 h serial download on a one-job-at-a-time
# account. Stratospheric flow at 50 hPa is driven by the QBO and the monsoon
# anticyclone, with autocorrelation measured in days, so 3-hourly loses
# essentially no signal at a 6-24 h lead time while cutting the download 3x.
STEP_HOURS = 3
HOURS = [f"{h:02d}:00" for h in range(0, 24, STEP_HOURS)]
STEPS_PER_DAY = len(HOURS)

# ------------------------------------------------------------- target location
# Domain-centre column used as the "airship station" for the time-series model.
TARGET_LAT = 20.0
TARGET_LON = 80.0

# ------------------------------------------------------------------ windowing
INPUT_HOURS = 24           # history fed to the model, in hours
HORIZONS = [6, 12, 24]     # forecast lead times in hours
# The model works in timesteps, not hours; at 3-hourly sampling a 24 h history
# is 8 steps and the horizons are 2 / 4 / 8 steps ahead.
INPUT_STEPS = INPUT_HOURS // STEP_HOURS
HORIZON_STEPS = [h // STEP_HOURS for h in HORIZONS]
SPLITS = (0.70, 0.15, 0.15)

# ------------------------------------------------------------------- training
SEED = 42
HIDDEN = 128
NUM_LAYERS = 2
DROPOUT = 0.2
LR = 1e-3
WEIGHT_DECAY = 1e-5        # L2 regularisation
BATCH_SIZE = 32
MAX_EPOCHS = 100
PATIENCE = 10
