"""Shared paths and constants for the Exoplanet Finder pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
LC_DIR = DATA / "lightcurves"        # one cached .npz per star (time, flux)
TCE_CSV = DATA / "dr24_tce.csv"      # labeled TCE table + split column
VIEWS_NPZ = DATA / "views.npz"       # global / local views for every TCE we could build
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
METRICS_CSV = RESULTS / "metrics.csv"

SEED = 42

# view sizes from Shallue & Vanderburg (2018)
GLOBAL_BINS = 2001
LOCAL_BINS = 201
LOCAL_HALF_WIDTH_DURATIONS = 4       # local view spans +/- 4 transit durations

# Kepler-90 (the system where Kepler-90i was found). Always kept in the TEST split
# so the "rediscovery" demo is on a star the model never trained on.
KEPLER90_KEPID = 11442793

for d in (DATA, LC_DIR, MODELS, RESULTS, FIGURES):
    d.mkdir(parents=True, exist_ok=True)
