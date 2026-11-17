"""Step 6: the baseline your CNN has to beat.

    python -m src.baseline

Gradient boosting on period, duration and depth only. It knows how long and how deep the
dip is, but not its SHAPE, which is exactly what the CNN adds. It only needs the TCE table,
so it is evaluated on all labeled TCEs (and on the same TCEs as the CNN in src.evaluate).
"""
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from .config import SEED, TCE_CSV
from .metrics import classification_metrics, log_run

FEATURES = ["tce_period", "tce_duration", "tce_depth"]


def fit_baseline(tce):
    train = tce[tce.split == "train"]
    return HistGradientBoostingClassifier(random_state=SEED).fit(train[FEATURES], train.planet)


def predict_baseline(model, df):
    return model.predict_proba(df[FEATURES])[:, 1]


def main():
    tce = pd.read_csv(TCE_CSV)
    model = fit_baseline(tce)
    for split in ("val", "test"):
        df = tce[tce.split == split]
        log_run("baseline_gbm", split, classification_metrics(df.planet, predict_baseline(model, df)),
                notes="all labeled TCEs")


if __name__ == "__main__":
    main()
