"""Step 1 + 2: download the Kepler DR24 TCE table and split it BY STAR.

    python -m src.get_tce_table

Writes data/dr24_tce.csv with a `planet` label (PC = 1, AFP/NTP = 0) and a `split`
column (train / val / test). All TCEs of a star share one split.
"""
import io

import numpy as np
import pandas as pd
import requests

from .config import KEPLER90_KEPID, SEED, TCE_CSV

TAP = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
# Only the columns we need. Deliberately NOT av_pred_class / av_pp_*: those are the
# Autovetter's predictions of this same label, so using them would be leakage.
COLUMNS = ["kepid", "tce_plnt_num", "tce_period", "tce_time0bk", "tce_duration",
           "tce_depth", "tce_model_snr", "av_training_set"]


def download():
    query = f"select {','.join(COLUMNS)} from q1_q17_dr24_tce"
    r = requests.get(TAP, params={"query": query, "format": "csv"}, timeout=300)
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text))


def split_by_star(df, val_frac=0.1, test_frac=0.1, seed=SEED):
    rng = np.random.default_rng(seed)
    stars = df.kepid.unique()
    rng.shuffle(stars)
    stars = [s for s in stars if s != KEPLER90_KEPID]
    n_test, n_val = int(len(stars) * test_frac), int(len(stars) * val_frac)
    test = set(stars[:n_test]) | {KEPLER90_KEPID}
    val = set(stars[n_test:n_test + n_val])
    return df.kepid.map(lambda k: "test" if k in test else "val" if k in val else "train")


def main():
    raw = download()
    print(f"downloaded {len(raw):,} TCEs; label counts:\n{raw.av_training_set.value_counts(dropna=False)}")
    df = raw[raw.av_training_set.isin(["PC", "AFP", "NTP"])].copy()
    df["planet"] = (df.av_training_set == "PC").astype(int)
    df["split"] = split_by_star(df)
    df["tce_id"] = df.kepid.astype(str) + "_" + df.tce_plnt_num.astype(str)
    df.to_csv(TCE_CSV, index=False)

    print(f"\nkept {len(df):,} labeled TCEs on {df.kepid.nunique():,} stars -> {TCE_CSV}")
    print(df.groupby("split").agg(tces=("planet", "size"), stars=("kepid", "nunique"),
                                  planet_frac=("planet", "mean")).round(3))
    # sanity check: no star appears in two splits
    assert df.groupby("kepid").split.nunique().max() == 1


if __name__ == "__main__":
    main()
