# Exoplanet Finder

A two-view 1D CNN that looks at NASA Kepler light curves and separates real planet transits from signals that only look like one, such as eclipsing binary stars and instrument noise.

> **Results:** _fill in from `results/metrics.csv` after running the pipeline._

| Model (test stars, never seen in training) | AUC | Avg. precision | Precision @0.5 | Recall @0.5 |
|---|---|---|---|---|
| Baseline: gradient boosting on period, duration, depth | – | – | – | – |
| Two-view CNN | – | – | – | – |

![planet vs eclipsing binary](results/figures/planet_vs_eb.png)

## How it works

1. **Data:** the Kepler Q1–Q17 DR24 TCE table (NASA Exoplanet Archive), with 15,737 labeled signals: `PC` = planet candidate, `AFP` = astrophysical false positive, `NTP` = non-transiting phenomenon. Each star's light curve comes from NASA MAST via `lightkurve`.
2. **Split by star:** all signals from one star go into the same train/val/test split (80/10/10), so the test score measures performance on stars the model has never seen. Kepler-90 is always in the test split.
3. **Flatten:** remove slow stellar variability, with every known transit masked first so the dips aren't smoothed away.
4. **Fold and bin:** the **global view** shows the whole orbit in 2001 bins. The **local view** shows ±4 transit durations in 201 bins. Both are normalized so the median is 0 and the deepest point is −1.
5. **Model:** two convolutional branches (one per view) feed into fully connected layers, giving one output: P(planet).
6. **Baseline:** gradient boosting on period, duration and depth only. It knows how big the dip is but not its shape.

## Quick start (Windows / PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cu124   # GPU build; drop the URL for CPU
pip install -r requirements.txt

python -m src.get_tce_table                        # 1-2. download TCE table, split by star
python -m src.download_lightcurves --max-stars 300 # 3. start small (resumable; drop the flag for all stars)
python -m src.views                                # 4-5. flatten, fold, bin -> data/views.npz
python -m src.baseline                             # 6. baseline on the full table
python -m src.train                                # 7. train the CNN
python -m src.evaluate                             # 8. test metrics + figures
python -m src.kepler90                             # 9. try it on Kepler-90 / Kepler-90i
```

Every run appends a row to `results/metrics.csv`, which is where the numbers in the results table above come from.

## Project layout

```
data/            downloaded + generated data (git-ignored)
notebooks/       exploration
src/             pipeline scripts (run with python -m src.<name>)
models/          trained weights (git-ignored)
results/         metrics.csv, test_predictions.csv, figures/
```

## Things this project is careful about

- **No leakage from NASA's own classifier:** the `av_pred_class` and `av_pp_*` columns are never used.
- **Split by star, not by signal:** a random split would put the same star's noise in both train and test.
- **Units:** `tce_duration` is in hours, while `tce_period` and `tce_time0bk` are in days.
- **No accuracy headline:** about 75% of signals are not planets, so always answering "not a planet" already scores about 75% accuracy. AUC and precision/recall are reported instead.

## Limitations

- Trained on Kepler data only. TESS light curves have different cadence and noise.
- Labels are imperfect: some "planet candidates" are really eclipsing binaries, and some of the model's "false positives" are probably label errors.
- `PC` means planet *candidate*, not a confirmed planet.

## References

- Shallue & Vanderburg (2018), *Identifying Exoplanets with Deep Learning: A Five-planet Resonant Chain around Kepler-80 and an Eighth Planet around Kepler-90*, AJ 155, 94.
- NASA Exoplanet Archive, Kepler Q1–Q17 DR24 TCE table.
