"""Step 3: download each star's Kepler long-cadence light curve from MAST and cache it.

    python -m src.download_lightcurves --max-stars 300      # quick start
    python -m src.download_lightcurves                      # every star (~hours, resumable)

Each star is saved once as data/lightcurves/<kepid>.npz (time in BKJD days, normalized
PDCSAP flux). The raw FITS files are deleted after stitching to save disk space.
Re-running skips stars that are already cached, so you can stop and resume any time.

Network drops and MAST rate limits (HTTP 429/503) are retried with growing pauses
(30 s, 1, 2, 4, 8, 10 min), so a short outage pauses the run instead of failing every star.
"""
import argparse
import shutil
import tempfile
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import DATA, KEPLER90_KEPID, LC_DIR, SEED, TCE_CSV

warnings.filterwarnings("ignore")  # lightkurve/astropy are noisy about units and quality flags

RETRY_WAITS = [30, 60, 120, 240, 480, 600]  # seconds; ~25 min total before a star is given up


def lc_path(kepid):
    return LC_DIR / f"{int(kepid)}.npz"


def fetch(kepid, tmp):
    import lightkurve as lk

    search = lk.search_lightcurve(f"KIC {kepid}", mission="Kepler", author="Kepler",
                                  cadence="long")
    if len(search) == 0:
        return "no data"
    lcs = search.download_all(download_dir=tmp, flux_column="pdcsap_flux")
    lc = lcs.stitch().remove_nans()          # stitch normalizes each quarter to median 1
    np.savez_compressed(lc_path(kepid), time=lc.time.value.astype(np.float64),
                        flux=np.asarray(lc.flux.value, dtype=np.float32))
    return "ok"


def download_star(kepid):
    if lc_path(kepid).exists():
        return kepid, "cached"
    error = None
    for wait in [0] + RETRY_WAITS:
        time.sleep(wait)
        tmp = tempfile.mkdtemp(prefix=f"kic{kepid}_")   # fresh folder: no half-written files
        try:
            return kepid, fetch(kepid, tmp)
        except Exception as e:  # network drop, rate limit, corrupt file...
            error = f"error: {type(e).__name__}: {e}"
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    return kepid, error


def choose_stars(tce, max_stars):
    """Random subset of stars (same seed every time, so a bigger run extends a smaller one)."""
    stars = tce.kepid.drop_duplicates().sample(frac=1, random_state=SEED).tolist()
    if max_stars:
        stars = stars[:max_stars]
    if KEPLER90_KEPID not in stars:
        stars.append(KEPLER90_KEPID)  # always needed for the Kepler-90 demo
    return stars


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-stars", type=int, default=None, help="limit for a quick first run")
    ap.add_argument("--workers", type=int, default=4, help="parallel downloads (be nice to MAST)")
    args = ap.parse_args()

    tce = pd.read_csv(TCE_CSV)
    stars = choose_stars(tce, args.max_stars)
    todo = [s for s in stars if not lc_path(s).exists()]
    print(f"{len(stars):,} stars selected, {len(stars) - len(todo):,} already cached, "
          f"{len(todo):,} to download", flush=True)
    failed_csv = DATA / "failed_stars.csv"
    failed_csv.unlink(missing_ok=True)   # only list failures from THIS run

    failures, ok = [], 0
    # processes, not threads: lightkurve silences output by swapping sys.stdout globally,
    # which breaks (closed-file errors) when several threads download at once
    with ProcessPoolExecutor(args.workers) as pool:
        futures = [pool.submit(download_star, s) for s in todo]
        bar = tqdm(as_completed(futures), total=len(futures), unit="star")
        for f in bar:
            kepid, status = f.result()
            if status in ("ok", "cached"):
                ok += 1
            else:
                failures.append((kepid, status))
            bar.set_postfix(saved=ok, failed=len(failures))

    if failures:
        pd.DataFrame(failures, columns=["kepid", "status"]).to_csv(failed_csv, index=False)
        print(f"{len(failures)} stars failed (see data/failed_stars.csv); re-run to retry them")
    print(f"cached stars: {len(list(LC_DIR.glob('*.npz'))):,}")


if __name__ == "__main__":
    main()
