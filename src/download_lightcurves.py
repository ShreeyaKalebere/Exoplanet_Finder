"""Step 3: download each star's Kepler long-cadence light curve from MAST and cache it.

    python -m src.download_lightcurves --max-stars 300      # quick start
    python -m src.download_lightcurves                      # every star (~hours, resumable)

Each star is saved once as data/lightcurves/<kepid>.npz (time in BKJD days, normalized
PDCSAP flux). The raw FITS files are deleted after stitching to save disk space.
Re-running skips stars that are already cached, so you can stop and resume any time.
"""
import argparse
import shutil
import tempfile
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import DATA, KEPLER90_KEPID, LC_DIR, SEED, TCE_CSV

warnings.filterwarnings("ignore")  # lightkurve/astropy are noisy about units and quality flags


def lc_path(kepid):
    return LC_DIR / f"{int(kepid)}.npz"


def download_star(kepid):
    import lightkurve as lk

    if lc_path(kepid).exists():
        return kepid, "cached"
    tmp = tempfile.mkdtemp(prefix=f"kic{kepid}_")
    try:
        search = lk.search_lightcurve(f"KIC {kepid}", mission="Kepler", author="Kepler",
                                      cadence="long")
        if len(search) == 0:
            return kepid, "no data"
        lcs = search.download_all(download_dir=tmp, flux_column="pdcsap_flux")
        lc = lcs.stitch().remove_nans()          # stitch normalizes each quarter to median 1
        np.savez_compressed(lc_path(kepid), time=lc.time.value.astype(np.float64),
                            flux=np.asarray(lc.flux.value, dtype=np.float32))
        return kepid, "ok"
    except Exception as e:  # network hiccups, corrupt files...
        return kepid, f"error: {type(e).__name__}: {e}"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


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
          f"{len(todo):,} to download")

    failures = []
    with ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(download_star, s) for s in todo]
        for f in tqdm(as_completed(futures), total=len(futures), unit="star"):
            kepid, status = f.result()
            if status not in ("ok", "cached"):
                failures.append((kepid, status))

    if failures:
        pd.DataFrame(failures, columns=["kepid", "status"]).to_csv(DATA / "failed_stars.csv", index=False)
        print(f"{len(failures)} stars failed (see data/failed_stars.csv); re-run to retry them")
    print(f"cached stars: {len(list(LC_DIR.glob('*.npz'))):,}")


if __name__ == "__main__":
    main()
