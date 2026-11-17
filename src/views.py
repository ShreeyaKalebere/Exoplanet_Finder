"""Steps 4 + 5: flatten each light curve and fold/bin every TCE into a global and a local view.

    python -m src.views

Global view: the whole orbit, 2001 bins.  Local view: +/- 4 transit durations, 201 bins.
Both are normalized so the median is 0 and the deepest point is -1.
Writes data/views.npz (tce_id, global, local).
"""
import os
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import (GLOBAL_BINS, LC_DIR, LOCAL_BINS, LOCAL_HALF_WIDTH_DURATIONS, TCE_CSV,
                     VIEWS_NPZ)

warnings.filterwarnings("ignore")


def load_star(kepid):
    import lightkurve as lk
    d = np.load(LC_DIR / f"{int(kepid)}.npz")
    return lk.LightCurve(time=d["time"], flux=d["flux"])


def flatten(lc, periods, t0s, durations_hr, window_length=401):
    """Remove slow stellar variability. Transits of ALL TCEs on the star are masked first,
    otherwise the trend fit follows the dips and makes them shallower."""
    mask = lc.create_transit_mask(period=list(periods), transit_time=list(t0s),
                                  duration=[2 * d / 24 for d in durations_hr])  # hours -> days
    flat = lc.flatten(window_length=window_length, mask=mask)
    return flat.time.value, np.asarray(flat.flux.value, dtype=np.float64)


def binned_median(phase, flux, lo, hi, n):
    """Median flux in n equal bins over [lo, hi). Empty bins are filled by interpolation."""
    sel = (phase >= lo) & (phase < hi) & np.isfinite(flux)
    p, f = phase[sel], flux[sel]
    if len(p) == 0:
        return None
    idx = np.clip(((p - lo) / (hi - lo) * n).astype(int), 0, n - 1)
    order = np.argsort(idx, kind="stable")
    idx, f = idx[order], f[order]
    bounds = np.searchsorted(idx, np.arange(n + 1))
    out = np.full(n, np.nan)
    for i in range(n):
        if bounds[i] < bounds[i + 1]:
            out[i] = np.median(f[bounds[i]:bounds[i + 1]])
    ok = np.isfinite(out)
    if ok.sum() < 2:
        return None
    out[~ok] = np.interp(np.flatnonzero(~ok), np.flatnonzero(ok), out[ok])
    return out


def normalize(v):
    v = v - np.median(v)
    depth = -v.min()
    return v / depth if depth > 0 else v


def make_views(time, flux, period, t0, duration_hr):
    """Fold on the period so every transit lands at phase 0, then bin into the two views."""
    dur = duration_hr / 24  # TCE table: duration in hours, period and t0 in days
    phase = (time - t0 + period / 2) % period - period / 2
    glob = binned_median(phase, flux, -period / 2, period / 2, GLOBAL_BINS)
    half = min(LOCAL_HALF_WIDTH_DURATIONS * dur, period / 2)
    loc = binned_median(phase, flux, -half, half, LOCAL_BINS)
    if glob is None or loc is None:
        return None
    return normalize(glob).astype(np.float32), normalize(loc).astype(np.float32)


def in_transit(time, period, t0, duration_hr):
    """True within +/- one transit duration of each transit centre."""
    dur = duration_hr / 24
    return np.abs((time - t0 + period / 2) % period - period / 2) < dur


def views_for_star(time, flux, signals):
    """Views for every signal on one star. For each signal, the points inside the OTHER
    signals' transits are dropped first; otherwise in multi-planet systems the other
    planets' dips land at random phases and can bury a shallow transit."""
    masks = [in_transit(time, s.tce_period, s.tce_time0bk, s.tce_duration)
             for s in signals.itertuples()]
    out = []
    for i, s in enumerate(signals.itertuples()):
        others = np.zeros_like(time, dtype=bool)
        for j, m in enumerate(masks):
            if j != i:
                others |= m
        out.append(make_views(time[~others], flux[~others], s.tce_period, s.tce_time0bk,
                              s.tce_duration))
    return out


def process_star(args):
    kepid, tces = args
    try:
        lc = load_star(kepid)
        t, f = flatten(lc, tces.tce_period, tces.tce_time0bk, tces.tce_duration)
    except Exception as e:
        return [], [(kepid, f"{type(e).__name__}: {e}")]
    out, errors = [], []
    for row, v in zip(tces.itertuples(), views_for_star(t, f, tces)):
        if v is None:
            errors.append((row.tce_id, "empty view"))
        else:
            out.append((row.tce_id, *v))
    return out, errors


def main():
    tce = pd.read_csv(TCE_CSV)
    cached = {int(p.stem) for p in LC_DIR.glob("*.npz")}
    tce = tce[tce.kepid.isin(cached)]
    jobs = list(tce.groupby("kepid"))
    print(f"building views for {len(tce):,} TCEs on {len(jobs):,} cached stars")

    ids, globs, locs, errors = [], [], [], []
    with ProcessPoolExecutor(max(1, (os.cpu_count() or 2) - 1)) as pool:
        for out, errs in tqdm(pool.map(process_star, jobs, chunksize=8), total=len(jobs), unit="star"):
            errors += errs
            for tce_id, g, l in out:
                ids.append(tce_id)
                globs.append(g)
                locs.append(l)

    np.savez_compressed(VIEWS_NPZ, tce_id=np.array(ids), global_view=np.stack(globs),
                        local_view=np.stack(locs))
    print(f"saved {len(ids):,} views -> {VIEWS_NPZ}  ({len(errors)} skipped)")
    for e in errors[:10]:
        print("  skipped:", e)


if __name__ == "__main__":
    main()
