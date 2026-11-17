"""Step 9: run the trained model on Kepler-90 (KIC 11442793), where Kepler-90i was found.

    python -m src.kepler90 [--model cnn.pt]

Kepler-90 is forced into the TEST split, so the model never trained on this star.

Only Kepler-90h is a labeled DR24 TCE, so the ephemerides of all eight planets are listed
here. b-h come from the NASA Exoplanet Archive cumulative KOI table (K00351.01-.07).
Kepler-90i was found by Shallue & Vanderburg's own search and is not in the KOI table; its
parameters are from their public AstroNet example. A box search on this light curve, with
b-h masked, recovers it at P = 14.449 d with a depth of about 80-90 ppm.
"""
import argparse

import pandas as pd
import torch

from .config import FIGURES, KEPLER90_KEPID, MODELS
from .model import build_model
from .plots import grid_of_views
from .views import flatten, load_star, views_for_star

# name, period (d), t0 (BKJD d), duration (h), how it is known
KEPLER90 = pd.DataFrame([
    ("Kepler-90 b", 7.00821787, 137.67164, 3.812, "confirmed"),
    ("Kepler-90 c", 8.71980135, 139.52134, 3.7454, "confirmed"),
    ("Kepler-90 i", 14.44912, 2.2, 0.11267 * 24, "found 2017 by CNN"),
    ("Kepler-90 d", 59.7371443, 158.95974, 8.0006, "confirmed"),
    ("Kepler-90 e", 91.9401253, 134.3, 9.195, "confirmed"),
    ("Kepler-90 f", 124.922516, 129.7295, 11.942, "confirmed"),
    ("Kepler-90 g", 210.6013843, 147.056867, 11.9922, "confirmed"),
    ("Kepler-90 h", 331.5972732, 140.498929, 14.404, "confirmed (DR24 PC)"),
], columns=["name", "tce_period", "tce_time0bk", "tce_duration", "status"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="cnn.pt")
    args = ap.parse_args()

    lc = load_star(KEPLER90_KEPID)
    t, f = flatten(lc, KEPLER90.tce_period, KEPLER90.tce_time0bk, KEPLER90.tce_duration)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(device)
    model.load_state_dict(torch.load(MODELS / args.model, map_location=device))
    model.eval()

    rows, items = [], []
    for s, (g, l) in zip(KEPLER90.itertuples(), views_for_star(t, f, KEPLER90)):
        with torch.no_grad():
            p = torch.sigmoid(model(torch.from_numpy(g)[None].to(device),
                                    torch.from_numpy(l)[None].to(device))).item()
        rows.append({"planet": s.name, "period_days": round(s.tce_period, 3),
                     "status": s.status, "p_planet": round(p, 3)})
        items.append((g, l, f"{s.name}\nP={s.tce_period:.2f} d  p={p:.2f}"))

    print(pd.DataFrame(rows).to_string(index=False))
    grid_of_views(items, FIGURES / "kepler90.png", "Kepler-90 (KIC 11442793): model predictions")


if __name__ == "__main__":
    main()
