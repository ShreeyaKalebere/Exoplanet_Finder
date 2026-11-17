"""Step 9: run the trained model on Kepler-90 (KIC 11442793), where Kepler-90i was found.

    python -m src.kepler90

Kepler-90 is forced into the TEST split, so the model never trained on this star.
Kepler-90i was NOT one of the DR24 TCEs: Shallue & Vanderburg found it with their own
search. Its parameters below are the ones from their public AstroNet example
(period 14.44912 d, t0 2.2 BKJD, duration 0.11267 d).
"""
import pandas as pd
import torch

from .config import FIGURES, KEPLER90_KEPID, MODELS, TCE_CSV
from .model import build_model
from .plots import grid_of_views
from .views import flatten, load_star, make_views

KEPLER_90I = {"tce_id": "Kepler-90i", "tce_period": 14.44912, "tce_time0bk": 2.2,
              "tce_duration": 0.11267 * 24, "av_training_set": "new (2017)"}


def main():
    tce = pd.read_csv(TCE_CSV)
    signals = pd.concat([tce[tce.kepid == KEPLER90_KEPID], pd.DataFrame([KEPLER_90I])],
                        ignore_index=True)
    signals = signals.sort_values("tce_period").reset_index(drop=True)

    lc = load_star(KEPLER90_KEPID)
    t, f = flatten(lc, signals.tce_period, signals.tce_time0bk, signals.tce_duration)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(device)
    model.load_state_dict(torch.load(MODELS / "cnn.pt", map_location=device))
    model.eval()

    rows, items = [], []
    for s in signals.itertuples():
        g, l = make_views(t, f, s.tce_period, s.tce_time0bk, s.tce_duration)
        with torch.no_grad():
            p = torch.sigmoid(model(torch.from_numpy(g)[None].to(device),
                                    torch.from_numpy(l)[None].to(device))).item()
        rows.append({"signal": s.tce_id, "period_days": round(s.tce_period, 3),
                     "label": s.av_training_set, "p_planet": round(p, 3)})
        items.append((g, l, f"{s.tce_id}  ({s.av_training_set})\nP={s.tce_period:.2f} d  p={p:.2f}"))

    print(pd.DataFrame(rows).to_string(index=False))
    grid_of_views(items, FIGURES / "kepler90.png", "Kepler-90 (KIC 11442793): model predictions")


if __name__ == "__main__":
    main()
