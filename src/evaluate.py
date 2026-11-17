"""Step 8 + 9: evaluate on the TEST stars (never seen in training) and make the figures.

    python -m src.evaluate

Compares the CNN with the period/duration/depth baseline on exactly the same test TCEs,
logs both to results/metrics.csv, and saves:
  results/test_predictions.csv
  results/figures/pr_curve.png          precision-recall, CNN vs baseline
  results/figures/planet_vs_eb.png      confident planet next to a confident eclipsing binary
  results/figures/false_positives.png   non-planets the CNN was most sure were planets
  results/figures/missed_planets.png    planet candidates the CNN missed
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.metrics import precision_recall_curve
from torch.utils.data import DataLoader

from .baseline import fit_baseline, predict_baseline
from .config import FIGURES, MODELS, RESULTS, TCE_CSV
from .data import ViewDataset, load_dataset
from .metrics import classification_metrics, log_run
from .model import build_model
from .plots import grid_of_views
from .train import predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="cnn.pt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    table, glob, loc = load_dataset()
    te = (table.split == "test").values
    test = table[te].copy()

    model = build_model(device)
    model.load_state_dict(torch.load(MODELS / args.model, map_location=device))
    test["p_cnn"] = predict(model, DataLoader(ViewDataset(glob[te], loc[te], test.planet), batch_size=256), device)
    test["p_baseline"] = predict_baseline(fit_baseline(pd.read_csv(TCE_CSV)), test)

    log_run("cnn_two_view", "test", classification_metrics(test.planet, test.p_cnn))
    log_run("baseline_gbm", "test", classification_metrics(test.planet, test.p_baseline),
            notes="same test TCEs as the CNN")
    test.drop(columns="row").to_csv(RESULTS / "test_predictions.csv", index=False)

    # precision-recall curves
    fig, ax = plt.subplots(figsize=(5, 4))
    for col, name in (("p_cnn", "two-view CNN"), ("p_baseline", "period/duration/depth GBM")):
        p, r, _ = precision_recall_curve(test.planet, test[col])
        ax.plot(r, p, label=name)
    ax.set_xlabel("recall (planets found)")
    ax.set_ylabel("precision")
    ax.set_title("Test stars: planet class")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "pr_curve.png", dpi=150)
    plt.close(fig)

    gv, lv = glob[te], loc[te]
    pos = {tid: i for i, tid in enumerate(test.tce_id)}

    def items(df, label):
        return [(gv[pos[r.tce_id]], lv[pos[r.tce_id]],
                 f"KIC {r.kepid} ({r.av_training_set})\nP={r.tce_period:.2f} d  p={r.p_cnn:.2f}{label}")
                for r in df.itertuples()]

    planet = test[test.planet == 1].nlargest(1, "p_cnn")
    eb = test[test.av_training_set == "AFP"].nsmallest(1, "p_cnn")
    grid_of_views(items(planet, "") + items(eb, ""), FIGURES / "planet_vs_eb.png",
                  "Real planet candidate vs. astrophysical false positive")
    grid_of_views(items(test[test.planet == 0].nlargest(5, "p_cnn"), ""), FIGURES / "false_positives.png",
                  "Non-planets the CNN rated most planet-like (some may be label errors)")
    grid_of_views(items(test[test.planet == 1].nsmallest(5, "p_cnn"), ""), FIGURES / "missed_planets.png",
                  "Planet candidates the CNN missed")


if __name__ == "__main__":
    main()
