"""Metric helpers. Every run appends one row to results/metrics.csv -> your README results table."""
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score

from .config import METRICS_CSV


def classification_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "n": len(y_true),
        "n_planets": int(y_true.sum()),
        "auc": roc_auc_score(y_true, y_prob),
        "avg_precision": average_precision_score(y_true, y_prob),
        "precision@0.5": precision_score(y_true, y_pred, zero_division=0),
        "recall@0.5": recall_score(y_true, y_pred, zero_division=0),
        # accuracy shown only to make the point that it misleads (~75% for "never a planet")
        "accuracy": float((y_pred == y_true).mean()),
    }


def log_run(model_name, split, metrics, notes=""):
    row = {"time": datetime.now().strftime("%Y-%m-%d %H:%M"), "model": model_name,
           "split": split, **{k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()},
           "notes": notes}
    df = pd.DataFrame([row])
    df.to_csv(METRICS_CSV, mode="a", header=not METRICS_CSV.exists(), index=False)
    print(f"[{model_name} | {split}] " + ", ".join(f"{k}={v}" for k, v in row.items()
                                                if k not in ("time", "model", "split", "notes")))
