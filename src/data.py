"""Join the saved views with labels and splits."""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import TCE_CSV, VIEWS_NPZ


def load_dataset():
    """Returns (table, global_views, local_views) aligned row by row."""
    v = np.load(VIEWS_NPZ)
    order = pd.DataFrame({"tce_id": v["tce_id"], "row": np.arange(len(v["tce_id"]))})
    table = pd.read_csv(TCE_CSV).merge(order, on="tce_id").reset_index(drop=True)
    return table, v["global_view"][table.row.values], v["local_view"][table.row.values]


class ViewDataset(Dataset):
    def __init__(self, glob, loc, labels, augment=False):
        self.g = torch.from_numpy(glob)
        self.l = torch.from_numpy(loc)
        self.y = torch.from_numpy(np.asarray(labels, dtype=np.float32))
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        g, l = self.g[i], self.l[i]
        if self.augment and torch.rand(1).item() < 0.5:   # time reversal: a transit looks the same
            g, l = g.flip(0), l.flip(0)
        return g, l, self.y[i]
