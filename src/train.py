"""Step 7: train the two-view CNN with binary cross-entropy, early stopping on validation AUC.

    python -m src.train
    python -m src.train --epochs 60 --lr 3e-4
"""
import argparse
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import MODELS, SEED
from .data import ViewDataset, load_dataset
from .metrics import classification_metrics, log_run
from .model import build_model


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    out = []
    for g, l, _ in loader:
        out.append(torch.sigmoid(model(g.to(device), l.to(device))).cpu())
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--out", default="cnn.pt")
    args = ap.parse_args()

    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    table, glob, loc = load_dataset()
    tr, va = (table.split == "train").values, (table.split == "val").values
    print(f"device={device}  train={tr.sum():,}  val={va.sum():,}  "
          f"(planet fraction train={table.planet[tr].mean():.2f})")

    train_dl = DataLoader(ViewDataset(glob[tr], loc[tr], table.planet[tr], augment=True),
                          batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(ViewDataset(glob[va], loc[va], table.planet[va]), batch_size=256)

    model = build_model(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    best_auc, bad_epochs = -1, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for g, l, y in train_dl:
            g, l, y = g.to(device), l.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(g, l), y)
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
        val_auc = classification_metrics(table.planet[va], predict(model, val_dl, device))["auc"]
        print(f"epoch {epoch:3d}  train_loss={total / tr.sum():.4f}  val_auc={val_auc:.4f}")
        if val_auc > best_auc:
            best_auc, bad_epochs = val_auc, 0
            torch.save(model.state_dict(), MODELS / args.out)
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"early stop: no val improvement for {args.patience} epochs")
                break

    model.load_state_dict(torch.load(MODELS / args.out, map_location=device))
    log_run("cnn_two_view", "val", classification_metrics(table.planet[va], predict(model, val_dl, device)),
            notes=f"lr={args.lr} bs={args.batch_size} train_tces={tr.sum()}")
    print(f"best model -> {MODELS / args.out}")


if __name__ == "__main__":
    main()
