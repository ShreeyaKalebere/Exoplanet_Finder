"""Step 7: two-view 1D CNN (a smaller version of Shallue & Vanderburg's AstroNet)."""
import torch
from torch import nn


def conv_branch(channels, pool_kernel):
    layers, in_ch = [], 1
    for ch in channels:
        layers += [nn.Conv1d(in_ch, ch, 5, padding=2), nn.ReLU(),
                   nn.Conv1d(ch, ch, 5, padding=2), nn.ReLU(),
                   nn.MaxPool1d(pool_kernel, stride=2)]
        in_ch = ch
    return nn.Sequential(*layers, nn.Flatten())


class TwoViewCNN(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        self.global_branch = conv_branch([16, 32, 64, 128], pool_kernel=5)   # 2001 bins
        self.local_branch = conv_branch([16, 32], pool_kernel=7)             # 201 bins
        self.head = nn.Sequential(
            nn.LazyLinear(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self, global_view, local_view):
        g = self.global_branch(global_view.unsqueeze(1))
        l = self.local_branch(local_view.unsqueeze(1))
        return self.head(torch.cat([g, l], dim=1)).squeeze(1)   # logit of P(planet)


def build_model(device):
    model = TwoViewCNN().to(device)
    with torch.no_grad():  # materialize the lazy layer
        model(torch.zeros(2, 2001, device=device), torch.zeros(2, 201, device=device))
    return model
