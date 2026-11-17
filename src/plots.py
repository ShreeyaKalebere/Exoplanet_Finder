"""Plot helpers for folded views."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_views(ax_global, ax_local, g, l, title):
    ax_global.plot(np.linspace(-0.5, 0.5, len(g)), g, ".", ms=1.5, color="#1f4e79")
    ax_global.set_title(title, fontsize=9)
    ax_global.set_xlabel("phase (orbits)", fontsize=8)
    ax_local.plot(np.linspace(-4, 4, len(l)), l, ".-", ms=2, lw=0.5, color="#c0392b")
    ax_local.set_xlabel("time from transit (durations)", fontsize=8)
    for ax in (ax_global, ax_local):
        ax.tick_params(labelsize=7)


def grid_of_views(items, path, suptitle):
    """items: list of (global_view, local_view, title). One column per item, global on top."""
    n = len(items)
    if n == 0:
        return
    fig, axes = plt.subplots(2, n, figsize=(3.2 * n, 5), squeeze=False)
    for i, (g, l, title) in enumerate(items):
        plot_views(axes[0, i], axes[1, i], g, l, title)
    fig.suptitle(suptitle)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"figure -> {path}")
