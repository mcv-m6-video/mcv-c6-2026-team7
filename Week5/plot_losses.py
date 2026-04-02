import json
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# ── Style ──────────────────────────────────────────────────────────────────────
mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "plots")

os.makedirs(OUTPUT_DIR, exist_ok=True)

MODELS = {
   "baseline":            "Baseline",
    #"depthwise":           "Depthwise Conv",
   # "dino":                "DINO",
   # "convnext":            "ConvNeXt",
   # "tcn":                 "TCN",
    "temporal_transformer_v1":"Temporal Transformer",
    "ablation_focal": "Focal Loss",
    "ablation_regularization": "Regularization",
    "ablation_backbone": "Backbone Change",
    "temporal_transformer":"Temporal Transformer V2",
}

# Colour-blind-friendly palette (Okabe–Ito)
COLORS = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#C9F9BA"]


def load(model_key):
    path = os.path.join(RESULTS_DIR, model_key, "loss.json")
    with open(path) as f:
        data = json.load(f)
    epochs = [d["epoch"] + 1 for d in data]   # 1-indexed for plotting
    train  = [d["train"]  for d in data]
    val    = [d["val"]    for d in data]
    return epochs, train, val


# ── Figure 1: all models in one grid (train + val per model) ──────────────────
def plot_grid():
    n = len(MODELS)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows), sharey=False)
    axes = axes.flatten()

    for idx, (key, label) in enumerate(MODELS.items()):
        ax = axes[idx]
        epochs, train, val = load(key)
        color = COLORS[idx]
        ax.plot(epochs, train, color=color, lw=2, label="Train")
        ax.plot(epochs, val,   color=color, lw=2, linestyle="--", label="Val")
        best_epoch = epochs[int(np.argmin(val))]
        best_val   = min(val)
        ax.axvline(best_epoch, color="gray", lw=0.8, linestyle=":")
        ax.scatter([best_epoch], [best_val], color=color, zorder=5, s=40)
        ax.set_title(label, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(loc="upper right")
        ax.set_xlim(1, max(epochs))

    # Hide unused subplots
    for i in range(n, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("Training and Validation Loss per Model", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, "losses_grid.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")



if __name__ == "__main__":
    plot_grid()
