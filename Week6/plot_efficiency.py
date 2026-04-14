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

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Model data ─────────────────────────────────────────────────────────────────
# Each entry: label, GMACs, mAP10, num_params
MODELS = [
    ("Baseline",          18.11,     4.83,   2_798_793 ),   # no complete log found
    ("Baseline Rny004",            36.22,  14.53, 3_908_877),
    ("Baseline Rny008",            71.45,  14.70, 5_504_165),
    ("Temporal Transformer + Rny004", 36.22,  32.25, 8_591_797),
    ("ConvNextV2 Pico", 47.37,  5.05, 3_700_000),
    ("ConvNextV2 Atto", 47.37,  5.05, 3_700_000),
    ("Temporal Transformer + Rny002", 18.18,  12.64, 6_079_513),
    ("Residual BiGRU", 36.31,  29.64, 5_657_437),
    ("Residual BiGRU + Focal Loss", 36.31,  32.87, 5_657_437),
    ("Residual BiGRU + TGLS", 36.31,  33.79, 5_657_437),
    ("Temporal Transformer + Rny004 + TGLS", 36.22,  27.9, 8_591_797),
    ("1D Convs", 36.22,  12.94, 3_908_877),
]

# Colour-blind-friendly palette (Okabe–Ito)
COLORS = [
    "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7",
    "#4B0082", "#FF6F61", "#2ECC71", "#A0522D", "#00CED1",
]

# ── Bubble scaling ─────────────────────────────────────────────────────────────
# Map num_params → marker area (s). We scale relative to the largest model so
# the biggest bubble has a fixed reference size and smaller ones scale down.
MAX_AREA   = 2000   # area (s units) for the largest model
params_arr = np.array([m[3] for m in MODELS], dtype=float)
areas      = (params_arr / params_arr.max()) * MAX_AREA



def plot_bubble():
    fig, ax = plt.subplots(figsize=(10, 5))

    xs     = [m[1] for m in MODELS]
    ys     = [m[2] for m in MODELS]

    for i, (label, gmacs, map10, params) in enumerate(MODELS):
        color = COLORS[i % len(COLORS)]
        ax.scatter(
            gmacs, map10,
            s=areas[i],
            color=color,
            alpha=0.85,
            edgecolors="white",
            linewidths=1.2,
            zorder=3,
            label=label,
        )

    all_x, all_y = xs, ys
    x_pad = (max(all_x) - min(all_x)) * 0.05 + 5
    y_pad = (max(all_y) - min(all_y)) * 0.10 + 2
    ax.set_xlim(min(all_x) - x_pad, max(all_x) + x_pad)
    ax.set_ylim(min(all_y) - y_pad, max(all_y) + y_pad)

    ax.set_xlabel("GMACs")
    ax.set_ylabel("mAP10")
    ax.set_title("Model Efficiency: GMACs vs mAP10\n(bubble size ∝ # parameters)",
                 fontweight="bold")

    handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=COLORS[i % len(COLORS)],
                   markersize=8, label=label)
        for i, (label, *_) in enumerate(MODELS)
    ]
    ax.legend(handles=handles, loc="center left",
              bbox_to_anchor=(1.02, 0.5),
              framealpha=0.8, borderaxespad=0)

    fig.tight_layout()
    fig.subplots_adjust(right=0.72)
    out = os.path.join(OUTPUT_DIR, "efficiency_bubble.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    plot_bubble()
