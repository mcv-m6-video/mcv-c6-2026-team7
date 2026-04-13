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
    # ("Baseline (rny002)",          ??,     ??,    ?),   # no complete log found
    ("Baseline Rny004",            36.22,  14.53, 3_908_877),
    ("Baseline Rny008",            71.45,  14.70, 5_504_165),
    ("Temporal Transformer Rny004",36.22,  32.25, 8_591_797),
]

# Colour-blind-friendly palette (Okabe–Ito)
COLORS = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#C9F9BA"]

# ── Bubble scaling ─────────────────────────────────────────────────────────────
# Map num_params → marker area (s). We scale relative to the largest model so
# the biggest bubble has a fixed reference size and smaller ones scale down.
MAX_AREA   = 2000   # area (s units) for the largest model
params_arr = np.array([m[3] for m in MODELS], dtype=float)
areas      = (params_arr / params_arr.max()) * MAX_AREA


def _repulsion_offset(i, xy_disp, area, pad_pts=10):
    """Return (dx, dy) in typographic points for label i, pushed away from all
    other points using a sum-of-unit-vectors repulsion in display space."""
    p = xy_disp[i]
    others = np.delete(xy_disp, i, axis=0)
    if len(others):
        diffs = p - others                               # vectors away from others
        dists = np.linalg.norm(diffs, axis=1, keepdims=True).clip(1e-6)
        direction = (diffs / dists).sum(axis=0)          # sum of unit vectors
    else:
        direction = np.array([0.0, 1.0])
    norm = np.linalg.norm(direction)
    direction /= max(norm, 1e-6)
    total_pts = np.sqrt(area / np.pi) + pad_pts          # bubble radius + padding
    return direction * total_pts                         # (dx, dy) in pts


def plot_bubble():
    fig, ax = plt.subplots(figsize=(9, 6))

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
        )
        ax.text(
            gmacs, map10,
            f"{params/1e6:.2f}M",
            ha="center", va="center",
            fontsize=8, fontweight="bold",
            color="white", zorder=4,
        )

    # Compute repulsion directions after axis limits are set
    all_x, all_y = xs, ys
    x_pad = (max(all_x) - min(all_x)) * 0.35 + 5
    y_pad = (max(all_y) - min(all_y)) * 0.25 + 2
    ax.set_xlim(min(all_x) - x_pad, max(all_x) + x_pad)
    ax.set_ylim(min(all_y) - y_pad, max(all_y) + y_pad)

    fig.canvas.draw()                                    # needed for transData
    xy_disp = ax.transData.transform(np.column_stack([xs, ys]))

    for i, (label, gmacs, map10, _) in enumerate(MODELS):
        dx, dy = _repulsion_offset(i, xy_disp, areas[i])
        ax.annotate(
            label,
            xy=(gmacs, map10),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="center", va="center",
            fontsize=9,
        )

    ax.set_xlabel("GMACs")
    ax.set_ylabel("mAP10")
    ax.set_title("Model Efficiency: GMACs vs mAP10\n(bubble size ∝ # parameters)",
                 fontweight="bold")

    ax.legend().set_visible(False)

    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, "efficiency_bubble.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    plot_bubble()
