import argparse
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


# ── colours ──────────────────────────────────────────────────────────────────
CAMERA_COLORS = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B2", "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD",
]


# ═════════════════════════════════════════════════════════════════════════════
#  Collect
# ═════════════════════════════════════════════════════════════════════════════

def collect_summaries(results_root: Path) -> pd.DataFrame:
    rows = []
    for cam_dir in sorted(results_root.iterdir()):
        if not cam_dir.is_dir():
            continue
        summary_csv = cam_dir / "summary.csv"
        if not summary_csv.exists():
            continue

        parts = cam_dir.name.split("_", 1)
        seq_name, cam_name = (parts[0], parts[1]) if len(parts) == 2 else (cam_dir.name, cam_dir.name)

        df = pd.read_csv(summary_csv)
        df.insert(0, "sequence", seq_name)
        df.insert(1, "camera", cam_name)
        df = df.rename(columns={"method": "tracking_method"})

        keep = ["sequence", "camera", "model", "tracking_method", "HOTA", "IDF1"]
        df = df[[c for c in keep if c in df.columns]]
        rows.append(df)

    if not rows:
        return pd.DataFrame(columns=["sequence", "camera", "model", "tracking_method", "HOTA", "IDF1"])
    return pd.concat(rows, ignore_index=True)


# ═════════════════════════════════════════════════════════════════════════════
#  Print helpers
# ═════════════════════════════════════════════════════════════════════════════

def print_mean_per_method_per_sequence(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("  MEAN SCORES PER TRACKING METHOD PER SEQUENCE")
    print("=" * 70)
    for (seq, method), row in df.groupby(["sequence", "tracking_method"])[["HOTA", "IDF1"]].mean().iterrows():
        print(f"  {seq:6s}  |  {method:20s}  |  HOTA={row['HOTA']:.2f}%  IDF1={row['IDF1']:.2f}%")


def print_mean_per_model_per_sequence(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("  MEAN SCORES PER MODEL PER SEQUENCE")
    print("=" * 70)
    for (seq, model), row in df.groupby(["sequence", "model"])[["HOTA", "IDF1"]].mean().iterrows():
        print(f"  {seq:6s}  |  {model:20s}  |  HOTA={row['HOTA']:.2f}%  IDF1={row['IDF1']:.2f}%")


def print_mean_per_model(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("  MEAN SCORES PER MODEL  (across all sequences & cameras)")
    print("=" * 70)
    for model, row in df.groupby("model")[["HOTA", "IDF1"]].mean().iterrows():
        print(f"  {model:20s}  |  HOTA={row['HOTA']:.2f}%  IDF1={row['IDF1']:.2f}%")


def print_best_per_camera(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("  BEST RESULT PER CAMERA  (by HOTA)")
    print("=" * 70)
    for (seq, cam), group in df.groupby(["sequence", "camera"]):
        best = group.loc[group["HOTA"].idxmax()]
        print(
            f"  {seq}/{cam:6s}  |  model={best['model']:15s}  "
            f"method={best['tracking_method']:20s}  |  "
            f"HOTA={best['HOTA']:.2f}%  IDF1={best['IDF1']:.2f}%"
        )


# ═════════════════════════════════════════════════════════════════════════════
#  Plots – YOLO models per sequence (one plot per sequence × tracking_method)
# ═════════════════════════════════════════════════════════════════════════════

def make_yolo_sequence_plots(df: pd.DataFrame, plots_dir: Path) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)

    yolo_df = df[df["model"].str.contains("yolo", case=False, na=False)].copy()
    if yolo_df.empty:
        print("\n[INFO] No YOLO models found in the data – skipping YOLO plots.")
        return

    yolo_models   = sorted(yolo_df["model"].unique())
    sequences     = sorted(yolo_df["sequence"].unique())
    track_methods = sorted(yolo_df["tracking_method"].unique())
    n_models      = len(yolo_models)
    model_colors  = {m: CAMERA_COLORS[i % len(CAMERA_COLORS)] for i, m in enumerate(yolo_models)}

    total_plots = 0

    for seq in sequences:
        for method in track_methods:
            sub = yolo_df[(yolo_df["sequence"] == seq) & (yolo_df["tracking_method"] == method)]
            if sub.empty:
                continue

            cameras    = sorted(sub["camera"].unique())
            n_cams     = len(cameras)
            n_bars     = n_models * 2          # HOTA + IDF1 per model
            bar_width  = 0.70 / n_bars         # all bars share 70% of each camera slot
            group_gap  = 0.30                  # remaining 30% is whitespace between camera groups

            fig, ax = plt.subplots(figsize=(max(7, n_cams * 1.8 + 1.5), 5))
            x_centers = np.arange(n_cams, dtype=float)

            for m_idx, model in enumerate(yolo_models):
                model_sub = sub[sub["model"] == model]
                color     = model_colors[model]

                hota_vals, idf1_vals = [], []
                for cam in cameras:
                    cam_row = model_sub[model_sub["camera"] == cam]
                    hota_vals.append(float(cam_row["HOTA"].iloc[0]) if not cam_row.empty else 0.0)
                    idf1_vals.append(float(cam_row["IDF1"].iloc[0])  if not cam_row.empty else 0.0)

                # Pack bars tightly: HOTA then IDF1 for each model, no gap between them
                slot_hota = m_idx * 2
                slot_idf1 = m_idx * 2 + 1
                total_width = bar_width * n_bars
                offset_hota = -total_width / 2 + (slot_hota + 0.5) * bar_width
                offset_idf1 = -total_width / 2 + (slot_idf1 + 0.5) * bar_width

                bars_hota = ax.bar(
                    x_centers + offset_hota, hota_vals, bar_width * 0.95,
                    label=f"{model} – HOTA",
                    color=color, alpha=0.90, zorder=3,
                )
                bars_idf1 = ax.bar(
                    x_centers + offset_idf1, idf1_vals, bar_width * 0.95,
                    label=f"{model} – IDF1",
                    color=color, alpha=0.45, hatch="//", edgecolor="white", zorder=3,
                )

                for bar in list(bars_hota) + list(bars_idf1):
                    h = bar.get_height()
                    if h > 1:
                        ax.text(
                            bar.get_x() + bar.get_width() / 2, h + 0.4,
                            f"{h:.1f}", ha="center", va="bottom", fontsize=10, rotation=90,
                        )

            ax.set_xticks(x_centers)
            ax.set_xticklabels(cameras, fontsize=9)
            ax.set_xlabel("Camera", fontsize=10)
            ax.set_ylabel("Score (%)", fontsize=10)
            ax.set_ylim(0, 118)
            ax.set_title(
                f"YOLO comparison – {seq}  |  {method}",
                fontsize=11, fontweight="bold",
            )
            ax.legend(fontsize=7, ncol=n_models, loc="upper right",
                      framealpha=0.7, borderpad=0.4, labelspacing=0.3)
            ax.grid(axis="y", alpha=0.35, zorder=0)

            plt.tight_layout(pad=0.8)
            fname = plots_dir / f"yolo_comparison_{seq}_{method}.png"
            plt.savefig(fname, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  → YOLO plot saved: {fname}")
            total_plots += 1

    print(f"\n  {total_plots} YOLO plot(s) saved to {plots_dir}")


# ═════════════════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Aggregate all summary.csv files into a single CSV with stats."
    )
    ap.add_argument("--results-root", required=True,
                    help="Folder produced by run_all_tracking.py (contains S01_c001/, …).")
    ap.add_argument("--output-csv", default="all_results.csv",
                    help="Path for the aggregated output CSV (default: ./all_results.csv).")
    ap.add_argument("--plots-dir", default="yolo_plots",
                    help="Folder where YOLO comparison plots are saved (default: ./yolo_plots).")
    args = ap.parse_args()

    results_root = Path(args.results_root).resolve()
    output_csv   = Path(args.output_csv).resolve()
    plots_dir    = Path(args.plots_dir).resolve()

    df = collect_summaries(results_root)
    if df.empty:
        print(f"[WARN] No summary.csv files found under {results_root}")
        return

    df.to_csv(output_csv, index=False)
    print(f"\nAggregated CSV saved → {output_csv}  ({len(df)} rows)")

    print_mean_per_method_per_sequence(df)
    print_mean_per_model_per_sequence(df)
    print_mean_per_model(df)
    print_best_per_camera(df)

    make_yolo_sequence_plots(df, plots_dir)

    print(f"\n{'=' * 70}")
    print("Done.")


if __name__ == "__main__":
    main()