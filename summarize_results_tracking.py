import os
import argparse
import pandas as pd
from pathlib import Path


def collect_summaries(results_root: Path) -> pd.DataFrame:
    rows = []
    for cam_dir in sorted(results_root.iterdir()):
        if not cam_dir.is_dir():
            continue
        summary_csv = cam_dir / "summary.csv"
        if not summary_csv.exists():
            continue

        parts = cam_dir.name.split("_", 1)
        if len(parts) == 2:
            seq_name, cam_name = parts[0], parts[1]
        else:
            seq_name, cam_name = cam_dir.name, cam_dir.name

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


def print_mean_per_method_per_sequence(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("  MEAN SCORES PER TRACKING METHOD PER SEQUENCE")
    print("=" * 70)
    grouped = df.groupby(["sequence", "tracking_method"])[["HOTA", "IDF1"]].mean()
    for (seq, method), row in grouped.iterrows():
        print(f"  {seq:6s}  |  {method:20s}  |  HOTA={row['HOTA']:.2f}%  IDF1={row['IDF1']:.2f}%")


def print_mean_per_model_per_sequence(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("  MEAN SCORES PER MODEL PER SEQUENCE")
    print("=" * 70)
    grouped = df.groupby(["sequence", "model"])[["HOTA", "IDF1"]].mean()
    for (seq, model), row in grouped.iterrows():
        print(f"  {seq:6s}  |  {model:20s}  |  HOTA={row['HOTA']:.2f}%  IDF1={row['IDF1']:.2f}%")


def print_mean_per_model(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("  MEAN SCORES PER MODEL  (across all sequences & cameras)")
    print("=" * 70)
    grouped = df.groupby("model")[["HOTA", "IDF1"]].mean()
    for model, row in grouped.iterrows():
        print(f"  {model:20s}  |  HOTA={row['HOTA']:.2f}%  IDF1={row['IDF1']:.2f}%")


def print_best_per_camera(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("  BEST RESULT PER CAMERA  (by HOTA)")
    print("=" * 70)
    for (seq, cam), group in df.groupby(["sequence", "camera"]):
        best_idx = group["HOTA"].idxmax()
        best = group.loc[best_idx]
        print(
            f"  {seq}/{cam:6s}  |  model={best['model']:15s}  "
            f"method={best['tracking_method']:20s}  |  "
            f"HOTA={best['HOTA']:.2f}%  IDF1={best['IDF1']:.2f}%"
        )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Aggregate all summary.csv files into a single CSV with stats."
    )
    ap.add_argument(
        "--results-root",
        required=True,
        help="Folder produced by run_all_tracking.py (contains S01_c001/, S03_c010/, …).",
    )
    ap.add_argument(
        "--output-csv",
        default="all_results.csv",
        help="Path for the aggregated output CSV (default: ./all_results.csv).",
    )
    args = ap.parse_args()

    results_root = Path(args.results_root).resolve()
    output_csv   = Path(args.output_csv).resolve()

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

    print(f"\n{'=' * 70}")
    print("Done.")


if __name__ == "__main__":
    main()