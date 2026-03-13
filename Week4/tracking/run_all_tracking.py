"""
run_all_tracking.py
====================
Iterates over every sequence/camera in an AI-City challenge folder,
runs the tracking methods [overlap, kalman, overlap_flow] PLUS the
detections that already ship with the dataset, evaluates each with
TrackEval (HOTA + IDF1) and produces a per-camera comparative bar plot.

Usage
-----
python run_all_tracking.py \
    --data-root  /path/to/AICity_data/train \
    --repo-root  /path/to/project/repo \
    --gt-xml     data/ai_challenge_s03_c010-full_annotation.xml   # optional override

Folder structure assumed inside --data-root:
    <seq>/          e.g.  S01/  S03/  S04/
      <cam>/        e.g.  c001/ c002/ c010/
        vdo.avi
        det/
          det.txt           # dataset detections  (MOT format: frame,id,x,y,w,h,conf,-1,-1,-1)
        gt/
          gt.txt            # ground truth        (MOT format, used when no XML annotation given)

Our model's detections are expected at:
    <repo_root>/Week3/detections/<seq>/<cam>/detections.txt
  e.g.  Week3/detections/S03/c010/detections.txt

Ground truth is derived from an XML annotation file (same logic as data_processor.py /
AICityFrames) when --gt-xml is provided; otherwise falls back to gt/gt.txt inside the
camera folder.
"""

import os
import sys
import argparse
import shutil
import subprocess
import re
import ast
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── compatibility shims ──────────────────────────────────────────────────────
if not hasattr(np, "float"):  np.float  = float
if not hasattr(np, "int"):    np.int    = int
if not hasattr(np, "bool"):   np.bool   = bool

# ── colours used in bar plots ────────────────────────────────────────────────
METHOD_COLORS = {
    "overlap":       "#4C72B0",
    "kalman":        "#DD8452",
    "overlap_flow":  "#55A868",
    "dataset_det":   "#C44E52",   # built-in detections from the dataset
}
FALLBACK_COLORS = ["#8172B2", "#937860", "#DA8BC3", "#8C8C8C", "#CCB974"]


# ═════════════════════════════════════════════════════════════════════════════
#  Helper – discover cameras
# ═════════════════════════════════════════════════════════════════════════════

def discover_cameras(data_root: Path) -> list[tuple[Path, str, str]]:
    """
    Walk data_root looking for directories that contain vdo.avi.
    Returns list of (cam_path, seq_name, cam_name).
    """
    cameras = []
    for seq_dir in sorted(data_root.iterdir()):
        if not seq_dir.is_dir():
            continue
        for cam_dir in sorted(seq_dir.iterdir()):
            if not cam_dir.is_dir():
                continue
            if (cam_dir / "vdo.avi").exists():
                cameras.append((cam_dir, seq_dir.name, cam_dir.name))
    return cameras


# ═════════════════════════════════════════════════════════════════════════════
#  Helper – count frames in a video
# ═════════════════════════════════════════════════════════════════════════════


def count_frames(video_path: Path) -> int:
    """Fallback: ask OpenCV for the frame count."""
    try:
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return max(n, 1)
    except Exception:
        return 2141   # last-resort fallback


def seq_length_from_file(txt_path: Path) -> int | None:
    """Read the max frame number from any MOT-format txt file."""
    try:
        max_frame = 0
        with open(txt_path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                if parts:
                    max_frame = max(max_frame, int(float(parts[0])))
        return max_frame if max_frame > 0 else None
    except Exception:
        return None


# ← replace the old seq_length_from_gt + get_seq_length with these two:

def seq_length_from_gt(gt_txt: Path) -> int | None:
    return seq_length_from_file(gt_txt)


def get_seq_length(gt_txt: Path, video_path: Path, tracker_txt: Path | None = None) -> int:
    """
    Take the MAX of:
      1. max frame in gt.txt
      2. max frame in tracker output (tracks.txt), if available
      3. OpenCV frame count
    This prevents seqinfo.ini from under-counting when annotations
    end before the video does, causing TrackEval to reject valid frames.
    """
    candidates = []

    from_gt = seq_length_from_file(gt_txt)
    if from_gt:
        candidates.append(from_gt)

    if tracker_txt and tracker_txt.exists():
        from_tracker = seq_length_from_file(tracker_txt)
        if from_tracker:
            candidates.append(from_tracker)

    from_cv = count_frames(video_path)
    candidates.append(from_cv)

    return max(candidates)


# ═════════════════════════════════════════════════════════════════════════════
#  Helper – convert dataset det.txt → our internal CSV format
# ═════════════════════════════════════════════════════════════════════════════

def dataset_det_to_internal_csv(det_txt: Path, out_csv: Path) -> None:
    """
    Dataset det.txt is already MOT-style:
        frame, id(-1), x, y, w, h, conf, -1, -1, -1
    We convert it to our internal detections.csv format:
        frame_id, x1, y1, x2, y2, confidence
    (track_id is irrelevant at this stage – the tracker will assign IDs)
    """
    rows = []
    with open(det_txt) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 7:
                continue
            frame = int(float(parts[0]))
            x  = float(parts[2])
            y  = float(parts[3])
            w  = float(parts[4])
            h  = float(parts[5])
            conf = float(parts[6])
            rows.append({
                "frame_id": frame,
                "x1": x,
                "y1": y,
                "x2": x + w,
                "y2": y + h,
                "confidence": conf,
            })
    pd.DataFrame(rows).to_csv(out_csv, index=False)


# ═════════════════════════════════════════════════════════════════════════════
#  Helper – run a tracking method via main.py
# ═════════════════════════════════════════════════════════════════════════════

def run_tracking_method(
    repo_root: Path,
    method: str,
    det_csv: Path,
    video_path: Path,
    output_dir: Path,
    extra_args: list[str] | None = None,
) -> Path | None:
    """
    Call Week3/tracking/main.py (from the repository) for one method.
    Returns the output directory or None on failure.
    """
    run_tracking = repo_root / "main.py"
    if not run_tracking.exists():
        # Try one level up – some repos have it at tracking/main.py
        run_tracking = repo_root / "tracking" / "main.py"
    if not run_tracking.exists():
        print(f"  [WARN] Cannot find main.py under {repo_root}")
        return None

    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(run_tracking),
        "--method",      method,
        "--detections",  str(det_csv),
        "--video",       str(video_path),
        "--show_IDs_video", "False",
        "--show_comp_video", "False",
    ]
    if extra_args:
        cmd += extra_args

    # Override the output destination: run_tracking always creates its own
    # timestamped folder; we call it and then copy tracks.txt out.
    print(f"    Running: {method} …")
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(repo_root),
        )
        if result.returncode != 0:
            print(f"    [ERROR] {method} failed:\n{result.stderr[-800:]}")
            return None

        # Locate the freshest output folder the script produced
        tracking_outputs = repo_root  / "outputs"
        if not tracking_outputs.exists():
            tracking_outputs = repo_root / "tracking" / "outputs"

        candidates = sorted(
            [d for d in tracking_outputs.iterdir() if d.is_dir() and d.name.startswith(method)],
            key=lambda d: d.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            print(f"    [WARN] No output folder found for {method}")
            return None

        src = candidates[0]
        # Copy tracks.txt into our organised output_dir
        tracks_src = src / "tracks.txt"
        if not tracks_src.exists():
            print(f"    [WARN] tracks.txt not found in {src}")
            return None

        output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(tracks_src, output_dir / "tracks.txt")
        return output_dir

    except Exception as exc:
        print(f"    [ERROR] Exception running {method}: {exc}")
        return None


# ═════════════════════════════════════════════════════════════════════════════
#  Helper – run TrackEval evaluation
# ═════════════════════════════════════════════════════════════════════════════

def run_evaluation(
    repo_root: Path,
    tracker_results_dir: Path,
    tracker_name: str,
    gt_txt: Path,
    seq_name: str,
    seq_length: int,
    benchmark_name: str = "AICity",
    split: str = "train",
) -> dict | None:
    """
    Calls eval_tracking.py (from the repository).
    Returns dict with keys 'HOTA' and 'IDF1', or None on failure.
    """
    evaluate_script = repo_root /  "eval_tracking.py"
    if not evaluate_script.exists():
        evaluate_script = repo_root / "tracking" / "eval_tracking.py"
    if not evaluate_script.exists():
        print(f"  [WARN] Cannot find eval_tracking.py under {repo_root}")
        return None

    # The evaluation script needs the GT in the expected place; we pre-copy it.
    gt_mot_base = repo_root / "data" / "gt_mot_format"
    bm_split_dir = gt_mot_base / f"{benchmark_name}-{split}"
    gt_seq_dir   = bm_split_dir / seq_name / "gt"
    gt_seq_dir.mkdir(parents=True, exist_ok=True)
    dest_gt = gt_seq_dir / "gt.txt"
    if not dest_gt.exists():
        shutil.copy(gt_txt, dest_gt)

    # seqinfo.ini — always (re)write so seq_length is never stale
    seqinfo = bm_split_dir / seq_name / "seqinfo.ini"
    seqinfo.parent.mkdir(parents=True, exist_ok=True)
    seqinfo.write_text(
        f"[Sequence]\nname={seq_name}\nimDir=img1\nframeRate=10\n"
        f"seqLength={seq_length}\nimWidth=1920\nimHeight=1080\nimExt=.jpg\n"
    )

    cmd = [
        sys.executable, str(evaluate_script),
        "--tracker-results", str(tracker_results_dir),
        "--tracker-name",    tracker_name,
        "--seq-name",        seq_name,
        "--seq-length",      str(seq_length),
        "--benchmark-name",  benchmark_name,
        "--split",           split,
    ]

    print(f"    Evaluating {tracker_name} …")
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(repo_root),
        )
        if result.returncode != 0:
            print(f"    [ERROR] Evaluation of {tracker_name} failed:\n{result.stderr[-800:]}")
            return None

        # ── parse metrics from stdout ────────────────────────────────────────
        stdout = result.stdout
        hota  = _parse_metric(stdout, "HOTA")
        idf1  = _parse_metric(stdout, "IDF1")
        if hota is None or idf1 is None:
            # Fallback: read the detailed CSV that eval_tracking.py writes
            detailed = (
                repo_root / "data" / "trackers_mot_format"
                / f"{benchmark_name}-{split}"
                / tracker_name
                / "pedestrian_detailed.csv"
            )
            hota, idf1 = _read_detailed_csv(detailed)

        if hota is None:
            print(f"    [WARN] Could not parse metrics for {tracker_name}")
            return None

        print(f"    HOTA={hota:.2f}%  IDF1={idf1:.2f}%")
        return {"HOTA": hota, "IDF1": idf1}

    except Exception as exc:
        print(f"    [ERROR] Exception evaluating {tracker_name}: {exc}")
        return None


def _parse_metric(text: str, metric: str) -> float | None:
    """Very simple regex parser for lines like '| HOTA    | 45.23 |'."""
    pattern = re.compile(
        rf"(?:^|\s){re.escape(metric)}\s*[|:,\s]+([0-9]+\.?[0-9]*)", re.IGNORECASE | re.MULTILINE
    )
    m = pattern.search(text)
    if m:
        return float(m.group(1))
    return None


def _read_detailed_csv(csv_path: Path) -> tuple[float | None, float | None]:
    if not csv_path.exists():
        return None, None
    try:
        df = pd.read_csv(csv_path)
        row = df[df["seq"] == "COMBINED"].iloc[0] if "COMBINED" in df["seq"].values else df.iloc[0]

        _ALPHA_COL_SUFFIXES = [str(int(round(a * 100))) for a in np.round(np.arange(0.05, 0.99, 0.05), 2)]
        hota_cols = [f"HOTA___{s}" for s in _ALPHA_COL_SUFFIXES if f"HOTA___{s}" in df.columns]
        if hota_cols:
            hota = float(np.mean([float(row[c]) for c in hota_cols])) * 100
        else:
            hota = None

        idf1 = float(row["IDF1"]) * 100 if "IDF1" in df.columns else None
        return hota, idf1
    except Exception as exc:
        print(f"    [WARN] Could not read {csv_path}: {exc}")
        return None, None


# ═════════════════════════════════════════════════════════════════════════════
#  Helper – comparative bar plot
# ═════════════════════════════════════════════════════════════════════════════

def make_comparison_plot(
    results: dict[str, dict],   # {method_label: {"HOTA": float, "IDF1": float}}
    seq_name: str,
    cam_name: str,
    output_path: Path,
) -> None:
    methods = list(results.keys())
    hota_vals = [results[m]["HOTA"] for m in methods]
    idf1_vals = [results[m]["IDF1"] for m in methods]

    x = np.arange(len(methods))
    width = 0.35

    colors = []
    for i, m in enumerate(methods):
        key = m.split("_")[0] if m not in METHOD_COLORS else m
        colors.append(METHOD_COLORS.get(m, FALLBACK_COLORS[i % len(FALLBACK_COLORS)]))

    fig, ax = plt.subplots(figsize=(max(8, len(methods) * 1.6), 6))

    bars_hota = ax.bar(x - width / 2, hota_vals, width, label="HOTA", color=colors, alpha=0.85, zorder=3)
    bars_idf1 = ax.bar(x + width / 2, idf1_vals, width, label="IDF1",
                       color=colors, alpha=0.50, hatch="//", edgecolor="white", zorder=3)

    # value labels
    for bar in bars_hota:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.6,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=8)
    for bar in bars_idf1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.6,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 105)
    ax.set_title(f"Tracking comparison – {seq_name} / {cam_name}", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.4, zorder=0)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Plot saved: {output_path}")


# ═════════════════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser(description="Run all trackers on all cameras and compare.")
    ap.add_argument("--data-root", required=True,
                    help="Root folder containing the sequences (S01, S03, S04 …).")
    ap.add_argument("--repo-root", required=True,
                    help="Root of the tracking repository (contains Week3/tracking/).")
    ap.add_argument("--detections-subpath", default="det",
                    help="Sub-folder inside a camera folder containing det_<model>.txt files "
                         "(default: det).")
    ap.add_argument("--gt-subpath", default="gt/gt.txt",
                    help="Sub-path inside a camera folder where the GT lives "
                         "(default: gt/gt.txt).")
    ap.add_argument("--output-dir", default="multi_cam_results",
                    help="Where to save all outputs (default: ./multi_cam_results).")
    ap.add_argument("--benchmark-name", default="AICity")
    ap.add_argument("--split", default="train")
    ap.add_argument("--methods", nargs="+",
                    default=["overlap", "kalman"],
                    choices=["overlap", "kalman", "overlap_flow", "deep_SORT", "deep_SORT_flow"])
    args = ap.parse_args()

    data_root = Path(args.data_root).resolve()
    repo_root = Path(args.repo_root).resolve()
    out_root  = Path(args.output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    cameras = discover_cameras(data_root)
    if not cameras:
        print(f"No cameras found under {data_root}. "
              "Make sure subdirectories contain vdo.avi files.")
        return

    print(f"Found {len(cameras)} camera(s).")

    for cam_path, seq_name, cam_name in cameras:
        print(f"\n{'='*70}")
        print(f"  Sequence: {seq_name}   Camera: {cam_name}")
        print(f"{'='*70}")

        video_path = cam_path / "vdo.avi"
        gt_txt     = cam_path / args.gt_subpath
        det_dir    = cam_path / args.detections_subpath

        if not gt_txt.exists():
            print(f"  [SKIP] GT file not found: {gt_txt}")
            continue

        base_seq_length = get_seq_length(gt_txt, video_path)
        print(f"  Frames (baseline seq_length): {base_seq_length}")

        seq_cam     = f"{seq_name}_{cam_name}"
        cam_out_dir = out_root / seq_cam
        cam_out_dir.mkdir(parents=True, exist_ok=True)

        # ── build the list of detection sources ──────────────────────────────
        # Each entry: (model_label, det_csv_path)
        # Discovers every det_<model>.txt inside the det/ folder automatically.
        det_sources: list[tuple[str, Path]] = []

        if det_dir.is_dir():
            det_files = sorted(det_dir.glob("det_*.txt"))
            if not det_files:
                print(f"  [INFO] No det_<model>.txt files found in {det_dir}, skipping.")
            for det_file in det_files:
                # Extract model name from filename: det_yolo.txt → "yolo"
                model_label = det_file.stem[len("det_"):]   # strip leading "det_"
                converted_csv = cam_out_dir / f"{model_label}_det.csv"
                dataset_det_to_internal_csv(det_file, converted_csv)
                det_sources.append((model_label, converted_csv))
                print(f"  [INFO] Found detection model: {model_label}  ({det_file.name})")
        else:
            print(f"  [INFO] Detection folder not found: {det_dir}, skipping.")

        if not det_sources:
            print("  [SKIP] No detection sources available for this camera.")
            continue

        # ── for each detection source × method → track + evaluate ────────────
        all_results: dict[str, dict] = {}

        for det_label, det_csv in det_sources:
            print(f"\n  Detection source: {det_label}  ({det_csv.name})")

            for method in args.methods:
                label = f"{det_label}\n{method}"   # two-line label for plot
                tracker_name  = f"{seq_cam}__{det_label}__{method}"
                method_out    = cam_out_dir / det_label / method

                # ── run tracker ──────────────────────────────────────────────
                result_dir = run_tracking_method(
                    repo_root=repo_root,
                    method=method,
                    det_csv=det_csv,
                    video_path=video_path,
                    output_dir=method_out,
                )
                if result_dir is None:
                    print(f"    [SKIP] Tracking failed for {label}")
                    continue
                
                # Recompute seq_length now that we have the tracker output,
                # so seqinfo.ini covers every frame the tracker actually produced.
                tracks_txt = result_dir / "tracks.txt"
                seq_length = get_seq_length(gt_txt, video_path, tracker_txt=tracks_txt)
                
                # ── evaluate ─────────────────────────────────────────────────
                metrics = run_evaluation(
                    repo_root=repo_root,
                    tracker_results_dir=result_dir,
                    tracker_name=tracker_name,
                    gt_txt=gt_txt,
                    seq_name=f"{seq_cam}__{det_label}__{method}",
                    seq_length=seq_length,
                    benchmark_name=args.benchmark_name,
                    split=args.split,
                )
                if metrics:
                    all_results[label] = metrics

        # ── plots ─────────────────────────────────────────────────────────────
        if all_results:
            # One plot per model: x-axis = methods, title includes model name
            for model_label, _ in det_sources:
                model_results = {
                    k.split("\n")[1]: v
                    for k, v in all_results.items()
                    if k.startswith(model_label + "\n")
                }
                if not model_results:
                    continue
                plot_path = cam_out_dir / f"comparison_{model_label}.png"
                make_comparison_plot(
                    results=model_results,
                    seq_name=seq_name,
                    cam_name=f"{cam_name}  [det: {model_label}]",
                    output_path=plot_path,
                )

            # Grand comparison across all models × methods (label = "model / method")
            if len(det_sources) > 1:
                grand_results = {
                    f"{k.split(chr(10))[0]} / {k.split(chr(10))[1]}": v
                    for k, v in all_results.items()
                }
                make_comparison_plot(
                    results=grand_results,
                    seq_name=seq_name,
                    cam_name=f"{cam_name}  [all models]",
                    output_path=cam_out_dir / "comparison_all_models.png",
                )

        # Summary CSV
        if all_results:
            summary_rows = [
                {"model": k.split("\n")[0], "method": k.split("\n")[1], **v}
                for k, v in all_results.items()
            ]
            pd.DataFrame(summary_rows).to_csv(cam_out_dir / "summary.csv", index=False)
            print(f"\n  Summary saved: {cam_out_dir / 'summary.csv'}")

    print(f"\n{'='*70}")
    print(f"All done. Results in: {out_root}")


if __name__ == "__main__":
    main()