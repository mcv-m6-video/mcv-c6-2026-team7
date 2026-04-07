"""
run_all_tracking.py
====================
Iterates over every sequence/camera in an AI-City challenge folder,
runs the configured tracking methods, evaluates each with TrackEval
(HOTA + IDF1) and produces per-camera comparative bar plots.

Detection files are expected in an external folder tree:
    <detections_root>/<model_name>/<seq_name>/<cam_name>/<any>.txt

Tracking results are saved mirroring that structure:
    <out_root>/<model_name>/<seq_name>/<cam_name>/<method>/

ROI masks are loaded from (relative to this script's location):
    ROIs/train/<cam_name>/roi.jpg
Any detection whose bounding-box centre falls outside the white region of
the mask is discarded before the CSV is written.  If no mask is found for
a camera the detections are kept as-is and a warning is printed.

Can be called:
  - Standalone:  python run_all_tracking.py --data-root ... --repo-root ...
  - From pipeline: main(SimpleNamespace(...))  — no subprocess, all in-process
"""

import os
import sys
import argparse
import shutil
import re
from pathlib import Path
from types import SimpleNamespace

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
# This script lives at  Week4/run_all_tracking.py
# We want imports to resolve as if Python were launched from  Week4/
# (i.e.  "from tracking.main import ..."  works regardless of cwd).
_WEEK4_DIR = Path(__file__).resolve().parent          # …/Week4
if str(_WEEK4_DIR) not in sys.path:
    sys.path.insert(0, str(_WEEK4_DIR))
# ---------------------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── numpy compatibility shims ─────────────────────────────────────────────────
if not hasattr(np, "float"):  np.float  = float
if not hasattr(np, "int"):    np.int    = int
if not hasattr(np, "bool"):   np.bool   = bool

# ── colours used in bar plots ─────────────────────────────────────────────────
METHOD_COLORS = {
    "overlap":       "#4C72B0",
    "kalman":        "#DD8452",
    "overlap_flow":  "#55A868",
    "dataset_det":   "#C44E52",
}
FALLBACK_COLORS = ["#8172B2", "#937860", "#DA8BC3", "#8C8C8C", "#CCB974"]


# =============================================================================
#  Camera discovery
# =============================================================================

def discover_cameras(data_root: Path) -> list:
    """
    Walk data_root looking for directories containing vdo.avi.
    Returns list of (cam_path, seq_name, cam_name).
    """
    cameras = []
    for seq_dir in sorted(data_root.iterdir()):
        if not seq_dir.is_dir():
            continue
        for cam_dir in sorted(seq_dir.iterdir()):
            if cam_dir.is_dir() and (cam_dir / "vdo.avi").exists():
                cameras.append((cam_dir, seq_dir.name, cam_dir.name))
    return cameras


# =============================================================================
#  Detection discovery  (new nested folder layout)
# =============================================================================

def discover_detections(detections_root: Path) -> dict:
    """
    Walk detections_root expecting the layout:
        <detections_root>/<model>/<seq>/<cam>/<det_file>.txt

    Returns a nested dict:
        { (seq_name, cam_name): [ (model_name, det_txt_path), ... ] }

    All .txt files found at the leaf camera level are collected.
    Multiple .txt files inside the same camera folder are all added under
    the same model label (the file stem is appended if there are siblings).
    """
    mapping: dict = {}

    if not detections_root.is_dir():
        print(f"[WARN] detections_root not found: {detections_root}")
        return mapping

    for model_dir in sorted(detections_root.iterdir()):
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name

        for seq_dir in sorted(model_dir.iterdir()):
            if not seq_dir.is_dir():
                continue
            seq_name = seq_dir.name

            for cam_dir in sorted(seq_dir.iterdir()):
                if not cam_dir.is_dir():
                    continue
                cam_name = cam_dir.name

                txt_files = sorted(cam_dir.glob("*.txt"))
                if not txt_files:
                    continue

                key = (seq_name, cam_name)
                if key not in mapping:
                    mapping[key] = []

                if len(txt_files) == 1:
                    # Single file → label is just the model name
                    mapping[key].append((model_name, txt_files[0]))
                else:
                    # Multiple files → disambiguate with the file stem
                    for txt in txt_files:
                        label = f"{model_name}_{txt.stem}"
                        mapping[key].append((label, txt))

    return mapping


# =============================================================================
#  Frame / sequence length helpers
# =============================================================================

def count_frames(video_path: Path) -> int:
    try:
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return max(n, 1)
    except Exception:
        return 2141


def seq_length_from_file(txt_path: Path):
    try:
        max_frame = 0
        with open(txt_path) as fh:
            for line in fh:
                parts = line.strip().split(",")
                if parts:
                    max_frame = max(max_frame, int(float(parts[0])))
        return max_frame if max_frame > 0 else None
    except Exception:
        return None


def get_seq_length(gt_txt: Path, video_path: Path, tracker_txt=None) -> int:
    candidates = []
    from_gt = seq_length_from_file(gt_txt)
    if from_gt:
        candidates.append(from_gt)
    if tracker_txt and Path(tracker_txt).exists():
        from_tracker = seq_length_from_file(Path(tracker_txt))
        if from_tracker:
            candidates.append(from_tracker)
    candidates.append(count_frames(video_path))
    return max(candidates)


# =============================================================================
#  ROI mask helpers
# =============================================================================

# This file lives at:  <repo_root>/Week4/run_all_tracking.py
# ROIs live at:        <repo_root>/Week4/ROIs/train/<cam_name>/roi.jpg
_SCRIPT_DIR = Path(__file__).resolve().parent          # …/Week4
_ROI_ROOT   = _SCRIPT_DIR / "ROIs" / "train"           # …/Week4/ROIs/train


def load_roi_mask(cam_name: str, roi_root: Path = _ROI_ROOT):
    """
    Load the binary ROI mask for *cam_name* as a numpy array (H×W, uint8).
    Returns None if the file does not exist or cv2 is unavailable.

    The mask image is thresholded at 127 so any grey-scale roi.jpg works:
    white (≥128) = valid region, black (<128) = outside ROI.
    """
    roi_path = roi_root / cam_name / "roi.jpg"
    if not roi_path.exists():
        return None
    try:
        import cv2
        img = cv2.imread(str(roi_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        _, mask = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        return mask
    except Exception as exc:
        print(f"  [WARN] Could not load ROI mask {roi_path}: {exc}")
        return None


def is_inside_roi(mask, cx: float, cy: float) -> bool:
    """
    Return True if the point (cx, cy) falls within the white region of *mask*.
    Coordinates are clipped to the mask bounds to handle edge detections.
    """
    h, w = mask.shape
    px = int(min(max(round(cx), 0), w - 1))
    py = int(min(max(round(cy), 0), h - 1))
    return mask[py, px] > 0


# =============================================================================
#  Detection format conversion
# =============================================================================

def _detect_format(det_txt: Path) -> str:
    """
    Sniff the detection file format by inspecting the header or first data row.

    Supported formats:
      "mot"    — MOT-challenge: frame, id, x, y, w, h, conf, ...  (x/y/w/h)
      "xyxy8"  — Our model output: frame_id, timestamp, class_id, conf, x1, y1, x2, y2
      "xyxy6"  — Compact: frame_id, class_id, conf, x1, y1, x2, y2  (6+ cols)
    """
    with open(det_txt) as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            # Header row → inspect column names
            if not parts[0].replace(".", "").replace("-", "").lstrip().isdigit():
                header = [p.lower() for p in parts]
                if "timestamp" in header or "class_id" in header:
                    return "xyxy8"
                return "mot"   # unknown header, fall back to MOT
            # First numeric row: decide by number of columns
            if len(parts) >= 8:
                return "xyxy8"
            return "mot"
    return "mot"


def dataset_det_to_internal_csv(
    det_txt: Path,
    out_csv: Path,
    roi_mask=None,          # numpy uint8 array (H×W) or None
) -> tuple[int, int]:
    """
    Convert a detection file to our internal CSV format:
        frame_id, x1, y1, x2, y2, confidence

    Auto-detects two input layouts:

    MOT-challenge (7+ columns):
        frame, track_id, x, y, w, h, conf, ...
        Coordinates are x/y/w/h (top-left + width/height).

    Model output (8 columns):
        frame_id, timestamp, class_id, conf, x1, y1, x2, y2
        Coordinates are already x1/y1/x2/y2 (top-left + bottom-right).

    If *roi_mask* is provided, detections whose bounding-box centre falls
    outside the white region are silently dropped.

    Returns (total_detections, kept_detections) for logging.
    """
    fmt   = _detect_format(det_txt)
    rows  = []
    total = 0

    with open(det_txt) as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]

            # Skip header / non-numeric rows
            try:
                frame = int(float(parts[0]))
            except ValueError:
                continue

            try:
                if fmt == "xyxy8":
                    # frame_id, timestamp, class_id, conf, x1, y1, x2, y2
                    if len(parts) < 8:
                        continue
                    conf       = float(parts[3])
                    x1, y1     = float(parts[4]), float(parts[5])
                    x2, y2     = float(parts[6]), float(parts[7])
                else:
                    # MOT: frame, id, x, y, w, h, conf, ...
                    if len(parts) < 7:
                        continue
                    x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
                    conf       = float(parts[6])
                    x1, y1     = x, y
                    x2, y2     = x + w, y + h
            except ValueError:
                continue

            total += 1

            if roi_mask is not None:
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                if not is_inside_roi(roi_mask, cx, cy):
                    continue

            rows.append({"frame_id": frame,
                         "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                         "confidence": conf})

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    return total, len(rows)


# =============================================================================
#  Tracking output sanitisation
# =============================================================================

def _sanitise_tracks(src: Path, dst: Path) -> None:
    """
    Write a clean MOT-format tracks file to *dst* from *src*.

    Guarantees TrackEval will not crash with 'cannot convert to float':
      - opens with universal newlines so \r\n is normalised on Windows
      - strips all whitespace from every field before float-parsing
      - skips blank lines, header rows, and lines with fewer than 6 columns
      - handles in-place sanitisation safely (reads fully before writing)
    """
    good_lines = []
    with open(src, newline="") as fh:       # universal newline handled below
        for raw in fh:
            line = raw.strip()              # removes \r, \n, leading spaces
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 6:
                continue
            try:
                [float(p) for p in parts]
            except ValueError:
                continue                   # header or corrupt row — skip
            good_lines.append(",".join(parts))

    # Write atomically (safe even when src == dst)
    with open(dst, "w", newline="\n") as fh:
        for l in good_lines:
            fh.write(l + "\n")


# =============================================================================
#  In-process tracking  (replaces subprocess call)
# =============================================================================

def run_tracking_method_inprocess(
    tracking_main_fn,
    method: str,
    det_csv: Path,
    video_path: Path,
    output_dir: Path,
) -> Path | None:
    """
    Call tracking/main.py's main() directly in-process.
    Returns the output directory (output_dir) or None on failure.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"    Running: {method} …")
    try:
        exp_dir = tracking_main_fn(SimpleNamespace(
            method          = method,
            detections      = str(det_csv),
            video           = str(video_path),
            conf_thr_video  = 0.30,
            iou_thr         = 0.65,
            show_IDs_video  = False,
            show_comp_video = False,
            memory_frames   = 20,
            memory_iou_thr  = 0.4,
            output_dir      = str(output_dir),
        ))
        # tracking_main returns exp_dir (a timestamped subfolder of output_dir)
        # Sanitise and copy tracks.txt to a stable path so TrackEval can find it.
        src_tracks = Path(exp_dir) / "tracks.txt"
        dst_tracks = output_dir / "tracks.txt"
        if src_tracks.exists():
            _sanitise_tracks(src_tracks, dst_tracks)
        return output_dir if dst_tracks.exists() else None
    except Exception as exc:
        print(f"    [ERROR] Tracking failed for {method}: {exc}")
        return None


# =============================================================================
#  In-process evaluation  (replaces subprocess call)
# =============================================================================

def run_evaluation_inprocess(
    eval_main_fn,
    tracker_results_dir: Path,
    tracker_name: str,
    gt_txt: Path,
    seq_name: str,
    seq_length: int,
    repo_root: Path,
    benchmark_name: str = "AICity",
    split: str = "train",
) -> dict | None:
    """
    Call tracking/eval_tracking.py's main() directly in-process.
    Returns dict with keys 'HOTA' and 'IDF1', or None on failure.
    """
    # Pre-stage GT file where TrackEval expects it
    gt_mot_base = repo_root / "data" / "gt_mot_format"
    bm_split_dir = gt_mot_base / f"{benchmark_name}-{split}"
    gt_seq_dir   = bm_split_dir / seq_name / "gt"
    gt_seq_dir.mkdir(parents=True, exist_ok=True)
    dest_gt = gt_seq_dir / "gt.txt"
    if not dest_gt.exists():
        shutil.copy(gt_txt, dest_gt)

    seqinfo = bm_split_dir / seq_name / "seqinfo.ini"
    seqinfo.parent.mkdir(parents=True, exist_ok=True)
    seqinfo.write_text(
        f"[Sequence]\nname={seq_name}\nimDir=img1\nframeRate=10\n"
        f"seqLength={seq_length}\nimWidth=1920\nimHeight=1080\nimExt=.jpg\n"
    )

    print(f"    Evaluating {tracker_name} …")
    try:
        eval_main_fn(SimpleNamespace(
            tracker_results = str(tracker_results_dir.resolve()),
            tracker_name    = tracker_name,
            gt_annotation   = None,   # we pre-staged gt.txt above
            seq_name        = seq_name,
            seq_length      = seq_length,
            benchmark_name  = benchmark_name,
            split           = split,
        ))

        # Read metrics from the CSV eval_tracking.py writes
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
        print(f"    [ERROR] Evaluation failed for {tracker_name}: {exc}")
        return None


def _read_detailed_csv(csv_path: Path):
    if not csv_path.exists():
        return None, None
    try:
        df = pd.read_csv(csv_path)
        row = df[df["seq"] == "COMBINED"].iloc[0] if "COMBINED" in df["seq"].values else df.iloc[0]
        alpha_suffixes = [str(int(round(a * 100))) for a in np.round(np.arange(0.05, 0.99, 0.05), 2)]
        hota_cols = [f"HOTA___{s}" for s in alpha_suffixes if f"HOTA___{s}" in df.columns]
        hota = float(np.mean([float(row[c]) for c in hota_cols])) * 100 if hota_cols else None
        idf1 = float(row["IDF1"]) * 100 if "IDF1" in df.columns else None
        return hota, idf1
    except Exception as exc:
        print(f"    [WARN] Could not read {csv_path}: {exc}")
        return None, None


# =============================================================================
#  Comparison plot
# =============================================================================

def make_comparison_plot(results: dict, seq_name: str, cam_name: str, output_path: Path) -> None:
    methods    = list(results.keys())
    hota_vals  = [results[m]["HOTA"] for m in methods]
    idf1_vals  = [results[m]["IDF1"]  for m in methods]
    x          = np.arange(len(methods))
    width      = 0.35
    colors     = [METHOD_COLORS.get(m, FALLBACK_COLORS[i % len(FALLBACK_COLORS)])
                  for i, m in enumerate(methods)]

    fig, ax = plt.subplots(figsize=(max(8, len(methods) * 1.6), 6))
    bars_hota = ax.bar(x - width/2, hota_vals, width, label="HOTA",
                       color=colors, alpha=0.85, zorder=3)
    bars_idf1 = ax.bar(x + width/2, idf1_vals, width, label="IDF1",
                       color=colors, alpha=0.50, hatch="//", edgecolor="white", zorder=3)

    for bar in list(bars_hota) + list(bars_idf1):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.6,
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


# =============================================================================
#  Core logic — extracted so it can be called from pipeline or CLI
# =============================================================================

def run_all_cameras(
    data_root: Path,
    repo_root: Path,
    out_root: Path,
    methods: list,
    gt_subpath: str,
    detections_root: Path,          # root of the model/seq/cam detection tree
    roi_root: Path,                 # root of the ROI masks  (…/Week4/ROIs/train)
    benchmark_name: str,
    split: str,
    tracking_main_fn,
    eval_main_fn,
) -> None:
    """
    Iterate every camera under data_root, run tracking + evaluation for each
    method × detection source, save summary CSVs and comparison plots.

    Detection files are resolved from:
        detections_root/<model>/<seq>/<cam>/*.txt

    ROI masks are loaded from:
        roi_root/<cam_name>/roi.jpg  (default roi_root = …/Week4/ROIs/train)
    Script is expected at Week4/run_all_tracking.py

    Tracking outputs are saved to:
        out_root/<model>/<seq>/<cam>/<method>/

    Called by both main() (CLI) and stage_run_all_tracking() (pipeline).
    """
    out_root.mkdir(parents=True, exist_ok=True)

    cameras = discover_cameras(data_root)
    if not cameras:
        print(f"No cameras found under {data_root}.")
        return
    print(f"Found {len(cameras)} camera(s).")

    # Build the (seq, cam) → [(model, det_txt), ...] index once
    det_index = discover_detections(detections_root)
    if not det_index:
        print(f"[WARN] No detection files found under: {detections_root}")

    for cam_path, seq_name, cam_name in cameras:
        print(f"\n{'='*70}")
        print(f"  Sequence: {seq_name}   Camera: {cam_name}")
        print(f"{'='*70}")

        video_path = cam_path / "vdo.avi"
        gt_txt     = cam_path / gt_subpath

        if not gt_txt.exists():
            print(f"  [SKIP] GT file not found: {gt_txt}")
            continue

        base_seq_length = get_seq_length(gt_txt, video_path)
        print(f"  Frames (baseline seq_length): {base_seq_length}")

        # ── ROI mask for this camera ──────────────────────────────────────────
        roi_mask = load_roi_mask(cam_name, roi_root)
        if roi_mask is not None:
            print(f"  [ROI] Mask loaded for {cam_name}  "
                  f"({roi_mask.shape[1]}×{roi_mask.shape[0]} px)")
        else:
            print(f"  [ROI] No mask found for {cam_name} — all detections kept")

        # ── resolve detection sources for this (seq, cam) ────────────────────
        raw_sources = det_index.get((seq_name, cam_name), [])
        if not raw_sources:
            print(f"  [SKIP] No detections found for {seq_name}/{cam_name} "
                  f"under {detections_root}")
            continue

        # Convert each .txt → internal CSV, applying ROI filter if available.
        # CSVs live next to the tracking outputs for traceability:
        #   out_root/<model>/<seq>/<cam>/det_converted.csv
        det_sources = []
        for model_label, det_txt in raw_sources:
            converted_dir = out_root / model_label / seq_name / cam_name
            converted_dir.mkdir(parents=True, exist_ok=True)
            converted_csv = converted_dir / "det_converted.csv"
            total, kept = dataset_det_to_internal_csv(det_txt, converted_csv, roi_mask)
            det_sources.append((model_label, converted_csv))
            if roi_mask is not None:
                dropped = total - kept
                print(f"  [INFO] Detection model: {model_label}  "
                      f"({kept}/{total} detections kept after ROI filter, "
                      f"{dropped} dropped)")
            else:
                print(f"  [INFO] Detection model: {model_label}  "
                      f"({total} detections)")

        # ── track + evaluate each detection × method ─────────────────────────
        all_results: dict = {}

        for det_label, det_csv in det_sources:
            print(f"\n  Detection source: {det_label}")

            for method in methods:
                label        = f"{det_label}\n{method}"
                tracker_name = f"{seq_name}__{cam_name}__{det_label}__{method}"

                # Output folder mirrors the detection tree
                # out_root/<model>/<seq>/<cam>/<method>/
                method_out = out_root / det_label / seq_name / cam_name / method

                result_dir = run_tracking_method_inprocess(
                    tracking_main_fn=tracking_main_fn,
                    method=method,
                    det_csv=det_csv,
                    video_path=video_path,
                    output_dir=method_out,
                )
                if result_dir is None:
                    print(f"    [SKIP] Tracking failed for {label}")
                    continue

                tracks_txt = result_dir / "tracks.txt"
                seq_length = get_seq_length(gt_txt, video_path, tracker_txt=tracks_txt)

                metrics = run_evaluation_inprocess(
                    eval_main_fn=eval_main_fn,
                    tracker_results_dir=result_dir,
                    tracker_name=tracker_name,
                    gt_txt=gt_txt,
                    seq_name=f"{seq_name}__{cam_name}__{det_label}__{method}",
                    seq_length=seq_length,
                    repo_root=repo_root,
                    benchmark_name=benchmark_name,
                    split=split,
                )
                if metrics:
                    all_results[label] = metrics

        # ── plots ─────────────────────────────────────────────────────────────
        # Plots are saved alongside the converted CSV for each model:
        #   out_root/<model>/<seq>/<cam>/comparison_<model>.png
        if all_results:
            for model_label, _ in det_sources:
                model_results = {
                    k.split("\n")[1]: v
                    for k, v in all_results.items()
                    if k.startswith(model_label + "\n")
                }
                if model_results:
                    plot_dir = out_root / model_label / seq_name / cam_name
                    make_comparison_plot(
                        results=model_results,
                        seq_name=seq_name,
                        cam_name=f"{cam_name}  [det: {model_label}]",
                        output_path=plot_dir / f"comparison_{model_label}.png",
                    )

            if len(det_sources) > 1:
                grand = {
                    f"{k.split(chr(10))[0]} / {k.split(chr(10))[1]}": v
                    for k, v in all_results.items()
                }
                # Grand comparison plot goes in out_root/<seq>/<cam>/
                grand_dir = out_root / seq_name / cam_name
                make_comparison_plot(
                    results=grand,
                    seq_name=seq_name,
                    cam_name=f"{cam_name}  [all models]",
                    output_path=grand_dir / "comparison_all_models.png",
                )

        # ── summary CSV ───────────────────────────────────────────────────────
        if all_results:
            summary_rows = [
                {"model": k.split("\n")[0], "method": k.split("\n")[1], **v}
                for k, v in all_results.items()
            ]
            # One summary per (seq, cam) for easy cross-model comparison
            summary_dir = out_root / seq_name / cam_name
            summary_dir.mkdir(parents=True, exist_ok=True)
            summary_path = summary_dir / "summary.csv"
            pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
            print(f"\n  Summary saved: {summary_path}")

    print(f"\n{'='*70}")
    print(f"All done. Results in: {out_root}")


# =============================================================================
#  Public entry point — called from pipeline or CLI
# =============================================================================

def main(args=None) -> None:
    """
    Run all single-camera trackings across a multi-camera sequence.

    Called from pipeline with a SimpleNamespace, or standalone via CLI.
    When args is None, arguments are parsed from the command line as usual.
    """
    if args is None:
        ap = argparse.ArgumentParser(description="Run all trackers on all cameras and compare.")
        ap.add_argument("--data-root",       required=True,
                        help="Root folder containing the sequences (S01, S03, S04 …).")
        ap.add_argument("--repo-root",       required=True,
                        help="Root of the tracking repository.")
        ap.add_argument("--output-dir",      default="multi_cam_results")
        ap.add_argument("--detections-root", default=None,
                        help="Root of the detection folder tree "
                             "(<model>/<seq>/<cam>/*.txt). "
                             "Defaults to <data-root>/detections if omitted.")
        ap.add_argument("--roi-root",        default=None,
                        help="Root of the ROI masks folder "
                             "(<cam_name>/roi.jpg). "
                             "Defaults to ROIs/train relative to this script.")
        ap.add_argument("--gt-subpath",      default="gt/gt.txt")
        ap.add_argument("--benchmark-name",  default="AICity")
        ap.add_argument("--split",           default="train")
        ap.add_argument("--methods", nargs="+",
                        default=["kalman"],
                        choices=["overlap", "kalman", "overlap_flow",
                                 "deep_SORT", "deep_SORT_flow"])
        args = ap.parse_args()

    # Lazy-import the in-process functions so this file can be imported
    # without the tracking package on sys.path
    from tracking.main import main as tracking_main_fn
    from tracking.eval_tracking import main as eval_main_fn

    data_root = Path(args.data_root).resolve()

    # Allow callers (pipeline) to pass detections_root directly as an attribute
    raw_det_root = getattr(args, "detections_root", None)
    detections_root = (
        Path(raw_det_root).resolve() if raw_det_root
        else data_root / "detections"
    )

    # ROI root: explicit arg > pipeline attr > default (../ROIs from this script)
    raw_roi_root = getattr(args, "roi_root", None)
    roi_root = (
        Path(raw_roi_root).resolve() if raw_roi_root
        else _ROI_ROOT
    )

    run_all_cameras(
        data_root        = data_root,
        repo_root        = Path(args.repo_root).resolve(),
        out_root         = Path(args.output_dir).resolve(),
        methods          = args.methods,
        gt_subpath       = args.gt_subpath,
        detections_root  = detections_root,
        roi_root         = roi_root,
        benchmark_name   = args.benchmark_name,
        split            = args.split,
        tracking_main_fn = tracking_main_fn,
        eval_main_fn     = eval_main_fn,
    )


if __name__ == "__main__":
    main()