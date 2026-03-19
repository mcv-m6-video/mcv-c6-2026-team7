import os
import sys
import glob
import argparse
import subprocess
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
import time
from collections import defaultdict, Counter
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import linkage, fcluster

# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="MTMC baseline for AI City Challenge",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("sequence", type=str,
                   help="Sequence name, e.g. S01")
    p.add_argument("--root", type=str,
                   default="Week4/tracking_overlap_yolov3u_base/output_detections",
                   help="Root folder that contains S01/, S03/, …")
    p.add_argument("--output", type=str, default="Week4/results",
                   help="Directory where the output .txt will be written")

    p.add_argument("--dist-threshold", type=float, default=0.45,
                   help="[Single-run mode] Cosine distance threshold for merging tracklets. "
                        "Ignored when --grid-search is set.")
    p.add_argument("--n-hist-bins", type=int, default=8,
                   help="[Single-run mode] Number of bins per HSV channel. "
                        "Ignored when --grid-search is set.")
    p.add_argument("--max-frames-sample", type=int, default=4,
                   help="Max frames to sample per tracklet when extracting colour features.")
    p.add_argument("--gt", type=str,
                   default="./AI_CITY_CHALLENGE_2022_TRAIN/eval/ground_truth_train.txt",
                   help="Path to ground-truth .txt file. If given, eval.py is run.")
    p.add_argument("--eval-script", type=str,
                   default="./AI_CITY_CHALLENGE_2022_TRAIN/eval/eval.py",
                   help="Path to the official AI City eval.py script.")
    p.add_argument("--dstype", type=str, default="train",
                   help="Dataset type passed to eval.py (train/validation/test).")
    p.add_argument("--roi-dir", type=str, default="ROIs",
                   help="ROI directory passed to eval.py.")

    p.add_argument("--grid-search", action="store_true",
                   help="Run a grid search over n-hist-bins and dist-threshold. "
                        "Overrides --n-hist-bins and --dist-threshold.")
    return p.parse_args()

# ─────────────────────────────────────────────
# Grid-search parameter space
# ─────────────────────────────────────────────

GRID_N_HIST_BINS = [4, 8, 16, 24, 32, 64, 96, 128]
GRID_DIST_THRESH = [round(v, 2) for v in np.arange(0.10, 0.46, 0.05)]
# → [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]

# ─────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────

MTSC_COLS = ["frame_id", "timestamp", "class_id", "confidence",
             "x1", "y1", "x2", "y2", "track_id"]

def load_mtsc(txt_path: str, cam_id: int) -> pd.DataFrame:
    """Load a single tracking_overlap.txt and add camera id."""
    df = pd.read_csv(txt_path, header=0, names=MTSC_COLS)
    df["cam_id"] = cam_id
    df["width"]  = df["x2"] - df["x1"]
    df["height"] = df["y2"] - df["y1"]
    return df

def load_sequence(root: str, sequence: str):
    """Return a list of (cam_id:int, DataFrame) tuples for the sequence."""
    seq_dir = os.path.join(root, sequence)
    if not os.path.isdir(seq_dir):
        sys.exit(f"[ERROR] Sequence directory not found: {seq_dir}")

    cam_dirs = sorted(glob.glob(os.path.join(seq_dir, "c[0-9][0-9][0-9]")))
    if not cam_dirs:
        sys.exit(f"[ERROR] No camera directories (cXXX) found in {seq_dir}")

    cameras = []
    for cam_dir in cam_dirs:
        cam_name = os.path.basename(cam_dir)
        cam_id   = int(cam_name[1:])
        txt_file = os.path.join(cam_dir, "tracking_overlap.txt")
        if not os.path.isfile(txt_file):
            print(f"[WARN] Missing {txt_file}, skipping.")
            continue
        df = load_mtsc(txt_file, cam_id)
        cameras.append((cam_id, df))
        print(f"  Loaded cam {cam_id:>3d}  ({cam_name})  —  "
              f"{df['track_id'].nunique()} tracklets, "
              f"{len(df)} detections")
    return cameras

# ─────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────

FRAMES_ROOT = "AI_CITY_CHALLENGE_2022_TRAIN/train"

def color_histogram_feature(tracklet_df: pd.DataFrame,
                             sequence: str,
                             cam_name: str,
                             n_bins: int = 16,
                             max_samples: int = 10) -> np.ndarray:
    """
    Build a concatenated HSV colour histogram (H + S + V channels) by
    cropping the bounding-boxes of up to `max_samples` frames and averaging.

    Frames are read from:  AI_CITY_CHALLENGE_2022_TRAIN/<sequence>/<cam>/img1/<frame>.jpg
    """
    img_dir = os.path.join(FRAMES_ROOT, sequence, cam_name, "img1")
    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"[ERROR] Frames directory not found: {img_dir}")

    rows = tracklet_df.sample(min(max_samples, len(tracklet_df)),
                              random_state=0).sort_values("frame_id")

    hist_sum = None
    n_valid  = 0

    for _, row in rows.iterrows():
        fid      = int(row["frame_id"])
        img_path = os.path.join(img_dir, f"{fid:06d}.jpg")
        if not os.path.isfile(img_path):
            continue

        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue

        x1 = max(0, int(row["x1"]))
        y1 = max(0, int(row["y1"]))
        x2 = min(img.shape[1], int(row["x2"]))
        y2 = min(img.shape[0], int(row["y2"]))
        if x2 <= x1 or y2 <= y1:
            continue

        crop = img[y1:y2, x1:x2]
        hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        h_h = cv2.calcHist([hsv], [0], None, [n_bins], [0, 180]).flatten()
        h_s = cv2.calcHist([hsv], [1], None, [n_bins], [0, 256]).flatten()
        h_v = cv2.calcHist([hsv], [2], None, [n_bins], [0, 256]).flatten()
        hist = np.concatenate([h_h, h_s, h_v]).astype(np.float32)

        if hist_sum is None:
            hist_sum = hist
        else:
            hist_sum += hist
        n_valid += 1

    if n_valid == 0 or hist_sum is None:
        frame_ids   = sorted(tracklet_df["frame_id"].unique().tolist())
        print(f"  [DEBUG] No valid frames for tracklet in {img_dir}")
        print(f"  [DEBUG] frame_ids in tracklet: {frame_ids[:20]}")
        sample_path = os.path.join(img_dir, f"{frame_ids[0]:06d}.jpg")
        print(f"  [DEBUG] First candidate path: {sample_path!r}, exists={os.path.isfile(sample_path)}")
        raise ValueError("[ERROR] No valid frames found for tracklet; cannot extract HSV features.")

    hist_avg = hist_sum / n_valid
    norm = np.linalg.norm(hist_avg)
    if norm > 0:
        hist_avg /= norm
    return hist_avg

def build_tracklet_features(cameras, sequence, n_bins, max_samples):
    """
    Returns:
      tracklets        : list of dicts with keys cam_id, track_id, cam_name, feature, df
      cam_track_to_idx : dict (cam_id, track_id) → index in tracklets
    """
    tracklets        = []
    cam_track_to_idx = {}

    for cam_id, df in tqdm(cameras, desc="Building tracklet features", unit="camera"):
        cam_name = f"c{cam_id:03d}"
        for tid, tdf in df.groupby("track_id"):
            feat = color_histogram_feature(tdf, sequence, cam_name, n_bins, max_samples)

            idx = len(tracklets)
            tracklets.append({
                "cam_id":   cam_id,
                "track_id": tid,
                "cam_name": cam_name,
                "feature":  feat,
                "df":       tdf,
            })
            cam_track_to_idx[(cam_id, tid)] = idx

    print(f"\n  Total tracklets across all cameras: {len(tracklets)}")
    return tracklets, cam_track_to_idx

# ─────────────────────────────────────────────
# Cross-camera matching
# ─────────────────────────────────────────────

def build_distance_matrix(tracklets):
    """
    Compute pairwise cosine distance between all tracklets.
    Same-camera pairs are forced to distance 1.0 (never merged).
    """
    feats = np.vstack([t["feature"] for t in tracklets])

    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    feats_normed = feats / norms

    cos_dist = cdist(feats_normed, feats_normed, metric="cosine")
    cos_dist = np.clip(cos_dist, 0.0, 2.0)

    for i, ti in enumerate(tracklets):
        for j, tj in enumerate(tracklets):
            if ti["cam_id"] == tj["cam_id"]:
                cos_dist[i, j] = 1.0

    np.fill_diagonal(cos_dist, 0.0)
    return cos_dist.astype(np.float32)

def cluster_tracklets(tracklets, dist_threshold, dist_matrix=None):
    """
    Agglomerative clustering (average linkage) on the distance matrix.
    Accepts a pre-computed dist_matrix to avoid recomputing it.
    Returns an array of global IDs, one per tracklet.
    """
    if len(tracklets) == 1:
        return np.array([1])

    if dist_matrix is None:
        dist_matrix = build_distance_matrix(tracklets)

    n = len(tracklets)
    condensed = []
    for i in range(n):
        for j in range(i + 1, n):
            condensed.append(dist_matrix[i, j])
    condensed = np.array(condensed, dtype=np.float64)

    Z      = linkage(condensed, method="average")
    labels = fcluster(Z, t=dist_threshold, criterion="distance")
    return labels   # 1-based cluster IDs

# ─────────────────────────────────────────────
# Build output DataFrame
# ─────────────────────────────────────────────

def build_output(tracklets, global_ids, min_cams=2):
    """
    Assemble the final DataFrame in AI City eval format:
      CameraId, Id, FrameId, X, Y, Width, Height, Xworld, Yworld

    Only global IDs spanning >= min_cams cameras are included.
    """
    gid_to_cams = defaultdict(set)
    for t, gid in zip(tracklets, global_ids):
        gid_to_cams[gid].add(t["cam_id"])
    valid_gids = {gid for gid, cams in gid_to_cams.items()
                  if len(cams) >= min_cams}

    records = []
    for t, gid in zip(tracklets, global_ids):
        if gid not in valid_gids:
            continue
        df  = t["df"]
        cam = t["cam_id"]
        for _, row in df.iterrows():
            records.append({
                "CameraId": cam,
                "Id":       int(gid),
                "FrameId":  int(row["frame_id"]),
                "X":        int(row["x1"]),
                "Y":        int(row["y1"]),
                "Width":    int(row["width"]),
                "Height":   int(row["height"]),
                "Xworld":   -1,
                "Yworld":   -1,
            })

    out_df = pd.DataFrame(records, columns=[
        "CameraId", "Id", "FrameId", "X", "Y",
        "Width", "Height", "Xworld", "Yworld"
    ])

    n_filtered = len(set(global_ids)) - len(valid_gids)
    print(f"  Kept {len(valid_gids)} global IDs spanning >={min_cams} cameras "
          f"({n_filtered} single-camera IDs discarded)")
    return out_df

def save_output(df: pd.DataFrame, out_path: str):
    df.to_csv(out_path, sep=" ", header=False, index=False)
    print(f"\n  Saved MTMC results → {out_path}")
    print(f"  Rows: {len(df)} | Global IDs: {df['Id'].nunique()} | "
          f"Cameras: {df['CameraId'].nunique()}")

# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────

def run_evaluation(eval_script, gt_path, pred_path, dstype, roi_dir,
                   capture_output=False):
    """
    Call the official eval.py.
    If capture_output=True, returns stdout as a string instead of printing.
    """
    cmd = [
        sys.executable, eval_script,
        gt_path, pred_path,
        "--dstype", dstype,
        "--roidir", roi_dir,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if capture_output:
        return result.stdout + (("\n[STDERR] " + result.stderr) if result.stderr else "")

    print("\n" + "═" * 60)
    print("  Running official AI City evaluation…")
    print("  CMD:", " ".join(cmd))
    print("═" * 60)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print("[STDERR]", result.stderr)
    if result.returncode != 0:
        print(f"[WARN] eval.py exited with code {result.returncode}")

# ─────────────────────────────────────────────
# Grid search
# ─────────────────────────────────────────────

def run_grid_search(args, cameras):
    """
    Outer loop  : n_hist_bins  – features + distance matrix + linkage are
                  computed ONCE per bins value, then cached.
    Inner loop  : dist_threshold – only fcluster() is called per threshold,
                  reusing the cached linkage matrix Z.
    """
    total_combos = len(GRID_N_HIST_BINS) * len(GRID_DIST_THRESH)
    print(f"\n{'═'*60}")
    print(f"  GRID SEARCH")
    print(f"  n_hist_bins   : {GRID_N_HIST_BINS}")
    print(f"  dist_threshold: {GRID_DIST_THRESH}")
    print(f"  Total combos  : {total_combos}")
    print(f"{'═'*60}\n")

    os.makedirs(args.output, exist_ok=True)

    results   = []
    combo_idx = 0

    for n_bins in GRID_N_HIST_BINS:

        # ── Feature extraction (once per n_bins) ──────────────────
        print(f"\n{'─'*60}")
        print(f"  [n_hist_bins={n_bins}]  Extracting features…")
        t0 = time.perf_counter()
        tracklets, _ = build_tracklet_features(
            cameras,
            sequence=args.sequence,
            n_bins=n_bins,
            max_samples=args.max_frames_sample,
        )
        t_feat = time.perf_counter() - t0
        print(f"  Feature extraction : {t_feat:.3f}s")

        # ── Distance matrix (once per n_bins) ─────────────────────
        print(f"  Building distance matrix…")
        t0 = time.perf_counter()
        dist_matrix = build_distance_matrix(tracklets)
        t_dist = time.perf_counter() - t0
        print(f"  Distance matrix    : {t_dist:.3f}s")

        # ── Hierarchical linkage (once per n_bins) ─────────────────
        print(f"  Computing linkage…")
        t0 = time.perf_counter()
        n = len(tracklets)
        condensed = dist_matrix[np.triu_indices(n, k=1)].astype(np.float64)
        Z = linkage(condensed, method="average")
        t_link = time.perf_counter() - t0
        print(f"  Linkage            : {t_link:.3f}s")

        # ── Sweep dist_threshold (cheap fcluster calls) ───────────
        for dist_thr in GRID_DIST_THRESH:
            combo_idx += 1
            print(f"\n  [{combo_idx}/{total_combos}] "
                  f"n_hist_bins={n_bins:>3d}  dist_threshold={dist_thr:.2f}", end="  ")

            t0         = time.perf_counter()
            global_ids = fcluster(Z, t=dist_thr, criterion="distance")
            t_cluster  = time.perf_counter() - t0

            n_global    = len(set(global_ids))
            gid_to_cams = defaultdict(set)
            for t, gid in zip(tracklets, global_ids):
                gid_to_cams[gid].add(t["cam_id"])
            multi_cam_ids = {gid for gid, cams in gid_to_cams.items()
                             if len(cams) > 1}

            print(f"→ {n_global} global IDs  "
                  f"({len(multi_cam_ids)} multi-cam)  "
                  f"[fcluster: {t_cluster*1000:.1f}ms]")

            # ── Save output ───────────────────────────────────────
            out_name = (f"{args.sequence}"
                        f"_bins{n_bins:03d}"
                        f"_thr{dist_thr:.2f}"
                        f"_mtmc.txt")
            out_path = os.path.join(args.output, out_name)
            out_df   = build_output(tracklets, global_ids)
            out_df.to_csv(out_path, sep=" ", header=False, index=False)

            row = {
                "n_hist_bins":    n_bins,
                "dist_threshold": dist_thr,
                "n_tracklets":    len(tracklets),
                "n_global_ids":   n_global,
                "n_multi_cam":    len(multi_cam_ids),
                "n_detections":   len(out_df),
                "t_feat_s":       round(t_feat, 3),
                "t_dist_s":       round(t_dist, 3),
                "t_link_s":       round(t_link, 3),
                "t_cluster_ms":   round(t_cluster * 1000, 2),
                "output_file":    out_path,
                "eval_output":    "",
            }

            # ── Evaluate (optional) ───────────────────────────────
            if (args.gt
                    and os.path.isfile(args.gt)
                    and os.path.isfile(args.eval_script)):
                eval_out = run_evaluation(
                    eval_script=args.eval_script,
                    gt_path=args.gt,
                    pred_path=out_path,
                    dstype=args.dstype,
                    roi_dir=args.roi_dir,
                    capture_output=True,
                )
                row["eval_output"] = eval_out.strip()

            results.append(row)

    return results


def print_grid_results(results):
    """Pretty-print the full grid-search results table."""
    sep = "═" * 88
    print(f"\n\n{sep}")
    print("  GRID SEARCH RESULTS SUMMARY")
    print(sep)

    header = (f"{'bins':>5}  {'thr':>5}  {'tracklets':>10}  "
              f"{'global_IDs':>10}  {'multi_cam':>9}  "
              f"{'detections':>10}  {'t_feat(s)':>9}  {'t_clust(ms)':>11}")
    print(header)
    print("─" * len(header))

    for r in results:
        print(
            f"{r['n_hist_bins']:>5d}  "
            f"{r['dist_threshold']:>5.2f}  "
            f"{r['n_tracklets']:>10d}  "
            f"{r['n_global_ids']:>10d}  "
            f"{r['n_multi_cam']:>9d}  "
            f"{r['n_detections']:>10d}  "
            f"{r['t_feat_s']:>9.3f}  "
            f"{r['t_cluster_ms']:>11.1f}"
        )

    print("─" * len(header))

    # If evaluation scores are present, print them grouped
    eval_present = any(r["eval_output"] for r in results)
    if eval_present:
        print(f"\n{sep}")
        print("  EVALUATION OUTPUTS")
        print(sep)
        for r in results:
            if r["eval_output"]:
                print(f"\n  [bins={r['n_hist_bins']:>3d}  thr={r['dist_threshold']:.2f}]"
                      f"  →  {r['output_file']}")
                for line in r["eval_output"].splitlines():
                    print(f"    {line}")

    print(f"\n{sep}\n")

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    args = parse_args()

    print("\n" + "═" * 60)
    print(f"  Sequence      : {args.sequence}")
    print(f"  Root dir      : {args.root}")
    print(f"  Feature mode  : HSV colour histograms")
    if args.grid_search:
        print(f"  Mode          : GRID SEARCH")
    else:
        print(f"  Dist threshold: {args.dist_threshold}")
        print(f"  n-hist-bins   : {args.n_hist_bins}")
    print("═" * 60 + "\n")

    # ── Load MTSC data ───────────────────────────────────────────
    print(f"[1] Loading MTSC tracklets for {args.sequence}…")
    cameras = load_sequence(args.root, args.sequence)
    if not cameras:
        sys.exit("[ERROR] No cameras loaded. Aborting.")

    # ── Grid search branch ───────────────────────────────────────
    if args.grid_search:
        results = run_grid_search(args, cameras)
        print_grid_results(results)
        print("Done.\n")
        return

    # ── Single-run branch (original behaviour) ───────────────────
    print(f"\n[2/4] Extracting features…")
    t_feat_start = time.perf_counter()
    tracklets, _ = build_tracklet_features(
        cameras,
        sequence=args.sequence,
        n_bins=args.n_hist_bins,
        max_samples=args.max_frames_sample,
    )
    t_feat_elapsed = time.perf_counter() - t_feat_start
    print(f"  → Feature extraction completed in {t_feat_elapsed:.3f}s")

    print(f"\n[3/4] Clustering tracklets (threshold={args.dist_threshold})…")
    global_ids = cluster_tracklets(tracklets, args.dist_threshold)
    n_global   = len(set(global_ids))
    print(f"  → Assigned {n_global} global identities "
          f"from {len(tracklets)} per-camera tracklets")

    gid_to_cams = defaultdict(set)
    for t, gid in zip(tracklets, global_ids):
        gid_to_cams[gid].add(t["cam_id"])
    multi_cam_ids = {gid for gid, cams in gid_to_cams.items() if len(cams) > 1}
    print(f"  → IDs spanning ≥2 cameras: {len(multi_cam_ids)} "
          f"(these are the ones evaluated)")

    print(f"\n[4/4] Writing output…")
    os.makedirs(args.output, exist_ok=True)
    out_path = os.path.join(args.output, f"{args.sequence}_mtmc.txt")
    out_df   = build_output(tracklets, global_ids)
    save_output(out_df, out_path)

    if args.gt:
        if not os.path.isfile(args.gt):
            print(f"[WARN] Ground-truth file not found: {args.gt}. Skipping eval.")
        elif not os.path.isfile(args.eval_script):
            print(f"[WARN] eval.py not found at {args.eval_script}. Skipping eval.")
        else:
            run_evaluation(
                eval_script=args.eval_script,
                gt_path=args.gt,
                pred_path=out_path,
                dstype=args.dstype,
                roi_dir=args.roi_dir,
            )
    else:
        print("\n  [INFO] No --gt provided. Skipping evaluation.")
        print(f"  To evaluate later, run:")
        print(f"    python eval.py <ground_truth.txt> {out_path} --dstype {args.dstype}")

    print("\nDone.\n")

if __name__ == "__main__":
    main()