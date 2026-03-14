import os
import sys
import glob
import argparse
import subprocess
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
from collections import defaultdict
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
                   help="Cosine distance threshold for merging tracklets into the same "
                        "global identity. Lower → fewer merges (more IDs), "
                        "higher → more aggressive merging.")
    p.add_argument("--n-hist-bins", type=int, default=8,
                   help="Number of bins per HSV channel for colour histograms.")
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
    return p.parse_args()

# ─────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────

MTSC_COLS = ["frame_id", "timestamp", "class_id", "confidence",
             "x1", "y1", "x2", "y2", "track_id"]

def load_mtsc(txt_path: str, cam_id: int) -> pd.DataFrame:
    """Load a single tracking_overlap.txt and add camera id."""
    df = pd.read_csv(txt_path, header=0, names=MTSC_COLS)
    df["cam_id"] = cam_id
    # Derive bbox in (x, y, w, h) format expected by the eval script
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
        cam_name = os.path.basename(cam_dir)               # e.g. "c001"
        cam_id   = int(cam_name[1:])                       # → 1
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

    # Sample uniformly across the tracklet
    rows = tracklet_df.sample(min(max_samples, len(tracklet_df)),
                              random_state=0).sort_values("frame_id")

    hist_sum = None
    n_valid  = 0

    for _, row in rows.iterrows():
        # Try common zero-padded filename conventions
        fid  = int(row["frame_id"])
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
        frame_ids = sorted(tracklet_df["frame_id"].unique().tolist())
        print(f"  [DEBUG] No valid frames for tracklet in {img_dir}")
        print(f"  [DEBUG] frame_ids in tracklet: {frame_ids[:20]}")
        sample_path = os.path.join(img_dir, f"{frame_ids[0]:06d}.jpg")
        print(f"  [DEBUG] First candidate path: {sample_path!r}, exists={os.path.isfile(sample_path)}")
        raise ValueError("[ERROR] No valid frames found for tracklet; cannot extract HSV features.")

    # L2-normalise
    hist_avg = hist_sum / n_valid
    norm = np.linalg.norm(hist_avg)
    if norm > 0:
        hist_avg /= norm
    return hist_avg

def build_tracklet_features(cameras, sequence, n_bins, max_samples):
    """
    Returns:
      tracklets : list of dicts with keys
                  cam_id, track_id, cam_name, feature, df
      cam_track_to_idx : dict (cam_id, track_id) → index in tracklets
    """
    tracklets = []
    cam_track_to_idx = {}

    for cam_id, df in tqdm(cameras, desc="Building tracklet features", unit="camera"):
        cam_name = f"c{cam_id:03d}"
        for tid, tdf in df.groupby("track_id"):
            feat = color_histogram_feature(
                tdf, sequence, cam_name, n_bins, max_samples)

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
    n = len(tracklets)
    feats = np.vstack([t["feature"] for t in tracklets])

    # Handle zero vectors gracefully
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    feats_normed = feats / norms

    cos_dist = cdist(feats_normed, feats_normed, metric="cosine")
    cos_dist = np.clip(cos_dist, 0.0, 2.0)

    # Block same-camera pairs from being merged
    for i, ti in enumerate(tracklets):
        for j, tj in enumerate(tracklets):
            if ti["cam_id"] == tj["cam_id"]:
                cos_dist[i, j] = 1.0   # max cosine distance = 2, using 1.0 as a soft block

    # Diagonal to 0
    np.fill_diagonal(cos_dist, 0.0)
    return cos_dist.astype(np.float32)

def cluster_tracklets(tracklets, dist_threshold):
    """
    Agglomerative clustering (average linkage) on the distance matrix.
    Returns an array of global IDs, one per tracklet.
    """
    if len(tracklets) == 1:
        return np.array([1])

    dist_matrix = build_distance_matrix(tracklets)

    # Convert square distance matrix to condensed form
    n = len(tracklets)
    condensed = []
    for i in range(n):
        for j in range(i + 1, n):
            condensed.append(dist_matrix[i, j])
    condensed = np.array(condensed, dtype=np.float64)

    # Average-linkage hierarchical clustering
    Z = linkage(condensed, method="average")
    labels = fcluster(Z, t=dist_threshold, criterion="distance")
    return labels   # 1-based cluster IDs

# ─────────────────────────────────────────────
# Build output DataFrame
# ─────────────────────────────────────────────

def build_output(tracklets, global_ids, min_cams=2):
    """
    Assemble the final DataFrame in AI City eval format:
      CameraId, Id, FrameId, X, Y, Width, Height, Xworld, Yworld

    Only global IDs spanning >= min_cams cameras are included,
    matching the ground-truth filtering used by the eval script.
    """
    # find which global IDs span enough cameras
    gid_to_cams = defaultdict(set)
    for t, gid in zip(tracklets, global_ids):
        gid_to_cams[gid].add(t["cam_id"])
    valid_gids = {gid for gid, cams in gid_to_cams.items()
                  if len(cams) >= min_cams}

    records = []
    for t, gid in zip(tracklets, global_ids):
        if gid not in valid_gids:
            continue
        df = t["df"]
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

def run_evaluation(eval_script, gt_path, pred_path, dstype, roi_dir):
    """Call the official eval.py and print its output."""
    cmd = [
        sys.executable, eval_script,
        gt_path, pred_path,
        "--dstype", dstype,
        "--roidir", roi_dir,
    ]
    print("\n" + "═" * 60)
    print("  Running official AI City evaluation…")
    print("  CMD:", " ".join(cmd))
    print("═" * 60)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print("[STDERR]", result.stderr)
    if result.returncode != 0:
        print(f"[WARN] eval.py exited with code {result.returncode}")

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    args = parse_args()

    print("\n" + "═" * 60)
    print(f"  Sequence      : {args.sequence}")
    print(f"  Root dir      : {args.root}")
    print(f"  Feature mode  : HSV colour histograms")
    print(f"  Dist threshold: {args.dist_threshold}")
    print("═" * 60 + "\n")

    # ── Load MTSC data ───────────────────────
    print(f"[1/4] Loading MTSC tracklets for {args.sequence}…")
    cameras = load_sequence(args.root, args.sequence)
    if not cameras:
        sys.exit("[ERROR] No cameras loaded. Aborting.")

    # ── Extract features ─────────────────────
    print(f"\n[2/4] Extracting features…")
    tracklets, _ = build_tracklet_features(
        cameras,
        sequence=args.sequence,
        n_bins=args.n_hist_bins,
        max_samples=args.max_frames_sample,
    )

    # ── Cluster / Re-ID ──────────────────────
    print(f"\n[3/4] Clustering tracklets (threshold={args.dist_threshold})…")
    global_ids = cluster_tracklets(tracklets, args.dist_threshold)
    n_global = len(set(global_ids))
    print(f"  → Assigned {n_global} global identities "
          f"from {len(tracklets)} per-camera tracklets")

    # Print a brief summary of cross-camera assignments
    from collections import Counter
    gid_to_cams = defaultdict(set)
    for t, gid in zip(tracklets, global_ids):
        gid_to_cams[gid].add(t["cam_id"])
    multi_cam_ids = {gid for gid, cams in gid_to_cams.items() if len(cams) > 1}
    print(f"  → IDs spanning ≥2 cameras: {len(multi_cam_ids)} "
          f"(these are the ones evaluated)")

    # ── Save output ──────────────────────────
    print(f"\n[4/4] Writing output…")
    os.makedirs(args.output, exist_ok=True)
    out_path = os.path.join(args.output, f"{args.sequence}_mtmc.txt")
    out_df = build_output(tracklets, global_ids)
    save_output(out_df, out_path)

    # ── Evaluate ─────────────────────────────
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