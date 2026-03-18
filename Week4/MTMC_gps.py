import os
import sys
import glob
import math
import argparse
import subprocess
import functools
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
from collections import defaultdict
from scipy.spatial.distance import cdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
from gps_utils import (
    load_homographies,
    tracklet_first_world_pos,
    tracklet_last_world_pos,
    estimate_world_scale,
    build_spatiotemporal_gate,
)

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
                   default="C:/Users/Usuario/Documents/MCV/C6/mcv-c6-2026-team7/Week4/tracking_overlap_yolov26x_base/output_detections",
                   help="Root folder that contains S01/, S03/, …")
    p.add_argument("--output", type=str, default="Week4/results",
                   help="Directory where the output .txt will be written")
    p.add_argument("--dist-threshold", type=float, default=0.15,
                   help="Cosine distance threshold for merging tracklets into the same "
                        "global identity. Lower → fewer merges (more IDs), "
                        "higher → more aggressive merging.")
    p.add_argument("--max-speed", type=float, default=30.0,
                   help="Maximum physically plausible car speed in m/s for the "
                        "spatio-temporal GPS gate (default 30 m/s = 108 km/h). "
                        "Pairs implying a higher speed are hard-blocked.")
    p.add_argument("--reproj-threshold", type=float, default=15.0,
                   help="Reprojection error (px) above which the GPS speed limit is "
                        "relaxed proportionally. Cameras with poor homographies get "
                        "a looser gate rather than being treated equally to accurate ones.")
    p.add_argument("--calib-root", type=str,
                   default="C:/Users/Usuario/Documents/MCV/C6/mcv-c6-2026-team7/Week4/AI_CITY_CHALLENGE_2022_TRAIN/train",
                   help="Root folder with calibration.txt files (same layout as --root). "
                        "Defaults to --root if not provided.")
    p.add_argument("--frames-root", type=str,
                   default="AI_CITY_CHALLENGE_2022_TRAIN/train",
                   help="Root folder containing per-sequence per-camera img1/ frame images.")
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

def _make_frame_loader(img_dir: str):
    """Return a per-camera cached frame loader to avoid redundant disk I/O."""
    @functools.lru_cache(maxsize=64)
    def load_frame(fid: int):
        img_path = os.path.join(img_dir, f"{fid:06d}.jpg")
        if not os.path.isfile(img_path):
            return None
        return cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    return load_frame


def color_histogram_feature(tracklet_df: pd.DataFrame,
                             sequence: str,
                             cam_name: str,
                             frames_root: str,
                             n_bins: int = 16,
                             max_samples: int = 10,
                             frame_loader=None) -> np.ndarray:
    """
    Build a concatenated, L2-normalised HSV colour histogram (H + S + V)
    by cropping bounding boxes of up to `max_samples` frames and averaging.
    """
    img_dir = os.path.join(frames_root, sequence, cam_name, "img1")
    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"[ERROR] Frames directory not found: {img_dir}")

    load_frame = frame_loader or _make_frame_loader(img_dir)

    rows = tracklet_df.sample(min(max_samples, len(tracklet_df)),
                              random_state=0).sort_values("frame_id")

    hist_sum = None
    n_valid  = 0

    for _, row in rows.iterrows():
        fid = int(row["frame_id"])
        img = load_frame(fid)
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

        hist_sum = hist if hist_sum is None else hist_sum + hist
        n_valid += 1

    if n_valid == 0 or hist_sum is None:
        frame_ids = sorted(tracklet_df["frame_id"].unique().tolist())
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


def build_tracklet_features(cameras, sequence, n_bins, max_samples,
                             frames_root: str,
                             homographies: dict = None):
    """
    Returns:
      tracklets        : list of dicts with keys
                         cam_id, track_id, cam_name, feature, df,
                         world_first, world_last,
                         _H, _reproj_error   ← stored for scale estimation & gate
      cam_track_to_idx : dict (cam_id, track_id) → index in tracklets
    """
    tracklets = []
    cam_track_to_idx = {}

    for cam_id, df in tqdm(cameras, desc="Building tracklet features", unit="camera"):
        cam_name = f"c{cam_id:03d}"

        H, reproj_err = None, None
        if homographies and cam_id in homographies:
            H, reproj_err = homographies[cam_id]

        img_dir = os.path.join(frames_root, sequence, cam_name, "img1")
        frame_loader = _make_frame_loader(img_dir) if os.path.isdir(img_dir) else None

        for tid, tdf in df.groupby("track_id"):
            feat = color_histogram_feature(
                tdf, sequence, cam_name, frames_root, n_bins, max_samples,
                frame_loader=frame_loader)

            world_first = tracklet_first_world_pos(tdf, H) if H is not None else None
            world_last  = tracklet_last_world_pos(tdf, H)  if H is not None else None

            idx = len(tracklets)
            tracklets.append({
                "cam_id":        cam_id,
                "track_id":      tid,
                "cam_name":      cam_name,
                "feature":       feat,
                "df":            tdf,
                "world_first":   world_first,
                "world_last":    world_last,
                # Stored privately for scale estimation and gate relaxation.
                # Prefixed with _ to signal they are internal, not output fields.
                "_H":            H,
                "_reproj_error": reproj_err if reproj_err is not None else 0.0,
            })
            cam_track_to_idx[(cam_id, tid)] = idx

    print(f"\n  Total tracklets across all cameras: {len(tracklets)}")
    n_with_gps = sum(1 for t in tracklets if t["world_first"] is not None)
    print(f"  Tracklets with GPS world positions: {n_with_gps}/{len(tracklets)}")
    return tracklets, cam_track_to_idx

# ─────────────────────────────────────────────
# Cross-camera matching
# ─────────────────────────────────────────────

def build_distance_matrix(tracklets,
                           st_gate: np.ndarray = None):
    """
    Compute pairwise cosine distance between all tracklets.

    Gate logic (applied in order):
      1. Same-camera pairs → always distance = 1.0 (hard block).
      2. Cross-camera pairs blocked by the spatio-temporal gate
         (st_gate[i,j] == False) → distance = 1.0.
      3. All remaining pairs → cosine distance of HSV colour histograms.

    Args:
        tracklets : list of tracklet dicts
        st_gate   : boolean mask from build_spatiotemporal_gate(), or None
                    to disable the GPS gate entirely (colour-only mode).
    """
    n = len(tracklets)
    feats = np.vstack([t["feature"] for t in tracklets])

    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    feats_normed = feats / norms

    cos_dist = cdist(feats_normed, feats_normed, metric="cosine")
    cos_dist = np.clip(cos_dist, 0.0, 2.0)

    # ── 1. Same-camera block (vectorised) ────────────────────────────────────
    cam_ids = np.array([t["cam_id"] for t in tracklets])
    same_cam_mask = cam_ids[:, None] == cam_ids[None, :]
    cos_dist[same_cam_mask] = 1.0
    np.fill_diagonal(cos_dist, 0.0)

    # ── 2. Spatio-temporal gate ───────────────────────────────────────────────
    if st_gate is not None:
        blocked = ~st_gate & ~same_cam_mask   # only apply to cross-cam pairs
        n_blocked = int(blocked.sum()) // 2
        cos_dist[blocked] = 1.0
        print(f"  [GPS] Distance matrix: {n_blocked} cross-cam pairs hard-blocked by GPS gate")
    else:
        print("  [GPS] Gate disabled — colour-only mode")

    return cos_dist.astype(np.float32)


def cluster_tracklets(tracklets, dist_threshold,
                       st_gate: np.ndarray = None):
    """
    Agglomerative clustering (average linkage) on the gated distance matrix.
    Returns an array of global IDs (1-based), one per tracklet.
    """
    if len(tracklets) == 1:
        return np.array([1])

    dist_matrix = build_distance_matrix(tracklets, st_gate=st_gate)
    condensed   = squareform(dist_matrix.astype(np.float64), checks=False)

    Z      = linkage(condensed, method="average")
    labels = fcluster(Z, t=dist_threshold, criterion="distance")
    return labels

# ─────────────────────────────────────────────
# Build output DataFrame
# ─────────────────────────────────────────────

def build_output(tracklets, global_ids, min_cams=2):
    """
    Assemble the final DataFrame in AI City eval format:
      CameraId, Id, FrameId, X, Y, Width, Height, Xworld, Yworld

    Only global IDs spanning >= min_cams cameras are included.

    NOTE: Xworld/Yworld are intentionally written as -1.
    The AI City eval script switches to world-coordinate scoring mode when
    these fields are non-(-1), which is not comparable to the baseline.
    World coordinates are used internally for the GPS gate only.
    """
    gid_to_cams = defaultdict(set)
    for t, gid in zip(tracklets, global_ids):
        gid_to_cams[gid].add(t["cam_id"])
    valid_gids = {gid for gid, cams in gid_to_cams.items() if len(cams) >= min_cams}

    frames = []
    for t, gid in zip(tracklets, global_ids):
        if gid not in valid_gids:
            continue
        df = t["df"][["frame_id", "x1", "y1", "width", "height"]].copy()
        df["CameraId"] = t["cam_id"]
        df["Id"]       = int(gid)
        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=[
            "CameraId", "Id", "FrameId", "X", "Y",
            "Width", "Height", "Xworld", "Yworld"]), gid_to_cams

    out_df = pd.concat(frames, ignore_index=True)
    out_df = out_df.rename(columns={
        "frame_id": "FrameId",
        "x1":       "X",
        "y1":       "Y",
        "width":    "Width",
        "height":   "Height",
    })
    out_df["Xworld"] = -1
    out_df["Yworld"] = -1
    out_df = out_df[["CameraId", "Id", "FrameId", "X", "Y",
                      "Width", "Height", "Xworld", "Yworld"]]
    out_df = out_df.astype({
        "CameraId": int, "Id": int, "FrameId": int,
        "X": int, "Y": int, "Width": int, "Height": int,
    })

    n_filtered = len(set(global_ids)) - len(valid_gids)
    print(f"  Kept {len(valid_gids)} global IDs spanning >={min_cams} cameras "
          f"({n_filtered} single-camera IDs discarded)")
    return out_df, gid_to_cams


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
    print(f"  Sequence        : {args.sequence}")
    print(f"  Root dir        : {args.root}")
    print(f"  Frames root     : {args.frames_root}")
    print(f"  Feature mode    : HSV colour histograms + spatio-temporal GPS gate")
    print(f"  Dist threshold  : {args.dist_threshold}")
    print(f"  Max car speed   : {args.max_speed} m/s")
    print(f"  Reproj threshold: {args.reproj_threshold} px")
    print("═" * 60 + "\n")

    # ── 1. Load MTSC data ────────────────────
    print("[1/6] Loading MTSC tracklets…")
    cameras = load_sequence(args.root, args.sequence)
    if not cameras:
        sys.exit("[ERROR] No cameras loaded. Aborting.")

    # ── 2. Load homographies ─────────────────
    print(f"\n[2/6] Loading camera homographies…")
    calib_root   = args.calib_root or args.root
    seq_dir      = os.path.join(calib_root, args.sequence)
    cam_ids      = [cam_id for cam_id, _ in cameras]
    homographies = load_homographies(seq_dir, cam_ids)

    # ── 3. Extract features + GPS positions ──
    print(f"\n[3/6] Extracting features and GPS positions…")
    tracklets, _ = build_tracklet_features(
        cameras,
        sequence=args.sequence,
        n_bins=args.n_hist_bins,
        max_samples=args.max_frames_sample,
        frames_root=args.frames_root,
        homographies=homographies,
    )

    # ── 4. Estimate world scale ──────────────
    # Done after feature extraction because tracklets now carry _H.
    # This replaces the old fixed geo_threshold with a physically meaningful
    # metres-based speed limit that auto-adapts to the dataset's coordinate system.
    print(f"\n[4/6] Estimating world-coordinate scale…")
    world_scale = 1.0

    # ── 5. Build spatio-temporal GPS gate ────
    # This is the gate that was previously defined in gps_utils.py but never
    # called. It is now the primary GPS mechanism, replacing the old static
    # geo_threshold distance check.
    print(f"\n[5/6] Building spatio-temporal GPS gate and clustering…")
    st_gate = build_spatiotemporal_gate(
        tracklets,
        max_speed_mps=args.max_speed,
        world_scale=world_scale,
        reproj_error_threshold=args.reproj_threshold,
    )

    global_ids = cluster_tracklets(
        tracklets,
        dist_threshold=args.dist_threshold,
        st_gate=st_gate,
    )
    n_global = len(set(global_ids))
    print(f"  → Assigned {n_global} global identities "
          f"from {len(tracklets)} per-camera tracklets")

    out_df, gid_to_cams = build_output(tracklets, global_ids)
    multi_cam_ids = {gid for gid, cams in gid_to_cams.items() if len(cams) > 1}
    print(f"  → IDs spanning ≥2 cameras: {len(multi_cam_ids)} "
          f"(these are the ones evaluated)")

    # ── 6. Save and evaluate ─────────────────
    print(f"\n[6/6] Writing output…")
    os.makedirs(args.output, exist_ok=True)
    out_path = os.path.join(args.output, f"{args.sequence}_mtmc.txt")
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