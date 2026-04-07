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



from gps_utils import (
    load_homographies,
    tracklet_first_world_pos,
    tracklet_last_world_pos,
    build_spatiotemporal_gate,
    row_to_world,
)
from camera_time_windows import build_timewindow_gate

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
    p.add_argument("--output", type=str, default="results",
                   help="Directory where the output .txt will be written")

    p.add_argument("--dist-threshold", type=float, default=0.10,
                   help="Cosine distance threshold for merging tracklets into the same "
                        "global identity. Lower → fewer merges (more IDs), "
                        "higher → more aggressive merging.")
    p.add_argument("--n-hist-bins", type=int, default=8,
                   help="Number of bins per HSV channel for colour histograms.")
    p.add_argument("--max-frames-sample", type=int, default=4,
                   help="Max frames to sample per tracklet when extracting colour features.")
    p.add_argument("--frames-root", type=str,
                   default="../data/AI_CITY_CHALLENGE_2022_TRAIN/train",
                   help="Root directory containing the raw frame images "
                        "<sequence>/<cam>/img1/<frame>.jpg. "
                        "Must be set to match your local dataset location.")

    # GPS arguments
    p.add_argument("--max-speed-mps", type=float, default=2.0,
                   help="Maximum physically plausible speed (in homography units/second) "
                        "between the end of one tracklet and the start of another. "
                        "Pairs implying a higher speed are hard-blocked before clustering. "
                        "Typical values: ~2.0 for pedestrians (metres/s), "
                        "~15.0 for vehicles (metres/s). Scale to your homography units.")
    p.add_argument("--min-dt-s", type=float, default=0.5,
                   help="Minimum time gap (seconds) required between two tracklets before "
                        "the speed gate is applied. Overlapping or near-simultaneous "
                        "tracklets are left unrestricted.")

    # Time-window gate (camera_time_windows.py output)
    p.add_argument("--time-windows", type=str, default=None,
                   help="Path to a time_windows.json produced by camera_time_windows.py. "
                        "When provided, a second gate is applied that hard-blocks cross-camera "
                        "pairs whose observed transit time falls outside the expected "
                        "[t_min, t_max] window for that camera pair. "
                        "Both the speed gate AND the time-window gate must pass. "
                        "If omitted, only the speed gate is used (original behaviour).")

    p.add_argument("--gt", type=str,
                   default="../data/AI_CITY_CHALLENGE_2022_TRAIN/eval/ground_truth_train.txt",
                   help="Path to ground-truth .txt file. If given, eval.py is run.")
    p.add_argument("--eval-script", type=str,
                   default="../data/AI_CITY_CHALLENGE_2022_TRAIN/eval/eval.py",
                   help="Path to the official AI City eval.py script.")
    p.add_argument("--dstype", type=str, default="train",
                   help="Dataset type passed to eval.py (train/validation/test).")
    p.add_argument("--roi-dir", type=str, default="ROIs",
                   help="ROI directory passed to eval.py.")
    return p.parse_args()

# ─────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────

def load_mtsc(txt_path: str, cam_id: int) -> pd.DataFrame:
    """
    Load a tracking file in MOT format and add camera id.
    Format: frame_id, track_id, x, y, w, h, conf, -1, -1, -1  (10 cols)
    """
    with open(txt_path, "r") as fh:
        first_line = fh.readline().strip()
    n_cols = len(first_line.split(","))
    cols = ["frame_id", "track_id", "x", "y", "w", "h", "confidence"] +            [f"_c{i}" for i in range(n_cols - 7)]

    df = pd.read_csv(txt_path, header=None, names=cols)
    df["cam_id"] = cam_id
    df["x1"] = df["x"]
    df["y1"] = df["y"]
    df["x2"] = df["x"] + df["w"]
    df["y2"] = df["y"] + df["h"]
    df["width"]  = df["w"]
    df["height"] = df["h"]
    df["timestamp"] = df["frame_id"]
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
        txt_file = os.path.join(cam_dir, r"overlap\tracks.txt")
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


def get_available_frame_ids(img_dir: str) -> set:
    """
    Scans img_dir for image files and returns a set of integer frame IDs
    that are actually available on disk.
    """
    available = set()
    for ext in ("*.jpg", "*.png"):
        for fpath in glob.glob(os.path.join(img_dir, ext)):
            stem = os.path.splitext(os.path.basename(fpath))[0]
            try:
                available.add(int(stem))
            except ValueError:
                pass
    return available


def color_histogram_feature(tracklet_df: pd.DataFrame,
                             sequence: str,
                             cam_name: str,
                             frames_root: str,
                             n_bins: int = 16,
                             max_samples: int = 10) -> np.ndarray:
    """
    Build a concatenated HSV colour histogram (H + S + V channels) by
    cropping the bounding-boxes of up to `max_samples` frames and averaging.

    Frames are read from:  <frames_root>/<sequence>/<cam>/img1/<frame>.jpg
    """
    img_dir = os.path.normpath(os.path.join(frames_root, sequence, cam_name, "img1"))
    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"[ERROR] Frames directory not found: {img_dir}")

    rows = tracklet_df.sample(min(max_samples, len(tracklet_df)),
                              random_state=0).sort_values("frame_id")

    hist_sum = None
    n_valid  = 0

    for _, row in rows.iterrows():
        fid = int(row["frame_id"])
        # Try .jpg first, then .png as fallback
        img_path = None
        for ext in (".jpg", ".png"):
            candidate = os.path.abspath(os.path.join(img_dir, f"{fid:06d}{ext}"))
            if os.path.isfile(candidate):
                img_path = candidate
                break
        if img_path is None:
            continue
        # Use cv2.imread with absolute path (more reliable than imdecode on Windows)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
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
        fids = sorted(tracklet_df["frame_id"].unique().tolist())
        print(f"  [WARN] No valid frames for cam={cam_name} track frames={fids[:5]} in {img_dir} — skipping tracklet.")
        return None

    hist_avg = hist_sum / n_valid
    norm = np.linalg.norm(hist_avg)
    if norm > 0:
        hist_avg /= norm
    return hist_avg


def build_tracklet_features(cameras, sequence, n_bins, max_samples,
                             homographies=None, frames_root=None):
    """
    Construye la lista de tracklets con:
      - feature   : histograma HSV L2-normalizado (apariencia pura, sin GPS)
      - world_first : ((wx, wy), t_start)  — primer punto GPS del tracklet
      - world_last  : ((wx, wy), t_end)    — último punto GPS del tracklet
      - cam_id, track_id, cam_name, df

    El GPS NO se fusiona en el vector de apariencia. Se almacena por separado
    para el gate espacio-temporal en build_distance_matrix.
    """
    tracklets = []
    cam_track_to_idx = {}

    _frames_root = frames_root or "../data/AI_CITY_CHALLENGE_2022_TRAIN/train"

    for cam_id, df in tqdm(cameras, desc="Building tracklet features", unit="camera"):
        cam_name = f"c{cam_id:03d}"
        H = (homographies[cam_id][0]
             if (homographies and homographies[cam_id][0] is not None)
             else None)

        # Pre-scan available frames for this camera once
        img_dir = os.path.abspath(os.path.join(_frames_root, sequence, cam_name, "img1"))
        available_frames = get_available_frame_ids(img_dir)
        print(f"  [INFO] {cam_name}: {len(available_frames)} frames available "
              f"(range {min(available_frames)}–{max(available_frames)})" if available_frames
              else f"  [WARN] {cam_name}: no frames found in {img_dir}")

        for tid, tdf in df.groupby("track_id"):
            # Filter detections to only frames that exist on disk
            tdf_valid = tdf[tdf["frame_id"].isin(available_frames)]
            if tdf_valid.empty:
                print(f"  [WARN] {cam_name} track {tid}: all {len(tdf)} detections outside available frame range — skipping.")
                continue
            feat = color_histogram_feature(
                tdf_valid, sequence, cam_name, _frames_root,
                n_bins, max_samples)
            if feat is None:
                continue  # skip tracklets with no readable frames

            # Calcular posiciones GPS de inicio y fin del tracklet
            if H is not None:
                world_first = tracklet_first_world_pos(tdf_valid, H)
                world_last  = tracklet_last_world_pos(tdf_valid, H)
            else:
                world_first = ((None, None), None)
                world_last  = ((None, None), None)

            idx = len(tracklets)
            tracklets.append({
                "cam_id":      cam_id,
                "track_id":    tid,
                "cam_name":    cam_name,
                "feature":     feat,
                "world_first": world_first,   # ((wx, wy), t_start)
                "world_last":  world_last,    # ((wx, wy), t_end)
                "df":          tdf_valid,
            })
            cam_track_to_idx[(cam_id, tid)] = idx

    print(f"\n  Total tracklets across all cameras: {len(tracklets)}")
    return tracklets, cam_track_to_idx

# ─────────────────────────────────────────────
# Cross-camera matching
# ─────────────────────────────────────────────

def build_distance_matrix(tracklets,
                           max_speed_mps: float = 2.0,
                           min_dt_s: float = 0.5,
                           time_windows_path: str = None):
    """
    Compute pairwise cosine distance between all tracklets (apariencia HSV pura).

    Blocking rules (distance forced to maximum):
      1. Same-camera pairs -> 1.0  (soft block, never merged)
      2. GPS spatio-temporal speed gate -> 2.0
             Hard-blocks pairs whose implied speed exceeds max_speed_mps.
             (physically impossible movement)
      3. Time-window gate -> 2.0  (optional, requires time_windows_path)
             Hard-blocks pairs whose observed transit time between cameras
             falls outside the [t_min, t_max] window derived from camera
             geometry and avg_speed in camera_time_windows.py.
             Both gate 2 AND gate 3 must pass for a pair to be considered.

    The GPS gates act as binary filters AFTER computing the appearance
    distance, keeping the two signals cleanly separated:
        - Appearance : do these tracklets look the same?
        - Speed gate : is the movement physically possible at all?
        - Time window: is the transit time consistent with camera layout?
    """
    n = len(tracklets)
    feats = np.vstack([t["feature"] for t in tracklets])

    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    feats_normed = feats / norms

    cos_dist = cdist(feats_normed, feats_normed, metric="cosine")
    cos_dist = np.clip(cos_dist, 0.0, 2.0)

    # Block 1: same-camera pairs (never merged in MTMC)
    for i, ti in enumerate(tracklets):
        for j, tj in enumerate(tracklets):
            if ti["cam_id"] == tj["cam_id"]:
                cos_dist[i, j] = 1.0

    # Block 2: GPS spatio-temporal speed gate
    speed_allowed = build_spatiotemporal_gate(
        tracklets,
        max_speed_mps=max_speed_mps,
        min_dt_s=min_dt_s,
    )
    cos_dist[~speed_allowed] = 2.0

    # Block 3: Time-window gate (optional).
    # A pair must survive BOTH the speed gate and the time-window gate.
    if time_windows_path is not None:
        tw_allowed = build_timewindow_gate(tracklets, time_windows_path)
        combined_blocked = ~speed_allowed | ~tw_allowed
        cos_dist[combined_blocked] = 2.0
        print(f"  [TimeWindow] Combined gate applied "
              f"({combined_blocked.sum() // 2} total pairs blocked after AND)")

    np.fill_diagonal(cos_dist, 0.0)
    return cos_dist.astype(np.float32)



def cluster_tracklets(tracklets, dist_threshold,
                      max_speed_mps: float = 2.0,
                      min_dt_s: float = 0.5,
                      time_windows_path: str = None):
    """
    Agglomerative clustering (average linkage) on the distance matrix.
    Returns an array of global IDs, one per tracklet.

    time_windows_path : optional path to a camera_time_windows.py JSON.
    When provided, the time-window gate is AND-ed with the speed gate
    inside build_distance_matrix before clustering.
    """
    if len(tracklets) == 1:
        return np.array([1])

    dist_matrix = build_distance_matrix(
        tracklets,
        max_speed_mps=max_speed_mps,
        min_dt_s=min_dt_s,
        time_windows_path=time_windows_path,
    )

    n = len(tracklets)
    condensed = []
    for i in range(n):
        for j in range(i + 1, n):
            condensed.append(dist_matrix[i, j])
    condensed = np.array(condensed, dtype=np.float64)

    Z = linkage(condensed, method="average")
    labels = fcluster(Z, t=dist_threshold, criterion="distance")
    return labels   # 1-based cluster IDs


# ─────────────────────────────────────────────
# Build output DataFrame
# ─────────────────────────────────────────────

def build_output(tracklets, global_ids, homographies=None, min_cams=2):
    """
    Assemble the final DataFrame in AI City eval format:
      CameraId, Id, FrameId, X, Y, Width, Height, Xworld, Yworld

    Xworld/Yworld are filled with real world-plane coordinates when a
    homography is available for the camera, otherwise -1.

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
        H   = (homographies[cam][0]
               if (homographies and homographies[cam][0] is not None) else None)

        for _, row in df.iterrows():
            xw, yw = row_to_world(row, H)
            records.append({
                "CameraId": cam,
                "Id":       int(gid),
                "FrameId":  int(row["frame_id"]),
                "X":        int(row["x1"]),
                "Y":        int(row["y1"]),
                "Width":    int(row["width"]),
                "Height":   int(row["height"]),
                "Xworld":   xw,
                "Yworld":   yw,
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
    print(f"  Feature mode  : HSV colour histograms (apariencia pura)")
    print(f"  GPS gate      : spatio-temporal (max_speed={args.max_speed_mps} u/s, "
          f"min_dt={args.min_dt_s} s)")
    tw_status = args.time_windows if args.time_windows else "disabled"
    print(f"  Time-win gate : {tw_status}")
    print(f"  Dist threshold: {args.dist_threshold}")
    print("=" * 60 + "\n")

    # ── Load MTSC data ───────────────────────
    print(f"[1/5] Loading MTSC tracklets for {args.sequence}…")
    cameras = load_sequence(args.root, args.sequence)
    if not cameras:
        sys.exit("[ERROR] No cameras loaded. Aborting.")

    # ── Load homographies ────────────────────
    print(f"\n[2/5] Loading GPS homographies…")
    seq_dir  = os.path.join("../data/AI_CITY_CHALLENGE_2022_TRAIN/train", args.sequence)
    cam_ids  = [cam_id for cam_id, _ in cameras]
    homographies = load_homographies(seq_dir, cam_ids)

    # ── Extract features ─────────────────────
    print(f"\n[3/5] Extracting appearance features…")
    tracklets, _ = build_tracklet_features(
        cameras,
        sequence=args.sequence,
        n_bins=args.n_hist_bins,
        max_samples=args.max_frames_sample,
        homographies=homographies,
        frames_root=args.frames_root,
    )
    # Nota: no se llama a fuse_gps_features — el GPS actúa solo como gate binario.

    # ── Cluster / Re-ID ──────────────────────
    print(f"\n[4/5] Clustering tracklets (threshold={args.dist_threshold})…")
    global_ids = cluster_tracklets(
        tracklets,
        dist_threshold=args.dist_threshold,
        max_speed_mps=args.max_speed_mps,
        min_dt_s=args.min_dt_s,
        time_windows_path=args.time_windows,
    )
    n_global = len(set(global_ids))
    print(f"  → Assigned {n_global} global identities "
          f"from {len(tracklets)} per-camera tracklets")

    gid_to_cams = defaultdict(set)
    for t, gid in zip(tracklets, global_ids):
        gid_to_cams[gid].add(t["cam_id"])
    multi_cam_ids = {gid for gid, cams in gid_to_cams.items() if len(cams) > 1}
    print(f"  → IDs spanning ≥2 cameras: {len(multi_cam_ids)} "
          f"(these are the ones evaluated)")

    # ── Save output ──────────────────────────
    print(f"\n[5/5] Writing output…")
    os.makedirs(args.output, exist_ok=True)
    out_path = os.path.join(args.output, f"{args.sequence}_mtmc.txt")
    out_df = build_output(tracklets, global_ids, homographies=homographies)
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