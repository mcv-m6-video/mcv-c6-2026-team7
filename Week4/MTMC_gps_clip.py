import os
import sys
import glob
import hashlib
import pickle
import argparse
import subprocess
import numpy as np
import pandas as pd
import cv2
from functools import lru_cache
from tqdm import tqdm
from collections import defaultdict
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import linkage, fcluster
from gps_utils import (
    load_homographies,
    tracklet_first_world_pos,
    tracklet_last_world_pos,
)

# Lazy-loaded CLIP globals (initialised on first use)
_clip_model     = None
_clip_processor = None
_clip_device    = None

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
                   default="Week4/tracking_overlap_yolov26x_fine/output_detections",
                   help="Root folder that contains S01/, S03/, …")
    p.add_argument("--output", type=str, default="Week4/results",
                   help="Directory where the output .txt will be written")

    p.add_argument("--dist-threshold", type=float, default=0.512,
                   help="Cosine distance threshold for merging tracklets into the same "
                        "global identity. Lower → fewer merges (more IDs), "
                        "higher → more aggressive merging.")
    p.add_argument("--geo-threshold", type=float, default=2000)
    p.add_argument("--calib-root", type=str, default="./AI_CITY_CHALLENGE_2022_TRAIN/train",
                   help="Root folder with calibration.txt files (same layout as --root). "
                        "Defaults to --root if not provided.")
    p.add_argument("--n-hist-bins", type=int, default=8,
                   help="Number of bins per HSV channel for colour histograms.")
    p.add_argument("--max-frames-sample", type=int, default=8,
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
    p.add_argument("--feature-mode", type=str, default="hsv",
                   choices=["hsv", "clip"],
                   help="Feature extraction mode: "
                        "'hsv' (colour histogram, default), "
                        "'clip' (CLIP ViT-L/14 Stanford-Cars, recommended threshold 0.10–0.20).")
    p.add_argument("--no-cam-whitening", action="store_true",
                   help="Disable per-camera feature whitening (subtract camera mean). "
                        "Whitening is ON by default to reduce camera-specific bias.")
    p.add_argument("--clip-min-crop-px", type=int, default=32,
                   help="Minimum crop side length (pixels) accepted for CLIP inference. "
                        "Smaller crops are skipped as they are too noisy.")
    p.add_argument("--clip-context-pad", type=float, default=0.0,
                   help="Fractional padding added around each bbox before CLIP inference. "
                        "0.4 expands the crop by 40%% on each side, giving the model "
                        "surrounding context instead of an upscaled blurry tight crop.")
    p.add_argument("--clip-batch-size", type=int, default=256,
                   help="Max crops per GPU batch during CLIP inference.")
    p.add_argument("--feature-cache-dir", type=str, default=None,
                   help="Directory to cache extracted features. "
                        "If set, features are saved/loaded as .pkl to skip re-extraction.")
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
# Frame cache (avoids re-reading the same JPEG)
# ─────────────────────────────────────────────

@lru_cache(maxsize=256)
def _read_frame(img_path: str):
    """Read and cache a frame from disk. Returns None if unreadable."""
    if not os.path.isfile(img_path):
        return None
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    return img

def _clear_frame_cache():
    _read_frame.cache_clear()

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
        fid  = int(row["frame_id"])
        img_path = os.path.join(img_dir, f"{fid:06d}.jpg")
        img = _read_frame(img_path)
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

_CLIP_MODEL_ID = "tanganke/clip-vit-large-patch14_stanford-cars"

# Map fine-tuned model IDs to their corresponding base CLIP model
_CLIP_BASE_MAP = {
    "tanganke/clip-vit-base-patch32_stanford-cars":  "openai/clip-vit-base-patch32",
    "tanganke/clip-vit-large-patch14_stanford-cars": "openai/clip-vit-large-patch14",
}

def _load_clip():
    """Load the CLIP model the first time it is needed."""
    global _clip_model, _clip_processor, _clip_device
    if _clip_model is not None:
        return
    try:
        import torch
        from transformers import CLIPModel, CLIPProcessor, CLIPVisionModel
    except ImportError:
        sys.exit("[ERROR] 'transformers' and 'torch' are required for CLIP features. "
                 "Install with: pip install transformers torch")

    import torch
    _clip_device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  [CLIP] Loading {_CLIP_MODEL_ID} on {_clip_device}…")

    base_model_id = _CLIP_BASE_MAP.get(_CLIP_MODEL_ID)
    if base_model_id is None:
        sys.exit(f"[ERROR] No base model mapping for '{_CLIP_MODEL_ID}'. "
                 f"Add an entry to _CLIP_BASE_MAP.")

    _clip_processor = CLIPProcessor.from_pretrained(base_model_id)
    _clip_model     = CLIPModel.from_pretrained(base_model_id,
                                                use_safetensors=True)
    finetuned       = CLIPVisionModel.from_pretrained(_CLIP_MODEL_ID,
                                                      use_safetensors=True)
    _clip_model.vision_model.load_state_dict(finetuned.vision_model.state_dict())
    del finetuned
    _clip_model = _clip_model.to(_clip_device)
    _clip_model.eval()
    print(f"  [CLIP] Model ready ({base_model_id} + {_CLIP_MODEL_ID}).")


def _collect_clip_crops(tracklet_df: pd.DataFrame,
                        sequence: str,
                        cam_name: str,
                        max_samples: int,
                        min_crop_px: int,
                        context_pad: float) -> list:
    """
    Collect RGB crops for a single tracklet (CPU only, no GPU).
    Returns a list of PIL Images ready for CLIP, or empty list.
    """
    from PIL import Image

    img_dir = os.path.join(FRAMES_ROOT, sequence, cam_name, "img1")
    if not os.path.isdir(img_dir):
        return []

    all_frames = tracklet_df.sort_values("frame_id")
    if len(all_frames) > max_samples:
        indices = np.linspace(0, len(all_frames) - 1, max_samples, dtype=int)
        rows = all_frames.iloc[indices]
    else:
        rows = all_frames

    crops = []
    for _, row in rows.iterrows():
        fid      = int(row["frame_id"])
        img_path = os.path.join(img_dir, f"{fid:06d}.jpg")
        img = _read_frame(img_path)
        if img is None:
            continue

        cx   = (int(row["x1"]) + int(row["x2"])) / 2
        cy   = (int(row["y1"]) + int(row["y2"])) / 2
        hw   = (int(row["x2"]) - int(row["x1"])) / 2 * (1 + context_pad)
        hh   = (int(row["y2"]) - int(row["y1"])) / 2 * (1 + context_pad)
        x1   = max(0, int(cx - hw))
        y1   = max(0, int(cy - hh))
        x2   = min(img.shape[1], int(cx + hw))
        y2   = min(img.shape[0], int(cy + hh))
        if x2 <= x1 or y2 <= y1:
            continue

        area = (x2 - x1) * (y2 - y1)
        crop_bgr = img[y1:y2, x1:x2]
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        crops.append((crop_rgb, area))

    if not crops:
        return []

    large_crops = [c for c, area in crops if area >= min_crop_px * min_crop_px]
    selected = large_crops if large_crops else [c for c, _ in crops]
    return [Image.fromarray(c) for c in selected]


def _run_clip_batched(all_pil_crops: list, tracklet_crop_counts: list,
                      batch_size: int = 64) -> list:
    """
    Run CLIP vision encoder on all crops across all tracklets in large batches.
    Uses fp16 autocast on CUDA for speed/memory.

    Returns a list of L2-normalised feature vectors (one per tracklet).
    """
    import torch

    _load_clip()

    total_crops = len(all_pil_crops)
    if total_crops == 0:
        return [None] * len(tracklet_crop_counts)

    # Encode all crops in batches
    all_embeds = []
    for start in range(0, total_crops, batch_size):
        batch_pil = all_pil_crops[start : start + batch_size]
        inputs = _clip_processor(images=batch_pil, return_tensors="pt").to(_clip_device)
        with torch.no_grad():
            if _clip_device == "cuda":
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    vision_out = _clip_model.vision_model(pixel_values=inputs["pixel_values"])
            else:
                vision_out = _clip_model.vision_model(pixel_values=inputs["pixel_values"])
            all_embeds.append(vision_out.pooler_output.float().cpu())

    all_embeds = torch.cat(all_embeds, dim=0)  # (total_crops, D)

    # Split back per tracklet and pool
    features = []
    offset = 0
    for count in tracklet_crop_counts:
        if count == 0:
            features.append(None)
            continue
        embeds = all_embeds[offset : offset + count]  # (count, D)
        offset += count

        # Adaptive pooling: mean for few crops, median for many
        if count < 4:
            pooled = embeds.mean(dim=0).numpy().astype(np.float32)
        else:
            pooled = embeds.median(dim=0).values.numpy().astype(np.float32)

        norm = np.linalg.norm(pooled)
        if norm > 0:
            pooled /= norm
        features.append(pooled)

    return features


def _feature_cache_path(cache_dir: str, sequence: str, feature_mode: str) -> str:
    """Deterministic cache file path for a (sequence, mode) pair."""
    key = f"{sequence}_{feature_mode}_{_CLIP_MODEL_ID}"
    h = hashlib.md5(key.encode()).hexdigest()[:12]
    return os.path.join(cache_dir, f"features_{sequence}_{feature_mode}_{h}.pkl")


def build_tracklet_features(cameras, sequence, n_bins, max_samples,
                             homographies: dict = None,
                             feature_mode: str = "hsv",
                             clip_min_crop_px: int = 32,
                             clip_context_pad: float = 0.4,
                             clip_batch_size: int = 64,
                             feature_cache_dir: str = None):
    """
    Returns:
      tracklets : list of dicts with keys
                  cam_id, track_id, cam_name, feature, df,
                  world_first, world_last
      cam_track_to_idx : dict (cam_id, track_id) → index in tracklets

    feature_mode : "hsv"  → concatenated HSV colour histograms (default)
                   "clip" → batched CLIP ViT-L/14 pooler_output (Stanford-Cars)
    """
    # ── Check feature cache ─────────────────────
    cache_path = None
    cached_features = None
    if feature_cache_dir:
        os.makedirs(feature_cache_dir, exist_ok=True)
        cache_path = _feature_cache_path(feature_cache_dir, sequence, feature_mode)
        if os.path.isfile(cache_path):
            print(f"  [Cache] Loading features from {cache_path}")
            with open(cache_path, "rb") as f:
                cached_features = pickle.load(f)

    tracklets = []
    cam_track_to_idx = {}

    # First pass: build tracklet metadata + collect crops for CLIP
    clip_crop_jobs = []  # list of (tracklet_index, pil_crops)
    all_pil_crops = []
    tracklet_crop_counts = []

    for cam_id, df in tqdm(cameras, desc="Building tracklet features", unit="camera"):
        cam_name = f"c{cam_id:03d}"

        H = None
        if homographies and cam_id in homographies:
            H, _ = homographies[cam_id]

        for tid, tdf in df.groupby("track_id"):
            world_first = tracklet_first_world_pos(tdf, H) if H is not None else None
            world_last  = tracklet_last_world_pos(tdf, H)  if H is not None else None

            idx = len(tracklets)
            tracklets.append({
                "cam_id":      cam_id,
                "track_id":    tid,
                "cam_name":    cam_name,
                "feature":     None,  # filled below
                "df":          tdf,
                "world_first": world_first,
                "world_last":  world_last,
            })
            cam_track_to_idx[(cam_id, tid)] = idx

            cache_key = (cam_id, tid)

            if feature_mode == "hsv":
                if cached_features and cache_key in cached_features:
                    tracklets[idx]["feature"] = cached_features[cache_key]
                else:
                    tracklets[idx]["feature"] = color_histogram_feature(
                        tdf, sequence, cam_name, n_bins, max_samples)
            elif feature_mode == "clip":
                if cached_features and cache_key in cached_features:
                    tracklets[idx]["feature"] = cached_features[cache_key]
                    tracklet_crop_counts.append(0)  # no crops needed
                else:
                    pil_crops = _collect_clip_crops(
                        tdf, sequence, cam_name, max_samples,
                        clip_min_crop_px, clip_context_pad)
                    clip_crop_jobs.append(idx)
                    all_pil_crops.extend(pil_crops)
                    tracklet_crop_counts.append(len(pil_crops))

    # ── Batched CLIP inference ──────────────────
    if feature_mode == "clip" and all_pil_crops:
        print(f"  [CLIP] Running batched inference on {len(all_pil_crops)} crops "
              f"across {len(clip_crop_jobs)} tracklets (batch_size={clip_batch_size})…")
        clip_features = _run_clip_batched(all_pil_crops, tracklet_crop_counts,
                                          batch_size=clip_batch_size)
        job_idx = 0
        for i, count in enumerate(tracklet_crop_counts):
            if count == 0:
                continue
            feat = clip_features[i]
            tidx = clip_crop_jobs[job_idx]
            job_idx += 1
            if feat is None:
                raise ValueError(
                    f"[ERROR] No valid crops for tracklet "
                    f"cam={tracklets[tidx]['cam_id']} track={tracklets[tidx]['track_id']}")
            tracklets[tidx]["feature"] = feat

    # Free frame cache memory
    _clear_frame_cache()

    # ── Save feature cache ──────────────────────
    if cache_path:
        features_to_save = {
            (t["cam_id"], t["track_id"]): t["feature"]
            for t in tracklets
        }
        with open(cache_path, "wb") as f:
            pickle.dump(features_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  [Cache] Saved features to {cache_path}")

    print(f"\n  Total tracklets across all cameras: {len(tracklets)}")
    n_with_gps = sum(1 for t in tracklets if t["world_first"] is not None)
    print(f"  Tracklets with GPS world positions: {n_with_gps}/{len(tracklets)}")
    return tracklets, cam_track_to_idx

# ─────────────────────────────────────────────
# Per-camera feature whitening
# ─────────────────────────────────────────────

def apply_camera_whitening(tracklets: list) -> list:
    """
    Subtract the per-camera mean feature vector from every tracklet in that
    camera, then re-L2-normalise.
    """
    cam_indices = defaultdict(list)
    for i, t in enumerate(tracklets):
        cam_indices[t["cam_id"]].append(i)

    for cam_id, idxs in cam_indices.items():
        feats = np.vstack([tracklets[i]["feature"] for i in idxs])  # (K, D)
        cam_mean = feats.mean(axis=0)                                # (D,)
        for i in idxs:
            centred = tracklets[i]["feature"] - cam_mean
            norm = np.linalg.norm(centred)
            tracklets[i]["feature"] = centred / norm if norm > 0 else centred

    print(f"  [Whitening] Applied per-camera mean subtraction to "
          f"{len(cam_indices)} cameras.")
    return tracklets


# ─────────────────────────────────────────────
# Cross-camera matching
# ─────────────────────────────────────────────

def build_distance_matrix(tracklets, geo_threshold=None):
    """
    Compute pairwise cosine distance between all tracklets.
    Same-camera pairs and GPS-gated pairs are blocked (distance = 1.0).
    """
    n = len(tracklets)
    feats = np.vstack([t["feature"] for t in tracklets])

    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    feats_normed = feats / norms

    cos_dist = cdist(feats_normed, feats_normed, metric="cosine")
    cos_dist = np.clip(cos_dist, 0.0, 2.0)

    # ── Same-camera blocking (vectorised) ────────────────────────────────
    cam_ids = np.array([t["cam_id"] for t in tracklets])
    same_cam = cam_ids[:, None] == cam_ids[None, :]
    cos_dist[same_cam] = 1.0
    np.fill_diagonal(cos_dist, 0.0)

    # ── GPS hard gate (vectorised) ───────────────────────────────────────
    n_blocked_geo = 0
    if geo_threshold is not None:
        # Build arrays of world positions for exit/entry
        # NaN for tracklets without GPS data
        exit_pos = np.full((n, 2), np.nan)
        entry_pos = np.full((n, 2), np.nan)
        exit_time = np.full(n, np.nan)

        for i, t in enumerate(tracklets):
            if t["world_last"] is not None:
                (wx, wy), ts = t["world_last"]
                exit_pos[i] = [wx, wy]
                exit_time[i] = ts
            if t["world_first"] is not None:
                (wx, wy), _ = t["world_first"]
                entry_pos[i] = [wx, wy]

        has_gps = ~np.isnan(exit_pos[:, 0]) & ~np.isnan(entry_pos[:, 0])
        # Only check pairs where both have GPS
        gps_idx = np.where(has_gps)[0]

        if len(gps_idx) > 1:
            # For each pair (i, j): compare exit of earlier to entry of later
            for ii in range(len(gps_idx)):
                i = gps_idx[ii]
                for jj in range(ii + 1, len(gps_idx)):
                    j = gps_idx[jj]
                    if cam_ids[i] == cam_ids[j]:
                        continue
                    # Determine which exits first
                    if exit_time[i] <= exit_time[j]:
                        dx = exit_pos[i, 0] - entry_pos[j, 0]
                        dy = exit_pos[i, 1] - entry_pos[j, 1]
                    else:
                        dx = exit_pos[j, 0] - entry_pos[i, 0]
                        dy = exit_pos[j, 1] - entry_pos[i, 1]

                    if dx * dx + dy * dy > geo_threshold * geo_threshold:
                        cos_dist[i, j] = 1.0
                        cos_dist[j, i] = 1.0
                        n_blocked_geo += 1

        print(f"  [GPS] AND-gate blocked {n_blocked_geo} cross-cam pairs "
              f"(world_dist > {geo_threshold:.0f})")
    else:
        print(f"  [GPS] Gate disabled — colour-only mode")

    return cos_dist.astype(np.float32)


def cluster_tracklets(tracklets, dist_threshold, geo_threshold=None):
    """
    Agglomerative clustering (average linkage) on the distance matrix.
    Returns an array of global IDs (1-based), one per tracklet.
    """
    if len(tracklets) == 1:
        return np.array([1])

    dist_matrix = build_distance_matrix(tracklets, geo_threshold=geo_threshold)

    # Extract upper triangle as condensed distance (vectorised)
    n = len(tracklets)
    iu = np.triu_indices(n, k=1)
    condensed = dist_matrix[iu].astype(np.float64)

    Z      = linkage(condensed, method="average")
    labels = fcluster(Z, t=dist_threshold, criterion="distance")
    return labels

# ─────────────────────────────────────────────
# Build output DataFrame
# ─────────────────────────────────────────────

def build_output(tracklets, global_ids, homographies: dict = None, min_cams=2):
    """
    Assemble the final DataFrame in AI City eval format.
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
    feat_labels = {"hsv": "HSV colour histograms",
                   "clip": "CLIP ViT-L/14 Stanford-Cars"}
    print(f"  Feature mode  : {feat_labels.get(args.feature_mode, args.feature_mode)} + GPS AND-gate")
    print(f"  Cam whitening : {'OFF' if args.no_cam_whitening else 'ON'}")
    print(f"  Dist threshold: {args.dist_threshold}")
    print(f"  Geo threshold : {args.geo_threshold}  (None = colour-only)")
    if args.feature_cache_dir:
        print(f"  Feature cache : {args.feature_cache_dir}")
    print("═" * 60 + "\n")

    # ── Load MTSC data ───────────────────────
    print(f"[1/5] Loading MTSC tracklets for {args.sequence}…")
    cameras = load_sequence(args.root, args.sequence)
    if not cameras:
        sys.exit("[ERROR] No cameras loaded. Aborting.")

    # ── Load homographies ────────────────────
    print(f"\n[2/5] Loading camera homographies…")
    calib_root = args.calib_root if args.calib_root else args.root
    seq_dir    = os.path.join(calib_root, args.sequence)
    cam_ids    = [cam_id for cam_id, _ in cameras]
    homographies = load_homographies(seq_dir, cam_ids)

    # ── Extract features ─────────────────────
    print(f"\n[3/5] Extracting features…")
    tracklets, _ = build_tracklet_features(
        cameras,
        sequence=args.sequence,
        n_bins=args.n_hist_bins,
        max_samples=args.max_frames_sample,
        homographies=homographies,
        feature_mode=args.feature_mode,
        clip_min_crop_px=args.clip_min_crop_px,
        clip_context_pad=args.clip_context_pad,
        clip_batch_size=args.clip_batch_size,
        feature_cache_dir=args.feature_cache_dir,
    )

    # ── Per-camera whitening ─────────────────
    if not args.no_cam_whitening:
        apply_camera_whitening(tracklets)

    # ── Cluster / Re-ID ──────────────────────
    print(f"\n[4/5] Clustering tracklets "
          f"(dist_threshold={args.dist_threshold}, "
          f"geo_threshold={args.geo_threshold})…")
    global_ids = cluster_tracklets(
        tracklets,
        dist_threshold=args.dist_threshold,
        geo_threshold=args.geo_threshold,
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
