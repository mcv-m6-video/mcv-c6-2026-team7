"""
debug_clip.py — Diagnose why CLIP features score lower than HSV for MTMC Re-ID.

What this script does
─────────────────────
For a given sequence it:
  1. Loads per-camera MTSC tracklets (same as MTMC_gps.py).
  2. Matches each predicted tracklet to a ground-truth global ID by majority vote
     on overlapping detections (frame + IoU).
  3. Extracts HSV, CLIP, and clip+hsv features for every tracklet.
  4. Builds the full set of cross-camera positive pairs (same GT id, diff cam)
     and negative pairs (different GT id, diff cam), then reports:
       a) Distance distributions (mean / median / std) for positives vs negatives
       b) ROC-AUC and the Equal Error Rate threshold for each feature mode
       c) Histogram plots saved to --output-dir
  5. Prints a per-tracklet table of the hardest positive pairs (GT same vehicle
     but high CLIP distance) so you can inspect why.
  6. Saves crops of the hardest positive pairs as image grids.

Usage
─────
  python Week4/debug_clip.py S01 \\
      --root   Week4/tracking_overlap_yolov26x_base/output_detections \\
      --gt     ./AI_CITY_CHALLENGE_2022_TRAIN/eval/ground_truth_train.txt \\
      --output-dir Week4/debug_clip_out
"""

import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use("Agg")          # headless — no display needed
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
from scipy.spatial.distance import cosine as cosine_dist

# ── reuse feature extractors from MTMC_gps ──────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from MTMC_gps import (
    load_sequence,
    color_histogram_feature,
    clip_feature as _clip_feature_normalized,
    apply_camera_whitening,
    FRAMES_ROOT,
)

def clip_feature(tracklet_df, sequence, cam_name, max_samples=10,
                 min_crop_px=32, context_pad=0.4):
    """Like MTMC_gps.clip_feature but WITHOUT L2 normalisation."""
    import torch
    from PIL import Image
    import MTMC_gps

    MTMC_gps._load_clip()
    _clip_model     = MTMC_gps._clip_model
    _clip_processor = MTMC_gps._clip_processor
    _clip_device    = MTMC_gps._clip_device

    img_dir = os.path.join(FRAMES_ROOT, sequence, cam_name, "img1")
    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"[ERROR] Frames directory not found: {img_dir}")

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
        if not os.path.isfile(img_path):
            continue
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        cx  = (int(row["x1"]) + int(row["x2"])) / 2
        cy  = (int(row["y1"]) + int(row["y2"])) / 2
        hw  = (int(row["x2"]) - int(row["x1"])) / 2 * (1 + context_pad)
        hh  = (int(row["y2"]) - int(row["y1"])) / 2 * (1 + context_pad)
        x1  = max(0, int(cx - hw)); y1 = max(0, int(cy - hh))
        x2  = min(img.shape[1], int(cx + hw)); y2 = min(img.shape[0], int(cy + hh))
        if x2 <= x1 or y2 <= y1:
            continue
        crop_bgr = img[y1:y2, x1:x2]
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        crops.append((crop_rgb, (x2 - x1) * (y2 - y1)))

    if not crops:
        raise ValueError("[ERROR] No valid frames found for tracklet; cannot extract CLIP features.")

    large_crops = [c for c, area in crops if area >= min_crop_px * min_crop_px]
    selected    = large_crops if large_crops else [c for c, _ in crops]
    pil_crops   = [Image.fromarray(c) for c in selected]

    with torch.no_grad():
        inputs    = _clip_processor(images=pil_crops, return_tensors="pt").to(_clip_device)
        vision_out = _clip_model.vision_model(pixel_values=inputs["pixel_values"])
        embeds    = vision_out.pooler_output   # (N, 768)

    # Median-pool — no L2 normalisation
    return embeds.median(dim=0).values.cpu().numpy().astype(np.float32)

# ─────────────────────────────────────────────────────────────────────────────
# Vehicle Re-ID model  (OSNet-AIN, torchreid)
#
# OSNet-AIN (Omni-Scale Network with Adaptive Instance Normalisation) is a
# lightweight re-ID backbone specifically designed for cross-camera matching.
# We use ImageNet-pretrained weights whose Google Drive links are publicly
# accessible and download automatically via gdown on first run.
#
# Weight files are cached in ~/.cache/torchreid/.
# ─────────────────────────────────────────────────────────────────────────────

_reid_model     = None
_reid_transform = None
_reid_device    = None
_reid_failed    = False   # set to True after first load failure — stops retrying

_OSNET_CACHE    = os.path.expanduser("~/.cache/torchreid")
_OSNET_IMAGENET = os.path.join(_OSNET_CACHE, "osnet_ain_x1_0_imagenet.pth")
_OSNET_IMAGENET_URL = "https://drive.google.com/uc?id=1-CaioD9NaqbHK_kzSMW8VE4_3KcsRjEo"


def _load_reid():
    """
    Load OSNet-AIN as a re-ID feature extractor.

    Strategy (tries in order, uses first that succeeds):
      1. OSNet-AIN with ImageNet pretrained weights (torchreid).
         The Drive link for these weights is publicly accessible and downloads
         reliably via gdown.  Weights are cached in ~/.cache/torchreid/.
      2. ResNet-50 ImageNet (torchvision) — always available, no extra download.
    """
    global _reid_model, _reid_transform, _reid_device, _reid_failed
    if _reid_model is not None or _reid_failed:
        return

    import sys, types, torch, torch.nn as nn
    from torchvision import transforms

    _reid_device = "cuda" if torch.cuda.is_available() else "cpu"

    # Stub tensorboard so torchreid.engine imports don't crash
    if "tensorboard" not in sys.modules:
        _tb = types.ModuleType("tensorboard")
        _tb.version = types.SimpleNamespace(VERSION="2.0.0")
        sys.modules["tensorboard"] = _tb
        sys.modules["tensorboard.version"] = _tb.version
    if "torch.utils.tensorboard" not in sys.modules:
        sys.modules["torch.utils.tensorboard"] = types.SimpleNamespace(SummaryWriter=None)

    # ── attempt 1: OSNet-AIN ImageNet (torchreid) ───────────────────────────
    osnet_loaded = False
    try:
        import torchreid
        from torchreid.reid.utils import load_pretrained_weights

        os.makedirs(_OSNET_CACHE, exist_ok=True)
        if not os.path.isfile(_OSNET_IMAGENET):
            print("  [ReID] Downloading OSNet-AIN ImageNet weights…")
            import gdown
            gdown.download(_OSNET_IMAGENET_URL, _OSNET_IMAGENET, quiet=False)

        m = torchreid.models.build_model(
            name="osnet_ain_x1_0", num_classes=1000, pretrained=False)
        load_pretrained_weights(m, _OSNET_IMAGENET)
        m.classifier = nn.Identity()   # drop head → 512-d features
        _reid_model  = m.to(_reid_device).eval()
        osnet_loaded = True
        print(f"  [ReID] OSNet-AIN (ImageNet) ready on {_reid_device}.")
    except Exception as e:
        print(f"  [ReID] OSNet-AIN unavailable ({e}); falling back to ResNet-50.")

    # ── attempt 2: plain ResNet-50 (ImageNet) from torchvision ─────────────
    if not osnet_loaded:
        try:
            import torchvision.models as tvm
            backbone = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V1)
            backbone.fc = nn.Identity()
            _reid_model = backbone.to(_reid_device).eval()
            print(f"  [ReID] ResNet-50 (ImageNet, fallback) ready on {_reid_device}.")
        except Exception as e:
            _reid_failed = True
            print(f"  [ReID] All re-ID models failed to load: {e}")
            return

    _reid_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 128)),     # OSNet standard input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def reid_feature(tracklet_df: pd.DataFrame,
                 sequence: str,
                 cam_name: str,
                 max_samples: int = 4,
                 min_crop_px: int = 32) -> np.ndarray:
    """
    Extract an OSNet-AIN re-ID embedding for a tracklet:
      1. Uniformly sample up to max_samples frames.
      2. Crop tight bounding boxes; skip crops smaller than min_crop_px.
      3. Run the OSNet-AIN encoder in eval mode → 512-d penultimate features.
      4. Mean-pool per-frame embeddings and L2-normalise.
    """
    import torch
    _load_reid()
    if _reid_model is None:
        raise RuntimeError("Re-ID model unavailable (load failed at startup).")

    img_dir = os.path.join(FRAMES_ROOT, sequence, cam_name, "img1")
    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"Frames dir not found: {img_dir}")

    all_frames = tracklet_df.sort_values("frame_id")
    if len(all_frames) > max_samples:
        idxs = np.linspace(0, len(all_frames) - 1, max_samples, dtype=int)
        rows = all_frames.iloc[idxs]
    else:
        rows = all_frames

    crops_all, crops_large = [], []
    for _, row in rows.iterrows():
        fid      = int(row["frame_id"])
        img_path = os.path.join(img_dir, f"{fid:06d}.jpg")
        if not os.path.isfile(img_path):
            continue
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        x1 = max(0, int(row["x1"])); y1 = max(0, int(row["y1"]))
        x2 = min(img.shape[1], int(row["x2"])); y2 = min(img.shape[0], int(row["y2"]))
        if x2 <= x1 or y2 <= y1:
            continue
        crop = img[y1:y2, x1:x2]
        if crop.shape[0] < 2 or crop.shape[1] < 2:
            continue
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crops_all.append(crop_rgb)
        if crop.shape[0] >= min_crop_px and crop.shape[1] >= min_crop_px:
            crops_large.append(crop_rgb)

    selected = crops_large if crops_large else crops_all
    if not selected:
        raise ValueError("No valid crops for re-ID feature extraction.")
    tensors = [_reid_transform(c) for c in selected]

    batch = torch.stack(tensors).to(_reid_device)
    with torch.no_grad():
        feats = _reid_model(batch)   # (N, 512) in eval mode
    feat = feats.mean(dim=0).cpu().numpy().astype(np.float32)
    return feat

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Debug CLIP vs HSV Re-ID quality",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("sequence")
    p.add_argument("--root", default="Week4/tracking_overlap_yolov26x_base/output_detections")
    p.add_argument("--gt",   default="./AI_CITY_CHALLENGE_2022_TRAIN/eval/ground_truth_train.txt")
    p.add_argument("--output-dir", default="Week4/debug_clip_out")
    p.add_argument("--n-hist-bins", type=int, default=8)
    p.add_argument("--max-frames-sample", type=int, default=4)
    p.add_argument("--clip-min-crop-px", type=int, default=32)
    p.add_argument("--clip-context-pad", type=float, default=0.4,
                   help="Fractional bbox expansion for CLIP crops (0 = tight crop).")
    p.add_argument("--max-pairs", type=int, default=5000,
                   help="Cap on random negative pairs to keep (positives always kept all).")
    p.add_argument("--n-hard-examples", type=int, default=10,
                   help="Number of hardest positive pairs to visualise as image grids.")
    p.add_argument("--shared-frame-id", type=int, default=None,
                   help="Specific frame number to use for the shared-frame panel. "
                        "If omitted the median frame shared by all cameras is used.")
    p.add_argument("--no-cam-whitening", action="store_true")
    p.add_argument("--no-reid", action="store_true",
                   help="Skip vehicle re-ID model (use if torchreid not installed).")
    return p.parse_args()

# ─────────────────────────────────────────────────────────────────────────────
# Ground-truth loader
# ─────────────────────────────────────────────────────────────────────────────

GT_COLS = ["cam_id", "global_id", "frame_id", "x", "y", "width", "height", "xw", "yw"]

def load_gt(gt_path: str, sequence: str) -> pd.DataFrame:
    """
    Load GT and filter to the requested sequence.
    The GT file has no header and uses 1-based camera IDs; camera IDs for a
    sequence (e.g. S01) run from c001 to c00N.
    We rely on the caller supplying camera IDs present in the MTSC data.
    """
    df = pd.read_csv(gt_path, sep=" ", header=None, names=GT_COLS)
    return df

# ─────────────────────────────────────────────────────────────────────────────
# Match predicted tracklets to GT global IDs
# ─────────────────────────────────────────────────────────────────────────────

def iou_1d(ax1, ax2, bx1, bx2):
    inter = max(0, min(ax2, bx2) - max(ax1, bx1))
    union = max(ax2, ax1) - min(ax1, bx1) + max(bx2, bx1) - min(bx1, bx2) - inter
    return inter / union if union > 0 else 0.0

def bbox_iou(r1, r2):
    ix1 = max(r1["x1"], r2["x"])
    iy1 = max(r1["y1"], r2["y"])
    ix2 = min(r1["x2"], r2["x"] + r2["width"])
    iy2 = min(r1["y2"], r2["y"] + r2["height"])
    iw  = max(0, ix2 - ix1)
    ih  = max(0, iy2 - iy1)
    inter = iw * ih
    a1 = (r1["x2"] - r1["x1"]) * (r1["y2"] - r1["y1"])
    a2 = r2["width"] * r2["height"]
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0

def assign_gt_ids(cameras, gt_df, iou_thresh=0.4):
    """
    For each predicted tracklet (cam_id, track_id) assign the GT global_id
    whose detections overlap the most frames/IoU with the predicted track.
    Returns dict (cam_id, track_id) → gt_global_id  (or None if no match).
    """
    # Build GT lookup: cam_id → frame_id → list of GT rows
    gt_by_cam_frame = defaultdict(lambda: defaultdict(list))
    for _, row in gt_df.iterrows():
        gt_by_cam_frame[int(row.cam_id)][int(row.frame_id)].append(row)

    assignment = {}
    for cam_id, df in cameras:
        for tid, tdf in df.groupby("track_id"):
            votes = defaultdict(int)
            for _, det in tdf.iterrows():
                fid = int(det.frame_id)
                gt_rows = gt_by_cam_frame[cam_id].get(fid, [])
                for gt_row in gt_rows:
                    iou = bbox_iou(det, gt_row)
                    if iou >= iou_thresh:
                        votes[int(gt_row.global_id)] += 1
            if votes:
                assignment[(cam_id, tid)] = max(votes, key=votes.get)
            else:
                assignment[(cam_id, tid)] = None
    return assignment

# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction for all tracklets
# ─────────────────────────────────────────────────────────────────────────────

def extract_all_features(cameras, sequence, n_bins, max_samples, clip_min_crop_px,
                          context_pad=0.4, use_reid=True):
    """
    Returns a list of dicts:
      cam_id, track_id, feat_hsv, feat_clip, feat_fused, feat_reid
    feat_reid is None when use_reid=False or extraction fails.
    """
    records = []
    for cam_id, df in tqdm(cameras, desc="Extracting features", unit="cam"):
        cam_name = f"c{cam_id:03d}"
        for tid, tdf in df.groupby("track_id"):
            try:
                f_hsv = color_histogram_feature(
                    tdf, sequence, cam_name, n_bins, max_samples)
            except Exception as e:
                print(f"  [WARN] HSV failed cam={cam_id} tid={tid}: {e}")
                f_hsv = None

            try:
                f_clip = clip_feature(
                    tdf, sequence, cam_name, max_samples, clip_min_crop_px,
                    context_pad=context_pad)
            except Exception as e:
                print(f"  [WARN] CLIP failed cam={cam_id} tid={tid}: {e}")
                f_clip = None

            if f_hsv is not None and f_clip is not None:
                f_fused = np.concatenate([f_clip, f_hsv]).astype(np.float32)
            else:
                f_fused = None

            f_reid = None
            if use_reid:
                try:
                    f_reid = reid_feature(tdf, sequence, cam_name, max_samples,
                                          min_crop_px=clip_min_crop_px)
                except Exception as e:
                    print(f"  [WARN] ReID failed cam={cam_id} tid={tid}: {e}")

            records.append({
                "cam_id":     cam_id,
                "track_id":   tid,
                "df":         tdf,
                "cam_name":   cam_name,
                "feat_hsv":   f_hsv,
                "feat_clip":  f_clip,
                "feat_fused": f_fused,
                "feat_reid":  f_reid,
            })
    return records

# ─────────────────────────────────────────────────────────────────────────────
# Pair analysis
# ─────────────────────────────────────────────────────────────────────────────

def build_pairs(records, assignment, max_neg_pairs):
    """
    Returns two lists of (key_i, key_j, d_hsv, d_clip, d_fused, d_reid):
      positives — same GT global_id, different cameras
      negatives — different GT global_id, different cameras (capped)
    d_reid / d_fused are None when the feature was not extracted.
    """
    by_key = {(r["cam_id"], r["track_id"]): r for r in records}

    keys   = list(by_key.keys())
    gt_ids = [assignment.get(k) for k in keys]
    n      = len(keys)

    positives, negatives = [], []

    for i in range(n):
        ri   = by_key[keys[i]]
        gt_i = gt_ids[i]
        if gt_i is None:
            continue
        if ri["feat_hsv"] is None or ri["feat_clip"] is None:
            continue

        for j in range(i + 1, n):
            rj   = by_key[keys[j]]
            gt_j = gt_ids[j]
            if gt_j is None:
                continue
            if rj["feat_hsv"] is None or rj["feat_clip"] is None:
                continue
            if ri["cam_id"] == rj["cam_id"]:
                continue   # same camera — not relevant for MTMC

            d_hsv   = float(cosine_dist(ri["feat_hsv"],  rj["feat_hsv"]))
            d_clip  = float(cosine_dist(ri["feat_clip"], rj["feat_clip"]))
            d_fused = float(cosine_dist(ri["feat_fused"], rj["feat_fused"])) \
                      if ri["feat_fused"] is not None and rj["feat_fused"] is not None \
                      else None
            d_reid  = float(cosine_dist(ri["feat_reid"], rj["feat_reid"])) \
                      if ri["feat_reid"] is not None and rj["feat_reid"] is not None \
                      else None

            entry = (keys[i], keys[j], d_hsv, d_clip, d_fused, d_reid)
            if gt_i == gt_j:
                positives.append(entry)
            else:
                negatives.append(entry)

    # Cap negatives to avoid memory issues
    rng = np.random.default_rng(0)
    if len(negatives) > max_neg_pairs:
        idx = rng.choice(len(negatives), max_neg_pairs, replace=False)
        negatives = [negatives[k] for k in idx]

    return positives, negatives

# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def roc_auc(pos_dists, neg_dists):
    from sklearn.metrics import roc_auc_score
    y_true  = np.array([1] * len(pos_dists) + [0] * len(neg_dists))
    # lower distance = more similar = positive → negate for AUC convention
    y_score = -np.array(pos_dists + neg_dists)
    return roc_auc_score(y_true, y_score)

def eer_threshold(pos_dists, neg_dists):
    """Equal Error Rate threshold (FPR ≈ FNR)."""
    thresholds = np.linspace(0, 1, 500)
    best_t, best_diff = 0.5, 1.0
    for t in thresholds:
        fnr = np.mean(np.array(pos_dists) > t)   # positives wrongly rejected
        fpr = np.mean(np.array(neg_dists) <= t)   # negatives wrongly accepted
        diff = abs(fnr - fpr)
        if diff < best_diff:
            best_diff = diff
            best_t    = t
    eer = (np.mean(np.array(pos_dists) > best_t) +
           np.mean(np.array(neg_dists) <= best_t)) / 2
    return best_t, eer

def print_stats(name, pos_dists, neg_dists):
    pos = np.array(pos_dists)
    neg = np.array(neg_dists)
    try:
        auc = roc_auc(pos_dists, neg_dists)
        t_eer, eer = eer_threshold(pos_dists, neg_dists)
        auc_str = f"{auc:.4f}"
        eer_str = f"EER={eer:.3f} @ threshold={t_eer:.3f}"
    except Exception:
        auc_str = "N/A (sklearn missing?)"
        eer_str = ""

    print(f"\n  [{name}]")
    print(f"    Positives ({len(pos):>5d}): mean={pos.mean():.4f}  "
          f"median={np.median(pos):.4f}  std={pos.std():.4f}  "
          f"max={pos.max():.4f}")
    print(f"    Negatives ({len(neg):>5d}): mean={neg.mean():.4f}  "
          f"median={np.median(neg):.4f}  std={neg.std():.4f}  "
          f"max={neg.max():.4f}")
    print(f"    Separability gap : {neg.mean() - pos.mean():.4f}  "
          f"(higher = better)")
    print(f"    ROC-AUC          : {auc_str}")
    if eer_str:
        print(f"    {eer_str}")

# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_distributions(pos_dists, neg_dists, name, out_dir):
    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.linspace(0, 1, 60)
    ax.hist(neg_dists, bins=bins, alpha=0.5, label="Negatives (diff ID)", color="tab:red",  density=True)
    ax.hist(pos_dists, bins=bins, alpha=0.5, label="Positives (same ID)", color="tab:blue", density=True)
    ax.set_xlabel("Cosine distance")
    ax.set_ylabel("Density")
    ax.set_title(f"Distance distribution — {name}")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(out_dir, f"dist_{name.replace('+','_').lower()}.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  Saved → {path}")


def plot_roc(methods, out_dir):
    try:
        from sklearn.metrics import roc_curve
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(6, 6))
    for name, pos_d, neg_d in methods:
        y_true  = np.array([1]*len(pos_d) + [0]*len(neg_d))
        y_score = -np.array(pos_d + neg_d)
        fpr, tpr, _ = roc_curve(y_true, y_score)
        ax.plot(fpr, tpr, label=name)
    ax.plot([0,1],[0,1],"k--", linewidth=0.8)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title("ROC curves")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(out_dir, "roc_curves.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  Saved → {path}")

# ─────────────────────────────────────────────────────────────────────────────
# Hard-example visualisation
# ─────────────────────────────────────────────────────────────────────────────

def load_first_crop(tdf, sequence, cam_name, target_h=128):
    """Return the first readable vehicle crop rescaled to target_h pixels tall."""
    img_dir = os.path.join(FRAMES_ROOT, sequence, cam_name, "img1")
    for _, row in tdf.sort_values("frame_id").iterrows():
        fid = int(row["frame_id"])
        path = os.path.join(img_dir, f"{fid:06d}.jpg")
        if not os.path.isfile(path):
            continue
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        x1 = max(0, int(row["x1"])); y1 = max(0, int(row["y1"]))
        x2 = min(img.shape[1], int(row["x2"])); y2 = min(img.shape[0], int(row["y2"]))
        if x2 <= x1 or y2 <= y1:
            continue
        crop = img[y1:y2, x1:x2]
        h, w = crop.shape[:2]
        if h == 0 or w == 0:
            continue
        scale = target_h / h
        crop  = cv2.resize(crop, (max(1, int(w * scale)), target_h))
        return crop
    return np.zeros((target_h, target_h, 3), dtype=np.uint8)


def save_hard_examples(hard_pairs, records_by_key, sequence, assignment, out_dir, n=10):
    """Save a grid image for each of the N hardest positive pairs (highest CLIP dist)."""
    hard_pairs = sorted(hard_pairs, key=lambda x: x[3], reverse=True)[:n]
    for rank, pair in enumerate(hard_pairs):
        ki, kj, d_hsv, d_clip, d_fused, d_reid = pair
        ri = records_by_key[ki]
        rj = records_by_key[kj]
        crop_i = load_first_crop(ri["df"], sequence, ri["cam_name"])
        crop_j = load_first_crop(rj["df"], sequence, rj["cam_name"])

        h = max(crop_i.shape[0], crop_j.shape[0])
        def pad(c):
            ph = h - c.shape[0]
            return np.pad(c, ((0, ph), (0, 0), (0, 0)))
        grid = np.hstack([pad(crop_i), np.ones((h, 4, 3), dtype=np.uint8) * 200, pad(crop_j)])
        grid = cv2.cvtColor(grid, cv2.COLOR_BGR2RGB)

        gt_id    = assignment.get(ki, "?")
        reid_str = f"  d_reid={d_reid:.3f}" if d_reid is not None else ""
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.imshow(grid)
        ax.axis("off")
        ax.set_title(
            f"Hard positive #{rank+1}  GT_id={gt_id}\n"
            f"cam{ki[0]}/tid{ki[1]}  ←→  cam{kj[0]}/tid{kj[1]}\n"
            f"d_hsv={d_hsv:.3f}  d_clip={d_clip:.3f}{reid_str}",
            fontsize=8,
        )
        fig.tight_layout()
        path = os.path.join(out_dir, f"hard_pos_{rank+1:02d}.png")
        fig.savefig(path, dpi=120)
        plt.close(fig)
    print(f"  Saved {min(n, len(hard_pairs))} hard-example images → {out_dir}/hard_pos_*.png")


def visualize_shared_frame(cameras, records, assignment, sequence, out_dir,
                            shared_frame_id=None):
    """
    For a single frame present in every camera, render one panel per camera with:
      • bounding boxes of every detection in that frame
      • label: track_id | GT:<global_id> | d_clip:<best cross-cam cosine dist>
               d_pos:<cosine dist to same-GT tracklet across cameras (if any)>
    Boxes are green when the best cross-cam CLIP match is the true GT match,
    orange otherwise (identity confusion), red when no GT match found.

    The chosen frame is the median of the shared-frame intersection unless
    --shared-frame-id overrides it.
    """
    # ── 1. find shared frames ────────────────────────────────────────────────
    frame_sets = [set(df["frame_id"].astype(int).tolist()) for _, df in cameras]
    shared = sorted(frame_sets[0].intersection(*frame_sets[1:]))
    if not shared:
        print("  [WARN] No frame shared across all cameras — skipping visualisation.")
        return
    if shared_frame_id is not None:
        if shared_frame_id not in shared:
            print(f"  [WARN] --shared-frame-id={shared_frame_id} not in shared set; "
                  f"using median instead.")
            fid = shared[len(shared) // 2]
        else:
            fid = shared_frame_id
    else:
        fid = shared[len(shared) // 2]
    print(f"  Shared frame chosen: {fid}  (total shared: {len(shared)})")

    # ── 2. build fast lookups ────────────────────────────────────────────────
    records_by_key = {(r["cam_id"], r["track_id"]): r for r in records}

    # For every tracklet that has feat_clip, compute:
    #   best_d      — min cosine dist to any other-cam tracklet (CLIP)
    #   best_is_gt  — whether that nearest neighbour has the same GT id
    #   pos_d       — min cosine dist to a same-GT other-cam tracklet (CLIP)
    #   reid_best_d — same as best_d but using feat_reid
    #   reid_pos_d  — same as pos_d  but using feat_reid
    clip_summary = {}
    keyed = [(k, r) for k, r in records_by_key.items() if r["feat_clip"] is not None]
    for (cam_i, tid_i), ri in keyed:
        gt_i = assignment.get((cam_i, tid_i))
        best_d, best_key = 1.0, None
        pos_d = None
        reid_best_d = None
        reid_pos_d  = None
        for (cam_j, tid_j), rj in keyed:
            if cam_j == cam_i:
                continue
            # CLIP distances
            d_clip = float(cosine_dist(ri["feat_clip"], rj["feat_clip"]))
            if d_clip < best_d:
                best_d, best_key = d_clip, (cam_j, tid_j)
            gt_j = assignment.get((cam_j, tid_j))
            if gt_i is not None and gt_i == gt_j:
                if pos_d is None or d_clip < pos_d:
                    pos_d = d_clip
            # ReID distances (only when both have feat_reid)
            if ri.get("feat_reid") is not None and rj.get("feat_reid") is not None:
                d_reid = float(cosine_dist(ri["feat_reid"], rj["feat_reid"]))
                if reid_best_d is None or d_reid < reid_best_d:
                    reid_best_d = d_reid
                if gt_i is not None and gt_i == gt_j:
                    if reid_pos_d is None or d_reid < reid_pos_d:
                        reid_pos_d = d_reid

        best_is_gt = (best_key is not None and
                      gt_i is not None and
                      assignment.get(best_key) == gt_i)
        clip_summary[(cam_i, tid_i)] = {
            "best_d":     best_d,
            "best_key":   best_key,
            "best_is_gt": best_is_gt,
            "pos_d":      pos_d,
            "reid_best_d": reid_best_d,
            "reid_pos_d":  reid_pos_d,
        }

    # ── 3. render one panel per camera ──────────────────────────────────────
    n_cams  = len(cameras)
    n_cols  = min(n_cams, 3)
    n_rows  = (n_cams + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(7 * n_cols, 5 * n_rows),
                             squeeze=False)

    img_dir_tmpl = os.path.join(FRAMES_ROOT, sequence, "{cam}", "img1")

    for ax_idx, (cam_id, df) in enumerate(cameras):
        ax = axes[ax_idx // n_cols][ax_idx % n_cols]
        cam_name = f"c{cam_id:03d}"

        # Load frame image
        img_dir = img_dir_tmpl.format(cam=cam_name)
        img_path = os.path.join(img_dir, f"{fid:06d}.jpg")
        if os.path.isfile(img_path):
            bgr = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            frame_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if bgr is not None else None
        else:
            frame_rgb = None

        if frame_rgb is None:
            ax.set_facecolor("black")
            ax.text(0.5, 0.5, "Frame not found", color="white",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title(cam_name)
            ax.axis("off")
            continue

        ax.imshow(frame_rgb)

        # Draw each detection in this frame
        frame_dets = df[df["frame_id"].astype(int) == fid]
        for _, det in frame_dets.iterrows():
            tid = int(det["track_id"])
            key = (cam_id, tid)
            gt_id       = assignment.get(key)
            cs          = clip_summary.get(key, {})
            best_d      = cs.get("best_d", float("nan"))
            pos_d       = cs.get("pos_d")
            is_gt       = cs.get("best_is_gt", False)
            reid_best_d = cs.get("reid_best_d")
            reid_pos_d  = cs.get("reid_pos_d")

            x1, y1 = float(det["x1"]), float(det["y1"])
            x2, y2 = float(det["x2"]), float(det["y2"])

            # Colour logic
            if gt_id is None:
                color = "red"
            elif is_gt:
                color = "limegreen"
            else:
                color = "orange"

            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                  linewidth=2, edgecolor=color, facecolor="none")
            ax.add_patch(rect)

            # Build label text
            gt_str  = str(gt_id) if gt_id is not None else "?"
            d_str   = f"{best_d:.2f}" if not np.isnan(best_d) else "?"
            pos_str = f"{pos_d:.2f}" if pos_d is not None else "—"
            reid_d_str   = f"{reid_best_d:.2f}" if reid_best_d is not None else "—"
            reid_pos_str = f"{reid_pos_d:.2f}"  if reid_pos_d  is not None else "—"
            label = (f"t{tid} GT:{gt_str}\n"
                     f"clip d={d_str} pos={pos_str}\n"
                     f"reid d={reid_d_str} pos={reid_pos_str}")

            ax.text(x1 + 2, y1 - 4, label,
                    color="white", fontsize=5.5, va="bottom",
                    bbox=dict(boxstyle="round,pad=0.1", fc=color, alpha=0.6, lw=0))

        ax.set_title(f"{cam_name}  |  frame {fid}", fontsize=10)
        ax.axis("off")

    # Hide unused axes
    for idx in range(n_cams, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].axis("off")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="limegreen", label="best CLIP match = true GT"),
        Patch(facecolor="orange",    label="CLIP confused (diff GT)"),
        Patch(facecolor="red",       label="no GT assigned"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3,
               fontsize=9, framealpha=0.8)

    fig.suptitle(
        f"Shared frame {fid} — all cameras\n"
        f"clip/reid d=best cross-cam dist  pos=same-GT cross-cam dist",
        fontsize=11, y=1.01,
    )
    fig.tight_layout()
    path = os.path.join(out_dir, f"shared_frame_{fid}.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


def save_crop_size_histogram(records, out_dir):
    """Histogram of vehicle crop areas to diagnose whether crops are too small."""
    areas = []
    for r in records:
        for _, row in r["df"].iterrows():
            w = int(row["x2"]) - int(row["x1"])
            h = int(row["y2"]) - int(row["y1"])
            if w > 0 and h > 0:
                areas.append(w * h)
    if not areas:
        return
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.hist(areas, bins=80, color="steelblue")
    ax.axvline(32*32, color="red", linestyle="--", label="32×32 px threshold")
    ax.set_xlabel("Crop area (px²)")
    ax.set_ylabel("Count")
    ax.set_title("Detection crop size distribution")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(out_dir, "crop_sizes.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    pct_small = 100 * np.mean(np.array(areas) < 32*32)
    print(f"  Crop sizes: median={int(np.median(areas))} px²  "
          f"  {pct_small:.1f}% below 32×32")
    print(f"  Saved → {path}")

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'═'*60}")
    print(f"  debug_clip.py  —  sequence: {args.sequence}")
    print(f"{'═'*60}\n")

    # 1. Load tracklets
    print("[1/7] Loading MTSC tracklets…")
    cameras = load_sequence(args.root, args.sequence)
    cam_ids = [c for c, _ in cameras]

    # 2. Load GT and assign GT IDs to predicted tracklets
    print("\n[2/7] Loading ground truth and matching tracklets…")
    gt_df      = load_gt(args.gt, args.sequence)
    gt_df_seq  = gt_df[gt_df["cam_id"].isin(cam_ids)]
    assignment = assign_gt_ids(cameras, gt_df_seq)
    n_matched  = sum(v is not None for v in assignment.values())
    print(f"  {n_matched}/{len(assignment)} tracklets matched to a GT global ID")

    # 3. Extract features (HSV + CLIP + ReID)
    use_reid = not args.no_reid
    print(f"\n[3/7] Extracting HSV + CLIP{' + ReID' if use_reid else ''} features…")
    records = extract_all_features(
        cameras, args.sequence,
        args.n_hist_bins, args.max_frames_sample, args.clip_min_crop_px,
        context_pad=args.clip_context_pad,
        use_reid=use_reid,
    )
    if use_reid:
        n_reid_ok = sum(1 for r in records if r["feat_reid"] is not None)
        print(f"  ReID features extracted for {n_reid_ok}/{len(records)} tracklets")

    # 3b. Per-camera whitening (same as MTMC_gps.py)
    if not args.no_cam_whitening:
        print("\n  Applying per-camera whitening…")
        def whiten_mode(feat_key):
            pseudo = [{"cam_id": r["cam_id"], "feature": r[feat_key]}
                      for r in records if r[feat_key] is not None]
            apply_camera_whitening(pseudo)
            idx = 0
            for r in records:
                if r[feat_key] is not None:
                    r[feat_key] = pseudo[idx]["feature"]
                    idx += 1
        whiten_mode("feat_hsv")
        whiten_mode("feat_clip")
        if use_reid:
            whiten_mode("feat_reid")
        # Recompute fused after whitening
        for r in records:
            if r["feat_hsv"] is not None and r["feat_clip"] is not None:
                r["feat_fused"] = np.concatenate(
                    [r["feat_clip"], r["feat_hsv"]]).astype(np.float32)

    # 4. Build positive / negative pairs
    print("\n[4/7] Building cross-camera pairs…")
    positives, negatives = build_pairs(records, assignment, args.max_pairs)
    print(f"  Positive pairs: {len(positives)}")
    print(f"  Negative pairs: {len(negatives)}")

    if not positives:
        print("\n[WARN] No positive pairs found — check that --gt covers this sequence "
              "and that cam_ids match.")
        return

    pos_hsv   = [p[2] for p in positives]
    pos_clip  = [p[3] for p in positives]
    pos_fused = [p[4] for p in positives if p[4] is not None]
    pos_reid  = [p[5] for p in positives if p[5] is not None]
    neg_hsv   = [p[2] for p in negatives]
    neg_clip  = [p[3] for p in negatives]
    neg_fused = [p[4] for p in negatives if p[4] is not None]
    neg_reid  = [p[5] for p in negatives if p[5] is not None]

    # 5. Print stats + plots
    print("\n[5/7] Statistics per feature mode:")
    print_stats("HSV",      pos_hsv,   neg_hsv)
    print_stats("CLIP",     pos_clip,  neg_clip)
    if pos_fused:
        print_stats("CLIP+HSV", pos_fused, neg_fused)
    if pos_reid:
        print_stats("ReID",   pos_reid,  neg_reid)

    print("\n  Saving plots…")
    plot_distributions(pos_hsv,   neg_hsv,   "HSV",      args.output_dir)
    plot_distributions(pos_clip,  neg_clip,  "CLIP",     args.output_dir)
    if pos_fused:
        plot_distributions(pos_fused, neg_fused, "CLIP+HSV", args.output_dir)
    if pos_reid:
        plot_distributions(pos_reid,  neg_reid,  "ReID",     args.output_dir)

    methods = [("HSV", pos_hsv, neg_hsv), ("CLIP", pos_clip, neg_clip)]
    if pos_fused:
        methods.append(("CLIP+HSV", pos_fused, neg_fused))
    if pos_reid:
        methods.append(("ReID", pos_reid, neg_reid))
    plot_roc(methods, args.output_dir)

    # Crop size diagnostic
    print("\n  Crop size diagnostic:")
    save_crop_size_histogram(records, args.output_dir)

    # 6. Hard-example visualisation
    print(f"\n[6/7] Saving top-{args.n_hard_examples} hardest CLIP positive pairs…")
    records_by_key = {(r["cam_id"], r["track_id"]): r for r in records}
    save_hard_examples(
        positives, records_by_key, args.sequence, assignment,
        args.output_dir, n=args.n_hard_examples,
    )

    # 7. Shared-frame panel
    print(f"\n[7/7] Rendering shared-frame panel…")
    visualize_shared_frame(
        cameras, records, assignment, args.sequence, args.output_dir,
        shared_frame_id=args.shared_frame_id,
    )

    print(f"\nDone. All outputs in: {args.output_dir}\n")


if __name__ == "__main__":
    main()
