"""
viz_MTMC.py – MTMC prediction vs ground-truth visualizer

Usage:
  python viz_MTMC.py S01 --cameras 1 2 3 --frames 100 500 \
      --frames-root AI_CITY_CHALLENGE_2022_TRAIN/train \
      --preds-root  Week4/tracking_overlap_yolov26x_base/output_detections \
      --preds-file  Week4/results/S01_mtmc.txt \
      --gt-file     AI_CITY_CHALLENGE_2022_TRAIN/eval/ground_truth_train.txt \
      --output      mtmc_visualization_S01.mp4
"""

import os
import sys
import argparse
import itertools
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


# ── Drawing constants ──────────────────────────────────────────────────────────

COLOUR_GT        = (220,  80,  40)
COLOUR_PRED      = ( 40,  60, 220)
COLOUR_LINE_GT   = (255, 150,  80)
COLOUR_LINE_PRED = ( 80, 120, 255)

BBOX_THICKNESS  = 2
LINE_THICKNESS  = 1
FONT            = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE_ID   = 0.45
FONT_SCALE_CAM  = 0.7
FONT_THICK      = 1
CAM_LABEL_ALPHA = 0.55


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Visualize MTMC predictions vs ground truth across cameras.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("sequence", type=str, help="Sequence name, e.g. S01")
    p.add_argument("--cameras", type=int, nargs="+", required=True,
                   help="Camera IDs to display (max 4)")
    p.add_argument("--frames", type=int, nargs=2, metavar=("START", "END"),
                   required=True, help="Frame range (inclusive)")
    p.add_argument("--frames-root", type=str,
                   default="AI_CITY_CHALLENGE_2022_TRAIN/train")
    p.add_argument("--preds-root", type=str,
                   default="Week4/tracking_overlap_yolov26x_base/output_detections")
    p.add_argument("--preds-file", type=str, default=None)
    p.add_argument("--gt-file", type=str,
                   default="AI_CITY_CHALLENGE_2022_TRAIN/eval/ground_truth_train.txt")
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--panel-width", type=int, default=640)
    p.add_argument("--no-gt",   action="store_true")
    p.add_argument("--no-pred", action="store_true")
    return p.parse_args()


# ── Data loading ───────────────────────────────────────────────────────────────

MTMC_COLS = ["CameraId", "Id", "FrameId", "X", "Y", "Width", "Height", "Xworld", "Yworld"]


def get_sequence_camera_ids(frames_root: str, sequence: str) -> set:
    """Return camera IDs belonging to this sequence by scanning its folder."""
    import glob as _glob
    cam_dirs = _glob.glob(os.path.join(frames_root, sequence, "c[0-9][0-9][0-9]"))
    return {int(os.path.basename(d)[1:]) for d in cam_dirs}


def load_cam_timestamps(frames_root: str, sequence: str) -> dict:
    """Read cam_timestamp/<sequence>.txt → {cam_id: float}. Returns {} if absent."""
    path = os.path.join(os.path.dirname(frames_root), "cam_timestamp", f"{sequence}.txt")
    if not os.path.isfile(path):
        return {}
    result = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2 or line.startswith("#"):
                continue
            try:
                result[int(parts[0][1:])] = float(parts[1])
            except (ValueError, IndexError):
                continue
    return result


def get_sequence_fps(frames_root: str, sequence: str, cam_ids: list) -> float:
    """Read FPS from seqinfo.ini; falls back to 10."""
    import configparser
    for cam_id in sorted(cam_ids):
        ini = os.path.join(frames_root, sequence, f"c{cam_id:03d}", "seqinfo.ini")
        if os.path.isfile(ini):
            cfg = configparser.ConfigParser()
            cfg.read(ini)
            try:
                fps = float(cfg["Sequence"]["frameRate"])
                print(f"  FPS from seqinfo.ini (cam {cam_id}): {fps}")
                return fps
            except (KeyError, ValueError):
                continue
    print("  [WARN] seqinfo.ini not found – assuming 10 fps")
    return 10.0


def compute_frame_offsets(timestamps: dict, fps: float) -> dict:
    """Compute per-camera frame offsets relative to the earliest camera."""
    if not timestamps:
        return {}
    min_ts = min(timestamps.values())
    return {cam: round((ts - min_ts) * fps) for cam, ts in timestamps.items()}


def load_mtmc_file(path: str, cameras: list, frame_start: int, frame_end: int,
                   valid_cam_ids: set = None, frame_offsets: dict = None) -> pd.DataFrame:
    """Load and filter an MTMC result/GT file to the relevant cameras & frames."""
    if not os.path.isfile(path):
        print(f"[WARN] File not found: {path}")
        return pd.DataFrame(columns=MTMC_COLS)

    df = pd.read_csv(path, sep=r"\s+", header=None, names=MTMC_COLS)

    # Filter out rows from other sequences
    if valid_cam_ids is not None:
        before = len(df)
        df = df[df["CameraId"].isin(valid_cam_ids)]
        if (dropped := before - len(df)):
            print(f"  [GT] Dropped {dropped} rows from other sequences")

    df = df[df["CameraId"].isin(cameras)]

    # Align GT frame IDs with image-file indices using per-camera offsets
    if frame_offsets:
        for cam_id, offset in frame_offsets.items():
            if offset == 0:
                continue
            mask = df["CameraId"] == cam_id
            if mask.any():
                df.loc[mask, "FrameId"] += offset

    df = df[(df["FrameId"] >= frame_start) & (df["FrameId"] <= frame_end)]
    return df.reset_index(drop=True)


def build_lookup(df: pd.DataFrame):
    """Build lookup[cam_id][frame_id] → list of (gid, x, y, w, h)."""
    lookup = defaultdict(lambda: defaultdict(list))
    for _, row in df.iterrows():
        lookup[int(row["CameraId"])][int(row["FrameId"])].append((
            int(row["Id"]), int(row["X"]), int(row["Y"]),
            int(row["Width"]), int(row["Height"]),
        ))
    return lookup


# ── Frame helpers ──────────────────────────────────────────────────────────────

def frame_path(frames_root: str, sequence: str, cam_id: int, frame_id: int) -> str:
    return os.path.join(frames_root, sequence, f"c{cam_id:03d}", "img1", f"{frame_id:06d}.jpg")


def read_frame(path: str, target_width: int):
    """Read and resize a frame; return black placeholder if missing."""
    placeholder = np.zeros((target_width * 9 // 16, target_width, 3), dtype=np.uint8), 1.0
    if not os.path.isfile(path):
        return placeholder
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return placeholder
    h, w = img.shape[:2]
    scale = target_width / w
    return cv2.resize(img, (target_width, int(h * scale))), scale


# ── Drawing helpers ────────────────────────────────────────────────────────────

def draw_bbox(img, x, y, w, h, gid, colour, scale):
    """Draw a scaled bbox with ID label; return centre (cx, cy)."""
    x1 = max(0, int(x * scale))
    y1 = max(0, int(y * scale))
    x2 = min(img.shape[1] - 1, int((x + w) * scale))
    y2 = min(img.shape[0] - 1, int((y + h) * scale))

    cv2.rectangle(img, (x1, y1), (x2, y2), colour, BBOX_THICKNESS)

    label = str(gid)
    (tw, th), baseline = cv2.getTextSize(label, FONT, FONT_SCALE_ID, FONT_THICK)
    tx, ty = x1, max(y1 - 4, th + 2)
    cv2.rectangle(img, (tx, ty - th - baseline), (tx + tw + 2, ty + baseline), colour, -1)
    cv2.putText(img, label, (tx + 1, ty), FONT, FONT_SCALE_ID, (255, 255, 255), FONT_THICK, cv2.LINE_AA)

    return (x1 + x2) // 2, (y1 + y2) // 2


def draw_cam_label(img, cam_id: int):
    """Overlay a semi-transparent camera label in the top-left corner."""
    label = f"Cam {cam_id}"
    (tw, th), baseline = cv2.getTextSize(label, FONT, FONT_SCALE_CAM, 2)
    pad = 6
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (tw + pad * 2, th + baseline + pad * 2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, CAM_LABEL_ALPHA, img, 1 - CAM_LABEL_ALPHA, 0, img)
    cv2.putText(img, label, (pad, th + pad), FONT, FONT_SCALE_CAM, (255, 255, 255), 2, cv2.LINE_AA)


# ── Layout helpers ─────────────────────────────────────────────────────────────

def make_canvas(panels: list, n_cams: int):
    """Arrange panels into layout; return (canvas, {panel_idx: (ox, oy)})."""
    ph, pw = panels[0].shape[:2]
    offsets = {}

    if n_cams == 1:
        return panels[0].copy(), {0: (0, 0)}

    elif n_cams == 2:
        canvas = np.zeros((ph, pw * 2, 3), dtype=np.uint8)
        for i, panel in enumerate(panels):
            canvas[:, i * pw:(i + 1) * pw] = panel
            offsets[i] = (i * pw, 0)

    elif n_cams == 3:
        # Top: 1 centred panel; bottom: 2 panels
        canvas = np.zeros((ph * 2, pw * 2, 3), dtype=np.uint8)
        top_ox = pw // 2
        canvas[0:ph, top_ox:top_ox + pw] = panels[0]
        offsets[0] = (top_ox, 0)
        for i in range(2):
            canvas[ph:, i * pw:(i + 1) * pw] = panels[i + 1]
            offsets[i + 1] = (i * pw, ph)

    else:  # 4 – 2×2 grid
        canvas = np.zeros((ph * 2, pw * 2, 3), dtype=np.uint8)
        for i, panel in enumerate(panels):
            row, col = divmod(i, 2)
            ox, oy = col * pw, row * ph
            canvas[oy:oy + ph, ox:ox + pw] = panel
            offsets[i] = (ox, oy)

    return canvas, offsets


# ── Cross-camera lines ─────────────────────────────────────────────────────────

def draw_cross_cam_lines(canvas, centres_by_gid, offsets, colour):
    """Connect same global ID across panels with lines."""
    for gid, locs in centres_by_gid.items():
        if len(locs) < 2:
            continue
        for (pi, cx1, cy1), (pj, cx2, cy2) in itertools.combinations(locs, 2):
            ox1, oy1 = offsets[pi]
            ox2, oy2 = offsets[pj]
            cv2.line(canvas, (ox1 + cx1, oy1 + cy1), (ox2 + cx2, oy2 + cy2),
                     colour, LINE_THICKNESS, cv2.LINE_AA)


# ── Per-frame render ───────────────────────────────────────────────────────────

def render_frame(frame_id, cameras, frames_root, sequence,
                 gt_lookup, pred_lookup, panel_width, show_gt, show_pred):
    """Build and return the composite canvas for a single frame."""
    panels, scales = [], []
    for cam_id in cameras:
        img, scale = read_frame(frame_path(frames_root, sequence, cam_id, frame_id), panel_width)
        panels.append(img)
        scales.append(scale)

    gt_centres_by_gid   = defaultdict(list)
    pred_centres_by_gid = defaultdict(list)

    for pi, (cam_id, panel, scale) in enumerate(zip(cameras, panels, scales)):
        if show_gt:
            for gid, x, y, w, h in gt_lookup[cam_id][frame_id]:
                cx, cy = draw_bbox(panel, x, y, w, h, gid, COLOUR_GT, scale)
                gt_centres_by_gid[gid].append((pi, cx, cy))
        if show_pred:
            for gid, x, y, w, h in pred_lookup[cam_id][frame_id]:
                cx, cy = draw_bbox(panel, x, y, w, h, gid, COLOUR_PRED, scale)
                pred_centres_by_gid[gid].append((pi, cx, cy))
        draw_cam_label(panel, cam_id)

    canvas, offsets = make_canvas(panels, len(cameras))

    if show_gt:
        draw_cross_cam_lines(canvas, gt_centres_by_gid, offsets, COLOUR_LINE_GT)
    if show_pred:
        draw_cross_cam_lines(canvas, pred_centres_by_gid, offsets, COLOUR_LINE_PRED)

    draw_legend(canvas, show_gt, show_pred)
    return canvas


def draw_legend(canvas, show_gt: bool, show_pred: bool):
    """Draw a small GT/Pred colour legend in the bottom-right corner."""
    items = []
    if show_gt:   items.append(("GT",   COLOUR_GT))
    if show_pred: items.append(("Pred", COLOUR_PRED))
    if not items:
        return

    pad, box_size = 6, 14
    lh = box_size + pad
    total_h, total_w = lh * len(items) + pad, 80
    h, w = canvas.shape[:2]
    ox, oy = w - total_w - pad, h - total_h - pad

    overlay = canvas.copy()
    cv2.rectangle(overlay, (ox - pad, oy - pad), (ox + total_w, oy + total_h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0, canvas)

    for i, (label, colour) in enumerate(items):
        y = oy + i * lh + box_size
        cv2.rectangle(canvas, (ox, y - box_size + 2), (ox + box_size, y + 2), colour, -1)
        cv2.putText(canvas, label, (ox + box_size + 5, y), FONT, 0.45, (230, 230, 230), 1, cv2.LINE_AA)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if len(args.cameras) > 4:
        sys.exit("[ERROR] Maximum 4 cameras supported.")
    cameras = sorted(set(args.cameras))
    n_cams = len(cameras)

    frame_start, frame_end = args.frames
    if frame_start > frame_end:
        sys.exit("[ERROR] Frame start must be <= frame end.")

    sequence  = args.sequence
    preds_file = args.preds_file or os.path.join("Week4", "results", f"{sequence}_mtmc.txt")
    cam_str   = "-".join(str(c) for c in cameras)
    out_path  = args.output or f"{sequence}_vis_c{cam_str}_f{frame_start}-{frame_end}.mp4"

    print(f"\n  Sequence: {sequence} | Cameras: {cameras} | Frames: {frame_start}–{frame_end}")
    print(f"  Preds: {preds_file}\n  GT:    {args.gt_file}\n  Out:   {out_path}\n")

    # Discover cameras & compute GT frame offsets
    valid_cam_ids = get_sequence_camera_ids(args.frames_root, sequence) or None
    if valid_cam_ids:
        print(f"  Sequence {sequence} camera IDs: {sorted(valid_cam_ids)}")
    else:
        print(f"  [WARN] Could not detect camera IDs for {sequence} – GT filtering disabled.")

    timestamps = load_cam_timestamps(args.frames_root, sequence)
    if timestamps:
        fps = get_sequence_fps(args.frames_root, sequence, list(valid_cam_ids or cameras))
        frame_offsets = compute_frame_offsets(timestamps, fps)
        non_zero = {c: o for c, o in frame_offsets.items() if o != 0}
        print(f"  Frame offsets: {non_zero}" if non_zero else "  No frame offset needed.")
    else:
        frame_offsets = {}
        print("  [INFO] cam_timestamp.txt not found – GT frame offsets not applied.")

    # Load data
    print("\n[1/3] Loading predictions …")
    pred_df = load_mtmc_file(preds_file, cameras, frame_start, frame_end)
    print(f"  {len(pred_df)} rows | {pred_df['Id'].nunique() if len(pred_df) else 0} IDs")

    print("[2/3] Loading ground truth …")
    gt_df = load_mtmc_file(args.gt_file, cameras, frame_start, frame_end,
                           valid_cam_ids=valid_cam_ids, frame_offsets=frame_offsets)
    print(f"  {len(gt_df)} rows | {gt_df['Id'].nunique() if len(gt_df) else 0} IDs")

    pred_lookup = build_lookup(pred_df)
    gt_lookup   = build_lookup(gt_df)
    show_gt, show_pred = not args.no_gt, not args.no_pred

    # Probe canvas dimensions from first frame
    probe_img, _ = read_frame(frame_path(args.frames_root, sequence, cameras[0], frame_start),
                               args.panel_width)
    dummy_canvas, _ = make_canvas([probe_img.copy() for _ in cameras], n_cams)
    canvas_h, canvas_w = dummy_canvas.shape[:2]

    # Set up video writer
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                             args.fps, (canvas_w, canvas_h))
    if not writer.isOpened():
        sys.exit(f"[ERROR] Cannot open video writer: {out_path}")

    # Render
    n_frames = frame_end - frame_start + 1
    print(f"\n[3/3] Rendering {n_frames} frames …")
    for fid in tqdm(range(frame_start, frame_end + 1), unit="frame"):
        writer.write(render_frame(fid, cameras, args.frames_root, sequence,
                                  gt_lookup, pred_lookup, args.panel_width,
                                  show_gt, show_pred))
    writer.release()

    print(f"\n  ✓ Saved → {out_path}  [{canvas_w}×{canvas_h}, {n_frames} frames @ {args.fps} fps, {n_frames/args.fps:.1f}s]")


if __name__ == "__main__":
    main()