import os
from typing import Dict, Tuple
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


def _track_id_to_color(track_id: int) -> Tuple[int, int, int]:
    """
    Map a track ID to a visually distinct BGR color using the HSV color wheel.
    Spreads IDs evenly across hue space with high saturation and value so colors
    pop against a grayscale background.
    """
    # Use golden-ratio hue stepping for maximum perceptual separation
    hue = int((track_id * 37) % 180)  # OpenCV hue range is 0-179
    hsv_pixel = np.array([[[hue, 230, 255]]], dtype=np.uint8)
    bgr_pixel = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)
    b, g, r = int(bgr_pixel[0, 0, 0]), int(bgr_pixel[0, 0, 1]), int(bgr_pixel[0, 0, 2])
    return (b, g, r)


# Path functions
def repo_root_from_this_file(file_path: str) -> str:
    """
    Resolve repo root assuming this file is at: <repo>/Week2/tracking/utils.py
    """
    here = os.path.dirname(os.path.abspath(file_path))
    return os.path.abspath(os.path.join(here, "..", ".."))


def resolve_path(p: str, repo_root: str) -> str:
    """
    If p is absolute -> keep it, else resolve relative to repo_root.
    """
    if os.path.isabs(p):
        return p
    return os.path.join(repo_root, p)


def ensure_dir_for_file(path: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)


# Render to a video
def _draw_dashed_rect(frame: np.ndarray, pt1: Tuple[int, int], pt2: Tuple[int, int],
                      dash_len: int = 10, gap_len: int = 6, thickness: int = 2) -> None:
    """
    Draw a dashed rectangle by alternating white and black segments along each edge.
    This gives a high-contrast 'ghost' look that stands out on any background without
    being confused with the solid colored tracked bounding boxes.
    """
    x1, y1 = pt1
    x2, y2 = pt2
    colors = [(255, 255, 255), (30, 30, 30)]  # alternating white / near-black

    def _dashed_line(p_start, p_end):
        sx, sy = p_start
        ex, ey = p_end
        length = max(abs(ex - sx), abs(ey - sy))
        if length == 0:
            return
        dx = (ex - sx) / length
        dy = (ey - sy) / length
        pos = 0
        color_idx = 0
        while pos < length:
            seg_end = min(pos + dash_len, length)
            gap_start = min(seg_end, length)
            gap_end = min(gap_start + gap_len, length)
            p0 = (int(sx + dx * pos), int(sy + dy * pos))
            p1 = (int(sx + dx * seg_end), int(sy + dy * seg_end))
            cv2.line(frame, p0, p1, colors[color_idx % 2], thickness)
            color_idx += 1
            pos = gap_end

    _dashed_line((x1, y1), (x2, y1))  # top
    _dashed_line((x2, y1), (x2, y2))  # right
    _dashed_line((x2, y2), (x1, y2))  # bottom
    _dashed_line((x1, y2), (x1, y1))  # left


def render_comparison_video(
    video_in: str,
    detections_df: pd.DataFrame,
    tracked_df: pd.DataFrame,
    video_out: str,
    conf_thr: float = 0.0,
    show_conf: bool = True,
) -> None:
    """
    Render a side-by-side comparison video overlaying both raw detections and
    tracked results on the same grayscale frame.

    Visual encoding
    ---------------
    - Raw detections  : dashed black-and-white bounding boxes (no label).
                        High contrast but visually 'passive' — they don't steal
                        attention from the tracks.
    - Tracked results : solid, per-ID color-coded boxes with 'Car ID <n>' labels,
                        identical to render_tracked_video.

    Drawing order: raw boxes first, tracked boxes on top, so track labels are
    never obscured by ghost detections.

    Parameters
    ----------
    video_in       : path to the source video file.
    detections_df  : raw detections DataFrame with columns
                     [frame_id, x1, y1, x2, y2] (and optionally confidence).
    tracked_df     : tracking output DataFrame with columns
                     [frame_id, track_id, x1, y1, x2, y2] (and optionally confidence).
    video_out      : path for the output .mp4 file.
    conf_thr       : minimum confidence to display a *tracked* detection.
                     Raw detections are shown regardless (no filtering).
    show_conf      : whether to append the confidence score to tracked labels.
    """
    ensure_dir_for_file(video_out)

    # --- Prepare tracked detections (filter by confidence) ---
    t_df = tracked_df.copy()
    if "confidence" in t_df.columns:
        t_df = t_df[t_df["confidence"] >= conf_thr].copy()

    # --- Prepare raw detections (no confidence filter) ---
    r_df = detections_df.copy()

    by_frame_tracked: Dict[int, pd.DataFrame] = {
        fid: g for fid, g in t_df.groupby("frame_id", sort=False)
    }
    by_frame_raw: Dict[int, pd.DataFrame] = {
        fid: g for fid, g in r_df.groupby("frame_id", sort=False)
    }

    cap = cv2.VideoCapture(video_in)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_in}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_out, fourcc, fps, (w, h))

    track_color_cache: Dict[int, Tuple[int, int, int]] = {}

    # Legend constants
    LEGEND_X, LEGEND_Y = 12, 20
    LEGEND_LINE_H = 22

    pbar = tqdm(total=total_frames, desc="Rendering comparison video", unit="frame")
    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Grayscale background so colored tracks pop
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            # 1) Draw raw detections (dashed ghost boxes)
            if frame_idx in by_frame_raw:
                for _, r in by_frame_raw[frame_idx].iterrows():
                    x1, y1, x2, y2 = int(r.x1), int(r.y1), int(r.x2), int(r.y2)
                    _draw_dashed_rect(frame, (x1, y1), (x2, y2), dash_len=10, gap_len=5, thickness=2)

            # 2) Draw tracked detections (solid color + label)
            if frame_idx in by_frame_tracked:
                for _, r in by_frame_tracked[frame_idx].iterrows():
                    x1, y1, x2, y2 = int(r.x1), int(r.y1), int(r.x2), int(r.y2)
                    tid = int(r.track_id)

                    if tid not in track_color_cache:
                        track_color_cache[tid] = _track_id_to_color(tid)
                    color = track_color_cache[tid]

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    label = f"Car ID {tid}"
                    if show_conf and "confidence" in r.index and pd.notna(r.confidence):
                        label += f" {float(r.confidence):.2f}"

                    cv2.putText(frame, label, (x1, max(15, y1 - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

            # 3) Legend (top-left corner)
            # Semi-transparent dark background for readability
            overlay = frame.copy()
            cv2.rectangle(overlay, (LEGEND_X - 6, LEGEND_Y - 16),
                          (LEGEND_X + 220, LEGEND_Y + LEGEND_LINE_H * 2 - 4), (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

            # Raw detections legend swatch
            _draw_dashed_rect(frame,
                              (LEGEND_X, LEGEND_Y - 10),
                              (LEGEND_X + 18, LEGEND_Y + 6),
                              dash_len=5, gap_len=3, thickness=2)
            cv2.putText(frame, "Raw detections", (LEGEND_X + 26, LEGEND_Y + 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)

            # Tracked detections legend swatch
            swatch_y = LEGEND_Y + LEGEND_LINE_H
            cv2.rectangle(frame, (LEGEND_X, swatch_y - 10), (LEGEND_X + 18, swatch_y + 6), (60, 220, 100), 2)
            cv2.putText(frame, "Tracked (color = ID)", (LEGEND_X + 26, swatch_y + 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)

            out.write(frame)
            frame_idx += 1
            pbar.update(1)
    finally:
        pbar.close()
        cap.release()
        out.release()


def render_tracked_video(video_in: str, tracked_df: pd.DataFrame, video_out: str, conf_thr: float = 0.0, 
                         show_conf: bool = True) -> None:
    """
    Draw (x1,y1,x2,y2) and track_id on the input video and write to video_out.

    Frames are rendered in grayscale (converted back to BGR for the writer) so
    that the per-track color-coded bounding boxes and labels stand out clearly.
    Each unique track_id is assigned a distinct color derived from the HSV wheel.
    """
    ensure_dir_for_file(video_out)

    df = tracked_df.copy()
    if "confidence" in df.columns:
        df = df[df["confidence"] >= conf_thr].copy()

    by_frame: Dict[int, pd.DataFrame] = {fid: g for fid, g in df.groupby("frame_id", sort=False)}

    cap = cv2.VideoCapture(video_in)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_in}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        total_frames = None  # tqdm will show indeterminate progress

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_out, fourcc, fps, (w, h))

    # Pre-compute a color for every track ID seen in this clip
    track_color_cache: Dict[int, Tuple[int, int, int]] = {}

    pbar = tqdm(total=total_frames, desc="Rendering tracking video", unit="frame")

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert to grayscale then back to BGR so annotations remain colorful
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            if frame_idx in by_frame:
                for _, r in by_frame[frame_idx].iterrows():
                    x1, y1, x2, y2 = int(r.x1), int(r.y1), int(r.x2), int(r.y2)
                    tid = int(r.track_id)

                    # Look up (or compute) this track's unique color
                    if tid not in track_color_cache:
                        track_color_cache[tid] = _track_id_to_color(tid)
                    color = track_color_cache[tid]

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    label = f"Car ID {tid}"
                    if show_conf and "confidence" in r.index and pd.notna(r.confidence):
                        label += f" {float(r.confidence):.2f}"

                    cv2.putText(frame, label, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                color, 2, cv2.LINE_AA)

            out.write(frame)
            frame_idx += 1
            pbar.update(1)
    finally:
        pbar.close()
        cap.release()
        out.release()