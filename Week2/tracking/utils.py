import os
from typing import Dict
import cv2
import pandas as pd
from tqdm import tqdm


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
def render_tracked_video(video_in: str, tracked_df: pd.DataFrame, video_out: str, conf_thr: float = 0.0, 
                         show_conf: bool = True) -> None:
    """
    Draw (x1,y1,x2,y2) and track_id on the input video and write to video_out.
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

    pbar = tqdm(total=total_frames, desc="Rendering video", unit="frame")

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx in by_frame:
                for _, r in by_frame[frame_idx].iterrows():
                    x1, y1, x2, y2 = int(r.x1), int(r.y1), int(r.x2), int(r.y2)
                    tid = int(r.track_id)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    label = f"Car ID {tid}"
                    if show_conf and "confidence" in r.index and pd.notna(r.confidence):
                        label += f" {float(r.confidence):.2f}"

                    cv2.putText(frame, label, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                                (0, 255, 0), 2, cv2.LINE_AA)

            out.write(frame)
            frame_idx += 1
            pbar.update(1)
    finally:
        pbar.close()
        cap.release()
        out.release()