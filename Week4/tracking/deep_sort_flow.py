"""
deep_sort_flow.py
=================
execute_deep_SORT_flow  –  Deep SORT + RAFT-small optical-flow box warping.

How it works
------------
Standard Deep SORT calls tracker.predict() (Kalman) then tracker.update().
Here we insert one extra step between predict and update:

    1. tracker.predict()          – Kalman advances every track state by 1 step
    2. flow_warp_tracks()         – RAFT-small estimates pixel motion from frame
                                    t-1 → t; each track's Kalman mean is shifted
                                    by the median flow vector inside its box
    3. tracker.update(dets)       – Hungarian + cosine matching as normal

This gives the association a much more accurate predicted position, especially
for fast or non-linearly moving vehicles.

Flow is cached to disk (same scheme as track_by_max_overlap_flow) so it is
computed only once across multiple runs / Optuna trials.
"""

from __future__ import annotations

import os
import cv2
import numpy as np
import pandas as pd
import torch
import tqdm
import sys
from typing import Optional

# Deep SORT internals  (same imports as execute_deep_SORT)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../external/deep_sort"))

from deep_sort import nn_matching
from deep_sort.tracker import Tracker
from deep_sort.detection import Detection
from application_util import preprocessing

# Reuse flow utilities already present in your codebase
from tracking.overlap import (
    np_to_torch_img,
    compute_optical_flow,
    flow_shift_box,
    get_flow_cache_dir,
    filter_overlapping_bboxes,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_or_compute_flow(
    model_name: str,
    raft_model,          # torchvision RAFT instance (lazy-init, passed in)
    prev_tensor: torch.Tensor,
    cur_tensor: torch.Tensor,
    flow_cache_dir: str,
    prev_f: int,
    cur_f: int,
) -> np.ndarray:
    """Return (H, W, 2) float32 flow array, using disk cache when available."""
    flow_file = os.path.join(flow_cache_dir, f"flow_{prev_f:06d}_{cur_f:06d}.npy")
    if os.path.exists(flow_file):
        return np.load(flow_file)

    flow = compute_optical_flow(model_name, raft_model, prev_tensor, cur_tensor)
    if isinstance(flow, torch.Tensor):
        flow = flow.cpu().numpy()
    np.save(flow_file, flow)
    return flow


def _warp_tracks(tracker_obj: Tracker, flow: np.ndarray, resize_flow_scale: float) -> None:
    """
    Shift the Kalman mean of every confirmed/tentative track by the median
    optical-flow vector inside its predicted bounding box.

    tracker.tracks[i].mean is [cx, cy, aspect, height, vx, vy, va, vh]
    We only touch cx and cy (indices 0, 1).
    """
    h_flow, w_flow = flow.shape[:2]

    for track in tracker_obj.tracks:
        # to_tlwh() reads from the Kalman mean
        tlwh = track.to_tlwh()
        x, y, w, h = tlwh

        # Map box to flow-space coordinates
        scale = resize_flow_scale
        x_f = int(np.clip(x * scale, 0, w_flow - 1))
        y_f = int(np.clip(y * scale, 0, h_flow - 1))
        x2_f = int(np.clip((x + w) * scale, 0, w_flow))
        y2_f = int(np.clip((y + h) * scale, 0, h_flow))

        if x2_f <= x_f or y2_f <= y_f:
            continue

        roi = flow[y_f:y2_f, x_f:x2_f]          # (rh, rw, 2)
        # Median is robust to background clutter
        dx = float(np.median(roi[..., 0])) / scale
        dy = float(np.median(roi[..., 1])) / scale

        # Apply shift directly to the Kalman filter mean (cx, cy)
        track.mean[0] += dx
        track.mean[1] += dy


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def execute_deep_SORT_flow(
    detections_per_frame: pd.DataFrame,
    video_path: str,
    *,
    # ---- Deep SORT params (mirror execute_deep_SORT signature) ----
    max_age: int = 1,
    min_hits: int = 3,
    iou_threshold: float = 0.3,
    show_tracks: bool = False,
    max_cosine_distance: float = 0.3,
    nn_budget: Optional[int] = None,
    nms_max_overlap: float = 1.0,
    # ---- Optical flow params ----
    model: str = "raft_small",
    resize_flow_scale: float = 1.0,
) -> pd.DataFrame:
    """
    Deep SORT with RAFT-small optical-flow warping.

    Parameters
    ----------
    detections_per_frame : pd.DataFrame
        Same format as execute_deep_SORT (frame_id, x1, y1, x2, y2, confidence).
    video_path : str
        Path to the source video (needed to extract frames for RAFT).
    max_age, min_hits, iou_threshold, show_tracks,
    max_cosine_distance, nn_budget, nms_max_overlap
        Identical semantics to execute_deep_SORT.
    model : str
        RAFT variant passed to compute_optical_flow.  "raft_small" (default)
        uses torchvision's built-in weights.
    resize_flow_scale : float
        Resize factor applied to frames before RAFT (1.0 = full resolution).
        Lower values (e.g. 0.5) are faster; use the same value you used when
        pre-computing cached flow with track_by_max_overlap_flow.

    Returns
    -------
    pd.DataFrame  –  same schema as execute_deep_SORT output.
    """
    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", max_cosine_distance, nn_budget
    )
    tracker_obj = Tracker(
        metric=metric,
        max_age=max_age,
        n_init=min_hits,
        max_iou_distance=iou_threshold,
    )

    flow_cache_dir = get_flow_cache_dir(video_path, resize_flow_scale, model=model)
    os.makedirs(flow_cache_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_ids = sorted(detections_per_frame["frame_id"].unique())

    prev_tensor: Optional[torch.Tensor] = None
    prev_f: Optional[int] = None
    all_tracks: list[pd.DataFrame] = []

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    for frame_id in tqdm.tqdm(frame_ids, desc="Deep SORT + Flow"):

        # ---- Read frame --------------------------------------------------
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame_bgr = cap.read()
        if not ret:
            continue

        height, width = frame_bgr.shape[:2]
        new_w = int(width * resize_flow_scale)
        new_h = int(height * resize_flow_scale)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        cur_tensor = np_to_torch_img(
            cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        )

        # ---- Compute / load optical flow ---------------------------------
        flow: Optional[np.ndarray] = None
        if prev_tensor is not None and prev_f is not None:
            flow = _load_or_compute_flow(
                model, [],          # empty list → compute_optical_flow lazy-inits model
                prev_tensor, cur_tensor,
                flow_cache_dir, prev_f, frame_id,
            )

        prev_tensor = cur_tensor
        prev_f = frame_id

        # ---- Build detections --------------------------------------------
        dets_frame = detections_per_frame[detections_per_frame["frame_id"] == frame_id]
        dets: list[Detection] = []
        if not dets_frame.empty:
            for _, row in dets_frame.iterrows():
                tlwh = np.array([
                    row["x1"], row["y1"],
                    row["x2"] - row["x1"],
                    row["y2"] - row["y1"],
                ])
                dets.append(Detection(tlwh, float(row["confidence"]), np.array([])))

            # NMS
            boxes  = np.array([d.tlwh for d in dets])
            scores = np.array([d.confidence for d in dets])
            kept   = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
            dets   = [dets[i] for i in kept]

        # ---- Kalman predict  +  flow warp  +  update --------------------
        tracker_obj.predict()

        if flow is not None:
            _warp_tracks(tracker_obj, flow, resize_flow_scale)

        tracker_obj.update(dets)

        # ---- Collect confirmed tracks ------------------------------------
        frame_tracks = []
        for track in tracker_obj.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            x1, y1, x2, y2 = track.to_tlbr()
            frame_tracks.append([x1, y1, x2, y2, track.track_id])

        if show_tracks:
            print(f"Frame {frame_id}: {frame_tracks}")

        if frame_tracks:
            df_t = pd.DataFrame(frame_tracks, columns=["x1", "y1", "x2", "y2", "track_id"])
            df_t["frame_id"] = frame_id
            all_tracks.append(df_t)

    cap.release()

    # ------------------------------------------------------------------
    # Assemble output (identical schema to execute_deep_SORT)
    # ------------------------------------------------------------------
    if all_tracks:
        result = pd.concat(all_tracks, ignore_index=True)
        result["confidence"] = 1
        result["x3"] = -1
        result["y3"] = -1
        result["z"]  = -1
        result["track_id"] = result["track_id"].astype(int)
        return result[["frame_id", "track_id", "x1", "y1", "x2", "y2",
                        "confidence", "x3", "y3", "z"]]
    else:
        return pd.DataFrame(
            columns=["frame_id", "track_id", "x1", "y1", "x2", "y2",
                     "confidence", "x3", "y3", "z"]
        )