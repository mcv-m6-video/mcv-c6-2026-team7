from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import cv2
import os

from metrics import compute_iou
from tqdm import tqdm
from optical_flow import compute_optical_flow
import torch

# Convert because metrics/compute_iou() expects (x,y,w,h)
def xyxy_to_xywh(box_xyxy: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = box_xyxy
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    return (float(x1), float(y1), float(w), float(h))


def filter_overlapping_bboxes(df_frame: pd.DataFrame, iou_dup_thr: float = 0.90, 
                              score_col: str = "confidence") -> pd.DataFrame:
    """
    Filter overlapping bounding boxes with IoU larger than 0.9 to remove repeated masks.
    Keep highest confidence.
    """
    if df_frame.empty:
        return df_frame

    # Sort detections in the same frame by confidence
    df_sorted = df_frame.sort_values(score_col, ascending=False).reset_index(drop=True)
    boxes_xyxy = df_sorted[["x1", "y1", "x2", "y2"]].to_numpy(dtype=float)

    keep: List[int] = []
    # Iterate over detections
    for i in range(len(df_sorted)):
        # Append first detection
        if not keep:
            keep.append(i)
            continue
        
        cur_xywh = xyxy_to_xywh(tuple(boxes_xyxy[i]))
        duplicate = False

        # Check overlap with already kept boxes
        for k in keep:
            kept_xywh = xyxy_to_xywh(tuple(boxes_xyxy[k]))
            if compute_iou(cur_xywh, kept_xywh) > iou_dup_thr:
                duplicate = True
                break

        if not duplicate:
            keep.append(i)

    return df_sorted.iloc[keep].copy()


def track_by_max_overlap(
    detections: pd.DataFrame,
    *,
    iou_match_thr: float = 0.40,
    iou_dup_thr: float = 0.90,
    memory_frames: int = 5,
    memory_iou_thr: float = 0.90,
) -> pd.DataFrame:
    """
    Tracking by maximum overlap (IoU) between consecutive frames with track memory.

    - First frame: assign new track_ids.
    - Next frames: for each detection, match to prev-frame detection with max IoU.
      If best IoU >= iou_match_thr and prev not already used => inherit ID, else new ID.
    - Memory: tracks that go unmatched are kept for up to *memory_frames* frames.
      An unmatched detection inherits a stored track ID if the IoU between the
      detection and the last known box is >= memory_iou_thr.

    Args:
        iou_match_thr:   Minimum IoU for frame-to-frame track continuation.
        iou_dup_thr:     IoU threshold to suppress duplicate detections within one frame.
        memory_frames:   How many frames a lost track is kept in memory (0 = disabled).
        memory_iou_thr:  Minimum IoU between a new detection and a remembered box to
                         re-use the stored track ID.
    """

    # Normalize and work on a copy to not modify the original
    det = detections.copy()
    det["frame_id"] = det["frame_id"].astype(int)

    # Sort frames and handle empty detections
    frames = sorted(det["frame_id"].unique())
    if not frames:
        det["track_id"] = np.array([], dtype=int)
        return det

    next_track_id = 1
    prev_df: Optional[pd.DataFrame] = None
    out_parts: List[pd.DataFrame] = []

    # Memory: track_id -> {"box": last known xyxy tuple, "age": frames since last seen}
    track_memory: Dict[int, Dict] = {}

    # Loop over frames
    for f in frames:
        # Detections of current frame
        cur_df = det[det["frame_id"] == f].copy()

        # Pre-processing: Filter overlapping bboxes inside each frame (IoU > 0.9)
        cur_df = filter_overlapping_bboxes(cur_df, iou_dup_thr=iou_dup_thr)
        cur_df["track_id"] = -1

        # ---- Age and prune memory
        expired = [tid for tid, v in track_memory.items() if v["age"] >= memory_frames]
        for tid in expired:
            del track_memory[tid]
        for mem in track_memory.values():
            mem["age"] += 1

        # First frame: assign a unique track to each detection
        if prev_df is None or prev_df.empty:
            for i in range(len(cur_df)):
                cur_df.iloc[i, cur_df.columns.get_loc("track_id")] = next_track_id
                next_track_id += 1

            out_parts.append(cur_df)
            prev_df = cur_df.copy()
            continue

        used_prev: Dict[int, bool] = {}
        prev_boxes_xyxy = prev_df[["x1", "y1", "x2", "y2"]].to_numpy(dtype=float)
        unmatched_cur: List[int] = []  # indices into cur_df that need memory check

        for i in range(len(cur_df)):
            row = cur_df.iloc[i]
            cur_xywh = xyxy_to_xywh((row["x1"], row["y1"], row["x2"], row["y2"]))

            # Compare bbox in frame N+1 to bboxes in frame N
            ious = np.array(
                [compute_iou(cur_xywh, xyxy_to_xywh(tuple(pb))) for pb in prev_boxes_xyxy],
                dtype=float,
            )

            order = np.argsort(-ious)  # Best IoU first
            assigned = False

            # For each detection in the previous frame
            for j in order:
                # Instead of taking the maximum, set a threshold of IoU >= 0.4
                if float(ious[j]) < iou_match_thr:
                    break
                
                # If the previous detection has not been assigned to another current detection
                if not used_prev.get(int(j), False):
                    used_prev[int(j)] = True
                    # Assign the same ID to the current detection 
                    # (This object in frame N+1 is the same object as detection j in frame N)
                    cur_df.iloc[i, cur_df.columns.get_loc("track_id")] = int(prev_df.iloc[int(j)]["track_id"])
                    assigned = True
                    break

            # If a new bounding box in the frame N+1 has no correspondence N then a new track label is assigned to it
            if not assigned:
                unmatched_cur.append(i)

        # ---- Memory-based re-identification for unmatched detections
        used_memory_tids: set = set()
        for i in unmatched_cur:
            row = cur_df.iloc[i]
            cur_xywh = xyxy_to_xywh((row["x1"], row["y1"], row["x2"], row["y2"]))

            best_iou = memory_iou_thr - 1e-9
            best_tid = None
            for tid, mem in track_memory.items():
                if tid in used_memory_tids:
                    continue
                iou = compute_iou(cur_xywh, xyxy_to_xywh(mem["box"]))
                if iou > best_iou:
                    best_iou = iou
                    best_tid = tid

            if best_tid is not None:
                cur_df.iloc[i, cur_df.columns.get_loc("track_id")] = best_tid
                used_memory_tids.add(best_tid)
                del track_memory[best_tid]
            else:
                cur_df.iloc[i, cur_df.columns.get_loc("track_id")] = next_track_id
                next_track_id += 1

        # ---- Update memory with unmatched prev tracks
        # Build history for tracks still active in prev_df
        for j in range(len(prev_df)):
            if used_prev.get(j, False):
                continue  # was matched -> stays alive, no need for memory
            tid = int(prev_df.iloc[j]["track_id"])
            box = tuple(prev_df.iloc[j][["x1", "y1", "x2", "y2"]].to_numpy(dtype=float))
            track_memory[tid] = {"box": box, "age": 0}

        out_parts.append(cur_df)
        prev_df = cur_df.copy()

    tracked = pd.concat(out_parts, ignore_index=True)
    tracked = tracked.sort_values(["frame_id", "track_id", "confidence"], ascending=[True, True, False]).reset_index(drop=True)
    return tracked

def np_to_torch_img(array: np.ndarray):
    return torch.from_numpy(array).permute(2, 0, 1)  # uint8 [0, 255]

def flow_shift_box(box, flow, scale):
    x1, y1, x2, y2 = box

    # scale box to flow resolution
    x1s = int(x1 * scale)
    y1s = int(y1 * scale)
    x2s = int(x2 * scale)
    y2s = int(y2 * scale)

    h, w = flow.shape[:2]

    x1s = np.clip(x1s, 0, w-1)
    x2s = np.clip(x2s, 0, w-1)
    y1s = np.clip(y1s, 0, h-1)
    y2s = np.clip(y2s, 0, h-1)

    # use the inner ~50% of the box
    cx, cy = (x1s + x2s) // 2, (y1s + y2s) // 2
    rw, rh = (x2s - x1s) // 4, (y2s - y1s) // 4
    region = flow[max(cy-rh,0):cy+rh, max(cx-rw,0):cx+rw]

    if region.size == 0:
        return box

    dx = np.median(region[..., 0]) / scale
    dy = np.median(region[..., 1]) / scale

    return (
        x1 + dx,
        y1 + dy,
        x2 + dx,
        y2 + dy
    )
    
def flow_to_rgb(flow: np.ndarray, mag_thresh: float = 1.0) -> np.ndarray:
    """
    Convert optical flow (HxWx2) to an RGB image for visualization.
    Only keep flow vectors above magnitude threshold to reduce noise.
    """
    h, w = flow.shape[:2]
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    mask = mag >= mag_thresh
    mag[~mask] = 0
    ang[~mask] = 0

    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[...,0] = (ang * 180 / np.pi / 2).astype(np.uint8)  # Hue: angle [0, 179]
    hsv[...,1] = 255                                        # Full saturation
    hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)  # Value: magnitude
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return rgb

def draw_bbox_flow(frame: np.ndarray, flow: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """
    Draw arrows representing median flow inside each bounding box.
    - frame: original frame (HxWx3)
    - flow: optical flow (HxWx2)
    - boxes: Nx4 array of [x1,y1,x2,y2]
    """
    vis = frame.copy()
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        # Extract flow inside bbox
        region_flow = flow[y1:y2, x1:x2]
        if region_flow.size == 0:
            continue
        # Compute median flow vector
        dx = np.median(region_flow[...,0])
        dy = np.median(region_flow[...,1])
        # Draw arrow at center of bbox
        cx = int((x1+x2)/2)
        cy = int((y1+y2)/2)
        tip = (int(cx+dx*10), int(cy+dy*10))  # scale for visibility
        cv2.arrowedLine(vis, (cx,cy), tip, (0,0,255), 2, tipLength=0.3)
        # Optional: draw bbox
        cv2.rectangle(vis, (x1,y1), (x2,y2), (0,255,0), 2)
    return vis

def track_by_max_overlap_flow(
    detections: pd.DataFrame,
    video_path: str,
    *,
    iou_match_thr: float = 0.40,
    iou_dup_thr: float = 0.90,
    memory_frames: int = 5,
    memory_iou_thr: float = 0.90,
    resize_flow_scale: float = 0.25,
    flow_frame_interval: int = 10,
    flow_dir: str = "flow_frames"
) -> pd.DataFrame:
    """
    Tracking with memory + save selected scaled optical flow frames as images.
    Flow is masked inside bounding boxes and denoised.
    """

    det = detections.copy()
    det["frame_id"] = det["frame_id"].astype(int)
    frames = sorted(det["frame_id"].unique())
    if not frames:
        det["track_id"] = np.array([], dtype=int)
        return det

    cap = cv2.VideoCapture(video_path)
    prev_frame_resized = None
    prev_df: Optional[pd.DataFrame] = None
    next_track_id = 1
    out_parts: list[pd.DataFrame] = []
    track_memory: dict[int, dict] = {}

    os.makedirs(flow_dir, exist_ok=True)

    for f in tqdm(frames):
        cur_df = det[det["frame_id"] == f].copy()
        cur_df = filter_overlapping_bboxes(cur_df, iou_dup_thr=iou_dup_thr)
        cur_df["track_id"] = -1

        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ret, frame = cap.read()
        if not ret:
            continue

        height, width = frame.shape[:2]
        new_dim = (int(width * resize_flow_scale), int(height * resize_flow_scale))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = np_to_torch_img(cv2.resize(frame_rgb, new_dim, interpolation=cv2.INTER_AREA))

        flow = None
        if prev_frame_resized is not None:
            flow = compute_optical_flow("raft_small", [], prev_frame_resized, frame_resized)
        prev_frame_resized = frame_resized

        # ---- Save denoised flow image every N frames
        if flow is not None and (f % flow_frame_interval == 0):
            # Resize flow to original frame size and scale values to full-res pixel units
            flow_vis = cv2.resize(flow, (width, height)) / resize_flow_scale
          
            # Create an empty image
            flow_image = np.zeros_like(frame)
            flow_image = flow_to_rgb(flow_vis, mag_thresh=0.4)

            '''# Mask flow inside bounding boxes
            for _, row in cur_df.iterrows():
                x1, y1, x2, y2 = map(int, row[["x1","y1","x2","y2"]])
                flow_crop = flow_vis[y1:y2, x1:x2]
                rgb_crop = flow_to_rgb(flow_crop, mag_thresh=0)
                flow_image[y1:y2, x1:x2] = rgb_crop
            '''
            # Save image
            fname = os.path.join(flow_dir, f"frame_{f:05d}_flow.png")
            cv2.imwrite(fname, flow_image)
            
            # Draw arrows and bounding boxes
            boxes = cur_df[["x1","y1","x2","y2"]].to_numpy()
            flow_image = draw_bbox_flow(frame, flow_vis, boxes)

            # Save
            fname = os.path.join(flow_dir, f"frame_{f:05d}_bboxes_flow.png")
            cv2.imwrite(fname, flow_image)

        # ---- Age and prune memory
        expired = [tid for tid, v in track_memory.items() if v["age"] >= memory_frames]
        for tid in expired:
            del track_memory[tid]
        for mem in track_memory.values():
            mem["age"] += 1

        # ---- Tracking logic
        if prev_df is None or prev_df.empty:
            for i in range(len(cur_df)):
                cur_df.iloc[i, cur_df.columns.get_loc("track_id")] = next_track_id
                next_track_id += 1
            prev_df = cur_df.copy()
            out_parts.append(cur_df)
            continue

        used_prev: dict[int, bool] = {}
        prev_boxes_xyxy = prev_df[["x1", "y1", "x2", "y2"]].to_numpy(dtype=float)

        unmatched_cur: list[int] = []
        for i in range(len(cur_df)):
            row = cur_df.iloc[i]
            cur_xywh = xyxy_to_xywh((row["x1"], row["y1"], row["x2"], row["y2"]))
            ious = np.array([compute_iou(cur_xywh, xyxy_to_xywh(tuple(pb))) for pb in prev_boxes_xyxy], dtype=float)
            order = np.argsort(-ious)
            assigned = False
            for j in order:
                if float(ious[j]) < iou_match_thr:
                    break
                if not used_prev.get(int(j), False):
                    used_prev[int(j)] = True
                    cur_df.iloc[i, cur_df.columns.get_loc("track_id")] = int(prev_df.iloc[int(j)]["track_id"])
                    assigned = True
                    break
            if not assigned:
                unmatched_cur.append(i)

        used_memory_tids: set = set()
        for i in unmatched_cur:
            row = cur_df.iloc[i]
            cur_xywh = xyxy_to_xywh((row["x1"], row["y1"], row["x2"], row["y2"]))
            best_iou = memory_iou_thr - 1e-9
            best_tid = None
            for tid, mem in track_memory.items():
                if tid in used_memory_tids:
                    continue
                iou = compute_iou(cur_xywh, xyxy_to_xywh(mem["box"]))
                if iou > best_iou:
                    best_iou = iou
                    best_tid = tid
            if best_tid is not None:
                cur_df.iloc[i, cur_df.columns.get_loc("track_id")] = best_tid
                used_memory_tids.add(best_tid)
                del track_memory[best_tid]
            else:
                cur_df.iloc[i, cur_df.columns.get_loc("track_id")] = next_track_id
                next_track_id += 1

        # Update memory
        for j in range(len(prev_df)):
            if used_prev.get(j, False):
                continue
            tid = int(prev_df.iloc[j]["track_id"])
            box = tuple(prev_df.iloc[j][["x1","y1","x2","y2"]].to_numpy(dtype=float))
            track_memory[tid] = {"box": box, "age": 0}

        prev_df = cur_df.copy()
        out_parts.append(cur_df)

    cap.release()

    tracked = pd.concat(out_parts, ignore_index=True)
    tracked = tracked.sort_values(["frame_id","track_id","confidence"], ascending=[True,True,False]).reset_index(drop=True)

    return tracked
