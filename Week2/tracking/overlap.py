from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from metrics import compute_iou



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


def track_by_max_overlap(detections: pd.DataFrame, *, iou_match_thr: float = 0.40, 
                         iou_dup_thr: float = 0.90) -> pd.DataFrame:
    """
    Tracking by maximum overlap (IoU) between consecutive frames:
    - First frame: assign new track_ids
    - Next frames: for each detection, match to prev-frame detection with max IoU.
      If best IoU >= iou_match_thr and prev not already used => inherit ID, else new ID.
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

    # Loop over frames
    for f in frames:
        # Detections of current frame
        cur_df = det[det["frame_id"] == f].copy()

        # Pre-processing: Filter overlapping bboxes inside each frame (IoU > 0.9)
        cur_df = filter_overlapping_bboxes(cur_df, iou_dup_thr=iou_dup_thr)
        cur_df["track_id"] = -1

        # First frame, we assign a different track to each object on the scene
        if prev_df is None or prev_df.empty:
            for i in range(len(cur_df)):
                cur_df.iloc[i, cur_df.columns.get_loc("track_id")] = next_track_id
                next_track_id += 1

            out_parts.append(cur_df)
            prev_df = cur_df.copy()
            continue

        used_prev: Dict[int, bool] = {}
        prev_boxes_xyxy = prev_df[["x1", "y1", "x2", "y2"]].to_numpy(dtype=float)

        # Following frames: One-to-one matching
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
                cur_df.iloc[i, cur_df.columns.get_loc("track_id")] = next_track_id
                next_track_id += 1

        out_parts.append(cur_df)
        prev_df = cur_df.copy()

    tracked = pd.concat(out_parts, ignore_index=True)
    tracked = tracked.sort_values(["frame_id", "track_id", "confidence"], ascending=[True, True, False]).reset_index(drop=True)
    return tracked