import os
import argparse
import pandas as pd
from typing import List

from tqdm import tqdm

from tracking.overlap import track_by_max_overlap


def find_camera_dirs(root: str) -> List[str]:
    """Yield all cXXX leaf directories under root that contain detections.txt."""
    cam_dirs = []
    for seq in sorted(os.listdir(root)):
        seq_path = os.path.join(root, seq)
        if not os.path.isdir(seq_path):
            continue
        for cam in sorted(os.listdir(seq_path)):
            cam_path = os.path.join(seq_path, cam)
            det_file = os.path.join(cam_path, "detections.txt")
            if os.path.isdir(cam_path) and cam.startswith("c") and os.path.isfile(det_file):
                cam_dirs.append(cam_path)
    return cam_dirs


def run(root: str, iou_thr: float, memory_frames: int, memory_iou_thr: float) -> None:
    cam_dirs = find_camera_dirs(root)
    print(f"Found {len(cam_dirs)} camera(s) under {root}\n")

    for cam_path in tqdm(cam_dirs):
        det_file = os.path.join(cam_path, "detections.txt")
        out_file = os.path.join(cam_path, "tracking_overlap.txt")
        rel = os.path.relpath(cam_path, root)

        detections = pd.read_csv(det_file)
        tracked = track_by_max_overlap(
            detections,
            iou_match_thr=iou_thr,
            memory_frames=memory_frames,
            memory_iou_thr=memory_iou_thr,
        )
        tracked.to_csv(out_file, index=False)
        print(f"  [{rel}]  {len(detections)} detections -> {tracked['track_id'].nunique()} tracks  =>  {out_file}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="Week4/detections_yolov3u_base/output_detections",
                    help="Root directory containing S01/S03/S04/... subfolders")
    ap.add_argument("--iou_thr",        type=float, default=0.40)
    ap.add_argument("--memory_frames",  type=int,   default=5)
    ap.add_argument("--memory_iou_thr", type=float, default=0.90)
    args = ap.parse_args()

    run(args.root, args.iou_thr, args.memory_frames, args.memory_iou_thr)