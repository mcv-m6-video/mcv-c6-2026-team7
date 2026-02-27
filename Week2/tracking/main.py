import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import argparse
import pandas as pd
from datetime import datetime

from utils import repo_root_from_this_file, resolve_path, ensure_dir_for_file, render_tracked_video
from overlap import track_by_max_overlap


# Path functions
def add_repo_root_to_syspath(repo_root: str) -> None:
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

def outputs_dir(repo_root: str, method: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(repo_root, "Week2", "tracking", "outputs", f"{method}_{ts}")
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir



def main():
    repo_root = repo_root_from_this_file(__file__)
    add_repo_root_to_syspath(repo_root)

    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["overlap", "kalman"], default="overlap")
    ap.add_argument("--detections", default="Week2/detections/detections.txt")
    ap.add_argument("--video", default="data/AICity_data/train/S03/c010/vdo.avi")
    ap.add_argument("--conf_thr_video", type=float, default=0.30)
    ap.add_argument("--iou_thr", type=float, default=0.40)
    args = ap.parse_args()

    # Inputs
    det_path = resolve_path(args.detections, repo_root)
    vid_path = resolve_path(args.video, repo_root)
    
    # Outputs
    exp_dir = outputs_dir(repo_root, args.method)
    out_csv = os.path.join(exp_dir, "tracks.csv")
    out_vid = os.path.join(exp_dir, "tracks.mp4")

    # Select method
    detections = pd.read_csv(det_path)
    if args.method == "overlap":
        tracked = track_by_max_overlap(
            detections,
            iou_match_thr=args.iou_thr,
        )
    # elif args.method == "kalman":

    # Save CSV
    ensure_dir_for_file(out_csv)
    tracked.to_csv(out_csv, index=False)

    # Render video with IDs
    render_tracked_video(vid_path, tracked, out_vid, conf_thr=args.conf_thr_video, show_conf=True)


if __name__ == "__main__":
    main()