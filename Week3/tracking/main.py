import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import argparse
import pandas as pd
from datetime import datetime

from utils import repo_root_from_this_file, resolve_path, ensure_dir_for_file, render_tracked_video, render_comparison_video
from overlap import track_by_max_overlap
from kalman import execute_kalman_SORT
from prepare_gt_for_trackeval import MOTChallengeConverter


# Path functions
def add_repo_root_to_syspath(repo_root: str) -> None:
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

def outputs_dir(repo_root: str, method: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(repo_root, "Week3", "tracking", "outputs", f"{method}_{ts}")
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
    ap.add_argument("--iou_thr", type=float, default=0.65)
    ap.add_argument("--show_IDs_video", default=True)
    ap.add_argument("--show_comp_video", default=True)

    ### MEMORY PARAMETERS (for re-identification of lost tracks)
    ap.add_argument("--memory_frames", type=int, default=5,
                    help="Number of frames to keep lost tracks in memory (0 = disabled)")
    ap.add_argument("--memory_iou_thr", type=float, default=0.90,
                    help="Min IoU to re-identify a detection with a remembered track")
    args = ap.parse_args()

    # Inputs
    det_path = resolve_path(args.detections, repo_root)
    vid_path = resolve_path(args.video, repo_root)
    
    # Outputs
    exp_dir = outputs_dir(repo_root, args.method)
    out_txt = os.path.join(exp_dir, "tracks.txt")
    out_vid = os.path.join(exp_dir, "tracks.mp4")
    out_cmp = os.path.join(exp_dir, "comparison.mp4")

    # Select method
    detections = pd.read_csv(det_path)
    if args.method == "overlap":
        tracked = track_by_max_overlap(
            detections,
            iou_match_thr=args.iou_thr,
            memory_frames=args.memory_frames,
            memory_iou_thr=args.memory_iou_thr,
        )
        # Convert to MOTChallenge format for saving
        tracked_mot = MOTChallengeConverter.dataframe_to_motchallenge(tracked)
    elif args.method == "kalman":
        tracked = execute_kalman_SORT(
            detections,
            max_age=1,
            min_hits=25,
            iou_threshold=args.iou_thr,
            show_tracks=False
        )

        tracked_mot = MOTChallengeConverter.dataframe_to_motchallenge(tracked)

    # Save CSV as TXT for TrackEval (required format)
    ensure_dir_for_file(out_txt)
    tracked_mot.to_csv(out_txt, index=False, header=False)

    # Render video with IDs
    if args.show_IDs_video:
        render_tracked_video(vid_path, tracked, out_vid, conf_thr=args.conf_thr_video, show_conf=True)
    
    # Render comparison video (detections vs tracks)
    if args.show_comp_video:
        render_comparison_video(vid_path, detections, tracked, out_cmp, conf_thr=args.conf_thr_video, show_conf=False)


if __name__ == "__main__":
    main()