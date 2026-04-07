import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import argparse
import pandas as pd
from datetime import datetime

from tracking.utils import repo_root_from_this_file, resolve_path, ensure_dir_for_file, render_tracked_video, render_comparison_video
from tracking.overlap import track_by_max_overlap, track_by_max_overlap_flow
from tracking.kalman import execute_kalman_SORT
from tracking.deep_sort_runner import execute_deep_SORT
from tracking.deep_sort_flow import execute_deep_SORT_flow
from tracking.prepare_gt_for_trackeval import MOTChallengeConverter

# Path functions
def add_repo_root_to_syspath(repo_root: str) -> None:
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
 
def outputs_dir(base: str, method: str) -> str:
    """Build a timestamped output directory under base/."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(base, f"{method}_{ts}")
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir
 
 
def main(args=None):
    """
    Run single-camera tracking.
 
    Can be called in two ways:
      - From CLI:            python main.py --method deep_SORT ...
      - From pipeline:       main(SimpleNamespace(method="deep_SORT", ...))
 
    When args is None, arguments are parsed from the command line as usual.
    When args is provided (e.g. a SimpleNamespace), argparse is skipped entirely.
    """
    repo_root = repo_root_from_this_file(__file__)
    add_repo_root_to_syspath(repo_root)
 
    if args is None:
        ap = argparse.ArgumentParser()
        ap.add_argument("--method", choices=["overlap", "kalman", "overlap_flow", "deep_SORT", "deep_SORT_flow"], default="overlap")
        ap.add_argument("--detections", default="Week2/detections/detections.txt")
        ap.add_argument("--video", default="data/AICity_data/train/S03/c010/vdo.avi")
        ap.add_argument("--conf_thr_video", type=float, default=0.30)
        ap.add_argument("--iou_thr", type=float, default=0.65)
        ap.add_argument("--show_IDs_video", default=True)
        ap.add_argument("--show_comp_video", default=True)
 
        ### MEMORY PARAMETERS (for re-identification of lost tracks)
        ap.add_argument("--memory_frames", type=int, default=20,
                        help="Number of frames to keep lost tracks in memory (0 = disabled)")
        ap.add_argument("--memory_iou_thr", type=float, default=0.4,
                        help="Min IoU to re-identify a detection with a remembered track")
        ap.add_argument("--output_dir", default=None,
                        help="Base directory for output files (default: <repo_root>/Week4/tracking/outputs)")
        args = ap.parse_args()
 
    # Inputs
    det_path = resolve_path(args.detections, repo_root)
    vid_path = resolve_path(args.video, repo_root)
 
    # Outputs — use explicit output_dir if provided, otherwise default to Week4/tracking/outputs
    base_out = args.output_dir if getattr(args, "output_dir", None) else os.path.join(repo_root, "Week4", "tracking", "outputs")
    exp_dir  = outputs_dir(base_out, args.method)
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
        tracked_mot = MOTChallengeConverter.dataframe_to_motchallenge(tracked)
 
    elif args.method == "overlap_flow":
        tracked = track_by_max_overlap_flow(
            detections,
            iou_match_thr=args.iou_thr,
            memory_frames=args.memory_frames,
            memory_iou_thr=args.memory_iou_thr,
            video_path=vid_path
        )
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
 
    elif args.method == "deep_SORT":
        tracked = execute_deep_SORT(
            detections,
            max_age=15,
            min_hits=30,
            iou_threshold=0.65,
            show_tracks=False,
            nms_max_overlap=0.9,
            max_cosine_distance=0.5,
            nn_budget=122
        )
        tracked_mot = MOTChallengeConverter.dataframe_to_motchallenge(tracked)
 
    elif args.method == "deep_SORT_flow":
        tracked = execute_deep_SORT_flow(
            detections,
            video_path=vid_path,
            max_age=15,
            min_hits=30,
            iou_threshold=0.65,
            show_tracks=False,
            nms_max_overlap=0.9,
            max_cosine_distance=0.5,
            nn_budget=122
        )
        tracked_mot = MOTChallengeConverter.dataframe_to_motchallenge(tracked)
 
    else:
        raise ValueError(f"Unknown tracking method: '{args.method}'")
 
    # Save tracks as TXT for TrackEval (required format)
    os.makedirs(exp_dir, exist_ok=True)   # guarantee dir exists before writing
    tracked_mot.to_csv(out_txt, index=False, header=False)
    print(f"  -> Tracks saved to: {out_txt}")
 
    # Render video with IDs
    # Coerce to bool — argparse may deliver 'False' as a string when called
    # from a pipeline via SimpleNamespace, which is truthy as a string.
    show_ids = args.show_IDs_video
    if isinstance(show_ids, str):
        show_ids = show_ids.strip().lower() not in ('false', '0', 'no', '')
    show_cmp = args.show_comp_video
    if isinstance(show_cmp, str):
        show_cmp = show_cmp.strip().lower() not in ('false', '0', 'no', '')

    if show_ids:
        render_tracked_video(vid_path, tracked, out_vid, conf_thr=args.conf_thr_video, show_conf=True)
        print(f"  -> Tracked video saved to: {out_vid}")

    # Render comparison video (detections vs tracks)
    if show_cmp:
        render_comparison_video(vid_path, detections, tracked, out_cmp, conf_thr=args.conf_thr_video, show_conf=False)
        print(f"  -> Comparison video saved to: {out_cmp}")
 
    return exp_dir   # caller uses this as tracker_results for evaluation
 
 
if __name__ == "__main__":
    main()