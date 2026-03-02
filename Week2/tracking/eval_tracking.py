import os
import sys
import argparse
from pathlib import Path
import pandas as pd
import shutil

# Add repo root to path
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Add TrackEval to path
TRACKEVAL_PATH = os.path.join(REPO_ROOT, "external", "TrackEval")
if TRACKEVAL_PATH not in sys.path:
    sys.path.insert(0, TRACKEVAL_PATH)

import trackeval
from prepare_gt_for_trackeval import MOTChallengeConverter


def prepare_tracker_file(tracker_csv: Path, output_file: Path):
    """
    Prepare tracker results in MOTChallenge format.
    
    Input CSV should have columns: frame, id, x, y, width, height, conf
    or frame_id, track_id, x1, y1, x2, y2, confidence
    Output should be: frame, id, x, y, width, height, conf, -1, -1, -1
    """
    print(f"Reading tracker results from {tracker_csv}...")
    
    # Try reading with header first
    df = pd.read_csv(tracker_csv)
    
    # Check if it's in the original format (frame_id, x1, y1, x2, y2, etc.)
    if 'frame_id' in df.columns:
        print("  Converting from original CSV format to MOTChallenge format...")
        # Use the converter class
        df = MOTChallengeConverter.dataframe_to_motchallenge(df, is_ground_truth=False)
    # Check if it's already in MOTChallenge format with header
    elif 'frame' in df.columns and 'id' in df.columns:
        print("  File already in MOTChallenge format...")
        required_cols = ['frame', 'id', 'x', 'y', 'width', 'height', 'conf']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"CSV missing required columns. Found: {df.columns.tolist()}")
        # Ensure world coordinates are present
        if 'x_world' not in df.columns:
            df['x_world'] = -1
            df['y_world'] = -1
            df['z_world'] = -1
    else:
        # No recognizable headers - might be already in correct format without header
        print("  Checking if file is in MOTChallenge format (no header)...")
        df = pd.read_csv(tracker_csv, header=None)
        if len(df.columns) >= 7:
            print("  File appears to be in MOTChallenge format (no header)")
            output_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(tracker_csv, output_file)
            return
        else:
            raise ValueError(f"Cannot parse tracker CSV file. Columns: {df.columns.tolist()}")
    
    # Prepare output directory
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Save without header
    print(f"Writing MOTChallenge format to {output_file}...")
    df[['frame', 'id', 'x', 'y', 'width', 'height', 'conf', 'x_world', 'y_world', 'z_world']].to_csv(
        output_file, index=False, header=False
    )
    print(f"Tracker file prepared: {output_file}")


def run_trackeval(
    gt_folder: Path,
    trackers_folder: Path,
    tracker_name: str,
    seq_name: str,
    benchmark_name: str = "AICity",
    split: str = "train"
):
    """Run TrackEval with HOTA and IDF1 metrics."""
    
    print("\n" + "="*60)
    print("Running TrackEval Evaluation")
    print("="*60)
    
    # Configure evaluation
    eval_config = {
        'USE_PARALLEL': False,
        'NUM_PARALLEL_CORES': 1,
        'BREAK_ON_ERROR': True,
        'RETURN_ON_ERROR': False,
        'LOG_ON_ERROR': os.path.join(str(trackers_folder), 'error_log.txt'),
        'PRINT_RESULTS': True,
        'PRINT_ONLY_COMBINED': False,
        'PRINT_CONFIG': True,
        'TIME_PROGRESS': True,
        'DISPLAY_LESS_PROGRESS': True,
        'OUTPUT_SUMMARY': True,
        'OUTPUT_DETAILED': True,
        'PLOT_CURVES': True,
    }
    
    # Configure dataset
    dataset_config = {
        'GT_FOLDER': str(gt_folder),
        'TRACKERS_FOLDER': str(trackers_folder),
        'OUTPUT_FOLDER': None,  # Will use TRACKERS_FOLDER
        'TRACKERS_TO_EVAL': [tracker_name],
        'CLASSES_TO_EVAL': ['pedestrian'],  # MOTChallenge uses 'pedestrian' even for cars
        'BENCHMARK': benchmark_name,
        'SPLIT_TO_EVAL': split,
        'INPUT_AS_ZIP': False,
        'PRINT_CONFIG': True,
        'DO_PREPROC': False,  # Don't preprocess (no distractor removal)
        'TRACKER_SUB_FOLDER': 'data',
        'OUTPUT_SUB_FOLDER': '',
        'TRACKER_DISPLAY_NAMES': None,
        'SEQMAP_FOLDER': None,  # Will use default
        'SEQMAP_FILE': None,
        'SEQ_INFO': {seq_name: None},  # Specify sequence to evaluate
        'GT_LOC_FORMAT': '{gt_folder}/{seq}/gt/gt.txt',
        'SKIP_SPLIT_FOL': False,
    }
    
    # Configure metrics
    metrics_config = {
        'METRICS': ['HOTA', 'Identity'],  # HOTA and Identity which includes IDF1
        'THRESHOLD': 0.5, # Important in IDF1
    }
    
    # Create evaluator
    evaluator = trackeval.Evaluator(eval_config)
    
    # Create dataset
    dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
    
    # Create metrics
    metrics_list = []
    for metric in [trackeval.metrics.HOTA, trackeval.metrics.Identity]:
        if metric.get_name() in metrics_config['METRICS']:
            metrics_list.append(metric(metrics_config))
    
    if len(metrics_list) == 0:
        raise Exception('No metrics selected for evaluation')
    
    # Run evaluation
    print("\nStarting evaluation...")
    output_res, output_msg = evaluator.evaluate(dataset_list, metrics_list)
    
    print("\n" + "="*60)
    print("Evaluation Complete!")
    print("="*60)
    
    return output_res, output_msg


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate tracking results with TrackEval (HOTA, IDF1)"
    )
    parser.add_argument(
        "--tracker-results",
        required=True,
        help="Path to tracker output directory (containing tracks.csv or tracks.txt)"
    )
    parser.add_argument(
        "--tracker-name",
        default=None,
        help="Name for this tracker (default: derived from folder name)"
    )
    parser.add_argument(
        "--gt-annotation",
        default="data/ai_challenge_s03_c010-full_annotation.xml",
        help="Path to ground truth XML annotation"
    )
    parser.add_argument(
        "--seq-name",
        default="S03c010",
        help="Sequence name"
    )
    parser.add_argument(
        "--benchmark-name",
        default="AICity",
        help="Benchmark name for folder structure"
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Split name (train/test)"
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    tracker_results_dir = Path(args.tracker_results)
    if not tracker_results_dir.is_absolute():
        tracker_results_dir = Path(REPO_ROOT) / tracker_results_dir
    
    print(tracker_results_dir)
    if not tracker_results_dir.exists():
        raise FileNotFoundError(f"Tracker results directory not found: {tracker_results_dir}")
    
    # Find tracker output file
    tracker_file = None
    for fname in ['tracks.txt', 'tracks.csv']:
        candidate = tracker_results_dir / fname
        if candidate.exists():
            tracker_file = candidate
            break
    
    if tracker_file is None:
        raise FileNotFoundError(f"No tracks.txt or tracks.csv found in {tracker_results_dir}")
    
    # Determine tracker name
    if args.tracker_name:
        tracker_name = args.tracker_name
    else:
        tracker_name = tracker_results_dir.name
    
    print(f"\nTracker: {tracker_name}")
    print(f"Results: {tracker_file}")
    
    # Set up folder structure
    gt_base = Path(REPO_ROOT) / "data" / "gt_mot_format"
    trackers_base = Path(REPO_ROOT) / "data" / "trackers_mot_format"
    
    benchmark_split = f"{args.benchmark_name}-{args.split}"
    
    gt_folder = gt_base / benchmark_split
    trackers_folder = trackers_base / benchmark_split
    
    gt_seq_folder = gt_folder / args.seq_name
    tracker_data_folder = trackers_folder / tracker_name / "data"
    
    print(f"\nPreparing folder structure...")
    print(f"  GT folder: {gt_folder}")
    print(f"  Trackers folder: {trackers_folder}")
    
    # Step 1: Prepare ground truth (if not already done)
    gt_file = gt_seq_folder / "gt" / "gt.txt"
    if not gt_file.exists():
        print(f"\n[1/2] Converting ground truth to MOTChallenge format...")
        annotation_path = Path(REPO_ROOT) / args.gt_annotation
        MOTChallengeConverter.ground_truth_to_motchallenge(
            annotation_path=annotation_path,
            output_file=gt_file,
            class_filter="car",
            verbose=True
        )
    else:
        print(f"\n[1/2] Ground truth already exists: {gt_file}")
    
    # Step 2: Prepare tracker results
    print(f"\n[2/2] Preparing tracker results...")
    tracker_output_file = tracker_data_folder / f"{args.seq_name}.txt"
    prepare_tracker_file(tracker_file, tracker_output_file)
    
    # Run TrackEval
    print(f"\nRunning evaluation...")
    results, messages = run_trackeval(
        gt_folder=gt_base,
        trackers_folder=trackers_base,
        tracker_name=tracker_name,
        seq_name=args.seq_name,
        benchmark_name=args.benchmark_name,
        split=args.split,
    )
    
    print(f"Evaluation complete!")
    print(f"\nResults saved in: {trackers_folder / tracker_name}")


if __name__ == "__main__":
    main()
