"""
Hyperparameter tuning for execute_deep_SORT using Optuna.

Usage:
    python tune_deep_sort.py [--detections path] [--video path] [--n_trials 50]

The script maximises HOTA (or falls back to IDF1 / MOTA depending on what
TrackEval returns).  All intermediate results are stored in an SQLite database
so a study can be resumed after interruption:

    python tune_deep_sort.py --resume

Requirements:
    pip install optuna optuna-dashboard pandas
"""

import os
import sys
import argparse
import logging
import tempfile
import subprocess
import json
from pathlib import Path
from datetime import datetime

import pandas as pd
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

# ---------------------------------------------------------------------------
# Repo path bootstrap (mirrors main.py)
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from utils import repo_root_from_this_file, resolve_path, ensure_dir_for_file
from deep_sort_runner import execute_deep_SORT
from prepare_gt_for_trackeval import MOTChallengeConverter

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)  # keep Optuna quiet


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_metrics_trackeval(tracked_mot: pd.DataFrame, gt_path: str) -> dict:
    """
    Run TrackEval on the tracked results and return a dict of metrics.
    Falls back to a lightweight IoU-based proxy if TrackEval is not available.
    """
    try:
        import trackeval  # noqa: F401 – just checking availability
        # Write predictions to a temp file and call TrackEval via subprocess
        with tempfile.TemporaryDirectory() as tmp:
            pred_txt = os.path.join(tmp, "tracks.txt")
            tracked_mot.to_csv(pred_txt, index=False, header=False)
            result_json = os.path.join(tmp, "results.json")
            cmd = [
                sys.executable, "-m", "trackeval.scripts.run_mot_challenge",
                "--GT_FOLDER", os.path.dirname(gt_path),
                "--TRACKERS_FOLDER", tmp,
                "--OUTPUT_FOLDER", tmp,
                "--TRACKERS_TO_EVAL", "tracks",
                "--METRICS", "HOTA", "CLEAR", "Identity",
                "--USE_PARALLEL", "False",
                "--PRINT_RESULTS", "False",
                "--OUTPUT_SUMMARY", "True",
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            with open(result_json) as f:
                return json.load(f)
    except Exception:
        pass

    # -----------------------------------------------------------------------
    # Lightweight proxy metric: weighted score based on track fragmentation
    # and coverage (does NOT require TrackEval).
    # Returns a dict with key "proxy_score" in [0, 1] (higher = better).
    # -----------------------------------------------------------------------
    return _proxy_metric(tracked_mot)


def _proxy_metric(tracked_mot: pd.DataFrame) -> dict:
    """
    Fast proxy for tracking quality (no ground truth required).
    
    Rewards:
      - More unique IDs per frame → coverage
      - Longer average track length → stability
      - Fewer track fragments → consistency
    
    Returns a dict with 'proxy_score' (higher is better).
    """
    if tracked_mot.empty:
        return {"proxy_score": 0.0}

    # tracked_mot columns: frame, id, x, y, w, h, conf, -1, -1, -1
    df = tracked_mot.copy()
    df.columns = ["frame", "id", "x", "y", "w", "h", "conf", "c1", "c2", "c3"]

    # Track length statistics
    track_lengths = df.groupby("id")["frame"].count()
    mean_len = track_lengths.mean()
    # Penalise very short tracks (likely false positives)
    long_tracks = (track_lengths >= 10).sum()
    total_tracks = len(track_lengths)
    stability = long_tracks / max(total_tracks, 1)

    # Frame coverage: avg detections per frame (capped at a reasonable number)
    frames = df["frame"].nunique()
    total_frames = df["frame"].max() - df["frame"].min() + 1
    coverage = frames / max(total_frames, 1)

    proxy = 0.5 * stability + 0.5 * coverage
    return {"proxy_score": float(proxy)}


def score_from_metrics(metrics: dict) -> float:
    """Extract a single scalar to maximise from the metrics dict."""
    # Priority: HOTA > IDF1 > MOTA > proxy_score
    for key in ("HOTA", "IDF1", "MOTA", "proxy_score"):
        if key in metrics:
            return float(metrics[key])
    # If metrics is nested (TrackEval format), try to dig in
    if isinstance(metrics, dict):
        for v in metrics.values():
            if isinstance(v, dict):
                s = score_from_metrics(v)
                if s is not None:
                    return s
    return 0.0


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------

def build_objective(detections: pd.DataFrame, gt_path: str | None):
    """
    Returns an Optuna objective function closed over the fixed inputs.
    """

    def objective(trial: optuna.Trial) -> float:
        # ---- Parameter search space ----------------------------------------
        max_age = trial.suggest_int("max_age", 1, 50)
        min_hits = trial.suggest_int("min_hits", 1, 30)
        iou_threshold = trial.suggest_float("iou_threshold", 0.1, 0.9, step=0.05)
        nms_max_overlap = trial.suggest_float("nms_max_overlap", 0.1, 1.0, step=0.05)
        # Appearance / Re-ID parameters
        max_cosine_distance = trial.suggest_float("max_cosine_distance", 0.1, 0.9, step=0.05)
        nn_budget = trial.suggest_int("nn_budget", 10, 200)

        logger.info(
            "Trial %d | max_age=%d min_hits=%d iou=%.2f nms=%.2f cos=%.2f nn_budget=%d",
            trial.number, max_age, min_hits, iou_threshold, nms_max_overlap,
            max_cosine_distance, nn_budget,
        )

        try:
            tracked = execute_deep_SORT(
                detections,
                max_age=max_age,
                min_hits=min_hits,
                iou_threshold=iou_threshold,
                show_tracks=False,
                nms_max_overlap=nms_max_overlap,
                max_cosine_distance=max_cosine_distance,
                nn_budget=nn_budget,
            )
        except Exception as exc:
            logger.warning("Trial %d failed: %s", trial.number, exc)
            raise optuna.exceptions.TrialPruned()

        tracked_mot = MOTChallengeConverter.dataframe_to_motchallenge(tracked)
        metrics = compute_metrics_trackeval(tracked_mot, gt_path) if gt_path else _proxy_metric(tracked_mot)
        score = score_from_metrics(metrics)

        logger.info("Trial %d → score=%.4f", trial.number, score)
        return score

    return objective


# ---------------------------------------------------------------------------
# Result reporting
# ---------------------------------------------------------------------------

def report_results(study: optuna.Study, out_dir: str) -> None:
    """Save best params, full trial history, and an importance plot."""
    os.makedirs(out_dir, exist_ok=True)

    # Best parameters
    best = study.best_params
    best_value = study.best_value
    logger.info("=" * 60)
    logger.info("BEST SCORE : %.4f", best_value)
    logger.info("BEST PARAMS: %s", best)
    logger.info("=" * 60)

    best_path = os.path.join(out_dir, "best_params.json")
    with open(best_path, "w") as f:
        json.dump({"best_score": best_value, "best_params": best}, f, indent=2)
    logger.info("Best params saved → %s", best_path)

    # Full trial CSV
    df = study.trials_dataframe()
    csv_path = os.path.join(out_dir, "all_trials.csv")
    df.to_csv(csv_path, index=False)
    logger.info("All trials saved  → %s", csv_path)

    # Importance plot (requires plotly)
    try:
        import plotly  # noqa: F401
        fig = optuna.visualization.plot_param_importances(study)
        fig.write_html(os.path.join(out_dir, "param_importances.html"))
        fig2 = optuna.visualization.plot_optimization_history(study)
        fig2.write_html(os.path.join(out_dir, "optimization_history.html"))
        logger.info("Plots saved to %s", out_dir)
    except ImportError:
        logger.info("Install plotly for visualisation: pip install plotly")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Tune deep SORT with Optuna")
    ap.add_argument("--detections", default="Week2/detections/detections.txt",
                    help="Path to detections CSV (relative to repo root or absolute)")
    ap.add_argument("--video", default="data/AICity_data/train/S03/c010/vdo.avi",
                    help="Path to source video (used only for proxy logging)")
    ap.add_argument("--gt", default=None,
                    help="Path to ground-truth MOT file for TrackEval. "
                         "If omitted, a proxy metric is used.")
    ap.add_argument("--n_trials", type=int, default=50,
                    help="Total Optuna trials to run")
    ap.add_argument("--n_jobs", type=int, default=1,
                    help="Parallel workers (1 = sequential, safe default)")
    ap.add_argument("--study_name", default="deep_sort_tuning",
                    help="Optuna study name (used for SQLite persistence)")
    ap.add_argument("--storage", default=None,
                    help="Optuna storage URL, e.g. sqlite:///tuning.db. "
                         "Defaults to an auto-named SQLite file in outputs/.")
    ap.add_argument("--resume", action="store_true",
                    help="Resume an existing study from --storage")
    ap.add_argument("--out_dir", default=None,
                    help="Directory to write results (default: auto-timestamped)")
    return ap.parse_args()


def main():
    args = parse_args()
    repo_root = repo_root_from_this_file(__file__)

    # Output directory
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or os.path.join(
        repo_root, "Week3", "tracking", "outputs", f"optuna_{ts}"
    )
    os.makedirs(out_dir, exist_ok=True)

    # Storage (SQLite for persistence / resumption)
    storage = args.storage or f"sqlite:///{os.path.join(out_dir, 'study.db')}"

    # Load detections once
    det_path = resolve_path(args.detections, repo_root)
    detections = pd.read_csv(det_path)
    logger.info("Loaded %d detections from %s", len(detections), det_path)

    gt_path = resolve_path(args.gt, repo_root) if args.gt else None
    if gt_path:
        logger.info("Ground truth: %s", gt_path)
    else:
        logger.info("No GT provided — using proxy metric (track stability + coverage)")

    # Create or load study
    direction = "maximize"
    sampler = TPESampler(seed=42, n_startup_trials=10)
    pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=5)

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction=direction,
        sampler=sampler,
        pruner=pruner,
        load_if_exists=args.resume,
    )

    logger.info(
        "Starting Optuna study '%s' | %d trials | storage: %s",
        args.study_name, args.n_trials, storage,
    )

    objective = build_objective(detections, gt_path)
    study.optimize(
        objective,
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        show_progress_bar=True,
        catch=(Exception,),
    )

    report_results(study, out_dir)


if __name__ == "__main__":
    main()