"""
Week 4 - Multi-Camera Tracking Pipeline
========================================
Modular pipeline that integrates:
  - Data processing
  - Fine-tuning (with optional k-fold cross-validation)
  - Detections  (precomputed file  OR  YOLO inference on video)
  - Single-camera tracking (DeepSORT / SORT / Kalman / Overlap)
  - [NEW] Multi-camera tracking

Optical flow is a standalone qualitative tool — it lives in optical_flow/ and
is NOT wired into this pipeline. Use run_optical_flow() below or call
optical_flow/main.py directly for image comparisons.

Usage:
    python main.py                        # uses config.yaml in same folder
    python main.py --config path/to/config.yaml
    python main.py --no-finetuning --single-cam-method kalman
"""

import argparse
import logging
from pathlib import Path
from types import SimpleNamespace

import yaml

# ---------------------------------------------------------------------------
# Intra-project imports
# ---------------------------------------------------------------------------
from data_processing.data_processor import extract_frames, AICityFrames

from finetuning.convert_to_yolo import main as convert_to_yolo_main
from finetuning.fine_tune import main as fine_tune_main
from finetuning.kfold_cross_validation import main as kfold_main

from tracking.main import main as tracking_main
from tracking.eval_tracking import main as eval_tracking_main
from tracking.summarize_results_tracking import main as summarize_results_main
from tracking.run_all_tracking import main as run_all_tracking_main

# YOLO inference — only used when detections.mode is "yolo"
from ultralytics import YOLO

"""
Multi-cam imports will have to match the methods we will implement:

from multi_cam_tracking.graph_matching import run as run_graph_matching
from multi_cam_tracking.appearance_clustering import run as run_appearance_clustering
from multi_cam_tracking.reid_based import run as run_reid_based
from multi_cam_tracking.zone_handoff import run as run_zone_handoff
"""

# Optical flow — imported only for the standalone helper below, not the pipeline
from optical_flow.main import main as optical_flow_main

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s - %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ===========================================================================
# CONFIG LOADER
# ===========================================================================
def load_config(config_path: str = "config.yaml") -> dict:
    """Load YAML config and return as a plain dict."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path.resolve()}")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    log.info("Config loaded from: %s", path.resolve())
    return cfg


# ===========================================================================
# STAGE 1 — Data processing
# ===========================================================================
def stage_data_processing(cfg: dict):
    dp         = cfg["data_processing"]
    output_dir = Path(dp["output_dir"])
    log.info("--- STAGE 1 | Data Processing ---")

    # Check whether frames already exist so we can skip extraction if requested
    existing_frames = list(output_dir.glob("frame_*.jpg")) if output_dir.exists() else []

    if dp.get("skip_if_exists", False) and existing_frames:
        log.info("  -> %d frames already found in '%s' — skipping extraction.",
                 len(existing_frames), output_dir)
    else:
        if not existing_frames:
            log.info("  -> No existing frames found, extracting from video ...")
        else:
            log.info("  -> Extracting frames (clean_output=%s) ...", dp["clean_output"])

        n_frames = extract_frames(
            video_path   = Path(dp["video_path"]),
            output_dir   = output_dir,
            clean_output = dp["clean_output"],
        )
        log.info("  -> Extracted %d frames to '%s'.", n_frames, output_dir)

    log.info("  -> Indexing annotations ...")
    dataset = AICityFrames(
        frames_dir       = output_dir,
        annotation_path  = Path(dp["annotation_path"]),
        image_index_base = dp["image_index_base"],
        scale            = dp["scale"],
    )
    log.info("  -> Annotation index built (%d frames with boxes).",
             len(list(dataset.frames_with_boxes())))

    log.info("Data processing complete.")
    return dataset


# ===========================================================================
# STAGE 2 — Fine-tuning
# ===========================================================================
def stage_finetuning(cfg: dict):
    ft = cfg["finetuning"]
    mode = ft["mode"]
    log.info("--- STAGE 2 | Fine-Tuning  [mode: %s] ---", mode)

    if mode == "skip":
        log.info("  -> Skipping fine-tuning, using weights at '%s'",
                 cfg["paths"]["model_weights"])
        return

    # Step 2a — convert annotations to YOLO dataset format
    log.info("  -> Converting annotations to YOLO format ...")
    c = ft["convert"]
    convert_to_yolo_main(SimpleNamespace(
        video       = c["video"],
        annotation  = c["annotation"],
        output_base = c["output_base"],
        folds       = c["folds"],
        seed        = c["seed"],
    ))

    # Step 2b — optional k-fold evaluation
    if ft["run_kfold"]:
        log.info("  -> Running k-fold cross-validation ...")
        kfold_main()

    # Step 2c — fine-tune YOLO
    log.info("  -> Fine-tuning model ...")
    t = ft["train"]
    fine_tune_main(SimpleNamespace(
        data            = t["data"],
        model           = t["model"],
        epochs          = t["epochs"],
        batch           = t["batch"],
        imgsz           = t["imgsz"],
        project         = t["project"],
        name            = t["name"],
        wandb_project   = t["wandb_project"],
        resume          = t["resume"],
        freeze          = t["freeze"],
        freeze_backbone = t["freeze_backbone"],
    ))

    log.info("Fine-tuning complete.")


# ===========================================================================
# STAGE 2b — Detections
# ===========================================================================
def stage_detections(cfg: dict) -> str:
    """
    Resolve where detections come from and return the final detections path
    that tracking will use.

    mode: "precomputed" — nothing to run, just return the configured path.
    mode: "yolo"        — run YOLO on the video, write results to yolo_output,
                          and return that path.
    """
    det = cfg["detections"]
    mode = det["mode"]
    log.info("--- Detections  [mode: %s] ---", mode)

    if mode == "precomputed":
        path = det["detections_path"]
        log.info("  -> Using precomputed detections: %s", path)
        return path

    if mode == "yolo":
        log.info("  -> Running YOLO inference ...")
        model = YOLO(det["yolo_weights"])
        results = model.predict(
            source = det["yolo_video"],
            conf   = det["yolo_conf"],
            imgsz  = det["yolo_imgsz"],
            stream = True,
        )

        out_path = det["yolo_output"]
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)

        with open(out_path, "w") as f:
            for frame_idx, r in enumerate(results):
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    cls  = int(box.cls[0])
                    f.write(f"{frame_idx},{cls},{conf},{x1:.2f},{y1:.2f},{x2:.2f},{y2:.2f}\n")

        log.info("  -> Detections written to: %s", out_path)
        return out_path

    raise ValueError(f"Unknown detections mode: '{mode}'. Choose 'precomputed' or 'yolo'.")


# ===========================================================================
# STAGE 3 — Single-camera tracking
# ===========================================================================
def stage_single_cam_tracking(cfg: dict, detections_path: str):
    sc  = cfg["single_cam_tracking"]
    ev  = cfg["evaluation"]
    sm  = cfg["summarize"]
    method = sc["method"]
    log.info("--- STAGE 3 | Single-Camera Tracking  [method: %s] ---", method)

    # --- 3a: run tracker ---------------------------------------------------
    exp_dir = str(Path(tracking_main(SimpleNamespace(
        method          = method,
        detections      = detections_path,
        video           = sc["video"],
        conf_thr_video  = sc["conf_thr_video"],
        iou_thr         = sc["iou_thr"],
        show_IDs_video  = sc["show_IDs_video"],
        show_comp_video = sc["show_comp_video"],
        memory_frames   = sc["memory_frames"],
        memory_iou_thr  = sc["memory_iou_thr"],
        output_dir      = sc["output_dir"],
    ))).resolve())
    log.info("  -> Tracker output written to: %s", exp_dir)

    # --- 3b: evaluate with TrackEval (HOTA + IDF1) -------------------------
    log.info("  -> Evaluating tracking ...")
    eval_tracking_main(SimpleNamespace(
        tracker_results = exp_dir,   # use the actual path created at runtime
        tracker_name    = ev["tracker_name"],
        gt_annotation   = ev["gt_annotation"],
        seq_name        = ev["seq_name"],
        seq_length      = ev["seq_length"],
        benchmark_name  = ev["benchmark_name"],
        split           = ev["split"],
    ))

    # --- 3c: aggregate and plot summaries ----------------------------------
    log.info("  -> Summarising results ...")
    summarize_results_main(SimpleNamespace(
        results_root = sm["results_root"],
        output_csv   = sm["output_csv"],
        plots_dir    = sm["plots_dir"],
    ))

    log.info("Single-camera tracking complete.")


# ===========================================================================
# STAGE 4 — Multi-camera tracking   <<<  WEEK 4  >>>
# ===========================================================================
def stage_run_all_tracking(cfg: dict) -> None:
    """
    Stage 3b — Run all single-camera trackings across every camera in a
    multi-camera sequence.  Discovers cameras automatically from data_root,
    runs each configured method × detection source, evaluates with TrackEval
    and writes per-camera summary CSVs + comparison plots.

    Designed to be called:
      (a) standalone via Stage 3b when stages.run_all_tracking is true, OR
      (b) as a pre-step inside stage_multi_cam_tracking before the multi-cam
          method runs (to ensure per-camera tracks exist).
    """
    ra  = cfg["run_all_tracking"]
    log.info("--- STAGE 3b | Run-All Single-Camera Tracking ---")
    log.info("  data_root  : %s", ra["data_root"])
    log.info("  output_dir : %s", ra["output_dir"])
    log.info("  methods    : %s", ra["methods"])

    run_all_tracking_main(SimpleNamespace(
        data_root          = ra["data_root"],
        repo_root          = ra["repo_root"],
        output_dir         = ra["output_dir"],
        detections_subpath = ra.get("detections_subpath", "det"),
        gt_subpath         = ra.get("gt_subpath", "gt/gt.txt"),
        benchmark_name     = ra.get("benchmark_name", "AICity"),
        split              = ra.get("split", "train"),
        methods            = ra["methods"],
    ))
    log.info("Run-all tracking complete.")


def stage_multi_cam_tracking(cfg: dict):
    mc = cfg["multi_cam_tracking"]
    method = mc["method"]
    log.info("--- STAGE 4 | Multi-Camera Tracking  [method: %s] ---", method)

    # Run per-camera single-cam tracking first if requested
    # (ensures all camera tracks exist before the multi-cam method runs)
    if cfg.get("run_all_tracking", {}).get("run_before_multicam", False):
        log.info("  -> Running per-camera tracking before multi-cam stage ...")
        stage_run_all_tracking(cfg)

    # Fill in when multi-cam methods are implemented:
    METHODS = {}
    """
    METHODS = {
        "graph_matching":        run_graph_matching,
        "appearance_clustering": run_appearance_clustering,
        "reid_based":            run_reid_based,
        "zone_handoff":          run_zone_handoff,
    }
    """

    if method not in METHODS:
        log.warning("Multi-cam method '%s' not yet implemented — skipping.", method)
        return

    METHODS[method](cfg)
    log.info("Multi-camera tracking complete.")


# ===========================================================================
# OPTICAL FLOW — standalone qualitative helper (not part of the pipeline)
# ===========================================================================
def run_optical_flow(cfg: dict):
    """
    Run optical flow for qualitative comparison between two images.
    Call this directly from a script or interactive session — it is intentionally
    decoupled from the main pipeline.

    Example:
        from main import load_config, run_optical_flow
        cfg = load_config("config.yaml")
        run_optical_flow(cfg)
    """
    of = cfg["optical_flow"]
    log.info("--- Optical Flow  [model: %s] ---", of["model"])

    optical_flow_main(SimpleNamespace(
        model           = of["model"],
        flowformer_ckpt = of["flowformer_ckpt"],
        img1_path       = of["img1_path"],
        img2_path       = of["img2_path"],
        gt_path         = of["gt_path"],
        plot            = of["plot"],
        plot_gt         = of["plot_gt"],
        plot_step       = of["plot_step"],
        plot_alpha      = of["plot_alpha"],
    ))

    log.info("Optical flow done.")


# ===========================================================================
# PIPELINE ORCHESTRATOR
# ===========================================================================
def _print_pipeline_summary(cfg: dict):
    """Print a human-readable summary of what the pipeline will do before running."""
    st  = cfg["stages"]
    det = cfg["detections"]
    ft  = cfg["finetuning"]
    sc  = cfg["single_cam_tracking"]
    ev  = cfg["evaluation"]
    sm  = cfg["summarize"]
    mc  = cfg["multi_cam_tracking"]
    of  = cfg["optical_flow"]

    ON  = "\033[92m✔\033[0m"   # green tick
    OFF = "\033[90m✘\033[0m"   # grey cross
    SEP = "─" * 52

    print()
    print("╔" + "═" * 52 + "╗")
    print("║    Week 4 — Multi-Camera Tracking Pipeline     ║")
    print("╚" + "═" * 52 + "╝")
    print()

    # ── Stages ──────────────────────────────────────────
    print("  STAGES")
    print(f"  {SEP}")

    print(f"  {ON if st['data_processing'] else OFF}  Stage 1 · Data Processing")
    if st["data_processing"]:
        dp = cfg["data_processing"]
        existing = list(Path(dp["output_dir"]).glob("frame_*.jpg")) if Path(dp["output_dir"]).exists() else []
        if dp.get("skip_if_exists", False) and existing:
            print(f"        Frames             : {len(existing)} already parsed — extraction will be SKIPPED")
        elif dp.get("skip_if_exists", False):
            print(f"        Frames             : none found yet — will extract from video")
        else:
            print(f"        Frames             : will extract from '{dp['video_path']}'")
            print(f"        Clean output       : {dp['clean_output']}")

    print(f"  {ON if st['finetuning'] else OFF}  Stage 2 · Fine-Tuning")
    print(f"        Detection source   : {det['mode'].upper()}")
    if det["mode"] == "precomputed":
        print(f"        Detections file    : {det['detections_path']}")
    else:
        print(f"        YOLO weights       : {det['yolo_weights']}")
        print(f"        YOLO video         : {det['yolo_video']}")
    if st["finetuning"]:
        print(f"        Finetune mode      : {ft['mode'].upper()}")
        if ft["mode"] == "finetune":
            t = ft["train"]
            print(f"        Base model         : {t['model']}")
            print(f"        Epochs             : {t['epochs']}  |  Batch: {t['batch']}  |  ImgSz: {t['imgsz']}")
            print(f"        K-Fold eval        : {'yes' if ft['run_kfold'] else 'no'}")

    print(f"  {ON if st['single_cam_tracking'] else OFF}  Stage 3 · Single-Camera Tracking")
    if st["single_cam_tracking"]:
        print(f"        Method             : {sc['method']}")
        print(f"        Video              : {sc['video']}")
        print(f"        IoU threshold      : {sc['iou_thr']}  |  Memory frames: {sc['memory_frames']}")
        print(f"        Evaluation         : seq={ev['seq_name']}  benchmark={ev['benchmark_name']}/{ev['split']}")
        print(f"        Tracker results    : set at runtime from tracking output")
        print(f"        Summary output     : {sm['output_csv']}  |  plots → {sm['plots_dir']}")

    print(f"  {ON if st['multi_cam_tracking'] else OFF}  Stage 4 · Multi-Camera Tracking")
    if st["multi_cam_tracking"]:
        print(f"        Method             : {mc['method']}")
        cams = mc.get("cameras", [])
        print(f"        Cameras            : {len(cams)}  ({', '.join(c['id'] for c in cams)})")

    print(f"  {SEP}")

    # ── Optical flow (standalone) ────────────────────────
    print()
    print("  OPTICAL FLOW  (standalone — not part of pipeline)")
    print(f"  {SEP}")
    print(f"        Model              : {of['model']}")
    print(f"        Image 1            : {of['img1_path']}")
    print(f"        Image 2            : {of['img2_path']}")
    print(f"        Plot modes         : {of['plot']}")
    print(f"  {SEP}")

    # ── Paths ────────────────────────────────────────────
    print()
    print("  PATHS")
    print(f"  {SEP}")
    print(f"        Output root        : {cfg['paths']['output_root']}")
    print(f"        Model weights      : {cfg['paths']['model_weights']}")
    print(f"  {SEP}")
    print()


def run_pipeline(cfg: dict):
    _print_pipeline_summary(cfg)

    Path(cfg["paths"]["output_root"]).mkdir(parents=True, exist_ok=True)

    if cfg["stages"]["data_processing"]:
        stage_data_processing(cfg)

    if cfg["stages"]["finetuning"]:
        stage_finetuning(cfg)

    # Detections are always resolved before tracking — either from file or YOLO
    detections_path = stage_detections(cfg)

    if cfg["stages"]["single_cam_tracking"]:
        stage_single_cam_tracking(cfg, detections_path)

    if cfg["stages"].get("run_all_tracking", False):
        stage_run_all_tracking(cfg)

    if cfg["stages"]["multi_cam_tracking"]:
        stage_multi_cam_tracking(cfg)

    log.info("Pipeline finished")


# ===========================================================================
# CLI entry point
# ===========================================================================
def _parse_args():
    parser = argparse.ArgumentParser(
        description="Week 4 multi-camera tracking pipeline"
    )

    parser.add_argument("--config", default="config.yaml",
                        help="Path to YAML config file (default: config.yaml)")

    # Stage toggle overrides
    parser.add_argument("--no-data-processing",  action="store_true")
    parser.add_argument("--no-finetuning",        action="store_true")
    parser.add_argument("--no-single-cam",        action="store_true")
    parser.add_argument("--no-multi-cam",         action="store_true")

    # Quick detection/finetuning overrides
    parser.add_argument("--detections-mode",
                        choices=["precomputed", "yolo"],
                        help="Override detections.mode from config")
    parser.add_argument("--detections-path",
                        dest="detections_path",
                        help="Override detections.detections_path from config")
    parser.add_argument("--finetune-mode",
                        choices=["finetune", "skip"],
                        help="Override finetuning.mode from config")
    parser.add_argument("--single-cam-method",
                        choices=["overlap", "overlap_flow", "kalman",
                                 "deep_SORT", "deep_SORT_flow"])
    parser.add_argument("--multi-cam-method",
                        choices=["graph_matching", "appearance_clustering",
                                 "reid_based", "zone_handoff"])

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg  = load_config(args.config)

    # Apply CLI overrides on top of the loaded config
    if args.no_data_processing: cfg["stages"]["data_processing"]     = False
    if args.no_finetuning:      cfg["stages"]["finetuning"]          = False
    if args.no_single_cam:      cfg["stages"]["single_cam_tracking"] = False
    if args.no_multi_cam:       cfg["stages"]["multi_cam_tracking"]  = False

    if args.detections_mode:    cfg["detections"]["mode"]                = args.detections_mode
    if args.detections_path:    cfg["detections"]["detections_path"]     = args.detections_path
    if args.finetune_mode:      cfg["finetuning"]["mode"]                = args.finetune_mode
    if args.single_cam_method:  cfg["single_cam_tracking"]["method"]     = args.single_cam_method
    if args.multi_cam_method:   cfg["multi_cam_tracking"]["method"]      = args.multi_cam_method

    run_pipeline(cfg)