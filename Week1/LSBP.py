import sys
from pathlib import Path
import os
sys.path.insert(0, str(Path(__file__).parents[1]))
import argparse
import numpy as np
import cv2
import cv2.bgsegm
from data_processor import AICityFrames
from tqdm import tqdm
from datetime import datetime
import json
from metrics import compute_map, compute_iou
from task1 import (
    preprocess_mask,
    detect_bboxes_in_frame,
)

# Unique run ID and output directory for this experiment
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = Path(__file__).parent / "results" / f"LSBP_{RUN_ID}"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LSBP background subtraction for vehicle detection")

    # Output options
    parser.add_argument("--save-videos", type=lambda x: x.lower() != "false", default=True,
                        help="Write visualisation videos")
    parser.add_argument("--save-mask-frames", type=lambda x: x.lower() != "false", default=True,
                        help="Save raw and processed mask frames to disk")

    # Scale
    parser.add_argument("--scale", type=float, default=0.25,
                        help="Frame resize factor to speed up experiments")

    # LSBP parameters
    parser.add_argument("--lsbp-radius", type=int, default=16,
                        help="Radius of the LSBP binary pattern neighbourhood (larger = more context, slower)")
    parser.add_argument("--t-lower", type=float, default=3.0,
                        help="Lower similarity threshold: smaller values = more foreground detections")

    # Post-processing / detection parameters
    parser.add_argument("--morph-kernel-size", type=int, default=7,
                        help="Morphological kernel size at scale 1, must be odd")
    parser.add_argument("--min-area", type=float, default=1000,
                        help="Minimum contour area to keep a bbox, at scale 1")
    parser.add_argument("--max-aspect-ratio", type=float, default=5.0,
                        help="Maximum aspect ratio (max/min side) for a bbox")
    parser.add_argument("--merge-iou-threshold", type=float, default=0.3,
                        help="IoU threshold to merge overlapping boxes")
    parser.add_argument("--merge-distance-threshold", type=float, default=100.0,
                        help="Pixel distance to merge nearby boxes, at scale 1")

    return parser.parse_args()


def build_lsbp_subtractor(args: argparse.Namespace) -> cv2.bgsegm.BackgroundSubtractorLSBP:
    return cv2.bgsegm.createBackgroundSubtractorLSBP(
        mc=cv2.bgsegm.LSBP_CAMERA_MOTION_COMPENSATION_NONE,
        LSBPRadius=args.lsbp_radius,
        Tlower=args.t_lower,
    )


def threshold_shadow(mask: np.ndarray) -> np.ndarray:
    """Replace shadow pixels (127) with background (0), keep foreground (255)."""
    return np.where(mask == 255, np.uint8(255), np.uint8(0))


def save_experiment(args: argparse.Namespace, result: dict) -> None:
    experiment = {
        "run_id": RUN_ID,
        "method": "LSBP",
        "params": {
            "scale": args.scale,
            "lsbp_radius": args.lsbp_radius,
            "t_lower": args.t_lower,
            "morph_kernel_size": args.morph_kernel_size,
            "min_area": args.min_area,
            "max_aspect_ratio": args.max_aspect_ratio,
            "merge_iou_threshold": args.merge_iou_threshold,
            "merge_distance_threshold": args.merge_distance_threshold,
        },
        "results": {
            "mAP": result["mAP"],
            "recall": result["recall"],
            "precision": result["precision"],
            "f1": result["f1"],
        },
    }

    with open(RUN_DIR / "results.json", "w") as f:
        json.dump(experiment, f, indent=2)

    print(f"Saved experiment to {RUN_DIR / 'results.json'}")


if __name__ == "__main__":

    args = parse_args()

    # Scale-adjusted parameters
    _ks = max(1, int(args.morph_kernel_size * args.scale))
    _ks = _ks if _ks % 2 == 1 else _ks + 1
    MORPH_KERNEL_SIZE_SCALED = (_ks, _ks)
    MIN_AREA_SCALED = args.min_area * (args.scale ** 2)
    MERGE_DISTANCE_THRESHOLD_SCALED = args.merge_distance_threshold * args.scale

    # Create dataloader
    dataloader = AICityFrames(scale=args.scale)
    total_frames = dataloader.frame_count
    warmup_end = int(total_frames * 0.25)
    print(f"Total frames: {total_frames}  |  Warm-up (skipped for eval): {warmup_end}")

    # Build LSBP subtractor
    subtractor = build_lsbp_subtractor(args)

    # Build ground truth
    # ============================================================================
    print("Building ground truth from dataloader...")
    ground_truth = {}
    label_to_class = {"car": 0, "bike": 0}

    for frame_idx in range(total_frames):
        boxes = dataloader.boxes(frame_idx)
        gt_boxes = []
        for box in boxes:
            if box.outside == 0 and box.label in label_to_class:
                if box.label == "car" and box.attributes.get("parked") == "true":
                    continue
                x = int(box.xtl * dataloader.scale)
                y = int(box.ytl * dataloader.scale)
                w = int((box.xbr - box.xtl) * dataloader.scale)
                h = int((box.ybr - box.ytl) * dataloader.scale)
                gt_boxes.append(((x, y, w, h), label_to_class[box.label]))
        if gt_boxes:
            ground_truth[frame_idx] = gt_boxes

    print(f"Loaded GT for {len(ground_truth)} frames")

    # Create output directories
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    if args.save_mask_frames:
        mask_frames_raw_path = RUN_DIR / "LSBP_mask_frames_raw"
        mask_frames_processed_path = RUN_DIR / "LSBP_mask_frames_processed"
        mask_frames_raw_path.mkdir(parents=True, exist_ok=True)
        mask_frames_processed_path.mkdir(parents=True, exist_ok=True)

    if args.save_videos:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        frame_h, frame_w = dataloader.image(0).shape[:2]
        output_path = RUN_DIR / "bg_mask_output.mp4"
        output_path_bboxes = RUN_DIR / "bg_mask_bboxes_output.mp4"
        output_path_comparison = RUN_DIR / "comparison_output.mp4"
        out = cv2.VideoWriter(str(output_path), fourcc, 10, (frame_w, frame_h))
        out_bboxes = cv2.VideoWriter(str(output_path_bboxes), fourcc, 10, (frame_w, frame_h))
        out_comparison = cv2.VideoWriter(str(output_path_comparison), fourcc, 10, (frame_w, frame_h))

    # Warm-up: feed 25% to the subtractor so it builds background model
    # Do NOT record predictions for evaluation.
    # ============================================================================
    for frame_idx in tqdm(range(warmup_end), "Warming up LSBP (first 25%%)"):
        frame = dataloader.image(frame_idx)
        subtractor.apply(frame)

    # Detection and prediction
    # ============================================================================
    predictions = {}

    for frame_idx in tqdm(range(warmup_end, total_frames - 1), "Processing frames"):

        frame = dataloader.image(frame_idx)

        # Raw mask from LSBP: 255 = foreground, 127 = shadow, 0 = background
        mask_raw = subtractor.apply(frame)

        # Remove shadow pixels before postprocessing
        mask = threshold_shadow(mask_raw)

        mask_processed = preprocess_mask(mask, MORPH_KERNEL_SIZE_SCALED)
        bboxes = detect_bboxes_in_frame(
            mask_processed,
            min_area_scaled=MIN_AREA_SCALED,
            max_aspect_ratio=args.max_aspect_ratio,
            merge_iou_threshold=args.merge_iou_threshold,
            merge_distance_threshold_scaled=MERGE_DISTANCE_THRESHOLD_SCALED,
            preprocessed=True,
        )
        predictions[frame_idx] = [(bbox, 1.0, 0) for bbox in bboxes]

        # Optional mask frame saving
        if args.save_mask_frames:
            cv2.imwrite(str(mask_frames_raw_path / f"mask_{frame_idx:06d}.jpg"),
                        cv2.cvtColor(mask_raw, cv2.COLOR_GRAY2BGR))
            cv2.imwrite(str(mask_frames_processed_path / f"mask_{frame_idx:06d}.jpg"),
                        mask_processed)

        # Optional video saving
        if args.save_videos:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            # Mask video (show thresholded mask, not raw, so shadows do not appear)
            out.write(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))

            # Bbox video
            frame_bboxes = frame_bgr.copy()
            for (x, y, w, h) in bboxes:
                cv2.rectangle(frame_bboxes, (x, y), (x + w, y + h), (0, 255, 0), 2)
            out_bboxes.write(frame_bboxes)

            # Comparison video (blue = GT, green = pred)
            gt_bboxes = ground_truth.get(frame_idx, [])
            pred_bboxes = predictions.get(frame_idx, [])
            frame_comparison = frame_bgr.copy()
            for (bbox, _) in gt_bboxes:
                x, y, w_box, h_box = bbox
                cv2.rectangle(frame_comparison, (x, y), (x + w_box, y + h_box), (255, 0, 0), 2)
            for (bbox, _, _) in pred_bboxes:
                x, y, w_box, h_box = bbox
                cv2.rectangle(frame_comparison, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
            for (gt_bbox, _) in gt_bboxes:
                for (pred_bbox, _, _) in pred_bboxes:
                    iou = compute_iou(gt_bbox, pred_bbox)
                    if iou > 0:
                        x = max(gt_bbox[0], pred_bbox[0])
                        y = max(gt_bbox[1], pred_bbox[1])
                        cv2.putText(frame_comparison, f"IoU: {iou:.2f}", (x, y - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            if frame_comparison.shape[1] == frame_w and frame_comparison.shape[0] == frame_h:
                out_comparison.write(frame_comparison)

    # Close video writers
    if args.save_videos:
        out.release()
        out_bboxes.release()
        out_comparison.release()
        print(f"Saved mask video to {output_path}")
        print(f"Saved bbox video to {output_path_bboxes}")
        print(f"Saved comparison video to {output_path_comparison}")

    # Evaluate
    # ============================================================================
    print("Computing mAP...")
    result = compute_map(predictions, ground_truth, num_classes=1, iou_threshold=0.5)
    print(f"\nmAP@0.5:   {result['mAP']:.4f}")
    print(f"Recall:    {result['recall']:.4f}")
    print(f"Precision: {result['precision']:.4f}")
    print(f"F1 Score:  {result['f1']:.4f}")

    # Save experiment results
    # ============================================================================
    save_experiment(args, result)