import sys
from pathlib import Path
import os
sys.path.insert(0, str(Path(__file__).parents[1]))
import argparse
import numpy as np
import cv2
from data_processor import AICityFrames
from tqdm import tqdm
from datetime import datetime
import json
from metrics import compute_map, compute_iou

# Directory for raw mask frames (shared across experiments, always the same)
MASK_FRAMES_PATH_RAW = Path("mask_frames_raw")

# Unique run ID and output directory for this experiment
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = Path(__file__).parent / "results" / RUN_ID


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gaussian background subtraction for vehicle detection")

    # Output options
    parser.add_argument("--save-videos", type=lambda x: x.lower() != "false", default=True, help="Write visualisation videos")
    parser.add_argument("--save-mask-frames", type=lambda x: x.lower() != "false", default=True, help="Save raw and processed mask frames to disk")

    # Scale
    parser.add_argument("--scale", type=float, default=0.25, help="Frame resize factor to speed up experiments")

    # Tunable detection parameters
    parser.add_argument("--alpha", type=float, default=6.0, help="Background subtraction threshold multiplier")
    parser.add_argument("--morph-kernel-size", type=int, default=7, help="Morphological kernel size at scale 1, must be odd")
    parser.add_argument("--min-area", type=float, default=1000, help="Minimum contour area to keep a bbox, at scale 1")
    parser.add_argument("--max-aspect-ratio", type=float, default=5.0, help="Maximum aspect ratio (max/min side) for a bbox")
    parser.add_argument("--merge-iou-threshold", type=float, default=0.3, help="IoU threshold to merge overlapping boxes")
    parser.add_argument("--merge-distance-threshold", type=float, default=100.0, help="Pixel distance to merge nearby boxes, at scale 1")

    # Modelling mode
    parser.add_argument("--adaptive", action="store_true", help="Use adaptive Gaussian modelling instead of non-adaptive")
    parser.add_argument("--rho", type=float, default=0.01, help="Learning rate for adaptive modelling")

    return parser.parse_args()


class GaussianModelling:
    
    def __init__(self, dataloader: AICityFrames):
        self.dataloader = dataloader
        self._compute_bg_model()
    
    def _compute_bg_model(self):
        bg_model_count = int(self.dataloader.frame_count * 0.25)
        first_img = self.dataloader.image(0)
        bg_frames = np.empty((bg_model_count, *first_img.shape), dtype=first_img.dtype)
        
        for i in tqdm(range(bg_model_count), 'Computing background model parameters'):
            bg_frames[i] = self.dataloader.image(i)
        
        self.pixelwise_mean = np.mean(bg_frames, axis=0)
        self.pixelwise_std = np.std(bg_frames, axis=0)
        
        mean_img = (self.pixelwise_mean / self.pixelwise_mean.max() * 255).astype(np.uint8)
        std_img = (self.pixelwise_std / self.pixelwise_std.max() * 255).astype(np.uint8)
        cv2.imwrite('bg_mean.png', mean_img)
        cv2.imwrite('bg_std.png', std_img)
    
    def compute_bg_mask(self, image_idx: int, alpha: float):
        image = self.dataloader.image(image_idx).astype(np.float64)
        threshold = alpha * (self.pixelwise_std + 2)
        mask = np.abs(image - self.pixelwise_mean) >= threshold
        return mask.astype(np.uint8) * 255
    
    def compute_bg_mask_and_update(self, image_idx: int, alpha: float, rho: float):
        """Used for adaptive modelling"""
        image = self.dataloader.image(image_idx).astype(np.float64)
        threshold = alpha * (self.pixelwise_std + 2)
        is_fg = np.abs(image - self.pixelwise_mean) >= threshold # foreground
        mask = is_fg.astype(np.uint8) * 255

        # Update model only for background pixels
        is_bg = ~is_fg
        self.pixelwise_mean[is_bg] = rho * image[is_bg] + (1 - rho) * self.pixelwise_mean[is_bg]
        diff = image[is_bg] - self.pixelwise_mean[is_bg]
        self.pixelwise_std[is_bg] = np.sqrt(rho * diff ** 2 + (1 - rho) * self.pixelwise_std[is_bg] ** 2)

        return mask


def preprocess_mask(mask: np.ndarray, morph_kernel_size: tuple) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, morph_kernel_size)
    mask_opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask_closed = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE, kernel)
    return mask_closed


def merge_close_bboxes(bboxes: list, iou_threshold: float, distance_threshold: float) -> list:
    if len(bboxes) <= 1:
        return bboxes
    
    bboxes = list(bboxes)
    
    def boxes_overlap_or_close(b1, b2, dist_thresh):
        x1_1, y1_1, w1, h1 = b1
        x1_2, y1_2, w2, h2 = b2
        x2_1, y2_1 = x1_1 + w1, y1_1 + h1
        x2_2, y2_2 = x1_2 + w2, y1_2 + h2
        
        if x2_1 < x1_2 or x2_2 < x1_1 or y2_1 < y1_2 or y2_2 < y1_1:
            dx = max(0, max(x1_2 - x2_1, x1_1 - x2_2))
            dy = max(0, max(y1_2 - y2_1, y1_1 - y2_2))
            return (dx * dx + dy * dy) ** 0.5 <= dist_thresh
        return True
    
    def merge_boxes(b1, b2):
        x1 = min(b1[0], b2[0])
        y1 = min(b1[1], b2[1])
        w = max(b1[0] + b1[2], b2[0] + b2[2]) - x1
        h = max(b1[1] + b1[3], b2[1] + b2[3]) - y1
        return (x1, y1, w, h)
    
    merged = True
    while merged:
        merged = False
        for i in range(len(bboxes)):
            for j in range(i + 1, len(bboxes)):
                if i >= len(bboxes) or j >= len(bboxes):
                    continue
                
                x1_i, y1_i, w_i, h_i = bboxes[i]
                x1_j, y1_j, w_j, h_j = bboxes[j]
                
                inside_i_in_j = x1_i >= x1_j and y1_i >= y1_j and x1_i + w_i <= x1_j + w_j and y1_i + h_i <= y1_j + h_j
                inside_j_in_i = x1_j >= x1_i and y1_j >= y1_i and x1_j + w_j <= x1_i + w_i and y1_j + h_j <= y1_i + h_i
                
                if inside_i_in_j:
                    bboxes.pop(j)
                    merged = True
                    break
                elif inside_j_in_i:
                    bboxes.pop(i)
                    merged = True
                    break
                
                iou = compute_iou(bboxes[i], bboxes[j])
                if iou > iou_threshold:
                    bboxes[i] = merge_boxes(bboxes[i], bboxes[j])
                    bboxes.pop(j)
                    merged = True
                    break
                
                if boxes_overlap_or_close(bboxes[i], bboxes[j], distance_threshold):
                    bboxes[i] = merge_boxes(bboxes[i], bboxes[j])
                    bboxes.pop(j)
                    merged = True
                    break
            if merged:
                break
    
    return bboxes


def detect_bboxes_in_frame(mask: np.ndarray, min_area_scaled: float, max_aspect_ratio: float,
                           merge_iou_threshold: float, merge_distance_threshold_scaled: float,
                           morph_kernel_size: tuple = None, preprocessed: bool = False) -> list:
    mask_closed = mask if preprocessed else preprocess_mask(mask, morph_kernel_size)
    contours, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = max(w, h) / min(w, h)
        if w * h > min_area_scaled and aspect_ratio <= max_aspect_ratio:
            bboxes.append((x, y, w, h))
    bboxes = merge_close_bboxes(bboxes, iou_threshold=merge_iou_threshold,
                                 distance_threshold=merge_distance_threshold_scaled)
    return bboxes


def save_experiment(args: argparse.Namespace, result: dict) -> None:
    experiment = {
        "run_id": RUN_ID,
        "params": {
            "scale": args.scale,
            "alpha": args.alpha,
            "adaptive": args.adaptive,
            "rho": args.rho if args.adaptive else None,
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
        }
    }

    with open(RUN_DIR / "results.json", "w") as f:
        json.dump(experiment, f, indent=2)

    print(f"Saved experiment to {RUN_DIR / 'results.json'}")


if __name__ == '__main__':

    args = parse_args()

    # Scale-adjusted parameters derived from args
    _ks = max(1, int(args.morph_kernel_size * args.scale))
    _ks = _ks if _ks % 2 == 1 else _ks + 1  # keep odd for morphological kernels
    MORPH_KERNEL_SIZE_SCALED = (_ks, _ks)
    MIN_AREA_SCALED = args.min_area * (args.scale ** 2)
    MERGE_DISTANCE_THRESHOLD_SCALED = args.merge_distance_threshold * args.scale

    # Create dataloader
    dataloader = AICityFrames(scale=args.scale)
    print(f"Total frames: {dataloader.frame_count}")

    # Instantiate Gaussian Modelling
    gm = GaussianModelling(dataloader)
    
    # Build ground truth
    # ============================================================================
    print("Building ground truth from dataloader...")
    ground_truth = {}

    # As stated in the instructions, we map both classes to the same one
    label_to_class = {'car': 0, 'bike': 0}

    for frame_idx in range(dataloader.frame_count):
        boxes = dataloader.boxes(frame_idx)
        gt_boxes = []
        for box in boxes:
            if box.outside == 0 and box.label in label_to_class:
                if box.label == 'car' and box.attributes.get('parked') == 'true':
                    continue
                x = int(box.xtl * dataloader.scale)
                y = int(box.ytl * dataloader.scale)
                w = int((box.xbr - box.xtl) * dataloader.scale)
                h = int((box.ybr - box.ytl) * dataloader.scale)
                gt_boxes.append(((x, y, w, h), label_to_class[box.label]))
        if gt_boxes:
            ground_truth[frame_idx] = gt_boxes

    print(f"Loaded GT for {len(ground_truth)} frames")

    # Create the experiment output directory
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    # Create mask frame directories (if mask frame saving enabled)
    if args.save_mask_frames:
        MASK_FRAMES_PATH_RAW.mkdir(parents=True, exist_ok=True)
        mask_frames_path_processed = RUN_DIR / "mask_frames_processed"
        mask_frames_path_processed.mkdir(parents=True, exist_ok=True)

    # Initialise video writers (if video saving enabled)
    if args.save_videos:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        frame_h, frame_w = dataloader.image(0).shape[:2]
        output_path = RUN_DIR / "bg_mask_output.mp4"
        output_path_bboxes = RUN_DIR / "bg_mask_bboxes_output.mp4"
        output_path_comparison = RUN_DIR / "comparison_output.mp4"
        out = cv2.VideoWriter(str(output_path), fourcc, 10, (frame_w, frame_h))
        out_bboxes = cv2.VideoWriter(str(output_path_bboxes), fourcc, 10, (frame_w, frame_h))
        out_comparison = cv2.VideoWriter(str(output_path_comparison), fourcc, 10, (frame_w, frame_h))

    # Detection and prediction
    # ============================================================================
    predictions = {}

    for frame_idx in tqdm(range(int(dataloader.frame_count * 0.25), dataloader.frame_count - 1), 'Processing frames'):

        # Main processing
        # -------------------------------
        if args.adaptive:
            mask = gm.compute_bg_mask_and_update(frame_idx, args.alpha, args.rho)
        else:
            mask = gm.compute_bg_mask(frame_idx, args.alpha)
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
        # -------------------------------
        if args.save_mask_frames:
            mask_bgr_frames = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            cv2.imwrite(str(MASK_FRAMES_PATH_RAW          / f"mask_{frame_idx:06d}.jpg"), mask_bgr_frames)
            cv2.imwrite(str(mask_frames_path_processed     / f"mask_{frame_idx:06d}.jpg"), mask_processed)

        # Optional video saving
        # -------------------------------
        if args.save_videos:
            frame_bgr = cv2.cvtColor(dataloader.image(frame_idx), cv2.COLOR_GRAY2BGR)

            # Mask video
            mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            out.write(mask_bgr)

            # Bbox video
            frame_bboxes = frame_bgr.copy()
            for (x, y, w, h) in bboxes:
                cv2.rectangle(frame_bboxes, (x, y), (x + w, y + h), (0, 255, 0), 2)
            out_bboxes.write(frame_bboxes)

            # Comparison video
            gt_bboxes   = ground_truth.get(frame_idx, [])
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
    print(f"\nmAP@0.5: {result['mAP']:.4f}")
    print(f"Recall: {result['recall']:.4f}")
    print(f"Precision: {result['precision']:.4f}")
    print(f"F1 Score: {result['f1']:.4f}")

    # Save experiment results
    # ============================================================================
    save_experiment(args, result)