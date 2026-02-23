import argparse
import json
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from metrics import compute_iou


def preprocess_mask(mask: np.ndarray, morph_kernel_size: tuple) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, morph_kernel_size)
    mask_opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask_closed = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE, kernel)
    return mask_closed


def merge_close_bboxes(
    bboxes: List[Tuple[int, int, int, int]],
    iou_threshold: float,
    distance_threshold: float
) -> List[Tuple[int, int, int, int]]:

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

                inside_i_in_j = (
                    x1_i >= x1_j and y1_i >= y1_j and
                    x1_i + w_i <= x1_j + w_j and
                    y1_i + h_i <= y1_j + h_j
                )
                inside_j_in_i = (
                    x1_j >= x1_i and y1_j >= y1_i and
                    x1_j + w_j <= x1_i + w_i and
                    y1_j + h_j <= y1_i + h_i
                )

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


def detect_bboxes_in_frame(
    mask: np.ndarray,
    min_area_scaled: float,
    max_aspect_ratio: float,
    merge_iou_threshold: float,
    merge_distance_threshold_scaled: float,
    morph_kernel_size: tuple = None,
    preprocessed: bool = False
) -> list:

    mask_closed = mask if preprocessed else preprocess_mask(mask, morph_kernel_size)
    contours, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bboxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = max(w, h) / min(w, h)
        if w * h > min_area_scaled and aspect_ratio <= max_aspect_ratio:
            bboxes.append((x, y, w, h))

    return merge_close_bboxes(
        bboxes,
        iou_threshold=merge_iou_threshold,
        distance_threshold=merge_distance_threshold_scaled,
    )


def threshold_shadow_to_fg(mask_raw: np.ndarray) -> np.ndarray:
    return np.where(mask_raw == 255, np.uint8(255), np.uint8(0))


def save_experiment(
    run_dir: Path,
    run_id: str,
    args: argparse.Namespace,
    result: dict
) -> None:

    params = {
        "scale": args.scale,
        "morph_kernel_size": args.morph_kernel_size,
        "min_area": args.min_area,
        "max_aspect_ratio": args.max_aspect_ratio,
        "merge_iou_threshold": args.merge_iou_threshold,
        "merge_distance_threshold": args.merge_distance_threshold,
        "num_random_ranks": args.num_random_ranks,
        "seed": args.seed,
    }

    if args.method == "gaussian":
        params.update({
            "alpha": args.alpha,
            "adaptive": args.adaptive,
            "rho": args.rho if args.adaptive else None,
        })
    elif args.method == "mog2":
        params.update({
            "mog2_history": args.mog2_history,
            "mog2_var_threshold": args.mog2_var_threshold,
            "mog2_detect_shadows": args.mog2_detect_shadows,
        })
    elif args.method == "lsbp":
        params.update({
            "lsbp_radius": args.lsbp_radius,
            "lsbp_t_lower": args.t_lower,
        })

    experiment = {
        "run_id": run_id,
        "method": args.method,
        "params": params,
        "results": {
            "mAP": result["mAP"],
            "recall": result["recall"],
            "precision": result["precision"],
            "f1": result["f1"],
        }
    }

    with open(run_dir / "results.json", "w") as f:
        json.dump(experiment, f, indent=2)