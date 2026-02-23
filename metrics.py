import numpy as np


def compute_iou(box1: tuple, box2: tuple) -> float:
    x1_1, y1_1, w1, h1 = box1
    x1_2, y1_2, w2, h2 = box2
    
    x2_1 = x1_1 + w1
    y2_1 = y1_1 + h1
    x2_2 = x1_2 + w2
    y2_2 = y1_2 + h2
    
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)
    
    inter_w = max(0, xi2 - xi1)
    inter_h = max(0, yi2 - yi1)
    inter_area = inter_w * inter_h
    
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area


def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """Compute AP using the Pascal VOC 11-point interpolation method.
    """
    ap = 0.0
    for r in np.arange(0.0, 1.1, 0.1):
        prec_at_recall = precisions[recalls >= r]
        p_interp = np.max(prec_at_recall) if prec_at_recall.size > 0 else 0.0
        ap += p_interp
    return ap / 11.0


def _compute_map_single_run(
    predictions: dict,
    ground_truth: dict,
    num_classes: int,
    iou_threshold: float,
    replace_confidence_at_random: bool,
) -> dict:
    aps = []
    total_tp = 0
    total_fp = 0
    total_gt = 0
    
    for class_id in range(num_classes):
        pred_boxes = []
        for frame_idx, preds in predictions.items():
            for bbox, score, cls in preds:
                if cls == class_id:
                    if replace_confidence_at_random:
                        score = np.random.rand()
                    pred_boxes.append((frame_idx, bbox, score))
        
        pred_boxes.sort(key=lambda x: x[2], reverse=True)
        
        gt_boxes = {}
        for frame_idx, gts in ground_truth.items():
            gt_boxes[frame_idx] = [(bbox, cls) for bbox, cls in gts if cls == class_id]
        
        tp = np.zeros(len(pred_boxes))
        fp = np.zeros(len(pred_boxes))
        matched = {frame_idx: set() for frame_idx in gt_boxes}
        
        for pred_idx, (frame_idx, pred_bbox, score) in enumerate(pred_boxes):
            if frame_idx not in gt_boxes or len(gt_boxes[frame_idx]) == 0:
                fp[pred_idx] = 1
                continue
            
            max_iou = 0
            max_gt_idx = -1
            for gt_idx, (gt_bbox, _) in enumerate(gt_boxes[frame_idx]):
                iou = compute_iou(pred_bbox, gt_bbox)
                if iou > max_iou:
                    max_iou = iou
                    max_gt_idx = gt_idx
            
            if max_iou >= iou_threshold and max_gt_idx not in matched[frame_idx]:
                tp[pred_idx] = 1
                matched[frame_idx].add(max_gt_idx)
            else:
                fp[pred_idx] = 1
        
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        num_gt = sum(len(boxes) for boxes in gt_boxes.values())
        total_gt += num_gt
        total_tp += int(tp_cumsum[-1]) if len(tp_cumsum) > 0 else 0
        total_fp += int(fp_cumsum[-1]) if len(fp_cumsum) > 0 else 0
        
        recalls = tp_cumsum / max(num_gt, 1)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
        
        ap = compute_ap(recalls, precisions)
        aps.append(ap)
    
    m_ap = np.mean(aps) if aps else 0.0
    recall = total_tp / max(total_gt, 1)
    precision = total_tp / max(total_tp + total_fp, 1)
    f1 = 2 * (precision * recall) / max(precision + recall, 1e-10)
    
    return {
        'mAP': m_ap,
        'AP_per_class': aps,
        'recall': recall,
        'precision': precision,
        'f1': f1,
    }


def compute_map(
    predictions: dict,  # {frame_idx: [(bbox, score, class_id), ...]}
    ground_truth: dict,  # {frame_idx: [(bbox, class_id), ...]}
    num_classes: int,
    *,
    iou_threshold: float = 0.5,
    replace_confidence_at_random: bool = True,
    N: int = 100
) -> dict:
    if not replace_confidence_at_random and N != 1:
        raise ValueError("N must be 1 when replace_confidence_at_random is False.")
    # Single deterministic run when not replacing confidence scores
    if not replace_confidence_at_random:
        return _compute_map_single_run(
            predictions, ground_truth, num_classes,
            iou_threshold, replace_confidence_at_random=False,
        )

    all_results = [
        _compute_map_single_run(
            predictions, ground_truth, num_classes,
            iou_threshold, replace_confidence_at_random=True,
        )
        for _ in range(N)
    ]

    mean_mAP = np.mean([r['mAP'] for r in all_results])
    std_mAP = np.std([r['mAP'] for r in all_results])
    mean_ap_per_class = np.mean(
        [r['AP_per_class'] for r in all_results], axis=0
    ).tolist()
    mean_recall = np.mean([r['recall'] for r in all_results])
    mean_precision = np.mean([r['precision'] for r in all_results])
    mean_f1 = np.mean([r['f1'] for r in all_results])

    return {
        'mAP': float(mean_mAP),
        'std_mAP': float(std_mAP),
        'AP_per_class': mean_ap_per_class,
        'recall': float(mean_recall),
        'precision': float(mean_precision),
        'f1': float(mean_f1),
    }
