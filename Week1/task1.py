import sys
from pathlib import Path
import os

sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np
import cv2
from data_processor import AICityFrames
from tqdm import tqdm
from metrics import compute_map, compute_iou


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
    
    def compute_bg_mask(self, image_idx: int, k=1):
        image = self.dataloader.image(image_idx).astype(np.float64)
        threshold = k * (self.pixelwise_std + 2)
        mask = np.abs(image - self.pixelwise_mean) >= threshold
        return mask.astype(np.uint8) * 255


def preprocess_mask(mask: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    mask_opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    mask_closed = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE, kernel)
    return mask_closed


def merge_close_bboxes(bboxes: list, iou_threshold: float = 0.5, distance_threshold: float = 50.0) -> list:
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
    
    while True:
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
        
        break
    
    return bboxes


def detect_bboxes_in_frame(mask: np.ndarray) -> list:
    mask_closed = preprocess_mask(mask)
    contours, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h > 300:
            bboxes.append((x, y, w, h))
    bboxes = merge_close_bboxes(bboxes, iou_threshold=0.3)
    return bboxes


if __name__ == '__main__':
    dataloader = AICityFrames(scale=0.5)
    h, w = dataloader.image(0).shape
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    
    MASK_FRAMES_PATH_RAW = Path("mask_frames_raw")
    MASK_FRAMES_PATH_PROCESSED = Path("mask_frames_processed")
    MASK_FRAMES_PATH_RAW.mkdir(parents=True, exist_ok=True)
    MASK_FRAMES_PATH_PROCESSED.mkdir(parents=True, exist_ok=True)
    
    generate_masks = True
    generate_bboxes = True
    evaluate = True
    
    if generate_masks:
        gm = GaussianModelling(dataloader)
        
        output_path = Path(__file__).parent / "bg_mask_output.mp4"
        out = cv2.VideoWriter(str(output_path), fourcc, 10, (w, h))
        
        print(f"Total frames: {dataloader.frame_count}")
        
        for frame_idx in tqdm(range(int(dataloader.frame_count * 0.25), dataloader.frame_count - 1), 'Processing frames'):
            mask = gm.compute_bg_mask(frame_idx, 6)
            mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            out.write(mask_bgr)
            cv2.imwrite(MASK_FRAMES_PATH_RAW / f"mask_{frame_idx:06d}.jpg", mask_bgr)
        
        out.release()
        print(f"Saved video to {output_path}")
    
    if generate_bboxes:
        output_path_bboxes = Path(__file__).parent / "bg_mask_bboxes_output.mp4"
        out_bboxes = cv2.VideoWriter(str(output_path_bboxes), fourcc, 10, (w, h))
        
        for frame_idx in tqdm(range(int(dataloader.frame_count * 0.25), dataloader.frame_count - 1), 'Processing frames with bboxes'):
            frame = dataloader.image(frame_idx)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            
            mask = cv2.imread(str(MASK_FRAMES_PATH_RAW / f"mask_{frame_idx:06d}.jpg"), cv2.IMREAD_GRAYSCALE)
            mask_processed = preprocess_mask(mask)
            cv2.imwrite(str(MASK_FRAMES_PATH_PROCESSED / f"mask_{frame_idx:06d}.jpg"), mask_processed)
            
            bboxes = detect_bboxes_in_frame(mask)
            
            for (x, y, w, h) in bboxes:
                cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            out_bboxes.write(frame_bgr)
        
        out_bboxes.release()
        print(f"Saved video with bboxes to {output_path_bboxes}")
    
    if evaluate:
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
        
        print("Running detection on all frames...")
        predictions = {}
        for frame_idx in tqdm(range(int(dataloader.frame_count * 0.25), dataloader.frame_count - 1)):
            mask_path = MASK_FRAMES_PATH_PROCESSED / f"mask_{frame_idx:06d}.jpg"
            if mask_path.exists():
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                bboxes = detect_bboxes_in_frame(mask)
                if bboxes:
                    predictions[frame_idx] = [(bbox, 1.0, 0) for bbox in bboxes]
        
        print(f"Total predictions: {sum(len(v) for v in predictions.values())}")
        
        print("Creating comparison video with GT and predicted bboxes...")
        output_path_comparison = Path(__file__).parent / "comparison_output.mp4"
        
        frame_h, frame_w = dataloader.image(0).shape[:2]
        out_comparison = cv2.VideoWriter(str(output_path_comparison), cv2.VideoWriter_fourcc(*'mp4v'), 10, (frame_w, frame_h))
        
        for frame_idx in tqdm(range(int(dataloader.frame_count * 0.25), dataloader.frame_count - 1), 'Creating comparison video'):
            frame = dataloader.image(frame_idx)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            
            gt_bboxes = ground_truth.get(frame_idx, [])
            pred_bboxes = predictions.get(frame_idx, [])
            
            for (bbox, class_id) in gt_bboxes:
                x, y, w_box, h_box = bbox
                cv2.rectangle(frame_bgr, (x, y), (x + w_box, y + h_box), (255, 0, 0), 2)
            
            for (bbox, score, class_id) in pred_bboxes:
                x, y, w_box, h_box = bbox
                cv2.rectangle(frame_bgr, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
            
            for (gt_bbox, _) in gt_bboxes:
                for (pred_bbox, _, _) in pred_bboxes:
                    iou = compute_iou(gt_bbox, pred_bbox)
                    if iou > 0:
                        x = max(gt_bbox[0], pred_bbox[0])
                        y = max(gt_bbox[1], pred_bbox[1])
                        cv2.putText(frame_bgr, f"IoU: {iou:.2f}", (x, y - 5), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            if frame_bgr is not None and frame_bgr.shape[1] == frame_w and frame_bgr.shape[0] == frame_h:
                out_comparison.write(frame_bgr)
        
        out_comparison.release()
        print(f"Saved comparison video to {output_path_comparison}")
        
        print("Computing mAP...")
        result = compute_map(predictions, ground_truth, num_classes=1, iou_threshold=0.5)
        print(f"\nmAP@0.5: {result['mAP']:.4f}")
        print(f"Recall: {result['recall']:.4f}")
        print(f"Precision: {result['precision']:.4f}")
        print(f"F1 Score: {result['f1']:.4f}")
