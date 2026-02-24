import sys
from pathlib import Path
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).parents[1]))

from data_processor import AICityFrames
from metrics import compute_map

DETECTIONS_FILE = Path(__file__).parent / "detections.txt"
OUTPUT_VIDEO = Path(__file__).parent / "output_yolo.mp4"


def main():
    scale = 0.25
    
    dataloader = AICityFrames(scale=scale)
    total_frames = dataloader.frame_count
    warmup_end = int(total_frames * 0.25)
    
    print(f"Total frames: {total_frames}")
    print(f"Warmup end (25%): {warmup_end}")
    print(f"Eval frames: {warmup_end} to {total_frames - 1}")
    
    label_to_class = {'car': 0, 'bike': 0}
    
    print("Building ground truth from dataloader...")
    ground_truth = {}
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
    
    print("Loading predictions from detections.txt...")
    predictions = {}
    
    with open(DETECTIONS_FILE, 'r') as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split(',')
            if len(parts) != 7:
                continue
            frame_id = int(parts[0])
            track_id = int(parts[1])
            class_id = int(parts[2])
            x1 = int(parts[3])
            y1 = int(parts[4])
            x2 = int(parts[5])
            y2 = int(parts[6])
            
            x = int(x1 * scale)
            y = int(y1 * scale)
            w = int((x2 - x1) * scale)
            h = int((y2 - y1) * scale)
            
            if frame_id not in predictions:
                predictions[frame_id] = []
            predictions[frame_id].append(((x, y, w, h), 1.0, 0))
    
    predictions = {k: v for k, v in predictions.items() if k >= warmup_end}
    
    print(f"Predictions for {len(predictions)} frames")
    
    print("Writing output video...")
    frame_h, frame_w = dataloader.image(0).shape[:2]
    if len(dataloader.image(0).shape) == 2:
        frame_w = frame_w
        frame_h = frame_h
    out = cv2.VideoWriter(str(OUTPUT_VIDEO), cv2.VideoWriter_fourcc(*'mp4v'), 10, (frame_w, frame_h))
    
    for frame_idx in range(warmup_end, total_frames - 1):
        frame = dataloader.image(frame_idx)
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        
        if frame_idx in ground_truth:
            for (bbox, cls) in ground_truth[frame_idx]:
                x, y, w, h = bbox
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        if frame_idx in predictions:
            for (x, y, w, h), conf, cls in predictions[frame_idx]:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        out.write(frame)
    
    out.release()
    print(f"Video saved to {OUTPUT_VIDEO}")
    
    print("Computing mAP...")
    np.random.seed(0)
    
    result = compute_map(
        predictions,
        ground_truth,
        num_classes=1,
        iou_threshold=0.5,
        replace_confidence_at_random=True,
        N=10,
    )
    
    print(f"\nmAP@0.5: {result['mAP']:.4f}")
    print(f"std_mAP: {result.get('std_mAP', 'N/A')}")
    print(f"Recall: {result['recall']:.4f}")
    print(f"Precision: {result['precision']:.4f}")
    print(f"F1 Score: {result['f1']:.4f}")


if __name__ == '__main__':
    main()
