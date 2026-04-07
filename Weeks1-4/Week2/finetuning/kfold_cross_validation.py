import os
import re
import glob
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict
from metrics import compute_map

BASE_DIR = Path("/home/master/C6/yolo/fine-tune")
DATA_DIR = BASE_DIR / "data/yolo_dataset"
OUTPUT_DIR = BASE_DIR / "output_masks" 

IMG_WIDTH, IMG_HEIGHT = 1920, 1080 
FOLD_TYPES = ["random", "sequential"]
NUM_FOLDS = 4
IOU_THRESHOLD = 0.5

def load_ground_truth(labels_dir: Path) -> Dict:
    gt = defaultdict(list)
    if not labels_dir.exists(): return gt
    
    for label_file in glob.glob(str(labels_dir / "*.txt")):
        match = re.search(r'frame_(\d+)', Path(label_file).stem)
        if not match: continue
        frame_idx = int(match.group(1))
        
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5: continue
                # YOLO format: class, xc, yc, w, h (normalized)
                _, xc, yc, w, h = map(float, parts[:5])
                
                px_w, px_h = w * IMG_WIDTH, h * IMG_HEIGHT
                px_x = (xc * IMG_WIDTH) - (px_w / 2)
                px_y = (yc * IMG_HEIGHT) - (px_h / 2)
                
                gt[frame_idx].append(((px_x, px_y, px_w, px_h), 0))
    return dict(gt)

def load_predictions_from_file(detections_file: Path) -> Dict:
    """Loads CSV detections and ensures Top-Left (x, y, w, h) format."""
    predictions = defaultdict(list)
    if not detections_file.exists(): return dict(predictions)
    
    try:
        df = pd.read_csv(detections_file)
        if 'frame_id' not in df.columns:
            df = pd.read_csv(detections_file, names=['frame_id', 'class_id', 'confidence', 'x1', 'y1', 'x2', 'y2'])
    except:
        return dict(predictions)

    for _, row in df.iterrows():
        f_idx = int(row['frame_id'])
        x1, y1, x2, y2 = row['x1'], row['y1'], row['x2'], row['y2']
        
        w, h = x2 - x1, y2 - y1
        predictions[f_idx].append(((float(x1), float(y1), float(w), float(h)), float(row['confidence']), 0))
        
    return dict(predictions)

def process_fold(fold_type: str, fold_num: int) -> Dict[str, Any]:
    fold_name = f"{fold_type}_fold_{fold_num}"
    
    det_path = Path(f"{OUTPUT_DIR}_{fold_name}") / "detections.txt"
    val_labels_dir = DATA_DIR / fold_name / "labels" / "val"
    
    if not det_path.exists():
        print(f"      [!] Skip: Detections not found for {fold_name}")
        return None

    raw_preds = load_predictions_from_file(det_path)
    raw_gts = load_ground_truth(val_labels_dir)

    # This ensures we only evaluate the specific validation set of this fold.
    common_frames = set(raw_preds.keys()).intersection(set(raw_gts.keys()))
    
    preds = {f: raw_preds[f] for f in common_frames}
    gts = {f: raw_gts[f] for f in common_frames}

    if not common_frames:
        print(f"      [!] Warning: No matching frames found between Preds and GT for {fold_name}")
        return None

    print(f"      Evaluating {len(common_frames)} synchronized frames...")
    
    metrics = compute_map(preds, gts, 1, iou_threshold=IOU_THRESHOLD, N=1)
    return {'fold': fold_name, 'metrics': metrics}

def main():
    print("\n" + "="*50)
    print("K-Fold Evaluation")
    print("="*50)
    
    all_summary = []
    for f_type in FOLD_TYPES:
        type_results = []
        for f_num in range(1, NUM_FOLDS + 1):
            res = process_fold(f_type, f_num)
            if res:
                m = res['metrics']
                type_results.append({
                    'type': f_type, 'fold': f_num, 
                    'mAP': m['mAP'], 'Prec': m['precision'], 'Rec': m['recall'], 'F1': m['f1']
                })
        
        if type_results:
            df = pd.DataFrame(type_results)
            print(f"\nResults for {f_type.upper()}:")
            print(df.to_string(index=False))
            print(f"Mean mAP: {df['mAP'].mean():.4f} ± {df['mAP'].std():.4f}")
            print(f"Mean Precision: {df['Prec'].mean():.4f} ± {df['Prec'].std():.4f}")
            print(f"Mean Recall: {df['Rec'].mean():.4f} ± {df['Rec'].std():.4f}")
            print(f"Mean F1: {df['F1'].mean():.4f} ± {df['F1'].std():.4f}")
            all_summary.extend(type_results)

    if all_summary:
        out_path = BASE_DIR / "final_cv_metrics.csv"
        pd.DataFrame(all_summary).to_csv(out_path, index=False)
        print(f"\nFinal results saved to: {out_path}")

if __name__ == "__main__":
    main()