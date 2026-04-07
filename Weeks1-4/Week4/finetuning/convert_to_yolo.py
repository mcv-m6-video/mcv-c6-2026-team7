import cv2
import xml.etree.ElementTree as ET
import os
import argparse
import shutil
import random
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="data/AICity_data/train/S03/c010/vdo.avi", help="Path to video")
    parser.add_argument("--annotation", default="data/ai_challenge_s03_c010-full_annotation.xml", help="Path to annotation XML")
    parser.add_argument("--output-base", default="data/yolo_dataset", help="Base output directory")
    parser.add_argument("--folds", type=int, default=4, help="Number of folds (25% chunks)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    all_img_dir = os.path.join(args.output_base, "frame_pool", "images")
    all_lbl_dir = os.path.join(args.output_base, "frame_pool", "labels")
    os.makedirs(all_img_dir, exist_ok=True)
    os.makedirs(all_lbl_dir, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Parse Annotations
    tree = ET.parse(args.annotation)
    root = tree.getroot()
    annotations = {}
    for track in root.findall(".//track[@label='car']"):
        for box in track.findall("box"):
            f_idx = int(box.get("frame"))
            xtl, ytl = float(box.get("xtl")), float(box.get("ytl"))
            xbr, ybr = float(box.get("xbr")), float(box.get("ybr"))
            
            x_center = ((xtl + xbr) / 2) / width
            y_center = ((ytl + ybr) / 2) / height
            w = (xbr - xtl) / width
            h = (ybr - ytl) / height

            if f_idx not in annotations:
                annotations[f_idx] = []
            annotations[f_idx].append(f"0 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

    print(f"Extracting annotated frames from video...")
    valid_indices = []
    for i in tqdm(range(total_frames)):
        ret, frame = cap.read()
        if not ret: break
        
        if i in annotations:
            name = f"frame_{i:05d}"
            cv2.imwrite(os.path.join(all_img_dir, f"{name}.jpg"), frame)
            with open(os.path.join(all_lbl_dir, f"{name}.txt"), "w") as f:
                f.write("\n".join(annotations[i]))
            valid_indices.append(i)
    cap.release()

    n = len(valid_indices)
    chunk_size = n // args.folds
    print(f"Total annotated frames: {n}. Chunk size (25%): {chunk_size}")

    #  Helper to create the folder structure of each fold
    def deploy_fold(fold_name, train_list, val_list):
        path = os.path.join(args.output_base, fold_name)
        for split, frames in [("train", train_list), ("val", val_list)]:
            img_out = os.path.join(path, "images", split)
            lbl_out = os.path.join(path, "labels", split)
            os.makedirs(img_out, exist_ok=True)
            os.makedirs(lbl_out, exist_ok=True)
            
            for f_idx in frames:
                name = f"frame_{f_idx:05d}"
                shutil.copy(os.path.join(all_img_dir, f"{name}.jpg"), os.path.join(img_out, f"{name}.jpg"))
                shutil.copy(os.path.join(all_lbl_dir, f"{name}.txt"), os.path.join(lbl_out, f"{name}.txt"))

        with open(os.path.join(path, "dataset.yaml"), "w") as f:
            f.write(f"path: {os.path.abspath(path)}\ntrain: images/train\nval: images/val\nnc: 1\nnames: ['car']")

    # 6. Sequential Folds (25% Train, 75% Val)
    print("\nCreating Sequential Folds (Moving 25% Training Window)...")
    for k in range(args.folds):
        start = k * chunk_size
        end = start + chunk_size if k < args.folds - 1 else n
        
        train_frames = valid_indices[start:end]
        val_frames = [idx for idx in valid_indices if idx not in train_frames]
        
        deploy_fold(f"sequential_fold_{k+1}", train_frames, val_frames)
        print(f"  Fold {k+1}: Train {len(train_frames)} frames | Val {len(val_frames)} frames")

    # 7. Random Folds (25% Train, 75% Val)
    print("\nCreating Random Folds (Random 25% Train, 75% Val)...")
    random.seed(args.seed)
    shuffled_indices = valid_indices.copy()
    random.shuffle(shuffled_indices)

    for k in range(args.folds):
        start = k * chunk_size
        end = start + chunk_size if k < args.folds - 1 else n
        
        train_frames = shuffled_indices[start:end]
        val_frames = [idx for idx in shuffled_indices if idx not in train_frames]
        
        deploy_fold(f"random_fold_{k+1}", train_frames, val_frames)
        print(f"  Random Fold {k+1}: Train {len(train_frames)} | Val {len(val_frames)}")

if __name__ == "__main__":
    main()