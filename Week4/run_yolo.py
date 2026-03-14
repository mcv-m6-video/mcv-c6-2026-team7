'''
    Individual file. Logic will be moved to the main.py
'''


import cv2
import numpy as np
from ultralytics import YOLO
import argparse
import os
import glob
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument("--source", default="../data/AICity_data/train", help="Path to source folder")
parser.add_argument("--output", default="output_detections_test", help="Base output directory")
parser.add_argument("--timestamp_dir", default="../data/AICity_data/cam_timestamp", help="Path to timestamp folder")
args = parser.parse_args()

SOURCE_DIR = args.source
BASE_OUTPUT = args.output
TIMESTAMP_DIR = args.timestamp_dir
TARGET_CLASSES = [2, 5, 7]

os.makedirs(BASE_OUTPUT, exist_ok=True)

model = YOLO("yolov3.pt")

def load_timestamps(timestamp_dir, scene):
    timestamp_file = os.path.join(timestamp_dir, f"{scene}.txt")
    timestamps = {}
    if os.path.exists(timestamp_file):
        with open(timestamp_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    timestamps[parts[0]] = float(parts[1])
    return timestamps

scene_dirs = sorted([d for d in os.listdir(SOURCE_DIR) if os.path.isdir(os.path.join(SOURCE_DIR, d)) and d.startswith('S')])

print(f"Found scenes: {scene_dirs}")

total_cameras = 0
for scene in scene_dirs:
    scene_path = os.path.join(SOURCE_DIR, scene)
    camera_dirs = sorted([d for d in os.listdir(scene_path) if os.path.isdir(os.path.join(scene_path, d)) and d.startswith('c')])
    total_cameras += len(camera_dirs)

print(f"Total cameras to process: {total_cameras}")

processed = 0

for scene in scene_dirs:
    scene_path = os.path.join(SOURCE_DIR, scene)
    camera_dirs = sorted([d for d in os.listdir(scene_path) if os.path.isdir(os.path.join(scene_path, d)) and d.startswith('c')])
    
    timestamps = load_timestamps(TIMESTAMP_DIR, scene)
    min_offset = min(timestamps.values()) if timestamps else 0
    sync_fps = 10
    
    for camera in tqdm(camera_dirs, desc=f"Processing {scene}"):
        camera_path = os.path.join(scene_path, camera)
        video_path = os.path.join(camera_path, "vdo.avi")
        
        if not os.path.exists(video_path):
            print(f"Warning: Video not found for {scene}/{camera}")
            continue
        
        output_camera_dir = os.path.join(BASE_OUTPUT, scene, camera)
        os.makedirs(output_camera_dir, exist_ok=True)
        detections_file = os.path.join(output_camera_dir, "detections.txt")
        
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        if scene == "S03" and camera == "c015":
            fps = 8
        
        vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        offset = timestamps.get(camera, 0)
        
        detections_lines = []
        
        results = model.predict(source=video_path, classes=TARGET_CLASSES, stream=True, conf=0.3, verbose=False)
        
        for frame_idx, r in tqdm(enumerate(results), desc=f"Processing camera {camera}"):
            timestamp = offset + (frame_idx / fps)
            sync_frame_id = int((timestamp - min_offset) * sync_fps)
            
            if r.boxes is not None and len(r.boxes) > 0:
                bboxes_xyxy = r.boxes.xyxy.cpu().numpy()
                class_ids = r.boxes.cls.int().cpu().tolist()
                confidences = r.boxes.conf.cpu().tolist()
                
                for i in range(len(bboxes_xyxy)):
                    x1, y1, x2, y2 = bboxes_xyxy[i].astype(int)
                    class_id = class_ids[i]
                    conf = confidences[i]
                    detections_lines.append(f"{sync_frame_id},{timestamp:.3f},{class_id},{conf:.4f},{x1},{y1},{x2},{y2}")
        
        with open(detections_file, 'w') as f:
            f.write("frame_id,timestamp,class_id,confidence,x1,y1,x2,y2\n")
            f.write("\n".join(detections_lines))
        
        processed += 1
        print(f"Saved detections for {scene}/{camera}")

print(f"Done! Processed {processed} cameras. Detections saved to each camera folder.")