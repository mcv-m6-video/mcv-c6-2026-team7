'''
    This was executed in a HPC. Some libraries might be not in the requirements.txt
'''

import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import os
import torch

# --- CONFIGURATION ---
SOURCE_VIDEO = "vdo.avi"
BASE_OUTPUT = "output_masks"
SEG_FOLDER = os.path.join(BASE_OUTPUT, "segmentation")
BBOX_FOLDER = os.path.join(BASE_OUTPUT, "bboxes")
VIDEO_OUTPUT = os.path.join(BASE_OUTPUT, "output_video.mp4")
DETECTIONS_FILE = os.path.join(BASE_OUTPUT, "detections.txt")

# Threshold: Total pixels moved across the whole video. 
GLOBAL_MOVE_THRESHOLD = 30.0 
TARGET_CLASSES = [1, 2, 3, 5, 7] # bicycle, car, motorcycle, bus, truck

os.makedirs(SEG_FOLDER, exist_ok=True)
os.makedirs(BBOX_FOLDER, exist_ok=True)

# 1. Load Model
model = YOLO("yolo11m-seg.pt")

trajectories = {}
moving_ids = set()

# Store all frame data for single-pass processing
all_frames_data = []

cap = cv2.VideoCapture(SOURCE_VIDEO)
fps = int(cap.get(cv2.CAP_PROP_FPS))
vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(VIDEO_OUTPUT, fourcc, fps, (vid_w, vid_h))
detections_lines = []

print("=== Single Pass: Analyzing Motion & Generating Masks ===")
results = model.track(source=SOURCE_VIDEO, persist=True, classes=TARGET_CLASSES, stream=True, conf=0.3)

for r in results:
    frame_data = {
        'orig_img': r.orig_img,
        'orig_shape': r.orig_shape,
        'boxes': r.boxes,
        'masks': r.masks
    }
    all_frames_data.append(frame_data)
    
    if r.boxes is not None and r.boxes.id is not None:
        boxes = r.boxes.xywh.cpu().numpy()
        track_ids = r.boxes.id.int().cpu().tolist()
        
        for box, tid in zip(boxes, track_ids):
            x, y, w, h = box
            if tid not in trajectories:
                trajectories[tid] = [x, y, 0.0]
            else:
                start_x, start_y, current_max = trajectories[tid]
                dist = np.sqrt((x - start_x)**2 + (y - start_y)**2)
                if dist > current_max:
                    trajectories[tid][2] = dist

# Decide which vehicle actually moved
for tid, (sx, sy, max_d) in trajectories.items():
    if max_d >= GLOBAL_MOVE_THRESHOLD:
        moving_ids.add(tid)

print(f"Analysis Complete. Found {len(trajectories)} objects, {len(moving_ids)} are moving.")

print("=== Generating Masks (Segmentation & BBoxes) ===")
for frame_idx, frame_data in enumerate(all_frames_data):
    orig_frame = frame_data['orig_img'].copy()
    h_orig, w_orig = frame_data['orig_shape']
    r_boxes = frame_data['boxes']
    r_masks = frame_data['masks']
    
    seg_mask_frame = np.zeros((h_orig, w_orig), dtype=np.uint8)
    bbox_mask_frame = np.zeros((h_orig, w_orig), dtype=np.uint8)

    if r_boxes is not None and r_boxes.id is not None:
        track_ids = r_boxes.id.int().cpu().tolist()
        bboxes_xyxy = r_boxes.xyxy.cpu().numpy() 
        class_ids = r_boxes.cls.int().cpu().tolist()
        
        has_masks = r_masks is not None
        if has_masks:
            masks = r_masks.data.cpu().numpy()

        for i, tid in enumerate(track_ids):
            if tid in moving_ids:
                x1, y1, x2, y2 = bboxes_xyxy[i].astype(int)
                class_id = class_ids[i]
                detections_lines.append(f"{frame_idx},{tid},{class_id},{x1},{y1},{x2},{y2}")
                
                cv2.rectangle(bbox_mask_frame, (x1, y1), (x2, y2), 255, -1)
                cv2.rectangle(orig_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(orig_frame, f"ID:{tid}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                if has_masks and i < len(masks):
                    m = cv2.resize(masks[i], (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
                    seg_mask_frame[m > 0] = 255

    filename = f"frame_{frame_idx:05d}.png"
    cv2.imwrite(os.path.join(SEG_FOLDER, filename), seg_mask_frame)
    cv2.imwrite(os.path.join(BBOX_FOLDER, filename), bbox_mask_frame)
    video_writer.write(orig_frame)

video_writer.release()

with open(DETECTIONS_FILE, 'w') as f:
    f.write("frame_id,track_id,class_id,x1,y1,x2,y2\n")
    f.write("\n".join(detections_lines))

print(f"Done! Results saved to {BASE_OUTPUT}")