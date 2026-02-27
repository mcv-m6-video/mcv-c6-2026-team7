import cv2
import numpy as np
from ultralytics import YOLO
import argparse
import os

# CONFIGURATION
parser = argparse.ArgumentParser()
parser.add_argument("--source", default="vdo.avi", help="Path to source video file")
parser.add_argument("--output", default="output_masks", help="Base output directory")
args = parser.parse_args()

SOURCE_VIDEO = args.source
BASE_OUTPUT = args.output
BBOX_FOLDER = os.path.join(BASE_OUTPUT, "bboxes")
VIDEO_OUTPUT = os.path.join(BASE_OUTPUT, "output_video.mp4")
DETECTIONS_FILE = os.path.join(BASE_OUTPUT, "detections.txt")
TARGET_CLASSES = [2]  # car only (COCO class 2)

os.makedirs(BBOX_FOLDER, exist_ok=True)

# Load YOLO model
model = YOLO("yolov3.pt")

# Get source video properties
cap = cv2.VideoCapture(SOURCE_VIDEO)
fps = int(cap.get(cv2.CAP_PROP_FPS))
vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(VIDEO_OUTPUT, fourcc, fps, (vid_w, vid_h))

print("=== Running Detection ===")
results = model.predict(source=SOURCE_VIDEO, classes=TARGET_CLASSES, stream=True, conf=0.3)

# Process results looping through each frame
detections_lines = []
for frame_idx, r in enumerate(results):
    
    orig_frame = r.orig_img.copy()
    h_orig, w_orig = r.orig_shape

    bbox_mask_frame = np.zeros((h_orig, w_orig), dtype=np.uint8)

    # Check if there are detections in the current frame
    if r.boxes is not None and len(r.boxes) > 0:
        bboxes_xyxy = r.boxes.xyxy.cpu().numpy()
        class_ids = r.boxes.cls.int().cpu().tolist()
        confidences = r.boxes.conf.cpu().tolist()

        # Loop through detections
        for i in range(len(bboxes_xyxy)):
            x1, y1, x2, y2 = bboxes_xyxy[i].astype(int)
            class_id = class_ids[i]
            conf = confidences[i]

            # Save detection information
            detections_lines.append(f"{frame_idx},{class_id},{conf:.4f},{x1},{y1},{x2},{y2}")

            # Draw bounding box on mask and original frame
            cv2.rectangle(bbox_mask_frame, (x1, y1), (x2, y2), 255, -1)
            cv2.rectangle(orig_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(orig_frame, f"car {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save bbox mask and write frame to video
    filename = f"frame_{frame_idx:05d}.png"
    cv2.imwrite(os.path.join(BBOX_FOLDER, filename), bbox_mask_frame)
    video_writer.write(orig_frame)

video_writer.release()

# Save detections to a text file
with open(DETECTIONS_FILE, 'w') as f:
    f.write("frame_id,class_id,confidence,x1,y1,x2,y2\n")
    f.write("\n".join(detections_lines))

print(f"Done! Results saved to {BASE_OUTPUT}")