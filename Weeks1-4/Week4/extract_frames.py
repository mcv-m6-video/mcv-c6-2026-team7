import os
import glob
import cv2

ROOT = "AI_CITY_CHALLENGE_2022_TRAIN/train"
SEQUENCES = ["S01"]
TIMESTAMP_DIR = "AI_CITY_CHALLENGE_2022_TRAIN/cam_timestamp"
SYNC_FPS = 10

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

for seq in SEQUENCES:
    timestamps = load_timestamps(TIMESTAMP_DIR, seq)
    min_offset = min(timestamps.values()) if timestamps else 0

    for vdo_path in sorted(glob.glob(os.path.join(ROOT, seq, "c*", "vdo.avi"))):
        cam_name = os.path.basename(os.path.dirname(vdo_path))  # e.g. "c001"
        out_dir = os.path.join(os.path.dirname(vdo_path), "img1")
        os.makedirs(out_dir, exist_ok=True)

        cap = cv2.VideoCapture(vdo_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        # Match the special-case fps override from run_yolo.py
        if seq == "S03" and cam_name == "c015":
            fps = 8

        offset = timestamps.get(cam_name, 0)

        print(f"Extracting {vdo_path} → {out_dir}  (fps={fps}, offset={offset:.3f}, min_offset={min_offset:.3f})")

        frame_idx = 0
        saved = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            sync_frame_id = int((offset + frame_idx / fps - min_offset) * SYNC_FPS)
            out_path = os.path.join(out_dir, f"{sync_frame_id:06d}.jpg")
            cv2.imwrite(out_path, frame)
            frame_idx += 1
            saved += 1

        cap.release()
        print(f"  ✓ {saved} frames extracted (sync_frame_id range: "
              f"{int((offset - min_offset) * SYNC_FPS)} – "
              f"{int((offset + (frame_idx - 1) / fps - min_offset) * SYNC_FPS)})")