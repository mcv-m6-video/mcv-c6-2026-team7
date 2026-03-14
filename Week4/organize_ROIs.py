import os
import shutil
import glob

src_root = "AI_CITY_CHALLENGE_2022_TRAIN/train"
dst_root = "ROIs/train"

for roi_path in glob.glob(os.path.join(src_root, "*", "*", "roi.jpg")):
    cam = os.path.basename(os.path.dirname(roi_path))  # e.g. c001
    dst = os.path.join(dst_root, cam)
    os.makedirs(dst, exist_ok=True)
    shutil.copy(roi_path, os.path.join(dst, "roi.jpg"))
    print(f"Copied: {roi_path} → {dst}/roi.jpg")

print("Done.")