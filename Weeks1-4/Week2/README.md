# MCV C6 - Video Analysis Project - Week 2

## Introduction

In Week 2 of the Video Analysis Project we focus on **object detection** and **object tracking**. Files specific to this Week can be found inside the `Week2/` folder.

## Environment Setup

1. **Create virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Data Setup

The project expects the following directory structure:
```
Root/
├── data/
│   ├── AICity_data
│   └── ai_challenge_s03_c010-full_annotation.xml
├── parsed_data/                              # Frames will be extracted here
└── WeekN/
```

**To extract frames from video** (run once):
```bash
cd ..
python data_processor.py
```

This will read the files in `data/` and generate the `parsed_data/` folder with the ready-to-read frames in `.jpg` format.

## Running the off-the-shelf model

We have chosen YOLOv3 (weights `yolov3u.pt`) as our off-the-shelf model. To run this model, use `run_yolo.py`, specifying the path to the video sequence using `--source`. This script will generate an `output_video.mp4` showing the detected bounding boxes and, most importantly, a `detections.txt` file with all the detections.

To evaluate the results, use `eval_yolo.py`. This script will use the `detections.txt` file and calculate the mAP@0.5, Precision, Recall, and F1, and will output a video comparing predicted and ground truth bounding boxes. This script also contains several global variables that can be modified to adjust ground truth generation.

## Running Tracking on Object Detection

Use `Week2/tracking/main.py` to track objects across frames using your detections:

```bash
python Week2/tracking/main.py \
    --method overlap \
    --detections Week2/detections/detections.txt
```

### Main Parameters

- `--method` (overlap | kalman): Tracking algorithm to use. Default: `overlap`
  - `overlap`: Maximum IoU-based tracking (faster, simpler)
  - `kalman`: Kalman filter with SORT algorithm (more sophisticated)

- `--detections`: Path to detections file in txt format. Default: `Week2/detections/detections.txt`. Can also be the fine-tuned detections.

### Matching Parameters

- `--iou_thr`: IoU threshold for detecting matches between detections and tracks. Default: `0.40`


### Memory Parameters (Re-identification)

- `--memory_frames`: Number of frames to keep lost tracks in memory for re-identification. Default: `5` (set to `0` to disable)

- `--memory_iou_thr`: Minimum IoU threshold to re-identify a detection with a remembered lost track. Default: `0.90`

### Kalman/SORT Parameters

These parameters are used when `--method kalman` is selected:

- `max_age`: Maximum frames a track can exist without detections before being deleted. Default: `1` (modify in code if needed)

- `min_hits`: Minimum number of consecutive hits before a track is confirmed. Default: `3` (modify in code if needed)

### Output

The script generates in `Week2/tracking/outputs/<method>_<timestamp>/` a .txt file; `tracks.txt`: Tracking results in MOTChallenge format

## Tracking Evaluation

### evaluate tracking results with TrackEval

Use `eval_tracking.py` to compute HOTA and IDF1 metrics for your tracking results:

```bash
python Week2/tracking/eval_tracking.py \
    --tracker-results Week2/tracking/outputs/overlap_20260228_191236 \
    --tracker-name overlap_iou40
```

This script:
1. Converts ground truth and tracking results to MOTChallenge format
2. Runs TrackEval to compute HOTA, IDF1, and other metrics

**Results will be saved in:**
```
data/trackers_mot_format/AICity-train/<tracker_name>/pedestrian_summary.txt
```

### Convert Ground Truth to MOTChallenge Format

The `prepare_gt_for_trackeval.py` module provides the `MOTChallengeConverter` class for format conversion:

```python
from Week2.tracking.prepare_gt_for_trackeval import MOTChallengeConverter

# Convert tracking results DataFrame
tracked_mot = MOTChallengeConverter.dataframe_to_motchallenge(
    df, is_ground_truth=False
)

# Convert ground truth from XML
MOTChallengeConverter.ground_truth_to_motchallenge(
    annotation_path="data/ai_challenge_s03_c010-full_annotation.xml",
    output_file="data/gt_mot_format/AICity-train/S03c010/gt/gt.txt",
    class_filter="car"
)
```

**Output format** (10 values per line, comma-separated):
```
<frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
``` 

