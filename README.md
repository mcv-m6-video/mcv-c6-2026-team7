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

