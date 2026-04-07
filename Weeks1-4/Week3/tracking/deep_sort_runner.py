import tqdm
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../external/deep_sort"))

from deep_sort import tracker
from deep_sort import nn_matching
from deep_sort.detection import Detection
from application_util import preprocessing


def execute_deep_SORT(detections_per_frame: pd.DataFrame, max_age=1, min_hits=3, iou_threshold=0.3, show_tracks=False,
                      max_cosine_distance: float=5.0, nn_budget: int=None, nms_max_overlap: float=1.0):
    """
    Executes the Deep SORT (Simple Online and Realtime Tracking with a Deep Association Metric)
    algorithm on a set of per-frame detections, assigning consistent track IDs across frames
    using a nearest-neighbor cosine distance metric combined with IoU-based matching.

    Parameters
    ----------
    detections_per_frame : pd.DataFrame
        DataFrame containing bounding box detections with at least the following columns:
        - 'frame_id'    : int, the frame number the detection belongs to.
        - 'x1'         : float, left coordinate of the bounding box.
        - 'y1'         : float, top coordinate of the bounding box.
        - 'x2'         : float, right coordinate of the bounding box.
        - 'y2'         : float, bottom coordinate of the bounding box.
        - 'confidence' : float, detection confidence score.
    max_age : int, optional
        Maximum number of frames a track is kept alive without receiving a detection. Default is 1.
    min_hits : int, optional
        Minimum number of consecutive detections before a track is confirmed. Default is 3.
    iou_threshold : float, optional
        Minimum IoU overlap required to associate a detection with an existing track. Default is 0.3.
    show_tracks : bool, optional
        If True, prints the track output for each frame to stdout. Default is False.
    max_cosine_distance : float, optional
        Maximum cosine distance threshold for the nearest-neighbor metric. Default is 5.0.
    nn_budget : int or None, optional
        Maximum size of the appearance descriptor gallery per track. Default is None.
    nms_max_overlap : float, optional
        Maximum allowed overlap (IoU) between detections after Non-Maximum Suppression.
        Set to 1.0 to disable NMS. Default is 1.0.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: frame_id, track_id, x1, y1, x2, y2, confidence, x3, y3, z.
        Returns an empty DataFrame with the same columns if no tracks are found.
    """
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracking = tracker.Tracker(metric=metric, max_age=max_age, n_init=min_hits, max_iou_distance=iou_threshold)
    all_tracks = []

    frame_ids = sorted(detections_per_frame['frame_id'].unique())

    for frame_id in tqdm.tqdm(frame_ids, desc="Tracking frames"):
        dets_frame = detections_per_frame[detections_per_frame['frame_id'] == frame_id]

        if not dets_frame.empty:
            dets = []
            for _, row in dets_frame.iterrows():
                # Convert [x1, y1, x2, y2] to tlwh format [x, y, w, h]
                tlwh = np.array([row['x1'], row['y1'], row['x2'] - row['x1'], row['y2'] - row['y1']])
                confidence = row['confidence']
                feature = np.array([])  # no appearance feature
                dets.append(Detection(tlwh, confidence, feature))

            # Apply Non-Maximum Suppression
            boxes = np.array([d.tlwh for d in dets])
            scores = np.array([d.confidence for d in dets])
            kept_indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
            dets = [dets[i] for i in kept_indices]
        else:
            dets = []

        tracking.predict()
        tracking.update(dets)

        frame_tracks = []
        for track in tracking.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            x1, y1, x2, y2 = track.to_tlbr()
            frame_tracks.append([x1, y1, x2, y2, track.track_id])

        if show_tracks:
            print(f"Frame {frame_id}: {frame_tracks}")

        if frame_tracks:
            df_tracks = pd.DataFrame(frame_tracks, columns=['x1', 'y1', 'x2', 'y2', 'track_id'])
            df_tracks['frame_id'] = frame_id
            all_tracks.append(df_tracks)

    if all_tracks:
        result = pd.concat(all_tracks, ignore_index=True)
        result['confidence'] = 1
        result['x3'] = -1
        result['y3'] = -1
        result['z'] = -1
        result['track_id'] = result['track_id'].astype(int)
        return result[['frame_id', 'track_id', 'x1', 'y1', 'x2', 'y2', 'confidence', 'x3', 'y3', 'z']]
    else:
        return pd.DataFrame(columns=['frame_id', 'track_id', 'x1', 'y1', 'x2', 'y2', 'confidence', 'x3', 'y3', 'z'])