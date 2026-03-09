from SORT.sort import *
import tqdm as tqdm
import numpy as np
import pandas as pd

def execute_kalman_SORT(detections_per_frame:pd.DataFrame, max_age=1, min_hits=3, iou_threshold=0.3,  show_tracks=False):

    """
    Executes the Kalman SORT (Simple Online and Realtime Tracking) algorithm on a set
    of per-frame detections, assigning consistent track IDs across frames using a
    Kalman filter for state estimation and IoU for detection-to-track association.

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
        Maximum number of frames a track is kept alive without receiving a detection.
        Default is 1.
    min_hits : int, optional
        Minimum number of consecutive detections before a track is confirmed.
        Default is 3.
    iou_threshold : float, optional
        Minimum IoU overlap required to associate a detection with an existing track.
        Default is 0.3.
    show_tracks : bool, optional
        If True, prints the track output for each frame to stdout. Default is False.

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per tracked detection, containing the following columns:
        - 'frame_id'    : int, the frame number.
        - 'track_id'    : int, the unique ID assigned to the track.
        - 'x1'         : float, left coordinate of the tracked bounding box.
        - 'y1'         : float, top coordinate of the tracked bounding box.
        - 'x2'         : float, right coordinate of the tracked bounding box.
        - 'y2'         : float, bottom coordinate of the tracked bounding box.
        - 'confidence' : float, set to 1 for all tracked outputs.
        - 'x3'         : float, placeholder, set to -1.
        - 'y3'         : float, placeholder, set to -1.
        - 'z'          : float, placeholder, set to -1.
        Returns an empty DataFrame with the same columns if no tracks are found.
    """

    tracker = Sort(max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold)
    all_tracks = []

    frame_ids = sorted(detections_per_frame['frame_id'].unique())

    for frame_id in tqdm.tqdm(frame_ids, desc="Tracking frames"):
        dets_frame = detections_per_frame[detections_per_frame['frame_id'] == frame_id]

        if not dets_frame.empty:
            dets = dets_frame[['x1','y1','x2','y2','confidence']].to_numpy()
        else:
            dets = np.empty((0, 5))

        tracks = tracker.update(dets)

        if show_tracks:
            print(f"Frame {frame_id}: {tracks}")

        if len(tracks) > 0:
            df_tracks = pd.DataFrame(tracks, columns=['x1', 'y1', 'x2', 'y2', 'track_id'])
            df_tracks['frame_id'] = frame_id
            all_tracks.append(df_tracks)

    if all_tracks:
        result = pd.concat(all_tracks, ignore_index=True)

        result['confidence']=1
        result['x3'] = -1
        result['y3'] = -1
        result['z'] = -1

        result['track_id'] = result['track_id'].astype(int)
        col_order = ['frame_id', 'track_id', 'x1', 'y1', 'x2', 'y2', 'confidence', 'x3', 'y3', 'z']

        return result[col_order]
    else:
        return pd.DataFrame(columns=['frame_id', 'track_id', 'x1', 'y1', 'x2', 'y2','confidence', 'x3', 'y3', 'z'])

