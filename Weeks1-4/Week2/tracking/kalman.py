from SORT.sort import *
import tqdm as tqdm
import numpy as np
import pandas as pd

def execute_kalman_SORT(detections_per_frame:pd.DataFrame, max_age=1, min_hits=3, iou_threshold=0.3,  show_tracks=False):

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

