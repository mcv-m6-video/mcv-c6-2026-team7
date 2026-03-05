import numpy as np
import pyflow
import png


def read_kitti_flow(flow_file: str) -> np.ndarray:

    reader = png.Reader(filename=flow_file)
    w, h, rows, info = reader.read()

    row_arr = np.asarray(list(rows), dtype=np.float64)  # Shape: (h, w*3)
    flow = row_arr.reshape(h, w, 3)                     # Shape: (h, w, 3)

    valid = flow[..., 2] > 0
    flow[..., :2] = (flow[..., :2] - 32768.0) / 64.0    # Decode u,v
    flow[~valid, :2] = 0.0                              # Zero invalid u,v

    return flow


def compute_optical_flow(model: str, params: list, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:

    if model == "pyflow":
        alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations, nSORIterations, colType = params

        u, v, _ = pyflow.coarse2fine_flow(
            img1, img2, alpha, ratio, minWidth, nOuterFPIterations, 
            nInnerFPIterations, nSORIterations, colType
        )

    return np.dstack((u, v))


def calculate_msen_pepn(gt_flow: np.ndarray, pred_flow: np.ndarray, th: int=3) -> float:

    mask = gt_flow[:, :, 2] == 1 
    du = gt_flow[:, :, 0] - pred_flow[:, :, 0]
    dv = gt_flow[:, :, 1] - pred_flow[:, :, 1]

    epe = np.sqrt(du ** 2 + dv ** 2)
    epe_valid = epe[mask]

    msen = float(np.mean(epe_valid))
    pepn = float(np.mean(epe_valid > th))

    return msen, pepn
