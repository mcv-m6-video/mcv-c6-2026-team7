import cv2
import numpy as np
import torch
import pyflow
import png
from torchvision.models.optical_flow import Raft_Large_Weights, Raft_Small_Weights, raft_large, raft_small
from torchvision.transforms.functional import resize
import torchvision.transforms.functional as F


class InputPadder:
    """Pads images such that dimensions are divisible by 8. Used for RAFT"""
    def __init__(self, dims):
        self.ht, self.wd = dims[-2], dims[-1]
        pad_ht = (((self.ht // 8) + 1) * 8 - self.ht) % 8
        pad_wd = (((self.wd // 8) + 1) * 8 - self.wd) % 8
        self._pad = [pad_wd // 2, pad_wd - pad_wd // 2,  # left, right
                     pad_ht // 2, pad_ht - pad_ht // 2]  # top, bottom

    def pad(self, *imgs):
        return [torch.nn.functional.pad(img, self._pad, mode='replicate') for img in imgs]

    def unpad(self, x):
        ht, wd = x.shape[-2], x.shape[-1]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]


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

    elif model == "raft_large" or model == "raft_small":
        # Get the default weights and transforms for RAFT
        if model == "raft_large":
            weights = Raft_Large_Weights.DEFAULT
        else:
            weights = Raft_Small_Weights.DEFAULT
        transforms = weights.transforms()

        # Apply RAFT transforms
        img1, img2 = transforms(img1.unsqueeze(0), img2.unsqueeze(0))

        # Pad to make dimensions divisible by 8
        padder = InputPadder(img1.shape)
        img1, img2 = padder.pad(img1, img2)

        # Select device
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load the RAFT model
        if model == "raft_large":
            model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(device)
        elif model == "raft_small":
            model = raft_small(weights=Raft_Small_Weights.DEFAULT, progress=False).to(device)
        model = model.eval()

        # RAFT outputs lists of predicted flows. Each entry is a (N, 2, H, W) batch of predicted
        # flows that corresponds to a given iteration. We take the last one as our final prediction.
        list_of_flows = model(img1.to(device), img2.to(device))
        predicted_flows = list_of_flows[-1]

        # Unpad the flow back to original resolution
        predicted_flows = padder.unpad(predicted_flows)

        # Convert the predicted flow to a numpy array and rearrange dimensions
        flow = predicted_flows[0].cpu().detach().numpy().transpose(1, 2, 0)  # Shape: (H, W, 2)

        return flow


def calculate_msen_pepn(gt_flow: np.ndarray, pred_flow: np.ndarray, th: int=3) -> float:

    mask = gt_flow[:, :, 2] == 1 
    du = gt_flow[:, :, 0] - pred_flow[:, :, 0]
    dv = gt_flow[:, :, 1] - pred_flow[:, :, 1]

    epe = np.sqrt(du ** 2 + dv ** 2)
    epe_valid = epe[mask]

    msen = float(np.mean(epe_valid))
    pepn = float(np.mean(epe_valid > th))

    return msen, pepn
