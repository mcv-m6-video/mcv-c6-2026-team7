import cv2
import numpy as np
import torch
import pyflow
import png

from torchvision.models.optical_flow import Raft_Large_Weights, Raft_Small_Weights, raft_large, raft_small
from torchvision.transforms.functional import resize
import torchvision.transforms.functional as F

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../external/flowformerpp"))
from configs.kitti import get_cfg
from core.FlowFormer import build_flowformer
from core.utils.utils import InputPadder as FlowFormerInputPadder


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
    
    elif model == "flowformer_pp":
        cfg = get_cfg()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_nn = build_flowformer(cfg).to(device)
        
        # Load checkpoint manually
        checkpoint = torch.load(params[0], map_location=device)

        # Strip 'module.' prefix from DataParallel checkpoint
        checkpoint = {k.replace("module.", "", 1): v for k, v in checkpoint.items()}

        model_nn.load_state_dict(checkpoint, strict=True)
        
        model_nn.eval()

        img1_t = img1.unsqueeze(0).float().to(device)
        img2_t = img2.unsqueeze(0).float().to(device)

        padder = FlowFormerInputPadder(img1_t.shape)
        img1_t, img2_t = padder.pad(img1_t, img2_t)

        with torch.no_grad():
            flow_predictions = model_nn(img1_t, img2_t)

        flow = padder.unpad(flow_predictions[0])  # (1, 2, H, W)
        return flow[0].cpu().numpy().transpose(1, 2, 0)  # (H, W, 2)


def calculate_msen_pepn(gt_flow: np.ndarray, pred_flow: np.ndarray, th: int=3) -> float:

    mask = gt_flow[:, :, 2] == 1 
    du = gt_flow[:, :, 0] - pred_flow[:, :, 0]
    dv = gt_flow[:, :, 1] - pred_flow[:, :, 1]

    epe = np.sqrt(du ** 2 + dv ** 2)
    epe_valid = epe[mask]

    msen = float(np.mean(epe_valid))
    pepn = float(np.mean(epe_valid > th))

    return msen, pepn

### Optical Flow plot functions ###
import matplotlib.pyplot as plt


def flow_to_magnitude(flow: np.ndarray) -> np.ndarray:
    """Returns a 2D array with the magnitude of the flow at each pixel."""
    u, v = flow[..., 0], flow[..., 1]
    return np.sqrt(u**2 + v**2)


def flow_to_color(flow: np.ndarray) -> np.ndarray:
    """Encodes flow as an HSV color image (hue=direction, value=magnitude)."""
    u, v = flow[..., 0], flow[..., 1]
    magnitude = np.sqrt(u**2 + v**2)
    angle = np.arctan2(v, u)  # Range: [-pi, pi]

    hsv = np.zeros((*flow.shape[:2], 3), dtype=np.uint8)
    hsv[..., 0] = ((angle + np.pi) / (2 * np.pi) * 179).astype(np.uint8)        # Hue
    hsv[..., 1] = 255                                                           # Saturation
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)       # Value

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def _load_as_rgb(img1) -> np.ndarray:
    """Converts img1 to a (H, W, 3) uint8 RGB array regardless of input format."""
    img = np.array(img1)
    # Torch tensor (C, H, W)
    if img.ndim == 3 and img.shape[0] in (1, 3):
        img = img.transpose(1, 2, 0)
    # Single channel -> repeat to 3
    if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
        img = np.squeeze(img)
        img = np.stack([img, img, img], axis=-1)
    # Float [0, 1] -> uint8
    if img.dtype != np.uint8:
        img = (img * 255).clip(0, 255).astype(np.uint8)
    return img


def plot_flow(flow: np.ndarray, mode: str, save_path: str, img1=None, alpha: float = 0.5, step: int = 16):
    """
    Saves an optical flow visualization as a plain PNG image (no axes, no title, no borders).
    If img1 is provided, it is blended as a background behind the flow visualization.

    Args:
        flow      : (H, W, 2) array with u and v components.
        mode      : One of 'magnitude', 'color', 'arrows', 'quiver'.
        save_path : Path where the PNG will be saved.
        img1      : Original image to overlay. Accepts numpy arrays or torch tensors.
        alpha     : Opacity of the flow layer over the original image (0=only image, 1=only flow).
        step      : Subsampling step for arrows/quiver plots.
    """
    u, v = flow[..., 0], flow[..., 1]
    H, W = flow.shape[:2]

    bg = _load_as_rgb(img1) if img1 is not None else None

    def blend(flow_rgb: np.ndarray) -> np.ndarray:
        if bg is None:
            return flow_rgb
        bg_resized = cv2.resize(bg, (W, H))
        return cv2.addWeighted(bg_resized, 1 - alpha, flow_rgb, alpha, 0)

    if mode == "magnitude":
        mag = flow_to_magnitude(flow)
        mag_u8 = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        flow_rgb = cv2.cvtColor(cv2.applyColorMap(mag_u8, cv2.COLORMAP_PLASMA), cv2.COLOR_BGR2RGB)
        result = blend(flow_rgb)
        cv2.imwrite(save_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))

    elif mode == "color":
        flow_rgb = flow_to_color(flow)  # already RGB
        result = blend(flow_rgb)
        cv2.imwrite(save_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))

    elif mode == "arrows" or mode == "quiver":
        # Use exact pixel size to avoid any rounding gaps
        fig = plt.figure(frameon=False)
        fig.set_size_inches(W / fig.dpi, H / fig.dpi)
        ax = plt.Axes(fig, [0, 0, 1, 1])  # axes fills the entire figure
        ax.set_axis_off()
        fig.add_axes(ax)

        ys = np.arange(0, H, step)
        xs = np.arange(0, W, step)
        X, Y = np.meshgrid(xs, ys)
        U = u[::step, ::step]
        V = v[::step, ::step]

        # Background: original image, magnitude, or black
        if bg is not None:
            ax.imshow(cv2.resize(bg, (W, H)), aspect="auto")
        elif mode == "arrows":
            ax.imshow(flow_to_magnitude(flow), cmap="gray", aspect="auto")
        else:
            ax.set_facecolor("black")

        if mode == "arrows":
            magnitude = np.sqrt(U**2 + V**2)
            ax.quiver(X, Y, U, V, magnitude, cmap="autumn_r", angles="xy",
                      scale_units="xy", scale=1, width=0.002)
        else:  # quiver
            magnitude = np.sqrt(U**2 + V**2)
            ax.quiver(X, Y, U, V, magnitude, cmap="viridis", angles="xy",
                      scale_units="xy", scale=1)
            ax.invert_yaxis()

        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        plt.savefig(save_path, dpi=fig.dpi, pad_inches=0)
        plt.close()

    else:
        raise ValueError(f"Unknown plot mode '{mode}'. Choose from: magnitude, color, arrows, quiver.")
