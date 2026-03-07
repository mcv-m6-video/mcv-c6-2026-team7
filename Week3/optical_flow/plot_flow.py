import cv2
import numpy as np
import matplotlib.pyplot as plt



def flow_to_magnitude(flow: np.ndarray) -> np.ndarray:
    """Returns a 2D array with the per-pixel magnitude of the flow."""
    u, v = flow[..., 0], flow[..., 1]
    return np.sqrt(u**2 + v**2)


def flow_to_color(flow: np.ndarray) -> np.ndarray:
    """Encodes flow as an HSV color image (hue=direction, value=magnitude). Returns RGB uint8."""
    u, v = flow[..., 0], flow[..., 1]
    magnitude = np.sqrt(u**2 + v**2)
    angle = np.arctan2(v, u)  # Range: [-pi, pi]

    hsv = np.zeros((*flow.shape[:2], 3), dtype=np.uint8)
    hsv[..., 0] = ((angle + np.pi) / (2 * np.pi) * 179).astype(np.uint8)  # Hue
    hsv[..., 1] = 255                                                        # Saturation
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)   # Value

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def flow_to_error(pred_flow: np.ndarray, gt_flow: np.ndarray) -> np.ndarray:
    """
    Computes per-pixel EPE (endpoint error) between prediction and ground truth.
    gt_flow must be (H, W, 3) with the valid mask in channel 2, or (H, W, 2) without it.
    Invalid pixels are set to 0 in the output.
    Returns a 2D float array.
    """
    if gt_flow.shape[2] == 3:
        valid = gt_flow[..., 2] == 1
        gt_uv = gt_flow[..., :2]
    else:
        valid = np.ones(gt_flow.shape[:2], dtype=bool)
        gt_uv = gt_flow

    du = pred_flow[..., 0] - gt_uv[..., 0]
    dv = pred_flow[..., 1] - gt_uv[..., 1]
    epe = np.sqrt(du**2 + dv**2)
    epe[~valid] = 0.0
    return epe


def _load_as_rgb(img1) -> np.ndarray:
    """Converts img1 to a (H, W, 3) uint8 RGB array regardless of input format."""
    img = np.array(img1)
    if img.ndim == 3 and img.shape[0] in (1, 3):   # Torch tensor (C, H, W)
        img = img.transpose(1, 2, 0)
    if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):  # Single channel
        img = np.squeeze(img)
        img = np.stack([img, img, img], axis=-1)
    if img.dtype != np.uint8:                        # Float [0,1] -> uint8
        img = (img * 255).clip(0, 255).astype(np.uint8)
    return img


def _make_fig(W: int, H: int):
    """Creates a borderless matplotlib figure of exactly W x H pixels."""
    fig = plt.figure(frameon=False)
    fig.set_size_inches(W / fig.dpi, H / fig.dpi)
    ax = plt.Axes(fig, [0, 0, 1, 1])
    ax.set_axis_off()
    fig.add_axes(ax)
    return fig, ax

COLORSCALE_MODES = ["colorbar", "text", "separate"]

def save_colorscale(image_bgr: np.ndarray, scalar_map: np.ndarray, cmap_cv2: int,
                    base_path: str, colorscale: list, label: str = ""):
    """
    Generates one or more color scale visualizations alongside the main image.

    Args:
        image_bgr   : The already-rendered BGR image.
        scalar_map  : Raw float 2D array (used for min/max).
        cmap_cv2    : OpenCV colormap constant (e.g. cv2.COLORMAP_HOT).
        base_path   : Path of the main image (e.g. results/error.png).
        colorscale  : List of modes: 'colorbar', 'text', 'separate'.
        label       : Label string shown on the scale (e.g. 'EPE (px)').
    """
    stem = base_path.replace(".png", "")

    if "colorbar" in colorscale:
        _colorscale_colorbar(image_bgr, scalar_map, cmap_cv2,
                             f"{stem}_colorbar.png", label)
    if "text" in colorscale:
        _colorscale_text(image_bgr, scalar_map,
                         f"{stem}_text.png", label)
    if "separate" in colorscale:
        _colorscale_separate(scalar_map, cmap_cv2,
                             f"{stem}_scale.png", label)



PLOT_MODES = ["magnitude", "color", "arrows", "quiver", "error"]


def plot_flow(
    flow: np.ndarray,
    mode: str,
    save_path: str,
    img1=None,
    alpha: float = 0.5,
    step: int = 16,
    gt_flow: np.ndarray = None,
):
    """
    Saves an optical flow visualization as a plain PNG (no axes, no title, no borders).

    Args:
        flow      : (H, W, 2) predicted flow array.
        mode      : One of 'magnitude', 'color', 'arrows', 'quiver', 'error'.
        save_path : Output PNG path.
        img1      : Optional background image (numpy or torch tensor).
        alpha     : Opacity of the flow layer (0=only image, 1=only flow).
        step      : Subsampling step for arrows/quiver.
        gt_flow   : (H, W, 3) or (H, W, 2) ground truth flow. Required for 'error' mode.
    """
    if mode not in PLOT_MODES:
        raise ValueError(f"Unknown mode '{mode}'. Choose from: {PLOT_MODES}")
    if mode == "error" and gt_flow is None:
        raise ValueError("'error' mode requires gt_flow to be provided.")

    u, v = flow[..., 0], flow[..., 1]
    H, W = flow.shape[:2]
    bg = _load_as_rgb(img1) if img1 is not None else None

    def blend(flow_rgb: np.ndarray) -> np.ndarray:
        if bg is None:
            return flow_rgb
        return cv2.addWeighted(cv2.resize(bg, (W, H)), 1 - alpha, flow_rgb, alpha, 0)

    if mode == "magnitude":
        mag = flow_to_magnitude(flow)
        mag_u8 = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        flow_rgb = cv2.cvtColor(cv2.applyColorMap(mag_u8, cv2.COLORMAP_PLASMA), cv2.COLOR_BGR2RGB)
        cv2.imwrite(save_path, cv2.cvtColor(blend(flow_rgb), cv2.COLOR_RGB2BGR))

    elif mode == "color":
        flow_rgb = flow_to_color(flow)
        cv2.imwrite(save_path, cv2.cvtColor(blend(flow_rgb), cv2.COLOR_RGB2BGR))

    elif mode == "error":
        epe = flow_to_error(flow, gt_flow)
        epe_u8 = cv2.normalize(epe, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        # Hot colormap: blue=low error, red=high error
        flow_rgb = cv2.cvtColor(cv2.applyColorMap(epe_u8, cv2.COLORMAP_HOT), cv2.COLOR_BGR2RGB)
        cv2.imwrite(save_path, cv2.cvtColor(blend(flow_rgb), cv2.COLOR_RGB2BGR))

    elif mode in ("arrows", "quiver"):
        fig, ax = _make_fig(W, H)

        ys = np.arange(0, H, step)
        xs = np.arange(0, W, step)
        X, Y = np.meshgrid(xs, ys)
        U = u[::step, ::step]
        V = v[::step, ::step]
        magnitude = np.sqrt(U**2 + V**2)

        # Background
        if bg is not None:
            ax.imshow(cv2.resize(bg, (W, H)), aspect="auto")
        elif mode == "arrows":
            ax.imshow(flow_to_magnitude(flow), cmap="gray", aspect="auto")
        else:
            ax.set_facecolor("black")

        if mode == "arrows":
            ax.quiver(X, Y, U, V, magnitude, cmap="autumn_r", angles="xy",
                      scale_units="xy", scale=1, width=0.002)
        else:
            ax.quiver(X, Y, U, V, magnitude, cmap="viridis", angles="xy",
                      scale_units="xy", scale=1)
            ax.invert_yaxis()

        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        plt.savefig(save_path, dpi=fig.dpi, pad_inches=0)
        plt.close()
