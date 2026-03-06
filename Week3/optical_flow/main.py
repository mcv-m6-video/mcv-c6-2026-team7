import cv2
import time
import argparse
import numpy as np
from torchvision.io import read_image

from .utils import (
    read_kitti_flow,
    compute_optical_flow,
    calculate_msen_pepn
)


def evaluate_model(model: str, params: list, img1: np.ndarray, img2: np.ndarray, gt: np.ndarray) -> tuple[float, float, float]:
    
    # Compute inference time
    start_time = time.time()
    optical_flow = compute_optical_flow(model, params, img1, img2)
    runtime = time.time() - start_time
    
    # Calculate MSEN and PEPN
    msen, pepn = calculate_msen_pepn(gt, optical_flow)

    return msen, pepn, runtime


def main(args):

    gt = read_kitti_flow(args.gt_path)

    if args.model == "pyflow":

        params = [0.012, 0.75, 20, 7, 1, 30, 0]

        img1 = cv2.imread(args.img1_path, cv2.IMREAD_GRAYSCALE)  
        img1 = np.atleast_3d(img1.astype(float) / 255.0)
        img2 = cv2.imread(args.img2_path, cv2.IMREAD_GRAYSCALE)
        img2 = np.atleast_3d(img2.astype(float) / 255.0)

    elif args.model == "raft_large" or args.model == "raft_small":
        
        params = []
        
        # We use torchvision's read_image
        img1 = read_image(args.img1_path)  # CxHxW uint8 tensor
        img2 = read_image(args.img2_path)  # CxHxW uint8 tensor

        # Images are grayscale but RAFT expects 3 channels
        img1 = img1.repeat(3, 1, 1)  # (3, H, W)
        img2 = img2.repeat(3, 1, 1)  # (3, H, W)
    
    msen, pepn, runtime = evaluate_model(args.model, params, img1, img2, gt)

    print(f"Model   : {args.model}")
    print(f"MSEN    : {msen:.4f}")
    print(f"PEPN    : {pepn * 100:.2f}%")
    print(f"Runtime : {runtime:.2f} s")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=["pyflow", "raft_large", "raft_small"], default="pyflow")
    parser.add_argument('--gt_path', help="Path to the Ground Truth stereo flow", default="data/data_stereo_flow/training/flow_noc/000045_10.png")
    parser.add_argument('--img1_path', help="Path to the first image.", default="data/data_stereo_flow/training/image_0/000045_10.png")
    parser.add_argument('--img2_path', help="Path to the second image.", default="data/data_stereo_flow/training/image_0/000045_11.png")
    args = parser.parse_args()

    main(args)