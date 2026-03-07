import os
import cv2
import time
import argparse
import numpy as np
from datetime import datetime
from torchvision.io import read_image

from .utils import (
    read_kitti_flow,
    compute_optical_flow,
    calculate_msen_pepn,
)

from .plot_flow import plot_flow, PLOT_MODES


def evaluate_model(model: str, params: list, img1: np.ndarray, img2: np.ndarray, gt: np.ndarray) -> tuple[float, float, float]:
    
    # Compute inference time
    start_time = time.time()
    optical_flow = compute_optical_flow(model, params, img1, img2)
    runtime = time.time() - start_time
    
    # Calculate MSEN and PEPN
    msen, pepn = calculate_msen_pepn(gt, optical_flow)

    return msen, pepn, runtime, optical_flow


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
    
    elif args.model == "flowformer_pp":
        params = [args.flowformer_ckpt]
        img1 = read_image(args.img1_path)
        img2 = read_image(args.img2_path)
        img1 = img1.repeat(3, 1, 1)
        img2 = img2.repeat(3, 1, 1)
    
    msen, pepn, runtime, optical_flow = evaluate_model(args.model, params, img1, img2, gt)

    print(f"Model   : {args.model}")
    print(f"MSEN    : {msen:.4f}")
    print(f"PEPN    : {pepn * 100:.2f}%")
    print(f"Runtime : {runtime:.2f} s")

    if args.plot or args.plot_gt:
        dt = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join("optical_flow/results", f"{args.model}_{dt}")
        os.makedirs(run_dir, exist_ok=True)

        if args.plot:
            for mode in args.plot:
                save_path = os.path.join(run_dir, f"{mode}.png")
                plot_flow(optical_flow, mode=mode, save_path=save_path, img1=img1,
                          alpha=args.plot_alpha, step=args.plot_step, gt_flow=gt)

        if args.plot_gt:
            gt_dir = os.path.join(run_dir, "ground_truth")
            os.makedirs(gt_dir, exist_ok=True)
            for mode in args.plot_gt:
                save_path = os.path.join(gt_dir, f"{mode}.png")
                plot_flow(gt[:, :, :2], mode=mode, save_path=save_path, img1=img1,
                          alpha=args.plot_alpha, step=args.plot_step, gt_flow=gt)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=["pyflow", "raft_large", "raft_small", "flowformer_pp"], default="pyflow")
    parser.add_argument('--flowformer_ckpt', default="external/flowformerpp/checkpoints/kitti.pth")
    parser.add_argument('--gt_path', help="Path to the Ground Truth stereo flow", default="data/data_stereo_flow/training/flow_noc/000045_10.png")
    parser.add_argument('--img1_path', help="Path to the first image.", default="data/data_stereo_flow/training/image_0/000045_10.png")
    parser.add_argument('--img2_path', help="Path to the second image.", default="data/data_stereo_flow/training/image_0/000045_11.png")
    
    # Plot arguments
    parser.add_argument('--plot', nargs="+", choices=PLOT_MODES, default=["color"], metavar="MODE")
    parser.add_argument('--plot_gt', nargs="+", choices=PLOT_MODES)
    parser.add_argument('--plot_step', type=int, default=12)
    parser.add_argument('--plot_alpha', type=float, default=0.5)

    
    args = parser.parse_args()
    main(args)