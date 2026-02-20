import sys
from pathlib import Path
import os

sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np
import cv2
from data_processor import AICityFrames
from tqdm import tqdm

'''
    Gaussian Modelling with a single Gaussian
'''

class GaussianModelling:
    
    def __init__(self, dataloader: AICityFrames):
        self.dataloader = dataloader
        
        self._compute_bg_model()
        pass
    
    def _compute_bg_model(self):
        
        bg_model_count = int(self.dataloader.frame_count * 0.25)
                
        first_img = self.dataloader.image(0)
        bg_frames = np.empty((bg_model_count, *first_img.shape), dtype=first_img.dtype)
        
        for i in tqdm(range(bg_model_count), 'Computing background model parameters'):
            bg_frames[i] = self.dataloader.image(i)
        
        self.pixelwise_mean = np.mean(bg_frames, axis=0)
        self.pixelwise_std = np.std(bg_frames, axis=0)
    
    def compute_bg_mask(self, image_idx: int, k=1):
        image = self.dataloader.image(image_idx).astype(np.float64)
        threshold = k * (self.pixelwise_std + 2)
        mask = np.abs(image - self.pixelwise_mean) >= threshold
        return mask.astype(np.uint8) * 255
        

def detect_bboxes_in_frame(image: np.ndarray):
    print(image)
    


dataloader = AICityFrames(scale=0.3)


MASK_FRAMES_PATH = Path("mask_frames")

#if not MASK_FRAMES_PATH.exists() or False:
if True:
    MASK_FRAMES_PATH.mkdir(parents=True, exist_ok=True)

    gm = GaussianModelling(dataloader)

    # Create video writer
    output_path = Path(__file__).parent / "bg_mask_output.mp4"
    h, w = dataloader.image(0).shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, 10, (w, h))

    print(f"Total frames: {dataloader.frame_count}")

    # Process all frames
    for frame_idx in tqdm(range(int(dataloader.frame_count * 0.25), dataloader.frame_count - 1), 'Processing frames'):
        mask = gm.compute_bg_mask(frame_idx, 1)
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        out.write(mask_bgr)
        cv2.imwrite(MASK_FRAMES_PATH / f"mask{ frame_idx:06d}.jpg", mask_bgr)

    out.release()
    print(f"Saved video to {output_path}")

