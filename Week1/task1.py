import sys
from pathlib import Path

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
        imgs_differences = np.empty((bg_model_count - 1, *first_img.shape), dtype=first_img.dtype)
        
        prev_img = first_img
        for i in tqdm(range(1, bg_model_count), 'Computing background model parameters'):
            curr_img = self.dataloader.image(i)
            imgs_differences[i - 1] = np.abs(curr_img.astype(np.int16) - prev_img.astype(np.int16)).astype(np.uint8)
            prev_img = curr_img
        
        self.pixelwise_mean = np.mean(imgs_differences, axis=0)
        self.pixelwise_variance = np.var(imgs_differences, axis=0)
    
    def compute_bg_mask(self, image_idx: int, k=2.5):
        image = self.dataloader.image(image_idx).astype(np.float64)
        threshold = k * np.sqrt(self.pixelwise_variance)
        mask = np.abs(image - self.pixelwise_mean) <= threshold
        return mask.astype(np.uint8) * 255
        


dataloader = AICityFrames(scale=0.3)

gm = GaussianModelling(dataloader)

# Create video writer
output_path = Path(__file__).parent / "bg_mask_output.mp4"
h, w = dataloader.image(0).shape
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(str(output_path), fourcc, 10, (w, h))

print(f"Total frames: {dataloader.frame_count}")

# Process all frames
for frame_idx in tqdm(range(int(dataloader.frame_count * 0.25), dataloader.frame_count - 1), 'Processing frames'):
    mask = gm.compute_bg_mask(image_idx=frame_idx)
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    out.write(mask_bgr)

out.release()
print(f"Saved video to {output_path}")