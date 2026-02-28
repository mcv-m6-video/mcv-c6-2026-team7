import pandas as pd
from pathlib import Path
from typing import Optional


class MOTChallengeConverter:
    """Converter for MOTChallenge format used by TrackEval."""
    
    @staticmethod
    def dataframe_to_motchallenge(df: pd.DataFrame, is_ground_truth: bool = False) -> pd.DataFrame:
        """
        Convert tracking DataFrame to MOTChallenge format.
        
        Args:
            df: DataFrame with columns frame_id, track_id, x1, y1, x2, y2, confidence, etc.
            is_ground_truth: If True, uses conf=1 for all entries
            
        Returns:
            DataFrame with MOTChallenge format: frame, id, x, y, width, height, conf, x_world, y_world, z_world
        """
        result = pd.DataFrame()
        
        # Frame (1-indexed for MOTChallenge)
        result["frame"] = df["frame_id"] + 1
        
        # Track ID
        result["id"] = df["track_id"]
        
        # Bounding box (top-left corner + width/height)
        result["x"] = df["x1"]
        result["y"] = df["y1"]
        result["width"] = df["x2"] - df["x1"]
        result["height"] = df["y2"] - df["y1"]
        
        # Confidence
        if is_ground_truth:
            result["conf"] = 1
        else:
            result["conf"] = df["confidence"]
        
        # World coordinates (not used in 2D tracking)
        result["x_world"] = -1
        result["y_world"] = -1
        result["z_world"] = -1
        
        return result
    
    @staticmethod
    def ground_truth_to_motchallenge(
        annotation_path: Path,
        output_file: Path,
        class_filter: Optional[str] = "car",
        verbose: bool = True
    ) -> None:
        """
        Convert ground truth annotations from XML to MOTChallenge format.
        
        Args:
            annotation_path: Path to the XML annotation file
            output_file: Path to save the MOTChallenge format file
            class_filter: Class label to include (e.g., 'car'), None for all classes
            verbose: Whether to print progress messages
        """
        # Import here to avoid circular dependencies
        import sys
        import os
        from data_processor import AICityFrames
        
        if verbose:
            print(f"Loading annotations from {annotation_path}...")
        
        dataloader = AICityFrames(annotation_path=annotation_path, image_index_base=0)
        
        # Prepare output directory
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to MOTChallenge format
        if verbose:
            print(f"Converting to MOTChallenge format...")
        
        lines = []
        
        for frame_idx in dataloader.frames_with_boxes():
            boxes = dataloader.boxes(frame_idx, include_outside=False)
            
            for box in boxes:
                # Only include specified class (e.g., 'car')
                if class_filter and box.label.lower() != class_filter.lower():
                    continue
                
                # MOTChallenge format is 1-indexed for frames
                frame_num = frame_idx + 1
                track_id = box.track_id
                bb_left = box.xtl
                bb_top = box.ytl
                bb_width = box.xbr - box.xtl
                bb_height = box.ybr - box.ytl
                
                # For ground truth: conf = 1 (active), 0 (ignore)
                conf = 1 if box.occluded == 0 else 0
                
                # World coordinates (not used in 2D)
                x, y, z = -1, -1, -1
                
                # Create line
                line = f"{frame_num},{track_id},{bb_left:.2f},{bb_top:.2f},{bb_width:.2f},{bb_height:.2f},{conf},{x},{y},{z}\n"
                lines.append(line)
        

        print(f"Writing {len(lines)} detections to {output_file}...")
        
        with open(output_file, 'w') as f:
            f.writelines(lines)
        
        if verbose:
            print(f"Ground truth converted successfully!")
            print(f"  Total annotations: {len(lines)}")
            print(f"  Output: {output_file}")
