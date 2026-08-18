import cv2
import numpy as np
import torch
from typing import Optional, Tuple

from .weights import load_tracknet_model
from loeuf_cv.ball import BallObservations
from loeuf_cv.pose_extractor import read_video_meta

def extract_ball_coordinates(heatmap: np.ndarray, threshold: float = 0.5) -> Optional[Tuple[float, float, float]]:
    """
    Extract (x, y, confidence) from 1-channel TrackNet heatmap.
    heatmap: (H, W) array of floats [0, 1]
    returns: (x_norm, y_norm, confidence) or None if below threshold
    """
    # Find the maximum value in the heatmap
    max_val = np.max(heatmap)
    if max_val < threshold:
        return None
        
    # Find the coordinates of the maximum value
    y_idx, x_idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    
    H, W = heatmap.shape
    x_norm = float(x_idx) / W
    y_norm = float(y_idx) / H
    
    return (x_norm, y_norm, float(max_val))

def track_ball_with_tracknet(
    video_path: str,
    conf_threshold: float = 0.5,
    batch_size: int = 16,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Optional[BallObservations]:
    """
    Process video with TrackNetV2 and return BallObservations.
    Returns None if TrackNet model fails to load.
    """
    model = load_tracknet_model(device=device)
    if model is None:
        return None
        
    meta = read_video_meta(video_path)
    total_frames = meta.frame_count
    
    cap = cv2.VideoCapture(video_path)
    frames = []
    positions = np.full((total_frames, 2), np.nan)
    confidences = np.zeros(total_frames)
    
    TARGET_H, TARGET_W = 360, 640
    
    def process_batch(batch_frames, frame_indices):
        if not batch_frames:
            return
            
        # batch_frames is list of shape (3, H, W, 3)
        # We need (B, 9, 360, 640)
        batch_tensors = []
        for triplet in batch_frames:
            # triplet: 3 frames
            resized = [cv2.resize(f, (TARGET_W, TARGET_H)) for f in triplet]
            # Convert to float and normalize [0, 1]
            resized = [f.astype(np.float32) / 255.0 for f in resized]
            # Concatenate along channel axis: (360, 640, 9)
            concat = np.concatenate(resized, axis=2)
            # Transpose to (9, 360, 640)
            concat = np.transpose(concat, (2, 0, 1))
            batch_tensors.append(concat)
            
        tensor = torch.tensor(np.array(batch_tensors), dtype=torch.float32, device=device)
        
        with torch.no_grad():
            heatmaps = model(tensor) # (B, 1, 360, 640)
            
        heatmaps_np = heatmaps.cpu().numpy()[:, 0, :, :] # (B, 360, 640)
        
        for i, hm in enumerate(heatmaps_np):
            res = extract_ball_coordinates(hm, conf_threshold)
            if res is not None:
                x_norm, y_norm, conf = res
                # Map back to original resolution
                x = x_norm * meta.width
                y = y_norm * meta.height
                
                # The output corresponds to the middle frame of the triplet
                f_idx = frame_indices[i]
                positions[f_idx] = (x, y)
                confidences[f_idx] = conf

    # We need 3 consecutive frames (t-1, t, t+1) for each prediction at time t.
    # Buffer stores recent frames.
    buffer = []
    batch_frames = []
    frame_indices = []
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        buffer.append(frame)
        if len(buffer) > 3:
            buffer.pop(0)
            
        if len(buffer) == 3:
            # We have frames for t-1, t, t+1. 
            # The middle frame (t) is at index frame_idx - 1
            target_idx = frame_idx - 1
            batch_frames.append(list(buffer))
            frame_indices.append(target_idx)
            
            if len(batch_frames) >= batch_size:
                process_batch(batch_frames, frame_indices)
                batch_frames = []
                frame_indices = []
                
        frame_idx += 1
        
    # Process remaining
    if batch_frames:
        process_batch(batch_frames, frame_indices)
        
    cap.release()
    
    # TrackNet does not produce predictions for the very first (t=0) and last (t=N-1) frames
    # because it needs a 3-frame window. We could pad or duplicate frames, but returning NaN is safe.
    
    return BallObservations(positions=positions, confidence=confidences, fps=meta.fps)
