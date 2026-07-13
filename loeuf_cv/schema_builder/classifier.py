import numpy as np
from ..config import L_SHOULDER, R_SHOULDER, L_WRIST, R_WRIST, NOSE

def classify_stroke(pose_series, impact_frame, dominant_side="right"):
    """
    Classify the stroke type (FH, BH, SV) based on pose at impact.
    """
    if impact_frame is None or impact_frame >= len(pose_series.landmarks):
        return "FH" # Default fallback
        
    lm = pose_series.landmarks[impact_frame]
    vis = pose_series.visibility[impact_frame]
    
    # Check if necessary landmarks are visible
    if vis[NOSE] < 0.3 or vis[R_SHOULDER] < 0.3 or vis[L_SHOULDER] < 0.3:
        return "FH"
        
    wrist_idx = R_WRIST if dominant_side == "right" else L_WRIST
    if vis[wrist_idx] < 0.3:
        return "FH"
        
    wrist_y = lm[wrist_idx][1]
    head_y = lm[NOSE][1]
    shoulder_y = min(lm[R_SHOULDER][1], lm[L_SHOULDER][1])
    
    # 1. Check for Serve (SV): Contact point is significantly above the head
    # Note: Y-axis is 0 at top of image, so lower Y means higher physical position
    if wrist_y < head_y - 0.05: # Arbitrary threshold for overhead reach
        return "SV"
        
    # 2. Check for FH vs BH based on lateral position
    wrist_x = lm[wrist_idx][0]
    spine_x = (lm[R_SHOULDER][0] + lm[L_SHOULDER][0]) / 2.0
    
    if dominant_side == "right":
        # Right-handed player (from behind): Right is +X direction
        if wrist_x > spine_x:
            return "FH"
        else:
            return "BH"
    else:
        # Left-handed player (from behind): Left is -X direction
        if wrist_x < spine_x:
            return "FH"
        else:
            return "BH"
