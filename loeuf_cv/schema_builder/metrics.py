import numpy as np

def calculate_angle(a, b, c):
    """
    Calculate the angle between three points.
    Returns angle in degrees [0, 180].
    """
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba < 1e-6 or norm_bc < 1e-6:
        return 0.0
    cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def calculate_shoulder_rotation(lm, L_SHOULDER, R_SHOULDER):
    dx = lm[R_SHOULDER][0] - lm[L_SHOULDER][0]
    dy = lm[R_SHOULDER][1] - lm[L_SHOULDER][1]
    return np.degrees(np.arctan2(dy, dx))

def calculate_hip_rotation(lm, L_HIP, R_HIP):
    dx = lm[R_HIP][0] - lm[L_HIP][0]
    dy = lm[R_HIP][1] - lm[L_HIP][1]
    return np.degrees(np.arctan2(dy, dx))

def calculate_cm_per_px(lm, vis, L_ANKLE, R_ANKLE, NOSE, actual_height_cm):
    """
    Find pixel-to-cm ratio based on player height.
    """
    if vis[NOSE] < 0.3: return None
    ankle_y = []
    if vis[L_ANKLE] > 0.3: ankle_y.append(lm[L_ANKLE][1])
    if vis[R_ANKLE] > 0.3: ankle_y.append(lm[R_ANKLE][1])
    
    if not ankle_y: return None
    
    avg_ankle_y = np.mean(ankle_y)
    head_y = lm[NOSE][1]
    
    px_height = abs(avg_ankle_y - head_y)
    if px_height < 1e-6: return None
    
    return actual_height_cm / px_height

def get_body_metrics(pose_series, kf, actual_height_cm=170.0, dominant_side="right"):
    """
    Extract full C metrics block based on keyframes.
    """
    from ..config import (
        L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST,
        L_HIP, R_HIP, L_ANKLE, R_ANKLE, NOSE
    )
    
    metrics = {
        "body": {}, "arm": {}, "timing": {}, "contact": {}, "movement": {}, "depth_estimated": None
    }
    
    # Need at least impact frame
    if kf.get("impact") is None or kf["impact"] >= len(pose_series.landmarks):
        return metrics
        
    impact = kf["impact"]
    unit_turn = kf.get("unit_turn")
    backswing = kf.get("backswing_peak")
    follow_through = kf.get("follow_through_peak")
    
    lm_imp = pose_series.landmarks[impact]
    vis_imp = pose_series.visibility[impact]
    
    s_idx = R_SHOULDER if dominant_side == "right" else L_SHOULDER
    e_idx = R_ELBOW if dominant_side == "right" else L_ELBOW
    w_idx = R_WRIST if dominant_side == "right" else L_WRIST
    
    # 1. Body Metrics
    cm_per_px = calculate_cm_per_px(lm_imp, vis_imp, L_ANKLE, R_ANKLE, NOSE, actual_height_cm)
    if cm_per_px:
        metrics["body"]["body_height_cm"] = actual_height_cm
        metrics["body"]["arm_span_cm"] = actual_height_cm * 1.0
        
    if unit_turn is not None and unit_turn < len(pose_series.landmarks):
        lm_ut = pose_series.landmarks[unit_turn]
        metrics["body"]["shoulder_rotation_at_unit_turn_deg"] = calculate_shoulder_rotation(lm_ut, L_SHOULDER, R_SHOULDER)
        metrics["body"]["hip_rotation_at_unit_turn_deg"] = calculate_hip_rotation(lm_ut, L_HIP, R_HIP)
        
    metrics["body"]["shoulder_rotation_at_impact_deg"] = calculate_shoulder_rotation(lm_imp, L_SHOULDER, R_SHOULDER)
    metrics["body"]["hip_rotation_at_impact_deg"] = calculate_hip_rotation(lm_imp, L_HIP, R_HIP)
    metrics["body"]["shoulder_hip_separation_at_impact_deg"] = metrics["body"]["shoulder_rotation_at_impact_deg"] - metrics["body"]["hip_rotation_at_impact_deg"]

    # 2. Arm Metrics
    if follow_through is not None and follow_through < len(pose_series.landmarks):
        lm_ft = pose_series.landmarks[follow_through]
        metrics["arm"]["follow_through_angle_deg"] = calculate_angle(lm_ft[s_idx], lm_ft[e_idx], lm_ft[w_idx])
        
    if backswing is not None and backswing < len(pose_series.landmarks):
        lm_bs = pose_series.landmarks[backswing]
        metrics["arm"]["backswing_depth_deg"] = calculate_angle(lm_bs[s_idx], lm_bs[e_idx], lm_bs[w_idx])

    # 3. Contact
    if cm_per_px and vis_imp[w_idx] > 0.3:
        # Distance from ground (ankles) to wrist
        ankle_y = []
        if vis_imp[L_ANKLE] > 0.3: ankle_y.append(lm_imp[L_ANKLE][1])
        if vis_imp[R_ANKLE] > 0.3: ankle_y.append(lm_imp[R_ANKLE][1])
        if ankle_y:
            ground_y = np.mean(ankle_y)
            contact_px = abs(ground_y - lm_imp[w_idx][1])
            metrics["contact"]["contact_height_cm"] = contact_px * cm_per_px
            
        # Distance from torso center to wrist
        torso_x = (lm_imp[L_SHOULDER][0] + lm_imp[R_SHOULDER][0] + lm_imp[L_HIP][0] + lm_imp[R_HIP][0]) / 4.0
        torso_y = (lm_imp[L_SHOULDER][1] + lm_imp[R_SHOULDER][1] + lm_imp[L_HIP][1] + lm_imp[R_HIP][1]) / 4.0
        dist_px = np.linalg.norm([lm_imp[w_idx][0] - torso_x, lm_imp[w_idx][1] - torso_y])
        metrics["contact"]["contact_distance_from_body_cm"] = dist_px * cm_per_px

    return metrics
