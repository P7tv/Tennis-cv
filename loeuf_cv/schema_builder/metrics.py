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


def _pick_racket_detection(dets, wrist_px):
    """เลือก racket detection ที่ centroid ใกล้ข้อมือ dominant สุด — กันเลือก
    ไม้ผิดตัวถ้ามีมากกว่า 1 ไม้ในเฟรม (คนป้อนบอล ฯลฯ) — fallback ตัวแรกถ้าไม่มี wrist"""
    if not dets:
        return None
    if wrist_px is None:
        return dets[0]

    best_det, best_d = dets[0], float("inf")
    for det in dets:
        valid = [(x, y) for x, y, v in det if v > 0.3]
        if not valid:
            continue
        cx = sum(x for x, y in valid) / len(valid)
        cy = sum(y for x, y in valid) / len(valid)
        d = (cx - wrist_px[0]) ** 2 + (cy - wrist_px[1]) ** 2
        if d < best_d:
            best_d = d
            best_det = det
    return best_det


def _racket_tip_center_px(det):
    """centroid ของ keypoint ที่มั่นใจ (v>0.3) = center, จุดที่ไกล centroid สุด = tip
    (สมมติฐานเบื้องต้น — 4 จุดจาก label เดิมไม่มี index ความหมายตายตัว)"""
    valid = [(x, y) for x, y, v in det if v > 0.3]
    if len(valid) < 2:
        return None
    cx = sum(x for x, y in valid) / len(valid)
    cy = sum(y for x, y in valid) / len(valid)
    tip = max(valid, key=lambda p: (p[0] - cx) ** 2 + (p[1] - cy) ** 2)
    return {"center_px": (cx, cy), "tip_px": tip}


def get_racket_metrics(racket_keypoints, kf, wrist_path, fps, frame_width, frame_height, cm_per_px=None):
    """
    KN5 (racket_tip_position) / KN6 (racket_center_position) / KN7
    (racket_head_speed_mps) — คำนวณจาก keypoint จริงของ racket pose model
    (train_model/build_pose_dataset.py + yolo11m-pose.pt) ถ้ามี

    racket_keypoints: dict[frame_idx -> list ต่อ detection -> list[(x_px, y_px, conf)] ต่อ keypoint]
    (พิกัด pixel จาก webui/yolo_track.py) — normalize เป็น 0-1 ก่อนเก็บ position
    ให้ตรงฟอร์แมตเดียวกับ pose landmark อื่นในระบบ

    ถ้าไม่มี racket_keypoints (โมเดลเก่าไม่มี keypoint / ตรวจไม่เจอ) คืน dict
    ว่างเปล่า ไม่ error — caller ใช้แทนที่ "kinematics": {} เดิมได้ตรงๆ
    """
    result = {}
    if not racket_keypoints or not frame_width or not frame_height:
        return result

    def pts_at(frame_idx):
        dets = racket_keypoints.get(frame_idx)
        if not dets:
            return None
        wrist_px = None
        if wrist_path is not None and frame_idx < len(wrist_path):
            wx, wy = wrist_path[frame_idx]
            if not (np.isnan(wx) or np.isnan(wy)):
                wrist_px = (wx * frame_width, wy * frame_height)
        det = _pick_racket_detection(dets, wrist_px)
        return _racket_tip_center_px(det) if det else None

    tip_positions, center_positions = {}, {}
    for name in ("unit_turn", "backswing_peak", "impact", "follow_through_peak"):
        f = kf.get(name)
        if f is None:
            continue
        pts = pts_at(f)
        if pts is None:
            continue
        tip_positions[name] = {"x": round(pts["tip_px"][0] / frame_width, 4),
                                "y": round(pts["tip_px"][1] / frame_height, 4)}
        center_positions[name] = {"x": round(pts["center_px"][0] / frame_width, 4),
                                   "y": round(pts["center_px"][1] / frame_height, 4)}

    if tip_positions:
        result["racket_tip_position"] = tip_positions
    if center_positions:
        result["racket_center_position"] = center_positions

    # KN7: peak racket head speed รอบๆ impact (±5 เฟรม) จาก tip trajectory จริง
    impact_f = kf.get("impact")
    if impact_f is not None and cm_per_px:
        traj = []
        for f in range(max(0, impact_f - 5), impact_f + 6):
            pts = pts_at(f)
            if pts is not None:
                traj.append((f, pts["tip_px"]))

        speeds_mps = []
        for (f1, p1), (f2, p2) in zip(traj, traj[1:]):
            dt = (f2 - f1) / fps
            if dt <= 0:
                continue
            dist_norm = (((p2[0] - p1[0]) / frame_width) ** 2 + ((p2[1] - p1[1]) / frame_height) ** 2) ** 0.5
            dist_cm = dist_norm * cm_per_px
            speeds_mps.append((dist_cm / 100.0) / dt)

        if speeds_mps:
            result["racket_head_speed_mps"] = round(max(speeds_mps), 2)

    return result
