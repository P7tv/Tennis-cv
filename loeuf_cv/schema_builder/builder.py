import datetime
import numpy as np
from .keyframes import extract_keyframes
from .metrics import calculate_cm_per_px, get_body_metrics, get_racket_metrics
from .classifier import classify_stroke
from .aggregator import build_phase2_aggregations
from ..ball import BallObservations, build_ball_block

def build_loeuf_schema(tracks, hit_events, fps, video_meta, config, racket_keypoints=None, ball_traj=None):
    """
    สร้าง Loeuf Full JSON Schema (17 Layers)
    Phase 1 & Phase 2 Intelligence

    racket_keypoints: dict[frame_idx -> ...] จาก webui/yolo_track.py (ถ้าใช้
    custom model แบบ pose ที่มี keypoint จริง) — None = kinematics racket
    fields (KN5-7) จะว่างเปล่าเหมือนเดิม ไม่ error

    ball_traj: np.ndarray (total_frames, 2) พิกัด pixel จาก
    extract_ball_trajectory_kalman() (NaN = ไม่เจอลูกเฟรมนั้น) — ใช้คำนวณ BL
    block จริง (available + _tracked_fraction) แทน placeholder เดิม —
    ball_speed_kmh/landing_* ยังเป็น None เพราะต้องมี court calibration ก่อน
    (ดู loeuf_cv/ball.py, scope แยก) — None = "ball": {"available": False} เหมือนเดิม
    """
    import uuid
    session_id = f"sess-{datetime.datetime.now().strftime('%Y%m%d-%H%M')}"
    
    # 011-MT: session_metadata
    mt = {
        "cv_model_version": "0.2.0",
        "pose_model": "mediapipe_blazepose",
        "schema_version": "1.0",
        "session_id": session_id,
        "session_date": datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "fps": fps,
        "camera_angle": "behind_baseline",
        "total_duration_sec": round(video_meta.get("total_frames", 0) / fps, 1) if fps else 0,
        "total_session_frame": video_meta.get("total_frames", 0),
        "total_strokes_detected": len(hit_events),
        "usable_strokes": len(hit_events),
        "stroke_type_distribution": {} # Will calculate at the end
    }
    
    strokes = []
    stroke_counts = {"FH": 0, "BH": 0, "SV": 0, "VL": 0, "SL": 0, "RS": 0}
    
    for i, hit in enumerate(hit_events):
        impact_frame = hit["frame"]
        player_id = hit["player_id"]
        
        # Find player track
        track = next((t for t in tracks if t.track_id == player_id), None)
        if not track: continue
        
        from ..config import R_WRIST, L_WRIST, L_ANKLE, R_ANKLE, NOSE
        dominant_side = config.dominant_side if hasattr(config, 'dominant_side') else "right"
        wrist_idx = R_WRIST if dominant_side == "right" else L_WRIST
        wrist_path = track.pose.landmarks[:, wrist_idx, :2]

        # Extract Keyframes (B)
        kf = extract_keyframes(impact_frame, wrist_path, fps, video_meta.get("total_frames", 0))

        # Metrics (C) — คำนวณก่อน classify_stroke เพื่อส่งเป็น feature เสริมให้ ML-assist (ถ้ามี stroke_classifier.pkl)
        actual_height = config.subject_height_cm if hasattr(config, 'subject_height_cm') and config.subject_height_cm else 170.0
        c_metric = get_body_metrics(track.pose, kf, actual_height, dominant_side)

        # Stroke Type (A)
        stype = classify_stroke(track.pose, impact_frame, dominant_side, keyframe_metrics=c_metric)
        if stype in stroke_counts:
            stroke_counts[stype] += 1

        # Kinematics (KN) — racket_tip/center_position + racket_head_speed_mps
        # จาก keypoint จริง (ถ้า custom model เป็น pose model) — ไม่มีก็ได้ dict ว่างเปล่า
        cm_per_px = calculate_cm_per_px(
            track.pose.landmarks[impact_frame], track.pose.visibility[impact_frame],
            L_ANKLE, R_ANKLE, NOSE, actual_height)
        kn_metric = get_racket_metrics(
            racket_keypoints, kf, wrist_path, fps,
            video_meta.get("width", 0), video_meta.get("height", 0), cm_per_px)
        
        # 021-M: stroke_metadata
        m = {
            "debug_pose_confidence": 0.95,
            "processed_at": datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "resolution": f"{video_meta.get('width', 1920)}x{video_meta.get('height', 1080)}",
            "camera_angle": "behind_baseline",
            "velocity_normalized": True,
            "normalization_method": "resampled_to_60fps"
        }
        
        # 022-VQ: video_quality
        vq = {
            "usable": True,
            "issues": [],
            "partial_swing_detail": None
        }
        
        # 023-B: keyframe
        b_keyframe = {
            "unit_turn": {"frame_index": kf["unit_turn"], "detected": kf["unit_turn"] is not None},
            "backswing_peak": {"frame_index": kf["backswing_peak"], "detected": kf["backswing_peak"] is not None},
            "impact": {"frame_index": kf["impact"], "detected": True},
            "follow_through_peak": {"frame_index": kf["follow_through_peak"], "detected": kf["follow_through_peak"] is not None},
            "recovery_position": {"frame_index": kf["recovery_position"], "detected": kf["recovery_position"] is not None}
        }
        
        stroke_start_frame = kf["unit_turn"] if kf["unit_turn"] else max(0, impact_frame - 30)
        stroke_end_frame = kf["recovery_position"] if kf["recovery_position"] else min(video_meta.get("total_frames", 0), impact_frame + 30)

        # Ball (BL) — available/_tracked_fraction จริงจาก ball trajectory (Kalman-filtered)
        # ในช่วงเฟรมของ stroke นี้ — ball_speed_kmh/landing_* ยังเป็น None เสมอ
        # เพราะต้องมี court calibration ก่อน (ดู docstring ด้านบน)
        ball_metric = {"available": False}
        if ball_traj is not None and video_meta.get("width") and video_meta.get("height"):
            seg = np.asarray(ball_traj[stroke_start_frame:stroke_end_frame], dtype=float)
            if len(seg):
                norm_seg = seg / [video_meta["width"], video_meta["height"]]
                conf = (~np.isnan(norm_seg[:, 0])).astype(float)
                ball_metric = build_ball_block(BallObservations(norm_seg, conf, fps))

        # 024-A: stroke_root
        a_root = {
            "stroke_id": f"{session_id}_stroke-{i+1}",
            "stroke_index": i + 1,
            "stroke_type": stype,
            "stroke_start_frame": stroke_start_frame,
            "stroke_end_frame": stroke_end_frame,
            "dominant_side": dominant_side,
            "detection_status": "valid",
            "is_clean_stroke": True,
            "stroke_recommended_action": "auto_accept"
        }

        stroke = {
            "stroke_metadata": m,
            "video_quality": vq,
            "keyframe": b_keyframe,
            "stroke_root": a_root,
            "metric": c_metric,
            "kinematics": kn_metric,
            "stroke_specific": {}, # TBC
            "ball": ball_metric,
            "derived": {},
            "visibility_flag": {},
            "confidence_flag": {"pose_estimation_confidence": 0.9, "inference_clean": True},
            "visualization": {"wrist_path": []}
        }
        
        strokes.append(stroke)
        
    mt["stroke_type_distribution"] = {k: v for k, v in stroke_counts.items() if v > 0}
    
    # Phase 2 Analytics
    agg, trend, pattern, summary = build_phase2_aggregations(strokes)
    
    # Remove _index used internally
    for s in strokes:
        s.pop("_index", None)
        
    session = {
        "session_metadata": mt,
        "strokes": strokes,
        "aggregated_metric": agg,
        "trend": trend,
        "pattern": pattern,
        "session_summary": summary
    }
    
    return session
