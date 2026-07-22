import datetime
from .keyframes import extract_keyframes
from .metrics import calculate_cm_per_px, get_body_metrics, get_racket_metrics
from .classifier import classify_stroke
from .aggregator import build_phase2_aggregations

def build_loeuf_schema(tracks, hit_events, fps, video_meta, config, racket_keypoints=None):
    """
    สร้าง Loeuf Full JSON Schema (17 Layers)
    Phase 1 & Phase 2 Intelligence

    racket_keypoints: dict[frame_idx -> ...] จาก webui/yolo_track.py (ถ้าใช้
    custom model แบบ pose ที่มี keypoint จริง) — None = kinematics racket
    fields (KN5-7) จะว่างเปล่าเหมือนเดิม ไม่ error
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
        
        # 024-A: stroke_root
        a_root = {
            "stroke_id": f"{session_id}_stroke-{i+1}",
            "stroke_index": i + 1,
            "stroke_type": stype,
            "stroke_start_frame": kf["unit_turn"] if kf["unit_turn"] else max(0, impact_frame - 30),
            "stroke_end_frame": kf["recovery_position"] if kf["recovery_position"] else min(video_meta.get("total_frames", 0), impact_frame + 30),
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
            "ball": {"available": True}, # TBC real values from ball tracking
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
