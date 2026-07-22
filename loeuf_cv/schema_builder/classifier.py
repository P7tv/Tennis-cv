import os
import pickle

import numpy as np
from ..config import L_SHOULDER, R_SHOULDER, L_WRIST, R_WRIST, NOSE

# ML เอาชนะ rule-based ก็ต่อเมื่อมั่นใจเกิน threshold นี้เท่านั้น (ไม่งั้น fallback
# ไป rule-based) — ค่าเดียวกับ PipelineConfig.coach_review_threshold (config.py)
# เพราะ class ที่มีตัวอย่างเทรนน้อยมาก (เช่น BH ใน set2 มีแค่ 3 ตัวอย่าง) โมเดลจะ
# ไม่ค่อยมั่นใจเป็นธรรมชาติอยู่แล้ว — threshold นี้เลยกัน BH ให้กลับไปใช้ rule-based
# (geometry ล้วน ไม่พึ่งข้อมูลเทรน) โดยอัตโนมัติ ไม่ต้อง hardcode แยกเป็นกรณีพิเศษ
ML_CONFIDENCE_THRESHOLD = 0.60

_stroke_model_cache = None
_stroke_model_loaded = False


def _load_stroke_model():
    """lazy-load stroke_classifier.pkl จาก CWD ถ้ามี (เทรนจาก
    train_model/train_stroke_classifier.py) — เลียนแบบ pattern เดียวกับที่
    loeuf_cv/hit_detection.py ใช้กับ hit_classifier.pkl: ถ้าไม่มีไฟล์ →
    fallback ไป rule-based ทั้งหมด ไม่ error"""
    global _stroke_model_cache, _stroke_model_loaded
    if not _stroke_model_loaded:
        _stroke_model_loaded = True
        if os.path.exists("stroke_classifier.pkl"):
            try:
                with open("stroke_classifier.pkl", "rb") as f:
                    _stroke_model_cache = pickle.load(f)  # (clf, feature_cols, ml_supported_classes)
            except Exception as e:
                print(f"Failed to load stroke classifier: {e}")
    return _stroke_model_cache


def _rule_based_classify(lm, wrist_idx, dominant_side):
    wrist_y = lm[wrist_idx][1]
    head_y = lm[NOSE][1]

    # 1. Check for Serve (SV): Contact point is significantly above the head
    # Note: Y-axis is 0 at top of image, so lower Y means higher physical position
    if wrist_y < head_y - 0.05:  # Arbitrary threshold for overhead reach
        return "SV"

    # 2. Check for FH vs BH based on lateral position
    wrist_x = lm[wrist_idx][0]
    spine_x = (lm[R_SHOULDER][0] + lm[L_SHOULDER][0]) / 2.0

    if dominant_side == "right":
        # Right-handed player (from behind): Right is +X direction
        return "FH" if wrist_x > spine_x else "BH"
    else:
        # Left-handed player (from behind): Left is -X direction
        return "FH" if wrist_x < spine_x else "BH"


def classify_stroke(pose_series, impact_frame, dominant_side="right", keyframe_metrics: dict | None = None):
    """
    Classify the stroke type (FH, BH, SV) based on pose at impact.

    ถ้ามี stroke_classifier.pkl (เทรนจาก client-labeled data ผ่าน
    train_model/train_stroke_classifier.py) จะใช้ ML predict แทน rule-based —
    แต่เฉพาะ stroke type ที่ตอนเทรนมีตัวอย่างพอ (ML_supported_classes) และ ML
    มั่นใจเกิน ML_CONFIDENCE_THRESHOLD เท่านั้น ไม่งั้น fallback ไป rule-based
    เสมอ (กัน class ที่ข้อมูลน้อยอย่าง BH ให้ยังพึ่ง geometry ได้)
    — ยิ่งส่ง keyframe_metrics (ผลจาก get_body_metrics() ที่ keyframe เดียวกัน)
    เข้ามาด้วย ยิ่งได้ feature ครบขึ้น ไม่ส่งมาก็ยังทำงานได้ (fallback เป็น 0
    เหมือนตอนเทรนที่ fillna(0))
    """
    if impact_frame is None or impact_frame >= len(pose_series.landmarks):
        return "FH"  # Default fallback

    lm = pose_series.landmarks[impact_frame]
    vis = pose_series.visibility[impact_frame]

    # Check if necessary landmarks are visible
    if vis[NOSE] < 0.3 or vis[R_SHOULDER] < 0.3 or vis[L_SHOULDER] < 0.3:
        return "FH"

    wrist_idx = R_WRIST if dominant_side == "right" else L_WRIST
    if vis[wrist_idx] < 0.3:
        return "FH"

    rule_result = _rule_based_classify(lm, wrist_idx, dominant_side)

    model_bundle = _load_stroke_model()
    if model_bundle is None:
        return rule_result

    if len(model_bundle) == 3:
        clf, feature_cols, ml_supported_classes = model_bundle
    else:  # เผื่อ pkl เก่าที่เทรนก่อนมี ml_supported_classes
        clf, feature_cols = model_bundle
        ml_supported_classes = None

    # rule-based ทายว่าเป็น class ที่ ML ไม่เคยเห็นข้อมูลพอ (เช่น BH ตอนเทรนมีน้อย)
    # -> เชื่อ rule-based ไปเลย ไม่ถาม ML เพราะ ML ไม่มีทางรู้ตัวว่าไม่มั่นใจสำหรับ
    # class ที่ตัวเองไม่เคยเรียนรู้ decision boundary จริงจัง (ดูเหตุผลใน
    # train_model/train_stroke_classifier.py: MIN_CLASS_SAMPLES)
    if ml_supported_classes is not None and rule_result not in ml_supported_classes:
        return rule_result

    spine_x = (lm[L_SHOULDER][0] + lm[R_SHOULDER][0]) / 2.0
    raw_dx = lm[wrist_idx][0] - spine_x
    feats = {
        "wrist_minus_head_y": float(lm[wrist_idx][1] - lm[NOSE][1]),
        "wrist_minus_spine_x_dominant_relative": float(raw_dx if dominant_side == "right" else -raw_dx),
    }
    if keyframe_metrics:
        feats.update({
            "body_shoulder_rotation_at_impact_deg": keyframe_metrics.get("body", {}).get("shoulder_rotation_at_impact_deg"),
            "body_hip_rotation_at_impact_deg": keyframe_metrics.get("body", {}).get("hip_rotation_at_impact_deg"),
            "body_shoulder_hip_separation_at_impact_deg": keyframe_metrics.get("body", {}).get("shoulder_hip_separation_at_impact_deg"),
            "arm_follow_through_angle_deg": keyframe_metrics.get("arm", {}).get("follow_through_angle_deg"),
            "arm_backswing_depth_deg": keyframe_metrics.get("arm", {}).get("backswing_depth_deg"),
            "contact_height_cm": keyframe_metrics.get("contact", {}).get("contact_height_cm"),
            "contact_distance_from_body_cm": keyframe_metrics.get("contact", {}).get("contact_distance_from_body_cm"),
        })

    try:
        import pandas as pd
        X = pd.DataFrame([{col: feats.get(col, 0) or 0 for col in feature_cols}])
        proba = clf.predict_proba(X)[0]
        best_idx = int(np.argmax(proba))
        best_prob = float(proba[best_idx])
        if best_prob < ML_CONFIDENCE_THRESHOLD:
            return rule_result
        return str(clf.classes_[best_idx])
    except Exception as e:
        print(f"Stroke ML predict failed, falling back to rule-based: {e}")
        return rule_result
