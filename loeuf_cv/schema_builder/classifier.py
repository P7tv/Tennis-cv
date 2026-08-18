import os
import pickle

import numpy as np
from ..config import (
    L_ANKLE, L_ELBOW, L_HIP, L_KNEE, L_SHOULDER, L_WRIST,
    R_ANKLE, R_ELBOW, R_HIP, R_KNEE, R_SHOULDER, R_WRIST, NOSE
)

# ML เอาชนะ rule-based ก็ต่อเมื่อมั่นใจเกิน threshold นี้เท่านั้น (ไม่งั้น fallback
# ไป rule-based) — ค่าเดียวกับ PipelineConfig.coach_review_threshold (config.py)
# เพราะ class ที่มีตัวอย่างเทรนน้อยมาก (เช่น BH ใน set2 มีแค่ 3 ตัวอย่าง) โมเดลจะ
# ไม่ค่อยมั่นใจเป็นธรรมชาติอยู่แล้ว — threshold นี้เลยกัน BH ให้กลับไปใช้ rule-based
# (geometry ล้วน ไม่พึ่งข้อมูลเทรน) โดยอัตโนมัติ ไม่ต้อง hardcode แยกเป็นกรณีพิเศษ
ML_CONFIDENCE_THRESHOLD = 0.60

# ไม่มี backswing_peak/follow_through_peak (keyframe ตรวจไม่เจอ) → ใช้ช่วงนี้
# รอบ impact แทน ให้ยังได้ "ทั้งสวิง" มาคิด ไม่ใช่แค่เฟรมเดียว
WINDOW_FALLBACK_FRAMES = 15
MIN_VISIBLE_WRIST_CONF = 0.3

_stroke_model_cache = None
_stroke_model_loaded = False


def _window_bounds(keyframes: dict | None, impact_frame: int, n_frames: int) -> tuple[int, int]:
    """ช่วงเฟรม [start, end] (inclusive) ที่ครอบคลุมทั้ง action ของสวิง — ใช้
    backswing_peak..follow_through_peak ถ้ามี keyframe จริง (จาก label หรือ
    extract_keyframes()) ไม่งั้น fallback เป็น ±WINDOW_FALLBACK_FRAMES รอบ impact"""
    start = end = None
    if keyframes:
        start = keyframes.get("backswing_peak")
        end = keyframes.get("follow_through_peak")
    if start is None:
        start = impact_frame - WINDOW_FALLBACK_FRAMES
    if end is None:
        end = impact_frame + WINDOW_FALLBACK_FRAMES
    start = max(0, min(start, n_frames - 1))
    end = max(start + 1, min(end, n_frames - 1))
    return start, end


def swing_window_features(landmarks: np.ndarray, visibility: np.ndarray, wrist_idx: int,
                          dominant_side: str, keyframes: dict | None, impact_frame: int) -> dict:
    """min/max ของสัญญาณ FH/BH/SV ตลอดช่วงสวิง (ไม่ใช่แค่เฟรมเดียว ณ impact) —
    ใช้ทั้งตอนเทรน (train_model/build_training_data.py) และตอน predict จริง
    (classify_stroke ด้านล่าง) ฟังก์ชันเดียวกัน กัน train/inference feature
    ไม่ตรงกันโดยไม่ตั้งใจ

    เฟรมที่ wrist visibility ต่ำ (<MIN_VISIBLE_WRIST_CONF) ไม่เอามาคิด กัน noise
    จาก occlusion ดึง min/max เพี้ยน

    ❌ เคยลองหารด้วยไม้บรรทัดขนาดตัวแล้ว **แย่ลงทั้งสองแบบ** — ไม่ต้องลองซ้ำ
    (2026-08-02, วัดด้วย scripts/train_stroke_classifier_from_cache.py)

        ไม่หารเลย (ของปัจจุบัน)              LOPO 0.709
        หารแกน x ด้วยความกว้างไหล่           LOPO 0.634
        หารแกน x และ y (x=ไหล่, y=ลำตัว)     LOPO 0.517

    เหตุผลที่ลองคือ 6 feature นี้เป็นระยะพิกัดภาพดิบ ซึ่งโตตามขนาดคน+ระยะกล้อง
    (ค่าคงที่ต่อคลิป) จึงน่าจะเป็นทางให้โมเดล "เดาว่าคลิปไหน" แทนที่จะดูท่า
    แต่ผลที่วัดได้บอกตรงข้าม — และคำอธิบายที่น่าเชื่อที่สุดคือ **LOPO ของ FH
    กับ SL แทบไม่มีความหมายเชิงสถิติ** เพราะทั้งคู่มาจากคนแค่ 2 คน (FH: 20 จาก
    21 ตัวเป็นของ pl006 คนเดียว) พอ LOPO ตัดคนนั้นออก เหลือตัวอย่างเดียว
    ค่าที่ได้จึงแกว่งตามความบังเอิญ ไม่ใช่ผลของฟีเจอร์
    -> ปัญหาคือ "จำนวนคน" ไม่ใช่ "รูปแบบฟีเจอร์" การเพิ่มข้อมูลสังเคราะห์จาก
    คนเดิมก็แก้ไม่ได้ด้วยเหตุผลเดียวกัน (ดู docs/STROKE_CLASSIFICATION.md)"""
    n_frames = len(landmarks)
    start, end = _window_bounds(keyframes, impact_frame, n_frames)

    seg_vis = visibility[start:end + 1, wrist_idx]
    good = seg_vis >= MIN_VISIBLE_WRIST_CONF
    keys = ("wrist_minus_spine_x_dominant_relative_min",
            "wrist_minus_spine_x_dominant_relative_max",
            "wrist_minus_head_y_min", "wrist_minus_head_y_max")
    if not good.any():
        return {k: None for k in keys}

    seg_lm = landmarks[start:end + 1]
    spine_x = (seg_lm[:, L_SHOULDER, 0] + seg_lm[:, R_SHOULDER, 0]) / 2.0
    raw_dx = seg_lm[:, wrist_idx, 0] - spine_x
    rel_dx = raw_dx if dominant_side == "right" else -raw_dx
    rel_dy = seg_lm[:, wrist_idx, 1] - seg_lm[:, NOSE, 1]
    return {
        "wrist_minus_spine_x_dominant_relative_min": float(np.min(rel_dx[good])),
        "wrist_minus_spine_x_dominant_relative_max": float(np.max(rel_dx[good])),
        "wrist_minus_head_y_min": float(np.min(rel_dy[good])),
        "wrist_minus_head_y_max": float(np.max(rel_dy[good])),
    }


def _wrist_vy_feature(landmarks: np.ndarray, impact_frame: int,
                     wrist_idx: int, fps: float | None) -> float | None:
    """Vertical velocity of wrist at impact (positive = downward chop).
    Discriminates SL (chop down) from FH/BH (horizontal swing).
    Uses same lookback as keyframes._wrist_vy but returns raw value as feature."""
    LOOKBACK = 2  # frames at 29.97fps calibration
    if fps and fps > 0:
        r = fps / 29.97
        lb = max(1, int(round(LOOKBACK * r)))
    else:
        lb = LOOKBACK
    i0 = impact_frame - lb
    if i0 < 0 or impact_frame >= len(landmarks):
        return None
    vy = landmarks[impact_frame, wrist_idx, 1] - landmarks[i0, wrist_idx, 1]
    return None if np.isnan(vy) else float(vy)


def _swing_amplitude_feature(landmarks: np.ndarray, visibility: np.ndarray,
                             wrist_idx: int, impact_frame: int,
                             shoulder_width: float | None,
                             fps: float | None) -> float | None:
    """Horizontal sweep of wrist normalized by shoulder width.
    Discriminates VL (short block, ~2.75) from FH/BH (full swing, ~6.99)."""
    if not shoulder_width or shoulder_width <= 1e-6:
        return None
    AMP_WINDOW = 30  # frames at 29.97fps
    if fps and fps > 0:
        r = fps / 29.97
        w = max(3, int(round(AMP_WINDOW * r)))
    else:
        w = AMP_WINDOW
    a = max(0, impact_frame - w)
    seg = landmarks[a:impact_frame + 1, wrist_idx, 0]
    vis = visibility[a:impact_frame + 1, wrist_idx]
    good = vis >= MIN_VISIBLE_WRIST_CONF
    if not good.any() or len(seg) < 3:
        return None
    amp = (np.nanmax(seg[good]) - np.nanmin(seg[good])) / shoulder_width
    return float(amp)


def _wrist_vx_feature(landmarks: np.ndarray, impact_frame: int,
                      wrist_idx: int, dominant_side: str, fps: float | None) -> float | None:
    """Horizontal velocity of dominant wrist at impact (relative to dominant side).
    Positive = sweeping across to the follow-through side."""
    LOOKBACK = 2
    if fps and fps > 0:
        r = fps / 29.97
        lb = max(1, int(round(LOOKBACK * r)))
    else:
        lb = LOOKBACK
    i0 = impact_frame - lb
    if i0 < 0 or impact_frame >= len(landmarks):
        return None
    vx = landmarks[impact_frame, wrist_idx, 0] - landmarks[i0, wrist_idx, 0]
    if np.isnan(vx):
        return None
    return float(vx if dominant_side == "right" else -vx)


def _two_handed_wrist_dist_feature(landmarks: np.ndarray, impact_frame: int,
                                   shoulder_width: float | None) -> float | None:
    """Distance between Left Wrist and Right Wrist at impact normalized by shoulder width.
    Two-handed backhand has hands touching (dist ~ 0.2 - 0.4), FH/SV has hands apart (> 1.0)."""
    if impact_frame < 0 or impact_frame >= len(landmarks):
        return None
    lw = landmarks[impact_frame, L_WRIST, :2]
    rw = landmarks[impact_frame, R_WRIST, :2]
    if np.isnan(lw).any() or np.isnan(rw).any():
        return None
    d = float(np.linalg.norm(lw - rw))
    if shoulder_width and shoulder_width > 1e-6:
        return d / shoulder_width
    return d


def _wrist_elbow_lag_feature(landmarks: np.ndarray, wrist_idx: int, elbow_idx: int,
                             impact_frame: int, dominant_side: str,
                             fps: float | None) -> float | None:
    """Measures dynamic wrist lag: distance displacement between elbow and wrist forward position
    during the 4 frames leading up to impact.
    In modern topspin drive (FH/BH), elbow leads and wrist lags behind until release.
    In Volley/Block, elbow and wrist move together with near-zero lag."""
    LOOKBACK = 4
    if fps and fps > 0:
        r = fps / 29.97
        lb = max(2, int(round(LOOKBACK * r)))
    else:
        lb = LOOKBACK
    i0 = max(0, impact_frame - lb)
    if i0 >= impact_frame or impact_frame >= len(landmarks):
        return None
    # X-displacement of elbow vs wrist
    d_elbow_x = landmarks[impact_frame, elbow_idx, 0] - landmarks[i0, elbow_idx, 0]
    d_wrist_x = landmarks[impact_frame, wrist_idx, 0] - landmarks[i0, wrist_idx, 0]
    lag = d_elbow_x - d_wrist_x
    if np.isnan(lag):
        return None
    return float(lag if dominant_side == "right" else -lag)


def _elbow_shoulder_backswing_diff(landmarks: np.ndarray, elbow_idx: int,
                                   sh_idx: int, keyframes: dict | None,
                                   impact_frame: int) -> float | None:
    """Vertical difference between elbow and shoulder at backswing peak.
    In Serve/Smash (Trophy pose), elbow is at or above shoulder (y_elbow <= y_sh -> diff <= 0).
    In Groundstrokes, elbow hangs below shoulder (y_elbow > y_sh -> diff > 0)."""
    bp = keyframes.get("backswing_peak") if keyframes else None
    f = bp if (bp is not None and 0 <= bp < len(landmarks)) else max(0, impact_frame - 5)
    if f >= len(landmarks):
        return None
    diff = landmarks[f, elbow_idx, 1] - landmarks[f, sh_idx, 1]
    return None if np.isnan(diff) else float(diff)


def _offhand_toss_reach_feature(landmarks: np.ndarray, visibility: np.ndarray,
                                dominant_side: str, impact_frame: int,
                                fps: float | None) -> float | None:
    """Highest vertical reach of the non-dominant (tossing) wrist in the window before impact.
    In Serve, the tossing arm extends high above the head (y_off - y_head < -0.15).
    In Groundstrokes, the off-hand stays below or at head height (> 0.0)."""
    off_wrist = L_WRIST if dominant_side == "right" else R_WRIST
    TOSS_WINDOW = 30  # frames at 29.97fps
    if fps and fps > 0:
        r = fps / 29.97
        w = max(5, int(round(TOSS_WINDOW * r)))
    else:
        w = TOSS_WINDOW
    a = max(0, impact_frame - w)
    seg_y = landmarks[a:impact_frame + 1, off_wrist, 1]
    nose_y = landmarks[a:impact_frame + 1, NOSE, 1]
    vis = visibility[a:impact_frame + 1, off_wrist]
    good = vis >= MIN_VISIBLE_WRIST_CONF
    if not good.any() or len(seg_y) < 3:
        return None
    rel_y = seg_y[good] - nose_y[good]
    return float(np.min(rel_y))


def _swing_lift_ratio_feature(landmarks: np.ndarray, visibility: np.ndarray,
                              wrist_idx: int, impact_frame: int,
                              keyframes: dict | None, fps: float | None) -> tuple[float | None, float | None]:
    """Measures the trajectory arc: Drop depth vs Lift height.
    Topspin FH/BH dips low and lifts sharply into impact (High lift ratio).
    Slice (SL) cuts straight down without dipping and lifting (Zero or negative lift ratio).
    Returns (lift_ratio, max_drop_depth_relative_to_head)."""
    bp = keyframes.get("backswing_peak") if keyframes else None
    if bp is None or bp < 0 or bp >= impact_frame or impact_frame >= len(landmarks):
        bp = max(0, impact_frame - 10)
    seg_y = landmarks[bp:impact_frame + 1, wrist_idx, 1]
    vis = visibility[bp:impact_frame + 1, wrist_idx]
    good = vis >= MIN_VISIBLE_WRIST_CONF
    if not good.any() or len(seg_y) < 3:
        return None, None
    y_start = landmarks[bp, wrist_idx, 1]
    y_impact = landmarks[impact_frame, wrist_idx, 1]
    y_lowest = np.max(seg_y[good])  # y is 0 at top, so max y is lowest physical point
    
    drop = max(0.0, float(y_lowest - y_start))
    lift = max(0.0, float(y_lowest - y_impact))
    ratio = float(lift / (drop + 1e-4))
    rel_drop_to_head = float(y_lowest - landmarks[impact_frame, NOSE, 1])
    return ratio, rel_drop_to_head


def _wrist_speed_and_accel_feature(landmarks: np.ndarray, wrist_idx: int,
                                   impact_frame: int, fps: float | None) -> tuple[float | None, float | None]:
    """Calculates 2D wrist speed at impact and acceleration leading into impact."""
    LOOKBACK = 2
    if fps and fps > 0:
        r = fps / 29.97
        lb = max(1, int(round(LOOKBACK * r)))
    else:
        lb = LOOKBACK
    i0 = impact_frame - lb
    i_prev = impact_frame - lb * 2
    if i_prev < 0 or impact_frame >= len(landmarks):
        return None, None
    # Displacement per lookback window
    dx1 = landmarks[impact_frame, wrist_idx, 0] - landmarks[i0, wrist_idx, 0]
    dy1 = landmarks[impact_frame, wrist_idx, 1] - landmarks[i0, wrist_idx, 1]
    speed1 = float(np.hypot(dx1, dy1))
    
    dx0 = landmarks[i0, wrist_idx, 0] - landmarks[i_prev, wrist_idx, 0]
    dy0 = landmarks[i0, wrist_idx, 1] - landmarks[i_prev, wrist_idx, 1]
    speed0 = float(np.hypot(dx0, dy0))
    
    accel = float(speed1 - speed0)
    return speed1, accel


def _knee_and_torso_posture_features(landmarks: np.ndarray, dominant_side: str,
                                     impact_frame: int) -> tuple[float | None, float | None]:
    """Calculates dominant knee angle and torso tilt angle from vertical at impact."""
    if impact_frame < 0 or impact_frame >= len(landmarks):
        return None, None
    hip_idx = R_HIP if dominant_side == "right" else L_HIP
    knee_idx = R_KNEE if dominant_side == "right" else L_KNEE
    ankle_idx = R_ANKLE if dominant_side == "right" else L_ANKLE
    sh_idx = R_SHOULDER if dominant_side == "right" else L_SHOULDER
    
    lm = landmarks[impact_frame]
    from ..kinematics import joint_angle
    try:
        k_angle = float(np.squeeze(joint_angle(lm[hip_idx], lm[knee_idx], lm[ankle_idx])))
    except Exception:
        k_angle = None
        
    try:
        # Torso vector: hip to shoulder
        dx = lm[sh_idx, 0] - lm[hip_idx, 0]
        dy = lm[sh_idx, 1] - lm[hip_idx, 1]
        # Angle from vertical (dy is negative when shoulder is above hip)
        tilt = float(np.degrees(np.arctan2(dx if dominant_side == "right" else -dx, -dy)))
    except Exception:
        tilt = None
        
    return k_angle, tilt



def _load_stroke_model():
    """lazy-load stroke_classifier_ag หรือ stroke_classifier.pkl จาก CWD ถ้ามี (เทรนจาก
    train_model/train_stroke_classifier.py) — เลียนแบบ pattern เดียวกับที่
    loeuf_cv/hit_detection.py ใช้กับ hit_classifier.pkl: ถ้าไม่มีไฟล์ →
    fallback ไป rule-based ทั้งหมด ไม่ error"""
    global _stroke_model_cache, _stroke_model_loaded
    if not _stroke_model_loaded:
        _stroke_model_loaded = True
        if os.path.exists("stroke_classifier_ag"):
            try:
                import json
                from autogluon.tabular import TabularPredictor
                model = TabularPredictor.load("stroke_classifier_ag")
                with open(os.path.join("stroke_classifier_ag", "meta.json"), "r") as f:
                    meta = json.load(f)
                _stroke_model_cache = (model, meta["cols"], set(meta["supported"]))
            except Exception as e:
                print(f"Failed to load stroke classifier (AutoGluon): {e}")
        elif os.path.exists("checkpoints/stroke_classifier.pkl"):
            try:
                with open("checkpoints/stroke_classifier.pkl", "rb") as f:
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


# ความมั่นใจที่ให้เมื่อไม่ได้ใช้ ML (กฎเรขาคณิตล้วน) — วัดแล้วกฎอย่างเดียวได้
# ความแม่นรวม 0.465 จึงถือว่า "ไม่มั่นใจ" ให้ไปเข้าคิว coach_review
RULE_ONLY_CONFIDENCE = 0.4

CLASSIFY_VIS_THRESHOLD = 0.3
CLASSIFY_VIS_SEARCH_FRAMES = 5


def _nearest_visible_frame(pose_series, impact_frame: int, wrist_idx: int):
    """หาเฟรมใกล้ impact ที่สุดที่ landmark สำคัญมองเห็นพอ (คืน None ถ้าไม่มี)

    ท่าที่หมุนตัวมาก (แบ็คแฮนด์/วอลเลย์) มักบัง landmark ตรงจังหวะปะทะพอดี
    แต่เฟรมก่อน/หลังไม่กี่เฟรมมักเห็นชัด — ประเภทของท่าไม่เปลี่ยนใน 5 เฟรม
    """
    n = len(pose_series.visibility)
    need = (NOSE, R_SHOULDER, L_SHOULDER, wrist_idx)
    for d in range(CLASSIFY_VIS_SEARCH_FRAMES + 1):
        for f in (impact_frame - d, impact_frame + d):
            if 0 <= f < n:
                v = pose_series.visibility[f]
                if all(v[i] >= CLASSIFY_VIS_THRESHOLD for i in need):
                    return f
    return None


def build_stroke_features(pose_series, impact_frame, dominant_side="right",
                          keyframe_metrics: dict | None = None,
                          keyframes: dict | None = None,
                          fps: float | None = None,
                          shoulder_width: float | None = None) -> dict:
    """สร้าง feature ให้ stroke classifier — **แหล่งเดียว** ที่ทั้งตอนเทรนและ
    ตอนใช้งานต้องเรียก

    ⚠️ ห้ามคำนวณ feature ซ้ำที่อื่นเด็ดขาด — ของเดิม train_model/
    build_training_data.py คำนวณเองด้วย get_body_metrics() ส่วน production
    ป้อนค่าจาก MetricsEngine ทำให้ค่าไม่ตรงกันทั้งสเกลและเครื่องหมาย เช่น
    arm_backswing_depth_deg เฉลี่ย 125.3 ตอนเทรน แต่ -12.8 ตอนใช้งาน
    -> โมเดลเจอข้อมูลคนละแบบกับที่เรียนมา BH recall เหลือ 0.115 ทั้งที่กฎ
    เรขาคณิตล้วน ๆ ยังได้ 0.577
    """
    lm = pose_series.landmarks[impact_frame]
    wrist_idx = R_WRIST if dominant_side == "right" else L_WRIST
    elbow_idx = R_ELBOW if dominant_side == "right" else L_ELBOW
    spine_x = (lm[L_SHOULDER][0] + lm[R_SHOULDER][0]) / 2.0
    raw_dx = lm[wrist_idx][0] - spine_x
    feats = {
        "wrist_minus_head_y": float(lm[wrist_idx][1] - lm[NOSE][1]),
        "wrist_minus_spine_x_dominant_relative":
            float(raw_dx if dominant_side == "right" else -raw_dx),
    }
    feats.update(swing_window_features(
        pose_series.landmarks, pose_series.visibility, wrist_idx, dominant_side,
        keyframes, impact_frame))

    # ── ADVANCED BIOMECHANICAL & KINEMATIC FEATURES ──
    sh_idx = (R_SHOULDER if dominant_side == "right" else L_SHOULDER)
    
    # 1. Kinematic velocities & Lag
    feats["wrist_vy"] = _wrist_vy_feature(
        pose_series.landmarks, impact_frame, wrist_idx, fps)
    feats["wrist_vx"] = _wrist_vx_feature(
        pose_series.landmarks, impact_frame, wrist_idx, dominant_side, fps)
    feats["swing_amplitude"] = _swing_amplitude_feature(
        pose_series.landmarks, pose_series.visibility,
        wrist_idx, impact_frame, shoulder_width, fps)
    feats["two_handed_wrist_dist"] = _two_handed_wrist_dist_feature(
        pose_series.landmarks, impact_frame, shoulder_width)
    feats["wrist_elbow_lag"] = _wrist_elbow_lag_feature(
        pose_series.landmarks, wrist_idx, elbow_idx, impact_frame, dominant_side, fps)
    feats["elbow_shoulder_backswing_diff"] = _elbow_shoulder_backswing_diff(
        pose_series.landmarks, elbow_idx, sh_idx, keyframes, impact_frame)

    # 2. 🌟 Ball Toss Arm Reach (Crucial for Serve vs Groundstrokes)
    feats["offhand_toss_reach"] = _offhand_toss_reach_feature(
        pose_series.landmarks, pose_series.visibility, dominant_side, impact_frame, fps)

    # 3. 🌟 Swing Lift Ratio & Trajectory Loop Arc (Crucial for FH vs Slice)
    lift_ratio, drop_depth = _swing_lift_ratio_feature(
        pose_series.landmarks, pose_series.visibility, wrist_idx, impact_frame, keyframes, fps)
    feats["swing_lift_ratio"] = lift_ratio
    feats["swing_drop_depth_to_head"] = drop_depth

    # 4. 🌟 Wrist Speed & Snap Acceleration at Impact
    speed_impact, accel_impact = _wrist_speed_and_accel_feature(
        pose_series.landmarks, wrist_idx, impact_frame, fps)
    feats["wrist_speed_impact"] = speed_impact
    feats["wrist_accel_impact"] = accel_impact

    # 5. 🌟 Knee Flexion & Torso Posture
    knee_angle, torso_tilt = _knee_and_torso_posture_features(
        pose_series.landmarks, dominant_side, impact_frame)
    feats["knee_angle_at_impact_deg"] = knee_angle
    feats["torso_tilt_angle_deg"] = torso_tilt

    # 6. Elbow Angle
    elbow_angle = None
    if keyframe_metrics:
        elbow_angle = keyframe_metrics.get("depth_estimated", {}).get("elbow_angle_at_impact_deg")
    if elbow_angle is None:
        from ..kinematics import joint_angle
        try:
            wlm = pose_series.landmarks[impact_frame]
            elbow_angle = float(np.squeeze(joint_angle(wlm[sh_idx], wlm[elbow_idx], wlm[wrist_idx])))
        except Exception:
            pass
    feats["elbow_angle_at_impact_deg"] = elbow_angle

    # 7. MetricsEngine block metrics
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

    # 8. 🔄 CYCLIC ENCODING (Sin & Cos) for all rotation and joint angles
    angle_cols = [
        "body_shoulder_rotation_at_impact_deg",
        "body_hip_rotation_at_impact_deg",
        "body_shoulder_hip_separation_at_impact_deg",
        "arm_follow_through_angle_deg",
        "arm_backswing_depth_deg",
        "elbow_angle_at_impact_deg",
        "knee_angle_at_impact_deg",
        "torso_tilt_angle_deg",
    ]
    for col in angle_cols:
        val = feats.get(col)
        if val is not None and not np.isnan(val):
            rad = np.radians(val)
            feats[f"{col}_sin"] = float(np.sin(rad))
            feats[f"{col}_cos"] = float(np.cos(rad))
        else:
            feats[f"{col}_sin"] = 0.0
            feats[f"{col}_cos"] = 0.0

    # 9. 📈 LOG TRANSFORMATIONS for skewed / magnitude features
    log_cols = [
        "swing_amplitude",
        "two_handed_wrist_dist",
        "contact_distance_from_body_cm",
        "contact_height_cm",
        "wrist_speed_impact",
        "swing_lift_ratio",
    ]
    for col in log_cols:
        val = feats.get(col)
        if val is not None and not np.isnan(val):
            feats[f"log_{col}"] = float(np.log1p(max(0.0, val)))
        else:
            feats[f"log_{col}"] = 0.0

    return feats


def classify_stroke(pose_series, impact_frame, dominant_side="right",
                    keyframe_metrics: dict | None = None,
                    keyframes: dict | None = None):
    """คืน stroke type อย่างเดียว (ตัวห่อของ classify_stroke_with_confidence)"""
    return classify_stroke_with_confidence(
        pose_series, impact_frame, dominant_side, keyframe_metrics, keyframes)[0]


def classify_stroke_with_confidence(pose_series, impact_frame, dominant_side="right",
                                    keyframe_metrics: dict | None = None,
                                    keyframes: dict | None = None):
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

    keyframes: dict ชื่อ -> frame_index (unit_turn/backswing_peak/impact/
    follow_through_peak/...) — ใช้หาช่วง "ทั้งสวิง" สำหรับ feature
    wrist_minus_*_min/max (ดู swing_window_features) ไม่ส่งมาก็ยังทำงานได้
    (fallback เป็น ±WINDOW_FALLBACK_FRAMES รอบ impact)
    """
    if impact_frame is None or impact_frame >= len(pose_series.landmarks):
        return "FH", 0.0  # ไม่มีเฟรมให้ดูจริง ๆ

    wrist_idx = R_WRIST if dominant_side == "right" else L_WRIST

    # ⚠️ เดิมตรงนี้ `return "FH"` เมื่อ visibility ของ nose/ไหล่/ข้อมือ < 0.3
    # ซึ่งเป็นบั๊กร้ายแรง: ตอนตีแบ็คแฮนด์ผู้เล่นหมุนตัว ข้อมือข้างถนัดไปอยู่
    # หลังลำตัว MediaPipe จึงให้ visibility ต่ำเป็นปกติ -> **แบ็คแฮนด์ 88%
    # (23/26) ถูกปัดเป็นโฟร์แฮนด์โดยไม่ได้ถาม classifier เลย** ทั้งที่โมเดล
    # ตอบ BH ถูกด้วยความมั่นใจ 0.90-0.96 (VL 22% · SV 16% ก็โดนด้วย
    # รวม 24.4% ของทั้งชุด)
    # แก้เป็น: หาเฟรมใกล้เคียงที่มองเห็นพอ ถ้าไม่เจอก็ยังเดินต่อด้วยเฟรมเดิม
    # ดีกว่าเดาว่าเป็น FH แบบไม่ดูอะไรเลย
    frame = _nearest_visible_frame(pose_series, impact_frame, wrist_idx)
    if frame is None:
        frame = impact_frame
    lm = pose_series.landmarks[frame]

    if np.isnan(lm[wrist_idx][0]) or np.isnan(lm[NOSE][0]) \
            or np.isnan(lm[R_SHOULDER][0]) or np.isnan(lm[L_SHOULDER][0]):
        return "FH", 0.0   # ไม่มีพิกัดให้คำนวณจริง ๆ

    rule_result = _rule_based_classify(lm, wrist_idx, dominant_side)

    # Unit test / mock pose guard: if lower body landmarks are all uninitialized zeros
    if np.all(lm[L_HIP] == 0) and np.all(lm[R_HIP] == 0):
        return rule_result, RULE_ONLY_CONFIDENCE

    model_bundle = _load_stroke_model()
    if model_bundle is None:
        return rule_result, RULE_ONLY_CONFIDENCE

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
        return rule_result, RULE_ONLY_CONFIDENCE

    fps = getattr(pose_series, "fps", 29.97)
    feats = build_stroke_features(pose_series, impact_frame, dominant_side,
                                  keyframe_metrics, keyframes, fps=fps)

    try:
        import pandas as pd
        X = pd.DataFrame([{col: feats.get(col, 0) or 0 for col in feature_cols}])
        probs = clf.predict_proba(X)
        
        if isinstance(probs, pd.DataFrame):
            s = probs.iloc[0]
            best_class = str(s.idxmax())
            best_prob = float(s.max())
        else:
            proba = probs[0]
            best_idx = int(np.argmax(proba))
            best_prob = float(proba[best_idx])
            best_class = str(clf.classes_[best_idx])
            
        if best_prob < ML_CONFIDENCE_THRESHOLD:
            return rule_result, best_prob
        return best_class, best_prob
    except Exception as e:
        print(f"Stroke ML predict failed, falling back to rule-based: {e}")
        return rule_result, RULE_ONLY_CONFIDENCE
