"""024-A · stroke_root + 042-CF · confidence_flag

A7 detection_status / A8 is_clean_stroke / A9 priority logic
และ CF1-CF3 — implement ตาม logic ที่ schema doc เขียนไว้ตายตัว
"""

import numpy as np

from .config import CORE_LANDMARKS, PipelineConfig
from .keyframes import Keyframe
from .pose_extractor import PoseTimeseries

TRACKING_KEYFRAMES = ("unit_turn", "backswing_peak", "impact",
                      "follow_through_peak")  # B1–B4 (B5 ไม่นับ — เจตนา player ได้)


def pose_estimation_confidence(ts: PoseTimeseries,
                               keyframes: dict[str, Keyframe]) -> float:
    """CF1: calibrate จาก visibility ของ landmark หลักช่วง stroke window"""
    imp = keyframes.get("impact")
    if imp and imp.detected:
        lo = keyframes.get("unit_turn")
        hi = keyframes.get("follow_through_peak")
        start = lo.frame_index if (lo and lo.detected) else 0
        end = (hi.frame_index + 1) if (hi and hi.detected) else ts.n_frames
    else:
        start, end = 0, ts.n_frames
    vis = ts.visibility[start:end][:, list(CORE_LANDMARKS)]
    return round(float(np.mean(vis)), 2) if vis.size else 0.0


def inference_clean(ts: PoseTimeseries, config: PipelineConfig) -> bool:
    """CF2: skeleton ขยับต่อเนื่อง ไม่มี jump ผิดธรรมชาติ"""
    from .kinematics import body_height_norm
    height = body_height_norm(ts)
    core = ts.landmarks[:, list(CORE_LANDMARKS), :2]
    if np.isnan(core).all():
        return False
    disp = np.linalg.norm(np.diff(np.nan_to_num(core, nan=0.0), axis=0), axis=2)
    max_jump = float(np.nanmax(disp)) if disp.size else 0.0
    return bool(max_jump < config.inference_jump_ratio * max(height, 1e-6) * 3)


def issues_during_inference(ts: PoseTimeseries) -> list[str]:
    """CF3: ปัญหาที่เจอระหว่าง inference (≠ VQ2 ที่เห็นก่อนรัน)"""
    issues = []
    core_vis = ts.visibility[:, list(CORE_LANDMARKS)]
    frac_occluded = float(np.mean(core_vis.mean(axis=1) < 0.5))
    if frac_occluded > 0.2:
        issues.append("partial_occlusion")
    return issues


def build_confidence_flag(ts: PoseTimeseries, keyframes: dict[str, Keyframe],
                          config: PipelineConfig) -> dict:
    return {
        "pose_estimation_confidence": pose_estimation_confidence(ts, keyframes),
        "inference_clean": inference_clean(ts, config),
        "issues_detected": issues_during_inference(ts),
    }


def build_stroke_root(session_id: str, stroke_index: int, stroke_type: str,
                      start_frame: int, end_frame: int,
                      keyframes: dict[str, Keyframe], video_quality: dict,
                      confidence_flag: dict, config: PipelineConfig) -> dict:
    detected = {name: (k.detected) for name, k in keyframes.items()}

    # A7: valid = ครบทุก keyframe / failed = B1-B4 ไม่ได้เลย / partial = บางส่วน
    if all(detected.values()):
        status = "valid"
    elif not any(detected[k] for k in TRACKING_KEYFRAMES):
        status = "failed"
    else:
        status = "partial"

    # A8: B1–B4 ครบ + VQ ไม่มี issue
    is_clean = bool(all(detected[k] for k in TRACKING_KEYFRAMES)
                    and not video_quality.get("issues"))

    # A9 priority logic (ตาม doc เป๊ะ ๆ 6 ขั้น)
    cf1 = confidence_flag["pose_estimation_confidence"]
    if status == "failed":
        action = "discard"
    elif cf1 < config.coach_review_threshold:
        action = "discard"
    elif status == "partial":
        action = "coach_review"
    elif not is_clean:
        action = "coach_review"
    elif cf1 < config.auto_accept_threshold:
        action = "coach_review"
    else:
        action = "auto_accept"

    return {
        "stroke_id": f"{session_id}_stroke-{stroke_index:02d}",
        "stroke_index": stroke_index,
        "stroke_type": stroke_type,
        "stroke_start_frame": start_frame,
        "stroke_end_frame": end_frame,
        "dominant_side": config.dominant_side,
        "detection_status": status,
        "is_clean_stroke": is_clean,
        "stroke_recommended_action": action,
    }
