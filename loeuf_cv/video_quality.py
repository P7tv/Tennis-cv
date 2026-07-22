"""022-VQ · video_quality — ประเมิน clip quality ก่อน process

Pre-check จาก frame stats ที่เก็บระหว่าง pose extraction
(ไม่มี stats เช่น synthetic → usable=true, issues=[])

Thresholds เป็น heuristic เริ่มต้น — จูนกับคลิปจริงใน Phase 0
"""

import warnings

import numpy as np

from .config import UPPER_BODY, PipelineConfig
from .kinematics import motion_energy
from .pose_extractor import PoseTimeseries

LUMA_DARK = 45.0            # mean luma ต่ำกว่านี้ = low_light
BLUR_VAR_MIN = 40.0         # Laplacian variance ต่ำกว่านี้ = motion_blur
PERSON_MIN_FRAC = 0.18      # bbox สูง < 18% ของเฟรม = player_too_far
OCCLUSION_VIS = 0.5         # mean visibility ต่ำกว่านี้ = partial_occlusion
EDGE_ENERGY_RATIO = 0.5     # motion ที่ขอบคลิป > 50% ของ peak = partial_swing
EDGE_MARGIN = 0.04          # landmark ใกล้ขอบเฟรมกว่านี้ = ชิดขอบ
EDGE_FRACTION = 0.25        # ชิดขอบเกิน 25% ของคลิป → player_near_edge


def assess_video_quality(ts: PoseTimeseries, config: PipelineConfig) -> dict:
    issues: list[str] = []
    stats = ts.frame_stats

    if stats.mean_luma is not None and np.median(stats.mean_luma) < LUMA_DARK:
        issues.append("low_light")
    if stats.blur_score is not None and np.median(stats.blur_score) < BLUR_VAR_MIN:
        issues.append("motion_blur")
    if (stats.person_height_frac is not None
            and np.median(stats.person_height_frac) < PERSON_MIN_FRAC):
        issues.append("player_too_far")

    core_vis = float(np.mean(ts.visibility[:, list(UPPER_BODY)]))
    if core_vis < OCCLUSION_VIS:
        issues.append("partial_occlusion")

    # ผู้เล่นชิดขอบเฟรมนาน → เสี่ยง lens distortion + ตัวหลุดเฟรม
    # (⚠️ "player_near_edge" = enum เสนอเพิ่มนอก VQ2 — รอ confirm, ดู D0.2)
    valid_x = ts.landmarks[:, list(UPPER_BODY), 0]
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        # แถวที่ UPPER_BODY landmark ทั้งหมด NaN (occlusion ยาวเกิน gap-fill limit)
        # ทำ nanmin/nanmax วอร์น "All-NaN slice" ได้ — errstate ไม่กันวอร์นนี้เพราะ
        # มันมาจาก warnings module ไม่ใช่ floating point state
        warnings.simplefilter("ignore", RuntimeWarning)
        near_edge = np.nanmin(valid_x, axis=1) < EDGE_MARGIN
        near_edge |= np.nanmax(valid_x, axis=1) > 1.0 - EDGE_MARGIN
    if np.nanmean(near_edge.astype(float)) > EDGE_FRACTION:
        issues.append("player_near_edge")

    # partial_swing: motion ยังสูงอยู่ที่ขอบคลิป = โดนตัดกลาง swing
    energy = motion_energy(ts, UPPER_BODY)
    peak = float(np.nanmax(energy)) if len(energy) else 0.0
    detail = None
    if peak > 0:
        edge = max(2, int(ts.n_frames * 0.08))
        start_hot = float(np.mean(energy[:edge])) > EDGE_ENERGY_RATIO * peak
        end_hot = float(np.mean(energy[-edge:])) > EDGE_ENERGY_RATIO * peak
        if start_hot and end_hot:
            issues.append("partial_swing")
            detail = "multiple"
        elif start_hot:
            issues.append("partial_swing")
            detail = "backswing_cut"
        elif end_hot:
            issues.append("partial_swing")
            detail = "follow_through_cut"

    # VQ1 usable: มีคนให้ detect + แสงพอ + ไม่เบลอเกิน + clip ไม่สั้นเกิน
    person_detected = float(np.mean(ts.visibility.max(axis=1) > 0.3)) > 0.5
    long_enough = ts.n_frames >= int(0.5 * ts.fps)
    usable = bool(person_detected and long_enough
                  and "low_light" not in issues)

    return {
        "usable": usable,
        "issues": issues,
        "partial_swing_detail": detail,
    }
