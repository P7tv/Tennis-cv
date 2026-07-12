"""Signal helpers: ความเร็ว, มุมข้อต่อ, rotation องศาเทียบ baseline, CoM

CoM ใช้สูตร segment weighting ตาม schema doc (031-C, "Reference for
CoM calculation") — น้ำหนักรวม 1.00 พอดี
"""

import numpy as np

from .config import (
    COM_SEGMENT_WEIGHTS, L_ANKLE, L_ELBOW, L_HIP, L_KNEE, L_SHOULDER, L_WRIST,
    NOSE, R_ANKLE, R_ELBOW, R_HIP, R_KNEE, R_SHOULDER, R_WRIST,
)
from .pose_extractor import PoseTimeseries


def dominant(handedness: str) -> dict:
    if handedness == "left":
        return {"wrist": L_WRIST, "elbow": L_ELBOW, "shoulder": L_SHOULDER,
                "hip": L_HIP, "ear": 7, "index": 19, "sign": -1.0}
    return {"wrist": R_WRIST, "elbow": R_ELBOW, "shoulder": R_SHOULDER,
            "hip": R_HIP, "ear": 8, "index": 20, "sign": 1.0}


def speed(points: np.ndarray, timestamps_ms: np.ndarray) -> np.ndarray:
    """ความเร็ว (units/s) ของ point timeseries (T, D)"""
    dt = np.gradient(timestamps_ms) / 1000.0
    vel = np.gradient(points, axis=0) / dt[:, None]
    return np.linalg.norm(vel, axis=1)


def joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """มุมที่ข้อ b (องศา) จากจุด a-b-c ต่อเฟรม (T, 3) หรือ (3,)"""
    a, b, c = np.atleast_2d(a), np.atleast_2d(b), np.atleast_2d(c)
    v1, v2 = a - b, c - b
    cos = np.sum(v1 * v2, axis=1) / (
        np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def rotation_vs_baseline_deg(ts: PoseTimeseries, left_id: int,
                             right_id: int) -> np.ndarray:
    """มุมหมุนของเส้น (shoulder/hip line) เทียบ baseline axis ต่อเฟรม

    กล้องอยู่หลัง baseline → baseline ขนานแกน x ของภาพ การหมุนใน
    ระนาบพื้น (transverse) อ่านจาก world x-z:
        angle = atan2(Δz, Δx)  → 0° = ขนาน baseline

    ⚠️ Sign convention (+ = หันออกจาก net) ต้อง validate กับคลิป
    labeled จริงใน Phase 0 — depth (z) จาก monocular มี noise
    → field เหล่านี้ camera_dep ตามตาราง แต่ magnitude ใช้ได้
    """
    wlm = ts.world_landmarks
    d = wlm[:, right_id] - wlm[:, left_id]
    return np.degrees(np.arctan2(d[:, 2], np.abs(d[:, 0]) + 1e-9))


def com_xy(ts: PoseTimeseries, frame: int | None = None) -> np.ndarray:
    """CoM (x, y) normalized image coords — สูตรถ่วงน้ำหนัก segment
    ตาม schema doc. คืน (T, 2) หรือ (2,) ถ้าระบุ frame"""
    lm = ts.landmarks[:, :, :2]
    w = COM_SEGMENT_WEIGHTS

    def mid(a, b):
        return (lm[:, a] + lm[:, b]) / 2.0

    shoulder_mid = mid(L_SHOULDER, R_SHOULDER)
    hip_mid = mid(L_HIP, R_HIP)
    trunk = (shoulder_mid + hip_mid) / 2.0

    com = (
        trunk * w["trunk"]
        + mid(L_SHOULDER, L_ELBOW) * w["upper_arm"]
        + mid(R_SHOULDER, R_ELBOW) * w["upper_arm"]
        + mid(L_ELBOW, L_WRIST) * w["forearm"]
        + mid(R_ELBOW, R_WRIST) * w["forearm"]
        + mid(L_HIP, L_KNEE) * w["thigh"]
        + mid(R_HIP, R_KNEE) * w["thigh"]
        + mid(L_KNEE, L_ANKLE) * w["shank"]
        + mid(R_KNEE, R_ANKLE) * w["shank"]
    )
    return com[frame] if frame is not None else com


def hip_center(ts: PoseTimeseries) -> np.ndarray:
    return (ts.landmarks[:, L_HIP, :2] + ts.landmarks[:, R_HIP, :2]) / 2.0


def torso_center(ts: PoseTimeseries) -> np.ndarray:
    """(shoulder_mid + hip_mid) / 2 ตามนิยาม C17"""
    shoulder_mid = (ts.landmarks[:, L_SHOULDER, :2]
                    + ts.landmarks[:, R_SHOULDER, :2]) / 2.0
    hip_mid = hip_center(ts)
    return (shoulder_mid + hip_mid) / 2.0


def torso_rotation_proxy(ts: PoseTimeseries) -> np.ndarray:
    """Rotation proxy 0-1 จาก projected shoulder width — ใช้ขับ
    keyframe detection (ทนกว่า z-based สำหรับหา onset)"""
    lm = ts.landmarks
    shoulder_w = np.abs(lm[:, R_SHOULDER, 0] - lm[:, L_SHOULDER, 0])
    hip_w = np.abs(lm[:, R_HIP, 0] - lm[:, L_HIP, 0]) + 1e-9
    ratio = shoulder_w / hip_w
    baseline = np.nanpercentile(ratio, 90)
    return np.clip(1.0 - ratio / (baseline + 1e-9), 0.0, 1.0)


def body_height_norm(ts: PoseTimeseries) -> float:
    """ankle→shoulder (norm) — ใช้ normalize ภายใน keyframe logic"""
    lm = ts.landmarks
    ankle_y = np.nanmedian((lm[:, L_ANKLE, 1] + lm[:, R_ANKLE, 1]) / 2.0)
    shoulder_y = np.nanmedian((lm[:, L_SHOULDER, 1] + lm[:, R_SHOULDER, 1]) / 2.0)
    return float(abs(ankle_y - shoulder_y)) + 1e-9


def ankle_to_head_norm(ts: PoseTimeseries) -> float:
    """ankle→head (norm) — คู่กับ body_height_cm (C1) เพื่อทำ px→cm scale"""
    lm = ts.landmarks
    ankle_y = np.nanmedian((lm[:, L_ANKLE, 1] + lm[:, R_ANKLE, 1]) / 2.0)
    head_y = np.nanmedian(lm[:, NOSE, 1])
    return float(abs(ankle_y - head_y)) + 1e-9


def motion_energy(ts: PoseTimeseries, landmark_ids: tuple) -> np.ndarray:
    total = np.zeros(ts.n_frames)
    for lm_id in landmark_ids:
        pts = ts.landmarks[:, lm_id, :2]
        s = speed(np.nan_to_num(pts, nan=0.0), ts.timestamps_ms)
        total += np.nan_to_num(s, nan=0.0)
    return total
