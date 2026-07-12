"""Camera normalization — ระบบ calibrate ต่อคลิป (per-clip, self-contained)

รองรับ "คลิปมุมไม่เท่ากัน" โดยไม่ต้องรู้ค่ากล้องใด ๆ:
1. Roll correction   — กล้องเอียง: estimate จากแกนลำตัว + เส้นข้อเท้า
                       ช่วง ready แล้วหมุน landmark กลับ
2. Per-frame scale   — ผู้เล่นวิ่งเข้า/ออกจากกล้อง: ใช้ trunk length
                       (ไม่ยืดหดตามท่า) เป็น anchor ต่อเฟรม
3. Rotation zero-ref — กล้องเยื้องกลาง (yaw): ตั้ง 0° ของ rotation
                       จาก shoulder/hip line ณ ท่า ready ของคลิปนั้น
4. Deviation envelope — ประเมิน "กล้องเบี่ยงจากหลังตรงกี่องศา" →
                       ok / degraded / reject → VQ flag + confidence cap

สมมติฐานที่ลูกค้าต้อง sign-off ใน D0.2: ท่า ready ผู้เล่นยืนตรง
ไหล่ขนาน baseline (Brief Section 9: "Tenn proposes, Loeuf confirms")
"""

from dataclasses import dataclass

import numpy as np

from .config import (
    L_ANKLE, L_HIP, L_SHOULDER, R_ANKLE, R_HIP, R_SHOULDER, UPPER_BODY,
    PipelineConfig,
)
from .kinematics import motion_energy, rotation_vs_baseline_deg
from .pose_extractor import PoseTimeseries

ROLL_FLAG_DEG = 8.0          # เกินนี้ → VQ flag camera_tilted (TBC Phase 0)
DEVIATION_DEGRADED_DEG = 15.0  # envelope ระดับ 2 (TBC Phase 0)
DEVIATION_REJECT_DEG = 30.0    # envelope ระดับ 3 → side_view (TBC Phase 0)


@dataclass
class Calibration:
    roll_deg: float                    # กล้องเอียง (ประเมินจาก ready)
    camera_deviation_deg: float        # กล้องเบี่ยงจากหลังตรง (yaw proxy)
    shoulder_zero_deg: float           # rotation reference ของ shoulder line
    hip_zero_deg: float                # rotation reference ของ hip line
    scale_per_frame: np.ndarray        # cm ต่อ normalized unit ต่อเฟรม
    ready_frames: np.ndarray           # เฟรมที่ใช้เป็น reference
    tier: str                          # "ok" / "degraded" / "reject"

    def to_debug_dict(self) -> dict:
        return {
            "roll_deg": round(self.roll_deg, 2),
            "camera_deviation_deg": round(self.camera_deviation_deg, 2),
            "tier": self.tier,
        }


def _ready_frames(ts: PoseTimeseries) -> np.ndarray:
    """เฟรมช่วง ready = motion ต่ำในครึ่งแรกของคลิป (ก่อนเริ่ม swing)"""
    energy = motion_energy(ts, UPPER_BODY)
    half = max(3, ts.n_frames // 2)
    seg = energy[:half]
    threshold = np.nanpercentile(seg, 30)
    idx = np.where(seg <= threshold)[0]
    return idx if len(idx) >= 3 else np.arange(min(5, ts.n_frames))


def _px(ts: PoseTimeseries, lm_id: int) -> np.ndarray:
    """normalized → aspect-corrected coords (มุมองศาถูกต้องตามภาพจริง)"""
    aspect = ts.meta.width / max(ts.meta.height, 1)
    pts = ts.landmarks[:, lm_id, :2].copy()
    pts[:, 0] *= aspect
    return pts


def estimate_calibration(ts: PoseTimeseries, config: PipelineConfig) -> Calibration:
    ready = _ready_frames(ts)

    # --- roll: เฉลี่ยจาก (ก) เส้นข้อเท้าควรนอน (ข) แกนลำตัวควรตั้ง ---
    ankle_l, ankle_r = _px(ts, L_ANKLE)[ready], _px(ts, R_ANKLE)[ready]
    d_ankle = ankle_r - ankle_l
    ankle_roll = np.degrees(np.arctan2(d_ankle[:, 1], d_ankle[:, 0] + 1e-9))

    sh_mid = (_px(ts, L_SHOULDER) + _px(ts, R_SHOULDER))[ready] / 2.0
    hip_mid = (_px(ts, L_HIP) + _px(ts, R_HIP))[ready] / 2.0
    axis = sh_mid - hip_mid  # ชี้ขึ้น (y ลบ)
    body_roll = np.degrees(np.arctan2(axis[:, 0], -axis[:, 1] + 1e-9))

    roll = float(np.nanmedian(np.concatenate([ankle_roll, body_roll])))

    # --- camera deviation (yaw proxy): world-z asymmetry ของไหล่ตอน ready ---
    shoulder_rot = rotation_vs_baseline_deg(ts, L_SHOULDER, R_SHOULDER)
    hip_rot = rotation_vs_baseline_deg(ts, L_HIP, R_HIP)
    deviation = float(abs(np.nanmedian(shoulder_rot[ready])))

    # --- rotation zero-reference ---
    shoulder_zero = float(np.nanmedian(shoulder_rot[ready]))
    hip_zero = float(np.nanmedian(hip_rot[ready]))

    # --- per-frame scale จาก trunk length ---
    trunk = np.linalg.norm(
        (_px(ts, L_SHOULDER) + _px(ts, R_SHOULDER)) / 2.0
        - (_px(ts, L_HIP) + _px(ts, R_HIP)) / 2.0, axis=1)
    trunk_ready = float(np.nanmedian(trunk[ready])) + 1e-9

    from .kinematics import ankle_to_head_norm
    base_scale = config.body_height_cm / ankle_to_head_norm(ts)
    ratio = np.clip(trunk_ready / np.where(trunk > 1e-6, trunk, trunk_ready),
                    0.5, 2.0)
    scale_per_frame = base_scale * ratio

    tier = ("reject" if deviation > DEVIATION_REJECT_DEG
            else "degraded" if deviation > DEVIATION_DEGRADED_DEG else "ok")

    return Calibration(
        roll_deg=roll, camera_deviation_deg=deviation,
        shoulder_zero_deg=shoulder_zero, hip_zero_deg=hip_zero,
        scale_per_frame=scale_per_frame, ready_frames=ready, tier=tier)


def apply_roll_correction(ts: PoseTimeseries, cal: Calibration) -> PoseTimeseries:
    """หมุน landmark กลับด้วย -roll รอบจุดกลางลำตัว (pixel space)"""
    if abs(cal.roll_deg) < 0.5:
        return ts

    aspect = ts.meta.width / max(ts.meta.height, 1)
    theta = np.radians(-cal.roll_deg)
    c, s = np.cos(theta), np.sin(theta)

    lm = ts.landmarks.copy()
    xy = lm[:, :, :2].copy()
    xy[:, :, 0] *= aspect
    center = np.nanmean(
        xy[:, [L_HIP, R_HIP, L_SHOULDER, R_SHOULDER], :], axis=1, keepdims=True)
    rel = xy - center
    rot = np.empty_like(rel)
    rot[:, :, 0] = rel[:, :, 0] * c - rel[:, :, 1] * s
    rot[:, :, 1] = rel[:, :, 0] * s + rel[:, :, 1] * c
    xy = rot + center
    xy[:, :, 0] /= aspect
    lm[:, :, :2] = xy

    wlm = ts.world_landmarks.copy()
    wx = wlm[:, :, 0] * c - wlm[:, :, 1] * s
    wy = wlm[:, :, 0] * s + wlm[:, :, 1] * c
    wlm[:, :, 0], wlm[:, :, 1] = wx, wy

    out = PoseTimeseries(
        landmarks=lm, world_landmarks=wlm, visibility=ts.visibility,
        timestamps_ms=ts.timestamps_ms, meta=ts.meta,
        frame_stats=ts.frame_stats)
    out.effective_fps = ts.effective_fps
    out.source_fps = ts.source_fps
    return out


def calibration_vq_issues(cal: Calibration) -> list[str]:
    """แปลงผล calibration เป็น VQ issues

    - deviation > reject → "side_view" (enum เดิมของลูกค้า)
    - roll เกิน → "camera_tilted" (⚠️ enum ใหม่ที่เสนอเพิ่ม — รอลูกค้า
      confirm ใน Phase 0; ระบุใน D0.2)
    """
    issues = []
    if cal.tier == "reject":
        issues.append("side_view")
    if abs(cal.roll_deg) > ROLL_FLAG_DEG:
        issues.append("camera_tilted")
    return issues
