"""ตรวจจับ Keyframes 5 จุด (TOR Phase 1)

unit_turn → backswing_peak → impact → follow_through_peak
→ ready_position_restored

- impact ใช้ sub-frame parabolic interpolation บน wrist-speed peak
  → ได้ timestamp ละเอียดกว่า frame spacing (แก้ปัญหา 30fps ที่เฟรม
  impact จริงอาจไม่ถูกถ่าย)
- detect ไม่ได้ → {"frame_index": null, "timestamp_ms": null,
  "detected": false} ตาม Brief Section 6
- ready_position_restored = false ถือเป็นข้อมูลปกติ ไม่ใช่ error
"""

from dataclasses import dataclass

import numpy as np

from .config import KEYFRAME_NAMES, L_ANKLE, R_ANKLE, PipelineConfig
from .kinematics import (
    body_height_norm, dominant, hip_center, speed, torso_rotation_proxy,
)
from .pose_extractor import PoseTimeseries


@dataclass
class Keyframe:
    name: str
    frame_index: int | None
    timestamp_ms: float | None
    detected: bool

    def to_dict(self) -> dict:
        return {
            "frame_index": self.frame_index,
            "timestamp_ms": round(self.timestamp_ms, 2)
            if self.timestamp_ms is not None else None,
            "detected": self.detected,
        }


def _missing(name: str) -> Keyframe:
    return Keyframe(name, None, None, False)


def _parabolic_refine(signal: np.ndarray, peak_idx: int,
                      timestamps_ms: np.ndarray) -> float:
    """Sub-frame peak time จาก parabola ผ่านจุด peak และเพื่อนบ้าน"""
    if peak_idx <= 0 or peak_idx >= len(signal) - 1:
        return float(timestamps_ms[peak_idx])
    y0, y1, y2 = signal[peak_idx - 1: peak_idx + 2]
    denom = y0 - 2 * y1 + y2
    if abs(denom) < 1e-12:
        return float(timestamps_ms[peak_idx])
    offset = np.clip(0.5 * (y0 - y2) / denom, -0.5, 0.5)
    dt = timestamps_ms[min(peak_idx + 1, len(signal) - 1)] - timestamps_ms[peak_idx]
    return float(timestamps_ms[peak_idx] + offset * dt)


def detect_keyframes(ts: PoseTimeseries, config: PipelineConfig,
                     stroke_type: str) -> dict[str, Keyframe]:
    side = dominant(config.dominant_side)
    T = ts.n_frames
    t_ms = ts.timestamps_ms

    wrist = ts.landmarks[:, side["wrist"], :2]
    wrist_speed = speed(np.nan_to_num(wrist, nan=0.0), t_ms)
    rotation = torso_rotation_proxy(ts)
    hip_c = hip_center(ts)
    height = body_height_norm(ts)

    kf: dict[str, Keyframe] = {name: _missing(name) for name in KEYFRAME_NAMES}

    # --- impact: peak wrist speed (ตัดขอบคลิปกัน false peak ตอนเดิน) ---
    margin = max(1, int(T * config.impact_search_margin))
    search = wrist_speed.copy()
    search[:margin] = 0
    search[T - margin:] = 0
    if search.max() <= 0:
        return kf
    impact_idx = int(np.argmax(search))
    kf["impact"] = Keyframe(
        "impact", impact_idx,
        _parabolic_refine(wrist_speed, impact_idx, t_ms), True,
    )

    # --- backswing_peak: ก่อน impact, wrist ห่างลำตัว (x) มากสุด ---
    if impact_idx > 2:
        lateral = np.abs(wrist[:impact_idx, 0] - hip_c[:impact_idx, 0])
        lateral = np.nan_to_num(lateral, nan=0.0)
        bs_idx = int(np.argmax(lateral))
        if lateral[bs_idx] > 0.02:  # กัน degenerate case ที่ wrist ไม่เคลื่อน
            kf["backswing_peak"] = Keyframe(
                "backswing_peak", bs_idx, float(t_ms[bs_idx]), True)

    # --- unit_turn: onset ของ torso rotation ก่อน backswing_peak ---
    bs = kf["backswing_peak"]
    if bs.detected and bs.frame_index > 1:
        window = rotation[: bs.frame_index]
        peak_rot = np.nanmax(window)
        if peak_rot > 0.05:
            onset_level = config.unit_turn_onset_ratio * peak_rot
            above = np.where(window >= onset_level)[0]
            if len(above):
                ut_idx = int(above[0])
                kf["unit_turn"] = Keyframe(
                    "unit_turn", ut_idx, float(t_ms[ut_idx]), True)

    # --- follow_through_peak: หลัง impact, wrist สุดทางฝั่งตรงข้าม/สูงสุด ---
    ft_limit_ms = t_ms[impact_idx] + config.follow_through_max_ms
    ft_range = np.where((t_ms > t_ms[impact_idx]) & (t_ms <= ft_limit_ms))[0]
    if len(ft_range) >= 2:
        # ระยะ wrist จาก hip center (รวมความสูง) — peak = สุด follow-through
        rel = wrist[ft_range] - hip_c[ft_range]
        dist = np.nan_to_num(np.linalg.norm(rel, axis=1), nan=0.0)
        ft_idx = int(ft_range[np.argmax(dist)])
        kf["follow_through_peak"] = Keyframe(
            "follow_through_peak", ft_idx, float(t_ms[ft_idx]), True)

    # --- ready_position_restored: กลับสู่ท่าเตรียมใกล้เคียงต้นคลิป ---
    ft = kf["follow_through_peak"]
    if ft.detected:
        n_ref = max(2, int(T * 0.1))
        ref_wrist = np.nanmean(wrist[:n_ref] - hip_c[:n_ref], axis=0)
        ref_stance = np.nanmean(
            np.abs(ts.landmarks[:n_ref, R_ANKLE, 0]
                   - ts.landmarks[:n_ref, L_ANKLE, 0]))
        hold = 0
        for i in range(ft.frame_index + 1, T):
            wrist_dev = np.linalg.norm((wrist[i] - hip_c[i]) - ref_wrist) / height
            stance = abs(ts.landmarks[i, R_ANKLE, 0] - ts.landmarks[i, L_ANKLE, 0])
            stance_dev = abs(stance - ref_stance) / height
            if wrist_dev < config.ready_pose_tolerance and \
               stance_dev < config.ready_pose_tolerance:
                hold += 1
                if hold >= config.ready_min_hold_frames:
                    idx = i - config.ready_min_hold_frames + 1
                    kf["ready_position_restored"] = Keyframe(
                        "ready_position_restored", idx, float(t_ms[idx]), True)
                    break
            else:
                hold = 0

    return kf
