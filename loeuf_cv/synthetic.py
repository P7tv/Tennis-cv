"""Synthetic landmark timeseries — จำลอง stroke จากมุมหลัง

ใช้ทดสอบ logic ชั้นบน (keyframes / measurements / spotting) โดยไม่ต้อง
มีวิดีโอหรือ mediapipe และใช้เดโมใน notebook ได้
"""

import numpy as np
from scipy.signal import savgol_filter

from .config import (
    L_ANKLE, L_ELBOW, L_FOOT, L_HIP, L_KNEE, L_SHOULDER, L_WRIST,
    NOSE, R_ANKLE, R_ELBOW, R_FOOT, R_HIP, R_KNEE, R_SHOULDER, R_WRIST,
)
from .pose_extractor import PoseTimeseries, VideoMeta


def _track(t: np.ndarray, key_times: list, key_values: list) -> np.ndarray:
    sig = np.interp(t, key_times, key_values)
    window = min(9, len(t) if len(t) % 2 == 1 else len(t) - 1)
    if window >= 5:
        sig = savgol_filter(sig, window, polyorder=2)
    return sig


def synthetic_forehand(fps: float = 30.0, duration_s: float = 3.0,
                       t_offset_s: float = 0.0,
                       wrist_x_track: tuple | None = None,
                       wrist_y_track: tuple | None = None) -> PoseTimeseries:
    """Forehand มือขวา มุมหลัง: ready → unit turn/backswing → forward
    swing เร็ว (impact ~1.3s) → follow-through → กลับ ready

    override wrist track ได้เพื่อจำลอง stroke อื่น (ดู variants ท้ายไฟล์)"""
    T = int(fps * duration_s)
    t = np.arange(T) / fps

    lm = np.zeros((T, 33, 3))
    vis = np.full((T, 33), 0.95)

    cx = 0.5   # ลำตัวอยู่กลางเฟรม
    shoulder_y, hip_y, knee_y, ankle_y = 0.40, 0.55, 0.70, 0.85

    # torso rotation: shoulder width หด 0.16 → 0.08 ช่วง 0.5–1.3s แล้วคืน
    half_sw = _track(t, [0, 0.5, 0.9, 1.3, 1.8, duration_s],
                     [0.08, 0.08, 0.045, 0.04, 0.08, 0.08])

    # dominant (R) wrist: หลัก ๆ ของ stroke
    xt = wrist_x_track or ([0, 0.5, 1.2, 1.4, 1.9, 2.4, duration_s],
                           [0.55, 0.55, 0.75, 0.35, 0.30, 0.55, 0.55])
    yt = wrist_y_track or ([0, 0.5, 1.2, 1.3, 1.9, 2.4, duration_s],
                           [0.55, 0.55, 0.60, 0.50, 0.35, 0.55, 0.55])
    wrist_x = _track(t, *xt)
    wrist_y = _track(t, *yt)

    for i in range(T):
        sw, hw = half_sw[i], 0.06
        lm[i, NOSE] = (cx, shoulder_y - 0.08, 0)
        lm[i, L_SHOULDER] = (cx - sw, shoulder_y, 0)
        lm[i, R_SHOULDER] = (cx + sw, shoulder_y, 0)
        lm[i, L_HIP] = (cx - hw, hip_y, 0)
        lm[i, R_HIP] = (cx + hw, hip_y, 0)
        lm[i, L_KNEE] = (cx - hw, knee_y, 0)
        lm[i, R_KNEE] = (cx + hw, knee_y, 0)
        lm[i, L_ANKLE] = (cx - 0.06, ankle_y, 0)
        lm[i, R_ANKLE] = (cx + 0.06, ankle_y, 0)
        lm[i, L_FOOT] = (cx - 0.06, ankle_y + 0.03, 0)
        lm[i, R_FOOT] = (cx + 0.06, ankle_y + 0.03, 0)

        lm[i, R_WRIST] = (wrist_x[i], wrist_y[i], 0)
        lm[i, R_ELBOW] = ((lm[i, R_SHOULDER, 0] + wrist_x[i]) / 2,
                          (shoulder_y + wrist_y[i]) / 2 + 0.02, 0)
        lm[i, L_WRIST] = (cx - 0.10, hip_y, 0)
        lm[i, L_ELBOW] = (cx - sw - 0.02, (shoulder_y + hip_y) / 2, 0)

    # แขนขวาโดน occlude เล็กน้อยช่วง backswing (สมจริงกับมุมหลัง)
    occl = (t > 0.9) & (t < 1.25)
    vis[occl, R_WRIST] = 0.65
    vis[occl, R_ELBOW] = 0.70

    # world landmarks: scale image coords เป็นเมตรหยาบ ๆ + depth เล็กน้อย
    wlm = (lm - [cx, hip_y, 0]) * [1.7, 1.7, 1.0]
    wlm[:, R_WRIST, 2] = _track(t, [0, 1.2, 1.35, duration_s],
                                [0.0, 0.15, -0.25, 0.0])

    # torso rotation ใน world z: coil (-30°) ก่อน impact แล้วเปิดตัว (+45°)
    rot = np.radians(_track(t, [0, 0.5, 1.2, 1.35, 1.9, duration_s],
                            [0, 0, -30, 45, 10, 0]))
    sw_world = half_sw * 1.7
    wlm[:, L_SHOULDER, 2] = -sw_world * np.sin(rot)
    wlm[:, R_SHOULDER, 2] = sw_world * np.sin(rot)
    hw_world = 0.06 * 1.7
    wlm[:, L_HIP, 2] = -hw_world * np.sin(rot * 0.7)
    wlm[:, R_HIP, 2] = hw_world * np.sin(rot * 0.7)

    meta = VideoMeta(path=f"synthetic_fh_{fps:.0f}fps.mp4", fps=fps,
                     frame_count=T, width=1920, height=1080)
    return PoseTimeseries(
        landmarks=lm, world_landmarks=wlm, visibility=vis,
        timestamps_ms=(t + t_offset_s) * 1000.0, meta=meta)


def synthetic_session(fps: float = 30.0, n_strokes: int = 3,
                      idle_s: float = 2.5) -> PoseTimeseries:
    """ต่อ stroke หลายอันคั่นด้วยช่วง idle → ใช้ทดสอบ action spotting"""
    parts, t_cursor = [], 0.0

    def idle(duration: float, start: float) -> PoseTimeseries:
        p = synthetic_forehand(fps, 3.0)
        n = int(fps * duration)
        return PoseTimeseries(
            landmarks=np.repeat(p.landmarks[:1], n, axis=0),
            world_landmarks=np.repeat(p.world_landmarks[:1], n, axis=0),
            visibility=np.repeat(p.visibility[:1], n, axis=0),
            timestamps_ms=(np.arange(n) / fps + start) * 1000.0,
            meta=p.meta)

    for _ in range(n_strokes):
        gap = idle(idle_s, t_cursor)
        t_cursor += idle_s
        stroke = synthetic_forehand(fps, 3.0, t_offset_s=t_cursor)
        t_cursor += 3.0
        parts.extend([gap, stroke])
    parts.append(idle(idle_s, t_cursor))
    t_cursor += idle_s

    meta = VideoMeta(path="synthetic_session.mp4", fps=fps,
                     frame_count=int(t_cursor * fps), width=1920, height=1080)
    return PoseTimeseries(
        landmarks=np.concatenate([p.landmarks for p in parts]),
        world_landmarks=np.concatenate([p.world_landmarks for p in parts]),
        visibility=np.concatenate([p.visibility for p in parts]),
        timestamps_ms=np.concatenate([p.timestamps_ms for p in parts]),
        meta=meta)


def synthetic_backhand(fps: float = 30.0) -> PoseTimeseries:
    """BH มือขวา: backswing ข้ามไปฝั่งซ้ายของลำตัว"""
    return synthetic_forehand(
        fps,
        wrist_x_track=([0, 0.5, 1.2, 1.4, 1.9, 2.4, 3.0],
                       [0.45, 0.45, 0.25, 0.65, 0.70, 0.45, 0.45]))


def synthetic_slice(fps: float = 30.0) -> PoseTimeseries:
    """SL: โครง BH + swing path สูง→ต่ำชัดเจน"""
    return synthetic_forehand(
        fps,
        wrist_x_track=([0, 0.5, 1.2, 1.4, 1.9, 2.4, 3.0],
                       [0.45, 0.45, 0.25, 0.65, 0.70, 0.45, 0.45]),
        wrist_y_track=([0, 0.5, 1.2, 1.4, 1.9, 2.4, 3.0],
                       [0.55, 0.55, 0.36, 0.54, 0.52, 0.55, 0.55]))


def synthetic_serve_like(fps: float = 30.0) -> PoseTimeseries:
    """SV: wrist ขึ้นเหนือหัว ณ impact (overhead)"""
    return synthetic_forehand(
        fps,
        wrist_x_track=([0, 0.5, 1.0, 1.25, 1.4, 1.9, 2.4, 3.0],
                       [0.55, 0.55, 0.60, 0.62, 0.50, 0.52, 0.55, 0.55]),
        wrist_y_track=([0, 0.5, 1.0, 1.2, 1.35, 1.9, 2.4, 3.0],
                       [0.55, 0.55, 0.60, 0.45, 0.18, 0.42, 0.55, 0.55]))


def synthetic_volley_like(fps: float = 30.0) -> PoseTimeseries:
    """VL: backswing สั้น punch ไปหน้า"""
    return synthetic_forehand(
        fps,
        wrist_x_track=([0, 0.5, 1.2, 1.35, 1.7, 2.4, 3.0],
                       [0.55, 0.55, 0.63, 0.42, 0.47, 0.55, 0.55]),
        wrist_y_track=([0, 0.5, 1.2, 1.35, 1.9, 2.4, 3.0],
                       [0.52, 0.52, 0.50, 0.48, 0.50, 0.52, 0.52]))
