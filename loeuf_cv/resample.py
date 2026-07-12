"""fps normalization ตาม M4/M5: input fps < 60 → resample เป็น 60fps

velocity_normalized = true เมื่อมีการ resample
(linear interpolation บน landmark timeseries — ไม่ใช่ optical flow
ดังนั้น velocity fidelity ยังจำกัดที่ fps ต้นทาง → KN fields จาก
คลิป 30fps ติด visibility low_confidence)
"""

import numpy as np

from .config import PipelineConfig
from .pose_extractor import PoseTimeseries


def _interp_array(arr: np.ndarray, t_old: np.ndarray, t_new: np.ndarray) -> np.ndarray:
    """Linear interp ทีละ channel รองรับ NaN (ปล่อย NaN ไว้ถ้า segment ขาด)"""
    flat = arr.reshape(len(t_old), -1)
    out = np.empty((len(t_new), flat.shape[1]))
    for c in range(flat.shape[1]):
        sig = flat[:, c]
        valid = ~np.isnan(sig)
        if valid.sum() < 2:
            out[:, c] = np.nan
            continue
        out[:, c] = np.interp(t_new, t_old[valid], sig[valid])
        # ช่วงที่ต้นทางเป็น NaN ยาว ๆ → คง NaN (อย่า interpolate ข้าม gap ใหญ่)
        nan_mask = np.interp(t_new, t_old, (~valid).astype(float)) > 0.5
        out[nan_mask, c] = np.nan
    return out.reshape((len(t_new),) + arr.shape[1:])


def resample_to_target_fps(ts: PoseTimeseries,
                           config: PipelineConfig) -> tuple[PoseTimeseries, bool]:
    """คืน (timeseries, velocity_normalized)

    fps >= target → คืนเดิม, velocity_normalized = False
    """
    if ts.fps >= config.target_fps:
        return ts, False

    t_old = ts.timestamps_ms
    duration = t_old[-1] - t_old[0]
    n_new = max(2, int(round(duration / 1000.0 * config.target_fps)) + 1)
    t_new = t_old[0] + np.arange(n_new) * (1000.0 / config.target_fps)
    t_new = t_new[t_new <= t_old[-1] + 1e-6]

    resampled = PoseTimeseries(
        landmarks=_interp_array(ts.landmarks, t_old, t_new),
        world_landmarks=_interp_array(ts.world_landmarks, t_old, t_new),
        visibility=_interp_array(ts.visibility, t_old, t_new),
        timestamps_ms=t_new,
        meta=ts.meta,
        frame_stats=ts.frame_stats,
    )
    resampled.effective_fps = config.target_fps
    resampled.source_fps = ts.fps
    return resampled, True
