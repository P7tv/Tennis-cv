"""Temporal smoothing + gap filling สำหรับ landmark timeseries

มุมหลังทำให้แขนข้างถือแร็กเก็ตโดน occlude ช่วงสั้น ๆ (backswing/serve)
— gap สั้นกว่า max_gap_fill_frames จะถูก interpolate จากเฟรมข้างเคียง
gap ยาวปล่อยเป็น NaN ให้ชั้น visibility flag รายงานตามจริง
"""

import numpy as np
from scipy.signal import savgol_filter

from .config import PipelineConfig
from .pose_extractor import PoseTimeseries


def _fill_short_gaps(series: np.ndarray, max_gap: int) -> np.ndarray:
    """Interpolate NaN runs ที่สั้นกว่า max_gap (ต่อ 1 สัญญาณ 1D)"""
    out = series.copy()
    isnan = np.isnan(out)
    if not isnan.any() or isnan.all():
        return out

    idx = np.arange(len(out))
    # หา run ของ NaN
    edges = np.diff(isnan.astype(int))
    starts = list(np.where(edges == 1)[0] + 1)
    ends = list(np.where(edges == -1)[0] + 1)
    if isnan[0]:
        starts.insert(0, 0)
    if isnan[-1]:
        ends.append(len(out))

    valid = ~isnan
    for s, e in zip(starts, ends):
        if e - s <= max_gap:
            out[s:e] = np.interp(idx[s:e], idx[valid], out[valid])
    return out


def smooth_timeseries(ts: PoseTimeseries, config: PipelineConfig) -> PoseTimeseries:
    """Gap-fill + Savitzky-Golay smoothing ทั้ง image และ world landmarks"""
    T = ts.n_frames
    window = min(config.smoothing_window_frames, T if T % 2 == 1 else T - 1)

    def process(arr: np.ndarray) -> np.ndarray:
        out = arr.copy()
        for lm in range(arr.shape[1]):
            for axis in range(arr.shape[2]):
                sig = _fill_short_gaps(out[:, lm, axis], config.max_gap_fill_frames)
                valid = ~np.isnan(sig)
                if window >= 5 and valid.sum() == T:
                    sig = savgol_filter(sig, window, polyorder=2)
                out[:, lm, axis] = sig
        return out

    return PoseTimeseries(
        landmarks=process(ts.landmarks),
        world_landmarks=process(ts.world_landmarks),
        visibility=ts.visibility,
        timestamps_ms=ts.timestamps_ms,
        meta=ts.meta,
    )
