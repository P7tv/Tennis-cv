"""One-Euro filter — adaptive low-pass smoothing สำหรับ landmark timeseries

ทางเลือกเสริมจาก Savitzky-Golay ใน smoothing.py (opt-in, default ยังเป็น
savgol เหมือนเดิม — ดู PipelineConfig.smoothing_method)

หลักการ (Casiez, Roussel, Vogel 2012 — "1€ Filter"): cutoff frequency ของ
low-pass ปรับตาม "ความเร็ว" ของสัญญาณเอง — นิ่ง (ความเร็วต่ำ) ใช้ cutoff
ต่ำ (กรอง jitter แรง), เคลื่อนไหวเร็ว (เช่นช่วง swing/impact) ใช้ cutoff
สูงขึ้นอัตโนมัติ (กรองน้อยลง ลด lag) — ต่างจาก Savitzky-Golay ที่ window
คงที่ตลอดคลิป จึงลด jitter ตอนนิ่งได้โดยไม่หน่วงตอนตวัดเร็ว (ตรงกับ
ลักษณะการสวิงเทนนิส: นิ่งช่วง ready/หลัง contact เร็วช่วง backswing→impact)

รวม confidence-weighting เข้าไปด้วย: เฟรมที่ visibility ต่ำ (ค่าที่
MediaPipe เดามั่ว) จะถูกดึงเข้าหาค่าที่ filter คาดไว้ (persistence) แทน
ที่จะเชื่อค่าดิบเต็มที่ — "เชื่อเฟรมมั่นใจ เดาเฟรมไม่มั่นใจ"
"""

import numpy as np

from .config import PipelineConfig
from .pose_extractor import PoseTimeseries


class _LowPassFilter:
    """Exponential low-pass ตัวช่วยของ OneEuroFilter (ไม่ได้ตั้งใจให้ใช้ตรง ๆ)"""

    def __init__(self):
        self.y: float | None = None

    def filter(self, value: float, alpha: float) -> float:
        if self.y is None:
            self.y = value
        else:
            self.y = alpha * value + (1.0 - alpha) * self.y
        return self.y


class OneEuroFilter:
    """1€ Filter — stateful causal filter ทีละ sample (เรียก `__call__`
    ตามลำดับเวลาจริง ห้ามข้าม/สลับลำดับ)"""

    def __init__(self, mincutoff: float = 1.0, beta: float = 0.0,
                dcutoff: float = 1.0, freq_hz: float = 30.0):
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self.freq_hz = freq_hz
        self._x_filter = _LowPassFilter()
        self._dx_filter = _LowPassFilter()
        self._last_t: float | None = None

    @staticmethod
    def _alpha(cutoff: float, freq_hz: float) -> float:
        te = 1.0 / freq_hz
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x: float, timestamp_s: float | None = None) -> float:
        if timestamp_s is not None:
            if self._last_t is not None:
                dt = timestamp_s - self._last_t
                if dt > 1e-9:
                    self.freq_hz = 1.0 / dt
            self._last_t = timestamp_s

        prev_x = self._x_filter.y if self._x_filter.y is not None else x
        dx = (x - prev_x) * self.freq_hz
        edx = self._dx_filter.filter(dx, self._alpha(self.dcutoff, self.freq_hz))
        cutoff = self.mincutoff + self.beta * abs(edx)
        return self._x_filter.filter(x, self._alpha(cutoff, self.freq_hz))


def _filter_1d_confidence_weighted(series: np.ndarray, vis: np.ndarray,
                                   timestamps_ms: np.ndarray, mincutoff: float,
                                   beta: float, dcutoff: float,
                                   low_vis_threshold: float) -> np.ndarray:
    """รัน OneEuroFilter บน 1 สัญญาณ — NaN ใช้ค่าที่ filter เก็บไว้ล่าสุด
    (persistence) แทน, เฟรม visibility ต่ำถูกดึงเข้าหาค่าที่ filter คาดไว้
    ตามสัดส่วนความเชื่อถือ ก่อนป้อนเข้า filter จริง"""
    T = len(series)
    out = np.full(T, np.nan)
    f = OneEuroFilter(mincutoff=mincutoff, beta=beta, dcutoff=dcutoff)
    last_valid = None
    for i in range(T):
        x = series[i]
        c = float(vis[i])
        if np.isnan(x):
            if last_valid is None:
                continue
            x = last_valid
        elif c < low_vis_threshold and last_valid is not None:
            w = max(c, 0.0) / max(low_vis_threshold, 1e-6)
            x = w * x + (1.0 - w) * last_valid
        y = f(x, timestamp_s=timestamps_ms[i] / 1000.0)
        out[i] = y
        last_valid = y
    return out


def apply_one_euro_filter(ts: PoseTimeseries, config: PipelineConfig) -> PoseTimeseries:
    """ทางเลือกแทน smooth_timeseries() (savgol) — ใช้ OneEuroFilter ต่อ
    landmark/แกน พร้อม confidence-weighting จาก visibility"""
    def process(arr: np.ndarray) -> np.ndarray:
        out = np.empty_like(arr)
        for lm in range(arr.shape[1]):
            vis_lm = ts.visibility[:, lm]
            for axis in range(arr.shape[2]):
                out[:, lm, axis] = _filter_1d_confidence_weighted(
                    arr[:, lm, axis], vis_lm, ts.timestamps_ms,
                    config.one_euro_mincutoff, config.one_euro_beta,
                    config.one_euro_dcutoff, config.low_confidence_threshold)
        return out

    out_ts = PoseTimeseries(
        landmarks=process(ts.landmarks),
        world_landmarks=process(ts.world_landmarks),
        visibility=ts.visibility,
        timestamps_ms=ts.timestamps_ms,
        meta=ts.meta,
        frame_stats=ts.frame_stats,
    )
    out_ts.effective_fps = ts.effective_fps
    out_ts.source_fps = ts.source_fps
    return out_ts
