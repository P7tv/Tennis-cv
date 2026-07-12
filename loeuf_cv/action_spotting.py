"""Phase 2: Action Spotting — หาจังหวะตีจากวิดีโอ session ยาว

แนวทาง: heuristic signal processing ไม่ใช่ deep learning
1. motion energy ของ upper body ต่อเฟรม (ผลรวมความเร็ว landmark)
2. find_peaks บนสัญญาณ smoothed → candidate impact ต่อจังหวะตี
3. ตัด window ±1.5s รอบ peak → ส่งเข้า Phase 1 pipeline (reuse ทั้งหมด)
4. window ที่ detect impact ไม่ได้ / confidence = discard → คัดทิ้ง

จุดอ่อนที่รู้: volley เบา ๆ motion ต่ำ อาจหลุด (known limitation —
เป้าคือ recall ~85-90% ไม่ใช่ 100%)
"""

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks, savgol_filter

from .config import UPPER_BODY, PipelineConfig
from .kinematics import motion_energy
from .pose_extractor import PoseTimeseries


@dataclass
class StrokeWindow:
    start_frame: int
    end_frame: int          # exclusive
    peak_frame: int
    peak_energy: float

    @property
    def offset_ms(self) -> float:
        return self._offset_ms

    def set_offset(self, timestamps_ms: np.ndarray):
        self._offset_ms = float(timestamps_ms[self.start_frame])


def spot_strokes(ts: PoseTimeseries, config: PipelineConfig) -> list[StrokeWindow]:
    energy = motion_energy(ts, UPPER_BODY)
    T = len(energy)
    if T < 10:
        return []

    window = min(11, T if T % 2 == 1 else T - 1)
    if window >= 5:
        energy = savgol_filter(energy, window, polyorder=2)

    min_gap_frames = max(1, int(config.spotting_min_stroke_gap_s * ts.fps))
    peak_max = float(np.nanmax(energy))
    if peak_max <= 0:
        return []

    peaks, _ = find_peaks(
        energy,
        distance=min_gap_frames,
        prominence=config.spotting_prominence_ratio * peak_max,
    )

    before = int(config.spotting_window_before_s * ts.fps)
    after = int(config.spotting_window_after_s * ts.fps)

    windows = []
    for p in peaks:
        w = StrokeWindow(
            start_frame=max(0, int(p) - before),
            end_frame=min(T, int(p) + after),
            peak_frame=int(p),
            peak_energy=float(energy[p]),
        )
        w.set_offset(ts.timestamps_ms)
        windows.append(w)
    return windows
