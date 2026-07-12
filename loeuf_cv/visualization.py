"""043-VZ · visualization — render payload for overlay

VZ1 wrist_path = REQUIRED เสมอ (ทุกเฟรม backswing_peak →
follow_through_peak; keyframe หายให้ fallback เป็นทั้งคลิป)
VZ2-VZ4 ต้องการ ball/racket detector → null
VZ5 body_center_path = CoM trajectory (คำนวณได้ ส่งให้เลย)
"""

import numpy as np

from .config import PipelineConfig
from .keyframes import Keyframe
from .kinematics import com_xy, dominant
from .pose_extractor import PoseTimeseries


def _path(points: np.ndarray, frames: range) -> list[dict]:
    out = []
    for i in frames:
        x, y = points[i]
        if np.isnan(x) or np.isnan(y):
            continue
        out.append({"frame": int(i), "x": round(float(x), 3),
                    "y": round(float(y), 3)})
    return out


def build_visualization(ts: PoseTimeseries, keyframes: dict[str, Keyframe],
                        config: PipelineConfig) -> dict:
    b2 = keyframes.get("backswing_peak")
    b4 = keyframes.get("follow_through_peak")
    start = b2.frame_index if (b2 and b2.detected) else 0
    end = (b4.frame_index + 1) if (b4 and b4.detected) else ts.n_frames
    frames = range(start, end)

    side = dominant(config.dominant_side)
    wrist = ts.landmarks[:, side["wrist"], :2]

    return {
        "wrist_path": _path(wrist, frames),          # VZ1 REQUIRED
        "ball_path": None,                           # VZ2 ต้องการ ball detection
        "racket_tip_path": None,                     # VZ3 ต้องการ racket detection
        "racket_center_path": None,                  # VZ4 ต้องการ racket detection
        "body_center_path": _path(com_xy(ts), frames),  # VZ5
    }
