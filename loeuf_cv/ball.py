"""Ball detection interface + impact fusion (Phase 2 stretch goal)

ออกแบบเป็น adapter: โมเดล ball detector (TrackNet / YOLO ที่จะ train
บน VM) แค่ส่ง observations ต่อเฟรมเข้ามา — pipeline ไม่ผูกกับโมเดลใด

ประโยชน์หลักตอนนี้: **impact fusion** — จุดที่ลูกเปลี่ยนทิศกะทันหัน
คือหลักฐาน impact ที่แม่นกว่า wrist peak → ดัน accuracy เข้าเกณฑ์ ≥80%

BL2–BL10 (speed km/h, landing, service box) ต้องการ court calibration
→ ยังส่ง null จนกว่าจะมี court line detection (scope แยก)
"""

from dataclasses import dataclass

import numpy as np

from .config import PipelineConfig
from .keyframes import Keyframe
from .pose_extractor import PoseTimeseries
from .schema_fields import empty_ball_block

FUSION_WINDOW_MS = 150.0     # หา direction change ภายใน ±นี้รอบ pose impact
MIN_BALL_CONFIDENCE = 0.3
MIN_TRACKED_FRACTION = 0.3   # ball ต้องเห็น ≥30% ของ window ถึงจะ fuse


@dataclass
class BallObservations:
    """ตำแหน่งลูกต่อเฟรมของวิดีโอต้นทาง (normalized 0–1)

    positions: (N, 2) — NaN = detect ไม่ได้เฟรมนั้น
    confidence: (N,)
    fps: fps ของวิดีโอต้นทาง (แปลง frame → ms)
    """
    positions: np.ndarray
    confidence: np.ndarray
    fps: float

    @property
    def timestamps_ms(self) -> np.ndarray:
        return np.arange(len(self.positions)) / self.fps * 1000.0


def load_ball_csv(path: str, fps: float) -> BallObservations:
    """Adapter: CSV จาก detector (คอลัมน์ frame,x,y,confidence) → observations"""
    import csv

    rows = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            rows.append((int(r["frame"]), float(r["x"]), float(r["y"]),
                         float(r.get("confidence", 1.0))))
    n = max(fr for fr, *_ in rows) + 1 if rows else 0
    pos = np.full((n, 2), np.nan)
    conf = np.zeros(n)
    for fr, x, y, c in rows:
        if c >= MIN_BALL_CONFIDENCE:
            pos[fr] = (x, y)
            conf[fr] = c
    return BallObservations(pos, conf, fps)


def _interp_to(ball: BallObservations, t_ms: np.ndarray) -> np.ndarray:
    """ball positions → timeline ของ PoseTimeseries (รองรับ resample แล้ว)"""
    src_t = ball.timestamps_ms
    out = np.full((len(t_ms), 2), np.nan)
    for axis in range(2):
        sig = ball.positions[:, axis]
        valid = ~np.isnan(sig)
        if valid.sum() < 2:
            continue
        out[:, axis] = np.interp(t_ms, src_t[valid], sig[valid])
        gap = np.interp(t_ms, src_t, (~valid).astype(float)) > 0.5
        out[gap, axis] = np.nan
    return out


def fuse_impact(ts: PoseTimeseries, keyframes: dict[str, Keyframe],
                ball: BallObservations,
                config: PipelineConfig) -> dict[str, Keyframe]:
    """Refine impact ด้วยจุดที่ลูกเปลี่ยนความเร็ว/ทิศทางกะทันหัน

    fusion rule: impact = peak ของ |Δvelocity| ของลูก ภายใน window
    ±150ms รอบ pose impact — หาไม่ได้ → คง pose impact เดิม
    """
    imp = keyframes.get("impact")
    if imp is None or not imp.detected:
        return keyframes

    t_ms = ts.timestamps_ms
    pos = _interp_to(ball, t_ms)
    window = (t_ms >= imp.timestamp_ms - FUSION_WINDOW_MS) & \
             (t_ms <= imp.timestamp_ms + FUSION_WINDOW_MS)
    idx = np.where(window & ~np.isnan(pos[:, 0]))[0]
    if len(idx) < max(4, int(MIN_TRACKED_FRACTION * window.sum())):
        return keyframes

    lo, hi = idx[0], idx[-1] + 1
    seg = pos[lo:hi]
    seg_t = t_ms[lo:hi]
    dt = np.gradient(seg_t) / 1000.0
    vel = np.gradient(np.nan_to_num(seg, nan=0.0), axis=0) / dt[:, None]
    accel = np.linalg.norm(np.gradient(vel, axis=0) / dt[:, None], axis=1)
    if not np.isfinite(accel).any():
        return keyframes

    peak = int(np.nanargmax(accel))
    frame_idx = lo + peak

    from .keyframes import _parabolic_refine
    refined_ms = _parabolic_refine(np.nan_to_num(accel, nan=0.0), peak, seg_t)

    out = dict(keyframes)
    out["impact"] = Keyframe("impact", frame_idx, refined_ms, True)
    return out


def build_ball_block(ball: BallObservations) -> dict:
    """BL block — มี detector แล้วแต่ยังไม่มี court calibration
    → available true, field อื่น null (ตาม gate BL1)

    key ทั้งหมดมาจาก schema_fields.BALL_FIELDS — ห้ามเขียน literal เองซ้ำ
    (เดิม fallback ใน schema_builder/builder.py ส่งแค่ {"available": False}
    แล้ว drift ห่างจาก block นี้)"""
    tracked = float(np.mean(~np.isnan(ball.positions[:, 0])))
    block = empty_ball_block(available=True)
    block["_tracked_fraction"] = round(tracked, 2)   # debug extension
    return block


def build_ball_path(ball: BallObservations, ts: PoseTimeseries,
                    keyframes: dict[str, Keyframe]) -> list[dict] | None:
    """VZ2 ball_path — ช่วง backswing_peak → follow_through_peak"""
    b2 = keyframes.get("backswing_peak")
    b4 = keyframes.get("follow_through_peak")
    start = b2.frame_index if (b2 and b2.detected) else 0
    end = (b4.frame_index + 1) if (b4 and b4.detected) else ts.n_frames

    pos = _interp_to(ball, ts.timestamps_ms)
    out = []
    for i in range(start, end):
        x, y = pos[i]
        if np.isnan(x) or np.isnan(y):
            continue
        out.append({"frame": int(i), "x": round(float(x), 3),
                    "y": round(float(y), 3)})
    return out or None
