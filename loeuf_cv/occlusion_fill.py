"""Kinematic gap-fill สำหรับ landmark ที่ visibility ต่ำต่อเนื่อง (เช่น
ข้อมือโดนลำตัวบังระหว่างสวิง) — เสริมจาก smoothing.py (ซึ่งเติมเฉพาะ NaN
runs สั้น ๆ ด้วย linear interpolation ตรง ๆ บนตำแหน่ง ไม่รู้เรื่อง
โครงสร้างร่างกาย)

แนวคิด (เหมือน motion-capture gap-filling เวลา marker หาย): landmark
"แม่" ที่ visibility สูงกว่ามาก (เช่น ข้อศอก) ใช้เป็นจุดยึด + ความยาว
ท่อนแขน/ขา (คงที่โดยประมาณ, calibrate เป็น median จากทั้งคลิป) + สมมติ
ทิศทางแขน/ขาหมุนต่อเนื่องราบเรียบ (cubic spline บน unit vector
แม่→ลูก) → ผสม (blend 50/50) กับ linear interpolation ตรงบนตำแหน่งเอง

⚠️ Validated ด้วย held-out test จริง (2026-07-10 — ซ่อนข้อมูลที่รู้คำตอบ
อยู่แล้วจากคลิปจริงแล้ววัด error, ทั้งใน world_landmarks และ landmarks
normalized): blend ให้ error สม่ำเสมอกว่าใช้ linear หรือ kinematic เดี่ยว ๆ
(ไม่มีกรณีไหนพังยับแบบที่แต่ละวิธีเดี่ยวเจอ) แต่ error สัมบูรณ์ยังสูง —
ประมาณ 35-75% ของความยาวท่อนแขนเอง ขึ้นกับความยาว gap ไม่ใช่ค่าที่แม่นยำ
เทียบเท่าการ detect จริง จึงจำกัดให้ใช้เฉพาะ gap สั้น
(<= MAX_KINEMATIC_GAP_FRAMES) เท่านั้น และตั้งใจ "ไม่แก้ visibility เดิม"
เพื่อให้ downstream confidence flag (CF1/CF3/VF ใน confidence.py/config.py)
ยังรายงานว่าเป็นค่าประมาณ ไม่ใช่ค่าที่ detect ได้จริง — เติมตำแหน่งให้
ต่อเนื่องใช้งานได้ (เช่น วาด wrist_path, KN7 fallback) แต่ไม่หลอกระบบ
ความเชื่อมั่นว่าแม่นกว่าที่เป็นจริง
"""

import numpy as np
from scipy.interpolate import CubicSpline

from .config import (
    L_ANKLE, L_ELBOW, L_KNEE, L_WRIST, R_ANKLE, R_ELBOW, R_KNEE, R_WRIST,
)
from .pose_extractor import PoseTimeseries

# child landmark idx -> parent landmark idx (parent = จุดยึด+ทิศทางอ้างอิง)
KINEMATIC_PARENT = {
    L_WRIST: L_ELBOW,
    R_WRIST: R_ELBOW,
    L_ANKLE: L_KNEE,
    R_ANKLE: R_KNEE,
}

LOW_VISIBILITY_THRESHOLD = 0.5   # ต่ำกว่านี้ถือว่า "หาย" ต้อง gap-fill
MAX_KINEMATIC_GAP_FRAMES = 20    # ยาวกว่านี้ error สูงเกินจะเชื่อ (validated)
BLEND_WEIGHT_KINEMATIC = 0.5     # 50/50 กับ linear — validated ว่าให้ผลสม่ำเสมอสุด
MIN_CALIBRATION_FRAMES = 10      # ต้องมีเฟรมดีพอจะ calibrate ความยาวท่อน


def _find_low_vis_runs(vis: np.ndarray, threshold: float) -> list[tuple[int, int]]:
    """ช่วง (start, end) ที่ visibility < threshold ต่อเนื่อง (end ไม่รวม)"""
    low = vis < threshold
    runs = []
    start = None
    for i, is_low in enumerate(low):
        if is_low and start is None:
            start = i
        elif not is_low and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(vis)))
    return runs


def kinematic_fill_joint(coords: np.ndarray, visibility: np.ndarray,
                         child_idx: int, parent_idx: int,
                         low_visibility_threshold: float = LOW_VISIBILITY_THRESHOLD,
                         max_gap_frames: int = MAX_KINEMATIC_GAP_FRAMES,
                         blend_weight: float = BLEND_WEIGHT_KINEMATIC,
                         min_calibration_frames: int = MIN_CALIBRATION_FRAMES
                         ) -> np.ndarray:
    """เติมตำแหน่งของ child_idx (ใน coords รูปแบบ (T, 33, D) — ใช้ได้ทั้ง
    landmarks 2D หรือ world_landmarks 3D) ช่วงที่ visibility ต่ำต่อเนื่อง
    สั้น ๆ ด้วย blend(kinematic, linear) — คืน array ใหม่ (ไม่แก้ของเดิม)
    """
    out = coords.copy()
    child_vis = visibility[:, child_idx]
    parent_vis = visibility[:, parent_idx]
    T = len(child_vis)

    good = child_vis >= low_visibility_threshold
    if good.sum() < min_calibration_frames:
        return out  # ข้อมูลดีไม่พอจะ calibrate ความยาวท่อน — ไม่แตะต้อง

    good_t = np.where(good)[0]
    child_good = coords[good, child_idx, :]
    parent_good = coords[good, parent_idx, :]
    limb_len = np.median(np.linalg.norm(child_good - parent_good, axis=1))
    vec = child_good - parent_good
    norms = np.linalg.norm(vec, axis=1, keepdims=True)
    unit = vec / np.where(norms == 0, 1.0, norms)
    n_dims = coords.shape[2]
    splines = [CubicSpline(good_t, unit[:, k]) for k in range(n_dims)]

    for start, end in _find_low_vis_runs(child_vis, low_visibility_threshold):
        gap_len = end - start
        if gap_len > max_gap_frames:
            continue  # ยาวเกินจะเชื่อ — ปล่อยตามเดิม (ดู module docstring)
        if start == 0 or end == T:
            continue  # ไม่มีขอบสองข้าง (ต้น/ท้ายคลิป) — interpolate ไม่ได้
        if np.any(parent_vis[start:end] < low_visibility_threshold):
            continue  # parent เองก็ไม่น่าเชื่อถือช่วงนี้ — ข้าม

        t_gap = np.arange(start, end)
        lin = np.stack([np.interp(t_gap, good_t, coords[good, child_idx, k])
                       for k in range(n_dims)], axis=1)

        pred_unit = np.stack([sp(t_gap) for sp in splines], axis=1)
        pred_norm = np.linalg.norm(pred_unit, axis=1, keepdims=True)
        pred_unit /= np.where(pred_norm == 0, 1.0, pred_norm)
        kin = coords[start:end, parent_idx, :] + limb_len * pred_unit

        out[start:end, child_idx, :] = blend_weight * kin + (1 - blend_weight) * lin

    return out


def kinematic_fill(ts: PoseTimeseries) -> PoseTimeseries:
    """รัน kinematic_fill_joint ทุกคู่ใน KINEMATIC_PARENT บนทั้ง landmarks
    (2D image-normalized) และ world_landmarks (3D เมตร) — คืน
    PoseTimeseries ใหม่ visibility เดิมไม่เปลี่ยนโดยเจตนา (ดู module
    docstring) ให้ downstream confidence flag ยังรายงานตามจริง"""
    lm = ts.landmarks.copy()
    wlm = ts.world_landmarks.copy()
    for child_idx, parent_idx in KINEMATIC_PARENT.items():
        lm = kinematic_fill_joint(lm, ts.visibility, child_idx, parent_idx)
        wlm = kinematic_fill_joint(wlm, ts.visibility, child_idx, parent_idx)

    out = PoseTimeseries(
        landmarks=lm,
        world_landmarks=wlm,
        visibility=ts.visibility,
        timestamps_ms=ts.timestamps_ms,
        meta=ts.meta,
        frame_stats=ts.frame_stats,
    )
    out.effective_fps = ts.effective_fps
    out.source_fps = ts.source_fps
    return out
