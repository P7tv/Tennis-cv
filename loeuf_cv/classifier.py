"""Hierarchical rule-based stroke classifier (Phase 2 — A3)

โครงตามที่ตกลง: จำแนกตามธรรมชาติของคลาส ไม่ใช่ flat 6-class
    ชั้น 1: SV (overhead) / VL (compact) หลุดก่อน — สัญญาณชัด
    ชั้น 2: groundstroke → FH-family / BH-family จากฝั่ง backswing
    ชั้น 3: BH-family + swing path ลงชัด → SL

⚠️ RS ไม่มีทาง classify จาก pose ใน window เดียว (เชิง kinematics
มันคือ FH/BH — ต่างที่บริบท) → classifier นี้ไม่คืน RS เด็ดขาด
ลูกค้าต้องเลือก: ระบุมาให้ / heuristic ลูกแรกของ rally / รวมกับ FH-BH

confidence ต่ำกว่า threshold → ตัวเรียกควรส่ง coach_review
"""

from dataclasses import dataclass

import numpy as np

from .config import NOSE, PipelineConfig
from .keyframes import detect_keyframes
from .kinematics import body_height_norm, dominant, torso_center
from .pose_extractor import PoseTimeseries
from .smoothing import smooth_timeseries

OVERHEAD_MARGIN = 0.03        # wrist สูงกว่า nose เกินนี้ ณ impact → SV (TBC)
COMPACT_AMPLITUDE = 0.35      # backswing amplitude ต่ำกว่านี้ → VL (TBC)
SLICE_PATH_DEG = -15.0        # swing path ลงชันกว่านี้ + BH → SL (ตาม D1)


@dataclass
class Classification:
    stroke_type: str
    confidence: float
    features: dict

    def to_debug_dict(self) -> dict:
        return {"predicted_type": self.stroke_type,
                "confidence": round(self.confidence, 2),
                "features": {k: round(v, 3) if isinstance(v, float) else v
                             for k, v in self.features.items()}}


def _conf(margin: float, scale: float) -> float:
    """แปลงระยะห่างจาก threshold เป็น confidence 0.5–0.95"""
    return float(np.clip(0.5 + abs(margin) / scale * 0.45, 0.5, 0.95))


def classify_stroke(ts: PoseTimeseries,
                    config: PipelineConfig) -> Classification:
    smoothed = smooth_timeseries(ts, config)
    side = dominant(config.dominant_side)
    height = body_height_norm(smoothed)

    # ใช้ keyframe detector เดิมหา B2/B3 (stroke type ไม่กระทบ detection)
    kf = detect_keyframes(smoothed, config, "FH")
    b2 = kf["backswing_peak"].frame_index if kf["backswing_peak"].detected else None
    b3 = kf["impact"].frame_index if kf["impact"].detected else None

    lm = smoothed.landmarks
    torso = torso_center(smoothed)
    if b3 is None:
        b3 = len(smoothed.landmarks) // 2

    features: dict = {}

    # --- ชั้น 1a: overhead → SV (กฎสรีระทางกายภาพ: การตีเหนือหัวสูงกว่าศีรษะคือ Serve / Smash) ---
    nose_y = float(np.nanmedian(lm[:, NOSE, 1]))
    lo, hi = max(0, b3 - 5), min(smoothed.n_frames, b3 + 6)
    wrist_y_impact = float(np.nanmin(lm[lo:hi, side["wrist"], 1]))
    overhead_margin = nose_y - wrist_y_impact  # + = wrist สูงกว่าหัว
    features["overhead_margin"] = overhead_margin
    if overhead_margin > OVERHEAD_MARGIN:
        return Classification("SV", _conf(overhead_margin - OVERHEAD_MARGIN, 0.15), features)

    # --- ชั้น 1b: compact swing → VL ---
    pre = slice(0, b3)
    amp = float(np.nanmax(
        np.abs(lm[pre, side["wrist"], 0] - torso[pre, 0]))) / height
    features["backswing_amplitude"] = amp
    if amp < COMPACT_AMPLITUDE:
        return Classification("VL", _conf(COMPACT_AMPLITUDE - amp, 0.2), features)

    if b3 is None:
        return Classification("FH", 0.5, {"note": "no impact detected"})

    # --- ชั้น 2: ฝั่งของ backswing → FH / BH family ---
    ref = b2 if b2 is not None else max(0, b3 - 5)
    side_offset = (float(lm[ref, side["wrist"], 0] - torso[ref, 0])
                   * side["sign"])
    features["backswing_side_offset"] = side_offset
    family = "FH" if side_offset > 0 else "BH"
    side_conf = _conf(side_offset, 0.25 * height)

    # --- ชั้น 3: swing path ลงชัด + BH → SL ---
    if family == "BH" and b2 is not None:
        dy_up = float(lm[b2, side["wrist"], 1] - lm[b3, side["wrist"], 1])
        dx = abs(float(lm[b3, side["wrist"], 0] - lm[b2, side["wrist"], 0])) + 1e-9
        path_deg = float(np.degrees(np.arctan2(dy_up, dx)))
        features["swing_path_deg"] = path_deg
        if path_deg < SLICE_PATH_DEG:
            return Classification(
                "SL", min(side_conf, _conf(SLICE_PATH_DEG - path_deg, 20.0)),
                features)

    return Classification(family, side_conf, features)
