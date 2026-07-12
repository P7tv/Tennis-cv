"""Phase 2 · 051-054 — AGG / TRE / PAT / SUM

- AGG แยกต่อ stroke_type, aggregate เฉพาะ usable strokes
  (is_clean_stroke = true) ตามกติกาใน doc
- TRE แบ่ง session เป็น thirds เทียบต้น vs ท้าย
- threshold ที่ doc mark "TBC after CV test" อ้าง config/ค่า default
"""

import numpy as np

from .config import STROKE_TYPES, PipelineConfig

TREND_STABLE_PCT = 0.05      # |Δ| < 5% = stable (ตาม doc)
MIN_STROKES_FOR_TREND = 3
MIN_STROKES_FOR_AVOIDANCE = 20


# ---------- helpers ----------

def _metric(stroke: dict, block: str, field: str):
    return stroke["metrics"].get(block, {}).get(field)


def _usable(strokes: list[dict]) -> list[dict]:
    return [s for s in strokes if s["stroke_root"]["is_clean_stroke"]]


def _mean(values):
    vals = [v for v in values if v is not None]
    return round(float(np.mean(vals)), 2) if vals else None


def _sd(values):
    vals = [v for v in values if v is not None]
    return round(float(np.std(vals)), 2) if vals else None


def _thirds(values: list):
    vals = [v for v in values if v is not None]
    n = len(vals)
    if n < MIN_STROKES_FOR_TREND:
        return None
    k = max(1, n // 3)
    return vals[:k], vals[k:n - k] or vals[:k], vals[n - k:]


def _trend(values: list, higher_is_better: bool = True) -> str:
    """improving/declining/stable/volatile — เทียบ mean ต้น vs ท้าย"""
    parts = _thirds(values)
    if parts is None:
        return "stable"
    first, mid, last = (float(np.mean(p)) for p in parts)
    ref = abs(first) + 1e-9
    delta = (last - first) / ref
    mid_delta = (mid - first) / ref
    if abs(delta) < TREND_STABLE_PCT:
        if abs(mid_delta) > 2 * TREND_STABLE_PCT:
            return "volatile"
        return "stable"
    got_better = (delta > 0) == higher_is_better
    return "improving" if got_better else "declining"


# ---------- AGG ----------

def build_aggregated_metric(strokes: list[dict]) -> list[dict]:
    usable = _usable(strokes)
    out = []
    for st in STROKE_TYPES:
        group = [s for s in usable if s["stroke_root"]["stroke_type"] == st]
        if not group:
            continue
        splits = [_metric(s, "movement", "split_step_detected") for s in group]
        split_known = [v for v in splits if v is not None]
        out.append({
            "stroke_type": st,
            "stroke_count": len(group),
            "shoulder_rotation_mean_deg": _mean(
                _metric(s, "body", "shoulder_rotation_at_impact_deg") for s in group),
            "shoulder_rotation_sd_deg": _sd(
                _metric(s, "body", "shoulder_rotation_at_impact_deg") for s in group),
            "hip_rotation_mean_deg": _mean(
                _metric(s, "body", "hip_rotation_at_impact_deg") for s in group),
            "tempo_ratio_mean": _mean(
                _metric(s, "timing", "tempo_ratio") for s in group),
            "tempo_ratio_sd": _sd(
                _metric(s, "timing", "tempo_ratio") for s in group),
            "contact_height_mean_cm": _mean(
                _metric(s, "contact", "contact_height_cm") for s in group),
            "swing_path_angle_mean_deg": _mean(
                _metric(s, "contact", "swing_path_angle_deg") for s in group),
            "racket_head_speed_mean_mps": _mean(
                _metric(s, "kinematics", "racket_head_speed_mps") for s in group),
            "split_step_rate_pct": (
                round(100.0 * sum(bool(v) for v in split_known) / len(split_known), 2)
                if split_known else None),
            "recovery_time_mean_ms": _mean(
                _metric(s, "movement", "recovery_time_ms") for s in group),
        })
    return out


# ---------- TRE ----------

def build_trend(strokes: list[dict]) -> dict:
    usable = _usable(strokes)

    shoulder = [_metric(s, "body", "shoulder_rotation_at_impact_deg")
                for s in usable]
    tempo = [_metric(s, "timing", "tempo_ratio") for s in usable]
    recovery = [_metric(s, "movement", "recovery_time_ms") for s in usable]
    split = [1.0 if _metric(s, "movement", "split_step_detected") else 0.0
             for s in usable
             if _metric(s, "movement", "split_step_detected") is not None]

    trends = {
        "shoulder_rotation_trend": _trend(shoulder, higher_is_better=True),
        "tempo_ratio_trend": _trend(tempo, higher_is_better=True),
        "recovery_time_trend": _trend(recovery, higher_is_better=False),
        "split_step_rate_trend": _trend(split, higher_is_better=True),
    }

    # TRE6 consistency: 1 − mean CV ของ key metrics (normalize หยาบ, TBC)
    consistency = None
    if len(usable) >= MIN_STROKES_FOR_TREND:
        cvs = []
        for series in (shoulder, tempo, recovery):
            vals = [v for v in series if v is not None]
            if len(vals) >= MIN_STROKES_FOR_TREND and abs(np.mean(vals)) > 1e-9:
                cvs.append(np.std(vals) / abs(np.mean(vals)))
        if cvs:
            consistency = round(float(np.clip(1.0 - np.mean(cvs), 0.0, 1.0)), 2)

    declining = sum(1 for v in trends.values() if v == "declining")
    fatigue = declining >= 2

    # TRE8: จุดเริ่มเหนื่อย — simplified เป็น stroke แรกของ third สุดท้าย (TBC)
    onset = None
    if fatigue and len(usable) >= MIN_STROKES_FOR_TREND:
        k = max(1, len(usable) // 3)
        onset = usable[len(usable) - k]["stroke_root"]["stroke_index"]

    return {
        "window_method": "thirds",
        **trends,
        "consistency_score": consistency,
        "fatigue_flag": fatigue,
        "fatigue_onset_stroke_index": onset,
    }


# ---------- PAT ----------

def build_pattern(strokes: list[dict]) -> dict:
    seq = [s["stroke_root"]["stroke_type"]
           for s in sorted(strokes, key=lambda s: s["stroke_root"]["stroke_index"])]
    counts = {t: seq.count(t) for t in STROKE_TYPES if seq.count(t)}

    ground_run = 0
    ground_detected = False
    for t in seq:
        ground_run = ground_run + 1 if t in ("FH", "BH") else 0
        if ground_run >= 3:
            ground_detected = True
            break

    total = len(seq)
    avoidance = None
    if total >= MIN_STROKES_FOR_AVOIDANCE:
        avoidance = (counts.get("BH", 0) / total * 100.0) < 20.0

    return {
        "dominant_stroke_type": max(counts, key=counts.get) if counts else None,
        "stroke_sequence": seq,
        "groundstroke_sequence_detected": ground_detected,
        # PAT4 ต้องการ BL10 (ball detection) → null จนกว่าจะมี
        "serve_fault_rate_pct": None,
        "backhand_avoidance_flag": avoidance,
    }


# ---------- SUM ----------

TREND_FIELD_TAGS = {
    "shoulder_rotation_trend": "shoulder_rotation",
    "tempo_ratio_trend": "tempo_ratio",
    "recovery_time_trend": "recovery_time",
    "split_step_rate_trend": "split_step_rate",
}


def build_session_summary(strokes: list[dict], trend: dict,
                          config: PipelineConfig) -> dict:
    consistency = trend.get("consistency_score")
    fatigue = trend.get("fatigue_flag", False)

    quality = None
    if consistency is not None:
        if consistency < 0.4 or fatigue:
            quality = "poor"
        elif consistency < 0.6:
            quality = "fair"
        elif consistency <= 0.8:
            quality = "good"
        else:
            quality = "excellent" if not fatigue else "good"

    declining = [TREND_FIELD_TAGS[k] for k in TREND_FIELD_TAGS
                 if trend.get(k) == "declining"]
    primary_issue = declining[0] if declining else None

    clean = _usable(strokes)
    highlight = None
    if clean:
        best = max(clean, key=lambda s:
                   s["confidence_flag"]["pose_estimation_confidence"])
        highlight = best["stroke_root"]["stroke_index"]

    low_conf = [s for s in strokes
                if s["confidence_flag"]["pose_estimation_confidence"]
                < config.coach_review_threshold]
    coach_flag = bool(quality == "poor" or fatigue
                      or (strokes and len(low_conf) / len(strokes) > 0.30))

    return {
        "session_quality": quality,
        "primary_issue": primary_issue,
        "highlight_stroke_index": highlight,
        "recommended_drill_tag": primary_issue,
        "coach_flag": coach_flag,
    }
