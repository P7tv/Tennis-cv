"""ประกอบ JSON output ตาม cv_schema_table_loeuf (17 layers)

Phase 1: session wrapper + strokes[] (MT8–MT10 = null,
AGG/TRE/PAT/SUM = null)
Phase 2: ครบทุก block
"""

from datetime import datetime, timezone

from .config import (
    CV_MODEL_VERSION, POSE_MODEL, SCHEMA_VERSION, STROKE_TYPES, PipelineConfig,
)
from .keyframes import Keyframe
from .pose_extractor import PoseTimeseries

try:
    import mediapipe as _mp
    POSE_MODEL_VERSION = _mp.__version__
except Exception:
    POSE_MODEL_VERSION = "unknown"


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


def default_session_id() -> str:
    return f"sess-{datetime.now().strftime('%Y%m%d')}-01"


def build_session_metadata(ts: PoseTimeseries, config: PipelineConfig,
                           session_stats: dict | None = None) -> dict:
    """011-MT — session_stats มาจาก Phase 2 (None = Phase 1 → null ตาม doc)"""
    stats = session_stats or {}
    return {
        "cv_model_version": CV_MODEL_VERSION,
        "pose_model": POSE_MODEL,
        "pose_model_version": POSE_MODEL_VERSION,
        "schema_version": SCHEMA_VERSION,
        "session_id": config.session_id or default_session_id(),
        "session_date": _now_iso(),
        "fps": int(round(ts.original_fps)),
        "camera_angle": "single_view",
        "total_duration_sec": round(ts.meta.duration_ms / 1000.0, 2),
        "total_session_frame": ts.meta.frame_count,
        "total_strokes_detected": stats.get("total_strokes_detected"),
        "usable_strokes": stats.get("usable_strokes"),
        "stroke_type_distribution": stats.get("stroke_type_distribution"),
    }


def build_stroke_metadata(ts: PoseTimeseries, config: PipelineConfig,
                          velocity_normalized: bool) -> dict:
    """021-M"""
    import numpy as np
    return {
        "debug_pose_confidence": round(float(np.mean(ts.visibility)), 2),
        "processed_at": _now_iso(),
        "resolution": f"{ts.meta.width}x{ts.meta.height}",
        "camera_angle": "behind_baseline",
        "velocity_normalized": velocity_normalized,
        "normalization_method": ("resampled_to_60fps"
                                 if velocity_normalized else "none"),
    }


def build_stroke_object(stroke_metadata: dict, video_quality: dict,
                        keyframes: dict[str, Keyframe], stroke_root: dict,
                        metrics: dict, derived: dict, visibility_flag: dict,
                        confidence_flag: dict, visualization: dict,
                        window_offset_ms: float = 0.0,
                        window_offset_frames: int = 0) -> dict:
    if stroke_root["stroke_type"] not in STROKE_TYPES:
        raise ValueError(f"Unknown stroke type {stroke_root['stroke_type']!r}")

    kf_out = {}
    for name, kf in keyframes.items():
        d = kf.to_dict()
        if d["frame_index"] is not None and window_offset_frames:
            d["frame_index"] += window_offset_frames
        if d["timestamp_ms"] is not None and window_offset_ms:
            d["timestamp_ms"] = round(d["timestamp_ms"] + window_offset_ms, 2)
        kf_out[name] = d

    return {
        "stroke_metadata": stroke_metadata,       # M
        "video_quality": video_quality,           # VQ
        "keyframes": kf_out,                      # B
        "stroke_root": stroke_root,               # A
        "metrics": metrics,                       # C / KN / SS / BL
        "derived": derived,                       # D
        "visibility_flag": visibility_flag,       # VF (flat, C/KN/SS/D เท่านั้น)
        "confidence_flag": confidence_flag,       # CF
        "visualization": visualization,           # VZ
    }


def build_output(session_metadata: dict, strokes: list[dict],
                 aggregated_metric: list | None = None,
                 trend: dict | None = None,
                 pattern: dict | None = None,
                 session_summary: dict | None = None) -> dict:
    """Full output — Phase 1 ส่ง 4 block ท้ายเป็น null"""
    return {
        "session_metadata": session_metadata,
        "strokes": strokes,
        "aggregated_metric": aggregated_metric,   # AGG (Phase 2)
        "trend": trend,                           # TRE (Phase 2)
        "pattern": pattern,                       # PAT (Phase 2)
        "session_summary": session_summary,       # SUM (Phase 2)
    }
