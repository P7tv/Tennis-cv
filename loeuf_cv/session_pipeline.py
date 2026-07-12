"""Phase 2: Full Session Video Inference

pose ทั้ง session ครั้งเดียว → resample → action spotting →
Phase 1 core ต่อ window → C5 post-pass → MT stats + AGG/TRE/PAT/SUM
"""

import numpy as np

from .action_spotting import spot_strokes
from .aggregation import (
    build_aggregated_metric, build_pattern, build_session_summary, build_trend,
)
from .config import PipelineConfig
from .gas_client import post_to_gas
from .pose_extractor import extract_pose
from .resample import resample_to_target_fps
from .schema import build_output, build_session_metadata, default_session_id
from .stroke_pipeline import StrokePipeline


def _apply_c5_normalization(strokes: list[dict]):
    """C5: Phase 2 = C4 ÷ max(C4 ทุก stroke ใน session) × 100"""
    c4s = [s["metrics"]["body"].get("shoulder_rotation_at_impact_deg")
           for s in strokes]
    valid = [abs(v) for v in c4s if v is not None]
    if not valid:
        return
    peak = max(valid) or 1e-9
    for s, c4 in zip(strokes, c4s):
        if c4 is not None:
            s["metrics"]["body"]["shoulder_rotation_normalized_pct"] = round(
                abs(c4) / peak * 100.0, 2)


class SessionPipeline:
    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()
        self.stroke_pipeline = StrokePipeline(self.config)

    def process_video(self, video_path: str,
                      stroke_type_provider=None,
                      send_to_gas: bool = False,
                      progress_callback=None) -> dict:
        """stroke_type_provider: callable(index, window) -> stroke type
        (เช่น label จากลูกค้า) — ไม่ระบุ → ใช้ hierarchical rule
        classifier (ไม่คืน RS — ดู classifier.py)"""
        from .classifier import classify_stroke

        raw = extract_pose(video_path, self.config, progress_callback)

        if self.config.auto_height_from_court and not self.config.height_calibrated:
            from .court_calibration import auto_detect_height_cm
            detected_cm, _court_cal = auto_detect_height_cm(
                video_path, raw, self.config)
            if detected_cm is not None:
                self.config.subject_height_cm = detected_cm

        ts, velocity_normalized = resample_to_target_fps(raw, self.config)
        windows = spot_strokes(ts, self.config)
        session_id = self.config.session_id or default_session_id()

        strokes, rejected = [], 0
        for idx, w in enumerate(windows):
            segment = ts.slice(w.start_frame, w.end_frame)
            classification = None
            if stroke_type_provider:
                stroke_type = stroke_type_provider(idx, w)
            else:
                classification = classify_stroke(segment, self.config)
                stroke_type = classification.stroke_type

            stroke = self.stroke_pipeline.process_timeseries(
                segment, stroke_type,
                stroke_index=len(strokes) + 1,
                session_id=session_id,
                window_offset_ms=w.offset_ms,
                window_offset_frames=w.start_frame,
                pre_resampled=velocity_normalized)

            if classification is not None:
                # debug extension (ระบุใน D0.2): ผล auto-classification
                stroke["stroke_metadata"]["debug_classification"] = \
                    classification.to_debug_dict()
                if classification.confidence < 0.7 and \
                        stroke["stroke_root"]["stroke_recommended_action"] \
                        == "auto_accept":
                    stroke["stroke_root"]["stroke_recommended_action"] = \
                        "coach_review"

            if stroke["stroke_root"]["stroke_recommended_action"] == "discard":
                rejected += 1
                continue
            strokes.append(stroke)

        _apply_c5_normalization(strokes)

        # MT8–MT10 (Phase 2 = populated)
        usable = [s for s in strokes if s["stroke_root"]["is_clean_stroke"]]
        distribution: dict = {}
        for s in strokes:
            st = s["stroke_root"]["stroke_type"]
            distribution[st] = distribution.get(st, 0) + 1
        session_stats = {
            "total_strokes_detected": len(windows),
            "usable_strokes": len(usable),
            "stroke_type_distribution": distribution,
        }

        trend = build_trend(strokes)
        output = build_output(
            build_session_metadata(raw, self.config, session_stats),
            strokes,
            aggregated_metric=build_aggregated_metric(strokes),
            trend=trend,
            pattern=build_pattern(strokes),
            session_summary=build_session_summary(strokes, trend, self.config),
        )
        output["session_metadata"]["_spotting_rejected"] = rejected

        if send_to_gas:
            if not self.config.gas_endpoint_url:
                raise ValueError("gas_endpoint_url not configured")
            output["gas_delivery"] = post_to_gas(
                output, self.config.gas_endpoint_url,
                self.config.gas_timeout_s, self.config.gas_max_retries)
        return output
