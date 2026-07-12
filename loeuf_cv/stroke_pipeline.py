"""Phase 1: Single Stroke Inference

video → pose → resample (fps<60 → 60) → smoothing → keyframes
→ metrics/derived/VF → CF/A → VZ → stroke object → session wrapper
"""

from .calibration import (
    apply_roll_correction, calibration_vq_issues, estimate_calibration,
)
from .config import PipelineConfig
from .confidence import build_confidence_flag, build_stroke_root
from .gas_client import post_to_gas
from .keyframes import detect_keyframes
from .metrics import MetricsEngine
from .occlusion_fill import kinematic_fill
from .one_euro_filter import apply_one_euro_filter
from .pose_extractor import PoseTimeseries, extract_pose
from .resample import resample_to_target_fps
from .schema import (
    build_output, build_session_metadata, build_stroke_metadata,
    build_stroke_object, default_session_id,
)
from .smoothing import smooth_timeseries
from .video_quality import assess_video_quality
from .visualization import build_visualization


class StrokePipeline:
    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()

    def process_timeseries(self, ts: PoseTimeseries, stroke_type: str,
                           stroke_index: int = 1,
                           session_id: str | None = None,
                           window_offset_ms: float = 0.0,
                           window_offset_frames: int = 0,
                           pre_resampled: bool | None = None,
                           ball_observations=None) -> dict:
        """Core: landmark timeseries → stroke object (1 ก้อนใน strokes[])"""
        if pre_resampled is None:
            ts, velocity_normalized = resample_to_target_fps(ts, self.config)
        else:
            velocity_normalized = pre_resampled

        ts = kinematic_fill(ts)
        if self.config.smoothing_method == "one_euro":
            smoothed = apply_one_euro_filter(ts, self.config)
        else:
            smoothed = smooth_timeseries(ts, self.config)

        # camera normalization: roll correction + per-frame scale + zero-ref
        calibration = estimate_calibration(smoothed, self.config)
        smoothed = apply_roll_correction(smoothed, calibration)

        video_quality = assess_video_quality(smoothed, self.config)
        for issue in calibration_vq_issues(calibration):
            if issue not in video_quality["issues"]:
                video_quality["issues"].append(issue)

        keyframes = detect_keyframes(smoothed, self.config, stroke_type)

        if ball_observations is not None:
            from .ball import fuse_impact
            keyframes = fuse_impact(smoothed, keyframes,
                                    ball_observations, self.config)

        metrics, derived, vf = MetricsEngine(
            smoothed, keyframes, self.config, stroke_type,
            calibration=calibration).compute()
        confidence_flag = build_confidence_flag(smoothed, keyframes, self.config)
        stroke_root = build_stroke_root(
            session_id or self.config.session_id or default_session_id(),
            stroke_index, stroke_type,
            start_frame=window_offset_frames,
            end_frame=window_offset_frames + smoothed.n_frames - 1,
            keyframes=keyframes, video_quality=video_quality,
            confidence_flag=confidence_flag, config=self.config)
        visualization = build_visualization(smoothed, keyframes, self.config)

        stroke_metadata = build_stroke_metadata(
            smoothed, self.config, velocity_normalized)
        # debug field (extension — ระบุใน D0.2): ผล camera calibration ต่อคลิป
        stroke_metadata["debug_camera"] = calibration.to_debug_dict()

        stroke = build_stroke_object(
            stroke_metadata, video_quality, keyframes, stroke_root,
            metrics, derived, vf, confidence_flag, visualization,
            window_offset_ms, window_offset_frames)

        if ball_observations is not None:
            from .ball import build_ball_block, build_ball_path
            stroke["metrics"]["ball"] = build_ball_block(ball_observations)
            stroke["visualization"]["ball_path"] = build_ball_path(
                ball_observations, smoothed, keyframes)
        return stroke

    def process_video(self, video_path: str, stroke_type: str,
                      send_to_gas: bool = False,
                      progress_callback=None) -> dict:
        """Phase 1 คลิปเดี่ยว → full JSON (session wrapper, Phase-2 blocks null)"""
        ts = extract_pose(video_path, self.config, progress_callback)

        if self.config.auto_height_from_court and not self.config.height_calibrated:
            from .court_calibration import auto_detect_height_cm
            detected_cm, _court_cal = auto_detect_height_cm(
                video_path, ts, self.config)
            if detected_cm is not None:
                self.config.subject_height_cm = detected_cm

        stroke = self.process_timeseries(ts, stroke_type)

        output = build_output(
            build_session_metadata(ts, self.config), [stroke])

        if send_to_gas:
            if not self.config.gas_endpoint_url:
                raise ValueError("gas_endpoint_url not configured")
            output["gas_delivery"] = post_to_gas(
                output, self.config.gas_endpoint_url,
                self.config.gas_timeout_s, self.config.gas_max_retries)
        return output
