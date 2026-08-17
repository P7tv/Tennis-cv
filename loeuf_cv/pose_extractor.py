"""Video → landmark timeseries via MediaPipe BlazePose

เก็บ frame stats (luma / blur / bbox) ไประหว่างอ่านเฟรม เพื่อให้
video_quality.py ประเมิน VQ block ได้โดยไม่ต้องอ่านวิดีโอซ้ำ

ทุกอย่างหลังไฟล์นี้ทำงานบน numpy timeseries ล้วน ๆ — logic ชั้นบน
ทดสอบได้โดยไม่ต้องมีวิดีโอ/mediapipe (ดู tests/)
"""

from dataclasses import dataclass, field

import cv2
import numpy as np

from .config import PipelineConfig


@dataclass
class VideoMeta:
    path: str
    fps: float
    frame_count: int
    width: int
    height: int

    @property
    def duration_ms(self) -> float:
        return self.frame_count / self.fps * 1000.0


@dataclass
class FrameStats:
    """สถิติภาพต่อเฟรมสำหรับ VQ pre-check (None = ไม่มีข้อมูล เช่น synthetic)"""
    mean_luma: np.ndarray | None = None       # (T,) 0-255
    blur_score: np.ndarray | None = None      # (T,) Laplacian variance
    person_height_frac: np.ndarray | None = None  # (T,) bbox height / frame height


@dataclass
class PoseTimeseries:
    """Landmark timeseries ของคลิปหนึ่ง

    landmarks:       (T, 33, 3) normalized image coords
    world_landmarks: (T, 33, 3) meters, hip-center origin (pseudo-3D)
    visibility:      (T, 33)
    timestamps_ms:   (T,) — ทุกการคำนวณอิงเวลา ไม่อิง frame index
    """

    landmarks: np.ndarray
    world_landmarks: np.ndarray
    visibility: np.ndarray
    timestamps_ms: np.ndarray
    meta: VideoMeta
    frame_stats: FrameStats = field(default_factory=FrameStats)

    # ตั้งโดย resample.py: fps ที่ใช้คำนวณจริง vs fps ต้นทาง
    effective_fps: float | None = None
    source_fps: float | None = None

    @property
    def n_frames(self) -> int:
        return len(self.timestamps_ms)

    @property
    def fps(self) -> float:
        return self.effective_fps or self.meta.fps

    @property
    def original_fps(self) -> float:
        return self.source_fps or self.meta.fps

    def slice(self, start: int, end: int) -> "PoseTimeseries":
        out = PoseTimeseries(
            landmarks=self.landmarks[start:end],
            world_landmarks=self.world_landmarks[start:end],
            visibility=self.visibility[start:end],
            timestamps_ms=self.timestamps_ms[start:end],
            meta=self.meta,
            frame_stats=self.frame_stats,
        )
        out.effective_fps = self.effective_fps
        out.source_fps = self.source_fps
        return out


@dataclass
class PlayerTrack:
    track_id: int
    role: str
    pose: PoseTimeseries
    n_frames_tracked: int

CROP_MARGIN_FRAC = 0.4
UPSCALE_TARGET_PX = 700

def crop_landmarks_to_frame(norm_in_crop: np.ndarray, crop_box: tuple,
                            frame_w: int, frame_h: int) -> np.ndarray:
    x0, y0, cw, ch = crop_box
    out = norm_in_crop.copy()
    out[:, 0] = (out[:, 0] * cw + x0) / frame_w
    out[:, 1] = (out[:, 1] * ch + y0) / frame_h
    return out


def _crop_and_run_pose(pose_model, frame: np.ndarray, box: tuple):
    fh, fw = frame.shape[:2]
    x, y, w, h = box
    mx, my = int(w * CROP_MARGIN_FRAC), int(h * CROP_MARGIN_FRAC)
    x0, y0 = max(0, x - mx), max(0, y - my)
    x1, y1 = min(fw, x + w + mx), min(fh, y + h + my)
    crop = frame[y0:y1, x0:x1]
    if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
        return None, None
    scale = max(1.0, UPSCALE_TARGET_PX / crop.shape[0])
    crop_big = cv2.resize(crop, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_CUBIC)
    result = pose_model.process(cv2.cvtColor(crop_big, cv2.COLOR_BGR2RGB))
    return result, (x0, y0, x1 - x0, y1 - y0)

def read_video_meta(path: str) -> VideoMeta:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    meta = VideoMeta(
        path=path,
        fps=cap.get(cv2.CAP_PROP_FPS) or 30.0,
        frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    cap.release()
    return meta


def extract_pose(path: str, config: PipelineConfig,
                 progress_callback=None) -> PoseTimeseries:
    """รัน BlazePose ทั้งคลิป → PoseTimeseries + FrameStats

    เฟรมที่ detect ไม่ได้เลย → NaN landmarks + visibility 0
    """
    import mediapipe as mp

    meta = read_video_meta(path)
    cap = cv2.VideoCapture(path)

    lm_list, wlm_list, vis_list, ts_list = [], [], [], []
    luma_list, blur_list, height_list = [], [], []

    with mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=config.model_complexity,
        smooth_landmarks=True,
        min_detection_confidence=config.min_detection_confidence,
        min_tracking_confidence=config.min_tracking_confidence,
    ) as pose:
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # VQ stats วัดจากเฟรมดิบ (ก่อน normalize) — รายงานคุณภาพตามจริง
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            raw_luma = float(gray.mean())
            luma_list.append(raw_luma)
            blur_list.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))

            if config.normalize_frames:
                from .image_norm import normalize_frame
                frame = normalize_frame(frame, raw_luma)

            result = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if result.pose_landmarks:
                lm = np.array([[p.x, p.y, p.z] for p in result.pose_landmarks.landmark])
                vis = np.array([p.visibility for p in result.pose_landmarks.landmark])
                wlm = np.array(
                    [[p.x, p.y, p.z] for p in result.pose_world_landmarks.landmark])
                ys = lm[:, 1]
                height_list.append(float(np.nanmax(ys) - np.nanmin(ys)))
            else:
                lm = np.full((33, 3), np.nan)
                wlm = np.full((33, 3), np.nan)
                vis = np.zeros(33)
                height_list.append(0.0)

            lm_list.append(lm)
            wlm_list.append(wlm)
            vis_list.append(vis)
            ts_list.append(frame_idx / meta.fps * 1000.0)

            if progress_callback:
                progress_callback(frame_idx, meta.frame_count)
            frame_idx += 1

    cap.release()
    if not lm_list:
        raise ValueError(f"No frames decoded from video: {path}")

    return PoseTimeseries(
        landmarks=np.stack(lm_list),
        world_landmarks=np.stack(wlm_list),
        visibility=np.stack(vis_list),
        timestamps_ms=np.array(ts_list),
        meta=meta,
        frame_stats=FrameStats(
            mean_luma=np.array(luma_list),
            blur_score=np.array(blur_list),
            person_height_frac=np.array(height_list),
        ),
    )
