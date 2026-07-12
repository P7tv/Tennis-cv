"""Pipeline configuration — aligned to cv_schema_table_loeuf (17-layer schema)

ค่าที่ mark TBC = ลูกค้าระบุ "TBC after CV test" ใน schema doc เอง
ต้อง recalibrate กับคลิปจริงใน Phase 0
"""

from dataclasses import dataclass

STROKE_TYPES = ("FH", "BH", "SV", "VL", "SL", "RS")

KEYFRAME_NAMES = (
    "unit_turn",                 # B1
    "backswing_peak",            # B2
    "impact",                    # B3
    "follow_through_peak",       # B4
    "ready_position_restored",   # B5
)

SCHEMA_VERSION = "1.0"
CV_MODEL_VERSION = "0.2.0"
POSE_MODEL = "mediapipe_blazepose"

# MediaPipe Pose landmark indices
NOSE = 0
L_EAR, R_EAR = 7, 8
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_PINKY, R_PINKY = 17, 18
L_INDEX, R_INDEX = 19, 20
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28
L_FOOT, R_FOOT = 31, 32

UPPER_BODY = (L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST)
CORE_LANDMARKS = UPPER_BODY + (L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE)

# น้ำหนักมวลต่อ segment สำหรับ CoM — สูตรจาก schema doc (รวม = 1.00)
COM_SEGMENT_WEIGHTS = {
    "trunk": 0.50,
    "upper_arm": 0.03,   # ต่อข้าง
    "forearm": 0.02,     # ต่อข้าง
    "thigh": 0.10,       # ต่อข้าง
    "shank": 0.10,       # ต่อข้าง
}

NET_HEIGHT_CM = 91.4     # ใช้เทียบ VL2 hands_height_at_contact (calibration TBC)
DEFAULT_BODY_HEIGHT_CM = 170.0


@dataclass
class PipelineConfig:
    # --- pose model ---
    model_complexity: int = 1
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5

    # --- subject / session ---
    dominant_side: str = "right"          # A6
    subject_height_cm: float | None = None  # C1 calibration; None → DEFAULT + low_confidence
    auto_height_from_court: bool = False   # ลอง auto-detect จากเส้นคอร์ทก่อน (court_calibration.py)
    session_id: str | None = None         # MT2; None → auto "sess-YYYYMMDD-01"

    # --- fps normalization (M4/M5): input < 60fps → resample เป็น 60 ---
    target_fps: float = 60.0

    # --- image normalization (ลดความต่างมือถือแต่ละรุ่น/เลนส์) ---
    normalize_frames: bool = True   # gray-world WB + exposure + CLAHE เมื่อมืด

    # --- confidence thresholds (CF1 → A9, ตาม schema doc) ---
    auto_accept_threshold: float = 0.85
    coach_review_threshold: float = 0.60

    # --- visibility flag thresholds ---
    visible_threshold: float = 0.80
    low_confidence_threshold: float = 0.50

    # --- smoothing ---
    smoothing_window_frames: int = 7
    max_gap_fill_frames: int = 5
    smoothing_method: str = "savgol"  # "savgol" (default เดิม) | "one_euro"
    one_euro_mincutoff: float = 1.0   # cutoff ต่ำสุด (ตอนนิ่ง) — ต่ำ = กรอง jitter แรงขึ้น
    one_euro_beta: float = 0.3        # ความไวต่อความเร็ว — สูง = ลด lag ตอนเคลื่อนไหวเร็วมากขึ้น
    one_euro_dcutoff: float = 1.0     # cutoff ของตัวประมาณความเร็ว (derivative)

    # --- keyframe detection ---
    impact_search_margin: float = 0.15
    unit_turn_onset_ratio: float = 0.30
    ready_pose_tolerance: float = 0.12
    ready_min_hold_frames: int = 3
    follow_through_max_ms: float = 1000.0

    # --- TBC thresholds (ลูกค้า mark "TBC after CV test") ---
    split_step_lift_ratio: float = 0.015      # C22: เท้าลอย ≥ 1.5% ของ body height
    split_step_ideal_offset_ms: float = 0.0   # C23 ideal window TBC
    leg_drive_knee_delta_deg: float = 15.0    # SV3: X° TBC
    punch_forward_dx: float = 0.02            # VL3 threshold TBC
    com_shift_neutral_cm: float = 2.0         # D2 forward/neutral/backward TBC
    inference_jump_ratio: float = 0.08        # CF2: landmark jump ต่อเฟรม > 8% height = artifact

    # --- action spotting (Phase 2) ---
    spotting_min_stroke_gap_s: float = 1.2
    spotting_window_before_s: float = 1.5
    spotting_window_after_s: float = 1.5
    spotting_prominence_ratio: float = 0.25

    # --- GAS ---
    gas_endpoint_url: str | None = None
    gas_timeout_s: float = 30.0
    gas_max_retries: int = 3

    @property
    def body_height_cm(self) -> float:
        return self.subject_height_cm or DEFAULT_BODY_HEIGHT_CM

    @property
    def height_calibrated(self) -> bool:
        return self.subject_height_cm is not None

    def recommended_action(self, confidence: float) -> str:
        if confidence >= self.auto_accept_threshold:
            return "auto_accept"
        if confidence >= self.coach_review_threshold:
            return "coach_review"
        return "discard"

    def visibility_flag(self, mean_visibility: float) -> str:
        if mean_visibility >= self.visible_threshold:
            return "visible"
        if mean_visibility >= self.low_confidence_threshold:
            return "low_confidence"
        return "not_detectable"
