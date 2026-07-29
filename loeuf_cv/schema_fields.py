"""Field manifest ตาม client schema doc — "ส่งทุก field เสมอ ห้าม omit key"

โมดูลนี้เป็น single source of truth ของ "รายชื่อ key ที่ต้องมีเสมอ" ของแต่ละ
block เพื่อไม่ให้ literal เดียวกันถูกเขียนซ้ำหลายที่แล้ว drift กัน (เคสเดิม:
loeuf_cv/ball.py::build_ball_block ส่ง 7 key แต่ fallback ใน
schema_builder/builder.py ส่งแค่ {"available": False})

ห้าม import อะไรจาก loeuf_cv นอกจาก .config (กัน circular import — ball.py /
metrics.py / schema_builder/builder.py import โมดูลนี้ทั้งหมด)
"""

from .config import (
    L_ANKLE, L_ELBOW, L_HIP, L_KNEE, L_SHOULDER, L_WRIST, NOSE,
    R_ANKLE, R_ELBOW, R_HIP, R_KNEE, R_SHOULDER, R_WRIST,
)

# ---------- 023-B keyframe ----------

# ทุก keyframe ต้องมีครบ 4 key นี้เสมอ (undetected → null/false ห้าม omit)
KEYFRAME_ENTRY_FIELDS = ("frame_index", "timestamp_ms", "detected", "joint")

# B1-B6 — trophy_position (B6) เป็น serve-only
KEYFRAME_BLOCK_NAMES = (
    "unit_turn",            # B1
    "backswing_peak",       # B2
    "impact",               # B3
    "follow_through_peak",  # B4
    "recovery_position",    # B5
    "trophy_position",      # B6 serve-only
)

# A7 detection_status นับจาก B1-B5 เท่านั้น / A8 is_clean_stroke จาก B1-B4
# ⚠️ ห้ามเติม "trophy_position" ลงสองอันนี้ — จะทำให้ non-serve ทุก stroke
# กลายเป็น partial/ไม่ clean ทันที
A7_KEYFRAME_NAMES = KEYFRAME_BLOCK_NAMES[:5]
A8_KEYFRAME_NAMES = KEYFRAME_BLOCK_NAMES[:4]

# joint = 13 จุด 2 มิติ (normalized image coords, ปัด 2 ตำแหน่ง)
JOINT_NAMES = (
    "head",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
)

JOINT_LANDMARK_IDS = {
    "head": NOSE,
    "left_shoulder": L_SHOULDER, "right_shoulder": R_SHOULDER,
    "left_elbow": L_ELBOW, "right_elbow": R_ELBOW,
    "left_wrist": L_WRIST, "right_wrist": R_WRIST,
    "left_hip": L_HIP, "right_hip": R_HIP,
    "left_knee": L_KNEE, "right_knee": R_KNEE,
    "left_ankle": L_ANKLE, "right_ankle": R_ANKLE,
}

# ---------- 034-BL ball ----------

BALL_FIELDS = (
    "available",                 # BL1 gate
    "ball_speed_kmh",            # BL2
    "trajectory_clearance_cm",   # BL3
    "landing_position",          # BL4
    "landing_zone",              # BL5
    "landing_call",              # BL6
    "server_position",           # BL7  serve-only
    "target_box_correct",        # BL8  serve-only
    "serve_attempt_number",      # BL9  serve-only
    "is_fault",                  # BL10 serve-only
    "_tracked_fraction",         # debug extension (ไม่ใช่ spec field)
)


def empty_ball_block(available: bool = False) -> dict:
    """BL block ที่มี key ครบทุกตัวเสมอ ค่าเริ่มต้น null (available เป็น bool)"""
    block = {name: None for name in BALL_FIELDS}
    block["available"] = available
    return block


# ---------- 043-VZ visualization ----------

VZ_FIELDS = (
    "wrist_path",          # VZ1 REQUIRED
    "ball_path",           # VZ2
    "racket_tip_path",     # VZ3
    "racket_center_path",  # VZ4
    "body_center_path",    # VZ5
)

# ---------- 033-SS stroke_specific ----------

# doc ระบุชัดว่า FH / BH / RS "ไม่มี field เพิ่ม" = {} → tuple ว่างโดยตั้งใจ
STROKE_SPECIFIC_FIELDS = {
    "FH": (),
    "BH": (),
    "RS": (),
    "SV": ("toss_deviation_cm", "trophy_position_achieved",
           "leg_drive_detected", "peak_contact_height_cm", "stance_type"),
    "VL": ("backswing_past_ear", "hands_height_at_contact",
           "punch_forward_detected"),
    "SL": ("swing_direction_verified",),
}
