"""031–035 · Metric blocks ตาม cv_schema_table_loeuf

body / arm / timing / contact / movement / depth_estimated /
kinematics / stroke_specific / ball + derived (D) + flat
visibility_flag (VF)

Conventions ที่เลือกไว้ (จุดที่ spec กำกวม — mark ⚠️ TBC):
- ค่า float ปัดทศนิยม 2 ตำแหน่ง (ตามตาราง type; ประเด็น full-precision
  vs 2dp เป็นคำถามค้างกับลูกค้า)
- C19/C20 คำนวณ ready(B5) − follow_through(B4) — label B4/B5 ใน
  สูตรของ doc สลับกัน implement ตาม intent ให้ค่าเป็นบวก
- C31 ไม่มี racket detector → ใช้ hand index keypoint แทน racket
- KN5/KN6/C40 ต้องการ racket detector → null; KN7 fallback wrist
  ตามที่ doc ระบุ ("ใช้ ≈ wrist_path")
- rotation deg ใช้ world x-z เทียบ baseline; sign validate Phase 0
"""

import numpy as np

from .config import (
    L_ANKLE, L_HIP, L_KNEE, L_SHOULDER, NET_HEIGHT_CM, NOSE,
    R_ANKLE, R_HIP, R_KNEE, R_SHOULDER, PipelineConfig,
)
from .keyframes import Keyframe
from .kinematics import (
    ankle_to_head_norm, body_height_norm, com_xy, dominant, joint_angle,
    rotation_vs_baseline_deg, speed, torso_center,
)
from .pose_extractor import PoseTimeseries
from .schema_fields import empty_ball_block

# D8 contact_timing ideal range (cm หน้าลำตัว) ต่อ stroke — ⚠️ TBC RS/VL/late
CONTACT_IDEAL_CM = {
    "FH": (15.0, 40.0), "BH": (15.0, 40.0), "SL": (15.0, 35.0),
    "RS": (10.0, 30.0), "VL": (15.0, 45.0),
}


def _safe_nanmax(arr: np.ndarray, default: float = np.nan) -> float:
    """np.nanmax แต่คืน default แทน ValueError เมื่อทั้ง slice เป็น NaN
    (occlusion ยาวเกิน gap-fill limit หลุดมาถึงตรงนี้ได้จริงบนคลิปลูกค้า)"""
    arr = np.asarray(arr)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return default
    return float(np.nanmax(arr))


def _safe_nanargmax(arr: np.ndarray) -> int | None:
    arr = np.asarray(arr)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return None
    return int(np.nanargmax(arr))


def _median3_smooth(arr: np.ndarray) -> np.ndarray:
    """median-of-3 (nan-safe) — กัน spike เฟรมเดียวจาก noise ก่อนหา peak
    (เช่น rotation angle จาก atan2 ไวต่อ noise มากตอน dx ใกล้ 0 → position
    noise เล็ก ๆ กลายเป็น angular velocity spike ใหญ่ผิดธรรมชาติได้ทีเดียว)
    ค่า peak ที่เกิดจากการหมุนจริงกินหลายเฟรมต่อเนื่อง ไม่ได้ถูกลดทอนจากนี้"""
    if len(arr) < 3:
        return arr
    out = arr.copy()
    for i in range(1, len(arr) - 1):
        window = arr[i - 1:i + 2]
        if not np.all(np.isnan(window)):
            out[i] = np.nanmedian(window)
    return out


def _r2(v):
    if v is None:
        return None
    if isinstance(v, (np.floating, float)):
        if np.isnan(v):
            return None
        return round(float(v), 2)
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, np.integer):
        return int(v)
    return v


class MetricsEngine:
    def __init__(self, ts: PoseTimeseries, keyframes: dict[str, Keyframe],
                 config: PipelineConfig, stroke_type: str,
                 calibration=None):
        self.ts = ts
        self.kf = keyframes
        self.config = config
        self.stroke = stroke_type
        self.side = dominant(config.dominant_side)
        self.cal = calibration

        self.f = {name: (k.frame_index if k.detected else None)
                  for name, k in keyframes.items()}
        self.t = {name: (k.timestamp_ms if k.detected else None)
                  for name, k in keyframes.items()}

        # px→cm calibration จาก body height (C1)
        self.height_norm = ankle_to_head_norm(ts)
        if calibration is not None:
            self.scale = calibration.scale_per_frame  # per-frame (trunk anchor)
        else:
            self.scale = np.full(ts.n_frames,
                                 config.body_height_cm / self.height_norm)
        self.cm_per_norm = float(np.nanmedian(self.scale))
        self.height_kf_norm = body_height_norm(ts)

        self.com = com_xy(ts)                       # (T, 2)
        self.torso_c = torso_center(ts)             # (T, 2)
        self.shoulder_rot = rotation_vs_baseline_deg(ts, L_SHOULDER, R_SHOULDER)
        self.hip_rot = rotation_vs_baseline_deg(ts, L_HIP, R_HIP)
        if calibration is not None:
            # zero-reference จากท่า ready → กิน camera yaw + ท่ายืนรายบุคคล
            self.shoulder_rot = self.shoulder_rot - calibration.shoulder_zero_deg
            self.hip_rot = self.hip_rot - calibration.hip_zero_deg

        self.vf: dict[str, str] = {}

    # ---------- helpers ----------

    def cm(self, norm_dist: float, frame: int | None = None) -> float:
        s = self.scale[frame] if frame is not None else self.cm_per_norm
        return norm_dist * float(s)

    def _vis_at(self, frame: int | None, ids: tuple,
                cap_low: bool = False) -> str:
        if frame is None:
            return "not_detectable"
        v = float(np.mean(self.ts.visibility[frame, list(ids)]))
        flag = self.config.visibility_flag(v)
        if cap_low and flag == "visible":
            flag = "low_confidence"
        return flag

    def _put(self, block: dict, name: str, value, visibility: str):
        value = _r2(value)
        if value is None and visibility != "derived":
            visibility = "not_detectable"
        block[name] = value
        self.vf[name] = visibility

    def _ankle_mid_x(self, frame: int) -> float:
        lm = self.ts.landmarks
        return float((lm[frame, L_ANKLE, 0] + lm[frame, R_ANKLE, 0]) / 2.0)

    def _wrist(self, frame: int) -> np.ndarray:
        return self.ts.landmarks[frame, self.side["wrist"], :2]

    def _elbow_angle(self, frame: int) -> float:
        wlm = self.ts.world_landmarks[frame]
        return float(joint_angle(wlm[self.side["shoulder"]],
                                 wlm[self.side["elbow"]],
                                 wlm[self.side["wrist"]])[0])

    def _rear_front_ankle(self) -> tuple[int, int]:
        """rear = ankle ใกล้กล้อง (world z ต่ำกว่า) — approximation"""
        wlm = self.ts.world_landmarks
        zl = np.nanmedian(wlm[:, L_ANKLE, 2])
        zr = np.nanmedian(wlm[:, R_ANKLE, 2])
        return (L_ANKLE, R_ANKLE) if zl < zr else (R_ANKLE, L_ANKLE)

    def _leg_load_rear_pct(self, frame: int) -> float:
        """% น้ำหนักขาหลัง ≈ ตำแหน่ง CoM_x ระหว่างเท้า (C37/C38 ⚠️ TBC)"""
        rear, front = self._rear_front_ankle()
        xr = self.ts.landmarks[frame, rear, 0]
        xf = self.ts.landmarks[frame, front, 0]
        if abs(xf - xr) < 1e-6:
            return 50.0
        frac_toward_front = (self.com[frame, 0] - xr) / (xf - xr)
        return float(np.clip(1.0 - frac_toward_front, 0.0, 1.0) * 100.0)

    def _split_step(self) -> tuple[bool | None, float | None]:
        """C22/C23: เท้าสองข้างลอยพร้อมกันช่วง pre-unit_turn window"""
        b1 = self.f.get("unit_turn")
        if b1 is None or b1 < 3:
            return None, None
        lm = self.ts.landmarks
        thresh = self.config.split_step_lift_ratio * self.height_kf_norm
        base_l = np.nanmedian(lm[:b1, L_ANKLE, 1])
        base_r = np.nanmedian(lm[:b1, R_ANKLE, 1])
        lift = ((base_l - lm[:b1, L_ANKLE, 1]) > thresh) & \
               ((base_r - lm[:b1, R_ANKLE, 1]) > thresh)
        idx = np.where(lift)[0]
        if len(idx) < 2:
            return False, None
        apex = int(idx[len(idx) // 2])
        # (-) = ก่อน unit_turn; ideal offset TBC after CV test
        offset = (self.ts.timestamps_ms[apex] - self.t["unit_turn"]
                  - self.config.split_step_ideal_offset_ms)
        return True, float(offset)

    # ---------- blocks ----------

    def body(self) -> dict:
        b: dict = {}
        c1 = self.config.body_height_cm
        c1_vis = "visible" if self.config.height_calibrated else "low_confidence"
        self._put(b, "body_height_cm", c1, c1_vis)
        self._put(b, "arm_span_cm", c1 * 1.0, c1_vis)  # C2 = C1 × 1.0 per doc

        for name, arr, frame_key, ids in (
            ("shoulder_rotation_at_unit_turn_deg", self.shoulder_rot,
             "unit_turn", (L_SHOULDER, R_SHOULDER)),
            ("shoulder_rotation_at_impact_deg", self.shoulder_rot,
             "impact", (L_SHOULDER, R_SHOULDER)),
            ("hip_rotation_at_unit_turn_deg", self.hip_rot,
             "unit_turn", (L_HIP, R_HIP)),
            ("hip_rotation_at_impact_deg", self.hip_rot,
             "impact", (L_HIP, R_HIP)),
        ):
            fr = self.f.get(frame_key)
            val = float(arr[fr]) if fr is not None else None
            if val is not None and self.stroke == "BH":
                val = -val  # ⚠️ BH sign flip ตาม C3/C4 note
            self._put(b, name, val, self._vis_at(fr, ids))

        # C5: Phase 1 = C4 raw (session pipeline คำนวณ % ทับใน Phase 2)
        self._put(b, "shoulder_rotation_normalized_pct",
                  b["shoulder_rotation_at_impact_deg"],
                  self.vf["shoulder_rotation_at_impact_deg"])

        c4, c7 = b["shoulder_rotation_at_impact_deg"], b["hip_rotation_at_impact_deg"]
        sep = (c4 - c7) if (c4 is not None and c7 is not None) else None
        self._put(b, "shoulder_hip_separation_at_impact_deg", sep,
                  self._vis_at(self.f.get("impact"),
                               (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP)))

        # C28 head_still_at_contact: nose movement ±5 เฟรมรอบ B3 < 3% ของ C1
        b3 = self.f.get("impact")
        if b3 is not None:
            lo, hi = max(0, b3 - 5), min(self.ts.n_frames, b3 + 6)
            nose = self.ts.landmarks[lo:hi, NOSE, :2]
            disp = _safe_nanmax(np.linalg.norm(nose - nose[b3 - lo], axis=1))
            if np.isnan(disp):
                self._put(b, "head_still_at_contact", None, "not_detectable")
            else:
                still = bool(self.cm(disp, b3) < 0.03 * self.config.body_height_cm)
                self._put(b, "head_still_at_contact", still,
                          self._vis_at(b3, (NOSE,)))
        else:
            self._put(b, "head_still_at_contact", None, "not_detectable")
        return b

    def arm(self) -> dict:
        a: dict = {}
        s = self.side
        ids = (s["shoulder"], s["elbow"], s["wrist"])

        # C9 follow_through_angle_deg (no SV)
        b4 = self.f.get("follow_through_peak")
        if self.stroke == "SV" or b4 is None:
            self._put(a, "follow_through_angle_deg", None,
                      "not_detectable" if self.stroke != "SV" else "not_detectable")
        else:
            self._put(a, "follow_through_angle_deg", self._elbow_angle(b4),
                      self._vis_at(b4, ids))

        # C10 backswing_depth_deg (no SV): มุมแขนจาก neutral, (-) = ข้างหลัง
        # ⚠️ doc เขียน "shoulder-elbow-wrist" แต่ตัวอย่าง -38.5 เป็นไปไม่ได้
        # กับ joint angle → implement เป็นมุม shoulder→wrist เทียบแนวดิ่ง
        # ติดลบเมื่อ wrist อยู่ฝั่ง backswing — confirm กับลูกค้า
        b2 = self.f.get("backswing_peak")
        if self.stroke == "SV" or b2 is None:
            self._put(a, "backswing_depth_deg", None, "not_detectable")
        else:
            lm = self.ts.landmarks[b2]
            v = lm[s["wrist"], :2] - lm[s["shoulder"], :2]
            ang = float(np.degrees(np.arctan2(abs(v[0]), v[1] + 1e-9)))
            behind = (lm[s["wrist"], 0] - self.torso_c[b2, 0]) * s["sign"] > 0
            self._put(a, "backswing_depth_deg", -ang if behind else ang,
                      self._vis_at(b2, ids))
        return a

    def timing(self) -> dict:
        tm: dict = {}

        def dur(a: str, b: str):
            ta, tb = self.t.get(a), self.t.get(b)
            return (tb - ta) if (ta is not None and tb is not None) else None

        c11 = None if self.stroke == "VL" else dur("unit_turn", "backswing_peak")
        c12 = dur("backswing_peak", "impact")
        c13 = dur("unit_turn", "impact")
        self._put(tm, "backswing_duration_ms", c11,
                  "visible" if c11 is not None else "not_detectable")
        self._put(tm, "forward_swing_duration_ms", c12,
                  "visible" if c12 is not None else "not_detectable")
        self._put(tm, "total_swing_duration_ms", c13,
                  "visible" if c13 is not None else "not_detectable")

        c14 = (c11 / c12) if (c11 and c12 and c12 > 0) else None
        self._put(tm, "tempo_ratio", c14,
                  "visible" if c14 is not None else "not_detectable")

        # C15 rhythm_class: <1.5 rushed / 1.5–3.0 balanced / >3.0 slow (TBC)
        c15 = None
        if c14 is not None:
            c15 = "rushed" if c14 < 1.5 else ("balanced" if c14 <= 3.0 else "slow")
        self._put(tm, "rhythm_class", c15, "derived")
        return tm

    def contact(self) -> dict:
        c: dict = {}
        b3 = self.f.get("impact")
        s = self.side

        if b3 is not None:
            lm = self.ts.landmarks
            ankle_y = (lm[b3, L_ANKLE, 1] + lm[b3, R_ANKLE, 1]) / 2.0
            wrist = self._wrist(b3)
            c16 = self.cm(ankle_y - wrist[1], b3)
            c16_vis = self._vis_at(b3, (s["wrist"],))
            # ความสูงจุดปะทะติดลบไม่มีจริง (ตีบอลใต้พื้นไม่ได้) — เกิดได้จาก noise
            # ตอนวอลเลย์ต่ำมาก (ข้อมือใกล้ระดับข้อเท้าจนติดลบนิดหน่อย) → clamp ที่ 0
            # แต่ลด vis ลง เพราะค่าที่ต้อง clamp แปลว่าใกล้ noise floor ไม่ใช่ค่าที่เชื่อได้เต็มที่
            if c16 < 0:
                c16 = 0.0
                c16_vis = "low_confidence"
            self._put(c, "contact_height_cm", c16, c16_vis)
            # C17: ระยะจาก torso center, (+) = หน้าตัว (ฝั่ง swing)
            lateral = (wrist[0] - self.torso_c[b3, 0]) * s["sign"]
            self._put(c, "contact_distance_from_body_cm", self.cm(lateral, b3),
                      self._vis_at(b3, (s["wrist"],)))
        else:
            self._put(c, "contact_height_cm", None, "not_detectable")
            self._put(c, "contact_distance_from_body_cm", None, "not_detectable")

        # C18 swing_path_angle_deg (no VL/SV): (+) = low to high
        b2 = self.f.get("backswing_peak")
        if self.stroke in ("VL", "SV") or b2 is None or b3 is None:
            self._put(c, "swing_path_angle_deg", None, "not_detectable")
        else:
            w2, w3 = self._wrist(b2), self._wrist(b3)
            dy_up = w2[1] - w3[1]  # image y กลับหัว
            dx = abs(w3[0] - w2[0]) + 1e-9
            self._put(c, "swing_path_angle_deg",
                      float(np.degrees(np.arctan2(dy_up, dx))),
                      self._vis_at(b3, (s["wrist"],)))
        return c

    def movement(self) -> dict:
        m: dict = {}
        b3, b4, b5 = (self.f.get(k) for k in
                      ("impact", "follow_through_peak", "ready_position_restored"))

        # C22/C23 split step (no SV)
        if self.stroke == "SV":
            self._put(m, "split_step_detected", None, "not_detectable")
            self._put(m, "split_step_timing_ms", None, "not_detectable")
        else:
            detected, offset = self._split_step()
            vis = "visible" if detected is not None else "not_detectable"
            self._put(m, "split_step_detected", detected, vis)
            self._put(m, "split_step_timing_ms", offset,
                      "visible" if offset is not None else "not_detectable")

        # C24/C25 footwork (no SV): displacement CoM จากต้นคลิป (ready) → B3
        if self.stroke == "SV" or b3 is None:
            self._put(m, "footwork_distance_cm", None, "not_detectable")
            self._put(m, "movement_speed_mps", None, "not_detectable")
        else:
            dist = float(np.linalg.norm(self.com[b3] - self.com[0]))
            c24 = self.cm(dist)
            dt_s = (self.ts.timestamps_ms[b3] - self.ts.timestamps_ms[0]) / 1000.0
            self._put(m, "footwork_distance_cm", c24, "visible")
            self._put(m, "movement_speed_mps",
                      (c24 / 100.0) / dt_s if dt_s > 0 else None, "visible")

        # C26/C27 stance ที่ B3
        if b3 is not None:
            lm = self.ts.landmarks
            width = abs(lm[b3, R_ANKLE, 0] - lm[b3, L_ANKLE, 0])
            self._put(m, "stance_width_at_impact_cm", self.cm(width, b3),
                      self._vis_at(b3, (L_ANKLE, R_ANKLE)))
            wlm = self.ts.world_landmarks[b3]
            d = wlm[R_ANKLE] - wlm[L_ANKLE]
            ang = float(np.degrees(np.arctan2(abs(d[2]), abs(d[0]) + 1e-9)))
            self._put(m, "stance_angle_deg", ang,
                      self._vis_at(b3, (L_ANKLE, R_ANKLE)))
        else:
            self._put(m, "stance_width_at_impact_cm", None, "not_detectable")
            self._put(m, "stance_angle_deg", None, "not_detectable")

        # C36–C38 weight transfer (⚠️ TBC feasibility ตาม doc)
        b2 = self.f.get("backswing_peak")
        cap = True  # approximation จาก CoM → cap เป็น low_confidence
        if self.stroke == "VL" or b2 is None:
            c37 = c38 = None
        else:
            c37 = self._leg_load_rear_pct(b2)
            c38 = (100.0 - self._leg_load_rear_pct(b3)) if b3 is not None else None
        offset_ms = None
        if b2 is not None and b3 is not None and b3 > b2:
            loads = np.array([self._leg_load_rear_pct(i) for i in range(b2, b3 + 1)])
            below = np.where(loads < 50.0)[0]
            if len(below):
                t_cross = self.ts.timestamps_ms[b2 + int(below[0])]
                offset_ms = float(t_cross - self.t["impact"])
        self._put(m, "weight_transfer_timing_offset_ms", offset_ms,
                  self._vis_at(b3, (L_HIP, R_HIP), cap_low=cap)
                  if offset_ms is not None else "not_detectable")
        self._put(m, "rear_leg_load_pct", c37,
                  self._vis_at(b2, (L_ANKLE, R_ANKLE), cap_low=cap)
                  if c37 is not None else "not_detectable")
        self._put(m, "front_leg_transfer_pct", c38,
                  self._vis_at(b3, (L_ANKLE, R_ANKLE), cap_low=cap)
                  if c38 is not None else "not_detectable")

        # C19/C20 recovery — intent: ready(B5) − follow_through(B4)
        # (สูตรใน doc สลับ label B4/B5)
        if b4 is not None and b5 is not None:
            self._put(m, "recovery_time_ms",
                      self.t["ready_position_restored"]
                      - self.t["follow_through_peak"], "visible")
            self._put(m, "return_to_ready_frames", int(b5 - b4), "visible")
        else:
            self._put(m, "recovery_time_ms", None, "not_detectable")
            self._put(m, "return_to_ready_frames", None, "not_detectable")

        # C21 recovery_position: B5 undetected → not_recovered (data ไม่ใช่ error)
        if b5 is None:
            self._put(m, "recovery_position", "not_recovered", "visible")
        else:
            x = self.com[b5, 0]
            zone = ("baseline_left" if x < 0.4
                    else "baseline_right" if x > 0.6 else "baseline_center")
            # "net" ต้องการ court-position calibration — Phase 0 question
            self._put(m, "recovery_position", zone, "visible")
        return m

    def depth_estimated(self) -> dict:
        """C29–C41: Limited from back view — visibility cap ที่ low_confidence"""
        d: dict = {}
        s = self.side
        b2, b3, b4 = (self.f.get(k) for k in
                      ("backswing_peak", "impact", "follow_through_peak"))
        arm_ids = (s["shoulder"], s["elbow"], s["wrist"])

        self._put(d, "elbow_angle_at_impact_deg",
                  self._elbow_angle(b3) if b3 is not None else None,
                  self._vis_at(b3, arm_ids, cap_low=True))
        self._put(d, "elbow_angle_at_backswing_peak_deg",
                  self._elbow_angle(b2) if b2 is not None else None,
                  self._vis_at(b2, arm_ids, cap_low=True))

        # C31: elbow-wrist-racket → ไม่มี racket ใช้ hand index keypoint (TBC)
        if b3 is not None:
            wlm = self.ts.world_landmarks[b3]
            c31 = float(joint_angle(wlm[s["elbow"]], wlm[s["wrist"]],
                                    wlm[s["index"]])[0])
        else:
            c31 = None
        self._put(d, "wrist_angle_at_impact_deg", c31,
                  self._vis_at(b3, (s["wrist"], s["index"]), cap_low=True))

        # C32 spine tilt vs vertical, (+) = เอนหลัง (sign validate Phase 0)
        if b3 is not None:
            lm = self.ts.landmarks[b3]
            shoulder_mid = (lm[L_SHOULDER, :2] + lm[R_SHOULDER, :2]) / 2.0
            hip_mid = (lm[L_HIP, :2] + lm[R_HIP, :2]) / 2.0
            v = shoulder_mid - hip_mid
            tilt = float(np.degrees(np.arctan2(v[0], -v[1] + 1e-9)))
            wz = self.ts.world_landmarks[b3]
            z_lean = (wz[L_SHOULDER, 2] + wz[R_SHOULDER, 2]) / 2.0 \
                - (wz[L_HIP, 2] + wz[R_HIP, 2]) / 2.0
            c32 = abs(tilt) * (1.0 if z_lean < 0 else -1.0)
        else:
            c32 = None
        self._put(d, "spine_tilt_at_impact_deg", c32,
                  self._vis_at(b3, (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP),
                               cap_low=True))

        # C33–C35 CoM
        com_ids = (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP, L_KNEE, R_KNEE)
        c33 = (self.cm(float(np.linalg.norm(self.com[b3] - self.com[b2])), b3)
               if (b2 is not None and b3 is not None) else None)
        self._put(d, "com_shift_distance_cm", c33,
                  self._vis_at(b3, com_ids, cap_low=True)
                  if c33 is not None else "not_detectable")

        for name, frame in (("com_deviation_at_impact_cm", b3),
                            ("com_deviation_at_finish_cm", b4)):
            if frame is not None:
                dev = self.cm(abs(self.com[frame, 0] - self._ankle_mid_x(frame)), frame)
                self._put(d, name, dev,
                          self._vis_at(frame, (L_ANKLE, R_ANKLE), cap_low=True))
            else:
                self._put(d, name, None, "not_detectable")

        # C39/C40 requires side view / racket detector
        self._put(d, "pronation_detected", None, "not_detectable")
        self._put(d, "racket_face_open_deg", None, "not_detectable")

        # C41 peak_racket_drop_cm (no SL/VL/RS) — wrist proxy, low_confidence
        if self.stroke in ("SL", "VL", "RS") or b2 is None or b3 is None:
            self._put(d, "peak_racket_drop_cm", None, "not_detectable")
        else:
            ys = self.ts.landmarks[b2:b3 + 1, s["wrist"], 1]
            peak_y = _safe_nanmax(ys)
            if np.isnan(peak_y) or np.isnan(ys[0]):
                self._put(d, "peak_racket_drop_cm", None, "not_detectable")
            else:
                drop = max(0.0, float(peak_y - ys[0]))
                self._put(d, "peak_racket_drop_cm", self.cm(drop, b3),
                          self._vis_at(b3, (s["wrist"],), cap_low=True))
        return d

    def kinematics_block(self) -> dict:
        kn: dict = {}
        b2, b3 = self.f.get("backswing_peak"), self.f.get("impact")
        low_fps = self.ts.original_fps < 60.0  # resample แล้วแต่ fidelity จำกัด
        t_ms = self.ts.timestamps_ms

        def peak_rot_velocity(rot: np.ndarray):
            if b2 is None or b3 is None or b3 <= b2:
                return None, None
            seg = rot[b2:b3 + 1]
            dt = np.gradient(t_ms[b2:b3 + 1]) / 1000.0
            vel = np.abs(np.gradient(seg) / dt)
            vel = _median3_smooth(vel)
            i = _safe_nanargmax(vel)
            if i is None:
                return None, None
            return float(vel[i]), float(t_ms[b2 + i])

        kn1, t_hip = peak_rot_velocity(self.hip_rot)
        kn2, t_shoulder = peak_rot_velocity(self.shoulder_rot)
        vis_kn = ("low_confidence" if low_fps else "visible")
        self._put(kn, "hip_rotation_velocity_deg_s", kn1,
                  vis_kn if kn1 is not None else "not_detectable")
        self._put(kn, "shoulder_rotation_velocity_deg_s", kn2,
                  vis_kn if kn2 is not None else "not_detectable")

        # KN3: (+) = hip นำก่อน shoulder (kinetic chain) — ⚠️ สูตรใน doc
        # ให้เครื่องหมายกลับด้าน implement ตาม intent
        kn3 = (t_shoulder - t_hip) if (t_hip is not None and t_shoulder is not None) else None
        self._put(kn, "hip_shoulder_velocity_lag_ms", kn3,
                  vis_kn if kn3 is not None else "not_detectable")

        # KN4 CoM velocity ณ impact
        if b3 is not None and b3 > 0:
            dt = (t_ms[b3] - t_ms[b3 - 1]) / 1000.0
            dist_cm = self.cm(float(np.linalg.norm(self.com[b3] - self.com[b3 - 1])), b3)
            kn4 = (dist_cm / 100.0) / dt if dt > 0 else None
        else:
            kn4 = None
        self._put(kn, "center_of_mass_velocity_mps", kn4,
                  "visible" if kn4 is not None else "not_detectable")

        # KN5/KN6 ต้องการ racket detector → null
        self._put(kn, "racket_tip_position", None, "not_detectable")
        self._put(kn, "racket_center_position", None, "not_detectable")

        # KN7 fallback: "ถ้าไม่มี racket detector → ใช้ ≈ wrist_path"
        if b2 is not None and b3 is not None and b3 > b2:
            ws = speed(np.nan_to_num(
                self.ts.landmarks[:, self.side["wrist"], :2], nan=0.0), t_ms)
            peak = float(np.nanmax(ws[b2:b3 + 1]))
            kn7 = self.cm(peak, b3) / 100.0
        else:
            kn7 = None
        self._put(kn, "racket_head_speed_mps", kn7,
                  "low_confidence" if kn7 is not None else "not_detectable")
        return kn

    def stroke_specific(self) -> dict:
        ss: dict = {}
        s = self.side
        lm = self.ts.landmarks
        b2, b3, b4 = (self.f.get(k) for k in
                      ("backswing_peak", "impact", "follow_through_peak"))

        if self.stroke == "SV":
            # SV1 ต้องการ ball detection
            self._put(ss, "toss_deviation_cm", None, "not_detectable")

            # SV2 trophy: elbow ≥ shoulder, upper arm ±20° จากแนวนอน, wrist ≥ shoulder
            if b2 is not None:
                sh, el, wr = (lm[b2, s[k], :2] for k in ("shoulder", "elbow", "wrist"))
                v = el - sh
                upper_ang = abs(np.degrees(np.arctan2(-v[1], abs(v[0]) + 1e-9)))
                trophy = bool(el[1] <= sh[1] and upper_ang <= 20.0 and wr[1] <= sh[1])
                self._put(ss, "trophy_position_achieved", trophy,
                          self._vis_at(b2, (s["shoulder"], s["elbow"], s["wrist"])))
            else:
                self._put(ss, "trophy_position_achieved", None, "not_detectable")

            # SV3 leg drive: knee เหยียด ≥ X° (TBC) + CoM สูงขึ้น B2→B3
            if b2 is not None and b3 is not None:
                wlm = self.ts.world_landmarks
                knee = joint_angle(wlm[:, R_HIP], wlm[:, R_KNEE], wlm[:, R_ANKLE])
                extend = (knee[b3] - knee[b2]) >= self.config.leg_drive_knee_delta_deg
                com_up = self.com[b3, 1] < self.com[b2, 1]
                self._put(ss, "leg_drive_detected", bool(extend and com_up),
                          self._vis_at(b3, (R_HIP, R_KNEE, R_ANKLE)))
            else:
                self._put(ss, "leg_drive_detected", None, "not_detectable")

            # SV4 peak contact height — ติดลบไม่มีจริง เหตุผลเดียวกับ C16 (contact())
            if b3 is not None:
                ankle_y = (lm[b3, L_ANKLE, 1] + lm[b3, R_ANKLE, 1]) / 2.0
                sv4 = self.cm(ankle_y - lm[b3, s["wrist"], 1], b3)
                sv4_vis = self._vis_at(b3, (s["wrist"],))
                if sv4 < 0:
                    sv4 = 0.0
                    sv4_vis = "low_confidence"
                self._put(ss, "peak_contact_height_cm", sv4, sv4_vis)
            else:
                self._put(ss, "peak_contact_height_cm", None, "not_detectable")

            # SV5 stance_type: เท้าหลังชิดเท้าหน้าก่อน impact → pinpoint
            if b3 is not None:
                rear, front = self._rear_front_ankle()
                gap0 = abs(lm[0, rear, 0] - lm[0, front, 0])
                gap_b3 = abs(lm[b3, rear, 0] - lm[b3, front, 0])
                stype = "pinpoint" if gap_b3 < 0.5 * gap0 else "platform"
                self._put(ss, "stance_type", stype,
                          self._vis_at(b3, (L_ANKLE, R_ANKLE)))
            else:
                self._put(ss, "stance_type", None, "not_detectable")

        elif self.stroke == "VL":
            # VL1 backswing เลยใบหู (Red Flag)
            if b2 is not None:
                past = (lm[b2, s["wrist"], 0] - lm[b2, s["ear"], 0]) * s["sign"] > 0
                self._put(ss, "backswing_past_ear", bool(past),
                          self._vis_at(b2, (s["wrist"], s["ear"])))
            else:
                self._put(ss, "backswing_past_ear", None, "not_detectable")

            # VL2 มือเทียบระดับเน็ต (net = 91.4cm, calibration TBC)
            if b3 is not None:
                ankle_y = (lm[b3, L_ANKLE, 1] + lm[b3, R_ANKLE, 1]) / 2.0
                h_cm = self.cm(ankle_y - lm[b3, s["wrist"], 1], b3)
                level = ("above_net" if h_cm > NET_HEIGHT_CM + 10
                         else "below_net" if h_cm < NET_HEIGHT_CM - 10 else "at_net")
                self._put(ss, "hands_height_at_contact", level,
                          self._vis_at(b3, (s["wrist"],)))
            else:
                self._put(ss, "hands_height_at_contact", None, "not_detectable")

            # VL3 punch forward: Δx wrist B3→B4 ฝั่ง contact > threshold (TBC)
            if b3 is not None and b4 is not None:
                dx = (lm[b4, s["wrist"], 0] - lm[b3, s["wrist"], 0]) * -s["sign"]
                self._put(ss, "punch_forward_detected",
                          bool(dx > self.config.punch_forward_dx),
                          self._vis_at(b4, (s["wrist"],)))
            else:
                self._put(ss, "punch_forward_detected", None, "not_detectable")

        elif self.stroke == "SL":
            # SL1 ทิศ swing จาก y ของ wrist B2 vs B3
            if b2 is not None and b3 is not None:
                dy_up = lm[b2, s["wrist"], 1] - lm[b3, s["wrist"], 1]
                band = 0.02 * self.height_kf_norm
                direction = ("low_to_high" if dy_up > band
                             else "high_to_low" if dy_up < -band else "flat")
                self._put(ss, "swing_direction_verified", direction,
                          self._vis_at(b3, (s["wrist"],)))
            else:
                self._put(ss, "swing_direction_verified", None, "not_detectable")

        # FH / BH / RS → {} ตาม doc
        return ss

    def ball(self) -> dict:
        # BL1 gate: ไม่มี ball detection → available false แต่ key ต้องครบทุกตัว
        # (spec: ห้าม omit key) — ดู schema_fields.BALL_FIELDS
        return empty_ball_block()

    def derived(self, blocks: dict) -> dict:
        d: dict = {}
        cfg = self.config

        def put(name, value):
            self._put(d, name, value, "derived")

        c18 = blocks["contact"].get("swing_path_angle_deg")
        put("swing_path_direction",
            None if c18 is None else
            ("low_to_high" if c18 > 15 else "high_to_low" if c18 < -15 else "flat"))

        c33 = blocks["depth_estimated"].get("com_shift_distance_cm")
        # D2 ทิศทางจาก CoM y (ขึ้น = forward approx) — ⚠️ TBC threshold
        if c33 is None:
            put("center_of_mass_shift", None)
        elif c33 < cfg.com_shift_neutral_cm:
            put("center_of_mass_shift", "neutral")
        else:
            b2, b3 = self.f.get("backswing_peak"), self.f.get("impact")
            fwd = (b2 is not None and b3 is not None
                   and self.com[b3, 1] <= self.com[b2, 1])
            put("center_of_mass_shift", "forward" if fwd else "backward")

        for name, key in (("balance_at_impact", "com_deviation_at_impact_cm"),
                          ("balance_at_finish", "com_deviation_at_finish_cm")):
            v = blocks["depth_estimated"].get(key)
            put(name, None if v is None else
                ("stable" if v < 5 else "slight_loss" if v <= 15 else "clear_loss"))

        c27 = blocks["movement"].get("stance_angle_deg")
        put("stance_type", None if c27 is None else
            ("closed" if c27 < 30 else "semi_open" if c27 < 60 else "open"))

        c36 = blocks["movement"].get("weight_transfer_timing_offset_ms")
        put("weight_transfer_timing",
            "none" if c36 is None else
            ("pre_impact" if c36 < -50 else "post_impact" if c36 > 50 else "at_impact"))

        c37 = blocks["movement"].get("rear_leg_load_pct")
        c38 = blocks["movement"].get("front_leg_transfer_pct")
        if c37 is None or c38 is None:
            put("weight_transfer_direction", None)
        elif abs(c38 - c37) < 5:
            put("weight_transfer_direction", "neutral")
        else:
            put("weight_transfer_direction", "forward" if c38 > c37 else "backward")

        # D8 contact_timing ต่อ stroke (SV ใช้ arm extension แทน — TBC)
        c17 = blocks["contact"].get("contact_distance_from_body_cm")
        if self.stroke == "SV":
            c29 = blocks["depth_estimated"].get("elbow_angle_at_impact_deg")
            put("contact_timing",
                None if c29 is None else ("ideal" if c29 >= 160 else "late"))
        elif c17 is None:
            put("contact_timing", None)
        else:
            lo, hi = CONTACT_IDEAL_CM[self.stroke]
            put("contact_timing",
                "ideal" if lo <= c17 <= hi else ("early" if c17 < lo else "late"))

        c29 = blocks["depth_estimated"].get("elbow_angle_at_impact_deg")
        put("arm_extension_at_impact", None if c29 is None else
            ("full" if c29 >= 160 else "partial" if c29 >= 120 else "bent"))
        return d

    # ---------- entry ----------

    def compute(self) -> tuple[dict, dict, dict]:
        """คืน (metrics, derived, visibility_flag)"""
        blocks = {
            "body": self.body(),
            "arm": self.arm(),
            "timing": self.timing(),
            "contact": self.contact(),
            "movement": self.movement(),
            "depth_estimated": self.depth_estimated(),
            "kinematics": self.kinematics_block(),
            "stroke_specific": self.stroke_specific(),
            "ball": self.ball(),
        }
        derived = self.derived(blocks)
        return blocks, derived, dict(self.vf)
