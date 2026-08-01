import dataclasses
import datetime
import numpy as np
from .keyframes import (extract_keyframes, refine_keyframes_for_type,
                        trophy_position_frame)
from .metrics import calculate_cm_per_px, get_racket_metrics, _pick_racket_detection, _racket_tip_center_px
from .classifier import classify_stroke_with_confidence
from .aggregator import build_phase2_aggregations
from ..ball import BallObservations, build_ball_block
from ..config import CORE_LANDMARKS, STROKE_TYPES
from ..keyframes import Keyframe
from ..kinematics import com_xy
from ..metrics import MetricsEngine
from ..occlusion_fill import kinematic_fill
from ..pose_extractor import FrameStats
from ..schema_fields import (
    A7_KEYFRAME_NAMES, A8_KEYFRAME_NAMES, JOINT_LANDMARK_IDS, JOINT_NAMES,
    VZ_FIELDS, empty_ball_block,
)
from ..smoothing import smooth_timeseries
from ..video_quality import assess_video_quality

# schema_builder เดิมตั้งชื่อ B5 "recovery_position" (ตรงกับ schema doc field name
# ในบล็อก 023-B) แต่ loeuf_cv/config.py (KEYFRAME_NAMES, ใช้โดย MetricsEngine)
# ตั้งชื่อ "ready_position_restored" — คนละชื่อ ความหมายเดียวกัน
_KF_NAME_TO_ENGINE = {"recovery_position": "ready_position_restored"}


def _build_engine_keyframes(kf: dict, fps: float, stroke_start_frame: int) -> dict[str, Keyframe]:
    """schema_builder.keyframes.extract_keyframes() คืน dict ชื่อ -> frame_index (int|None)
    ธรรมดา — MetricsEngine (loeuf_cv/metrics.py, ของเดิมที่ทดสอบแล้ว 63 tests) ต้องการ
    dict ชื่อ -> Keyframe(frame_index, timestamp_ms, detected) โดย frame_index ต้อง
    relative กับ PoseTimeseries ที่ตัดมาเฉพาะ stroke นี้ (ดู _slice_pose_for_stroke)"""
    out = {}
    for name, frame_idx in kf.items():
        engine_name = _KF_NAME_TO_ENGINE.get(name, name)
        detected = frame_idx is not None
        rel_idx = (frame_idx - stroke_start_frame) if detected else None
        ts_ms = (rel_idx / fps * 1000.0) if detected else None
        out[engine_name] = Keyframe(engine_name, rel_idx, ts_ms, detected)
    return out


def _slice_pose_for_stroke(pose, start_frame: int, end_frame: int, config):
    """MetricsEngine ออกแบบมาสำหรับ PoseTimeseries ของ stroke เดียว (บาง field เช่น
    split_step/footwork_distance ใช้ 'เฟรม 0 ของ ts' เป็นจุดอ้างอิงก่อน stroke) —
    track.pose ในระบบนี้ครอบคลุมทั้งคลิป (หลาย stroke) ต้องตัดมาเฉพาะช่วงของ stroke
    นี้ก่อนส่งเข้า MetricsEngine ไม่งั้นค่าที่คำนวณจะอ้างอิงผิดช่วง

    track.pose มาจาก YOLO+MediaPipe ดิบ (ไม่ผ่าน resample/gap-fill/smoothing
    แบบที่ StrokePipeline เดิมทำก่อนส่งเข้า MetricsEngine — ดู stroke_pipeline.py)
    ช่วงที่ track หลุด/occluded ทำให้ landmark เป็น NaN ต่อเนื่องยาว ๆ ได้ ซึ่งทำให้
    MetricsEngine (เช่น kinematics_block ที่ทำ nanargmax) crash ได้ตรง ๆ — จึงต้อง
    kinematic_fill (เติม wrist/ankle เฉพาะจุด) + smooth_timeseries (gap-fill ทุก
    landmark สั้น ๆ + savgol) ก่อนเสมอ เหมือน pipeline เดิมทำ"""
    n = len(pose.landmarks)
    s = max(0, min(start_frame, n - 1))
    e = max(s + 1, min(end_frame + 1, n))
    sliced = dataclasses.replace(
        pose,
        landmarks=pose.landmarks[s:e],
        world_landmarks=pose.world_landmarks[s:e],
        visibility=pose.visibility[s:e],
        timestamps_ms=pose.timestamps_ms[s:e],
    )
    filled = kinematic_fill(sliced)
    smoothed = smooth_timeseries(filled, config)
    # smooth_timeseries() (loeuf_cv/smoothing.py) สร้าง PoseTimeseries ใหม่โดยไม่
    # copy frame_stats มาด้วย (ไม่ใช่ field ที่มันแตะ) → ต้องตัดแปะ frame_stats ของ
    # ช่วง stroke นี้เองทีหลัง ไม่งั้น VQ (assess_video_quality) จะได้ None ทุก field
    def _slice_stat(arr):
        return arr[s:e] if arr is not None else None

    fs = pose.frame_stats
    return dataclasses.replace(smoothed, frame_stats=FrameStats(
        mean_luma=_slice_stat(fs.mean_luma),
        blur_score=_slice_stat(fs.blur_score),
        person_height_frac=_slice_stat(fs.person_height_frac),
    ))


def _pose_confidence(pose, frame_idx: int) -> float:
    """CF1 pose_estimation_confidence — ค่าเฉลี่ย visibility ของ core landmark
    ณ impact frame (ประมาณการรวมผล lighting/occlusion/motion blur/pose quality
    ที่ landmark visibility ของ MediaPipe สะท้อนอยู่แล้วในตัวมันเอง)"""
    if frame_idx is None or frame_idx >= len(pose.visibility):
        return 0.0
    return float(np.mean(pose.visibility[frame_idx, list(CORE_LANDMARKS)]))


def _inference_clean(pose, start_frame: int, end_frame: int, config) -> bool:
    """CF2 — เช็ค landmark jump ต่อเฟรมของ core landmark ตลอดช่วง stroke นี้ ถ้ากระโดด
    เกิน inference_jump_ratio ของ body height (ในเฟรมเดียว) ถือว่าน่าจะเป็น artifact
    (skeleton jump / landmark หายกลางคัน) ไม่ใช่ track จริง"""
    n = len(pose.landmarks)
    s, e = max(0, start_frame), min(n, end_frame + 1)
    if e - s < 2:
        return True
    seg = pose.landmarks[s:e, list(CORE_LANDMARKS), :2]
    ankle_y = np.nanmean(pose.landmarks[s:e, [27, 28], 1])
    head_y = np.nanmean(pose.landmarks[s:e, 0, 1])
    height_norm = abs(ankle_y - head_y)
    if not np.isfinite(height_norm) or height_norm < 1e-6:
        return True
    jumps = np.linalg.norm(np.diff(seg, axis=0), axis=2)
    max_jump = np.nanmax(jumps) if jumps.size and not np.all(np.isnan(jumps)) else 0.0
    return bool(max_jump < config.inference_jump_ratio * height_norm)


def _pose_model_version() -> str:
    """MT0-3 — เวอร์ชันจริงของ mediapipe ที่รันอยู่ (spec: value-example 0.10.3)

    mediapipe เป็น optional dependency ของ schema_builder (ทั้งโมดูลนี้ไม่ได้
    import mediapipe เลย — pose มาจาก track ที่คำนวณไว้แล้ว) จึงห้าม import
    ระดับโมดูล ถ้าไม่มีให้คืน "unknown" ไม่ใช่ปล่อย ImportError
    """
    try:
        from importlib.metadata import version
        return version("mediapipe")
    except Exception:
        return "unknown"


def _recommended_action(detection_status: str, cf1: float, is_clean: bool, config,
                        stroke_type_confidence: float = 1.0) -> str:
    """A9 — priority logic ตรงตาม schema doc (031-043 field definition, A9 row)

    stroke_type_confidence: ความมั่นใจของการแยกประเภทท่า — ถ้าต่ำ ต้องส่งให้
    โค้ชดูเอง ไม่ใช่ปล่อยผ่านพร้อมชื่อท่าที่อาจผิด
    เหตุผล: ชุดข้อมูลปัจจุบันถ่ายแบบ "หนึ่งคลิปหนึ่งท่า" ทุกคลิป และ SL กับ VL
    ไม่เคยอยู่คลิปเดียวกันเลย -> ยืนยันไม่ได้ว่าโมเดลแยกสองท่านี้ได้จริง
    (LOPO: SL recall 0.000 ไปปนกับ VL หมด) ดู docs/STROKE_CLASSIFICATION.md
    """
    if detection_status == "failed":
        return "discard"
    if cf1 < config.coach_review_threshold:
        return "discard"
    if detection_status == "partial":
        return "coach_review"
    if not is_clean:
        return "coach_review"
    if stroke_type_confidence < config.stroke_type_review_threshold:
        return "coach_review"
    if cf1 < config.auto_accept_threshold:
        return "coach_review"
    return "auto_accept"


# ITF ground-plane geometry เดียวกับ loeuf_cv/court_calibration.py — ใช้ zoning
# ตำแหน่ง bounce ตาม homography (BL2-4). "left"/"right" อ้างอิงเครื่องหมาย x_m
# ดิบ (ไม่ใช่ deuce/ad — deuce/ad ขึ้นกับว่ากล้องอยู่ฝั่งไหนของ server ซึ่งไม่รู้
# แน่ชัดจาก camera_angle="behind_baseline" อย่างเดียว ใช้ deuce/ad ไปตรงๆ
# เสี่ยงป้าย label ผิดฝั่งได้)
_HALF_W_M = 4.115
_SERVICE_DEPTH_M = 5.485
_NET_DEPTH_M = 11.885
_COURT_LENGTH_M = 23.77
_ZONE_MARGIN_M = 0.5


def _landing_zone(x_m: float, z_m: float) -> str:
    if abs(x_m) > _HALF_W_M + _ZONE_MARGIN_M or not (-_ZONE_MARGIN_M <= z_m <= _COURT_LENGTH_M + _ZONE_MARGIN_M):
        return "out_of_bounds"
    side = "right" if x_m >= 0 else "left"
    if z_m <= _SERVICE_DEPTH_M:
        return f"near_{side}_service_box"
    if z_m <= _NET_DEPTH_M:
        return f"near_{side}_backcourt"
    if z_m <= _NET_DEPTH_M + _SERVICE_DEPTH_M:
        return f"far_{side}_service_box"
    return f"far_{side}_backcourt"


def _ball_landing_from_hit(hit: dict) -> tuple[dict | None, str | None, str | None]:
    """BL2-4 (landing_position/landing_zone/landing_call) จาก bounce data ที่
    loeuf_cv/bounce_detection.py::add_bounce_to_hits() คำนวณไว้แล้วต่อ hit
    (ต้องมี court_homography ตอนเรียก — ไม่งั้น bounce_court_x_m/z_m เป็น None
    และฟังก์ชันนี้คืน (None, None, None) เหมือนเดิม)

    ball_speed_kmh/trajectory_clearance_cm (BL อื่น) ยังคง None เสมอ — ground
    homography ให้แค่พิกัดบนพื้นสนาม (Z=0 plane) ไม่มีแกนความสูง จะเอาไป
    ประมาณตำแหน่ง/ความเร็วลูกตอนลอยกลางอากาศ (เหนือพื้น) ตรงๆ จะเพี้ยน
    ต้องมี depth/height model เพิ่ม (scope แยก)"""
    x_m, z_m = hit.get("bounce_court_x_m"), hit.get("bounce_court_z_m")
    if x_m is None or z_m is None:
        return None, None, None
    landing_position = {"x_m": x_m, "z_m": z_m}
    landing_zone = _landing_zone(x_m, z_m)
    bounce_in = hit.get("bounce_in_court")
    landing_call = ("in" if bounce_in else "out") if bounce_in is not None else None
    return landing_position, landing_zone, landing_call


def _path_vz(values_per_frame, start_frame: int, end_frame: int) -> list[dict] | None:
    """VZ format ทั่วไป: [{frame, x, y}] ทุกเฟรมระหว่าง backswing_peak → follow_through_peak
    values_per_frame: callable(frame_idx) -> (x, y) normalized 0-1, หรือ None ถ้าไม่มีค่า"""
    if start_frame is None or end_frame is None or end_frame < start_frame:
        return None
    out = []
    for f in range(start_frame, end_frame + 1):
        pt = values_per_frame(f)
        if pt is None:
            continue
        out.append({"frame": int(f), "x": round(float(pt[0]), 4), "y": round(float(pt[1]), 4)})
    return out or None


def _joint_dict(pose, frame_index: int | None) -> dict | None:
    """B-block `joint` — 13 จุดตามชื่อ spec, normalized image coords, ปัด 2 ตำแหน่ง

    INDEX SPACE: frame_index เป็น ABSOLUTE frame ของทั้งคลิป (ตรงกับที่
    schema_builder/keyframes.py::extract_keyframes คืนมา) และ `pose` ต้องเป็น
    track.pose (ทั้งคลิป) เท่านั้น — ห้ามส่ง sliced_pose เข้ามา (sliced_pose ถูก
    reindex ให้ relative กับ stroke_start_frame แล้ว จะได้พิกัดผิดเฟรม)

    คืนค่า:
      - frame_index เป็น None หรืออยู่นอกช่วง [0, len(landmarks)) → None
        (เคสนี้ keyframe นั้น detected=False อยู่แล้ว)
      - ปกติ → dict ที่มีครบ 13 key เสมอ; joint ที่พิกัดเป็น NaN → ค่าเป็น None
        แต่ key ยังอยู่ (ห้าม omit)

    VISIBILITY: landmark ที่ visibility ต่ำแต่พิกัดเป็นตัวเลขจริง → ยังส่ง x/y
    ตามปกติ ไม่ null ทิ้ง — ทั้งไฟล์นี้ใช้ NaN เป็น sentinel เดียวของ "ไม่มีค่า"
    (_wrist_pt / _path_vz ก็ใช้เกณฑ์เดียวกัน) ส่วน visibility เป็นคนละมิติและ
    รายงานแยกอยู่แล้ว การ null ตามเกณฑ์ visibility จะทำให้ overlay กะพริบ
    """
    if frame_index is None:
        return None
    lm = pose.landmarks
    if frame_index < 0 or frame_index >= len(lm):
        return None
    out = {}
    for name in JOINT_NAMES:
        x, y = lm[frame_index, JOINT_LANDMARK_IDS[name], :2]
        if np.isnan(x) or np.isnan(y):
            out[name] = None
        else:
            out[name] = {"x": round(float(x), 2), "y": round(float(y), 2)}
    return out


def _b_keyframe_entry(pose, frame_index: int | None, fps: float) -> dict:
    """023-B keyframe 1 ตัว — ต้องมีครบ 4 key เสมอ (ห้าม omit)
    undetected → {"frame_index": None, "timestamp_ms": None,
                  "detected": False, "joint": None}

    frame_index เป็น absolute frame ของทั้งคลิป (ดู _joint_dict)"""
    if frame_index is None:
        return {"frame_index": None, "timestamp_ms": None,
                "detected": False, "joint": None}
    ts = pose.timestamps_ms
    ts_ms = (float(ts[frame_index]) if 0 <= frame_index < len(ts)
             else frame_index / fps * 1000.0)
    return {
        "frame_index": int(frame_index),
        "timestamp_ms": round(ts_ms, 2),
        "detected": True,
        "joint": _joint_dict(pose, frame_index),
    }


def build_loeuf_schema(tracks, hit_events, fps, video_meta, config, racket_keypoints=None, ball_traj=None):
    """
    สร้าง Loeuf Full JSON Schema (17 Layers)
    Phase 1 & Phase 2 Intelligence

    racket_keypoints: dict[frame_idx -> ...] จาก webui/yolo_track.py (ถ้าใช้
    custom model แบบ pose ที่มี keypoint จริง) — None = kinematics racket
    fields (KN5-7) จะว่างเปล่าเหมือนเดิม ไม่ error

    ball_traj: np.ndarray (total_frames, 2) พิกัด pixel จาก
    extract_ball_trajectory_kalman() (NaN = ไม่เจอลูกเฟรมนั้น) — ใช้คำนวณ BL
    block จริง (available + _tracked_fraction) แทน placeholder เดิม —
    landing_position/landing_zone/landing_call (BL2-4) มาจาก hit_events[i]
    ที่ loeuf_cv/bounce_detection.py::add_bounce_to_hits() เติม bounce_court_*
    ไว้แล้ว (ต้องรัน add_bounce_to_hits ด้วย court_homography ก่อนส่งเข้ามา
    ไม่งั้นเป็น None ทั้ง 3 field เหมือนเดิม) — ball_speed_kmh/
    trajectory_clearance_cm ยังคง None เสมอ เพราะ ground homography ไม่มี
    แกนความสูง คำนวณความเร็ว/ความสูงเหนือพื้นของลูกลอยกลางอากาศตรงๆ ไม่ได้
    (scope แยก, ดู _ball_landing_from_hit)
    """
    import uuid
    session_id = f"sess-{datetime.datetime.now().strftime('%Y%m%d-%H%M')}"
    
    # 011-MT: session_metadata
    mt = {
        "cv_model_version": "0.2.0",
        "pose_model": "mediapipe_blazepose",
        # MT0-3 — spec แยก pose_model (MT0-2) กับ pose_model_version (MT0-3)
        # เป็นคนละ field ("For debug: not to compare over model version")
        "pose_model_version": _pose_model_version(),
        "schema_version": "1.0",
        "session_id": session_id,
        "session_date": datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        # MT4 — spec ให้เป็นเลขจำนวนเต็ม (value-example: 60) ไม่ใช่ float ดิบ
        # 29.973518... ที่ cv2 อ่านได้ · ⚠️ spec เขียนว่า "≥ 60 required for KN"
        # ต้นทาง 29.97fps จึงได้ KN visibility = low_confidence (metrics.py:513)
        "fps": int(round(fps)) if fps else 0,
        # MT5 — enum ของ session คือ single_view / multi_view (คนละตัวกับ M3
        # ที่เป็น behind_baseline / side_view) เดิมใส่ค่าของ M3 ผิด block
        "camera_angle": "single_view",
        "total_duration_sec": round(video_meta.get("total_frames", 0) / fps, 1) if fps else 0,
        "total_session_frame": video_meta.get("total_frames", 0),
        "total_strokes_detected": len(hit_events),
        # MT9 — spec: นับเฉพาะ detection_status=valid และ is_clean_stroke=true
        # (ไม่ใช่จำนวน hit ทั้งหมด) — เติมค่าจริงหลังลูป
        "usable_strokes": 0,
        "stroke_type_distribution": {} # Will calculate at the end
    }
    
    strokes = []
    stroke_counts = {"FH": 0, "BH": 0, "SV": 0, "VL": 0, "SL": 0, "RS": 0}
    
    for i, hit in enumerate(hit_events):
        impact_frame = hit["frame"]
        player_id = hit["player_id"]
        
        # Find player track
        track = next((t for t in tracks if t.track_id == player_id), None)
        if not track: continue
        
        from ..config import (R_WRIST, L_WRIST, L_ANKLE, R_ANKLE, NOSE,
                              R_SHOULDER, L_SHOULDER)
        dominant_side = config.dominant_side if hasattr(config, 'dominant_side') else "right"
        wrist_idx = R_WRIST if dominant_side == "right" else L_WRIST
        wrist_path = track.pose.landmarks[:, wrist_idx, :2]
        # head_path ใช้แยกท่าเสิร์ฟใน extract_keyframes (ข้อมือเหนือหัว = SV) —
        # ต้องส่งไป ไม่งั้น backswing_peak ของ SV จะใช้ค่า groundstroke แล้วเพี้ยน
        head_path = track.pose.landmarks[:, NOSE, :2]
        # ความกว้างไหล่ (median ทั้งคลิป) — ใช้ normalize amplitude การเหวี่ยงเพื่อ
        # แยก volley ออกจาก groundstroke ใน extract_keyframes
        _sh = np.abs(track.pose.landmarks[:, R_SHOULDER, 0]
                     - track.pose.landmarks[:, L_SHOULDER, 0])
        shoulder_width = float(np.nanmedian(_sh)) if not np.isnan(_sh).all() else None

        # Extract Keyframes (B)
        kf = extract_keyframes(impact_frame, wrist_path, fps, video_meta.get("total_frames", 0),
                               head_path=head_path, shoulder_width=shoulder_width)

        # ต้องครอบคลุมทุก keyframe ที่ตรวจเจอจริง ไม่ใช่แค่ unit_turn/recovery_position —
        # backswing_peak ค้นหาได้ไกลถึง impact-1.5s ซึ่งอาจไกลกว่า fallback impact-30 เฟรม
        # ถ้า stroke_start_frame ตัดมาสั้นกว่านั้น พอเอาไป slice+reindex ให้ MetricsEngine
        # จะได้ relative frame_index ติดลบ (ตัดหัวคิวทิ้งไปทั้งที่ยังต้องใช้) → crash/ผิดพลาด
        detected_frames = [f for f in kf.values() if f is not None]
        stroke_start_frame = max(0, min(detected_frames + [impact_frame - 30]))
        stroke_end_frame = min(video_meta.get("total_frames", 0) - 1,
                                max(detected_frames + [impact_frame + 30]))

        actual_height = config.subject_height_cm if hasattr(config, 'subject_height_cm') and config.subject_height_cm else 170.0

        # Metrics (C/D/VF/KN1-4/SS) — ใช้ MetricsEngine เดิม (loeuf_cv/metrics.py, ทดสอบ
        # แล้วผ่าน 63 tests) แทนสูตรมือใน schema_builder เอง — engine นี้ implement ครบ
        # ตาม schema doc อยู่แล้ว (D block, VF ทุก field, KN1-4, SS ต่อ stroke type) ไม่ต้อง
        # เขียนใหม่ ต้องตัด track.pose (ครอบคลุมทั้งคลิปหลาย stroke) ให้เหลือแค่ช่วง stroke
        # นี้ก่อน เพราะ engine สมมติว่า ts ที่ส่งเข้าไปเป็น stroke เดียว (ดู _slice_pose_for_stroke)
        engine_config = dataclasses.replace(config, dominant_side=dominant_side, subject_height_cm=actual_height)
        sliced_pose = _slice_pose_for_stroke(track.pose, stroke_start_frame, stroke_end_frame, engine_config)
        engine_kf = _build_engine_keyframes(kf, fps, stroke_start_frame)
        blocks, derived, vf = MetricsEngine(sliced_pose, engine_kf, engine_config, "FH").compute()
        # stroke_type ยังไม่รู้ตอนนี้ (classify_stroke ต้องใช้ c_metric ก่อน) — เรียกซ้ำรอบสอง
        # ด้วย stroke_type จริงหลังจำแนกได้ ค่า C/D/VF ส่วนใหญ่ไม่ขึ้นกับ stroke_type แต่
        # SS (stroke_specific)/บาง field ใน arm/movement/derived ที่กัน SV/VL ออก ขึ้นกับมันจริง
        c_metric = blocks

        # Stroke Type (A)
        stype, stype_conf = classify_stroke_with_confidence(
            track.pose, impact_frame, dominant_side, keyframe_metrics=c_metric,
            keyframes=kf)
        if stype in stroke_counts:
            stroke_counts[stype] += 1

        # B4/B5 ขึ้นกับ stroke_type ซึ่งเพิ่งรู้ตอนนี้ → ปรับแล้วคำนวณ metric ใหม่
        # ทั้งชุด (ไม่ใช่แค่ตอน stype != "FH" เหมือนเดิม) ไม่งั้น B block ที่ส่ง
        # ลูกค้าจะรายงานเฟรมที่ปรับแล้ว แต่ metric ยังคิดจากเฟรมชุดเก่า
        kf = refine_keyframes_for_type(kf, stype, impact_frame,
                                       video_meta.get("total_frames", 0), fps)
        detected_frames = [f for f in kf.values() if f is not None]
        # กันชนหน้า/หลัง impact = 1 วินาที (เดิมเขียน 30 เฟรมตรง ๆ ซึ่งกลายเป็น
        # 0.5 วิ ที่คลิป 60fps) — ดู keyframes.py หัวไฟล์เรื่องหน่วยเวลา
        pad = int(round(fps)) if fps else 30
        stroke_start_frame = max(0, min(detected_frames + [impact_frame - pad]))
        stroke_end_frame = min(video_meta.get("total_frames", 0) - 1,
                               max(detected_frames + [impact_frame + pad]))
        sliced_pose = _slice_pose_for_stroke(track.pose, stroke_start_frame,
                                             stroke_end_frame, engine_config)
        engine_kf = _build_engine_keyframes(kf, fps, stroke_start_frame)
        blocks, derived, vf = MetricsEngine(sliced_pose, engine_kf, engine_config, stype).compute()
        c_metric = blocks

        # Kinematics (KN) — racket_tip/center_position + racket_head_speed_mps จาก keypoint
        # จริง (custom pose model) — แม่นกว่า MetricsEngine เดิมที่ไม่มี racket detector
        # (ตั้ง null/wrist-fallback เสมอ) เลย override เฉพาะ 3 field นี้ทับของเดิมถ้ามีค่าจริง
        cm_per_px = calculate_cm_per_px(
            track.pose.landmarks[impact_frame], track.pose.visibility[impact_frame],
            L_ANKLE, R_ANKLE, NOSE, actual_height)
        racket_kn = get_racket_metrics(
            racket_keypoints, kf, wrist_path, fps,
            video_meta.get("width", 0), video_meta.get("height", 0), cm_per_px)
        for key in ("racket_tip_position", "racket_center_position", "racket_head_speed_mps"):
            if key in racket_kn:
                c_metric["kinematics"][key] = racket_kn[key]
                vf[key] = "visible"

        # metric (031-C) ตาม schema เก็บแค่ body/arm/timing/contact/movement/depth_estimated —
        # kinematics/stroke_specific/ball เป็น block แยก (032-KN/033-SS/034-BL) ต้องแยกออกมา
        kn_metric = c_metric.pop("kinematics")
        ss_metric = c_metric.pop("stroke_specific")
        c_metric.pop("ball", None)

        # 021-M: stroke_metadata
        cf1 = _pose_confidence(track.pose, impact_frame)
        m = {
            "debug_pose_confidence": round(cf1, 2),
            "processed_at": datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "resolution": f"{video_meta.get('width', 1920)}x{video_meta.get('height', 1080)}",
            "camera_angle": "behind_baseline",
            # M4/M5 — spec: true = ต้นทาง <60fps แล้ว "CV ทำการ resample ขึ้นเป็น
            # 60fps" / false = ต้นทาง >=60fps อยู่แล้ว ไม่ต้อง resample
            # ⚠️ path นี้ (build_loeuf_schema) ไม่ได้ resample เลย — hit_events,
            # keyframe, ball_traj, racket_keypoints ทั้งหมดอยู่บน index space ของ
            # เฟรมต้นฉบับ การ resample ต้อง remap ทุกตัวพร้อมกัน (ทำที่ชั้น
            # tracking ก่อน detect_hit_events ไม่ใช่ที่นี่) — ดู
            # loeuf_cv/resample.py::resample_to_target_fps ที่ StrokePipeline ใช้
            # ตราบใดที่ยังไม่ resample ต้องรายงานตามจริง: เดิมโค้ดนี้กลับตรรกะ
            # (fps>=60 -> "resampled_to_60fps") ซึ่งบอกลูกค้าว่า resample แล้ว
            # ทั้งที่ไม่เคยทำ ส่วน "native_fps_no_resample" ก็ไม่ใช่ค่าใน spec
            # KN ที่ได้รับผลกระทบถูกตีเป็น low_confidence อยู่แล้ว (metrics.py)
            "velocity_normalized": False,
            "normalization_method": "none",
        }

        # 022-VQ: video_quality — ใช้ loeuf_cv/video_quality.py เดิม (ของ StrokePipeline)
        # ตรงๆ กับ sliced_pose ของ stroke นี้ (ผ่าน smooth_timeseries + frame_stats
        # ที่ตัดมาเฉพาะช่วงแล้ว — ดู _slice_pose_for_stroke) ได้ partial_occlusion/
        # player_near_edge/partial_swing/player_too_far จริงเสมอ ส่วน low_light/
        # motion_blur ได้จริงเมื่อ track.pose.frame_stats มี mean_luma/blur_score
        # (เติมจาก webui/yolo_track.py) ไม่งั้นสองอันนี้แค่ไม่ trigger (None-safe
        # ใน assess_video_quality เอง) ไม่ error
        vq = assess_video_quality(sliced_pose, engine_config)

        # 023-B: keyframe — spec บังคับ 4 key ต่อ keyframe เสมอ
        # (frame_index, timestamp_ms, detected, joint) + B6 trophy_position
        # index space: kf[...] เป็น absolute frame → ส่ง track.pose (ทั้งคลิป)
        # ไม่ใช่ sliced_pose
        #
        # B6 trophy_position เป็น serve-only — ผูกกับ impact ตรง ๆ (impact-14)
        # ไม่ใช่กับ backswing_peak เหมือนเดิม เพราะ backswing ของ SV เองยังต่ำ
        # (0.219) ทำให้ trophy ได้แค่ 0.094 ทั้งที่วัด GT แล้ว offset จาก impact
        # ให้ LOPO 37.5% — ดู schema_builder/keyframes.py::trophy_position_frame
        # ⚠️ trophy_frame ห้ามใส่กลับเข้า dict `kf` เด็ดขาด — builder.py ด้านบน
        # คำนวณ stroke_start_frame/stroke_end_frame จาก kf.values() การเติม
        # entry ใหม่จะไปยืดหน้าต่าง stroke
        trophy_frame = trophy_position_frame(
            stype, impact_frame, video_meta.get("total_frames", 0), fps)
        b_keyframe = {
            "unit_turn": _b_keyframe_entry(track.pose, kf["unit_turn"], fps),
            "backswing_peak": _b_keyframe_entry(track.pose, kf["backswing_peak"], fps),
            "impact": _b_keyframe_entry(track.pose, kf["impact"], fps),
            "follow_through_peak": _b_keyframe_entry(track.pose, kf["follow_through_peak"], fps),
            "recovery_position": _b_keyframe_entry(track.pose, kf["recovery_position"], fps),
            "trophy_position": _b_keyframe_entry(track.pose, trophy_frame, fps),
        }

        # Ball (BL) — available/_tracked_fraction จริงจาก ball trajectory (Kalman-filtered)
        # ในช่วงเฟรมของ stroke นี้ — ball_speed_kmh/landing_* ยังเป็น None เสมอ
        # เพราะต้องมี court calibration ก่อน (ดู docstring ด้านบน)
        ball_metric = empty_ball_block()
        if ball_traj is not None and video_meta.get("width") and video_meta.get("height"):
            seg = np.asarray(ball_traj[stroke_start_frame:stroke_end_frame], dtype=float)
            if len(seg):
                norm_seg = seg / [video_meta["width"], video_meta["height"]]
                conf = (~np.isnan(norm_seg[:, 0])).astype(float)
                ball_metric = build_ball_block(BallObservations(norm_seg, conf, fps))
                landing_position, landing_zone, landing_call = _ball_landing_from_hit(hit)
                ball_metric["landing_position"] = landing_position
                ball_metric["landing_zone"] = landing_zone
                ball_metric["landing_call"] = landing_call

        # A7 detection_status: valid = ครบ B1-B5, partial = บางส่วน, failed = ไม่มีเลย
        # ⚠️ นับเฉพาะ B1-B5 (A7_KEYFRAME_NAMES) — trophy_position (B6) เป็น
        # serve-only ถ้านับรวมจะทำให้ non-serve ทุก stroke กลายเป็น partial ทันที
        b_detected = [b_keyframe[n]["detected"] for n in A7_KEYFRAME_NAMES]
        n_detected = sum(b_detected)
        detection_status = ("valid" if n_detected == len(A7_KEYFRAME_NAMES)
                            else ("failed" if n_detected == 0 else "partial"))

        # A8 is_clean_stroke: B1-B4 ตรวจเจอครบ + ไม่มี issue จาก VQ2 (ตอนนี้ issues ว่างเสมอ
        # เพราะ VQ ยังไม่ implement จริง — ค่าจะแม่นขึ้นอัตโนมัติเมื่อ VQ พร้อม)
        is_clean = all(b_keyframe[n]["detected"] for n in A8_KEYFRAME_NAMES) and not vq["issues"]

        cf2 = _inference_clean(track.pose, stroke_start_frame, stroke_end_frame, config)
        recommended_action = _recommended_action(detection_status, cf1, is_clean,
                                                 config, stype_conf)

        # 024-A: stroke_root
        a_root = {
            "stroke_id": f"{session_id}_stroke-{i+1}",
            "stroke_index": i + 1,
            "stroke_type": stype,
            "stroke_start_frame": stroke_start_frame,
            "stroke_end_frame": stroke_end_frame,
            "dominant_side": dominant_side,
            "detection_status": detection_status,
            "is_clean_stroke": is_clean,
            "stroke_recommended_action": recommended_action
        }

        # 043-VZ: visualization — spec บังคับ 5 key เสมอ (wrist_path, ball_path,
        # racket_tip_path, racket_center_path, body_center_path) ไม่มีข้อมูล → null
        # ห้ามสร้าง key แบบมีเงื่อนไขเหมือนเดิม (VZ2-4 หายไปทั้ง key เมื่อไม่ส่ง
        # ball_traj/racket_keypoints) — ดู schema_fields.VZ_FIELDS
        # index space: vz_start/vz_end + f ทุกตัวในบล็อกนี้เป็น ABSOLUTE frame
        # ของทั้งคลิป → ใช้ track.pose / ball_traj ตรง ๆ (ไม่ใช่ sliced_pose)
        vz_start = kf.get("backswing_peak")
        vz_end = kf.get("follow_through_peak")

        def _wrist_pt(f):
            if f >= len(track.pose.landmarks):
                return None
            x, y = track.pose.landmarks[f, wrist_idx, :2]
            return None if (np.isnan(x) or np.isnan(y)) else (x, y)

        com_full = com_xy(track.pose)   # (T, 2) normalized, absolute frame index

        def _com_pt(f):
            if f >= len(com_full):
                return None
            x, y = com_full[f]
            return None if (np.isnan(x) or np.isnan(y)) else (x, y)

        visualization = {name: None for name in VZ_FIELDS}
        visualization["wrist_path"] = _path_vz(_wrist_pt, vz_start, vz_end) or []
        visualization["body_center_path"] = _path_vz(_com_pt, vz_start, vz_end)

        if ball_traj is not None and video_meta.get("width") and video_meta.get("height"):
            fw, fh = video_meta["width"], video_meta["height"]

            def _ball_pt(f):
                if f >= len(ball_traj):
                    return None
                x, y = ball_traj[f]
                return None if (np.isnan(x) or np.isnan(y)) else (x / fw, y / fh)

            visualization["ball_path"] = _path_vz(_ball_pt, vz_start, vz_end)

        if racket_keypoints and video_meta.get("width") and video_meta.get("height"):
            fw, fh = video_meta["width"], video_meta["height"]
            wrist_path_full = track.pose.landmarks[:, wrist_idx, :2]

            def _racket_pts(f):
                dets = racket_keypoints.get(f)
                if not dets:
                    return None, None
                wrist_px = None
                if f < len(wrist_path_full):
                    wx, wy = wrist_path_full[f]
                    if not (np.isnan(wx) or np.isnan(wy)):
                        wrist_px = (wx * fw, wy * fh)
                det = _pick_racket_detection(dets, wrist_px)
                pts = _racket_tip_center_px(det) if det else None
                if pts is None:
                    return None, None
                return ((pts["tip_px"][0] / fw, pts["tip_px"][1] / fh),
                        (pts["center_px"][0] / fw, pts["center_px"][1] / fh))

            visualization["racket_tip_path"] = _path_vz(lambda f: _racket_pts(f)[0], vz_start, vz_end)
            visualization["racket_center_path"] = _path_vz(lambda f: _racket_pts(f)[1], vz_start, vz_end)

        stroke = {
            "stroke_metadata": m,
            "video_quality": vq,
            "keyframe": b_keyframe,
            "stroke_root": a_root,
            "metric": c_metric,
            "kinematics": kn_metric,
            "stroke_specific": ss_metric,
            "ball": ball_metric,
            "derived": derived,
            "visibility_flag": vf,
            "confidence_flag": {"pose_estimation_confidence": round(cf1, 2), "inference_clean": cf2, "issues_detected": vq["issues"]},
            "visualization": visualization
        }

        strokes.append(stroke)
        
    # MT9 — spec: usable_strokes = detection_status "valid" + is_clean_stroke true
    # (เดิมใส่ len(hit_events) = นับทุก stroke ที่ detect ได้ ซึ่งเป็นค่าของ MT8)
    mt["usable_strokes"] = sum(
        1 for s in strokes
        if s["stroke_root"]["detection_status"] == "valid"
        and s["stroke_root"]["is_clean_stroke"])

    # MT10 — ต้องมีครบทุก stroke type เสมอ (ห้าม omit key ที่นับได้ 0)
    mt["stroke_type_distribution"] = {k: stroke_counts.get(k, 0) for k in STROKE_TYPES}
    
    # Phase 2 Analytics
    agg, trend, pattern, summary = build_phase2_aggregations(strokes, config)
    
    # Remove _index used internally
    for s in strokes:
        s.pop("_index", None)
        
    session = {
        "session_metadata": mt,
        "strokes": strokes,
        "aggregated_metric": agg,
        "trend": trend,
        "pattern": pattern,
        "session_summary": summary
    }
    
    return session
