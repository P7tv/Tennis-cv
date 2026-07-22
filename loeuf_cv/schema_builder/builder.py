import dataclasses
import datetime
import numpy as np
from .keyframes import extract_keyframes
from .metrics import calculate_cm_per_px, get_racket_metrics, _pick_racket_detection, _racket_tip_center_px
from .classifier import classify_stroke
from .aggregator import build_phase2_aggregations
from ..ball import BallObservations, build_ball_block
from ..config import CORE_LANDMARKS
from ..keyframes import Keyframe
from ..metrics import MetricsEngine
from ..occlusion_fill import kinematic_fill
from ..pose_extractor import FrameStats
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


def _recommended_action(detection_status: str, cf1: float, is_clean: bool, config) -> str:
    """A9 — priority logic ตรงตาม schema doc (031-043 field definition, A9 row)"""
    if detection_status == "failed":
        return "discard"
    if cf1 < config.coach_review_threshold:
        return "discard"
    if detection_status == "partial":
        return "coach_review"
    if not is_clean:
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
        "schema_version": "1.0",
        "session_id": session_id,
        "session_date": datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "fps": fps,
        "camera_angle": "behind_baseline",
        "total_duration_sec": round(video_meta.get("total_frames", 0) / fps, 1) if fps else 0,
        "total_session_frame": video_meta.get("total_frames", 0),
        "total_strokes_detected": len(hit_events),
        "usable_strokes": len(hit_events),
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
        
        from ..config import R_WRIST, L_WRIST, L_ANKLE, R_ANKLE, NOSE
        dominant_side = config.dominant_side if hasattr(config, 'dominant_side') else "right"
        wrist_idx = R_WRIST if dominant_side == "right" else L_WRIST
        wrist_path = track.pose.landmarks[:, wrist_idx, :2]

        # Extract Keyframes (B)
        kf = extract_keyframes(impact_frame, wrist_path, fps, video_meta.get("total_frames", 0))

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
        stype = classify_stroke(track.pose, impact_frame, dominant_side, keyframe_metrics=c_metric)
        if stype in stroke_counts:
            stroke_counts[stype] += 1

        if stype != "FH":
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
            "velocity_normalized": fps >= 60,
            "normalization_method": "resampled_to_60fps" if fps >= 60 else "native_fps_no_resample"
        }

        # 022-VQ: video_quality — ใช้ loeuf_cv/video_quality.py เดิม (ของ StrokePipeline)
        # ตรงๆ กับ sliced_pose ของ stroke นี้ (ผ่าน smooth_timeseries + frame_stats
        # ที่ตัดมาเฉพาะช่วงแล้ว — ดู _slice_pose_for_stroke) ได้ partial_occlusion/
        # player_near_edge/partial_swing/player_too_far จริงเสมอ ส่วน low_light/
        # motion_blur ได้จริงเมื่อ track.pose.frame_stats มี mean_luma/blur_score
        # (เติมจาก webui/yolo_track.py) ไม่งั้นสองอันนี้แค่ไม่ trigger (None-safe
        # ใน assess_video_quality เอง) ไม่ error
        vq = assess_video_quality(sliced_pose, engine_config)

        # 023-B: keyframe
        b_keyframe = {
            "unit_turn": {"frame_index": kf["unit_turn"], "detected": kf["unit_turn"] is not None},
            "backswing_peak": {"frame_index": kf["backswing_peak"], "detected": kf["backswing_peak"] is not None},
            "impact": {"frame_index": kf["impact"], "detected": True},
            "follow_through_peak": {"frame_index": kf["follow_through_peak"], "detected": kf["follow_through_peak"] is not None},
            "recovery_position": {"frame_index": kf["recovery_position"], "detected": kf["recovery_position"] is not None}
        }

        # Ball (BL) — available/_tracked_fraction จริงจาก ball trajectory (Kalman-filtered)
        # ในช่วงเฟรมของ stroke นี้ — ball_speed_kmh/landing_* ยังเป็น None เสมอ
        # เพราะต้องมี court calibration ก่อน (ดู docstring ด้านบน)
        ball_metric = {"available": False}
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
        b_detected = [b_keyframe[n]["detected"] for n in
                      ("unit_turn", "backswing_peak", "impact", "follow_through_peak", "recovery_position")]
        n_detected = sum(b_detected)
        detection_status = "valid" if n_detected == 5 else ("failed" if n_detected == 0 else "partial")

        # A8 is_clean_stroke: B1-B4 ตรวจเจอครบ + ไม่มี issue จาก VQ2 (ตอนนี้ issues ว่างเสมอ
        # เพราะ VQ ยังไม่ implement จริง — ค่าจะแม่นขึ้นอัตโนมัติเมื่อ VQ พร้อม)
        is_clean = all(b_keyframe[n]["detected"] for n in
                       ("unit_turn", "backswing_peak", "impact", "follow_through_peak")) and not vq["issues"]

        cf2 = _inference_clean(track.pose, stroke_start_frame, stroke_end_frame, config)
        recommended_action = _recommended_action(detection_status, cf1, is_clean, config)

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

        # 043-VZ: visualization — wrist_path (VZ1, required), + ball/racket path (VZ2-4)
        # ถ้ามีข้อมูล ทุกจุดระหว่าง backswing_peak -> follow_through_peak ตาม spec
        vz_start = kf.get("backswing_peak")
        vz_end = kf.get("follow_through_peak")

        def _wrist_pt(f):
            if f >= len(track.pose.landmarks):
                return None
            x, y = track.pose.landmarks[f, wrist_idx, :2]
            return None if (np.isnan(x) or np.isnan(y)) else (x, y)

        visualization = {"wrist_path": _path_vz(_wrist_pt, vz_start, vz_end) or []}

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
        
    mt["stroke_type_distribution"] = {k: v for k, v in stroke_counts.items() if v > 0}
    
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
