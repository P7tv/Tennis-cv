"""ทดสอบ logic กับ 17-layer schema (cv_schema_table_loeuf)

รัน: python -m pytest tests/ -v — ไม่ต้องใช้วิดีโอ/mediapipe จริง
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import pytest

from loeuf_cv import PipelineConfig, StrokePipeline
from loeuf_cv.action_spotting import spot_strokes
from loeuf_cv.aggregation import (
    build_aggregated_metric, build_pattern, build_session_summary, build_trend,
)
from loeuf_cv.config import KEYFRAME_NAMES
from loeuf_cv.resample import resample_to_target_fps
from loeuf_cv.synthetic import synthetic_forehand, synthetic_session

CONFIG = PipelineConfig(subject_height_cm=172.0, session_id="sess-20260702-01")
PIPELINE = StrokePipeline(CONFIG)

METRIC_BLOCKS = ("body", "arm", "timing", "contact", "movement",
                 "depth_estimated", "kinematics", "stroke_specific", "ball")
STROKE_LAYERS = ("stroke_metadata", "video_quality", "keyframes", "stroke_root",
                 "metrics", "derived", "visibility_flag", "confidence_flag",
                 "visualization")


def _stroke(fps=30, stroke_type="FH"):
    return PIPELINE.process_timeseries(synthetic_forehand(fps=fps), stroke_type)


def test_stroke_object_has_all_layers():
    stroke = _stroke()
    for layer in STROKE_LAYERS:
        assert layer in stroke, f"missing layer {layer}"
    for block in METRIC_BLOCKS:
        assert block in stroke["metrics"], f"missing metric block {block}"


def test_keyframes_detected_and_ordered():
    kf = _stroke()["keyframes"]
    assert set(kf) == set(KEYFRAME_NAMES)
    for name in ("unit_turn", "backswing_peak", "impact", "follow_through_peak"):
        assert kf[name]["detected"], f"{name} should be detected"
    order = [kf[n]["timestamp_ms"] for n in
             ("unit_turn", "backswing_peak", "impact", "follow_through_peak")]
    assert order == sorted(order)
    t_impact = kf["impact"]["timestamp_ms"]
    assert 1150 <= t_impact <= 1450


def test_null_rule_value_matches_visibility():
    """Null rule: not_detectable ⟺ value null (ห้าม omit field)"""
    stroke = _stroke()
    vf = stroke["visibility_flag"]
    flat = {}
    for block in METRIC_BLOCKS:
        if block == "ball":
            continue
        flat.update(stroke["metrics"][block])
    flat.update(stroke["derived"])

    for field, value in flat.items():
        assert field in vf, f"{field} missing from visibility_flag"
        if vf[field] == "not_detectable":
            assert value is None, f"{field}: not_detectable but value={value}"

    # VF ส่งเฉพาะ C/KN/SS/D — ต้องไม่มี key จาก M/A/VQ/B/CF
    for forbidden in ("usable", "impact", "stroke_id",
                      "pose_estimation_confidence"):
        assert forbidden not in vf


def test_fps_normalization_m4_m5():
    s30 = _stroke(fps=30)["stroke_metadata"]
    s60 = _stroke(fps=60)["stroke_metadata"]
    assert s30["velocity_normalized"] is True
    assert s30["normalization_method"] == "resampled_to_60fps"
    assert s60["velocity_normalized"] is False

    ts, normalized = resample_to_target_fps(synthetic_forehand(fps=30), CONFIG)
    assert normalized and abs(ts.fps - 60.0) < 1e-6
    # KN fields จากต้นทาง 30fps ถูก cap เป็น low_confidence
    vf = _stroke(fps=30)["visibility_flag"]
    assert vf["shoulder_rotation_velocity_deg_s"] == "low_confidence"
    assert _stroke(fps=60)["visibility_flag"][
        "shoulder_rotation_velocity_deg_s"] == "visible"


def test_rotation_degrees_nonzero():
    body = _stroke()["metrics"]["body"]
    assert body["shoulder_rotation_at_impact_deg"] is not None
    assert abs(body["shoulder_rotation_at_impact_deg"]) > 10.0
    assert body["shoulder_hip_separation_at_impact_deg"] is not None


def test_stroke_root_priority_logic():
    root = _stroke()["stroke_root"]
    assert root["detection_status"] == "valid"
    assert root["is_clean_stroke"] is True
    assert root["stroke_recommended_action"] == "auto_accept"
    assert root["stroke_id"].startswith("sess-20260702-01_stroke-")


def test_static_clip_fails_gracefully():
    ts = synthetic_forehand(fps=30)
    ts.landmarks = np.repeat(ts.landmarks[:1], ts.n_frames, axis=0)
    ts.world_landmarks = np.repeat(ts.world_landmarks[:1], ts.n_frames, axis=0)
    stroke = PIPELINE.process_timeseries(ts, "FH")

    impact = stroke["keyframes"]["impact"]
    assert impact == {"frame_index": None, "timestamp_ms": None,
                      "detected": False}
    assert stroke["stroke_root"]["detection_status"] == "failed"
    assert stroke["stroke_root"]["stroke_recommended_action"] == "discard"


def test_visualization_wrist_path_required():
    stroke = _stroke()
    vz = stroke["visualization"]
    assert isinstance(vz["wrist_path"], list) and len(vz["wrist_path"]) > 5
    b2 = stroke["keyframes"]["backswing_peak"]["frame_index"]
    b4 = stroke["keyframes"]["follow_through_peak"]["frame_index"]
    frames = [p["frame"] for p in vz["wrist_path"]]
    assert min(frames) >= b2 and max(frames) <= b4
    assert vz["ball_path"] is None and vz["racket_tip_path"] is None


def test_phase1_output_wrapper():
    from loeuf_cv.schema import build_output, build_session_metadata
    ts = synthetic_forehand(fps=60)
    out = build_output(build_session_metadata(ts, CONFIG), [_stroke(fps=60)])
    mt = out["session_metadata"]
    assert mt["total_strokes_detected"] is None      # MT8 Phase 1 = null
    assert mt["stroke_type_distribution"] is None    # MT10 Phase 1 = null
    assert out["aggregated_metric"] is None
    assert out["trend"] is None


def test_phase2_aggregation_blocks():
    session_ts = synthetic_session(fps=30, n_strokes=4)
    ts, normalized = resample_to_target_fps(session_ts, CONFIG)
    windows = spot_strokes(ts, CONFIG)
    assert len(windows) == 4

    strokes = []
    for i, w in enumerate(windows):
        strokes.append(PIPELINE.process_timeseries(
            ts.slice(w.start_frame, w.end_frame), "FH",
            stroke_index=i + 1, window_offset_ms=w.offset_ms,
            window_offset_frames=w.start_frame, pre_resampled=normalized))

    agg = build_aggregated_metric(strokes)
    fh = next(a for a in agg if a["stroke_type"] == "FH")
    assert fh["stroke_count"] >= 3
    assert fh["shoulder_rotation_mean_deg"] is not None
    assert fh["tempo_ratio_sd"] is not None

    trend = build_trend(strokes)
    assert trend["window_method"] == "thirds"
    assert trend["shoulder_rotation_trend"] in (
        "improving", "declining", "stable", "volatile")
    assert isinstance(trend["fatigue_flag"], bool)

    pattern = build_pattern(strokes)
    assert pattern["dominant_stroke_type"] == "FH"
    assert pattern["stroke_sequence"] == ["FH"] * len(strokes)
    assert pattern["backhand_avoidance_flag"] is None  # < 20 strokes → null

    summary = build_session_summary(strokes, trend, CONFIG)
    assert summary["highlight_stroke_index"] in [
        s["stroke_root"]["stroke_index"] for s in strokes]
    assert isinstance(summary["coach_flag"], bool)


def test_stroke_specific_serve_and_volley():
    sv = PIPELINE.process_timeseries(synthetic_forehand(fps=30), "SV")
    ss = sv["metrics"]["stroke_specific"]
    assert "trophy_position_achieved" in ss and "stance_type" in ss
    # SV null rules
    assert sv["metrics"]["arm"]["follow_through_angle_deg"] is None
    assert sv["metrics"]["movement"]["split_step_detected"] is None
    assert sv["metrics"]["contact"]["swing_path_angle_deg"] is None

    vl = PIPELINE.process_timeseries(synthetic_forehand(fps=30), "VL")
    assert "backswing_past_ear" in vl["metrics"]["stroke_specific"]
    assert vl["metrics"]["timing"]["backswing_duration_ms"] is None  # VL null
    assert vl["metrics"]["timing"]["tempo_ratio"] is None

    fh = _stroke()
    assert fh["metrics"]["stroke_specific"] == {}  # FH → {}


def _rotate_ts(ts, roll_deg):
    """หมุน landmark จำลองกล้องเอียง (pixel space, รอบกลางลำตัว)"""
    aspect = ts.meta.width / ts.meta.height
    th = np.radians(roll_deg)
    c, s = np.cos(th), np.sin(th)
    xy = ts.landmarks[:, :, :2].copy()
    xy[:, :, 0] *= aspect
    center = xy.mean(axis=1, keepdims=True)
    rel = xy - center
    out = np.empty_like(rel)
    out[:, :, 0] = rel[:, :, 0] * c - rel[:, :, 1] * s
    out[:, :, 1] = rel[:, :, 0] * s + rel[:, :, 1] * c
    xy = out + center
    xy[:, :, 0] /= aspect
    ts.landmarks[:, :, :2] = xy
    return ts


def test_calibration_recovers_injected_roll():
    from loeuf_cv.calibration import estimate_calibration
    ts = _rotate_ts(synthetic_forehand(fps=30), roll_deg=6.0)
    cal = estimate_calibration(ts, CONFIG)
    assert 4.0 <= cal.roll_deg <= 8.0, f"estimated roll {cal.roll_deg}"
    assert cal.tier == "ok"


def test_tilted_camera_flagged_and_metrics_survive():
    ts = _rotate_ts(synthetic_forehand(fps=30), roll_deg=12.0)
    stroke = PIPELINE.process_timeseries(ts, "FH")
    assert "camera_tilted" in stroke["video_quality"]["issues"]
    # tilt → is_clean false → coach_review (ไม่ discard — ยังวัดได้)
    assert stroke["stroke_root"]["is_clean_stroke"] is False
    assert stroke["stroke_root"]["stroke_recommended_action"] == "coach_review"
    # roll correction ทำให้ keyframes ยังหาเจอ
    assert stroke["keyframes"]["impact"]["detected"]
    assert "debug_camera" in stroke["stroke_metadata"]


def test_rotation_zero_reference():
    """rotation ที่ ready ต้อง ~0 หลัง zero-reference"""
    stroke = _stroke()
    c3 = stroke["metrics"]["body"]["shoulder_rotation_at_unit_turn_deg"]
    c4 = stroke["metrics"]["body"]["shoulder_rotation_at_impact_deg"]
    # unit_turn คือช่วงเริ่ม coil — ค่าควรเล็กกว่า impact ที่เปิดตัวเต็มที่
    assert c3 is not None and c4 is not None
    assert abs(c4) > abs(c3)


def test_benchmark_keyframe_accuracy():
    from loeuf_cv.benchmark import keyframe_accuracy

    stroke = _stroke(fps=30)
    preds = {"FH/fh_001": stroke}
    imp = stroke["keyframes"]["impact"]
    bs = stroke["keyframes"]["backswing_peak"]
    # GT ตรงกับ prediction (แปลง ms กลับเป็นเฟรมต้นทาง 30fps)
    gt_exact = {
        "clip_path": "FH/fh_001.mp4", "stroke_type": "FH", "fps": 30.0,
        "usable": True, "unit_turn_frame": None,
        "backswing_peak_frame": round(bs["timestamp_ms"] * 30 / 1000),
        "impact_frame": round(imp["timestamp_ms"] * 30 / 1000),
        "follow_through_peak_frame": None, "ready_frame": None,
        "ready_restored": None,
    }
    r = keyframe_accuracy([gt_exact], preds, tolerance_frames=1)
    assert r["acceptance_accuracy"] == 1.0 and r["acceptance_n"] == 2

    # GT เลื่อน 5 เฟรม → ต้องไม่ผ่าน
    gt_off = dict(gt_exact, impact_frame=gt_exact["impact_frame"] + 5,
                  backswing_peak_frame=gt_exact["backswing_peak_frame"] + 5)
    r2 = keyframe_accuracy([gt_off], preds, tolerance_frames=1)
    assert r2["acceptance_accuracy"] == 0.0
    assert len(r2["failures"]) == 2


def test_benchmark_agreement_and_spotting():
    from loeuf_cv.benchmark import annotator_agreement, spotting_metrics

    base = {"clip_path": "FH/a.mp4", "stroke_type": "FH", "fps": 30.0,
            "usable": True, "unit_turn_frame": None,
            "follow_through_peak_frame": None, "ready_frame": None,
            "ready_restored": None}
    rows = [dict(base, impact_frame=41, backswing_peak_frame=34, annotator_id="A"),
            dict(base, impact_frame=42, backswing_peak_frame=30, annotator_id="B")]
    ag = annotator_agreement(rows)
    assert ag["pairs_compared"] == 2
    assert ag["agreement"] == 0.5  # impact ห่าง 1 (ผ่าน), backswing ห่าง 4 (ไม่ผ่าน)

    session_out = {"strokes": [
        {"keyframes": {"impact": {"detected": True, "timestamp_ms": 13800}}},
        {"keyframes": {"impact": {"detected": True, "timestamp_ms": 99000}}},
    ]}
    gt = [{"start_time_s": 12.4, "end_time_s": 15.1},
          {"start_time_s": 30.0, "end_time_s": 33.0}]
    sp = spotting_metrics(gt, session_out)
    assert sp["recall"] == 0.5 and sp["precision"] == 0.5


def test_classifier_hierarchy():
    from loeuf_cv.classifier import classify_stroke
    from loeuf_cv.synthetic import (
        synthetic_backhand, synthetic_serve_like, synthetic_slice,
        synthetic_volley_like,
    )

    cases = [
        (synthetic_forehand(fps=30), "FH"),
        (synthetic_backhand(fps=30), "BH"),
        (synthetic_serve_like(fps=30), "SV"),
        (synthetic_volley_like(fps=30), "VL"),
        (synthetic_slice(fps=30), "SL"),
    ]
    for ts, expected in cases:
        c = classify_stroke(ts, CONFIG)
        assert c.stroke_type == expected, \
            f"expected {expected} got {c.stroke_type} ({c.features})"
        assert 0.5 <= c.confidence <= 0.95
        assert c.stroke_type != "RS"  # RS ห้ามออกจาก pose classifier


def test_ball_impact_fusion():
    from loeuf_cv.ball import BallObservations, build_ball_path, fuse_impact

    ts = synthetic_forehand(fps=30)
    from loeuf_cv.resample import resample_to_target_fps
    from loeuf_cv.smoothing import smooth_timeseries
    from loeuf_cv.keyframes import detect_keyframes
    rs, _ = resample_to_target_fps(ts, CONFIG)
    smoothed = smooth_timeseries(rs, CONFIG)
    kf = detect_keyframes(smoothed, CONFIG, "FH")
    t_pose_impact = kf["impact"].timestamp_ms

    # ball วิ่งเข้าแล้วเปลี่ยนทิศกะทันหันที่ 1300ms (ต้นทาง 30fps)
    n = 90
    pos = np.full((n, 2), np.nan)
    for f in range(n):
        t = f / 30.0 * 1000.0
        if 900 <= t <= 1800:
            pos[f, 0] = 0.9 - (t - 900) * 0.0006 if t < 1300 \
                else 0.66 + (t - 1300) * 0.0007
            pos[f, 1] = 0.5
    ball = BallObservations(pos, np.ones(n), fps=30.0)

    fused = fuse_impact(smoothed, kf, ball, CONFIG)
    assert fused["impact"].detected
    # fusion ต้องดึง impact เข้าใกล้ 1300ms (จุดลูกเปลี่ยนทิศ)
    assert abs(fused["impact"].timestamp_ms - 1300) < \
        abs(t_pose_impact - 1300) + 40

    # ผ่าน stroke pipeline เต็ม → BL block + VZ2
    stroke = PIPELINE.process_timeseries(ts, "FH", ball_observations=ball)
    assert stroke["metrics"]["ball"]["available"] is True
    assert stroke["visualization"]["ball_path"] is not None


def test_image_normalization_functions():
    import cv2
    from loeuf_cv.image_norm import (
        gray_world_white_balance, normalize_exposure, normalize_frame,
    )

    # ภาพอมส้ม (WB เพี้ยนแบบมือถือบางรุ่น) → หลัง WB channel ต้องสมดุลขึ้น
    warm = np.zeros((60, 80, 3), np.uint8)
    warm[:] = (60, 110, 190)  # BGR — แดงจัด
    balanced = gray_world_white_balance(warm)
    m = balanced.reshape(-1, 3).mean(axis=0)
    assert (m.max() - m.min()) < 40, f"channels still unbalanced: {m}"

    # ภาพมืด → exposure ดึงสว่างขึ้นเข้า target
    dark = np.full((60, 80, 3), 40, np.uint8)
    brightened = normalize_exposure(dark)
    assert cv2.cvtColor(brightened, cv2.COLOR_BGR2GRAY).mean() > 80

    # เฟรมมืด → normalize_frame ต้องไม่พัง (WB + exposure + CLAHE path)
    out = normalize_frame(dark, raw_luma=40.0)
    assert out.shape == dark.shape and out.dtype == np.uint8


def test_player_near_edge_flagged():
    ts = synthetic_forehand(fps=30)
    ts.landmarks[:, :, 0] -= 0.47  # เลื่อนผู้เล่นทั้งตัวไปชิดขอบซ้าย
    stroke = PIPELINE.process_timeseries(ts, "FH")
    assert "player_near_edge" in stroke["video_quality"]["issues"]

    centered = PIPELINE.process_timeseries(synthetic_forehand(fps=30), "FH")
    assert "player_near_edge" not in centered["video_quality"]["issues"]


def test_crop_landmarks_to_frame_coordinate_transform():
    from loeuf_cv.multi_person import crop_landmarks_to_frame

    # crop เริ่มที่ (100,50) ขนาด 200x300 ใน เฟรม 960x540
    norm_in_crop = np.array([[0.5, 0.5, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    crop_box = (100, 50, 200, 300)
    out = crop_landmarks_to_frame(norm_in_crop, crop_box, frame_w=960, frame_h=540)

    # จุดกึ่งกลาง crop -> pixel (100+100, 50+150) = (200,200) -> norm (200/960, 200/540)
    assert abs(out[0, 0] - 200 / 960) < 1e-9
    assert abs(out[0, 1] - 200 / 540) < 1e-9
    # มุมบนซ้าย crop -> (100,50) พอดี
    assert abs(out[1, 0] - 100 / 960) < 1e-9
    assert abs(out[1, 1] - 50 / 540) < 1e-9
    # มุมล่างขวา crop -> (300,350)
    assert abs(out[2, 0] - 300 / 960) < 1e-9
    assert abs(out[2, 1] - 350 / 540) < 1e-9


def _draw_person_blob(frame, cx, cy, w=40, h=100):
    cv2.rectangle(frame, (int(cx - w / 2), int(cy - h / 2)),
                 (int(cx + w / 2), int(cy + h / 2)), (200, 200, 200), -1)


def test_blob_detection_and_tracking_two_moving_people():
    from loeuf_cv.multi_person import detect_blobs, match_tracks

    W, H = 640, 480
    bg = cv2.createBackgroundSubtractorMOG2(history=30, varThreshold=25,
                                            detectShadows=False)
    # วอร์มอัพพื้นหลังนิ่ง (สีเดียวกันทุกเฟรม)
    for _ in range(20):
        frame = np.full((H, W, 3), 50, np.uint8)
        bg.apply(frame)

    # จำลอง 2 คนเดินสวนทางกัน 15 เฟรม
    active_centroids = {}
    tracks_seen = []
    for i in range(15):
        frame = np.full((H, W, 3), 50, np.uint8)
        _draw_person_blob(frame, cx=100 + i * 10, cy=240)   # คนซ้าย เดินไปขวา
        _draw_person_blob(frame, cx=500 - i * 8, cy=240)    # คนขวา เดินไปซ้าย
        boxes = detect_blobs(bg, frame, learning_rate=0.0)
        assert len(boxes) == 2, f"frame {i}: expected 2 blobs, got {len(boxes)}"

        assigned = match_tracks(active_centroids, boxes, frame_w=W)
        next_id = max(active_centroids.keys(), default=-1) + 1
        frame_ids = []
        for bi, box in enumerate(boxes):
            x, y, w, h = box
            if bi in assigned:
                tid = assigned[bi]
            else:
                tid = next_id
                next_id += 1
            active_centroids[tid] = (x + w / 2.0, y + h / 2.0)
            frame_ids.append(tid)
        tracks_seen.append(sorted(frame_ids))

    # ต้องมีแค่ 2 track ID ตลอดทั้งคลิป (ไม่สร้าง ID ใหม่มั่ว ๆ)
    all_ids = set(i for fr in tracks_seen for i in fr)
    assert len(all_ids) == 2, f"expected 2 stable track IDs, got {all_ids}"
    # ทุกเฟรมต้องเห็นครบ 2 track เดิม
    for fr in tracks_seen:
        assert set(fr) == all_ids


def test_track_gap_tolerance_and_role_assignment():
    from loeuf_cv.multi_person import match_tracks, MAX_TRACK_GAP_FRAMES

    # track หายไปสั้น ๆ (น้อยกว่า gap tolerance) ต้อง match กลับเป็น ID เดิมได้
    active = {0: (100, 100)}
    boxes = [(90, 90, 20, 20)]  # centroid (100,100) พอดี
    assigned = match_tracks(active, boxes, frame_w=640)
    assert assigned == {0: 0}

    # box ที่ไกลเกิน max_dist ต้องไม่ match (กลายเป็น track ใหม่)
    boxes_far = [(600, 400, 20, 20)]
    assigned_far = match_tracks(active, boxes_far, frame_w=640)
    assert assigned_far == {}


def _fake_track(frame_idxs, centroid_xs, centroid_ys, height_px=100.0):
    """track dict ขั้นต่ำสำหรับทดสอบ select_candidate_tracks (ไม่ต้องมี
    landmark/visibility จริง — ฟังก์ชันนี้ดูแค่ frames.keys()/centroid)
    frame_idxs ระบุเองได้ (ไม่ใช่ range(n) เสมอไป) เพื่อคุม temporal
    overlap ระหว่าง track ในเทสได้ตรง ๆ"""
    return {
        "frames": {i: (None, None, None, height_px) for i in frame_idxs},
        "centroid_xs": list(centroid_xs),
        "centroid_ys": list(centroid_ys),
    }


def test_select_candidate_tracks_rejects_stationary_bystander():
    """⚠️ ข้อค้นพบจริงจากคลิปลูกค้า (2026-07-10): คนยืนนิ่งข้างสนาม
    (ผู้ชม/เก็บบอล) track ต่อเนื่องได้ง่ายกว่าผู้เล่นจริงที่ขยับตลอด —
    ถ้าเลือกด้วย "track ไหนยาวสุด" อย่างเดียวจะได้คนผิด แม้ track ยาว
    กว่าก็ต้องไม่ถูกเลือกถ้าแทบไม่ขยับเลย"""
    from loeuf_cv.multi_person import select_candidate_tracks

    frame_w, frame_h = 960, 540
    bystander = _fake_track(range(211), [900] * 211, [260 + 0.01 * i for i in range(211)])
    real_player = _fake_track(range(100), [300 + i * 0.5 for i in range(100)],
                              [200 + 40 * np.sin(i / 10) for i in range(100)])
    tracks = {1: bystander, 2: real_player}

    clusters = select_candidate_tracks(tracks, frame_w, frame_h, max_players=2)
    kept_ids = {tid for cluster in clusters for tid, _ in cluster}
    assert 1 not in kept_ids
    assert 2 in kept_ids


def test_select_candidate_tracks_merges_same_person_fragments():
    """⚠️ ข้อค้นพบจริง (2026-07-10 บ่าย): คนเดียวกันอาจถูกตัดเป็นหลาย
    track ID (blob หายเกิน MAX_TRACK_GAP_FRAMES แล้วโผล่ใหม่ตำแหน่งเดิม
    ซ้ำ ๆ ระหว่างแรลลี่ — เจอจริง 7 ท่อนในคลิปเดียว) เดิมเคยแค่ "เลือก
    ท่อนที่ยาวสุด แล้วทิ้งท่อนอื่น" (เสียข้อมูลไปเกือบครึ่ง) ตอนนี้ต้อง
    stitch ท่อนที่เป็นคนเดียวกันเข้าเป็น cluster เดียว ไม่ใช่แค่เลือก
    ท่อนเดียวแล้วทิ้งที่เหลือ"""
    from loeuf_cv.multi_person import select_candidate_tracks

    frame_w, frame_h = 960, 540
    # track เดียวกันทางกายภาพ (x~300) แยกเป็น 2 ID เพราะ track หลุดกลางคลิป
    # — คนละช่วงเวลากันเลย (part1 เฟรม 0-199, part2 เฟรม 300-379)
    person_a_part1 = _fake_track(range(0, 200), [300 + i * 0.1 for i in range(200)],
                                 [220 + 30 * np.sin(i / 15) for i in range(200)])
    person_a_part2 = _fake_track(range(300, 380), [305 + i * 0.1 for i in range(80)],
                                 [225 + 30 * np.sin(i / 15) for i in range(80)])
    # คนที่สองจริง ๆ อยู่คนละตำแหน่งชัดเจน (x~650)
    person_b = _fake_track(range(0, 60), [650 + i * 0.2 for i in range(60)],
                           [210 + 35 * np.sin(i / 12) for i in range(60)])
    tracks = {10: person_a_part1, 11: person_a_part2, 12: person_b}

    clusters = select_candidate_tracks(tracks, frame_w, frame_h, max_players=2)
    assert len(clusters) == 2
    cluster_id_sets = [{tid for tid, _ in cluster} for cluster in clusters]
    assert {10, 11} in cluster_id_sets  # ทั้งสองท่อนของคนเดียวกัน stitch เข้าด้วยกัน
    assert {12} in cluster_id_sets


def test_select_candidate_tracks_keeps_two_close_people_tracked_simultaneously():
    """⚠️ ข้อค้นพบจริง (2026-07-10) — คนละกรณีกับเทสก่อนหน้า: สองคนยืน
    ใกล้กันจริงตลอดคลิป (เช่น คู่ double ที่ตาข่าย) mean centroid ใกล้
    กันมากเหมือนกรณี "คนเดียวกัน" ทุกประการ แต่เฟรมที่ track ได้ overlap
    กันสูง (active พร้อมกันจริง = เป็นไปไม่ได้ที่จะเป็นคนเดียวกัน) ต้อง
    ไม่ถูก merge เข้าด้วยกัน — ถ้าเช็คแค่ระยะห่างอย่างเดียวจะเผลอรวมคน
    สองคนเป็นคนเดียว (นี่คือบั๊กที่เจอจริงตอนแรกที่ implement เฉพาะ
    spatial check)"""
    from loeuf_cv.multi_person import select_candidate_tracks

    frame_w, frame_h = 960, 540
    person_a = _fake_track(range(0, 300), [490 + 5 * np.sin(i / 8) for i in range(300)],
                           [255 + 20 * np.sin(i / 9) for i in range(300)])
    # คนละคน แต่ยืนใกล้กันมาก (mean centroid ห่างกันแค่ ~10px) ตลอดช่วงเวลาเดียวกัน
    person_b = _fake_track(range(0, 250), [485 + 5 * np.cos(i / 8) for i in range(250)],
                           [260 + 20 * np.cos(i / 9) for i in range(250)])
    tracks = {0: person_a, 1: person_b}

    clusters = select_candidate_tracks(tracks, frame_w, frame_h, max_players=2)
    assert len(clusters) == 2
    cluster_id_sets = [{tid for tid, _ in cluster} for cluster in clusters]
    assert {0} in cluster_id_sets
    assert {1} in cluster_id_sets


def test_select_candidate_tracks_avoids_transitive_merge_of_different_people():
    """⚠️ ข้อค้นพบจริง (2026-07-10 บ่าย): union-find (single-linkage) merge
    คนสองคนที่เป็นคนละคนจริง (overlap เวลาสูง) เข้าด้วยกันได้ผ่าน fragment
    ตัวกลางที่ compatible กับทั้งสองฝั่งแยกกัน (transitive merge ผิดคน) —
    validated บนคลิปจริงว่าเกิดขึ้นจริง (near+far ถูกรวมเป็น cluster เดียว
    ผ่าน fragment ตัวกลาง) ห้ามเกิดขึ้นไม่ว่าจะมี fragment ตัวกลางกี่ตัว"""
    from loeuf_cv.multi_person import select_candidate_tracks

    frame_w, frame_h = 960, 540
    # คนสองคนจริง ยืนใกล้กัน active พร้อมกันตลอด (overlap เวลาสูง = คนละคนแน่)
    person_a = _fake_track(range(0, 200), [300 + 5 * np.sin(i / 8) for i in range(200)],
                           [220 + 25 * np.sin(i / 9) for i in range(200)])
    person_b = _fake_track(range(0, 200), [310 + 5 * np.cos(i / 8) for i in range(200)],
                           [225 + 25 * np.cos(i / 9) for i in range(200)])
    # fragment ตัวกลาง: ไม่ overlap เวลากับทั้งคู่ (มาทีหลัง) ตำแหน่งใกล้ทั้งสองฝั่ง
    bridge = _fake_track(range(250, 300), [305 + i * 0.05 for i in range(50)],
                         [222 + i * 0.05 for i in range(50)])
    tracks = {0: person_a, 1: person_b, 2: bridge}

    clusters = select_candidate_tracks(tracks, frame_w, frame_h, max_players=3)
    cluster_of = {tid: ci for ci, cluster in enumerate(clusters) for tid, _ in cluster}
    assert cluster_of[0] != cluster_of[1]  # คนละคนจริง ต้องไม่รวม cluster เดียวกัน


def test_select_candidate_tracks_average_linkage_tolerates_outlier_member():
    """⚠️ ข้อค้นพบจริง (2026-07-10 บ่าย): complete-linkage (ต้อง compatible
    กับสมาชิกทุกคนใน cluster) ทำให้ fragment ที่ควรรวมเข้า cluster หลัก
    ไม่ผ่านเพราะ fail กับสมาชิกเก่าแค่ตัวเดียว (ตำแหน่งเบี่ยงสะสมไปทีละนิด
    ตามเวลา) ทั้งที่ compatible กับ "จุดรวม" (aggregate) ของ cluster สบาย ๆ
    — validate บนคลิปจริงแล้วว่าทำให้ fragment ของ near player ถูกจัดเป็น
    "far" (คนละคน) ผิด ๆ average-linkage ต้องทนต่อ outlier สมาชิกตัวเดียวได้"""
    from loeuf_cv.multi_person import select_candidate_tracks

    frame_w, frame_h = 960, 540
    # member1 กับ member3 ห่างกันเกิน min_sep โดยตรง (180px > 144px) แต่
    # member2 (กลาง) เชื่อมทั้งคู่ผ่าน aggregate ได้ — คนเดียวกันที่ตำแหน่ง
    # ขยับไปทีละนิดตามเวลา (เช่น เดินไปมาระหว่างแรลลี่)
    member1 = _fake_track(range(0, 200), [100 + 5 * np.sin(i / 8) for i in range(200)],
                          [220 + 25 * np.sin(i / 9) for i in range(200)])
    member2 = _fake_track(range(300, 500), [220 + 5 * np.cos(i / 8) for i in range(200)],
                          [225 + 25 * np.cos(i / 9) for i in range(200)])
    member3 = _fake_track(range(600, 800), [280 + 5 * np.sin(i / 8) for i in range(200)],
                          [222 + 25 * np.sin(i / 9) for i in range(200)])
    tracks = {0: member1, 1: member2, 2: member3}

    clusters = select_candidate_tracks(tracks, frame_w, frame_h, max_players=1)
    assert len(clusters) == 1
    kept_ids = {tid for tid, _ in clusters[0]}
    assert kept_ids == {0, 1, 2}  # ทั้งสามควรรวมเป็นคนเดียวกัน แม้ 0 กับ 2 ไกลกันเกิน min_sep โดยตรง


def test_merge_split_person_boxes_merges_vertically_stacked():
    """⚠️ ข้อค้นพบจริงจากคลิปลูกค้า (2026-07-10 บ่าย): คนคนเดียวขยับเร็ว/
    แสงไม่สม่ำเสมอ ทำให้ background subtraction เห็นเป็น 2 blob แยกกัน
    (หัว/ไหล่ vs ลำตัว/ขา) ในเฟรมเดียวกัน — ตัวเลขจริงจาก frame 587 ของ
    805110616.390966.mp4"""
    from loeuf_cv.multi_person import _merge_split_person_boxes

    upper = (284, 174, 24, 39)
    lower = (294, 209, 34, 107)
    merged = _merge_split_person_boxes([upper, lower])
    assert len(merged) == 1
    x, y, w, h = merged[0]
    assert x <= 284 and y <= 174
    assert x + w >= 294 + 34
    assert y + h >= 209 + 107


def test_merge_split_person_boxes_keeps_separate_people_apart():
    """สองคนยืนคนละตำแหน่งชัดเจน (x ไม่ overlap กันเลย) ต้องไม่ถูกรวม —
    กัน _merge_split_person_boxes ไม่ให้ merge คนสองคนที่ยืนห่างกันจริง"""
    from loeuf_cv.multi_person import _merge_split_person_boxes

    person_a = (100, 200, 40, 100)
    person_b = (400, 200, 40, 100)
    merged = _merge_split_person_boxes([person_a, person_b])
    assert len(merged) == 2


def test_ground_homography_and_depth_roundtrip():
    from loeuf_cv.court_calibration import (
        WORLD_POINTS, ground_depth_from_image_point, solve_ground_homography,
    )

    # จำลองจุดภาพ (perspective หยาบ ๆ: ยิ่งลึกยิ่งแคบเข้ากลางและสูงขึ้นในภาพ)
    image_points = {
        "near_baseline_left": (150, 500),
        "near_baseline_right": (810, 500),
        "service_line_left": (330, 260),
        "service_line_right": (630, 260),
    }
    H = solve_ground_homography(image_points)
    assert H is not None

    # จุดที่ใช้แก้ homography ต้อง map กลับไปยังพิกัดโลกเดิมได้แม่น
    for name, (u, v) in image_points.items():
        x, z = ground_depth_from_image_point(H, (u, v))
        wx, wz = WORLD_POINTS[name]
        assert abs(x - wx) < 1e-6, f"{name}: X {x} != {wx}"
        assert abs(z - wz) < 1e-6, f"{name}: Z {z} != {wz}"


def test_homography_needs_four_points():
    from loeuf_cv.court_calibration import solve_ground_homography
    only_three = {"near_baseline_left": (150, 500),
                  "near_baseline_right": (810, 500),
                  "service_line_left": (330, 260)}
    assert solve_ground_homography(only_three) is None


def test_focal_length_and_height_roundtrip():
    from loeuf_cv.court_calibration import (
        estimate_focal_length_px, estimate_height_cm,
    )

    # สมมติ f_px จริง แล้วสร้าง pixel_height จากสูตร pinhole ไปข้างหน้า
    true_f_px = 900.0
    net_depth_m, net_real_height_m = 11.885, 0.914
    net_pixel_height = true_f_px * net_real_height_m / net_depth_m

    f_px = estimate_focal_length_px(net_pixel_height, net_depth_m, net_real_height_m)
    assert abs(f_px - true_f_px) < 1e-6

    # ผู้เล่นสูง 175cm ยืนที่ความลึก 8m -> คำนวณ pixel height แล้ว invert กลับ
    true_height_m = 1.75
    player_depth_m = 8.0
    player_pixel_height = f_px * true_height_m / player_depth_m

    height_cm = estimate_height_cm(player_pixel_height, player_depth_m, f_px)
    assert abs(height_cm - 175.0) < 1e-6


def test_net_band_detection_synthetic():
    from loeuf_cv.court_calibration import detect_net_band

    frame = np.full((400, 600, 3), 200, np.uint8)  # พื้นหลังสว่าง
    frame[150:180, :] = 20                          # แถบเน็ตเข้ม สูง 30px
    h = detect_net_band(frame, x_col=300)
    assert h is not None
    assert 25 <= h <= 35, f"expected ~30px, got {h}"


def test_court_corner_detection_on_synthetic_court():
    from loeuf_cv.court_calibration import detect_court_lines, find_court_corners

    # วาดคอร์ทสังเคราะห์: พื้นเข้ม + เส้นขาวตามพิกัดภาพที่กำหนดไว้ล่วงหน้า
    frame = np.full((540, 960, 3), 60, np.uint8)   # พื้นสนามเข้ม
    true_corners = {
        "near_baseline_left": (150, 480),
        "near_baseline_right": (810, 480),
        "service_line_left": (330, 260),
        "service_line_right": (630, 260),
    }
    bl, br = true_corners["near_baseline_left"], true_corners["near_baseline_right"]
    sl, sr = true_corners["service_line_left"], true_corners["service_line_right"]
    thickness = 4
    cv2.line(frame, bl, br, (255, 255, 255), thickness)   # baseline
    cv2.line(frame, sl, sr, (255, 255, 255), thickness)   # service line
    cv2.line(frame, bl, sl, (255, 255, 255), thickness)   # left sideline
    cv2.line(frame, br, sr, (255, 255, 255), thickness)   # right sideline

    lines = detect_court_lines(frame)
    assert len(lines) > 0, "ควรหาเส้นเจอในคอร์ทสังเคราะห์"

    corners = find_court_corners(lines)
    assert len(corners) == 4, f"ควรหาจุดตัดครบ 4 จุด, ได้ {list(corners)}"
    for name, (u, v) in corners.items():
        tx, ty = true_corners[name]
        dist = np.hypot(u - tx, v - ty)
        assert dist < 15, f"{name}: detected {(u,v)} vs true {(tx,ty)}, dist={dist:.1f}px"


def test_calibrate_court_from_frame_end_to_end_synthetic():
    from loeuf_cv.court_calibration import (
        BASELINE_TO_NET_M, NET_HEIGHT_CENTER_M, calibrate_court_from_frame,
    )

    frame = np.full((540, 960, 3), 60, np.uint8)
    bl, br = (150, 480), (810, 480)
    sl, sr = (330, 260), (630, 260)
    cv2.line(frame, bl, br, (255, 255, 255), 4)
    cv2.line(frame, sl, sr, (255, 255, 255), 4)
    cv2.line(frame, bl, sl, (255, 255, 255), 4)
    cv2.line(frame, br, sr, (255, 255, 255), 4)
    # แถบเน็ตอยู่เหนือ service line เล็กน้อย (ระหว่าง service line กับขอบบน)
    cv2.rectangle(frame, (280, 180), (680, 200), (20, 20, 20), -1)

    cal = calibrate_court_from_frame(frame)
    assert cal.valid, cal.reason
    assert cal.focal_length_px is not None and cal.focal_length_px > 0

    depth_at_net = cal.depth_of((480, 190))
    # จุดที่ทดสอบอยู่แถวเน็ต ความลึกควรใกล้เคียง baseline-to-net (คร่าว ๆ)
    assert depth_at_net is not None
    assert 5.0 < depth_at_net < 20.0


def test_estimate_height_from_pose_with_known_calibration():
    from loeuf_cv.config import L_ANKLE, NOSE, R_ANKLE
    from loeuf_cv.court_calibration import (
        BASELINE_TO_SERVICE_M, CourtCalibration, estimate_height_from_pose,
        solve_ground_homography,
    )
    from loeuf_cv.pose_extractor import PoseTimeseries, VideoMeta

    image_points = {
        "near_baseline_left": (150, 500),
        "near_baseline_right": (810, 500),
        "service_line_left": (330, 260),
        "service_line_right": (630, 260),
    }
    H = solve_ground_homography(image_points)
    f_px = 900.0
    true_height_m = 1.75
    depth_m = BASELINE_TO_SERVICE_M
    ankle_u, ankle_v = image_points["service_line_left"]
    pixel_height = f_px * true_height_m / depth_m
    head_v = ankle_v - pixel_height

    meta = VideoMeta(path="fake.mp4", fps=30.0, frame_count=5, width=960, height=540)
    T = 5
    landmarks = np.zeros((T, 33, 3))
    landmarks[:, L_ANKLE] = (ankle_u / 960, ankle_v / 540, 0)
    landmarks[:, R_ANKLE] = (ankle_u / 960, ankle_v / 540, 0)
    landmarks[:, NOSE] = (ankle_u / 960, head_v / 540, 0)
    ts = PoseTimeseries(
        landmarks=landmarks, world_landmarks=np.zeros((T, 33, 3)),
        visibility=np.ones((T, 33)), timestamps_ms=np.arange(T) * 33.3,
        meta=meta)

    cal = CourtCalibration(homography=H, focal_length_px=f_px, valid=True)
    height_cm = estimate_height_from_pose(cal, ts, ready_frames=np.arange(T))
    assert height_cm is not None
    assert abs(height_cm - 175.0) < 1.0


def test_auto_detect_height_invalid_calibration_returns_none():
    from loeuf_cv.court_calibration import CourtCalibration, estimate_height_from_pose

    invalid_cal = CourtCalibration(None, None, False, reason="test")
    result = estimate_height_from_pose(invalid_cal, None, None)
    assert result is None


def test_find_court_corners_rejects_out_of_frame_intersection():
    """เส้นที่เกือบขนานกัน (misclassify) ทำให้จุดตัดยิงออกนอกเฟรม —
    ต้องถูกปฏิเสธทั้งชุด ไม่ใช่ปล่อยผ่านแบบเงียบ ๆ"""
    from loeuf_cv.court_calibration import find_court_corners

    # baseline เกือบขนานกับ "sideline" (ต่างกันแค่ไม่กี่องศา) → จุดตัดไกลลิบ
    baseline = (100, 480, 800, 478)
    almost_parallel_left = (100, 480, 800, 470)   # เกือบขนาน baseline
    service_line = (300, 260, 600, 258)
    right_line = (630, 260, 812, 478)             # sideline ปกติ

    corners_no_bounds = find_court_corners(
        [baseline, almost_parallel_left, service_line, right_line])
    corners_with_bounds = find_court_corners(
        [baseline, almost_parallel_left, service_line, right_line],
        frame_w=960, frame_h=540)
    assert corners_with_bounds == {}, "ควรถูกปฏิเสธเพราะจุดตัดหลุดขอบเฟรม"


def test_build_reference_frame_removes_moving_occluder():
    from loeuf_cv.court_calibration import build_reference_frame

    T, H, W = 9, 100, 100
    frames = []
    for i in range(T):
        f = np.full((H, W, 3), 50, np.uint8)          # พื้นหลังนิ่ง (เข้ม)
        f[40:45, :] = 220                             # เส้นคอร์ทนิ่ง (สว่าง)
        ox = (i * 12) % W                             # วัตถุขยับ (ผู้เล่น)
        f[10:30, ox:ox + 8] = 255
        frames.append(f)

    ref = build_reference_frame(frames)
    # เส้นนิ่งต้องเหลืออยู่ชัดเจน (median ของค่าคงที่ = ค่าคงที่)
    assert ref[42, 50, 0] > 200
    # บริเวณที่วัตถุขยับผ่าน ต้องไม่ถูกจัดว่า "สว่างเหมือนวัตถุ" อีกต่อไป
    # (median เห็นพื้นหลังบ่อยกว่าวัตถุที่ผ่านเป็นครั้งคราว)
    assert ref[20, 50, 0] < 200


def test_auto_detect_height_cm_tries_fallback_ranges_and_gives_up_gracefully(tmp_path):
    from loeuf_cv.court_calibration import auto_detect_height_cm

    # วิดีโอปลอมที่ cv2.VideoCapture เปิดไม่ได้ (path ไม่มีจริง)
    # -> ต้อง fail gracefully คืน (None, invalid) ไม่ throw
    height_cm, cal = auto_detect_height_cm(
        str(tmp_path / "does_not_exist.mp4"), ts=None, config=None)
    assert height_cm is None
    assert cal.valid is False


def test_camera_pose_projection_sanity():
    from loeuf_cv.court_model import CameraPose

    frame_shape = (540, 960)
    pose = CameraPose(height_m=1.5, distance_m=3.0, tilt_deg=0.0,
                      pan_deg=0.0, focal_length_px=800)

    # จุดตรงหน้ากล้องพอดี (X=0, ความสูงเดียวกับกล้อง) -> กึ่งกลางเฟรมเป๊ะ
    center = pose.project((0, 10.0, 1.5), frame_shape)
    assert abs(center[0] - 480.0) < 1e-6
    assert abs(center[1] - 270.0) < 1e-6

    # จุดบนพื้น (ต่ำกว่ากล้อง) ต้องอยู่ใต้กึ่งกลางเฟรม (v มากกว่า)
    ground_near = pose.project((0, 10.0, 0.0), frame_shape)
    assert ground_near[1] > 270.0

    # จุดบนพื้นที่ไกลกว่า ต้องขยับเข้าใกล้ horizon (v ใกล้ 270 มากกว่า)
    ground_far = pose.project((0, 20.0, 0.0), frame_shape)
    assert 270.0 < ground_far[1] < ground_near[1]

    # จุดทางขวาของกล้อง ต้องมี u > กึ่งกลาง
    right_pt = pose.project((2.0, 10.0, 0.0), frame_shape)
    assert right_pt[0] > 480.0

    # จุดหลังกล้อง (Z < -distance_m) ต้องคืน None
    behind = pose.project((0, -10.0, 1.5), frame_shape)
    assert behind is None


def test_project_court_lines_predicts_unseen_parts():
    """เส้นที่ "ไม่เห็น" ในเฟรม (far baseline อยู่นอกจอ) ก็ยังทำนาย
    พิกัดออกมาได้ — นี่คือจุดที่ต่างจาก court_calibration.py เดิม"""
    from loeuf_cv.court_model import CameraPose, project_court_lines

    frame_shape = (540, 960)
    pose = CameraPose(height_m=1.5, distance_m=3.0, tilt_deg=5.0,
                      pan_deg=0.0, focal_length_px=900)
    lines = project_court_lines(
        pose, frame_shape, line_names=("near_baseline", "net_line", "far_baseline"))

    assert lines["net_line"] is not None
    # far_baseline อยู่ไกลลิบ (23.77m) แต่ยังคำนวณพิกัดภาพได้ (ไม่ None)
    # แม้ว่าอาจจะอยู่นอกขอบเฟรมจริง ๆ ก็ตาม
    assert lines["far_baseline"] is not None


def test_search_camera_pose_recovers_known_pose():
    """สร้าง edge map จากท่ากล้องที่รู้ค่าจริง แล้ว search ต้องหาท่าที่
    ใกล้เคียงได้ (คะแนน edge-alignment สูง)"""
    import cv2

    from loeuf_cv.court_model import (
        CameraPose, NEAR_COURT_LINES, build_edge_map, score_pose_against_edges,
        search_camera_pose,
    )

    frame_shape = (540, 960)
    # ท่ากล้องแบบ "ปกติ" (ไม่ใกล้/ก้มจนเส้นคอร์ทเลยขึ้นไปโดนโซนที่ถูก
    # mask เป็นพื้นหลัง — ดู COURT_SURFACE_TOP_FRAC)
    true_pose = CameraPose(height_m=1.3, distance_m=6.0, tilt_deg=4.0,
                           pan_deg=0.0, focal_length_px=1000)

    # วาดภาพสังเคราะห์: เส้นขาวตามที่ true_pose ทำนาย บนพื้นเข้ม
    frame = np.full((540, 960, 3), 60, np.uint8)
    from loeuf_cv.court_model import project_court_lines
    lines = project_court_lines(true_pose, frame_shape, NEAR_COURT_LINES)
    for seg in lines.values():
        if seg is None:
            continue
        p1, p2 = seg
        ok, c1, c2 = cv2.clipLine((0, 0, 960, 540),
                                  (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])))
        if ok:
            cv2.line(frame, c1, c2, (255, 255, 255), 3)

    edge_map = build_edge_map(frame)
    true_score = score_pose_against_edges(true_pose, edge_map, frame_shape)
    assert true_score > 0.8, f"true pose ควรได้คะแนนสูงกับ edge map ของตัวเอง: {true_score}"

    # หมายเหตุ: focal_length ↔ distance เป็น ambiguity แบบคลาสสิกใน
    # single-view calibration — ท่าที่ search เจออาจไม่ตรงกับ true_pose
    # เป๊ะ (พารามิเตอร์ต่างกันได้แต่ 2D projection เหมือนกัน) แต่ต้อง
    # ได้ alignment score สูงพอ ๆ กับ true pose เอง (validate ว่า search
    # machinery หาท่าที่ "อธิบายภาพที่เห็น" ได้ดีจริง แม้พารามิเตอร์เป๊ะ
    # อาจกู้คืนไม่ได้ — ดู README/CHANGELOG สำหรับนัยที่มีต่อ auto-height)
    #
    # search แบบ random+refine มีความไม่แน่นอน (บาง seed ติด local
    # optimum) — validated ว่ารันหลายรอบอิสระแล้วเลือกดีที่สุดช่วยได้
    # จริง (เป็นแนวทางที่แนะนำให้ใช้จริงด้วย เพราะแต่ละรอบเร็ว ~2-3 วิ)
    best_score = -1.0
    for seed in (42, 100):
        _pose, score = search_camera_pose(
            edge_map, frame_shape, n_random=10000, n_refine_iters=300,
            n_multistart=20, rng=np.random.default_rng(seed))
        best_score = max(best_score, score)
    assert best_score > 0.8, f"search (best of 2 runs) ควรได้ score สูง: {best_score}"


def test_point_on_line_interpolation():
    from loeuf_cv.court_model import COURT_LINES, point_on_line

    a, b = COURT_LINES["near_baseline"]
    assert point_on_line("near_baseline", 0.0) == a
    assert point_on_line("near_baseline", 1.0) == b
    mid = point_on_line("near_baseline", 0.5)
    assert mid == ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2)


def test_named_points_consistent_with_court_lines():
    """NAMED_POINTS (ใช้โดย court_click_tool.html) ต้องตรงกับ COURT_LINES
    เดิม — กันไม่ให้แก้ไฟล์หนึ่งแล้วลืมอีกไฟล์"""
    from loeuf_cv.court_model import COURT_LINES, NAMED_POINTS, NET_HEIGHT_CENTER_M

    assert NAMED_POINTS["near_baseline_left_doubles"] == COURT_LINES["near_baseline"][0]
    assert NAMED_POINTS["near_baseline_right_doubles"] == COURT_LINES["near_baseline"][1]
    assert NAMED_POINTS["net_post_left_base"] == COURT_LINES["net_post_left"][0]
    assert NAMED_POINTS["net_post_left_top"] == COURT_LINES["net_post_left"][1]
    # net_center_top คือกึ่งกลางเน็ต (ความสูงลดหย่อนกลางเน็ต 0.914m) ไม่ใช่
    # จุดปลายของ net_top ที่อยู่ที่เสา (1.07m) — คนละจุดกันโดยตั้งใจ
    assert NAMED_POINTS["net_center_top"][2] == NET_HEIGHT_CENTER_M
    assert len(NAMED_POINTS) == 18


def test_reprojection_error_zero_for_exact_correspondences():
    from loeuf_cv.court_model import (
        CameraPose, PointCorrespondence, reprojection_error_px,
        score_pose_against_points,
    )

    frame_shape = (540, 960)
    pose = CameraPose(height_m=1.3, distance_m=6.0, tilt_deg=4.0,
                      pan_deg=0.0, focal_length_px=1000)
    world_points = [
        (-4.115, 0.0, 0.0), (4.115, 0.0, 0.0),
        (-4.115, 5.485, 0.0), (4.115, 5.485, 0.0),
        (0.0, 11.885, 0.0),
    ]
    corrs = [PointCorrespondence(pose.project(wp, frame_shape), wp)
             for wp in world_points]

    assert reprojection_error_px(pose, corrs, frame_shape) < 1e-6
    assert score_pose_against_points(pose, corrs, frame_shape) > 0.999


def test_search_camera_pose_from_points_too_few_correspondences():
    from loeuf_cv.court_model import (
        CameraPose, PointCorrespondence, search_camera_pose_from_points,
    )

    frame_shape = (540, 960)
    pose = CameraPose(height_m=1.3, distance_m=6.0, tilt_deg=4.0,
                      pan_deg=0.0, focal_length_px=1000)
    corrs = [PointCorrespondence(pose.project(wp, frame_shape), wp)
             for wp in [(-4.115, 0.0, 0.0), (4.115, 0.0, 0.0)]]

    result_pose, score = search_camera_pose_from_points(corrs, frame_shape)
    assert result_pose is None
    assert score == 0.0


def test_search_camera_pose_from_points_recovers_known_pose_and_net_point_helps():
    """คนคลิกจุดที่เห็นได้จริง (ไม่ต้องครบ 4 มุม/ทุกเส้น) -> search หาท่า
    กล้องที่อธิบายจุดเหล่านั้นได้ดี และรวมจุดบนเน็ต (สูงจากพื้น จึง
    non-coplanar กับจุดอื่น) ช่วยตัด focal-length↔distance ambiguity"""
    from loeuf_cv.court_model import (
        CameraPose, NET_HEIGHT_POST_M, PointCorrespondence,
        search_camera_pose_from_points,
    )

    frame_shape = (540, 960)
    true_pose = CameraPose(height_m=1.3, distance_m=6.0, tilt_deg=4.0,
                           pan_deg=0.0, focal_length_px=1000)

    ground_points = [
        (-4.115, 0.0, 0.0), (4.115, 0.0, 0.0),
        (-4.115, 5.485, 0.0), (4.115, 5.485, 0.0),
        (0.0, 5.485, 0.0), (0.0, 11.885, 0.0),
    ]
    net_point = (5.485, 11.885, NET_HEIGHT_POST_M)  # เสาเน็ต — จุดเดียวที่ยกสูงจากพื้น

    def to_corrs(world_points):
        return [PointCorrespondence(true_pose.project(wp, frame_shape), wp)
                for wp in world_points]

    corrs_with_net = to_corrs(ground_points + [net_point])

    best_pose, best_score = None, -1.0
    for seed in (42, 100):
        pose, score = search_camera_pose_from_points(
            corrs_with_net, frame_shape, n_random=6000, n_refine_iters=250,
            n_multistart=15, rng=np.random.default_rng(seed))
        if score > best_score:
            best_pose, best_score = pose, score

    assert best_score > 0.95, f"search ควรอธิบายจุดที่คลิกได้แม่นมาก: {best_score}"
    # มีจุดเน็ต (non-coplanar) ช่วยตัด ambiguity -> พารามิเตอร์ที่กู้คืนได้
    # ควรใกล้เคียง true_pose จริง ไม่ใช่แค่ 2D projection ตรงกันเฉย ๆ
    assert abs(best_pose.focal_length_px - 1000) < 250
    assert abs(best_pose.height_m - 1.3) < 0.3
    assert abs(best_pose.distance_m - 6.0) < 1.0


def test_depth_spread_m():
    from loeuf_cv.court_model import NAMED_POINTS, PointCorrespondence, depth_spread_m

    net_only = [PointCorrespondence((0, 0), NAMED_POINTS[name]) for name in (
        "net_post_left_base", "net_post_left_top",
        "net_post_right_base", "net_post_right_top")]
    assert depth_spread_m(net_only) == 0.0  # ทุกจุดอยู่ Z=BASELINE_TO_NET_M เดียวกัน

    mixed = net_only + [PointCorrespondence((0, 0), NAMED_POINTS["near_baseline_left_doubles"])]
    assert depth_spread_m(mixed) > 10.0  # near_baseline (Z=0) ห่างจากเน็ต (Z=11.885) มาก


def test_same_depth_points_are_degenerate_even_without_noise():
    """ข้อค้นพบสำคัญจากการ validate จริง (2026-07-09): คลิกแค่ 4 มุมเสาเน็ต
    (Z เดียวกันหมด) เป็น degenerate case จริง ไม่ใช่แค่ noise-sensitive —
    แม้ไม่มี noise เลย search ก็ยังหา pose ผิด (ไม่ตรง true_pose) ที่ยัง
    ได้ score สูงพอกันได้ เพราะไม่มี depth ต่างกันให้แยก focal↔distance"""
    from loeuf_cv.court_model import (
        CameraPose, NAMED_POINTS, PointCorrespondence, search_camera_pose_from_points,
    )

    frame_shape = (540, 960)
    true_pose = CameraPose(height_m=1.3, distance_m=6.0, tilt_deg=4.0,
                           pan_deg=0.0, focal_length_px=1000)
    net_only = ["net_post_left_base", "net_post_left_top",
               "net_post_right_base", "net_post_right_top"]
    corrs = [PointCorrespondence(true_pose.project(NAMED_POINTS[n], frame_shape),
                                NAMED_POINTS[n]) for n in net_only]

    pose, score = search_camera_pose_from_points(
        corrs, frame_shape, n_random=6000, n_refine_iters=250,
        n_multistart=15, rng=np.random.default_rng(42))

    assert score > 0.9  # อธิบายจุดที่คลิกได้ดี (score สูง)...
    # ...แต่พารามิเตอร์ที่ได้ไม่ตรง true_pose เลย (นี่คือประเด็น: score สูง
    # ไม่ได้แปลว่าพารามิเตอร์ถูก เมื่อจุดไม่กระจายความลึกพอ)
    assert abs(pose.focal_length_px - 1000) > 100 or abs(pose.distance_m - 6.0) > 1.0


def _render_synthetic_court_frame(true_pose, frame_shape):
    import cv2

    from loeuf_cv.court_model import NEAR_COURT_LINES, project_court_lines

    frame = np.full((*frame_shape, 3), 60, np.uint8)
    lines = project_court_lines(true_pose, frame_shape, NEAR_COURT_LINES)
    for seg in lines.values():
        if seg is None:
            continue
        p1, p2 = seg
        ok, c1, c2 = cv2.clipLine((0, 0, frame_shape[1], frame_shape[0]),
                                  (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])))
        if ok:
            cv2.line(frame, c1, c2, (255, 255, 255), 3)
    return frame


def test_focal_length_from_vanishing_points_exact_with_clean_vps():
    """สูตรปิด (closed-form) ต้องคืน focal length ตรงเป๊ะเมื่อ VP มาจาก
    เส้นที่ตั้งฉากกันจริงแบบไม่มี noise เลย (ground-truth geometry)"""
    from loeuf_cv.court_model import CameraPose, COURT_LINES
    from loeuf_cv.vanishing_point import (
        _line_intersection, focal_length_from_vanishing_points,
    )

    frame_shape = (540, 960)
    true_pose = CameraPose(height_m=1.3, distance_m=6.0, tilt_deg=4.0,
                           pan_deg=8.0, focal_length_px=1000)

    def seg_of(name):
        a, b = COURT_LINES[name]
        return (*true_pose.project(a, frame_shape), *true_pose.project(b, frame_shape))

    vp1 = _line_intersection(seg_of("near_baseline"), seg_of("near_service_line"))
    vp2 = _line_intersection(seg_of("near_singles_sideline_left"),
                             seg_of("near_singles_sideline_right"))
    f = focal_length_from_vanishing_points(vp1, vp2, (480.0, 270.0))
    assert f is not None
    assert abs(f - 1000) < 1e-3


def test_vanishing_point_reliability_gate_rejects_centered_camera():
    """⚠️ ข้อค้นพบสำคัญ (2026-07-09): setup กล้องมาตรฐานของโปรเจกต์นี้ —
    อยู่กึ่งกลางหลัง baseline พอดี (pan_deg≈0) — ทำให้ vanishing point ของ
    เส้นทิศทางกว้างอยู่เกือบ infinity ระบบต้อง "รู้ตัว" และคืน None แทน
    การคืนค่า focal length ที่ไม่น่าเชื่อถือ"""
    from loeuf_cv.court_model import CameraPose
    from loeuf_cv.vanishing_point import estimate_focal_length_from_frame

    frame_shape = (540, 960)
    centered_pose = CameraPose(height_m=1.3, distance_m=6.0, tilt_deg=4.0,
                               pan_deg=0.0, focal_length_px=1000)
    frame = _render_synthetic_court_frame(centered_pose, frame_shape)

    result = estimate_focal_length_from_frame(
        frame, frame_shape, rng=np.random.default_rng(42))
    assert result is None


def test_vanishing_point_reliability_gate_accepts_well_conditioned_pan():
    """เมื่อกล้องมี pan มากพอ (มุมเบี่ยงจากเส้นกึ่งกลางคอร์ทชัดเจน) VP ของ
    เส้นทิศทางกว้างไม่ใกล้ infinity แล้ว — ระบบควรให้ผลลัพธ์ที่ใช้ได้"""
    from loeuf_cv.court_model import CameraPose
    from loeuf_cv.vanishing_point import estimate_focal_length_from_frame

    frame_shape = (540, 960)
    angled_pose = CameraPose(height_m=1.3, distance_m=6.0, tilt_deg=4.0,
                             pan_deg=15.0, focal_length_px=1000)
    frame = _render_synthetic_court_frame(angled_pose, frame_shape)

    result = estimate_focal_length_from_frame(
        frame, frame_shape, rng=np.random.default_rng(42))
    assert result is not None
    assert abs(result - 1000) / 1000 < 0.2


def test_kinematic_fill_short_gap_beats_leaving_it_noisy():
    """gap สั้น (<=MAX_KINEMATIC_GAP_FRAMES): เติมด้วย blend(kinematic,
    linear) ควรใกล้ตำแหน่งจริงมากกว่าปล่อยค่า noisy เดิมไว้ — validated
    ด้วย held-out test บนคลิปจริงแล้ว (2026-07-10, ดู occlusion_fill.py
    docstring) นี่คือเทสสังเคราะห์ตรวจ mechanic เท่านั้น"""
    from loeuf_cv.config import R_ELBOW, R_WRIST
    from loeuf_cv.occlusion_fill import kinematic_fill_joint
    from loeuf_cv.synthetic import synthetic_forehand

    ts = synthetic_forehand(fps=60.0, duration_s=3.0)
    true_wrist = ts.landmarks[:, R_WRIST, :2].copy()
    noisy = ts.landmarks.copy()
    vis = ts.visibility.copy()

    gap_start, gap_len = 40, 15
    gap = slice(gap_start, gap_start + gap_len)
    rng = np.random.default_rng(0)
    noisy[gap, R_WRIST, :2] += rng.normal(0, 0.15, size=(gap_len, 2))  # noisy ตำแหน่งช่วง occlude
    vis[gap, R_WRIST] = 0.1  # ต่ำกว่า LOW_VISIBILITY_THRESHOLD

    filled = kinematic_fill_joint(noisy, vis, R_WRIST, R_ELBOW)

    err_before = np.linalg.norm(noisy[gap, R_WRIST, :2] - true_wrist[gap], axis=1).mean()
    err_after = np.linalg.norm(filled[gap, R_WRIST, :2] - true_wrist[gap], axis=1).mean()
    assert err_after < err_before


def test_kinematic_fill_respects_max_gap_and_boundaries():
    from loeuf_cv.config import R_ELBOW, R_WRIST
    from loeuf_cv.occlusion_fill import MAX_KINEMATIC_GAP_FRAMES, kinematic_fill_joint
    from loeuf_cv.synthetic import synthetic_forehand

    ts = synthetic_forehand(fps=60.0, duration_s=3.0)
    coords = ts.landmarks.copy()
    vis = ts.visibility.copy()
    original = coords.copy()

    # gap ยาวเกิน max — ไม่ควรถูกแก้เลย
    long_gap = slice(40, 40 + MAX_KINEMATIC_GAP_FRAMES + 10)
    vis[long_gap, R_WRIST] = 0.1
    filled = kinematic_fill_joint(coords, vis, R_WRIST, R_ELBOW)
    assert np.allclose(filled[long_gap, R_WRIST, :], original[long_gap, R_WRIST, :])

    # gap ติดขอบเริ่มคลิป (ไม่มีขอบซ้าย) — ไม่ควรถูกแก้เช่นกัน
    vis2 = ts.visibility.copy()
    vis2[:10, R_WRIST] = 0.1
    filled2 = kinematic_fill_joint(coords, vis2, R_WRIST, R_ELBOW)
    assert np.allclose(filled2[:10, R_WRIST, :], original[:10, R_WRIST, :])


def test_kinematic_fill_preserves_visibility_array():
    """ตั้งใจไม่แก้ visibility เดิม — ให้ downstream confidence flag
    (CF1/CF3/VF) ยังรายงานว่าเป็นค่าประมาณ ไม่ใช่ detect ได้จริง"""
    from loeuf_cv.occlusion_fill import kinematic_fill
    from loeuf_cv.synthetic import synthetic_forehand

    ts = synthetic_forehand(fps=60.0, duration_s=3.0)
    ts.visibility[40:50, :] = 0.1
    out = kinematic_fill(ts)
    assert np.array_equal(out.visibility, ts.visibility)


def test_one_euro_filter_reduces_jitter_on_noisy_static_signal():
    """สัญญาณนิ่ง (ค่าคงที่) + noise แรง — filter ต้องลด jitter ได้จริง
    (frame-to-frame delta variance ลดลง) เทียบกับสัญญาณดิบ"""
    from loeuf_cv.one_euro_filter import OneEuroFilter

    rng = np.random.default_rng(0)
    n = 200
    raw = 0.5 + rng.normal(0, 0.02, n)  # นิ่งที่ 0.5 + noise
    f = OneEuroFilter(mincutoff=1.0, beta=0.0, dcutoff=1.0)
    filtered = np.array([f(x, timestamp_s=i / 30.0) for i, x in enumerate(raw)])

    raw_jitter = np.std(np.diff(raw))
    filtered_jitter = np.std(np.diff(filtered[20:]))  # ตัด warm-up ช่วงแรกทิ้ง
    assert filtered_jitter < raw_jitter * 0.5


def test_one_euro_filter_beta_reduces_lag_on_fast_ramp():
    """คุณสมบัติหลักของ 1€ filter: beta สูง = cutoff ปรับตามความเร็ว =
    lag น้อยลงตอนสัญญาณเปลี่ยนเร็ว (จำลอง swing เร็วช่วง backswing→impact)
    — beta=0.3 ต้อง track ปลายสัญญาณ ramp ได้ใกล้กว่า beta=0"""
    from loeuf_cv.one_euro_filter import OneEuroFilter

    n = 60
    ts_s = np.arange(n) / 60.0
    ramp = np.concatenate([np.zeros(20), np.linspace(0, 1, 20), np.ones(20)])

    f_no_beta = OneEuroFilter(mincutoff=1.0, beta=0.0, dcutoff=1.0)
    f_with_beta = OneEuroFilter(mincutoff=1.0, beta=2.0, dcutoff=1.0)
    out_no_beta = np.array([f_no_beta(x, timestamp_s=t) for x, t in zip(ramp, ts_s)])
    out_with_beta = np.array([f_with_beta(x, timestamp_s=t) for x, t in zip(ramp, ts_s)])

    # ที่ปลาย ramp (เพิ่งขึ้นถึง 1.0) beta สูงควร "ตามทัน" ได้ดีกว่า (error น้อยกว่า)
    err_no_beta = abs(out_no_beta[39] - ramp[39])
    err_with_beta = abs(out_with_beta[39] - ramp[39])
    assert err_with_beta < err_no_beta


def test_one_euro_filter_confidence_weighting_pulls_toward_trend():
    """เฟรม visibility ต่ำ ต้องถูกดึงเข้าหาค่าที่ filter คาดไว้ (persistence)
    แทนที่จะตามค่าดิบที่ noisy เต็มที่ — validate ว่าค่า noisy ที่ confidence
    ต่ำมีผลต่อ output น้อยกว่าตอน confidence สูง"""
    from loeuf_cv.one_euro_filter import _filter_1d_confidence_weighted

    n = 30
    series = np.full(n, 0.5)
    series[15] = 5.0  # spike ผิดปกติ 1 เฟรม
    ts_ms = np.arange(n) / 30.0 * 1000.0

    vis_high = np.full(n, 0.9)
    vis_low = np.full(n, 0.9)
    vis_low[15] = 0.1  # เฟรมเดียวกับ spike — confidence ต่ำ

    out_high_conf = _filter_1d_confidence_weighted(
        series, vis_high, ts_ms, mincutoff=1.0, beta=0.0, dcutoff=1.0, low_vis_threshold=0.5)
    out_low_conf = _filter_1d_confidence_weighted(
        series, vis_low, ts_ms, mincutoff=1.0, beta=0.0, dcutoff=1.0, low_vis_threshold=0.5)

    # spike ตอน confidence ต่ำ ต้องกระทบผลลัพธ์น้อยกว่าตอน confidence สูง
    assert out_low_conf[15] < out_high_conf[15]


def test_one_euro_filter_handles_nan_with_persistence():
    """ไม่มีค่าเลย (NaN) ต้องใช้ค่าที่ filter เก็บไว้ล่าสุดแทน ไม่ crash
    และไม่ทำให้ output เป็น NaN ต่อเนื่องเรื่อย ๆ"""
    from loeuf_cv.one_euro_filter import _filter_1d_confidence_weighted

    n = 20
    series = np.full(n, 0.5)
    series[10:13] = np.nan
    vis = np.full(n, 0.9)
    vis[10:13] = 0.0
    ts_ms = np.arange(n) / 30.0 * 1000.0

    out = _filter_1d_confidence_weighted(
        series, vis, ts_ms, mincutoff=1.0, beta=0.0, dcutoff=1.0, low_vis_threshold=0.5)
    assert not np.isnan(out[10:13]).any()
    assert np.isnan(out[:5]).sum() == 0  # ช่วงก่อนหน้าไม่ใช่ NaN อยู่แล้ว ต้องไม่เป็น NaN


def test_apply_one_euro_filter_opt_in_does_not_change_default_pipeline():
    """smoothing_method default ต้องยังเป็น savgol เดิม — ไม่แตะ behavior
    เดิมโดยไม่ตั้งใจ (opt-in เท่านั้น)"""
    from loeuf_cv.config import PipelineConfig

    config = PipelineConfig()
    assert config.smoothing_method == "savgol"


def _fake_player_track(n_frames=100, fps=30.0, width=1920, height=1080):
    from loeuf_cv.multi_person import PlayerTrack
    from loeuf_cv.pose_extractor import PoseTimeseries, VideoMeta

    meta = VideoMeta(path="fake.mp4", fps=fps, frame_count=n_frames, width=width, height=height)
    landmarks = np.zeros((n_frames, 33, 3)) + 0.5
    ts = PoseTimeseries(
        landmarks=landmarks, world_landmarks=np.zeros((n_frames, 33, 3)),
        visibility=np.ones((n_frames, 33)) * 0.9,
        timestamps_ms=np.arange(n_frames) * (1000.0 / fps), meta=meta)
    return PlayerTrack(track_id=1, role="near", pose=ts, n_frames_tracked=n_frames)


def test_build_loeuf_schema_ball_block_uses_real_tracked_fraction():
    """schema_builder.builder.build_loeuf_schema เดิมส่ง "ball": {"available": True}
    แบบ hardcode เสมอไม่ว่าจะเจอลูกจริงหรือไม่ — ตอนนี้ควรคำนวณจาก ball_traj
    จริง (Kalman trajectory) สโคปตามช่วงเฟรมของแต่ละ stroke"""
    from loeuf_cv.config import PipelineConfig
    from loeuf_cv.schema_builder.builder import build_loeuf_schema

    n = 100
    track = _fake_player_track(n_frames=n)
    hit_events = [{"frame": 50, "player_id": 1, "confidence": 0.9}]
    video_meta = {"width": 1920, "height": 1080, "total_frames": n}
    cfg = PipelineConfig()

    ball_traj = np.full((n, 2), np.nan)
    ball_traj[20:60, 0] = np.linspace(500, 1400, 40)
    ball_traj[20:60, 1] = np.linspace(300, 700, 40)

    with_ball = build_loeuf_schema([track], hit_events, 30.0, video_meta, cfg, ball_traj=ball_traj)
    without_ball = build_loeuf_schema([track], hit_events, 30.0, video_meta, cfg, ball_traj=None)

    ball_block = with_ball["strokes"][0]["ball"]
    assert ball_block["available"] is True
    assert 0.0 < ball_block["_tracked_fraction"] <= 1.0
    # ยังไม่มี court calibration — field พวกนี้ต้องเป็น None เสมอ ไม่เดาค่า
    assert ball_block["ball_speed_kmh"] is None
    assert ball_block["landing_position"] is None

    # ไม่ส่ง ball_traj มา -> fallback ตาม BL1 gate (available False) แต่ key ต้อง
    # ครบทุกตัวเหมือนตอนมี ball (spec: ห้าม omit key)
    from loeuf_cv.schema_fields import BALL_FIELDS
    no_ball_block = without_ball["strokes"][0]["ball"]
    assert set(no_ball_block) == set(BALL_FIELDS)
    assert no_ball_block["available"] is False
    assert all(no_ball_block[k] is None for k in BALL_FIELDS if k != "available")
    assert set(ball_block) == set(BALL_FIELDS)


def test_landing_zone_geometry():
    from loeuf_cv.schema_builder.builder import _landing_zone

    assert _landing_zone(1.0, 3.0) == "near_right_service_box"
    assert _landing_zone(-1.0, 3.0) == "near_left_service_box"
    assert _landing_zone(1.0, 8.0) == "near_right_backcourt"
    assert _landing_zone(1.0, 15.0) == "far_right_service_box"
    assert _landing_zone(1.0, 20.0) == "far_right_backcourt"
    assert _landing_zone(6.0, 3.0) == "out_of_bounds"
    assert _landing_zone(0.0, 30.0) == "out_of_bounds"


def test_build_loeuf_schema_wires_landing_from_bounce_data():
    """BL2-4 (landing_position/landing_zone/landing_call) ควรมาจาก
    bounce_court_x_m/z_m/bounce_in_court ที่ add_bounce_to_hits() เติมใน
    hit_events ไว้แล้ว (ต้องมี court_homography ตอน detect bounce) — ไม่ใช่
    หน้าที่ของ build_loeuf_schema เองที่จะคำนวณ homography ใหม่"""
    from loeuf_cv.config import PipelineConfig
    from loeuf_cv.schema_builder.builder import build_loeuf_schema

    n = 100
    track = _fake_player_track(n_frames=n)
    hit_events = [{
        "frame": 50, "player_id": 1, "confidence": 0.9,
        "bounce_court_x_m": 1.5, "bounce_court_z_m": 3.0, "bounce_in_court": True,
    }]
    video_meta = {"width": 1920, "height": 1080, "total_frames": n}
    cfg = PipelineConfig()

    ball_traj = np.full((n, 2), np.nan)
    ball_traj[20:60, 0] = np.linspace(500, 1400, 40)
    ball_traj[20:60, 1] = np.linspace(300, 700, 40)

    out = build_loeuf_schema([track], hit_events, 30.0, video_meta, cfg, ball_traj=ball_traj)
    ball_block = out["strokes"][0]["ball"]
    assert ball_block["landing_position"] == {"x_m": 1.5, "z_m": 3.0}
    assert ball_block["landing_zone"] == "near_right_service_box"
    assert ball_block["landing_call"] == "in"
    # BL อื่นที่ต้องการ depth/height เหนือพื้น ยังคง None เสมอ (ดู docstring)
    assert ball_block["ball_speed_kmh"] is None
    assert ball_block["trajectory_clearance_cm"] is None


def test_schema_field_manifest_matches_client_spec():
    """Freeze test — จุดเดียวในโค้ดเบสที่ hardcode ชื่อ field ตาม client doc
    เทสอื่นให้ import จาก schema_fields เอา ถ้า manifest เพี้ยน เทสนี้จะจับได้
    ที่เดียว (ถ้า hardcode ทุกเทส เวลา schema เปลี่ยนต้องแก้ 4 ที่ = ต้นเหตุ
    ของ drift เดิม)"""
    from loeuf_cv import schema_fields as sf

    assert sf.KEYFRAME_ENTRY_FIELDS == (
        "frame_index", "timestamp_ms", "detected", "joint")
    assert sf.KEYFRAME_BLOCK_NAMES == (
        "unit_turn", "backswing_peak", "impact", "follow_through_peak",
        "recovery_position", "trophy_position")
    # A7 = B1-B5, A8 = B1-B4 — trophy_position (B6, serve-only) ต้องไม่อยู่ในสองอันนี้
    assert "trophy_position" not in sf.A7_KEYFRAME_NAMES
    assert "trophy_position" not in sf.A8_KEYFRAME_NAMES
    assert len(sf.A7_KEYFRAME_NAMES) == 5 and len(sf.A8_KEYFRAME_NAMES) == 4

    assert len(sf.JOINT_NAMES) == 13
    assert set(sf.JOINT_NAMES) == {
        "head",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"}
    assert set(sf.JOINT_LANDMARK_IDS) == set(sf.JOINT_NAMES)
    assert sf.JOINT_LANDMARK_IDS["head"] == 0          # NOSE
    assert sf.JOINT_LANDMARK_IDS["right_wrist"] == 16  # R_WRIST

    for bl in ("server_position", "target_box_correct",
               "serve_attempt_number", "is_fault"):
        assert bl in sf.BALL_FIELDS, f"BL7-BL10: {bl} หายจาก manifest"

    assert sf.VZ_FIELDS == ("wrist_path", "ball_path", "racket_tip_path",
                            "racket_center_path", "body_center_path")

    empty = sf.empty_ball_block()
    assert set(empty) == set(sf.BALL_FIELDS)
    assert empty["available"] is False


def test_stroke_specific_fields_complete_per_stroke_type():
    """SS: doc ระบุ FH/BH/RS ไม่มี field เพิ่ม = {} ส่วน SV/VL/SL ต้องส่งครบ
    ทุก field ของ type นั้นเสมอ (ตรวจไม่ได้ → null ไม่ใช่ omit)"""
    from loeuf_cv.config import STROKE_TYPES
    from loeuf_cv.schema_fields import STROKE_SPECIFIC_FIELDS

    for stroke_type in STROKE_TYPES:
        ss = PIPELINE.process_timeseries(
            synthetic_forehand(fps=30), stroke_type)["metrics"]["stroke_specific"]
        expected = set(STROKE_SPECIFIC_FIELDS[stroke_type])
        assert set(ss) == expected, f"{stroke_type}: SS key ไม่ตรง manifest"


def test_build_loeuf_schema_never_omits_keys():
    """spec: "ส่งทุก field เสมอ — ห้าม omit key" — key set ของ B / BL / VZ / MT10
    ต้องเหมือนกันเป๊ะทั้งตอนมี ball_traj และไม่มี ball_traj (เคสที่พังเดิม)"""
    from loeuf_cv.config import STROKE_TYPES, PipelineConfig
    from loeuf_cv.schema_builder.builder import build_loeuf_schema
    from loeuf_cv.schema_fields import (
        BALL_FIELDS, JOINT_NAMES, KEYFRAME_BLOCK_NAMES, KEYFRAME_ENTRY_FIELDS,
        VZ_FIELDS,
    )

    n = 120
    track = _fake_player_track(n_frames=n)
    hit_events = [
        {"frame": 40, "player_id": 1, "confidence": 0.9},
        {"frame": 90, "player_id": 1, "confidence": 0.9},
    ]
    video_meta = {"width": 1920, "height": 1080, "total_frames": n}
    cfg = PipelineConfig()

    ball_traj = np.full((n, 2), np.nan)
    ball_traj[20:80, 0] = np.linspace(500, 1400, 60)
    ball_traj[20:80, 1] = np.linspace(300, 700, 60)

    outputs = {
        "with_ball": build_loeuf_schema([track], hit_events, 30.0, video_meta,
                                        cfg, ball_traj=ball_traj),
        "no_ball": build_loeuf_schema([track], hit_events, 30.0, video_meta,
                                      cfg, ball_traj=None),
    }

    for label, out in outputs.items():
        # MT10 ต้องมีครบทุก stroke type รวมตัวที่นับได้ 0
        dist = out["session_metadata"]["stroke_type_distribution"]
        assert set(dist) == set(STROKE_TYPES), f"{label}: MT10 ตกหล่น stroke type"

        assert out["strokes"], f"{label}: ไม่มี stroke เลย"
        for s in out["strokes"]:
            # --- 023-B ---
            assert set(s["keyframe"]) == set(KEYFRAME_BLOCK_NAMES), label
            for name, entry in s["keyframe"].items():
                assert set(entry) == set(KEYFRAME_ENTRY_FIELDS), f"{label}/{name}"
                if entry["detected"]:
                    assert entry["frame_index"] is not None
                    assert entry["timestamp_ms"] is not None
                    assert set(entry["joint"]) == set(JOINT_NAMES), f"{label}/{name}"
                    for jname, pt in entry["joint"].items():
                        assert pt is None or set(pt) == {"x", "y"}, f"{label}/{name}/{jname}"
                else:
                    assert entry["frame_index"] is None
                    assert entry["timestamp_ms"] is None
                    assert entry["joint"] is None

            # B6 serve-only — non-serve ต้อง undetected แต่ key ต้องอยู่
            if s["stroke_root"]["stroke_type"] != "SV":
                assert s["keyframe"]["trophy_position"]["detected"] is False

            # A7/A8 ต้องไม่ถูก B6 ทำพัง (non-serve ยัง valid ได้)
            assert s["stroke_root"]["detection_status"] in ("valid", "partial", "failed")

            # --- 034-BL ---
            assert set(s["ball"]) == set(BALL_FIELDS), label
            for bl in ("server_position", "target_box_correct",
                       "serve_attempt_number", "is_fault"):
                assert bl in s["ball"]

            # --- 043-VZ ---
            assert set(s["visualization"]) == set(VZ_FIELDS), label

    # key set ต้องเท่ากันเป๊ะระหว่างสองโหมด (เดิม no_ball หาย BL/VZ ไปหลาย key)
    a = outputs["with_ball"]["strokes"][0]
    b = outputs["no_ball"]["strokes"][0]
    assert set(a) == set(b)
    for block in ("keyframe", "ball", "visualization"):
        assert set(a[block]) == set(b[block]), f"{block} key set ต่างกันสองโหมด"


def test_keyframe_offsets_scale_with_fps():
    """ค่า offset ใน schema_builder/keyframes.py คาลิเบรตที่ 29.97fps

    ลูกค้าจะส่งคลิปชุดหน้าที่ >= 60fps (spec MT4) — ถ้าค่าพวกนี้ยังเป็น
    "จำนวนเฟรม" ตายตัว ระยะเวลาจริงจะเหลือครึ่งเดียวโดยไม่มี error ใด ๆ
    เทสต์นี้ยึดสองอย่างพร้อมกัน:
      1) ที่ 29.97fps ต้องได้ค่าเดิมเป๊ะ (benchmark ที่วัดไว้ไม่เป็นโมฆะ)
      2) ที่ 60fps ระยะห่างต้องเป็น "เวลาเท่ากัน" ไม่ใช่ "เฟรมเท่ากัน"
    """
    from loeuf_cv.schema_builder.keyframes import (
        CALIBRATION_FPS, FOLLOW_THROUGH_OFFSET, RECOVERY_OFFSET, TROPHY_OFFSET,
        refine_keyframes_for_type, trophy_position_frame)

    imp, n = 300, 900
    for stype in ("FH", "BH", "SL", "SV", "VL"):
        base = refine_keyframes_for_type({}, stype, imp, n, CALIBRATION_FPS)
        # 1) fps คาลิเบรต -> ตรงกับค่าคงที่ที่ประกาศไว้เป๊ะ
        assert base["follow_through_peak"] - imp == FOLLOW_THROUGH_OFFSET.get(
            stype, 10), stype

        fast = refine_keyframes_for_type({}, stype, imp, n, 60.0)
        for key in ("follow_through_peak", "recovery_position"):
            slow_ms = (base[key] - imp) / CALIBRATION_FPS
            fast_ms = (fast[key] - imp) / 60.0
            # 2) ระยะเวลาต้องเท่ากัน (คลาดได้ 1 เฟรมจากการปัดเศษ)
            assert abs(slow_ms - fast_ms) < 1.0 / 60.0 + 1e-9, \
                f"{stype}/{key}: {slow_ms*1000:.1f}ms vs {fast_ms*1000:.1f}ms"
        assert fast["recovery_position"] - imp == pytest.approx(
            RECOVERY_OFFSET.get(stype, 38) * 2, abs=1)

    assert trophy_position_frame("SV", imp, n, CALIBRATION_FPS) - imp == TROPHY_OFFSET
    assert trophy_position_frame("SV", imp, n, 60.0) - imp == pytest.approx(
        TROPHY_OFFSET * 2, abs=1)

    # ไม่ส่ง fps = ถือว่าเป็นคลิปคาลิเบรต (caller เก่าต้องไม่พฤติกรรมเปลี่ยน)
    assert refine_keyframes_for_type({}, "FH", imp, n) == \
        refine_keyframes_for_type({}, "FH", imp, n, CALIBRATION_FPS)


def test_hit_feature_units_are_fps_invariant():
    """ฟีเจอร์ที่ป้อน hit_classifier ต้องเป็นหน่วยเดียวกับตอนเทรน (29.97fps)

    speed_* เป็น "px ต่อเฟรม" — การเคลื่อนไหวเดียวกันที่ 60fps ให้ค่าครึ่งเดียว
    ถ้าไม่ชดเชย โมเดลที่เทรนจากคลิป 29.97fps ล้วนจะเจอ distribution คนละชุด
    (train/serve skew) แล้ว hit detection พังเงียบ ๆ ตอนคลิป 60fps มาถึง
    """
    from loeuf_cv.hit_detection import CALIBRATION_FPS, _scaled_features

    ctx = (100.0, {})   # bw = 100 px, ไม่มีสัญญาณต่อเฟรม
    SPEED_KEYS = ("wrist_speed", "speed_pre_mean", "speed_post_mean",
                  "speed_std_window")
    at_calib = {"wrist_speed": 30.0, "speed_pre_mean": 12.0,
                "speed_post_mean": 6.0, "speed_std_window": 4.0,
                "ball_dist": 50.0, "racket_dist": 20.0}
    # การเคลื่อนไหวทางกายภาพเดียวกัน วัดที่ 60fps -> px ต่อเฟรมลดลงตามสัดส่วน
    # (30 px/เฟรม ที่ 29.97fps = 899.1 px/วิ = 14.985 px/เฟรม ที่ 60fps)
    at_60 = dict(at_calib)
    for k in SPEED_KEYS:
        at_60[k] = at_calib[k] * CALIBRATION_FPS / 60.0

    slow = _scaled_features(0, at_calib, ctx, CALIBRATION_FPS)
    fast = _scaled_features(0, at_60, ctx, 60.0)
    for k in ("wrist_speed_bw", "speed_pre_bw", "speed_post_bw", "speed_std_bw"):
        assert slow[k] == pytest.approx(fast[k], rel=1e-6), k
    # ระยะทางไม่ขึ้นกับ fps — ห้ามถูกสเกล
    for k in ("ball_dist_bw", "racket_dist_bw"):
        assert slow[k] == pytest.approx(fast[k], rel=1e-6), k

    # ที่ fps คาลิเบรตต้องเท่ากับไม่ส่ง fps เลย (พฤติกรรมเดิมไม่เปลี่ยน)
    assert _scaled_features(0, at_calib, ctx) == _scaled_features(
        0, at_calib, ctx, CALIBRATION_FPS)


def test_session_metadata_matches_client_spec():
    """011-MT / 021-M ตาม cv_schema_table_loeuf — 4 จุดที่เคยไม่ตรง spec:

    MT0-3 pose_model_version  ขาดทั้ง field
    MT4   fps                 ส่ง float ดิบ 29.9735 (spec value-example = 60)
    MT5   camera_angle        ส่ง behind_baseline ซึ่งเป็น enum ของ M3 คนละ block
                              (MT5 = single_view / multi_view)
    MT9   usable_strokes      ส่ง len(hit_events) ซึ่งเป็นนิยามของ MT8
    M4/M5 velocity_normalized กลับตรรกะ: fps>=60 -> "resampled_to_60fps"
                              (spec: true = ต้นทาง <60fps แล้ว resample ขึ้น 60)
    """
    from loeuf_cv.config import PipelineConfig
    from loeuf_cv.schema_builder.builder import build_loeuf_schema

    n = 120
    track = _fake_player_track(n_frames=n)
    hit_events = [{"frame": 40, "player_id": 1, "confidence": 0.9},
                  {"frame": 90, "player_id": 1, "confidence": 0.9}]
    video_meta = {"width": 1920, "height": 1080, "total_frames": n}
    out = build_loeuf_schema([track], hit_events, 29.973518362654733,
                             video_meta, PipelineConfig())
    mt = out["session_metadata"]

    assert "pose_model_version" in mt                      # MT0-3
    assert isinstance(mt["fps"], int) and mt["fps"] == 30  # MT4
    assert mt["camera_angle"] in ("single_view", "multi_view")   # MT5
    assert mt["total_strokes_detected"] == len(hit_events)       # MT8

    # MT9 — นับเฉพาะ valid + clean ไม่ใช่จำนวน hit ทั้งหมด
    expected = sum(1 for s in out["strokes"]
                   if s["stroke_root"]["detection_status"] == "valid"
                   and s["stroke_root"]["is_clean_stroke"])
    assert mt["usable_strokes"] == expected
    assert mt["usable_strokes"] <= mt["total_strokes_detected"]

    for s in out["strokes"]:
        m = s["stroke_metadata"]
        # M3 เป็น enum คนละชุดกับ MT5 — ห้ามสลับกัน
        assert m["camera_angle"] in ("behind_baseline", "side_view")
        # path นี้ไม่ได้ resample -> ต้องรายงานตามจริง ห้ามอ้าง resampled_to_60fps
        assert m["velocity_normalized"] is False
        assert m["normalization_method"] == "none"
        # ...และ KN ที่ต้องการ fps>=60 ต้องไม่ถูกตีเป็น visible
        assert s["visibility_flag"]["hip_rotation_velocity_deg_s"] != "visible"


# ===========================================================================
# Keyframe benchmark (loeuf_cv/benchmark_keyframes.py) — วัดเกณฑ์ TOR >= 0.80
# ทั้งหมด pure ไม่ต้องใช้วิดีโอ/YOLO
# ===========================================================================


def _fake_stroke_label(stroke_no=1, stroke_type="FH", fps=29.97, **keyframes):
    """StrokeLabel ปลอม — เลียนแบบสิ่งที่ label_ingest.load_session_label() คืน"""
    from train_model.label_ingest import KEYFRAME_FIELD_MAP, StrokeLabel

    kf = {name: None for name in KEYFRAME_FIELD_MAP}
    kf.update(keyframes)
    return StrokeLabel(clip_id="IMG_TEST", stroke_no=stroke_no,
                       player_id="pl001", stroke_type=stroke_type,
                       fps=fps, usable=True, keyframes=kf)


class _FakeSession:
    def __init__(self, strokes, stem="IMG_TEST"):
        self.strokes = strokes
        self.video_path = Path(f"{stem}.MOV")


def test_label_ingest_extracts_trophy_position():
    """B6 trophy_position ต้องถูกดึงมาด้วย — SV 64 stroke มี GT ตัวนี้
    ถ้าไม่มีใน map จะถูกทิ้งเงียบ ๆ ทั้งหมด"""
    from train_model.label_ingest import KEYFRAME_FIELD_MAP, _extract_keyframes

    assert "trophy_position" in KEYFRAME_FIELD_MAP
    serve = {"backswing_peak_frame": 83, "trophy_position_frame": 100,
             "impact_frame": 115, "follow_through_peak_frame": 120}
    kf = _extract_keyframes(serve)
    assert kf["trophy_position"] == 100
    assert kf["impact"] == 115
    assert kf["unit_turn"] is None          # SV ไม่มี unit_turn ตาม spec
    assert kf["recovery_position"] is None
    # groundstroke ที่ไม่มี trophy → None ไม่ใช่ KeyError
    assert _extract_keyframes({"impact_frame": 50})["trophy_position"] is None


def test_stroke_key_roundtrips_through_match_pred():
    """_match_pred ทำ str(Path(k).with_suffix("")) แล้วเทียบตรง — คีย์ต้องรอด
    ทั้งจุด (จะถูกตัดเป็น suffix) และ separator (Windows normalize)"""
    from loeuf_cv.benchmark_keyframes import stroke_key

    for set_name, stem in [("set2", "IMG_0284"), ("set3", "IMG_0281(BH)"),
                           ("sessions_deferred/set4", "IMG_0294(SV1)")]:
        k = stroke_key(set_name, stem, 3)
        assert str(Path(k).with_suffix("")) == k, f"คีย์ไม่ roundtrip: {k}"
        assert "." not in k and "/" not in k

    # stroke ต่างกันในคลิปเดียวกันต้องไม่ชนกัน
    assert len({stroke_key("set2", "IMG_0284", i) for i in range(1, 24)}) == 23


def test_label_rows_one_per_stroke():
    from loeuf_cv.benchmark_keyframes import (
        BENCH_KEYFRAME_COLS, label_rows_from_session,
    )

    session = _FakeSession([
        _fake_stroke_label(1, "FH", impact=100, backswing_peak=80, unit_turn=60),
        _fake_stroke_label(2, "SV", impact=300, backswing_peak=270,
                           trophy_position=285),
    ])
    rows = label_rows_from_session(session, "set2")

    assert len(rows) == 2                       # ต่อ stroke ไม่ใช่ต่อคลิป
    assert len({r["clip_path"] for r in rows}) == 2
    for r in rows:
        for name in BENCH_KEYFRAME_COLS:
            assert name in r, f"ขาด column {name}"
    fh, sv = rows
    assert fh["unit_turn"] == 60 and fh["trophy_position"] is None
    assert sv["trophy_position"] == 285 and sv["unit_turn"] is None


def test_match_strokes_one_to_one():
    from loeuf_cv.benchmark_keyframes import match_strokes

    r = match_strokes([100, 200, 300], [98, 205, 500], tolerance=10)
    assert [(g, p) for g, p, _ in r["matches"]] == [(0, 0), (1, 1)]
    assert r["fn"] == [2] and r["fp"] == [2]

    # เลือกใกล้สุด ไม่ใช่ซ้ายสุด: 158 ควรจับกับ 160 (Δ2) ไม่ใช่ 150 (Δ8)
    r2 = match_strokes([150, 160], [158], tolerance=10)
    assert [(g, p) for g, p, _ in r2["matches"]] == [(1, 0)]
    assert r2["fn"] == [0]

    # pred 2 ตัวชน GT เดียว → จับได้ตัวเดียว อีกตัวเป็น FP
    r3 = match_strokes([100], [100, 108], tolerance=10)
    assert len(r3["matches"]) == 1 and r3["fp"] == [1]

    # delta มีเครื่องหมาย (pred - gt) เพื่อดู systematic lag
    assert match_strokes([100], [105], tolerance=10)["matches"][0][2] == 5


def test_match_tolerance_within_gap_bounds():
    """guard: ข้อพิสูจน์ว่า greedy = optimal พังทันทีถ้า tolerance ใหญ่เกิน
    ครึ่งหนึ่งของระยะห่างที่แคบที่สุด"""
    from loeuf_cv.benchmark_keyframes import (
        HIT_MIN_GAP_FRAMES_AT_30, MATCH_TOLERANCE_FRAMES, MIN_GT_IMPACT_GAP,
    )

    assert 2 * MATCH_TOLERANCE_FRAMES < MIN_GT_IMPACT_GAP
    assert 2 * MATCH_TOLERANCE_FRAMES < HIT_MIN_GAP_FRAMES_AT_30


def test_detection_metrics_handles_empty():
    from loeuf_cv.benchmark_keyframes import detection_metrics

    m = detection_metrics([
        {"clip": "a", "n_gt": 2, "n_pred": 2, "tp": 2, "fn": 0, "fp": 0,
         "duration_sec": 60.0, "deltas": [1, -2]},
        {"clip": "b", "n_gt": 2, "n_pred": 1, "tp": 1, "fn": 1, "fp": 0,
         "duration_sec": 60.0, "deltas": [3]},
    ])
    assert m["tp"] == 3 and m["fn"] == 1 and m["fp"] == 0
    assert m["recall"] == 0.75 and m["precision"] == 1.0
    assert m["fp_per_minute"] == 0.0

    # ไม่มี prediction เลย → precision None ไม่ใช่ ZeroDivisionError
    empty = detection_metrics([{"clip": "c", "n_gt": 3, "n_pred": 0, "tp": 0,
                                "fn": 3, "fp": 0, "duration_sec": 0.0,
                                "deltas": []}])
    assert empty["precision"] is None and empty["recall"] == 0.0


def test_pred_strokes_from_schema_renames_block():
    """production ใช้ key `keyframe` (เอกพจน์) แต่ keyframe_accuracy คาด
    `keyframes` (พหูพจน์)"""
    from loeuf_cv.benchmark_keyframes import (
        BENCH_KEYFRAME_COLS, pred_strokes_from_schema,
    )

    def _kf(idx):
        return {"frame_index": idx, "timestamp_ms": idx / 29.97 * 1000,
                "detected": True, "joint": None}

    schema = {"strokes": [{
        "stroke_root": {"stroke_index": 1, "stroke_type": "SV",
                        "detection_status": "valid"},
        "keyframe": {name: _kf(100 + i)
                     for i, name in enumerate(BENCH_KEYFRAME_COLS.values())},
    }]}
    preds = pred_strokes_from_schema(schema)
    assert len(preds) == 1
    p = preds[0]
    assert set(p["keyframes"]) == set(BENCH_KEYFRAME_COLS.values())
    assert p["stroke_type"] == "SV"
    assert p["impact_frame"] == p["keyframes"]["impact"]["frame_index"]


def test_keyframe_accuracy_with_bench_cols_scores_trophy():
    """keyframe_cols ใหม่ต้องนับ trophy_position ได้ และ keyframe ที่ไม่มี GT
    ต้องไม่ถูกนับเป็นผิด (ไม่มี GT = ไม่ได้วัด)"""
    from loeuf_cv.benchmark import keyframe_accuracy
    from loeuf_cv.benchmark_keyframes import BENCH_KEYFRAME_COLS

    fps = 29.97

    def _kf(frame):
        return {"frame_index": frame, "timestamp_ms": frame / fps * 1000.0,
                "detected": True}

    preds = {"k1": {"keyframes": {"impact": _kf(115), "backswing_peak": _kf(83),
                                  "trophy_position": _kf(100),
                                  "unit_turn": _kf(50),
                                  "follow_through_peak": _kf(120),
                                  "recovery_position": _kf(150)}}}
    row = {"clip_path": "k1", "stroke_type": "SV", "fps": fps, "usable": True,
           "impact": 115, "backswing_peak": 83, "trophy_position": 100,
           "unit_turn": None, "follow_through_peak": 120,
           "recovery_position": None}

    r = keyframe_accuracy([row], preds, tolerance_frames=1,
                          keyframe_cols=BENCH_KEYFRAME_COLS)
    assert r["per_stroke"]["SV"]["trophy_position"]["n"] == 1
    assert r["per_stroke"]["SV"]["trophy_position"]["accuracy"] == 1.0
    # unit_turn / recovery ไม่มี GT → ต้องไม่โผล่มาเป็น 0.0
    assert "unit_turn" not in r["per_stroke"]["SV"]
    assert r["acceptance_accuracy"] == 1.0 and r["acceptance_n"] == 2

    # row ที่ขาด column ไปเลย ต้องไม่ KeyError (พิสูจน์ .get())
    sparse = {"clip_path": "k1", "stroke_type": "SV", "fps": fps,
              "usable": True, "impact": 115}
    assert keyframe_accuracy([sparse], preds, 1,
                             BENCH_KEYFRAME_COLS)["acceptance_n"] == 1


def test_unmatched_gt_counts_as_failure_not_dropped():
    """GT ที่ hit detection หาไม่เจอ ต้องนับเป็นผิด ไม่ใช่หายจากตัวหาร
    (ไม่งั้น accuracy จะสูงหลอก)"""
    from loeuf_cv.benchmark import keyframe_accuracy
    from loeuf_cv.benchmark_keyframes import (
        BENCH_KEYFRAME_COLS, NOT_DETECTED_PRED,
    )

    row = {"clip_path": "missed", "stroke_type": "FH", "fps": 29.97,
           "usable": True, "impact": 115, "backswing_peak": 83}
    r = keyframe_accuracy([row], {"missed": NOT_DETECTED_PRED}, 1,
                          BENCH_KEYFRAME_COLS)
    assert r["acceptance_accuracy"] == 0.0
    assert r["acceptance_n"] == 2          # ยังอยู่ในตัวหาร
    assert all(f["reason"] == "not_detected" for f in r["failures"])


def test_anomaly_flags_detect_bad_labels():
    from loeuf_cv.benchmark_keyframes import anomaly_flags

    assert anomaly_flags({"impact": 100, "backswing_peak": 80,
                          "follow_through_peak": 120}) == []
    assert "backswing_not_before_impact" in anomaly_flags(
        {"impact": 1371, "backswing_peak": 1371, "follow_through_peak": 1379})
    assert "backswing_not_before_impact" in anomaly_flags(
        {"impact": 330, "backswing_peak": 361, "follow_through_peak": 344})
    assert "follow_through_before_impact" in anomaly_flags(
        {"impact": 2150, "backswing_peak": 2144, "follow_through_peak": 1569})
    assert "missing_follow_through" in anomaly_flags(
        {"impact": 100, "backswing_peak": 80})


def test_render_marks_missing_gt_not_as_zero():
    """ช่องที่ไม่มี GT ต้องขึ้น "no GT" ห้ามขึ้น 0.000 (จะอ่านเป็นสอบตก)"""
    from loeuf_cv.benchmark_keyframes import render_keyframe_markdown

    md = render_keyframe_markdown({
        "mode_a": {"acceptance_accuracy": 0.85, "acceptance_n": 128,
                   "per_stroke": {"SV": {"impact": {"n": 64, "accuracy": 0.9,
                                                    "mae_ms": 12.0}}},
                   "coverage": {"SV": {"impact": 64, "trophy_position": 64}},
                   "failures": [], "missing_predictions": 0},
        "detection": {"tp": 120, "fn": 22, "fp": 5, "recall": 0.845,
                      "precision": 0.96, "fp_per_minute": 0.17,
                      "delta_mean": 1.2, "delta_median": 1.0, "delta_p90": 4.0,
                      "per_clip": [{"clip": "IMG_0284", "n_gt": 2, "n_pred": 2,
                                    "tp": 2, "fn": 0, "fp": 0}]},
        "meta": {"n_gt_strokes": 142},
    })
    assert "no GT" in md                   # SV x unit_turn ไม่มี GT
    assert "TOR" in md and "0.8" in md
    assert "PASS" in md
    assert "Mode A" in md


def test_real_label_set_regression():
    """ล็อกข้อเท็จจริงของ label set ที่ตัวเลขในรายงานอ้างอิง — ถ้า label
    เปลี่ยน ตัวเลขที่ประกาศไปแล้วเป็นโมฆะ ต้องรู้ตัว"""
    from collections import Counter

    import pytest

    from train_model.label_ingest import find_session_labels, load_session_label

    dataset = Path(__file__).resolve().parent.parent / "dataset"
    if not dataset.exists():
        pytest.skip("ไม่มี dataset/ บนเครื่องนี้")

    # ต้อง glob จาก dataset/ ไม่ใช่ dataset/sessions/ ไม่งั้นตกคลิปใน
    # sessions_deferred/ ไป
    #
    # ⚠️ 2026-07-30: ชุด label ขยายจาก 13 ไฟล์ / 142 stroke เป็น 16 / 172 หลัง
    # ลูกค้าส่งคลิปเพิ่ม (IMG_0266-SL, IMG_0283A-VLfh, IMG_0283B-VLbh)
    # ที่สำคัญคือ SL เพิ่มจาก 1 คน (earth 11) เป็น 2 คน (+prem 10) ทำให้
    # Leave-One-Person-Out วัด SL ได้เป็นครั้งแรก — ก่อนหน้านี้ SL ให้ 0.00
    # ทุกโมเดลโดยโครงสร้าง ไม่ใช่เพราะโมเดลแย่
    label_paths = find_session_labels(dataset)
    assert len(label_paths) == 16

    strokes, impacts_by_clip = [], {}
    for p in label_paths:
        session = load_session_label(p)
        strokes.extend(session.strokes)
        impacts_by_clip[p.stem] = sorted(
            s.keyframes["impact"] for s in session.strokes)

    assert len(strokes) == 172
    assert dict(Counter(s.stroke_type for s in strokes)) == {
        "SV": 64, "VL": 40, "BH": 26, "SL": 21, "FH": 21}

    # เกณฑ์รับงานผูกกับ 2 ตัวนี้ — ต้องมี GT ครบทุก stroke
    assert sum(s.keyframes["impact"] is not None for s in strokes) == 172
    assert sum(s.keyframes["backswing_peak"] is not None for s in strokes) == 172
    assert sum(s.keyframes["trophy_position"] is not None for s in strokes) == 64

    # SL ต้องมาจากอย่างน้อย 2 คน ไม่งั้นตัวเลข SL พิสูจน์ข้ามคนไม่ได้อีก
    sl_players = {s.player_id for s in strokes if s.stroke_type == "SL"}
    assert len(sl_players) >= 2, f"SL มาจาก {len(sl_players)} คน — LOPO วัดไม่ได้"

    # ระยะห่างที่แคบสุดต้องยังรองรับ match tolerance ที่ใช้อยู่
    from loeuf_cv.benchmark_keyframes import MATCH_TOLERANCE_FRAMES
    gaps = [b - a for imp in impacts_by_clip.values()
            for a, b in zip(imp, imp[1:])]
    assert min(gaps) >= 2 * MATCH_TOLERANCE_FRAMES


# ---------------------------------------------------------------------------
# backswing_peak (B2) — offset ที่คาลิเบรตกับ GT ลูกค้า
# ดู loeuf_cv/schema_builder/keyframes.py สำหรับที่มาของค่าคงที่
# ---------------------------------------------------------------------------

def _straight_wrist_path(n=200, y=0.5):
    """ข้อมือนิ่ง — ไม่มี motion cue ให้กิ่ง slice จุดชนวน"""
    p = np.zeros((n, 2), dtype=float)
    p[:, 0] = 0.5
    p[:, 1] = y
    return p


def test_backswing_ground_uses_calibrated_offset():
    """ไม่ใช่เสิร์ฟ ไม่ใช่ slice → impact - GROUND_BACKSWING_OFFSET"""
    from loeuf_cv.schema_builder.keyframes import (GROUND_BACKSWING_OFFSET,
                                                   extract_keyframes)
    wrist = _straight_wrist_path()
    head = _straight_wrist_path(y=0.2)      # หัวอยู่สูงกว่าข้อมือ = ไม่ใช่เสิร์ฟ
    kf = extract_keyframes(100, wrist, 30.0, 200, head_path=head)
    assert kf["backswing_peak"] == 100 - GROUND_BACKSWING_OFFSET


def test_backswing_detects_serve_from_wrist_above_head():
    """ข้อมือเหนือหัวตอน impact → ใช้กิ่งเสิร์ฟ (หาเฟรมข้อมือสูงสุด)

    ไม่พึ่ง stroke classifier เพราะ extract_keyframes ถูกเรียกก่อน classify_stroke
    """
    from loeuf_cv.schema_builder.keyframes import (SERVE_SEARCH_MAX,
                                                   SERVE_SEARCH_MIN,
                                                   extract_keyframes)
    wrist = _straight_wrist_path(y=0.5)
    head = _straight_wrist_path(y=0.6)      # หัวต่ำกว่าข้อมือ (y มาก) = เสิร์ฟ
    imp = 100
    trophy = imp - 20                        # อยู่ในช่วงค้นหา 9..35
    wrist[trophy, 1] = 0.1                   # ข้อมือสูงสุดที่เฟรมนี้
    kf = extract_keyframes(imp, wrist, 30.0, 200, head_path=head)
    assert kf["backswing_peak"] == trophy
    assert imp - SERVE_SEARCH_MAX <= kf["backswing_peak"] <= imp - SERVE_SEARCH_MIN


def test_backswing_detects_slice_from_downward_wrist_velocity():
    """ข้อมือสับลง (vy > threshold) → ใช้ offset ของ slice ซึ่งยาวกว่า groundstroke"""
    from loeuf_cv.schema_builder.keyframes import (GROUND_BACKSWING_OFFSET,
                                                   SLICE_BACKSWING_OFFSET,
                                                   SLICE_VY_THRESHOLD,
                                                   extract_keyframes)
    wrist = _straight_wrist_path(y=0.5)
    head = _straight_wrist_path(y=0.2)
    imp = 100
    # y เพิ่ม = เคลื่อนลงในพิกัดภาพ ให้เกิน threshold ชัด ๆ
    wrist[imp, 1] = 0.5 + SLICE_VY_THRESHOLD * 3
    kf = extract_keyframes(imp, wrist, 30.0, 200, head_path=head)
    assert kf["backswing_peak"] == imp - SLICE_BACKSWING_OFFSET
    assert SLICE_BACKSWING_OFFSET != GROUND_BACKSWING_OFFSET


def test_backswing_without_head_path_falls_back_to_ground():
    """head_path=None → แยกเสิร์ฟไม่ได้ ต้องไม่ crash และใช้ค่า groundstroke"""
    from loeuf_cv.schema_builder.keyframes import (GROUND_BACKSWING_OFFSET,
                                                   extract_keyframes)
    kf = extract_keyframes(100, _straight_wrist_path(), 30.0, 200)
    assert kf["backswing_peak"] == 100 - GROUND_BACKSWING_OFFSET


def test_backswing_offsets_match_client_gt_convention():
    """guard: GT ลูกค้าวาง B2 ใกล้ impact มาก (groundstroke median 3 เฟรม)

    ถ้าใครเปลี่ยนค่าคงที่ให้ห่างขึ้นเยอะ = กลับไปเป็นบั๊กเดิมที่ prediction
    ไปกองที่ขอบหน้าต่างค้นหา (accuracy 0.000) — ต้อง fail ทันที
    """
    from loeuf_cv.schema_builder.keyframes import (GROUND_BACKSWING_OFFSET,
                                                   SLICE_BACKSWING_OFFSET)
    assert 1 <= GROUND_BACKSWING_OFFSET <= 8
    assert 1 <= SLICE_BACKSWING_OFFSET <= 12


def test_unit_turn_offset_matches_gt_median():
    """B1 = B2 - 0.5s ; 15 เฟรมที่ 30fps ตรงกับ GT median ของ (B2 - B1) พอดี"""
    from loeuf_cv.schema_builder.keyframes import extract_keyframes
    wrist = _straight_wrist_path()
    head = _straight_wrist_path(y=0.2)
    kf = extract_keyframes(100, wrist, 30.0, 200, head_path=head)
    assert kf["backswing_peak"] - kf["unit_turn"] == 15


# ─────────────────────────────────────────────────────────
# filter_static_ball_bboxes — คัดลูกที่นอนนิ่งบนพื้น/ในรถเข็นออก
# วัดจริงแล้ว 87.7% ของ detection ในคลิปมุมกว้างเป็นลูกนิ่ง
# ดู docs/BALL_DETECTION_ISSUE.md
# ─────────────────────────────────────────────────────────

def _bb(x, y, s=12):
    """bbox (x, y, w, h) ที่มีจุดกึ่งกลางอยู่ที่ (x, y)"""
    return (int(x - s / 2), int(y - s / 2), s, s)


def test_static_ball_is_filtered_out():
    """ลูกที่นอนอยู่ตำแหน่งเดิมตลอด ต้องถูกตัดออกทุกเฟรม"""
    from loeuf_cv.hit_detection import filter_static_ball_bboxes
    n = 120
    bb = {i: [_bb(500, 800)] for i in range(n)}
    assert filter_static_ball_bboxes(bb, n) == {}


def test_moving_ball_is_kept():
    """ลูกที่ลอยเป็นเส้นตรงเร็ว ต้องไม่ถูกตัดเลย"""
    from loeuf_cv.hit_detection import filter_static_ball_bboxes
    n = 120
    bb = {i: [_bb(100 + 25 * i, 400)] for i in range(n)}
    out = filter_static_ball_bboxes(bb, n)
    assert len(out) == n


def test_toss_apex_ball_is_kept():
    """จุดสูงสุดของการโยนเสิร์ฟ ความเร็วแนวดิ่ง = 0 ชั่วขณะ แต่แรงโน้มถ่วง
    ยังพาลูกออกจากรัศมี 8px ภายใน ~2 เฟรม -> ต้องไม่โดนตัด
    (นี่คือเคสที่อันตรายที่สุดของตัวกรองนี้)"""
    from loeuf_cv.hit_detection import filter_static_ball_bboxes
    n = 61
    g = 4.4  # px/frame^2 ที่ 30fps
    apex = 30
    bb = {i: [_bb(600, 300 + 0.5 * g * (i - apex) ** 2)] for i in range(n)}
    out = filter_static_ball_bboxes(bb, n)
    assert apex in out, "ลูกตรงจุดสูงสุดของการโยนต้องรอด"
    assert len(out) >= n - 6


def test_static_and_moving_together_keeps_only_moving():
    """เคสจริง: ลูกนิ่งหลายลูกในสนาม + ลูกที่กำลังเล่น 1 ลูก"""
    from loeuf_cv.hit_detection import filter_static_ball_bboxes
    n = 120
    statics = [(300, 900), (450, 950), (700, 880)]
    bb = {}
    for i in range(n):
        bb[i] = [_bb(sx, sy) for sx, sy in statics]
        bb[i].append(_bb(100 + 25 * i, 400))
    out = filter_static_ball_bboxes(bb, n)
    assert len(out) == n
    for i, kept in out.items():
        assert len(kept) == 1, f"เฟรม {i} ควรเหลือลูกเดียว ได้ {kept}"
        cx = kept[0][0] + kept[0][2] / 2
        assert abs(cx - (100 + 25 * i)) < 2


def test_kalman_ignores_static_ball_decoy():
    """end-to-end: Kalman ต้องตามลูกที่เคลื่อนที่ ไม่ใช่ลูกนิ่งที่อยู่ใกล้กว่า

    ก่อนแก้: association เลือก 'bbox ที่ใกล้ตำแหน่งทำนายที่สุด ภายใน 200px'
    โดยไม่สนว่าเคลื่อนที่ไหม -> พอลูกจริงวิ่งผ่านใกล้ลูกนิ่ง มันจะสลับไปเกาะ
    ลูกนิ่งแล้วค้างอยู่ตรงนั้น
    """
    from loeuf_cv.hit_detection import extract_ball_trajectory_kalman
    n = 100
    decoy = (1300, 402)
    bb = {}
    for i in range(n):
        bb[i] = [_bb(*decoy), _bb(100 + 25 * i, 400)]
    traj = extract_ball_trajectory_kalman(bb, n)
    # ปลายคลิปลูกจริงอยู่ไกลจาก decoy มาก — ถ้าไปเกาะ decoy จะค้างที่ x=1300
    assert abs(traj[n - 1, 0] - (100 + 25 * (n - 1))) < 60, \
        f"Kalman ไปเกาะลูกนิ่ง: x={traj[n-1, 0]}"


def test_drop_static_false_preserves_old_behaviour():
    """ต้องปิดกลับไปพฤติกรรมเดิมได้ (สำหรับเทียบผล/debug)"""
    from loeuf_cv.hit_detection import extract_ball_trajectory_kalman
    n = 60
    bb = {i: [_bb(500, 800)] for i in range(n)}
    on = extract_ball_trajectory_kalman(bb, n, drop_static=True)
    off = extract_ball_trajectory_kalman(bb, n, drop_static=False)
    assert np.isnan(on).all(), "เปิดกรอง: ลูกนิ่งล้วน ต้องไม่เหลือ trajectory"
    assert not np.isnan(off[:, 0]).all(), "ปิดกรอง: ต้องได้พฤติกรรมเดิม"


def test_static_filter_no_op_when_no_detections():
    from loeuf_cv.hit_detection import filter_static_ball_bboxes
    assert filter_static_ball_bboxes({}, 50) == {}


def test_intermittent_static_ball_is_filtered():
    """ลูกนิ่งที่ detector จับได้ติด ๆ ดับ ๆ ก็ต้องโดนตัด

    regression: เกณฑ์แรกใช้ 'สัดส่วนเฟรมที่เจอในหน้าต่าง ±20 > 50%' แล้ว
    ปล่อยหลุด เพราะลูกนิ่งใน dataset จริงถูกจับได้แค่ ~40-50% ของเฟรม
    (เจอที่ IMG_0281 เฟรม 937-955 จุด (1247.5, 539.0))
    """
    from loeuf_cv.hit_detection import filter_static_ball_bboxes
    n = 300
    bb = {}
    for i in range(n):
        if i % 5 in (0, 1):        # เจอแค่ 40% ของเฟรม
            bb[i] = [_bb(1247, 539)]
    assert filter_static_ball_bboxes(bb, n) == {}


def test_ball_passing_same_spot_twice_still_kept():
    """ลูกที่บังเอิญผ่านจุดเดิมสองครั้งห่างกันมาก ต้องไม่โดนตัด

    span ยาวก็จริง แต่จำนวนครั้งที่เจอน้อย -> ไม่เข้าเกณฑ์ 'ของนิ่ง'
    (นี่คือเหตุผลที่ต้องมี min_hits ไม่ใช่ดู span อย่างเดียว)
    """
    from loeuf_cv.hit_detection import filter_static_ball_bboxes
    n = 300
    bb = {}
    for i in range(60):                       # ตีที่ 1: วิ่งผ่าน x=100..1290
        bb[i] = [_bb(100 + 20 * i, 400)]
    for i in range(200, 260):                 # ตีที่ 2: วิ่งเส้นทางเดิมย้อนกลับ
        bb[i] = [_bb(1280 - 20 * (i - 200), 400)]
    out = filter_static_ball_bboxes(bb, n)
    assert len(out) == 120, \
        f"ลูกที่เคลื่อนที่ตลอดต้องรอดแม้ผ่านจุดเดิมซ้ำ (เหลือ {len(out)})"


def test_static_filter_thresholds_stay_physical():
    """guard: เกณฑ์ต้องห่างจาก 'ลูกที่จุดสูงสุดของการโยน' พอสมควร

    ลูกตรง apex อยู่ในวง 8px ได้ ~4 เฟรม (แรงโน้มถ่วง 4.4 px/frame²)
    ถ้าใครลด span ลงมาใกล้ค่านั้น = เริ่มตัดลูกจริงทิ้ง
    """
    from loeuf_cv.hit_detection import (STATIC_BALL_RADIUS, STATIC_BALL_SPAN,
                                        STATIC_BALL_MIN_HITS)
    assert STATIC_BALL_SPAN >= 15
    assert STATIC_BALL_MIN_HITS >= 3
    assert 4.0 <= STATIC_BALL_RADIUS <= 20.0


def test_ball_gate_ignores_kalman_coasting():
    """ด่าน 'ลูกอยู่ไกลข้อมือ' ต้องไม่ปัด candidate ทิ้งด้วยตำแหน่งที่ Kalman เดาเอง

    regression: ก่อนแก้ ด่านนี้เชื่อ ball_traj ทุกค่ารวมช่วง coasting ทำให้
    ตัด stroke จริงทิ้ง วัดแล้วกิน recall 17% (ดู scripts/diagnose_hit_recall.py)
    """
    from loeuf_cv.hit_detection import extract_ball_trajectory_kalman
    n = 60
    # ลูกถูกเห็นแค่ 5 เฟรมแรก ที่เหลือ Kalman เดาต่อ
    bb = {i: [_bb(100 + 25 * i, 400)] for i in range(5)}
    traj, measured = extract_ball_trajectory_kalman(bb, n, return_measured=True)
    assert measured[:5].all(), "เฟรมที่มี detection ต้องถูกทำเครื่องหมายว่าวัดจริง"
    assert not measured[5:].any(), "ช่วง Kalman เดาต่อต้องไม่ถูกนับว่าวัดจริง"
    # แต่ trajectory ยังมีค่าอยู่ (coasting) — นี่คือค่าที่ห้ามเอาไปตัดสินใจ
    assert not np.isnan(traj[10, 0])


def test_extract_ball_trajectory_kalman_default_return_unchanged():
    """ค่า default ต้องคืน array เดี่ยวเหมือนเดิม (ไม่ทำ caller เก่าพัง)"""
    from loeuf_cv.hit_detection import extract_ball_trajectory_kalman
    out = extract_ball_trajectory_kalman({0: [_bb(100, 200)]}, 10)
    assert isinstance(out, np.ndarray)
    assert out.shape == (10, 2)


# ─────────────────────────────────────────────────────────
# B4/B5/B6 — offset ต่อ stroke type (ดู scripts/probe_broken_keyframes.py)
# ─────────────────────────────────────────────────────────

def test_follow_through_uses_per_type_offset():
    """B4 ต้องเป็น impact + offset ของท่านั้น ไม่ใช่ 'จุดที่ข้อมือสูงสุด'

    วิธีเดิม (argmin y ใน 1 วิหลัง impact) วัดได้ 0.000-0.200
    offset คงที่ให้ LOPO 26.9% (BH 40% · VL 52%)
    """
    from loeuf_cv.schema_builder.keyframes import (FOLLOW_THROUGH_OFFSET,
                                                   refine_keyframes_for_type)
    for stype, off in FOLLOW_THROUGH_OFFSET.items():
        kf = {"impact": 100, "follow_through_peak": None,
              "recovery_position": None}
        refine_keyframes_for_type(kf, stype, 100, 500)
        assert kf["follow_through_peak"] == 100 + off, stype


def test_recovery_no_longer_collapses_onto_follow_through():
    """regression: เกณฑ์เดิม vel_mag < 0.5 บนพิกัด normalized 0-1
    = ครึ่งความกว้างภาพต่อเฟรม -> เข้าเงื่อนไขทันทีเฟรมแรกเสมอ
    ทำให้ recovery == follow_through ทุกครั้ง และได้ accuracy 0.000"""
    from loeuf_cv.schema_builder.keyframes import refine_keyframes_for_type
    kf = {"impact": 100, "follow_through_peak": None, "recovery_position": None}
    refine_keyframes_for_type(kf, "FH", 100, 500)
    assert kf["recovery_position"] > kf["follow_through_peak"]
    assert kf["recovery_position"] - 100 >= 20


def test_keyframe_offsets_clamped_to_clip_length():
    from loeuf_cv.schema_builder.keyframes import refine_keyframes_for_type
    kf = {"impact": 98, "follow_through_peak": None, "recovery_position": None}
    refine_keyframes_for_type(kf, "BH", 98, 100)
    assert kf["follow_through_peak"] <= 99
    assert kf["recovery_position"] <= 99


def test_trophy_is_serve_only_and_anchored_to_impact():
    """B6 เดิมใช้ค่าเดียวกับ backswing_peak ของ SV ซึ่งพาความผิดพลาดของ
    backswing มาด้วย (0.219 -> trophy 0.094) — ผูกกับ impact ตรง ๆ ได้ 0.344"""
    from loeuf_cv.schema_builder.keyframes import (TROPHY_OFFSET,
                                                   trophy_position_frame)
    assert trophy_position_frame("SV", 100, 500) == 100 + TROPHY_OFFSET
    for other in ("FH", "BH", "VL", "SL"):
        assert trophy_position_frame(other, 100, 500) is None
    assert trophy_position_frame("SV", 5, 500) == 0        # clamp ล่าง
    assert trophy_position_frame("SV", None, 500) is None


def test_unit_turn_offset_left_alone():
    """guard: เคยลองเปลี่ยน B1 ไปใช้ offset ต่อท่าแล้ว **แย่ลง**
    (LOPO 19.1% เทียบของเดิมที่ให้ BH 0.308 / SL 0.286) — ห้ามเปลี่ยนตาม
    B4/B5 โดยไม่วัดใหม่"""
    from loeuf_cv.schema_builder.keyframes import extract_keyframes
    wrist = _straight_wrist_path()
    head = _straight_wrist_path(y=0.2)
    kf = extract_keyframes(100, wrist, 30.0, 200, head_path=head)
    assert kf["backswing_peak"] - kf["unit_turn"] == 15


# ─────────────────────────────────────────────────────────
# stroke classifier — ด่าน visibility เคยปัดทุกอย่างเป็น FH
# ─────────────────────────────────────────────────────────

def _pose_with_visibility(vis_at_impact, n=60, impact=30):
    """สร้าง PoseTimeseries ปลอมที่ข้อมือขวาอยู่ซ้ายของลำตัว (= แบ็คแฮนด์)"""
    from loeuf_cv.config import (L_SHOULDER, L_WRIST, NOSE, R_SHOULDER,
                                 R_WRIST)
    from loeuf_cv.pose_extractor import PoseTimeseries, VideoMeta
    lm = np.zeros((n, 33, 3), dtype=float)
    lm[:, NOSE, :2] = (0.50, 0.30)
    lm[:, R_SHOULDER, :2] = (0.55, 0.40)
    lm[:, L_SHOULDER, :2] = (0.45, 0.40)
    lm[:, R_WRIST, :2] = (0.35, 0.50)   # ซ้ายของ spine(0.50) -> BH
    lm[:, L_WRIST, :2] = (0.40, 0.50)
    vis = np.full((n, 33), 0.9)
    vis[impact, :] = vis_at_impact
    return PoseTimeseries(
        landmarks=lm, world_landmarks=np.zeros((n, 33, 3)), visibility=vis,
        timestamps_ms=np.arange(n) * (1000 / 30.0),
        meta=VideoMeta(path="fake.mp4", fps=30.0, frame_count=n,
                       width=1920, height=1080))


def test_low_visibility_at_impact_no_longer_forces_forehand():
    """regression: เดิม visibility < 0.3 ที่เฟรม impact -> return "FH" ทันที
    โดยไม่ดูอะไรเลย

    ท่าแบ็คแฮนด์หมุนตัวจนข้อมือข้างถนัดถูกบัง MediaPipe จึงให้ visibility ต่ำ
    เป็นปกติ -> วัดจริงแล้วแบ็คแฮนด์ 23/26 (88%) ถูกปัดเป็นโฟร์แฮนด์
    (VL 22% · SV 16% ก็โดน รวม 24.4% ของทั้งชุด)
    """
    from loeuf_cv.schema_builder.classifier import classify_stroke
    ts = _pose_with_visibility(0.05)      # เฟรม impact มองไม่เห็น
    assert classify_stroke(ts, 30, "right") == "BH"


def test_nearest_visible_frame_falls_back_within_window():
    from loeuf_cv.config import R_WRIST
    from loeuf_cv.schema_builder.classifier import (
        CLASSIFY_VIS_SEARCH_FRAMES, _nearest_visible_frame)
    ts = _pose_with_visibility(0.05)
    f = _nearest_visible_frame(ts, 30, R_WRIST)
    assert f is not None and f != 30
    assert abs(f - 30) <= CLASSIFY_VIS_SEARCH_FRAMES


def test_nearest_visible_frame_returns_none_when_all_hidden():
    from loeuf_cv.config import R_WRIST
    from loeuf_cv.schema_builder.classifier import _nearest_visible_frame
    ts = _pose_with_visibility(0.9)
    ts.visibility[:, :] = 0.05
    assert _nearest_visible_frame(ts, 30, R_WRIST) is None


def test_stroke_features_single_source_of_truth():
    """guard: ตอนเทรนกับตอนใช้งานต้องเรียก build_stroke_features ตัวเดียวกัน

    เคยแยกกันแล้วเกิด train/serve skew (get_body_metrics vs MetricsEngine)
    ค่าเฉลี่ย arm_backswing_depth_deg 125.3 ตอนเทรน แต่ -12.8 ตอนใช้งาน
    """
    from loeuf_cv.schema_builder.classifier import build_stroke_features
    ts = _pose_with_visibility(0.9)
    f = build_stroke_features(ts, 30, "right")
    assert "wrist_minus_head_y" in f
    assert "wrist_minus_spine_x_dominant_relative" in f
    # ข้อมือขวาอยู่ซ้ายของ spine -> ค่าติดลบสำหรับคนถนัดขวา
    assert f["wrist_minus_spine_x_dominant_relative"] < 0


def test_low_stroke_type_confidence_routes_to_coach_review():
    """ไม่มั่นใจว่าเป็นท่าอะไร -> ต้องส่งให้โค้ชตรวจ ไม่ปล่อยผ่านพร้อมชื่อท่าที่อาจผิด

    เหตุผล: ชุดข้อมูลถ่ายแบบ 'หนึ่งคลิปหนึ่งท่า' ทุกคลิป และ SL/VL ไม่เคยอยู่
    คลิปเดียวกัน -> ยืนยันไม่ได้ว่าโมเดลแยกสองท่านี้ได้จริง (LOPO SL = 0.000)
    ดู docs/STROKE_CLASSIFICATION.md
    """
    from loeuf_cv.config import PipelineConfig
    from loeuf_cv.schema_builder.builder import _recommended_action
    cfg = PipelineConfig()
    ok = dict(detection_status="valid", cf1=0.99, is_clean=True, config=cfg)
    assert _recommended_action(**ok, stroke_type_confidence=0.95) == "auto_accept"
    assert _recommended_action(**ok, stroke_type_confidence=0.30) == "coach_review"


def test_recommended_action_defaults_stay_backward_compatible():
    """ไม่ส่ง stroke_type_confidence มา = พฤติกรรมเดิมทุกประการ"""
    from loeuf_cv.config import PipelineConfig
    from loeuf_cv.schema_builder.builder import _recommended_action
    cfg = PipelineConfig()
    assert _recommended_action("valid", 0.99, True, cfg) == "auto_accept"
    assert _recommended_action("failed", 0.99, True, cfg) == "discard"
    assert _recommended_action("partial", 0.99, True, cfg) == "coach_review"


def test_classify_stroke_with_confidence_returns_pair():
    from loeuf_cv.schema_builder.classifier import (
        classify_stroke, classify_stroke_with_confidence)
    ts = _pose_with_visibility(0.9)
    t, c = classify_stroke_with_confidence(ts, 30, "right")
    assert isinstance(t, str) and 0.0 <= c <= 1.0
    assert classify_stroke(ts, 30, "right") == t
