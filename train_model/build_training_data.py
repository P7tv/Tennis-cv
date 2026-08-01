"""แปลง client-labeled session clips (dataset/sessions/**/*_stroke_labels_*.json
+ วิดีโอคู่กัน) เป็น training CSV สำหรับ 2 โมเดล — reuse feature extraction
ที่มีอยู่แล้วทั้งหมด ไม่เขียนสูตรใหม่:
  - hit classifier   : loeuf_cv.hit_detection.detect_hit_events() (candidate +
                        features) auto-label is_hit จาก impact_frame ที่คน label ไว้
  - stroke classifier: loeuf_cv.schema_builder.metrics.get_body_metrics() ที่
                        keyframe จริงจาก label (ไม่ใช้ heuristic เดา keyframe)

รัน: python train_model/build_training_data.py
เอาต์พุต: dataset/training/hit_candidates.csv, dataset/training/stroke_features.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2  # noqa: E402

from label_ingest import find_session_labels, load_session_label  # noqa: E402
from loeuf_cv.config import L_SHOULDER, L_WRIST, NOSE, R_SHOULDER, R_WRIST, PipelineConfig  # noqa: E402
from loeuf_cv.hit_detection import detect_hit_events, extract_ball_trajectory_kalman  # noqa: E402
from loeuf_cv.schema_builder.classifier import swing_window_features  # noqa: E402
from loeuf_cv.schema_builder.metrics import get_body_metrics  # noqa: E402
from webui.yolo_track import track_players_with_yolo  # noqa: E402

HIT_CSV_COLUMNS = [
    "is_hit", "frame", "confidence", "wrist_speed", "ball_dist",
    "ball_vel_before", "ball_vel_after", "ball_vel_change", "ball_angle_change",
    "racket_dist",
    # รูปทรงของ wrist speed รอบๆ candidate frame (ดู
    # loeuf_cv/hit_detection.py::_hit_window_features) — แยกจังหวะตีจริง (peak
    # แหลมแล้วชะลอทันที) ออกจากการวิ่ง/แกว่งแขนทั่วไป (speed สูงต่อเนื่อง)
    "speed_pre_mean", "speed_post_mean", "speed_decel_ratio",
    "speed_peak_sharpness", "speed_std_window",
    # ต่อ session/วิดีโอ (ไม่ใช่ต่อ stroke) — ใช้ทำ grouped train/test split กัน
    # data leakage ข้าม candidate ในคลิปเดียวกัน (ดู train_hit_classifier.py)
    "clip_id",
]

STROKE_FEATURE_COLUMNS = [
    "body_shoulder_rotation_at_impact_deg",
    "body_hip_rotation_at_impact_deg",
    "body_shoulder_hip_separation_at_impact_deg",
    "arm_follow_through_angle_deg",
    "arm_backswing_depth_deg",
    "contact_height_cm",
    "contact_distance_from_body_cm",
    "wrist_minus_head_y",
    "wrist_minus_spine_x_dominant_relative",
    # min/max ตลอดช่วงสวิง (backswing_peak..follow_through_peak จาก label จริง)
    # ไม่ใช่แค่เฟรมเดียว ณ impact — ดู loeuf_cv/schema_builder/classifier.py::
    # swing_window_features (ใช้ฟังก์ชันเดียวกันตอน predict จริงด้วย กัน
    # train/inference feature ไม่ตรงกัน)
    "wrist_minus_spine_x_dominant_relative_min",
    "wrist_minus_spine_x_dominant_relative_max",
    "wrist_minus_head_y_min",
    "wrist_minus_head_y_max",
]


def _pick_target_track(tracks):
    """ผู้เล่นหลัก = track ที่มี pose coverage (จำนวนเฟรมที่เห็นตัว) สูงสุด
    — สมมติฐาน: คนป้อนบอล/คนอื่นในเฟรม ปกติถูก track สั้นกว่าผู้เล่นหลัก"""
    def coverage(t):
        vis = t.pose.visibility
        return float((vis.mean(axis=1) > 0).sum())
    return max(tracks, key=coverage)


def _flatten_stroke_features(pose, kf: dict, metrics: dict, dominant_side: str) -> dict:
    row = {
        "body_shoulder_rotation_at_impact_deg": metrics["body"].get("shoulder_rotation_at_impact_deg"),
        "body_hip_rotation_at_impact_deg": metrics["body"].get("hip_rotation_at_impact_deg"),
        "body_shoulder_hip_separation_at_impact_deg": metrics["body"].get("shoulder_hip_separation_at_impact_deg"),
        "arm_follow_through_angle_deg": metrics["arm"].get("follow_through_angle_deg"),
        "arm_backswing_depth_deg": metrics["arm"].get("backswing_depth_deg"),
        "contact_height_cm": metrics["contact"].get("contact_height_cm"),
        "contact_distance_from_body_cm": metrics["contact"].get("contact_distance_from_body_cm"),
        "wrist_minus_head_y": None,
        "wrist_minus_spine_x_dominant_relative": None,
        "wrist_minus_spine_x_dominant_relative_min": None,
        "wrist_minus_spine_x_dominant_relative_max": None,
        "wrist_minus_head_y_min": None,
        "wrist_minus_head_y_max": None,
    }

    impact = kf.get("impact")
    if impact is not None and impact < len(pose.landmarks):
        lm = pose.landmarks[impact]
        wrist_idx = R_WRIST if dominant_side == "right" else L_WRIST
        spine_x = (lm[L_SHOULDER][0] + lm[R_SHOULDER][0]) / 2.0
        raw_dx = lm[wrist_idx][0] - spine_x
        # normalize เครื่องหมายตาม dominant_side เหมือน classify_stroke() —
        # ค่าบวก = ฝั่ง forehand เสมอ ไม่ว่าถนัดซ้ายหรือขวา
        row["wrist_minus_head_y"] = float(lm[wrist_idx][1] - lm[NOSE][1])
        row["wrist_minus_spine_x_dominant_relative"] = float(raw_dx if dominant_side == "right" else -raw_dx)

        # ทั้งช่วงสวิง (backswing_peak..follow_through_peak จาก label จริง ไม่ใช่
        # heuristic) — ฟังก์ชันเดียวกับที่ classify_stroke() ใช้ตอน predict จริง
        # กัน train/inference feature ไม่ตรงกัน (ดู loeuf_cv/schema_builder/classifier.py)
        row.update(swing_window_features(
            pose.landmarks, pose.visibility, wrist_idx, dominant_side, kf, impact))

    return row


def process_session(label_path: Path, args, hit_writer, stroke_writer, stats) -> None:
    session = load_session_label(label_path)
    if not session.strokes:
        print("  ไม่มี usable stroke — ข้าม")
        return
    if len(session.players) > 1:
        print(f"  WARNING: มี {len(session.players)} players ในไฟล์นี้ — ใช้แค่คนแรก (ยังไม่รองรับหลายคน/คลิป)")
    player = next(iter(session.players.values()))

    cap = cv2.VideoCapture(str(session.video_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or session.strokes[0].fps
    n_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    config = PipelineConfig(dominant_side=player.dominant_side, subject_height_cm=player.height)

    print(f"  tracking {session.video_path.name} ({n_video_frames} frames, {width}x{height} @ {fps:.1f}fps)...")
    tracks, ball_bboxes, racket_bboxes, racket_keypoints = track_players_with_yolo(
        str(session.video_path), max_players=args.max_players, config=config,
        model_path=args.ball_model, base_model=args.base_model,
    )
    if not tracks:
        print("  WARNING: ไม่เจอ track เลย — ข้าม")
        return
    track = _pick_target_track(tracks)
    print(f"  เลือก track_id={track.track_id} role={track.role} (จาก {len(tracks)} track)")

    # ─── Hit candidates: auto-label จาก impact_frame ที่คน label ───
    n_pose_frames = len(track.pose.landmarks)
    traj, ball_measured = extract_ball_trajectory_kalman(
        ball_bboxes, n_pose_frames, return_measured=True, fps=fps)
    impact_frames = [
        s.keyframes["impact"] for s in session.strokes if s.keyframes.get("impact") is not None
    ]

    cwd = os.getcwd()
    scratch_dir = tempfile.mkdtemp(prefix="hit_candidates_scratch_")
    try:
        os.chdir(scratch_dir)
        events = detect_hit_events(traj, [track], fps, width, height,
                                   racket_bboxes=racket_bboxes,
                                   ball_measured=ball_measured)
    finally:
        os.chdir(cwd)
        shutil.rmtree(scratch_dir, ignore_errors=True)

    session_clip_id = session.video_path.stem
    for ev in events:
        is_hit = int(any(abs(ev["frame"] - fr) <= args.hit_tolerance_frames for fr in impact_frames))
        feats = ev.get("features", {})
        hit_writer.writerow([
            is_hit, ev["frame"], ev["confidence"],
            feats.get("wrist_speed", 0), feats.get("ball_dist", 9999),
            feats.get("ball_vel_before", 0), feats.get("ball_vel_after", 0),
            feats.get("ball_vel_change", 0), feats.get("ball_angle_change", 1),
            feats.get("racket_dist", 9999),
            feats.get("speed_pre_mean", 0), feats.get("speed_post_mean", 0),
            feats.get("speed_decel_ratio", 1), feats.get("speed_peak_sharpness", 1),
            feats.get("speed_std_window", 0),
            session_clip_id,
        ])
        stats["hit_pos" if is_hit else "hit_neg"] += 1
    print(f"  hit candidates: {len(events)} (ground-truth impacts: {len(impact_frames)})")

    # ─── Stroke features: ใช้ keyframe จริงจาก label (ไม่ใช่ heuristic เดา) ───
    for s in session.strokes:
        metrics = get_body_metrics(
            track.pose, s.keyframes,
            actual_height_cm=player.height or 170.0,
            dominant_side=player.dominant_side,
        )
        row = _flatten_stroke_features(track.pose, s.keyframes, metrics, player.dominant_side)
        row["stroke_type"] = s.stroke_type
        row["clip_id"] = s.clip_id
        row["stroke_no"] = s.stroke_no
        stroke_writer.writerow(row)
        stats["stroke_types"][s.stroke_type] = stats["stroke_types"].get(s.stroke_type, 0) + 1
    print(f"  stroke rows: {len(session.strokes)}")


def main():
    parser = argparse.ArgumentParser(
        description="แปลง dataset/sessions/**/*_stroke_labels_*.json เป็น training CSV "
                     "สำหรับ hit classifier + stroke type classifier")
    parser.add_argument("--sessions-dir", default="dataset/sessions")
    parser.add_argument("--out-dir", default="dataset/training")
    parser.add_argument("--ball-model", default="yolo26s.pt")
    parser.add_argument("--base-model", default="yolo11m.pt")
    parser.add_argument("--max-players", type=int, default=1)
    parser.add_argument("--hit-tolerance-frames", type=int, default=5,
                         help="candidate frame ห่างจาก impact_frame (label) ไม่เกินกี่เฟรม ถือว่า is_hit=1")
    args = parser.parse_args()

    sessions_dir = Path(args.sessions_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    label_paths = find_session_labels(sessions_dir)
    if not label_paths:
        print(f"ไม่เจอ label json ใต้ {sessions_dir} (คาดว่าไฟล์ลงท้าย _stroke_labels_*.json)")
        return

    hit_csv_path = out_dir / "hit_candidates.csv"
    stroke_csv_path = out_dir / "stroke_features.csv"
    stats = {"sessions": 0, "sessions_skipped": 0, "hit_pos": 0, "hit_neg": 0, "stroke_types": {}}

    with open(hit_csv_path, "w", newline="", encoding="utf-8") as hf, \
            open(stroke_csv_path, "w", newline="", encoding="utf-8") as sf:
        hit_writer = csv.writer(hf)
        hit_writer.writerow(HIT_CSV_COLUMNS)
        stroke_writer = csv.DictWriter(
            sf, fieldnames=STROKE_FEATURE_COLUMNS + ["stroke_type", "clip_id", "stroke_no"])
        stroke_writer.writeheader()

        for label_path in label_paths:
            print(f"\n=== {label_path.relative_to(sessions_dir)} ===")
            try:
                process_session(label_path, args, hit_writer, stroke_writer, stats)
                stats["sessions"] += 1
            except Exception as e:
                print(f"  ERROR: {e} — ข้ามไฟล์นี้")
                stats["sessions_skipped"] += 1

    print("\n===== สรุป =====")
    print(f"session ที่ประมวลผลสำเร็จ: {stats['sessions']} (ข้าม {stats['sessions_skipped']})")
    print(f"hit candidates: is_hit=1 -> {stats['hit_pos']}, is_hit=0 -> {stats['hit_neg']}")
    print(f"stroke rows ต่อ type: {stats['stroke_types']}")
    print(f"\nwrote: {hit_csv_path}")
    print(f"wrote: {stroke_csv_path}")


if __name__ == "__main__":
    main()
