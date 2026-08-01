"""ส่งออกข้อมูลสำหรับหน้าจอ 'โค้ชย้อนดูคลิป' — stroke + marker + metric

ใช้ cache v2 จึงไม่ต้องรัน YOLO ใหม่ ออกมาเป็น JSON ก้อนเดียวที่หน้าเว็บ
โหลดไปแสดงได้เลย (ไม่มี dependency ภายนอก)

โครงสร้าง output:
  {
    clip, fps, total_frames, duration_sec,
    strokes: [{
       index, stroke_type, start_frame, end_frame, confidence,
       markers: [{name, frame, ms, detected}],   <- ให้กดกระโดดไปดู
       joints:  {marker_name: {joint: {x, y}}},  <- วาดโครงร่างท่าทาง
       metrics: {ชื่อ: ค่า}                       <- ให้โค้ชอ่าน
    }]
  }
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loeuf_cv.config import PipelineConfig  # noqa: E402
from loeuf_cv.hit_detection import (detect_hit_events,  # noqa: E402
                                    extract_ball_trajectory_kalman)
from loeuf_cv.schema_builder.builder import build_loeuf_schema  # noqa: E402

MARKER_ORDER = ["unit_turn", "backswing_peak", "trophy_position", "impact",
                "follow_through_peak", "recovery_position"]

# metric ที่โค้ชอ่านรู้เรื่อง — คัดจาก 39 ตัวที่มีค่าจริง
# (ดู docs/QUALITY_GRADING_FEASIBILITY.md สำหรับที่มา)
COACH_METRICS = [
    ("metric.body.shoulder_rotation_at_unit_turn_deg", "หมุนไหล่ตอนตั้งท่า", "°"),
    ("metric.body.hip_rotation_at_unit_turn_deg", "หมุนสะโพกตอนตั้งท่า", "°"),
    ("metric.body.shoulder_hip_separation_at_impact_deg", "แยกไหล่-สะโพกตอนปะทะ", "°"),
    ("metric.contact.contact_height_cm", "ความสูงจุดปะทะ", "ซม."),
    ("metric.timing.backswing_duration_ms", "ช่วงง้างไม้", "ms"),
    ("metric.timing.forward_swing_duration_ms", "ช่วงสวิงไปหน้า", "ms"),
    ("metric.timing.tempo_ratio", "อัตราส่วนจังหวะ", ""),
    ("metric.movement.footwork_distance_cm", "ระยะที่เท้าขยับ", "ซม."),
    ("metric.movement.stance_width_at_impact_cm", "ความกว้างขาตอนปะทะ", "ซม."),
    ("kinematics.racket_head_speed_mps", "ความเร็วหน้าไม้", "ม./วิ"),
]


def dig(d, path):
    cur = d
    for part in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur if isinstance(cur, (int, float)) and not isinstance(cur, bool) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="path ของ tracking pkl (cache v2)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-strokes", type=int, default=12)
    args = ap.parse_args()

    with open(args.cache, "rb") as f:
        c = pickle.load(f)
    if c.get("cache_version") != 2:
        raise SystemExit("ต้องใช้ cache v2 (มี ball_bboxes)")

    track, vm, fps = c["track"], c["video_meta"], c["fps"]
    cfg = PipelineConfig(dominant_side=c["dominant_side"],
                         subject_height_cm=c["subject_height_cm"])
    traj, meas = extract_ball_trajectory_kalman(
        c["ball_bboxes"], len(track.pose.landmarks), return_measured=True)
    hits = detect_hit_events(traj, [track], fps, vm["width"], vm["height"],
                             racket_bboxes=c.get("racket_bboxes"),
                             ball_measured=meas)
    schema = build_loeuf_schema([track], hits, fps, vm, cfg,
                                racket_keypoints=c.get("racket_keypoints"),
                                ball_traj=traj)

    out_strokes = []
    for i, s in enumerate(schema.get("strokes", [])[:args.max_strokes], 1):
        root, kfb = s.get("stroke_root", {}), s.get("keyframe", {})
        markers, joints = [], {}
        for name in MARKER_ORDER:
            e = kfb.get(name) or {}
            if e.get("frame_index") is None:
                continue
            markers.append({"name": name, "frame": e["frame_index"],
                            "ms": e.get("timestamp_ms"),
                            "detected": bool(e.get("detected"))})
            if e.get("joint"):
                joints[name] = {k: v for k, v in e["joint"].items()
                                if isinstance(v, dict) and v.get("x") is not None}
        metrics = []
        for path, label, unit in COACH_METRICS:
            v = dig(s, path)
            if v is not None:
                metrics.append({"label": label, "value": round(v, 1), "unit": unit})
        out_strokes.append({
            "index": i,
            "stroke_type": root.get("stroke_type"),
            "start_frame": root.get("stroke_start_frame"),
            "end_frame": root.get("stroke_end_frame"),
            "detection_status": root.get("detection_status"),
            "is_clean": root.get("is_clean_stroke"),
            "markers": markers, "joints": joints, "metrics": metrics,
        })

    n = vm.get("total_frames", 0)
    bundle = {
        "clip": Path(args.cache).stem,
        "fps": round(float(fps), 2),
        "total_frames": n,
        "duration_sec": round(n / fps, 1) if fps else 0,
        "n_strokes_total": len(schema.get("strokes", [])),
        "strokes": out_strokes,
    }
    Path(args.out).write_text(json.dumps(bundle, ensure_ascii=False, indent=1),
                              encoding="utf-8")
    print(f"เขียน {args.out} — {len(out_strokes)} stroke "
          f"(จากทั้งหมด {bundle['n_strokes_total']})")


if __name__ == "__main__":
    main()
