#!/usr/bin/env python3
"""Solve camera pose from manually-clicked court points
(labels/court_click_tool.html output) → CameraPose (height/distance/tilt/
pan/focal_length) + optional overlay visualization for sanity-checking fit.

ใช้เมื่อ auto-detect เส้นคอร์ท (court_calibration.py / court_model.py
edge-based search) ไม่น่าเชื่อถือ หรือคอร์ทไม่เข้าเฟรมครบ (ซูมมาก) — คนคลิก
จุดที่เห็นได้จริงในเฟรม (ผ่าน court_click_tool.html) แล้ว solve ตรง ๆ

กล้องไม่ขยับต่อสนามหนึ่งตัว → solve ครั้งเดียวแล้ว reuse ผลลัพธ์
(camera_pose.json) กับทุกคลิปจากกล้องตัวนั้นได้เลย ไม่ต้อง solve ใหม่ทุกคลิป

ตัวอย่าง:
    python scripts/solve_camera_pose.py court_points_2026-07-09.json \
        --frame reference_frame.jpg -o camera_pose.json --overlay fit_check.jpg
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from loeuf_cv.court_model import (
    MIN_DEPTH_SPREAD_M, NAMED_POINTS, PointCorrespondence, depth_spread_m,
    project_court_lines, search_camera_pose_from_points,
)


def load_correspondences(points_json: dict) -> list[PointCorrespondence]:
    corrs = []
    for p in points_json["points"]:
        feature = p["feature"]
        if feature not in NAMED_POINTS:
            print(f"  ⚠️ ข้าม point ที่ไม่รู้จัก: {feature}")
            continue
        corrs.append(PointCorrespondence((p["x"], p["y"]), NAMED_POINTS[feature]))
    return corrs


def draw_overlay(frame_path: str, pose, frame_shape: tuple, corrs, out_path: str):
    import cv2

    from loeuf_cv.court_model import COURT_LINES

    frame = cv2.imread(frame_path)
    if frame is None:
        print(f"  ⚠️ เปิดรูป {frame_path} ไม่ได้ ข้าม overlay")
        return
    frame = cv2.resize(frame, (frame_shape[1], frame_shape[0]))

    lines = project_court_lines(pose, frame_shape, line_names=tuple(COURT_LINES))
    for seg in lines.values():
        if seg is None:
            continue
        p1, p2 = seg
        ok, c1, c2 = cv2.clipLine(
            (0, 0, frame_shape[1], frame_shape[0]),
            (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])))
        if ok:
            cv2.line(frame, c1, c2, (0, 255, 0), 2)

    for corr in corrs:
        x, y = int(corr.image_xy[0]), int(corr.image_xy[1])
        cv2.circle(frame, (x, y), 6, (0, 165, 255), 2)

    cv2.imwrite(out_path, frame)
    print(f"  overlay → {out_path} (เส้นเขียว = ที่โมเดลทำนาย, วงส้ม = จุดที่คลิก)")


def main():
    ap = argparse.ArgumentParser(description="Solve camera pose from clicked court points")
    ap.add_argument("points_json", help="output จาก labels/court_click_tool.html")
    ap.add_argument("--frame", default=None,
                    help="รูปเฟรมเดียวกับที่คลิก (สำหรับวาด --overlay)")
    ap.add_argument("-o", "--output", default=None,
                    help="เซฟผลลัพธ์ CameraPose เป็น JSON (default: <points_json>.pose.json)")
    ap.add_argument("--overlay", default=None,
                    help="เซฟรูป overlay ตรวจสอบ fit ด้วยตา (ต้องมี --frame)")
    ap.add_argument("--n-random", type=int, default=6000)
    ap.add_argument("--n-refine-iters", type=int, default=250)
    ap.add_argument("--n-multistart", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    with open(args.points_json) as f:
        points_json = json.load(f)
    frame_shape = (points_json["frame_height"], points_json["frame_width"])

    corrs = load_correspondences(points_json)
    print(f"โหลดจุดที่คลิกได้ {len(corrs)} จุด")
    if not any(p["feature"].startswith("net_") for p in points_json["points"]):
        print("  ⚠️ ไม่มีจุดเสาเน็ต — ผลลัพธ์อาจไวต่อ noise มากกว่าถ้ามี (ดู CHANGELOG 0.2.6)")
    spread = depth_spread_m(corrs)
    if spread < MIN_DEPTH_SPREAD_M:
        print(f"  🛑 จุดที่คลิกกระจายความลึกแค่ {spread:.1f}m (ต่ำกว่า {MIN_DEPTH_SPREAD_M}m) —"
              " นี่คือสภาวะ degenerate จริง ไม่ใช่แค่ noise: ถ้าจุดทั้งหมดอยู่ระดับความลึก"
              " เดียวกัน (เช่น คลิกแค่ 4 มุมเสาเน็ตอย่างเดียว) ระบบแยก focal length กับ"
              " ระยะกล้องไม่ออก แม้ไม่มี noise เลยก็ยังได้ pose ผิดที่ score สูงพอกัน"
              " → กรุณาคลิกเพิ่มอย่างน้อย 1-2 จุดที่ระดับความลึกอื่น (เช่น baseline หรือ"
              " service line) ก่อน แล้วค่อยรันใหม่")

    pose, score = search_camera_pose_from_points(
        corrs, frame_shape, n_random=args.n_random,
        n_refine_iters=args.n_refine_iters, n_multistart=args.n_multistart,
        rng=np.random.default_rng(args.seed))

    if pose is None:
        print(f"❌ solve ไม่สำเร็จ (ต้องมีอย่างน้อย 4 จุด, ตอนนี้มี {len(corrs)})")
        sys.exit(1)

    print(f"✅ score = {score:.4f}  (1/(1+RMSE_px) — ยิ่งใกล้ 1 ยิ่งดี)")
    print(f"   ท่ากล้อง: {pose.to_debug_dict()}")
    if score < 0.5:
        print("   ⚠️ score ต่ำ — RMSE > 1px คูณผกผัน มาก น่าจะมีจุดคลิกผิด/สลับ feature กัน ลองตรวจสอบ")

    out_path = args.output or str(Path(args.points_json).with_suffix("")) + ".pose.json"
    with open(out_path, "w") as f:
        json.dump({"camera_pose": pose.to_debug_dict(), "score": score,
                   "frame_width": frame_shape[1], "frame_height": frame_shape[0],
                   "n_points": len(corrs)}, f, indent=2)
    print(f"   บันทึก → {out_path} (reuse ผลนี้กับทุกคลิปจากกล้องตัวเดียวกันได้)")

    if args.overlay:
        if not args.frame:
            print("  ⚠️ ต้องระบุ --frame ด้วยถึงจะวาด --overlay ได้")
        else:
            draw_overlay(args.frame, pose, frame_shape, corrs, args.overlay)


if __name__ == "__main__":
    main()
