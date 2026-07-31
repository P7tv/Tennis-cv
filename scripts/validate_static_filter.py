"""เทียบคุณภาพ ball trajectory ก่อน/หลังเปิดตัวกรองลูกนิ่ง

ใช้ detection ที่ dump ไว้แล้วจาก scripts/dump_ball_dets.py (ไม่ต้องรัน YOLO ซ้ำ)

ตัวชี้วัด:
  - detection ที่เหลือหลังกรอง
  - สัดส่วนเฟรมที่ trajectory มาจาก 'ของจริง' (มี detection ในเฟรมนั้น)
    เทียบกับที่ Kalman เดาต่อเอง
  - ความเร่งแนวดิ่งกลาง: ลูกที่ลอยจริงต้อง != 0 (ตกอิสระ ~4.4 px/frame² ที่ 30fps)
    ถ้าเป็น 0 แปลว่าเป็นเส้นตรง = ของที่ Kalman เดา หรือเกาะลูกนิ่งอยู่
"""
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from loeuf_cv.hit_detection import (  # noqa: E402
    extract_ball_trajectory_kalman, filter_static_ball_bboxes)

CONF = 0.15
src = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "scratch_ball_dets.pkl"
dets = pickle.load(open(src, "rb"))


def to_bboxes(per_frame):
    """[(cx, cy, w, h, conf)] ต่อเฟรม -> dict[frame -> list[(x, y, w, h)]]"""
    out = {}
    for i, fr in enumerate(per_frame):
        keep = [(int(cx - w / 2), int(cy - h / 2), int(w), int(h))
                for cx, cy, w, h, c in fr if c >= CONF]
        if keep:
            out[i] = keep
    return out


def accel_median(traj):
    y = traj[:, 1]
    ok = ~np.isnan(y)
    a = []
    for i in range(1, len(y) - 1):
        if ok[i - 1] and ok[i] and ok[i + 1]:
            a.append(abs(y[i + 1] - 2 * y[i] + y[i - 1]))
    return float(np.median(a)) if a else float("nan")


print(f"{'คลิป':26s} {'det ก่อน':>9s} {'det หลัง':>9s} {'เหลือ':>7s} "
      f"{'|a| ก่อน':>9s} {'|a| หลัง':>9s} {'จริง ก่อน':>10s} {'จริง หลัง':>10s}")
print("-" * 100)
for name, per_frame in dets.items():
    n = len(per_frame)
    bb = to_bboxes(per_frame)
    bb_f = filter_static_ball_bboxes(bb, n)
    n0 = sum(len(v) for v in bb.values())
    n1 = sum(len(v) for v in bb_f.values())

    t0 = extract_ball_trajectory_kalman(bb, n, drop_static=False)
    t1 = extract_ball_trajectory_kalman(bb, n, drop_static=True)
    # สัดส่วนเฟรมที่ trajectory มีค่า *และ* เฟรมนั้นมี detection จริง
    real0 = sum(1 for i in range(n) if i in bb and not np.isnan(t0[i, 0])) / n
    real1 = sum(1 for i in range(n) if i in bb_f and not np.isnan(t1[i, 0])) / n

    print(f"{name:26s} {n0:9d} {n1:9d} {n1/max(1,n0):6.1%} "
          f"{accel_median(t0):9.2f} {accel_median(t1):9.2f} "
          f"{real0:9.1%} {real1:9.1%}")

print("\n|a| = ความเร่งแนวดิ่งกลางของ trajectory (px/frame²)")
print("     ลูกที่ลอยจริงควรใกล้ ~4.4 ที่ 30fps · 0.00 = เส้นตรง = ของปลอม")
print("จริง = สัดส่วนเฟรมที่ค่ามาจาก detection จริง ไม่ใช่ Kalman เดาต่อ")
