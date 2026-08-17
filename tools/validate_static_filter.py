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


def accel_median(traj, real_frames=None):
    """ความเร่งแนวดิ่งกลาง

    real_frames: ถ้าให้มา จะนับเฉพาะช่วงที่ 3 เฟรมติดกัน 'มี detection จริง'
    ทั้งหมด — จำเป็น เพราะถ้านับทุกเฟรม ช่วงที่ Kalman เดาต่อ (เส้นตรง
    ความเร่ง = 0) จะกลบค่ากลางจนอ่านอะไรไม่ได้
    """
    y = traj[:, 1]
    ok = ~np.isnan(y)
    a = []
    for i in range(1, len(y) - 1):
        if not (ok[i - 1] and ok[i] and ok[i + 1]):
            continue
        if real_frames is not None and not all(
                j in real_frames for j in (i - 1, i, i + 1)):
            continue
        a.append(abs(y[i + 1] - 2 * y[i] + y[i - 1]))
    return (float(np.median(a)), len(a)) if a else (float("nan"), 0)


hdr = (f"{'คลิป':24s} {'det ก่อน':>8s} {'det หลัง':>8s} {'เหลือ':>6s} "
       f"{'เฟรมจริง ก่อน':>13s} {'เฟรมจริง หลัง':>13s} "
       f"{'|a| ของจริง ก่อน':>17s} {'|a| ของจริง หลัง':>17s}")
print(hdr)
print("-" * len(hdr))
for name, per_frame in dets.items():
    n = len(per_frame)
    bb = to_bboxes(per_frame)
    bb_f = filter_static_ball_bboxes(bb, n)
    n0 = sum(len(v) for v in bb.values())
    n1 = sum(len(v) for v in bb_f.values())

    t0 = extract_ball_trajectory_kalman(bb, n, drop_static=False)
    t1 = extract_ball_trajectory_kalman(bb, n, drop_static=True)
    # เฟรมที่ค่ามาจาก detection จริง ไม่ใช่ Kalman เดาต่อ
    r0 = {i for i in range(n) if i in bb and not np.isnan(t0[i, 0])}
    r1 = {i for i in range(n) if i in bb_f and not np.isnan(t1[i, 0])}
    a0, c0 = accel_median(t0, r0)
    a1, c1 = accel_median(t1, r1)

    print(f"{name:24s} {n0:8d} {n1:8d} {n1/max(1,n0):5.1%} "
          f"{len(r0)/n:12.1%} {len(r1)/n:12.1%} "
          f"{a0:11.2f} (n={c0:4d}) {a1:11.2f} (n={c1:4d})")

print("\n|a| ของจริง = ความเร่งแนวดิ่งกลาง นับเฉพาะช่วง 3 เฟรมติดที่มี")
print("             detection จริงครบ (ตัดช่วงที่ Kalman เดาต่อออก)")
print("             ลูกที่ลอยจริงควรใกล้ ~4.4 px/frame² ที่ 30fps")
print("เฟรมจริง   = สัดส่วนเฟรมที่ค่ามาจาก detection ไม่ใช่การเดา")
print("             (ก่อนกรอง ตัวเลขนี้รวมลูกที่นอนนิ่งด้วย จึงดูสูงเกินจริง)")
