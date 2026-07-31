"""แยก detection ออกเป็น 'ลูกนอนนิ่ง' (ในสนาม/รถเข็น) vs 'ลูกที่ลอยอยู่จริง'

วิธี: detection ที่มีเพื่อนอยู่ตำแหน่งเดียวกัน (<8px) ในเฟรมข้างเคียง ±W
เกินครึ่ง → เป็นวัตถุนิ่ง
"""
import pickle, sys
from pathlib import Path
import numpy as np

ROOT = Path("D:/Work/Tennis-cv")
dets = pickle.load(open(ROOT / "scratch_ball_dets.pkl", "rb"))

W = 20        # หน้าต่างข้างเคียง (เฟรม)
RADIUS = 8.0  # px ถือว่า "ตำแหน่งเดียวกัน"
CONF = 0.15   # ค่าที่ pipeline ใช้จริง

for name, per_frame in dets.items():
    n = len(per_frame)
    # กรอง conf
    pf = [[d for d in f if d[4] >= CONF] for f in per_frame]

    static_frames = 0   # เฟรมที่มีเฉพาะลูกนิ่ง
    moving_frames = 0   # เฟรมที่มีลูกเคลื่อนที่ >=1
    n_static_det = n_moving_det = 0
    moving_mask = np.zeros(n, dtype=bool)

    for i, fr in enumerate(pf):
        has_moving = False
        for d in fr:
            x, y = d[0], d[1]
            lo, hi = max(0, i - W), min(n, i + W + 1)
            neigh = 0
            for j in range(lo, hi):
                if j == i:
                    continue
                if any((e[0]-x)**2 + (e[1]-y)**2 < RADIUS**2 for e in pf[j]):
                    neigh += 1
            frac = neigh / max(1, (hi - lo - 1))
            if frac > 0.5:
                n_static_det += 1
            else:
                n_moving_det += 1
                has_moving = True
        if has_moving:
            moving_frames += 1
            moving_mask[i] = True
        elif fr:
            static_frames += 1

    # gap ของ "ลูกที่เคลื่อนที่"
    gaps, cur = [], 0
    for h in moving_mask:
        if h:
            if cur: gaps.append(cur)
            cur = 0
        else:
            cur += 1
    if cur: gaps.append(cur)

    tot_det = n_static_det + n_moving_det
    print(f"\n=== {name}  ({n} เฟรม, conf>={CONF}) ===")
    print(f"  detection ทั้งหมด      : {tot_det}")
    print(f"    เป็นลูกนอนนิ่ง       : {n_static_det}  ({n_static_det/max(1,tot_det):.1%})")
    print(f"    เป็นลูกเคลื่อนที่     : {n_moving_det}  ({n_moving_det/max(1,tot_det):.1%})")
    print(f"  เฟรมที่มีลูกเคลื่อนที่ : {moving_frames}  ({moving_frames/n:.1%})  <<< สัญญาณจริง")
    print(f"  เฟรมที่มีแต่ลูกนิ่ง    : {static_frames}  ({static_frames/n:.1%})  <<< ตัวลวง")
    print(f"  ช่องว่างลูกเคลื่อนที่  : n={len(gaps)} median={np.median(gaps) if gaps else 0:.0f} "
          f"max={max(gaps) if gaps else 0}")
