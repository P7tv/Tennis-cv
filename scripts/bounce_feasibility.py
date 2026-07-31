"""โอกาสที่จะตรวจจับ 'จุดที่ลูกตก' ได้จริง

การหา bounce ต้องเห็นลูกเป็นรูป V: ลงมา -> แตะพื้น -> เด้งขึ้น
เงื่อนไขขั้นต่ำ = มี detection >=2 จุดใน 4 เฟรมก่อน และ >=2 จุดใน 4 เฟรมหลัง
"""
import pickle
from pathlib import Path
import numpy as np

ROOT = Path("D:/Work/Tennis-cv")
dets = pickle.load(open(ROOT / "scratch_ball_dets.pkl", "rb"))
W, RADIUS, CONF = 20, 8.0, 0.15
K = 4          # หน้าต่างข้างละกี่เฟรม
NEED = 2       # ต้องมีกี่จุดในแต่ละข้าง

print(f"เงื่อนไข: มี detection ลูกเคลื่อนที่ >= {NEED} จุด ทั้งใน {K} เฟรมก่อน และ {K} เฟรมหลัง\n")
for name, per_frame in dets.items():
    n = len(per_frame)
    pf = [[d for d in f if d[4] >= CONF] for f in per_frame]
    moving = np.zeros(n, dtype=bool)
    for i, fr in enumerate(pf):
        for d in fr:
            x, y = d[0], d[1]
            lo, hi = max(0, i - W), min(n, i + W + 1)
            neigh = sum(1 for j in range(lo, hi) if j != i and
                        any((e[0]-x)**2 + (e[1]-y)**2 < RADIUS**2 for e in pf[j]))
            if neigh / max(1, hi - lo - 1) <= 0.5:
                moving[i] = True
                break

    ok = 0
    for i in range(K, n - K):
        if moving[i-K:i].sum() >= NEED and moving[i+1:i+1+K].sum() >= NEED:
            ok += 1
    print(f"{name:26s}  เฟรมที่วัด bounce ได้: {ok:5d}/{n}  ({ok/n:6.1%})   "
          f"[เห็นลูกเลย {moving.sum()/n:5.1%}]")
