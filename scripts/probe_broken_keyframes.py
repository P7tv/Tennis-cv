"""วัดว่า keyframe ที่พังอยู่ (B1/B4/B5/B6) ควรอยู่ตรงไหนตาม GT

ตัวเลขที่วัดได้ (Mode B = ป้อน impact ที่ถูกต้องให้แล้ว) ยังต่ำมาก:
  unit_turn 0.000-0.308 · follow_through_peak 0.000-0.200
  trophy_position 0.094 · recovery_position 0.000

วิธี: ดูว่า GT วาง keyframe เหล่านี้ห่างจากจุดอ้างอิงเท่าไหร่ (median/sd)
แล้วเทียบว่า "ถ้าใช้ offset คงที่ที่ดีที่สุด" จะได้เท่าไหร่ — เป็นเพดานของ
วิธีที่ง่ายที่สุด ถ้าเพดานสูงพอก็ไม่ต้องทำอะไรซับซ้อน
(วิธีเดียวกับที่ใช้แก้ backswing_peak จนขึ้นจาก 0.05 เป็น 0.76)
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train_model.label_ingest import find_session_labels  # noqa: E402

# (ชื่อ keyframe, จุดอ้างอิง) — วัด offset จากจุดอ้างอิงนั้น
PAIRS = [
    ("unit_turn", "backswing_peak"),
    ("unit_turn", "impact"),
    ("follow_through_peak", "impact"),
    ("recovery_position", "follow_through_peak"),
    ("recovery_position", "impact"),
    ("trophy_position", "impact"),
    ("trophy_position", "backswing_peak"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default=str(ROOT / "dataset"))
    ap.add_argument("--tolerance", type=int, default=1)
    args = ap.parse_args()

    # อ่านจาก JSON ดิบ เพราะ label_ingest ไม่ได้ดึง trophy/recovery ครบ
    strokes = []
    for p in find_session_labels(Path(args.dataset_root)):
        raw = json.loads(Path(p).read_text(encoding="utf-8"))
        for s in raw["strokes"]:
            strokes.append({
                "type": s.get("stroke_type"),
                "player": s.get("player_id"),
                "unit_turn": s.get("unit_turn_frame"),
                "backswing_peak": s.get("backswing_peak_frame"),
                "impact": s.get("impact_frame"),
                "follow_through_peak": s.get("follow_through_peak_frame"),
                "recovery_position": (s.get("recovery_frame")
                                      if s.get("recovery_restored") else None),
                "trophy_position": s.get("trophy_position_frame"),
            })
    print(f"stroke ทั้งหมด {len(strokes)}\n")

    for kf, ref in PAIRS:
        by_type = defaultdict(list)
        for s in strokes:
            if s[kf] is None or s[ref] is None:
                continue
            by_type[s["type"]].append(int(s[kf]) - int(s[ref]))
        if not by_type:
            print(f"=== {kf} − {ref}: ไม่มี GT ===\n")
            continue
        allv = [v for vs in by_type.values() for v in vs]
        print(f"=== {kf} − {ref}  (n={len(allv)}) ===")
        print(f"{'ท่า':6s} {'n':>4s} {'median':>7s} {'mean':>7s} {'sd':>6s} "
              f"{'เพดาน offset':>13s} {'offset ดีสุด':>13s}")
        for t in sorted(by_type):
            d = np.array(by_type[t])
            best_off, best_acc = 0, 0.0
            for off in range(int(d.min()), int(d.max()) + 1):
                acc = float(np.mean(np.abs(d - off) <= args.tolerance))
                if acc > best_acc:
                    best_off, best_acc = off, acc
            print(f"{t:6s} {len(d):4d} {np.median(d):7.1f} {d.mean():7.2f} "
                  f"{d.std():6.2f} {best_acc:13.1%} {best_off:13d}")
        d = np.array(allv)
        best_off, best_acc = 0, 0.0
        for off in range(int(d.min()), int(d.max()) + 1):
            acc = float(np.mean(np.abs(d - off) <= args.tolerance))
            if acc > best_acc:
                best_off, best_acc = off, acc
        print(f"{'รวม':6s} {len(d):4d} {np.median(d):7.1f} {d.mean():7.2f} "
              f"{d.std():6.2f} {best_acc:13.1%} {best_off:13d}")

        # ── Leave-One-Person-Out: หา offset จากคนอื่น แล้ววัดกับคนที่กันไว้ ──
        # ตัวเลขนี้คือของจริงที่จะได้กับผู้เล่นใหม่ ส่วน 'เพดาน offset' ข้างบน
        # คือค่าที่ fit กับข้อมูลชุดนี้ (สูงเกินจริงเสมอ)
        pool = [(s["type"], s["player"], int(s[kf]) - int(s[ref]))
                for s in strokes if s[kf] is not None and s[ref] is not None]
        people = sorted({p for _, p, _ in pool if p})
        lopo_hit = defaultdict(list)
        for held in people:
            train = [(t, v) for t, p, v in pool if p != held]
            for t in {t for t, p, _ in pool if p == held}:
                tv = [v for tt, v in train if tt == t]
                if not tv:
                    continue
                off = int(round(float(np.median(tv))))
                for tt, p, v in pool:
                    if p == held and tt == t:
                        lopo_hit[t].append(abs(v - off) <= args.tolerance)
        if lopo_hit:
            parts = " · ".join(
                f"{t} {np.mean(v):.0%}({len(v)})" for t, v in sorted(lopo_hit.items()))
            allh = [x for v in lopo_hit.values() for x in v]
            print(f"{'LOPO':6s} {len(allh):4d} {'':7s} {'':7s} {'':6s} "
                  f"{np.mean(allh):13.1%}   <- ของจริงกับผู้เล่นใหม่")
            print(f"       {parts}")
        print()


if __name__ == "__main__":
    main()
