"""เฟรมปะทะที่รายงานออกไป คลาดจากเฉลยเพราะอะไร — เลือกผิด หรือไม่เคยมีให้เลือก

ที่มา: แยกส่วนที่ทำให้ Mode A acceptance ได้แค่ 0.410 ออกเป็น 3 ก้อน พบว่า

    หาลูกไม่เจอเลย                 23.3%
    เจอลูกแล้วแต่เฟรมปะทะเพี้ยน     23.5%   <- ก้อนนี้ไม่เคยมีใครแตะ
    เพดานของ keyframe logic เอง    12.2%

ก้อนกลางใหญ่พอ ๆ กับก้อนแรก: ลูกที่ "เจอแล้ว" ได้ keyframe ถูกแค่ 53.4% ทั้งที่
เพดาน (Mode B ที่ป้อนเฉลย) คือ 84.0% -> หายไป 30 จุดเพราะเฟรมปะทะไม่ตรงล้วน ๆ

สาเหตุเชิงกลไก: หน้าต่างที่ยอมรับได้ของ impact กว้าง median +-4.25 เฟรม แต่
ความคลาด p90 อยู่ที่ 7 เฟรม และ backswing_peak = impact - ค่าคงที่ ทำให้ความ
คลาดของ impact ส่งต่อไปทั้งดุ้น -> keyframe ทั้งสองตกพร้อมกัน

คำถามที่สคริปต์นี้ตอบ
────────────────────
  Q1  เฟรมที่ถูกอยู่ใน candidate ที่เราสร้างขึ้นมาแล้วหรือยัง
      ถ้าอยู่ = ปัญหาการ "เลือก" -> แก้ที่ NMS/การจัดอันดับ
      ถ้าไม่อยู่ = ปัญหาการ "สร้าง" -> ต้องหาเฟรมใหม่ระหว่าง candidate

  Q2  ความคลาดมีทิศทางแน่นอนไหม (เช่นช้าไปเสมอเพราะไปจับ follow-through)
      ถ้ามี = แก้ด้วยการเลื่อนค่าคงที่ก็ได้ผลทันที

  Q3  ml_prob ของ candidate ที่ "ตรงเฉลยที่สุด" ต่ำกว่าตัวที่ถูกเลือกแค่ไหน
      ถ้าต่ำกว่านิดเดียว = จัดอันดับใหม่ด้วยสัญญาณอื่นก็พอ

⚠️ ต้องรันด้วยโมเดล LOPO เท่านั้น (--hit-classifier-dir) ไม่งั้นได้ภาพที่ดีเกินจริง

รัน:  python scripts/diagnose_impact_offset.py --hit-classifier-dir <dir>
"""
import argparse
import json
import os
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loeuf_cv import hit_detection  # noqa: E402
from loeuf_cv.hit_detection import (ML_PROB_THRESHOLD,  # noqa: E402
                                    detect_hit_events,
                                    extract_ball_trajectory_kalman)
from tools.label_ingest import (find_session_labels,  # noqa: E402
                                      load_session_label)

MATCH_TOL = 10      # เดียวกับ benchmark harness
SEARCH_TOL = 25     # ค้นหา candidate ที่ตรงเฉลยกว่า ในรัศมีกว้างกว่า


def nms(frames_probs, gap):
    chosen = []
    for f, p in sorted(frames_probs, key=lambda x: -x[1]):
        if all(abs(f - k) >= gap for k, _ in chosen):
            chosen.append((f, p))
    return sorted(chosen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=str(ROOT / "dataset/benchmark_cache"))
    ap.add_argument("--dataset-root", default=str(ROOT / "dataset"))
    ap.add_argument("--hit-classifier-dir", default=None)
    ap.add_argument("--threshold", type=float, default=ML_PROB_THRESHOLD)
    args = ap.parse_args()

    lopo = None
    if args.hit_classifier_dir:
        d = Path(args.hit_classifier_dir)
        with open(d / "clip_person.json", encoding="utf-8") as f:
            lopo = (d, json.load(f))
    else:
        print("⚠️ ไม่ได้ใช้โมเดล LOPO — ตัวเลขจะดีเกินจริง\n")

    sessions = {}
    for p in find_session_labels(Path(args.dataset_root)):
        try:
            s = load_session_label(p)
        except Exception:
            continue
        sessions[f"{p.parent.name}/{Path(s.video_path).stem}"] = s

    rows = []
    for pkl in sorted((Path(args.cache_dir) / "tracking").rglob("*.pkl")):
        key = f"{pkl.parent.name}/{pkl.stem}"
        if key not in sessions:
            continue
        with open(pkl, "rb") as f:
            c = pickle.load(f)
        if c.get("cache_version") != 2:
            continue

        if lopo:
            d, mapping = lopo
            person = mapping.get(key)
            if person is None:
                print(f"  ข้าม {key} (ไม่รู้ว่าเป็นของใคร)")
                continue
            os.environ["LOEUF_HIT_CLASSIFIER"] = str(d / f"{person}.pkl")
            hit_detection._hit_clf_cache = None

        s = sessions[key]
        gt = sorted(st.keyframes["impact"] for st in s.strokes
                    if st.keyframes.get("impact") is not None)
        vm, fps = c["video_meta"], c["fps"]
        t = c["track"]
        traj, meas = extract_ball_trajectory_kalman(
            c["ball_bboxes"], len(t.pose.landmarks), return_measured=True,
            fps=fps)
        cands = detect_hit_events(traj, [t], fps, vm["width"], vm["height"],
                                  racket_bboxes=c.get("racket_bboxes"),
                                  ball_measured=meas, return_candidates=True)

        pool = [(ev["frame"], ev.get("ml_prob") or 0.0) for ev in cands]
        keep = [(f, p) for f, p in pool if p >= args.threshold]
        picked = nms(keep, int(fps * 0.75))

        for g in gt:
            near = [(f, p) for f, p in picked if abs(f - g) <= MATCH_TOL]
            if not near:
                continue                       # FN — คนละปัญหา ไม่นับตรงนี้
            sel_f, sel_p = min(near, key=lambda x: abs(x[0] - g))

            # candidate ที่ตรงเฉลยที่สุด "ในบรรดาที่มีอยู่" ไม่ว่า prob เท่าไร
            cand = [(f, p) for f, p in pool if abs(f - g) <= SEARCH_TOL]
            best_f, best_p = min(cand, key=lambda x: abs(x[0] - g))
            rows.append({"clip": key, "gt": g, "sel": sel_f, "sel_p": sel_p,
                         "best": best_f, "best_p": best_p})

    if not rows:
        print("ไม่มีข้อมูล")
        return

    d_sel = np.array([r["sel"] - r["gt"] for r in rows])
    d_best = np.array([r["best"] - r["gt"] for r in rows])
    n = len(rows)
    print(f"=== stroke ที่ตรวจเจอและจับคู่ได้ {n} ลูก ===\n")

    def stat(name, d):
        a = np.abs(d)
        print(f"{name:24s} median {np.median(d):+5.1f} | |Δ| median {np.median(a):4.1f} "
              f"p75 {np.percentile(a,75):4.1f} p90 {np.percentile(a,90):5.1f} "
              f"| ภายใน 4 เฟรม {np.mean(a<=4):5.1%}")

    print("Q2 — ความคลาดของเฟรมที่รายงานจริง")
    stat("  ที่เลือกไปจริง", d_sel)
    print("\nQ1 — ถ้าเลือกตัวที่ตรงเฉลยที่สุดในบรรดา candidate ที่มีอยู่แล้ว")
    stat("  ดีที่สุดที่เป็นไปได้", d_best)

    gain = np.mean(np.abs(d_best) <= 4) - np.mean(np.abs(d_sel) <= 4)
    print(f"\n  -> เพดานที่ได้จากการ 'เลือกใหม่' เพียงอย่างเดียว: +{gain:.1%} จุด")
    same = np.mean(d_sel == d_best)
    print(f"  -> เลือกถูกอยู่แล้ว {same:.1%} ของกรณี")

    print("\nQ3 — prob ของตัวที่ตรงเฉลยที่สุด เทียบตัวที่ถูกเลือก")
    dp = np.array([r["best_p"] - r["sel_p"] for r in rows])
    miss = [r for r in rows if r["sel"] != r["best"]]
    if miss:
        dpm = np.array([r["best_p"] - r["sel_p"] for r in miss])
        print(f"  เฉพาะ {len(miss)} เคสที่เลือกผิด: prob ต่างกัน median {np.median(dpm):+.3f} "
              f"| ตัวที่ควรเลือกมี prob ต่ำกว่า {np.mean(dpm<0):.0%} ของเคส")
        below = [r for r in miss if r["best_p"] < args.threshold]
        print(f"  ใน {len(miss)} เคสนั้น มี {len(below)} เคสที่ตัวควรเลือก "
              f"prob ต่ำกว่าเกณฑ์ {args.threshold} (ถูกกรองทิ้งไปก่อนแล้ว)")
    print(f"  ทุกเคส: prob ต่างกัน median {np.median(dp):+.3f}")

    print("\nทิศทางความคลาด (เฟรมที่เลือก − เฉลย)")
    c = Counter(np.sign(d_sel))
    print(f"  ช้ากว่าเฉลย {c[1]:3d} | ตรงเป๊ะ {c[0]:3d} | เร็วกว่าเฉลย {c[-1]:3d}")


if __name__ == "__main__":
    main()
