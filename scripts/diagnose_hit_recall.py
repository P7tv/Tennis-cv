"""ไล่ดูว่า stroke จริงที่ระบบหาไม่เจอ 'ตาย' ที่ขั้นตอนไหน

detect_hit_events มี 4 ด่านที่ candidate อาจหลุด:
  A. หา wrist peak ไม่เจอเลย            (สัญญาณต้นทางไม่มี)
  B. โดนด่าน 'ลูกอยู่ไกลข้อมือ' ปัดตก    (hit_detection.py:562-566)
  C. ml_prob ต่ำกว่า threshold
  D. โดน NMS กดเพราะมีตัวคะแนนสูงกว่าอยู่ใกล้

ใช้ cache v2 (มี ball_bboxes ดิบ) จึงไม่ต้องรัน YOLO ใหม่
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loeuf_cv.config import L_WRIST, R_WRIST  # noqa: E402
from loeuf_cv.hit_detection import (ML_PROB_THRESHOLD,  # noqa: E402
                                    _compute_wrist_speed, _find_wrist_peaks,
                                    detect_hit_events,
                                    extract_ball_trajectory_kalman)
from train_model.label_ingest import (find_session_labels,  # noqa: E402
                                      load_session_label)

TOL = 10  # ±เฟรม ถือว่า candidate ตรงกับ stroke จริง (เท่ากับ match tolerance)


def gt_impacts(dataset_root: Path) -> dict[str, list[int]]:
    out = {}
    for p in find_session_labels(dataset_root):
        try:
            s = load_session_label(p)
        except Exception:
            continue
        key = f"{p.parent.name}/{Path(s.video_path).stem}"
        out[key] = sorted(st.keyframes["impact"] for st in s.strokes
                          if st.keyframes.get("impact") is not None)
    return out


def covered(frames, gt, tol=TOL):
    """GT กี่ตัวที่มี candidate อยู่ใน ±tol"""
    fs = sorted(frames)
    if not fs:
        return 0, []
    miss = [g for g in gt if not any(abs(f - g) <= tol for f in fs)]
    return len(gt) - len(miss), miss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=str(ROOT / "dataset/benchmark_cache"))
    ap.add_argument("--dataset-root", default=str(ROOT / "dataset"))
    ap.add_argument("--ml-threshold", type=float, default=ML_PROB_THRESHOLD)
    ap.add_argument("--trust-coasting", action="store_true",
                    help="พฤติกรรมเดิม: เชื่อตำแหน่งลูกที่ Kalman เดาต่อด้วย")
    ap.add_argument("--sweep", action="store_true",
                    help="กวาด ml threshold แล้วรายงาน recall/precision")
    args = ap.parse_args()

    gts = gt_impacts(Path(args.dataset_root))
    tracking = Path(args.cache_dir) / "tracking"

    tot = {k: 0 for k in ("gt", "A", "B", "C", "D")}
    sweep_data = []
    print(f"{'คลิป':26s} {'GT':>4s} {'A.peak':>7s} {'B.ลูก':>7s} "
          f"{'C.ml':>6s} {'D.NMS':>7s}")
    print("-" * 66)

    for pkl in sorted(tracking.rglob("*.pkl")):
        key = f"{pkl.parent.name}/{pkl.stem}"
        if key not in gts:
            continue
        with open(pkl, "rb") as f:
            c = pickle.load(f)
        if c.get("cache_version") != 2:
            print(f"{key:26s}  (cache v1 — ข้าม)")
            continue

        gt = gts[key]
        track = c["track"]
        vm = c["video_meta"]
        w, h, fps = vm["width"], vm["height"], c["fps"]
        n = len(track.pose.landmarks)
        traj, meas = extract_ball_trajectory_kalman(
            c["ball_bboxes"], n, return_measured=True)
        if not args.trust_coasting:
            kw = {"ball_measured": meas}
        else:
            kw = {}

        # ── A: wrist peak ดิบ ──
        peaks = []
        for widx in (R_WRIST, L_WRIST):
            sp = _compute_wrist_speed(track.pose, widx, w, h)
            peaks += _find_wrist_peaks(sp)
        a, _ = covered(peaks, gt)

        # ── B: หลังด่าน 'ลูกไกลข้อมือ' (= candidate ที่ออกจาก return_candidates) ──
        cands = detect_hit_events(traj, [track], fps, w, h,
                                  racket_bboxes=c.get("racket_bboxes"),
                                  return_candidates=True, **kw)
        b, _ = covered([x["frame"] for x in cands], gt)

        # ── C: หลังกรอง ml_prob ──
        kept = [x for x in cands
                if x["ml_prob"] is None or x["ml_prob"] >= args.ml_threshold]
        cc, _ = covered([x["frame"] for x in kept], gt)

        # ── D: ผลสุดท้ายหลัง NMS ──
        final = detect_hit_events(traj, [track], fps, w, h,
                                  racket_bboxes=c.get("racket_bboxes"),
                                  ml_prob_threshold=args.ml_threshold, **kw)
        d, _ = covered([x["frame"] for x in final], gt)

        for k, v in (("gt", len(gt)), ("A", a), ("B", b), ("C", cc), ("D", d)):
            tot[k] += v
        print(f"{key:26s} {len(gt):4d} {a:7d} {b:7d} {cc:6d} {d:7d}")
        sweep_data.append((gt, cands, fps))

    g = max(1, tot["gt"])
    print("-" * 66)
    print(f"{'รวม':26s} {tot['gt']:4d} {tot['A']:7d} {tot['B']:7d} "
          f"{tot['C']:6d} {tot['D']:7d}")
    print(f"{'recall':26s} {'':4s} {tot['A']/g:7.1%} {tot['B']/g:7.1%} "
          f"{tot['C']/g:6.1%} {tot['D']/g:7.1%}")
    print(f"\nหายไปแต่ละด่าน: A(ไม่มี peak) {g-tot['A']} · "
          f"B(ด่านลูก) {tot['A']-tot['B']} · "
          f"C(ml) {tot['B']-tot['C']} · D(NMS) {tot['C']-tot['D']}")

    if not args.sweep:
        return

    print(f"\n{'threshold':>10s} {'TP':>5s} {'FN':>5s} {'FP':>5s} "
          f"{'recall':>8s} {'precision':>10s} {'F1':>7s}")
    print("-" * 56)
    for th in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
        TP = FN = FP = 0
        for gt, cands, fps in sweep_data:
            kept = [x for x in cands
                    if x["ml_prob"] is None or x["ml_prob"] >= th]
            gap = int(fps * 0.75)
            chosen = []
            for c in sorted(kept, key=lambda x: -(x["ml_prob"] or 0.0)):
                if all(abs(c["frame"] - k["frame"]) >= gap for k in chosen):
                    chosen.append(c)
            fs = sorted(x["frame"] for x in chosen)
            used = set()
            for gf in gt:
                m = [f for f in fs if abs(f - gf) <= TOL and f not in used]
                if m:
                    used.add(min(m, key=lambda f: abs(f - gf)))
                    TP += 1
                else:
                    FN += 1
            FP += len(fs) - len(used)
        rec = TP / max(1, TP + FN)
        pre = TP / max(1, TP + FP)
        f1 = 2 * rec * pre / max(1e-9, rec + pre)
        print(f"{th:10.2f} {TP:5d} {FN:5d} {FP:5d} "
              f"{rec:8.1%} {pre:10.1%} {f1:7.3f}")


if __name__ == "__main__":
    main()
