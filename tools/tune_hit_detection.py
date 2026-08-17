"""จูน hit detection จาก tracking cache — ไม่รัน YOLO ใหม่

hit detection คือคอขวดของเกณฑ์รับงาน: recall 0.606 x ตรง ±1 เฟรม 0.29 = 0.17
สคริปต์นี้ sweep 3 ปัจจัยแล้ววัดเทียบ GT จริงเพื่อดูว่าอันไหนได้ผล

  1. ml_prob_threshold   — ของเดิม 0.4 (permissive มาก precision 0.101)
  2. NMS mode            — ของเดิมไล่ตามลำดับเฟรม vs เรียงตามคะแนน
  3. ball snapping       — ดึงเฟรมที่ทายเข้าหา "จุดที่ลูกเปลี่ยนทิศ" ที่ใกล้สุด
                           (การปะทะลูกคือเหตุการณ์ที่ทำให้ลูกกลับทิศ → เป็น
                            สัญญาณตรงของ impact มากกว่าความเร็วข้อมือ)

ต้องมี cache จาก: python scripts/run_keyframe_benchmark.py track
รัน: python scripts/tune_hit_detection.py
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from loeuf_cv.benchmark_keyframes import (MATCH_TOLERANCE_FRAMES,  # noqa: E402
                                          match_strokes)
from loeuf_cv.hit_detection import (_find_ball_inflections,  # noqa: E402
                                    detect_hit_events)
from tools.label_ingest import (find_session_labels,  # noqa: E402
                                      load_session_label)

ROOT = Path(__file__).resolve().parent.parent
ACC_TOL = 1          # ±1 เฟรม = เกณฑ์ของลูกค้า
MATCH_TOL = MATCH_TOLERANCE_FRAMES


def load_clips(cache_dir: Path, limit: int | None = None):
    """คืน [(ชื่อคลิป, candidates ดิบพร้อม ml_prob, ball_traj, gt_impacts, fps, วินาที)]"""
    out = []
    for lp in find_session_labels(ROOT / "dataset"):
        sess = load_session_label(lp)
        f = cache_dir / "tracking" / sess.video_path.parent.name / f"{sess.video_path.stem}.pkl"
        if not f.exists():
            continue
        d = pickle.loads(f.read_bytes())
        vm = d.get("video_meta") or {}
        gt = sorted(s.keyframes["impact"] for s in sess.strokes
                    if s.keyframes.get("impact") is not None)
        cands = detect_hit_events(
            d["ball_traj"], [d["track"]], d["fps"],
            vm.get("width", 1920), vm.get("height", 1080),
            racket_bboxes=d.get("racket_bboxes"),
            return_candidates=True,
        )
        out.append({
            "clip": f"{sess.video_path.parent.name}/{sess.video_path.stem}",
            "cands": cands, "gt": gt, "fps": d["fps"],
            "sec": d.get("duration_sec") or 0.0,
            "inflections": _find_ball_inflections(d["ball_traj"]),
        })
        print(f"  loaded {out[-1]['clip']:34} candidates={len(cands):4} GT={len(gt):3}")
        if limit and len(out) >= limit:
            break
    return out


def postprocess(clip, thr: float, nms_by_prob: bool, snap: int) -> list[int]:
    """เลียนแบบ pass 2-3 ของ detect_hit_events แล้วคืนเฟรมที่ทาย"""
    fps = clip["fps"]
    min_gap = int(fps * 0.75)
    kept = [c for c in clip["cands"]
            if c["ml_prob"] is None or c["ml_prob"] >= thr]

    if nms_by_prob:
        chosen: list[dict] = []
        for c in sorted(kept, key=lambda x: -(x["ml_prob"] or 0.0)):
            if all(abs(c["frame"] - k["frame"]) >= min_gap for k in chosen):
                chosen.append(c)
        final = sorted(chosen, key=lambda x: x["frame"])
    else:
        final = []
        for c in kept:
            if not final or c["frame"] - final[-1]["frame"] >= min_gap:
                final.append(c)
            else:
                rank = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
                if rank.get(c["confidence"], 0) > rank.get(final[-1]["confidence"], 0):
                    final[-1] = c

    frames = [c["frame"] for c in final]
    if snap > 0 and clip["inflections"]:
        infl = np.array(sorted(clip["inflections"]))
        snapped = []
        for fr in frames:
            near = infl[np.abs(infl - fr) <= snap]
            snapped.append(int(near[np.argmin(np.abs(near - fr))]) if len(near) else fr)
        frames = snapped
    return sorted(frames)


def evaluate(clips, thr, nms_by_prob, snap):
    tp = fn = fp = 0
    within = 0
    total_gt = 0
    minutes = 0.0
    for c in clips:
        pred = postprocess(c, thr, nms_by_prob, snap)
        m = match_strokes(c["gt"], pred, MATCH_TOL)
        tp += len(m["matches"])
        fn += len(m["fn"])
        fp += len(m["fp"])
        within += sum(1 for _, _, d in m["matches"] if abs(d) <= ACC_TOL)
        total_gt += len(c["gt"])
        minutes += (c["sec"] or 0.0) / 60.0
    recall = tp / total_gt if total_gt else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    # impact accuracy end-to-end = สัดส่วน GT ที่ทั้ง "เจอ" และ "ตรง ±1 เฟรม"
    impact_acc = within / total_gt if total_gt else 0.0
    return dict(recall=recall, precision=prec, impact_acc=impact_acc,
                tp=tp, fn=fn, fp=fp,
                fp_per_min=(fp / minutes if minutes else 0.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="dataset/benchmark_cache")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    print("โหลด candidates ดิบจาก cache (ไม่รัน YOLO)...")
    clips = load_clips(ROOT / args.cache_dir, args.limit)
    if not clips:
        print("ไม่มี tracking cache — รัน scripts/run_keyframe_benchmark.py track ก่อน")
        return
    tot_cand = sum(len(c["cands"]) for c in clips)
    tot_gt = sum(len(c["gt"]) for c in clips)
    print(f"\nรวม {len(clips)} คลิป · candidates ดิบ {tot_cand} · GT {tot_gt}\n")

    print("=" * 100)
    print("baseline = ของเดิม (threshold 0.4 · NMS ตามลำดับเฟรม · ไม่ snap)")
    print("=" * 100)
    hdr = (f'{"threshold":>10}{"NMS":>12}{"snap":>6}'
           f'{"recall":>9}{"precision":>11}{"impact_acc":>12}'
           f'{"FP/min":>9}{"TP":>5}{"FN":>5}{"FP":>6}')
    print(hdr)

    rows = []
    for nms_by_prob in (False, True):
        for thr in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
            for snap in (0, 3):
                r = evaluate(clips, thr, nms_by_prob, snap)
                rows.append(((thr, nms_by_prob, snap), r))
                tag = "by_prob" if nms_by_prob else "by_frame"
                print(f'{thr:>10.2f}{tag:>12}{snap:>6}'
                      f'{r["recall"]:>9.3f}{r["precision"]:>11.3f}'
                      f'{r["impact_acc"]:>12.3f}{r["fp_per_min"]:>9.1f}'
                      f'{r["tp"]:>5}{r["fn"]:>5}{r["fp"]:>6}')

    best = max(rows, key=lambda kv: kv[1]["impact_acc"])
    (thr, nms, snap), r = best
    base = next(v for k, v in rows if k == (0.4, False, 0))
    print()
    print(f'ดีสุดตาม impact_acc: threshold={thr} NMS={"by_prob" if nms else "by_frame"} snap={snap}')
    print(f'  impact_acc {base["impact_acc"]:.3f} -> {r["impact_acc"]:.3f}'
          f'   recall {base["recall"]:.3f} -> {r["recall"]:.3f}'
          f'   precision {base["precision"]:.3f} -> {r["precision"]:.3f}')
    print()
    print("หมายเหตุ: impact_acc = สัดส่วน GT ที่ทั้ง detect เจอ *และ* ตรงภายใน ±1 เฟรม")
    print("  = ตัวที่เข้าสมการเกณฑ์รับงานโดยตรง (อีกครึ่งคือ backswing ซึ่งตามเฟรม impact)")
    print("⚠️ ค่าที่เลือกจากตารางนี้คือการจูนบนชุดเดียวกับที่วัด — ถ้าจะอ้างตัวเลข")
    print("   กับลูกค้าต้องยืนยันบน held-out (แบ่งตามผู้เล่น) ก่อน")


if __name__ == "__main__":
    main()
