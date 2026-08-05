"""ทำไม GT บางลูกไม่เกิด candidate เลย — พลาดตั้งแต่ด่านแรก ML ไม่มีสิทธิ์เห็น

_find_wrist_peaks มีเงื่อนไข 3 ข้อ ลูกที่หลุดต้องตกข้อใดข้อหนึ่ง:

  1. speed เป็น NaN            -> track หลุดช่วงนั้น
  2. speed < min_wrist_speed_px -> ขยับช้าเกินเกณฑ์
  3. ไม่ใช่ local max ใน +-8 เฟรม -> มีเฟรมข้าง ๆ สูงกว่า

ข้อ 2 น่าสงสัยที่สุด เพราะ min_wrist_speed_px = 8.0 เป็น **พิกเซลดิบ** ไม่ได้
หารความกว้างไหล่ — คนที่ยืนไกลกล้องเคลื่อนที่กี่ px ก็น้อยกว่าคนที่ยืนใกล้เสมอ
ทั้งที่ตีแรงเท่ากัน นี่คือบั๊กชนิดเดียวกับที่แก้ไปแล้วในชั้นฟีเจอร์
(SCALED_FEATURE_COLS: F1 0.365 -> 0.461) แต่ชั้นสร้าง candidate ยังไม่ได้แก้

สคริปต์นี้แยกให้เห็นว่าตกข้อไหน และเทียบ speed หน่วย px กับหน่วยความกว้างไหล่
ถ้าเป็นข้อ 2 จริง ลูกที่พลาดจะมี speed_px ต่ำแต่ speed_bw ปกติ

รัน:  python scripts/diagnose_missed_candidates.py
"""
import argparse
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loeuf_cv.config import L_SHOULDER, L_WRIST, R_SHOULDER, R_WRIST  # noqa: E402
from loeuf_cv.hit_detection import (_compute_wrist_speed,  # noqa: E402
                                    _f, _find_wrist_peaks, _fps_ratio,
                                    detect_hit_events,
                                    extract_ball_trajectory_kalman)
from train_model.label_ingest import (find_session_labels,  # noqa: E402
                                      load_session_label)

MATCH_TOL = 10
MIN_SPEED_PX = 8.0


def classify(speed_by_side, bw, g, fps):
    """บอกว่า GT frame นี้ตกด่านไหน + คืนค่าที่วัดได้"""
    W = _f(8, fps)
    thr = MIN_SPEED_PX / _fps_ratio(fps)
    best = {"reason": "NaN (track หลุด)", "px": np.nan, "bw": np.nan,
            "side": "-", "peak_px": np.nan}
    for side, sp in speed_by_side.items():
        lo, hi = max(0, g - MATCH_TOL), min(len(sp), g + MATCH_TOL + 1)
        seg = sp[lo:hi]
        if len(seg) == 0 or np.all(np.isnan(seg)):
            continue
        i = lo + int(np.nanargmax(seg))
        px = float(sp[i])
        if not np.isfinite(best["px"]) or px > best["px"]:
            wlo, whi = max(0, i - W), min(len(sp), i + W + 1)
            local_max = np.nanmax(sp[wlo:whi])
            if px < thr:
                reason = "speed ต่ำกว่าเกณฑ์"
            elif px < local_max:
                reason = "ไม่ใช่ local max"
            else:
                reason = "ผ่านด่าน peak แต่โดนด่านอื่นตัด"
            best = {"reason": reason, "px": px, "bw": px / bw if bw else np.nan,
                    "side": side, "peak_px": float(local_max)}
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=str(ROOT / "dataset/benchmark_cache"))
    ap.add_argument("--dataset-root", default=str(ROOT / "dataset"))
    args = ap.parse_args()

    sessions = {}
    for p in find_session_labels(Path(args.dataset_root)):
        try:
            s = load_session_label(p)
        except Exception:
            continue
        sessions[f"{p.parent.name}/{Path(s.video_path).stem}"] = s

    missed, hit_stats = [], []
    for pkl in sorted((Path(args.cache_dir) / "tracking").rglob("*.pkl")):
        key = f"{pkl.parent.name}/{pkl.stem}"
        if key not in sessions:
            continue
        with open(pkl, "rb") as f:
            c = pickle.load(f)
        if c.get("cache_version") != 2:
            continue

        s = sessions[key]
        gt = sorted(st.keyframes["impact"] for st in s.strokes
                    if st.keyframes.get("impact") is not None)
        pids = Counter(st.player_id for st in s.strokes if st.player_id)
        person = pids.most_common(1)[0][0] if pids else key

        track = c["track"]
        vm, fps = c["video_meta"], c["fps"]
        w, h = vm["width"], vm["height"]
        lm = track.pose.landmarks
        traj, meas = extract_ball_trajectory_kalman(
            c["ball_bboxes"], len(lm), return_measured=True, fps=fps)
        cands = detect_hit_events(traj, [track], fps, w, h,
                                  racket_bboxes=c.get("racket_bboxes"),
                                  ball_measured=meas, return_candidates=True)
        cf = [ev["frame"] for ev in cands]

        sw = np.abs(lm[:, R_SHOULDER, 0] - lm[:, L_SHOULDER, 0]) * w
        bw = float(np.nanmedian(sw)) if not np.all(np.isnan(sw)) else 0.0
        speed_by_side = {
            "right": _compute_wrist_speed(track.pose, R_WRIST, w, h),
            "left": _compute_wrist_speed(track.pose, L_WRIST, w, h)}

        for g in gt:
            info = classify(speed_by_side, bw, g, fps)
            if any(abs(g - f) <= MATCH_TOL for f in cf):
                hit_stats.append((info["px"], info["bw"], bw))
            else:
                missed.append((key, person, g, bw, info))

    print(f"=== GT ที่ไม่มี candidate เลย: {len(missed)} ลูก ===\n")
    print(f"{'คลิป':<26s} {'คน':<8s} {'เฟรม':>6s} {'ไหล่ px':>8s} "
          f"{'speed px':>9s} {'speed/ไหล่':>11s}  สาเหตุ")
    print("-" * 92)
    for key, person, g, bw, info in missed:
        print(f"{key:<26s} {person:<8s} {g:6d} {bw:8.1f} {info['px']:9.2f} "
              f"{info['bw']:11.3f}  {info['reason']}")

    print(f"\n=== เทียบกับ {len(hit_stats)} ลูกที่เกิด candidate ===")
    a = np.array([x for x in hit_stats if np.isfinite(x[0])])
    m = np.array([[i['px'], i['bw'], b] for _, _, _, b, i in missed
                  if np.isfinite(i['px'])])
    for lbl, col in (("speed ที่ GT (px)", 0), ("speed/ความกว้างไหล่", 1),
                     ("ความกว้างไหล่ (px)", 2)):
        print(f"{lbl:<24s} เจอ: median {np.median(a[:, col]):7.2f} · "
              f"พลาด: median {np.median(m[:, col]):7.2f}")

    reasons = Counter(i["reason"] for *_, i in missed)
    print("\nสรุปสาเหตุ:")
    for r, n in reasons.most_common():
        print(f"  {n:2d}  {r}")


if __name__ == "__main__":
    main()
