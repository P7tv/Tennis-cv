"""หาสัญญาณที่ระบุ 'เฟรมปะทะ' ได้แม่นกว่าจุดที่ข้อมือเร็วสุด

ปัญหาที่วัดไว้: ระบบหา stroke เจอแล้ว (recall 0.87) แต่ระบุเฟรมปะทะพลาด
— มีแค่ ~31% ที่อยู่ใน ±1 เฟรมตามเกณฑ์รับงาน

วิธีทดสอบ: จาก GT impact แต่ละตัว ดูสัญญาณต่าง ๆ ในหน้าต่างรอบ ๆ แล้ววัดว่า
จุดสูงสุด/ต่ำสุดของสัญญาณนั้นห่างจาก GT กี่เฟรม เทียบกับ baseline (wrist peak)

⚠️ นี่คือขั้น 'คัดกรอง' บนข้อมูลชุดเดียวกัน — สัญญาณที่ชนะต้องเอาไป
ยืนยันแบบ Leave-One-Person-Out อีกที ก่อนเชื่อ (บทเรียนจากรอบก่อน)
"""
import argparse
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loeuf_cv.config import L_WRIST, R_WRIST  # noqa: E402
from loeuf_cv.hit_detection import (_compute_wrist_speed,  # noqa: E402
                                    _find_wrist_peaks)
from train_model.label_ingest import (find_session_labels,  # noqa: E402
                                      load_session_label)

W = 6  # ค้นหาสัญญาณในหน้าต่าง ±W รอบ baseline


def racket_center_series(racket_bboxes: dict, n: int) -> np.ndarray:
    """จุดกึ่งกลาง bbox ไม้ต่อเฟรม (nan = ไม่เจอ)

    ถ้าเฟรมไหนเจอหลายอัน เลือกอันที่ใกล้เฟรมก่อนหน้าสุด
    """
    c = np.full((n, 2), np.nan)
    last = None
    for f in range(n):
        bs = racket_bboxes.get(f) or []
        if not bs:
            continue
        pts = [(x + w / 2.0, y + h / 2.0) for x, y, w, h in bs]
        p = (min(pts, key=lambda q: (q[0] - last[0]) ** 2 + (q[1] - last[1]) ** 2)
             if last else pts[0])
        c[f] = p
        last = p
    return c


def racket_area_series(racket_bboxes: dict, n: int) -> np.ndarray:
    """พื้นที่ bbox ไม้ — ตอนไม้หันหน้าเข้าหากล้อง (จังหวะปะทะ) จะกว้างสุด"""
    a = np.full(n, np.nan)
    for f in range(n):
        bs = racket_bboxes.get(f) or []
        if bs:
            a[f] = max(w * h for _, _, w, h in bs)
    return a


def _speed(series: np.ndarray) -> np.ndarray:
    n = len(series)
    s = np.full(n, np.nan)
    for i in range(1, n - 1):
        if not np.isnan(series[i - 1]).any() and not np.isnan(series[i + 1]).any():
            d = series[i + 1] - series[i - 1]
            s[i] = float(np.hypot(d[0], d[1]) / 2.0)
    return s


def _subframe_peak(sig: np.ndarray, center: int, w: int):
    """หา peak แบบ sub-frame ด้วยพาราโบลาผ่าน 3 จุด แล้วปัดเป็นเฟรม

    ถ้าจังหวะปะทะจริงตกอยู่ระหว่างเฟรม การปัดจากยอดพาราโบลาอาจใกล้กว่า
    การเอาเฟรมที่ค่าสูงสุดตรง ๆ
    """
    p = _argext(sig, center, w, "max")
    if p is None or p <= 0 or p >= len(sig) - 1:
        return p
    y0, y1, y2 = sig[p - 1], sig[p], sig[p + 1]
    if np.isnan([y0, y1, y2]).any():
        return p
    denom = y0 - 2 * y1 + y2
    if abs(denom) < 1e-9:
        return p
    return int(round(p + 0.5 * (y0 - y2) / denom))


def _dir_change(pose, width: int, height: int) -> np.ndarray:
    """cosine ของมุมหักเหของข้อมือ — ค่าต่ำ = เปลี่ยนทิศแรง"""
    lm = pose.landmarks
    n = len(lm)
    out = np.full(n, 1.0)
    for widx in (R_WRIST, L_WRIST):
        wx, wy = lm[:, widx, 0] * width, lm[:, widx, 1] * height
        for f in range(2, n - 2):
            if np.isnan([wx[f - 2], wx[f], wx[f + 2]]).any():
                continue
            v1 = np.array([wx[f] - wx[f - 2], wy[f] - wy[f - 2]])
            v2 = np.array([wx[f + 2] - wx[f], wy[f + 2] - wy[f]])
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 > 0 and n2 > 0:
                out[f] = min(out[f], float(np.dot(v1, v2) / (n1 * n2)))
    return out


def _argext(sig: np.ndarray, center: int, w: int, mode: str):
    lo, hi = max(0, center - w), min(len(sig), center + w + 1)
    seg = sig[lo:hi]
    if np.isnan(seg).all():
        return None
    idx = int(np.nanargmax(seg) if mode == "max" else np.nanargmin(seg))
    return lo + idx


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

    errs = defaultdict(list)
    by_type = defaultdict(lambda: defaultdict(list))
    signed = defaultdict(list)   # (wrist peak - GT) แยกตามท่า
    racket_cov = []

    for pkl in sorted((Path(args.cache_dir) / "tracking").rglob("*.pkl")):
        key = f"{pkl.parent.name}/{pkl.stem}"
        if key not in sessions:
            continue
        with open(pkl, "rb") as f:
            c = pickle.load(f)
        if c.get("cache_version") != 2:
            continue

        s = sessions[key]
        track = c["track"]
        vm = c["video_meta"]
        w_px, h_px = vm["width"], vm["height"]
        n = len(track.pose.landmarks)
        rb = c.get("racket_bboxes") or {}

        rc = racket_center_series(rb, n)
        rspeed = _speed(rc)
        rarea = racket_area_series(rb, n)
        raccel = np.full(n, np.nan)
        for i in range(1, n - 1):
            if not np.isnan([rspeed[i - 1], rspeed[i + 1]]).any():
                raccel[i] = rspeed[i + 1] - rspeed[i - 1]

        wspeed = {}
        for widx in (R_WRIST, L_WRIST):
            wspeed[widx] = _compute_wrist_speed(track.pose, widx, w_px, h_px)
        peaks = sorted(set(_find_wrist_peaks(wspeed[R_WRIST])
                           + _find_wrist_peaks(wspeed[L_WRIST])))

        for st in s.strokes:
            g = st.keyframes.get("impact")
            if g is None or not (W < g < n - W):
                continue
            # baseline = wrist peak ที่ใกล้ GT สุด (แยกปัญหา 'หาไม่เจอ' ออกไป)
            near = [p for p in peaks if abs(p - g) <= 12]
            if not near:
                continue
            base = min(near, key=lambda p: abs(p - g))
            errs["baseline (wrist peak)"].append(abs(base - g))
            by_type[st.stroke_type]["baseline (wrist peak)"].append(abs(base - g))
            signed[st.stroke_type].append(base - g)
            racket_cov.append(int(bool(rb.get(g))))

            wmax = np.fmax(wspeed[R_WRIST], wspeed[L_WRIST])
            wacc = np.full(n, np.nan)
            for i in range(1, n - 1):
                if not np.isnan([wmax[i - 1], wmax[i + 1]]).any():
                    wacc[i] = wmax[i + 1] - wmax[i - 1]
            cands = {
                "ไม้: เร็วสุด": _argext(rspeed, base, W, "max"),
                "ไม้: ชะลอแรงสุด": _argext(raccel, base, W, "min"),
                "ไม้: พื้นที่ bbox มากสุด": _argext(rarea, base, W, "max"),
                "ข้อมือ: เร็วสุดในหน้าต่าง": _argext(wmax, base, W, "max"),
                "ข้อมือ: ชะลอแรงสุด": _argext(wacc, base, W, "min"),
                "ข้อมือ: เร่งแรงสุด": _argext(wacc, base, W, "max"),
                "ข้อมือ: peak แบบ sub-frame": _subframe_peak(wmax, base, W),
                "ข้อมือ: หักเหมากสุด": _argext(
                    _dir_change(track.pose, w_px, h_px), base, W, "min"),
            }
            for name, fr in cands.items():
                if fr is not None:
                    errs[name].append(abs(fr - g))
                    by_type[st.stroke_type][name].append(abs(fr - g))

    print(f"racket bbox มีที่เฟรม impact จริง: {np.mean(racket_cov):.1%} "
          f"(n={len(racket_cov)})\n")
    print(f"{'สัญญาณ':30s} {'n':>5s} {'<=1 เฟรม':>10s} {'<=2':>7s} "
          f"{'median':>8s} {'mean':>7s}")
    print("-" * 74)
    for name, e in sorted(errs.items(), key=lambda kv: -np.mean(np.array(kv[1]) <= 1)):
        e = np.array(e)
        print(f"{name:30s} {len(e):5d} {np.mean(e <= 1):10.1%} "
              f"{np.mean(e <= 2):7.1%} {np.median(e):8.1f} {e.mean():7.2f}")

    print(f"\n{'แยกตามท่า (<=1 เฟรม)':30s}", end="")
    names = list(errs.keys())
    for nm in names:
        print(f" {nm[:14]:>15s}", end="")
    print()
    for t in sorted(by_type):
        print(f"{t:30s}", end="")
        for nm in names:
            e = by_type[t].get(nm, [])
            print(f" {np.mean(np.array(e) <= 1) if e else 0:15.1%}", end="")
        print(f"   (n={len(by_type[t].get('baseline (wrist peak)', []))})")

    # ─── มี bias เชิงระบบต่อท่ามั้ย? ถ้ามี แก้ด้วย offset คงที่ได้เลย ───
    print(f"\n{'ท่า':6s} {'n':>4s} {'mean':>7s} {'median':>7s} {'sd':>6s} "
          f"{'<=1 ตอนนี้':>11s} {'<=1 ถ้าเลื่อน':>13s} {'เลื่อนเท่าไหร่':>13s}")
    print("-" * 72)
    for t in sorted(signed):
        d = np.array(signed[t])
        best_sh, best_acc = 0, np.mean(np.abs(d) <= 1)
        for sh in range(-5, 6):
            acc = np.mean(np.abs(d - sh) <= 1)
            if acc > best_acc:
                best_sh, best_acc = sh, acc
        print(f"{t:6s} {len(d):4d} {d.mean():7.2f} {np.median(d):7.1f} "
              f"{d.std():6.2f} {np.mean(np.abs(d) <= 1):11.1%} "
              f"{best_acc:13.1%} {best_sh:13d}")
    all_d = np.array([v for vs in signed.values() for v in vs])
    print(f"{'รวม':6s} {len(all_d):4d} {all_d.mean():7.2f} "
          f"{np.median(all_d):7.1f} {all_d.std():6.2f} "
          f"{np.mean(np.abs(all_d) <= 1):11.1%}")
    print("\nค่าบวก = ระบบทายช้ากว่าเฉลย · sd สูง = กระจาย แก้ด้วย offset ไม่ได้")


if __name__ == "__main__":
    main()
