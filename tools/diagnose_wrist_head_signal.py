"""ข้อมือเทียบหัว เป็นสัญญาณหาจังหวะปะทะที่ดีกว่าความเร็วข้อมือหรือเปล่า

ที่มา: EDA ของหน้าต่าง stroke ที่ align กับเฉลย (scratchpad/stroke_windows_*)
แสดงว่า

  แถว wrist_speed          ค่าเฉลี่ยแทบไม่มียอดที่เฟรม 0 · spike กระจายทั่ว
                           แถบ ±1 SD ติดลบทั้งที่ speed ติดลบไม่ได้
                           = การแจกแจงเบ้ขวาจัด ยอดมาจาก outlier ไม่กี่ตัว
  แถว wrist_minus_head_y   จุดหักคมตรงเฟรม 0 · แถบแคบ · สม่ำเสมอทั้ง 5 ท่า

ข้อสงสัย: ตัวสร้าง candidate ทั้งระบบยืนอยู่บนยอดของ wrist_speed ซึ่งเป็น
อนุพันธ์ของตำแหน่งที่มี jitter -> abs() ทำให้ noise ไม่หักล้างกัน มีแต่บวกสะสม
ส่วน wrist_minus_head_y เป็น "ตำแหน่ง" ไม่ใช่อนุพันธ์ จึงไม่ขยาย jitter

🔴 สิ่งที่ EDA บอกไม่ได้: รูปนี้ครอบแค่ ±45 เฟรมรอบ impact จริง จึงบอกได้แค่ว่า
"ตรงจุดกระทบมีรูปทรงเฉพาะ" แต่บอกไม่ได้ว่ารูปทรงนั้นไปโผล่ที่อื่นในคลิปอีกกี่ครั้ง
ซึ่งเป็นคำถามที่ตัดสินว่าใช้เป็นตัวตรวจจับได้จริงไหม -> สคริปต์นี้วัดตรงนั้น

วัด 3 อย่าง เทียบกันตรง ๆ บน GT ชุดเดียวกัน:

  1. ครอบคลุม   GT กี่ % ที่มี landmark อยู่ใน ±tol
  2. ภาระ       landmark ทั้งหมดกี่ตัวต่อนาที (ยิ่งเยอะ ปลายทางยิ่งต้องคัดหนัก)
  3. ความคม     ระยะจาก landmark ที่ใกล้สุดถึงเฉลย

รัน:  python scripts/diagnose_wrist_head_signal.py
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

from loeuf_cv.config import L_WRIST, R_WRIST  # noqa: E402
from loeuf_cv.hit_detection import (PEAK_MERGE_WINDOW,  # noqa: E402
                                    PEAK_WINDOW, _compute_wrist_speed,
                                    _find_wrist_peaks)
from tools.label_ingest import (find_session_labels,  # noqa: E402
                                      load_session_label)

NOSE = 0
L_SHOULDER, R_SHOULDER = 11, 12
L_HIP, R_HIP = 23, 24


def wrist_minus_head(lm: np.ndarray, wrist_idx: int) -> np.ndarray:
    """(ข้อมือ − หัว) ตามแกน y หารด้วยความยาวลำตัว

    แกน y ในพิกัดภาพชี้ลง -> ค่าบวก = ข้อมืออยู่ต่ำกว่าหัว (ท่าปกติ ~1.2)
    ค่าลดลง = ข้อมือยกขึ้นเข้าหาหัว · ติดลบ = ข้อมือเหนือหัว (เสิร์ฟ)
    หารด้วยความยาวลำตัวเพื่อตัดผลของระยะกล้องและขนาดตัวคนออก
    """
    torso = np.abs((lm[:, [L_SHOULDER, R_SHOULDER], 1].mean(axis=1)
                    - lm[:, [L_HIP, R_HIP], 1].mean(axis=1)))
    torso = np.where(torso > 1e-6, torso, np.nan)
    return (lm[:, wrist_idx, 1] - lm[:, NOSE, 1]) / torso


def local_minima(sig: np.ndarray, window: int, merge: int,
                 prominence: float = 0.0) -> list[int]:
    """จุดต่ำสุดเฉพาะที่ — กติกาเดียวกับ _find_wrist_peaks เป๊ะ (กลับหัว)

    ใช้กติกาเดียวกันเพื่อให้เทียบกันได้จริง ถ้าใช้คนละกติกาแล้วตัวหนึ่งชนะ
    จะแยกไม่ออกว่าชนะเพราะสัญญาณดีกว่า หรือเพราะกติกาหลวมกว่า
    """
    n = len(sig)
    out = []
    for i in range(window, n - window):
        if np.isnan(sig[i]):
            continue
        seg = sig[max(0, i - window): i + window + 1]
        if np.all(np.isnan(seg)):
            continue
        if sig[i] != np.nanmin(seg):
            continue
        # ความลึกของหลุมเทียบกับขอบหน้าต่าง — กันจุดต่ำสุดที่เกิดจากพื้นราบ
        if prominence > 0 and (np.nanmax(seg) - sig[i]) < prominence:
            continue
        out.append(i)
    if not out or merge <= 1:
        return out
    merged = [out[0]]
    for p in out[1:]:
        if p - merged[-1] < merge:
            if sig[p] < sig[merged[-1]]:
                merged[-1] = p
        else:
            merged.append(p)
    return merged


def coverage(frames: list[int], gt: list[int], tol: int):
    if not frames:
        return 0, [abs(g) * 0 + 999 for g in gt]
    fs = np.asarray(sorted(frames))
    d = [int(np.min(np.abs(fs - g))) for g in gt]
    return sum(1 for x in d if x <= tol), d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=str(ROOT / "dataset/benchmark_cache"))
    ap.add_argument("--dataset-root", default=str(ROOT / "dataset"))
    ap.add_argument("--tol", type=int, default=4,
                    help="±เฟรม ที่ถือว่า landmark ตรงกับเฉลย")
    ap.add_argument("--prominence", type=float, default=0.0)
    args = ap.parse_args()

    rows = []
    for p in sorted(find_session_labels(Path(args.dataset_root))):
        try:
            s = load_session_label(p)
        except Exception:
            continue
        stem = Path(s.video_path).stem
        cand = list(Path(args.cache_dir).rglob(f"tracking/**/{stem}.pkl"))
        if not cand:
            continue
        c = pickle.load(open(cand[0], "rb"))
        track = c["track"]
        track = track[0] if isinstance(track, list) else track
        lm = track.pose.landmarks
        fps = c.get("fps") or 29.97
        vm = c["video_meta"]
        side = c.get("dominant_side", "right")
        wrist_idx = R_WRIST if side == "right" else L_WRIST

        gt = sorted(st.keyframes["impact"] for st in s.strokes
                    if st.keyframes.get("impact") is not None)
        gt = [g for g in gt if g < len(lm)]
        if not gt:
            continue

        speed = _compute_wrist_speed(track.pose, wrist_idx, vm["width"],
                                     vm["height"])
        sp = _find_wrist_peaks(speed, min_speed_px=8.0, window=PEAK_WINDOW,
                               merge_window=PEAK_MERGE_WINDOW)
        wh = wrist_minus_head(lm, wrist_idx)
        wp = local_minima(wh, PEAK_WINDOW, PEAK_MERGE_WINDOW, args.prominence)

        n_sp, d_sp = coverage(sp, gt, args.tol)
        n_wp, d_wp = coverage(wp, gt, args.tol)
        n_both, d_both = coverage(sorted(set(sp) | set(wp)), gt, args.tol)
        minutes = len(lm) / fps / 60.0
        rows.append(dict(clip=stem, n_gt=len(gt), minutes=minutes, gt=gt,
                         n_sp=n_sp, n_wp=n_wp, n_both=n_both,
                         c_sp=len(sp), c_wp=len(wp),
                         d_sp=d_sp, d_wp=d_wp, d_both=d_both,
                         sp_frames=sp, wp_frames=wp,
                         wp_values=[float(wh[i]) for i in wp]))
        print(f"  {stem:24s} GT {len(gt):3d} · ความเร็ว {n_sp:3d} "
              f"({len(sp):4d} จุด) · ข้อมือ-หัว {n_wp:3d} ({len(wp):4d} จุด) "
              f"· รวม {n_both:3d}")

    if not rows:
        print("ไม่มีข้อมูล")
        return

    tot_gt = sum(r["n_gt"] for r in rows)
    tot_min = sum(r["minutes"] for r in rows)
    print(f"\n=== รวม {tot_gt} ลูก · {len(rows)} คลิป · {tot_min:.1f} นาที ===")
    print(f"เกณฑ์: มี landmark อยู่ภายใน ±{args.tol} เฟรมจากเฉลย\n")

    print(f"{'สัญญาณ':<22s}{'ครอบคลุม':>12s}{'จุดทั้งหมด':>12s}"
          f"{'จุด/นาที':>11s}{'|Δ| median':>12s}")
    print("-" * 70)
    for name, kn, kc, kd in (("ความเร็วข้อมือ (ของเดิม)", "n_sp", "c_sp", "d_sp"),
                             ("ข้อมือ − หัว", "n_wp", "c_wp", "d_wp"),
                             ("ใช้ทั้งสองอย่าง", "n_both", None, "d_both")):
        hit = sum(r[kn] for r in rows)
        cnt = sum(r[kc] for r in rows) if kc else \
            sum(r["c_sp"] + r["c_wp"] for r in rows)
        d = np.concatenate([r[kd] for r in rows])
        print(f"{name:<22s}{hit/tot_gt:11.1%}{cnt:12d}{cnt/tot_min:11.1f}"
              f"{np.median(d):12.1f}")

    d_sp = np.concatenate([r["d_sp"] for r in rows])
    d_wp = np.concatenate([r["d_wp"] for r in rows])
    only_wp = int(np.sum((d_sp > args.tol) & (d_wp <= args.tol)))
    only_sp = int(np.sum((d_sp <= args.tol) & (d_wp > args.tol)))
    print(f"\nลูกที่ **เฉพาะ** ข้อมือ−หัว จับได้ (ความเร็วพลาด): {only_wp}")
    print(f"ลูกที่ **เฉพาะ** ความเร็ว จับได้ (ข้อมือ−หัวพลาด):  {only_sp}")
    print("\n⚠️ 'ครอบคลุม' สูงอย่างเดียวยังสรุปไม่ได้ — ถ้าจุด/นาทีสูงตามไปด้วย"
          "\n   แปลว่าได้มาจากการหว่านจุดเยอะขึ้น ปลายทางก็ต้องคัดหนักขึ้นตาม")

    # ── กรองเฉพาะจุดที่ข้อมืออยู่สูง = ท่าเหนือศีรษะ ──
    # เหตุผล: ตารางรายคลิปบอกว่าข้อมือ−หัวชนะเฉพาะคลิปเสิร์ฟ (IMG_0294 ความเร็ว
    # ได้ 4/10 แต่ข้อมือ−หัวได้ 10/10) ส่วนคลิป groundstroke แพ้ขาด
    # ถ้าจำกัดให้เสนอเฉพาะจังหวะที่ข้อมือยกสูงจริง จะได้ของที่ขาดโดยไม่ต้อง
    # หว่านจุดทั้งคลิป — สวีปหาว่าคุ้มที่ค่าไหน
    print(f"\n=== ถ้าให้ข้อมือ−หัวเสนอเฉพาะจังหวะที่ข้อมืออยู่สูง ===")
    print(f"{'เกณฑ์ (ข้อมือ−หัว <)':<22s}{'ครอบคลุมรวม':>13s}{'จุดที่เพิ่ม':>12s}"
          f"{'จุด/นาที':>11s}{'ได้เพิ่ม':>10s}")
    print("-" * 70)
    base_hit = sum(r["n_sp"] for r in rows)
    base_cnt = sum(r["c_sp"] for r in rows)
    for thr in (-0.5, 0.0, 0.25, 0.5, 0.75, 1.0, None):
        hit = extra = 0
        for r in rows:
            keep = [f for f, v in zip(r["wp_frames"], r["wp_values"])
                    if thr is None or v < thr]
            extra += len([f for f in keep if f not in set(r["sp_frames"])])
            hit += coverage(sorted(set(r["sp_frames"]) | set(keep)),
                            r["gt"], args.tol)[0]
        name = "ไม่จำกัด (ทุกจุด)" if thr is None else f"{thr:+.2f}"
        print(f"{name:<22s}{hit/tot_gt:12.1%}{extra:12d}"
              f"{(base_cnt+extra)/tot_min:11.1f}{hit-base_hit:+10d}")
    print(f"\nฐาน (ความเร็วอย่างเดียว): ครอบคลุม {base_hit/tot_gt:.1%} · "
          f"{base_cnt/tot_min:.1f} จุด/นาที")


if __name__ == "__main__":
    main()
