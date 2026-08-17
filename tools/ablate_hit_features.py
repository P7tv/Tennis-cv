"""Ablation: ชุดฟีเจอร์ไหน x augmentation เท่าไร ให้ hit detection ดีที่สุด

ทำไมต้องมีแยกจาก train_hit_classifier_from_cache.py: ตัวนั้นเทรน 1 ชุดต่อการรัน
1 ครั้ง ซึ่งต้องสร้าง candidate ใหม่ทุกรอบ (ส่วนที่แพงสุด) การเทียบ 7 ชุด x 2
โหมด augment จึงเสียเวลาสร้างแถวซ้ำ 14 รอบโดยไม่จำเป็น

ตัวนี้สร้างแถว **รอบเดียว** (เก็บทุกฟีเจอร์เสมอ ดู ALL_FEATURE_COLS) แล้ววน
เลือกคอลัมน์เทียบ -> ผลที่ได้เทียบกันได้จริงเพราะมาจาก candidate ชุดเดียวกันเป๊ะ

⚠️ Leave-One-Person-Out เท่านั้น และแถว augment ของ "คนที่กำลังทดสอบ" ถูกตัด
   ออกจาก fold ทดสอบเสมอ — ไม่งั้นกลายเป็นวัดผลบนข้อมูลสังเคราะห์

รัน:  python scripts/ablate_hit_features.py --n-aug 3
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_hit_classifier_from_cache import (FEATURE_SETS,  # noqa: E402
                                                     build_rows,
                                                     eval_thresholds)
from tools.label_ingest import (find_session_labels,  # noqa: E402
                                      load_session_label)

THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]


def lopo(df, cols, use_aug, n_estimators, max_depth):
    """out-of-fold prob — คืน array ยาวเท่า df (แถว augment เป็น NaN)"""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import LeaveOneGroupOut

    X, y = df[cols].fillna(0), df["is_hit"]
    is_aug = df["is_aug"].to_numpy()
    oof = np.full(len(df), np.nan)
    for tr, te in LeaveOneGroupOut().split(X, y, df["person"]):
        if not use_aug:
            tr = tr[is_aug[tr] == 0]
        te_real = te[is_aug[te] == 0]
        clf = RandomForestClassifier(n_estimators=n_estimators,
                                     max_depth=max_depth, random_state=42,
                                     class_weight="balanced", n_jobs=-1)
        clf.fit(X.iloc[tr], y.iloc[tr])
        if len(te_real):
            oof[te_real] = clf.predict_proba(X.iloc[te_real])[:, 1]
    return oof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=str(ROOT / "dataset/benchmark_cache"))
    ap.add_argument("--dataset-root", default=str(ROOT / "dataset"))
    ap.add_argument("--n-aug", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-estimators", type=int, default=300)
    ap.add_argument("--max-depth", type=int, default=6)
    args = ap.parse_args()

    print(f"=== สร้าง candidate (augment {args.n_aug} สำเนา/คลิป) ===")
    df = build_rows(Path(args.cache_dir), Path(args.dataset_root),
                    n_aug=args.n_aug, seed=args.seed)
    if df.empty:
        print("ไม่มีข้อมูล — cache v2 ยังไม่พร้อม?")
        return

    gts = {}
    for p in find_session_labels(Path(args.dataset_root)):
        try:
            s = load_session_label(p)
        except Exception:
            continue
        gts[f"{p.parent.name}/{Path(s.video_path).stem}"] = sorted(
            st.keyframes["impact"] for st in s.strokes
            if st.keyframes.get("impact") is not None)

    real = df["is_aug"] == 0
    print(f"\nแถว {len(df)} (ของจริง {int(real.sum())}) · "
          f"{df['clip'].nunique()} คลิป · {df['person'].nunique()} คน")

    print("\n=== LOPO ระดับ stroke — วัดบนแถวของจริงเท่านั้น ===")
    print(f"{'ชุดฟีเจอร์':<14s} {'n':>3s} {'augment':>8s} {'th':>5s} "
          f"{'TP':>4s} {'FN':>4s} {'FP':>4s} {'recall':>8s} "
          f"{'precision':>10s} {'F1':>7s}")
    print("-" * 76)

    rows, baseline, scored = [], None, {}
    for name, cols in FEATURE_SETS.items():
        for use_aug in (False, True):
            if use_aug and args.n_aug == 0:
                continue
            d = df.copy()
            d["p"] = lopo(d, cols, use_aug, args.n_estimators, args.max_depth)
            d = d[d["is_aug"] == 0].reset_index(drop=True)
            best = max(eval_thresholds(d, "p", gts, THRESHOLDS),
                       key=lambda r: r[6])
            if name == "all" and not use_aug:
                baseline = best[6]          # = production ปัจจุบัน
            tag = f"+{args.n_aug}" if use_aug else "ไม่ใช้"
            key = (name, tag)
            rows.append((name, len(cols), tag, *best))
            scored[key] = (d, best[0])
            delta = "" if baseline is None else f"  ({best[6] - baseline:+.3f})"
            print(f"{name:<14s} {len(cols):3d} {tag:>8s} {best[0]:5.2f} "
                  f"{best[1]:4d} {best[2]:4d} {best[3]:4d} {best[4]:8.1%} "
                  f"{best[5]:10.1%} {best[6]:7.3f}{delta}")

    top = max(rows, key=lambda r: r[-1])
    print(f"\nดีสุด: {top[0]} ({top[1]} ฟีเจอร์ · augment {top[2]}) — "
          f"F1 {top[-1]:.3f} · recall {top[-3]:.1%} · precision {top[-2]:.1%}")
    print("(baseline = 'all' + ไม่ใช้ augment = ชุดที่ production ใช้อยู่)")

    # ─── รายคน: ชุดที่ชนะ vs ชุดที่ production ใช้อยู่ ───
    # ตัวเลขรวมที่ดีขึ้นเพราะคนเดียวคือ overfit ที่ LOPO ก็ยังมองไม่เห็น
    # เพราะสุดท้าย LOPO ก็เอาทุก fold มารวมกันอยู่ดี
    print(f"\n=== รายคน: {top[0]}+{top[2]} เทียบ production (all/ไม่ใช้) ===")
    d_top, th_top = scored[(top[0], top[2])]
    d_base, th_base = scored[("all", "ไม่ใช้")]
    print(f"{'คน':<9s} {'GT':>4s} {'production':>22s} {'ชุดใหม่':>22s}")
    print("-" * 60)
    wins = 0
    for person in sorted(d_top["person"].unique()):
        cells = []
        for d, th in ((d_base, th_base), (d_top, th_top)):
            g = d[d["person"] == person]
            sub = {c: gts[c] for c in g["clip"].unique()}
            _, tp, fn, fp, _, _, f1 = eval_thresholds(g, "p", sub, [th])[0]
            cells.append((f1, tp, fn, fp))
        wins += cells[1][0] > cells[0][0]
        n_gt = cells[0][1] + cells[0][2]
        print(f"{person:<9s} {n_gt:4d} " + " ".join(
            f"{f1:8.3f} {tp:3d}/{n_gt:<3d} FP{fp:<4d}"
            for f1, tp, _, fp in cells))
    print(f"\nชุดใหม่ชนะ {wins}/{d_top['person'].nunique()} คน")


if __name__ == "__main__":
    main()
