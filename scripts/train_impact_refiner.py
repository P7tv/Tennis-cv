"""ตัวปรับเฟรมปะทะ — ตอบคำถามที่ hit classifier ตอบไม่ได้

ปัญหา
─────
hit classifier ถูกเทรนให้ตอบว่า "หน้าต่างนี้เหมือนการตีแค่ไหน" (label คือ
|frame - เฉลย| <= 5) ซึ่ง**ไม่ใช่**คำถามว่า "เฟรมนี้คือจังหวะปะทะพอดีหรือเปล่า"
พอ NMS เลือกด้วย ml_prob เฟรม follow-through ที่ความเร็วสูงกว่าจึงชนะบ่อย

วัดจากของจริง (LOPO, 132 stroke ที่ตรวจเจอ):

                              |Δ| median   ภายใน 4 เฟรม
  เฟรมที่เลือกไปจริง              3.0         71.2%
  ตัวที่ดีสุดที่มีอยู่ใน candidate   2.0         88.6%

  37 เคสที่เลือกผิด: ตัวที่ควรเลือกมี prob ต่ำกว่า **100%** ของเคส
  30 ใน 37 ผ่านเกณฑ์แล้วแต่แพ้ NMS ให้เฟรมข้าง ๆ

ทำไมสำคัญ: backswing_peak = impact - ค่าคงที่ ความคลาดของ impact จึงส่งต่อไป
ทั้งดุ้น -> keyframe ทั้งสองตกพร้อมกัน และหน้าต่างที่ยอมรับได้กว้างแค่ +-4.25
เฟรม (median) ขณะที่ความคลาด p90 อยู่ที่ 7 เฟรม
แยกส่วนแล้ว: ก้อนนี้กิน 23.5% ของคะแนนที่หายไป พอ ๆ กับก้อน "หาลูกไม่เจอ" (23.3%)

วิธี
────
เทรน regressor ทำนาย "ต้องเลื่อนไปกี่เฟรมถึงจะตรงเฉลย" จากฟีเจอร์ชุดเดียวกับ
ที่ classifier ใช้ แล้วเอาไปปรับเฟรมหลัง NMS

ข้อดีเหนือการ "เลือกใหม่ในบรรดา candidate": เลื่อนไปเฟรมที่ไม่ได้เป็น candidate
ได้ด้วย -> ไม่ติดเพดาน 88.6%

⚠️ วัดแบบ Leave-One-Person-Out เท่านั้น และตัดแถว augment ออกจาก fold ทดสอบ

รัน:  python scripts/train_impact_refiner.py --n-aug 3 --save
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loeuf_cv.hit_detection import ML_PROB_THRESHOLD  # noqa: E402
from scripts.train_hit_classifier_from_cache import (FEATURE_SETS,  # noqa: E402
                                                     build_rows, nms)

# ช่วงที่ถือว่าเฟรมที่เลือก "ตรงกับ" การตีครั้งนั้น — ตรงกับ MATCH_TOLERANCE
# ของ benchmark harness
REFINE_BAND = 10

# เลื่อนได้มากสุดกี่เฟรม — กันโมเดลที่มั่นใจผิดลากเฟรมไปไกลจนแย่กว่าเดิม
# 8 เฟรมครอบ p90 ของความคลาดที่วัดได้ (7 เฟรม) พอดี
MAX_SHIFT = 8


def selected_rows(df, cols, threshold, n_estimators, max_depth):
    """จำลองว่า pipeline จริงจะ "เลือก" แถวไหน แล้วคืนเฉพาะแถวนั้น

    🔴 ทำไมต้องมีขั้นนี้ ไม่เทรนบน candidate ทุกตัวในรัศมี: สองประชากรนี้มี
    ความเอนเอียงคนละทิศกัน — วัดจากของจริงได้

        candidate ทุกตัวในรัศมี +-10   ค่ากลางของ (เฉลย - เฟรม) = +1  (มาเร็วไป)
        เฟรมที่ NMS เลือกจริง            ค่ากลางของ (เฉลย - เฟรม) = -0.5 (ช้าไป)

    เทรนบนอันแรกแล้วเอาไปใช้กับอันหลัง = โมเดลเลื่อนผิดทาง เป็น train/serve
    skew ชนิดเดียวกับที่โปรเจกต์นี้โดนมาแล้ว 3 รอบ

    ใช้ out-of-fold probability (โมเดลไม่เคยเห็นคนในคลิปนั้น) จำลอง threshold
    + NMS -> ได้ชุดเฟรมเดียวกับที่ระบบจริงจะรายงานออกไป
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import LeaveOneGroupOut

    X, y = df[cols].fillna(0), df["is_hit"]
    is_aug = df["is_aug"].to_numpy()
    oof = np.full(len(df), np.nan)
    for tr, te in LeaveOneGroupOut().split(X, y, df["person"]):
        te_real = te[is_aug[te] == 0]
        clf = RandomForestClassifier(n_estimators=n_estimators,
                                     max_depth=max_depth, random_state=42,
                                     class_weight="balanced", n_jobs=-1)
        clf.fit(X.iloc[tr], y.iloc[tr])
        if len(te_real):
            oof[te_real] = clf.predict_proba(X.iloc[te_real])[:, 1]

    df = df.copy()
    df["oof"] = oof
    real = df[(df["is_aug"] == 0) & np.isfinite(df["oof"])]

    picked_idx = []
    for clip, g in real.groupby("clip"):
        gap = int(g["fps"].iloc[0] * 0.75)
        keep = g[g["oof"] >= threshold]
        if keep.empty:
            continue
        chosen = nms(list(zip(keep["frame"], keep["oof"])), gap)
        by_frame = {f: i for i, f in zip(keep.index, keep["frame"])}
        picked_idx += [by_frame[f] for f in chosen if f in by_frame]

    sel = df.loc[picked_idx]
    # เก็บเฉพาะเฟรมที่จับคู่กับเฉลยได้ — ตัวที่ไม่มีเฉลยใกล้ ๆ คือ false alarm
    # ซึ่งไม่มี "เฟรมที่ถูก" ให้เลื่อนไปหา
    return sel[sel["gt_offset"].abs() <= REFINE_BAND].reset_index(drop=True)


def evaluate(df, cols, n_estimators, max_depth, min_leaf):
    """LOPO — คืน (|Δ| ก่อน, |Δ| หลัง) ของแถวจริงในช่วง REFINE_BAND"""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import LeaveOneGroupOut

    X, y = df[cols].fillna(0), df["gt_offset"]
    is_aug = df["is_aug"].to_numpy()
    pred = np.full(len(df), np.nan)
    for tr, te in LeaveOneGroupOut().split(X, y, df["person"]):
        te_real = te[is_aug[te] == 0]
        m = RandomForestRegressor(n_estimators=n_estimators,
                                  max_depth=max_depth,
                                  min_samples_leaf=min_leaf,
                                  random_state=42, n_jobs=-1)
        m.fit(X.iloc[tr], y.iloc[tr])
        if len(te_real):
            pred[te_real] = m.predict(X.iloc[te_real])
    keep = (is_aug == 0) & np.isfinite(pred)
    # 🔴 คืนค่า **มีเครื่องหมาย** เสมอ — ผู้เรียกต้องทำ abs เอง
    # เคยคืน np.abs(before) แล้วผู้เรียกเอาไปคำนวณตัวเทียบเป็น | |offset| - k |
    # ซึ่งวัด "ขนาดความคลาดใกล้ k แค่ไหน" ไม่ใช่ "เลื่อนไป k แล้วตรงขึ้นไหม"
    # พอ |offset| ส่วนใหญ่ราว 3 ค่าคงที่ +3 จึงดูเหมือนได้ 91.7% ทั้งที่เป็นเลขปลอม
    before = df["gt_offset"].to_numpy()[keep]
    shift = np.clip(np.round(pred[keep]), -MAX_SHIFT, MAX_SHIFT)
    return before, before - shift, shift


def report(name, before_signed, after_signed, shift):
    before, after = np.abs(before_signed), np.abs(after_signed)

    def line(tag, a):
        return (f"  {tag:<12s} median {np.median(a):4.1f}  p75 {np.percentile(a,75):4.1f}"
                f"  p90 {np.percentile(a,90):5.1f}  ภายใน 4 เฟรม {np.mean(a<=4):6.1%}"
                f"  ภายใน 2 เฟรม {np.mean(a<=2):6.1%}")
    print(f"\n{name} (n={len(before)})")
    print(line("ก่อนปรับ", before))
    print(line("หลังปรับ", after))
    d = np.mean(after <= 4) - np.mean(before <= 4)
    worse = np.mean(after > before)
    print(f"  -> ภายใน 4 เฟรม {d:+.1%} จุด · แย่ลง {worse:.1%} ของแถว · "
          f"เลื่อนจริง median {np.median(np.abs(shift)):.1f} เฟรม "
          f"(ไม่เลื่อนเลย {np.mean(shift==0):.0%})")
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=str(ROOT / "dataset/benchmark_cache"))
    ap.add_argument("--dataset-root", default=str(ROOT / "dataset"))
    ap.add_argument("--out", default=str(ROOT / "impact_refiner.pkl"))
    ap.add_argument("--n-aug", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--features", choices=tuple(FEATURE_SETS),
                    default="swing+speed")
    ap.add_argument("--n-estimators", type=int, default=300)
    ap.add_argument("--max-depth", type=int, default=8)
    ap.add_argument("--min-leaf", type=int, default=5)
    ap.add_argument("--threshold", type=float, default=ML_PROB_THRESHOLD,
                    help="เกณฑ์ที่ใช้จำลองการเลือกเฟรม (ต้องตรงกับ production)")
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--lopo-out", default=None,
                    help="โฟลเดอร์เก็บ refiner ชุด LOPO (<person>.pkl) สำหรับ "
                         "benchmark ที่ไม่ปนข้อมูลที่เคยเห็น")
    args = ap.parse_args()

    print(f"=== สร้าง candidate (augment {args.n_aug}) ===")
    df = build_rows(Path(args.cache_dir), Path(args.dataset_root),
                    n_aug=args.n_aug, seed=args.seed)
    if df.empty:
        print("ไม่มีข้อมูล")
        return

    cols = FEATURE_SETS[args.features]
    print("\n=== จำลองว่า pipeline จริงเลือกเฟรมไหน (out-of-fold) ===")
    band = selected_rows(df, cols, args.threshold, args.n_estimators,
                         args.max_depth)
    print(f"เฟรมที่ถูกเลือกและจับคู่กับเฉลยได้: {len(band)} · "
          f"{band['person'].nunique()} คน")
    print(f"ระยะจากเฉลยก่อนปรับ: median "
          f"{band['gt_offset'].abs().median():.1f} เฟรม · "
          f"ค่ากลางแบบมีเครื่องหมาย {band['gt_offset'].median():+.1f}")

    before, after, shift = evaluate(band, cols, args.n_estimators,
                                    args.max_depth, args.min_leaf)
    gain = report(f"ชุดฟีเจอร์ {args.features} ({len(cols)} ตัว)",
                  before, after, shift)

    # ตัวเทียบขั้นต่ำ: เลื่อนทุกเฟรมด้วยค่าคงที่เดียวกัน
    # ถ้าโมเดลชนะตัวนี้ไม่ได้ แปลว่ามันไม่ได้เรียนอะไรที่ขึ้นกับแต่ละเคสเลย
    # แค่จำค่าเฉลี่ยของทั้งชุด ซึ่งเขียนเป็นค่าคงที่บรรทัดเดียวก็ได้ผลเท่ากัน
    off = band["gt_offset"].to_numpy()
    print("\nการกระจายของ (เฉลย - เฟรมที่เลือก):")
    hist = np.bincount(np.clip(off, -REFINE_BAND, REFINE_BAND).astype(int)
                       + REFINE_BAND, minlength=2 * REFINE_BAND + 1)
    for v, n in zip(range(-REFINE_BAND, REFINE_BAND + 1), hist):
        if n:
            print(f"  {v:+3d} เฟรม {'█' * n} {n}")
    print(f"  mean {off.mean():+.2f} · median {np.median(off):+.1f}")

    print("\nตัวเทียบขั้นต่ำ — เลื่อนทุกเฟรมเท่ากัน (เลือกค่าจากชุดทดสอบเอง = โกง):")
    base = float(np.mean(np.abs(before) <= 4))
    for k in range(-3, 4):
        acc = float(np.mean(np.abs(before - k) <= 4))
        print(f"  {k:+d} เฟรม: ภายใน 4 เฟรม {acc:6.1%} ({acc - base:+.1%} จุด)")

    # ค่าคงที่แบบยุติธรรม: เลือกค่าจาก fold ฝึก แล้ววัดบนคนที่ไม่เคยเห็น
    # เหมือนกับที่โมเดลถูกวัด — นี่คือตัวเทียบที่ถูกต้อง
    fair = np.zeros(len(band), dtype=bool)
    picks = []
    for person in band["person"].unique():
        te = (band["person"] == person).to_numpy()
        tr_off = off[~te]
        k = max(range(-3, 4), key=lambda k: np.mean(np.abs(tr_off - k) <= 4))
        picks.append((person, k))
        fair[te] = np.abs(off[te] - k) <= 4
    fair_acc = float(fair.mean())
    print(f"\nค่าคงที่แบบ LOPO (เลือกค่าจากคนอื่น วัดบนคนที่กันไว้): "
          f"{fair_acc:.1%}")
    print("  ค่าที่แต่ละ fold เลือก: " +
          " ".join(f"{p}:{k:+d}" for p, k in picks))

    edge = np.mean(np.abs(after) <= 4) - fair_acc
    print(f"\nโมเดลชนะค่าคงที่แบบยุติธรรม {edge:+.1%} จุด")
    if gain <= 0.02 or edge <= 0.02:
        print("⚠️ ยังไม่ควรเอาไปใช้ — ดีขึ้นไม่พอ หรือชนะค่าคงที่ไม่ขาด")
    def fit(rows):
        from sklearn.ensemble import RandomForestRegressor
        m = RandomForestRegressor(n_estimators=args.n_estimators,
                                  max_depth=args.max_depth,
                                  min_samples_leaf=args.min_leaf,
                                  random_state=42, n_jobs=-1)
        m.fit(rows[cols].fillna(0), rows["gt_offset"])
        return m

    if args.save:
        with open(args.out, "wb") as f:
            pickle.dump((fit(band), cols, MAX_SHIFT), f)
        print(f"\nบันทึก -> {args.out}")
    else:
        print("\n(ไม่ได้บันทึก — ใส่ --save ถ้าต้องการ)")

    # ชุด LOPO สำหรับ benchmark — ถ้าใช้ตัวเดียวกับที่เทรนจากทุกคน ตัวเลข
    # benchmark จะปนเปื้อนแบบเดียวกับที่เคยเจอกับ hit_classifier (0.506 vs 0.308)
    if args.lopo_out:
        out = Path(args.lopo_out)
        out.mkdir(parents=True, exist_ok=True)
        for person in sorted(band["person"].unique()):
            sub = band[band["person"] != person]
            with open(out / f"{person}.pkl", "wb") as f:
                pickle.dump((fit(sub), cols, MAX_SHIFT), f)
            print(f"  refiner ที่ไม่เคยเห็น {person}: {len(sub)} แถว")
        print(f"บันทึกชุด LOPO -> {out}")


if __name__ == "__main__":
    main()
