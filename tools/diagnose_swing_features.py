"""Diagnostic: ฟีเจอร์ "รูปทรงวงสวิง" ช่วย hit detection จริงไหม — ก่อนลงแรงสร้าง

คำถามที่ตอบ (3 ข้อ แยกกันชัด ๆ):

  Q1  ฟีเจอร์แต่ละตัวแยก hit ออกจาก non-hit ได้ไหม   -> AUC ต่อฟีเจอร์
  Q2  ใส่เข้าโมเดลแล้วดีขึ้นจริงไหม                  -> LOPO base vs base+swing
  Q3  ช่วย 8 ลูกที่ไม่เกิด candidate เลยได้ไหม        -> เทียบค่าที่เฟรม GT

เหตุผลที่ต้องแยก Q1 กับ Q2: AUC สูงไม่ได้แปลว่าโมเดลจะดีขึ้น ถ้าข้อมูลที่มัน
ให้ซ้ำกับที่ฟีเจอร์เดิมให้อยู่แล้ว ส่วน AUC ต่ำก็ไม่ได้แปลว่าไร้ค่า ถ้ามันช่วย
เฉพาะเคสที่ฟีเจอร์เดิมพลาด -> ตัวตัดสินจริงคือ Q2

⚠️ ทุกตัวเลขวัดแบบ Leave-One-Person-Out เท่านั้น — วัดแบบ in-sample เคยทำให้
   รายงานตัวเลขสูงเกินจริงมาแล้ว (recall 0.866 ที่จริงคือ 0.547)

รัน:  python scripts/diagnose_swing_features.py
"""
import argparse
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loeuf_cv.hit_detection import (detect_hit_events,  # noqa: E402
                                    extract_ball_trajectory_kalman)
from loeuf_cv.swing_shape import (SWING_FEATURE_COLS,  # noqa: E402
                                  swing_shape_features)
from scripts.train_hit_classifier_from_cache import (  # noqa: E402
    FEATURE_COLS, LABEL_TOL, MATCH_TOL, eval_thresholds)
from tools.label_ingest import (find_session_labels,  # noqa: E402
                                      load_session_label)


def build_rows(cache_dir: Path, dataset_root: Path):
    """คืน (df ของ candidate, gts ต่อคลิป, missed = GT ที่ไม่มี candidate ใกล้)

    โครงเดียวกับ train_hit_classifier_from_cache.build_rows แต่เพิ่มฟีเจอร์
    วงสวิง และเก็บ GT ที่ไม่มี candidate ไว้ตอบ Q3 (ตัวนั้นไม่ทำ)
    """
    sessions = {}
    for p in find_session_labels(dataset_root):
        try:
            s = load_session_label(p)
        except Exception:
            continue
        sessions[f"{p.parent.name}/{Path(s.video_path).stem}"] = s

    rows, gts, missed = [], {}, []
    for pkl in sorted((cache_dir / "tracking").rglob("*.pkl")):
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
        gts[key] = gt
        pids = Counter(st.player_id for st in s.strokes if st.player_id)
        person = pids.most_common(1)[0][0] if pids else key

        track = c["track"]
        vm = c["video_meta"]
        w, h, fps = vm["width"], vm["height"], c["fps"]
        lm = track.pose.landmarks
        traj, meas = extract_ball_trajectory_kalman(
            c["ball_bboxes"], len(lm), return_measured=True, fps=fps)
        cands = detect_hit_events(traj, [track], fps, w, h,
                                  racket_bboxes=c.get("racket_bboxes"),
                                  ball_measured=meas, return_candidates=True)

        for ev in cands:
            fe = ev.get("features", {})
            sw = swing_shape_features(lm, ev["frame"], fps=fps, width=w,
                                      height=h, side=ev.get("wrist_side"))
            rows.append({
                **{col: fe.get(col, 0.0) for col in FEATURE_COLS},
                **sw,
                "is_hit": int(any(abs(ev["frame"] - g) <= LABEL_TOL
                                  for g in gt)),
                "frame": ev["frame"], "clip": key, "person": person,
                "fps": fps, "is_aug": 0,
            })

        # Q3: GT ที่ไม่มี candidate ใด ๆ อยู่ใกล้ = พลาดตั้งแต่ด่านสร้าง candidate
        cf = [ev["frame"] for ev in cands]
        for g in gt:
            if not any(abs(g - f) <= MATCH_TOL for f in cf):
                missed.append({
                    **swing_shape_features(lm, g, fps=fps, width=w, height=h),
                    "frame": g, "clip": key, "person": person,
                })
        print(f"  {key:30s} candidate {len(cands):4d} · GT {len(gt):3d} · "
              f"ไม่มี candidate {sum(1 for m in missed if m['clip'] == key):2d}")
    return pd.DataFrame(rows), gts, pd.DataFrame(missed)


def auc(y: np.ndarray, x: np.ndarray) -> float:
    """AUC แบบ rank (Mann-Whitney) — ทน NaN, ไม่ต้องพึ่ง sklearn"""
    m = np.isfinite(x)
    y, x = y[m], x[m]
    n1, n0 = int(y.sum()), int((1 - y).sum())
    if n1 == 0 or n0 == 0:
        return 0.5
    r = pd.Series(x).rank().to_numpy()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def lopo_probs(df, cols, n_estimators, max_depth):
    """out-of-fold probability แบบ Leave-One-Person-Out"""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import LeaveOneGroupOut

    X, y = df[cols].fillna(0), df["is_hit"]
    oof = np.full(len(df), np.nan)
    for tr, te in LeaveOneGroupOut().split(X, y, df["person"]):
        clf = RandomForestClassifier(n_estimators=n_estimators,
                                     max_depth=max_depth, random_state=42,
                                     class_weight="balanced", n_jobs=-1)
        clf.fit(X.iloc[tr], y.iloc[tr])
        oof[te] = clf.predict_proba(X.iloc[te])[:, 1]
    return oof


def report(df, gts, label, cols, args):
    df = df.copy()
    df[f"p_{label}"] = lopo_probs(df, cols, args.n_estimators, args.max_depth)
    rs = eval_thresholds(df, f"p_{label}", gts, [0.3, 0.4, 0.5, 0.6, 0.7])
    best = max(rs, key=lambda r: r[6])
    return best, df[f"p_{label}"]


def per_person(df, gts, prob_col, th):
    """F1 แยกรายคน — ตัวชี้ขาดว่าผลรวมมาจากทุกคนหรือมาจากคนเดียว

    ผลรวมที่ดีขึ้นเพราะคน ๆ เดียวคือ overfit ที่มองไม่เห็นจากตัวเลขรวม
    เพราะ LOPO ก็ยังเฉลี่ยข้ามคนอยู่ดี
    """
    out = {}
    for person, g in df.groupby("person"):
        sub = {c: gts[c] for c in g["clip"].unique()}
        r = eval_thresholds(g, prob_col, sub, [th])[0]
        out[person] = (r[1], r[2], r[3], r[6])   # TP, FN, FP, F1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=str(ROOT / "dataset/benchmark_cache"))
    ap.add_argument("--dataset-root", default=str(ROOT / "dataset"))
    ap.add_argument("--n-estimators", type=int, default=300)
    ap.add_argument("--max-depth", type=int, default=6)
    args = ap.parse_args()

    print("=== สร้าง candidate + ฟีเจอร์วงสวิง ===")
    df, gts, missed = build_rows(Path(args.cache_dir), Path(args.dataset_root))
    if df.empty:
        print("ไม่มีข้อมูล — cache v2 ยังไม่พร้อม?")
        return

    y = df["is_hit"].to_numpy()
    print(f"\ncandidate {len(df)} · is_hit=1 {int(y.sum())} ({y.mean():.1%}) "
          f"· {df['clip'].nunique()} คลิป · {df['person'].nunique()} คน")

    # ─── Q1: ฟีเจอร์แต่ละตัวแยกได้แค่ไหน ───
    print("\n=== Q1: AUC ต่อฟีเจอร์ (0.5 = ไร้ค่า · >0.60 = มีสัญญาณ) ===")
    print(f"{'ฟีเจอร์':<22s} {'AUC':>7s}   {'mean hit':>10s} {'mean อื่น':>10s}")
    print("-" * 56)
    scored = sorted(((c, auc(y, df[c].to_numpy())) for c in SWING_FEATURE_COLS),
                    key=lambda t: -abs(t[1] - 0.5))
    for c, a in scored:
        mark = " <-" if abs(a - 0.5) >= 0.10 else ""
        print(f"{c:<22s} {a:7.3f}   {df.loc[y == 1, c].mean():10.3f} "
              f"{df.loc[y == 0, c].mean():10.3f}{mark}")
    print("\n(อ้างอิง — ฟีเจอร์เดิมที่แรงสุด)")
    for c, a in sorted(((c, auc(y, df[c].to_numpy())) for c in FEATURE_COLS),
                       key=lambda t: -abs(t[1] - 0.5))[:4]:
        print(f"{c:<22s} {a:7.3f}")

    # ─── Q2: ใส่เข้าโมเดลแล้วดีขึ้นไหม ───
    print("\n=== Q2: LOPO ระดับ stroke (threshold ที่ F1 ดีสุด) ===")
    combos = [("base", FEATURE_COLS),
              ("+swing", FEATURE_COLS + SWING_FEATURE_COLS),
              ("swing เดี่ยว", SWING_FEATURE_COLS)]
    print(f"{'ชุดฟีเจอร์':<14s} {'n':>4s} {'th':>5s} {'TP':>5s} {'FN':>5s} "
          f"{'FP':>5s} {'recall':>8s} {'precision':>10s} {'F1':>7s}")
    print("-" * 70)
    base_f1, probs = None, {}
    for label, cols in combos:
        best, p = report(df, gts, label, cols, args)
        probs[label] = (p, best[0])
        if label == "base":
            base_f1 = best[6]
        delta = "" if base_f1 is None or label == "base" else \
            f"  ({best[6] - base_f1:+.3f})"
        print(f"{label:<14s} {len(cols):4d} {best[0]:5.2f} {best[1]:5d} "
              f"{best[2]:5d} {best[3]:5d} {best[4]:8.1%} {best[5]:10.1%} "
              f"{best[6]:7.3f}{delta}")

    # ─── แยกรายคน: ผลรวมดีขึ้นเพราะทุกคน หรือเพราะคนเดียว ───
    print("\n=== F1 แยกรายคน (ที่ threshold ของแต่ละชุด) ===")
    for label, (p, th) in probs.items():
        df[f"p_{label}"] = p
    people = sorted(df["person"].unique())
    print(f"{'คน':<10s} {'GT':>4s} " +
          " ".join(f"{l:>14s}" for l in probs))
    print("-" * (16 + 15 * len(probs)))
    tables = {l: per_person(df, gts, f"p_{l}", th) for l, (_, th) in probs.items()}
    wins = 0
    for person in people:
        b = tables["base"][person]
        cells = []
        for l in probs:
            tp, fn, fp, f1 = tables[l][person]
            cells.append(f"{f1:6.3f} {tp:2d}/{tp+fn:<2d}F{fp:<3d}")
        if tables["+swing"][person][3] > b[3]:
            wins += 1
        print(f"{person:<10s} {b[0]+b[1]:4d} " + " ".join(f"{c:>14s}" for c in cells))
    print(f"\n'+swing' ชนะ base ที่ {wins}/{len(people)} คน "
          f"(รูปแบบ: F1 TP/GT F=FP)")

    # ─── Q3: ช่วยลูกที่ไม่เกิด candidate ได้ไหม ───
    print(f"\n=== Q3: GT ที่ไม่มี candidate เลย ({len(missed)} ลูก) ===")
    if missed.empty:
        print("ไม่มี — candidate generation ครอบคลุม GT ครบทุกลูก")
    else:
        print("ถ้าค่าที่เฟรมพวกนี้ใกล้ 'hit จริง' มากกว่า 'candidate ขยะ' "
              "แปลว่าสร้าง candidate จากวงสวิงเพิ่มได้")
        print(f"\n{'ฟีเจอร์':<22s} {'GT ที่พลาด':>11s} {'hit ที่เจอ':>11s} "
              f"{'ขยะ':>10s}")
        print("-" * 58)
        for c, _ in scored:
            print(f"{c:<22s} {missed[c].mean():11.3f} "
                  f"{df.loc[y == 1, c].mean():11.3f} "
                  f"{df.loc[y == 0, c].mean():10.3f}")

    print("\nสรุป: ดู Q2 เป็นหลัก — ถ้า '+swing' ไม่ชนะ 'base' อย่างมี"
          "นัยสำคัญ (> ~0.02) ไม่ควรลงแรงต่อ")


if __name__ == "__main__":
    main()
