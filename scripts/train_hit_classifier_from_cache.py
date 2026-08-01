"""เทรน hit classifier ใหม่จาก benchmark cache v2 — ไม่ต้องรัน YOLO

ทำไมต้องมีตัวนี้ทั้งที่มี train_model/build_training_data.py อยู่แล้ว:
  - ตัวเดิมรัน YOLO ใหม่ทุกครั้ง (~2 ชม. ต่อรอบ) ทำให้ทดลอง ML ไม่ไหว
  - cache v2 เก็บ ball_bboxes ดิบไว้แล้ว จึงสร้าง candidate + feature ใหม่ได้
    ในไม่กี่นาที
  - โมเดลตัวเก่าเทรนด้วยฟีเจอร์ที่คำนวณจาก ball trajectory ที่ยังพังอยู่
    (ดู docs/BALL_DETECTION_ISSUE.md) ฟีเจอร์เปลี่ยนไปหมดแล้ว ต้องเทรนใหม่

การประเมิน: Leave-One-Person-Out ระดับ "stroke" ไม่ใช่ระดับแถว —
วัด recall/precision หลังผ่าน threshold + NMS จริง เพราะนั่นคือสิ่งที่ใช้งาน
(leave-one-clip-out ยังรั่ว เพราะคนเดียวกันมีหลายคลิป โมเดลจำสไตล์คนได้)
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
from train_model.label_ingest import (find_session_labels,  # noqa: E402
                                      load_session_label)

BASE_COLS = [
    "wrist_speed", "ball_dist", "ball_vel_before", "ball_vel_after",
    "ball_vel_change", "ball_angle_change", "racket_dist",
    "speed_pre_mean", "speed_post_mean", "speed_decel_ratio",
    "speed_peak_sharpness", "speed_std_window",
]

# ฟีเจอร์ที่หารด้วยขนาดตัว (ความกว้างไหล่เป็น px) — เหตุผล: wrist_speed ดิบ
# เป็น px/frame ซึ่งขึ้นกับระยะกล้องและขนาดตัวคน คนที่ยืนไกลกล้องจะได้ค่า
# ต่ำกว่าคนที่ยืนใกล้ทั้งที่ตีแรงเท่ากัน -> โมเดลเรียน "ระยะกล้อง" แทนที่จะ
# เรียน "การตี" แล้วย้ายข้ามคนไม่ได้
SCALED_COLS = [
    "wrist_speed_bw", "ball_dist_bw", "racket_dist_bw",
    "speed_pre_bw", "speed_post_bw", "speed_std_bw",
    "wrist_dir_change", "wrist_y_rel", "wrist_x_rel",
]
FEATURE_COLS = BASE_COLS + SCALED_COLS
LABEL_TOL = 5   # candidate ห่าง GT impact <= 5 เฟรม ถือว่าเป็น hit จริง
MATCH_TOL = 10  # ตอนวัดผล ใช้ tolerance เดียวกับ benchmark harness


def build_rows(cache_dir: Path, dataset_root: Path) -> pd.DataFrame:
    sessions = {}
    for p in find_session_labels(dataset_root):
        try:
            s = load_session_label(p)
        except Exception:
            continue
        sessions[f"{p.parent.name}/{Path(s.video_path).stem}"] = s

    rows = []
    for pkl in sorted((cache_dir / "tracking").rglob("*.pkl")):
        key = f"{pkl.parent.name}/{pkl.stem}"
        if key not in sessions:
            continue
        with open(pkl, "rb") as f:
            c = pickle.load(f)
        if c.get("cache_version") != 2:
            print(f"  ข้าม {key} (cache v1)")
            continue

        s = sessions[key]
        gt = sorted(st.keyframes["impact"] for st in s.strokes
                    if st.keyframes.get("impact") is not None)
        # ผู้เล่นของคลิปนี้ = player_id ที่พบบ่อยสุดใน stroke (roster มีคนไม่ได้เล่นปนอยู่)
        pids = Counter(st.player_id for st in s.strokes if st.player_id)
        person = pids.most_common(1)[0][0] if pids else key

        track = c["track"]
        vm = c["video_meta"]
        w, h, fps = vm["width"], vm["height"], c["fps"]
        traj, meas = extract_ball_trajectory_kalman(
            c["ball_bboxes"], len(track.pose.landmarks), return_measured=True)
        cands = detect_hit_events(traj, [track], fps, w, h,
                                  racket_bboxes=c.get("racket_bboxes"),
                                  ball_measured=meas, return_candidates=True)
        for ev in cands:
            # detect_hit_events เติม feature ที่ normalize แล้วมาให้ครบตั้งแต่ต้นทาง
            fe = ev.get("features", {})
            rows.append({
                **{col: fe.get(col, 0.0) for col in FEATURE_COLS},
                "is_hit": int(any(abs(ev["frame"] - g) <= LABEL_TOL for g in gt)),
                "frame": ev["frame"],
                "clip": key, "person": person, "fps": fps,
                "n_gt": len(gt),
            })
        print(f"  {key:28s} candidate {len(cands):4d} · GT {len(gt):3d} · {person}")
    return pd.DataFrame(rows)


def nms(frames_probs, gap):
    chosen = []
    for f, p in sorted(frames_probs, key=lambda x: -x[1]):
        if all(abs(f - k) >= gap for k in chosen):
            chosen.append(f)
    return sorted(chosen)


def eval_thresholds(df, prob_col, gts, thresholds):
    """วัด recall/precision ระดับ stroke หลังผ่าน threshold + NMS จริง"""
    out = []
    for th in thresholds:
        TP = FN = FP = 0
        for clip, g in df.groupby("clip"):
            gt = gts[clip]
            gap = int(g["fps"].iloc[0] * 0.75)
            keep = g[g[prob_col] >= th]
            fs = nms(list(zip(keep["frame"], keep[prob_col])), gap)
            used = set()
            for gf in gt:
                m = [f for f in fs if abs(f - gf) <= MATCH_TOL and f not in used]
                if m:
                    used.add(min(m, key=lambda f: abs(f - gf)))
                    TP += 1
                else:
                    FN += 1
            FP += len(fs) - len(used)
        rec = TP / max(1, TP + FN)
        pre = TP / max(1, TP + FP)
        out.append((th, TP, FN, FP, rec, pre,
                    2 * rec * pre / max(1e-9, rec + pre)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=str(ROOT / "dataset/benchmark_cache"))
    ap.add_argument("--dataset-root", default=str(ROOT / "dataset"))
    ap.add_argument("--out", default=str(ROOT / "hit_classifier.pkl"))
    ap.add_argument("--rows-out", default=str(ROOT / "dataset/training/hit_candidates_cache.csv"))
    ap.add_argument("--n-estimators", type=int, default=300)
    ap.add_argument("--max-depth", type=int, default=6)
    ap.add_argument("--save", action="store_true",
                    help="เขียนทับโมเดลจริง (ไม่ใส่ = ประเมินอย่างเดียว)")
    ap.add_argument("--features", choices=("all", "scaled", "base"),
                    default="all",
                    help="all=ทั้งหมด · scaled=เฉพาะที่ normalize แล้ว "
                         "(+ ratio ที่ไม่มีหน่วย) · base=ชุดเดิม")
    args = ap.parse_args()

    global FEATURE_COLS
    if args.features == "scaled":
        # ratio/cosine ไม่มีหน่วย px อยู่แล้ว จึงข้ามข้ามคนได้ เก็บไว้ด้วย
        FEATURE_COLS = SCALED_COLS + ["speed_decel_ratio",
                                      "speed_peak_sharpness",
                                      "ball_angle_change"]
    elif args.features == "base":
        FEATURE_COLS = BASE_COLS

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import LeaveOneGroupOut

    print("=== สร้าง candidate + feature จาก cache ===")
    df = build_rows(Path(args.cache_dir), Path(args.dataset_root))
    if df.empty:
        print("ไม่มีข้อมูล — cache v2 ยังไม่พร้อม?")
        return
    Path(args.rows_out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.rows_out, index=False, encoding="utf-8")

    gts = {}
    for p in find_session_labels(Path(args.dataset_root)):
        try:
            s = load_session_label(p)
        except Exception:
            continue
        gts[f"{p.parent.name}/{Path(s.video_path).stem}"] = sorted(
            st.keyframes["impact"] for st in s.strokes
            if st.keyframes.get("impact") is not None)

    n_person = df["person"].nunique()
    print(f"\nแถวทั้งหมด {len(df)} · is_hit=1 {int(df['is_hit'].sum())} "
          f"({df['is_hit'].mean():.1%}) · {df['clip'].nunique()} คลิป "
          f"· {n_person} คน")

    X, y = df[FEATURE_COLS].fillna(0), df["is_hit"]

    # ─── LOPO: กันคนเดียวกันหลุดไปอยู่ทั้ง train และ test ───
    print(f"\n=== Leave-One-Person-Out ({n_person} คน) ===")
    df["oof_prob"] = np.nan
    logo = LeaveOneGroupOut()
    for tr, te in logo.split(X, y, df["person"]):
        clf = RandomForestClassifier(n_estimators=args.n_estimators,
                                     max_depth=args.max_depth,
                                     random_state=42, class_weight="balanced",
                                     n_jobs=-1)
        clf.fit(X.iloc[tr], y.iloc[tr])
        df.iloc[te, df.columns.get_loc("oof_prob")] = \
            clf.predict_proba(X.iloc[te])[:, 1]

    ths = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    print(f"{'threshold':>10s} {'TP':>5s} {'FN':>5s} {'FP':>5s} "
          f"{'recall':>8s} {'precision':>10s} {'F1':>7s}")
    print("-" * 56)
    best = None
    for r in eval_thresholds(df, "oof_prob", gts, ths):
        print(f"{r[0]:10.2f} {r[1]:5d} {r[2]:5d} {r[3]:5d} "
              f"{r[4]:8.1%} {r[5]:10.1%} {r[6]:7.3f}")
        if best is None or r[6] > best[6]:
            best = r
    print(f"\nF1 ดีสุดที่ threshold {best[0]:.2f} — "
          f"recall {best[4]:.1%} precision {best[5]:.1%}")

    # ─── โมเดลจริง: เทรนบนข้อมูลทั้งหมด ───
    clf = RandomForestClassifier(n_estimators=args.n_estimators,
                                 max_depth=args.max_depth, random_state=42,
                                 class_weight="balanced", n_jobs=-1)
    clf.fit(X, y)
    print("\nความสำคัญของฟีเจอร์:")
    for name, imp in sorted(zip(FEATURE_COLS, clf.feature_importances_),
                            key=lambda x: -x[1]):
        print(f"  {name:24s} {imp*100:5.1f}%")

    if args.save:
        with open(args.out, "wb") as f:
            pickle.dump((clf, FEATURE_COLS), f)
        print(f"\nบันทึกโมเดล -> {args.out}")
    else:
        print("\n(ไม่ได้บันทึกโมเดล — ใส่ --save ถ้าต้องการเขียนทับตัวจริง)")


if __name__ == "__main__":
    main()
