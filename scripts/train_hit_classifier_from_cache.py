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
import json
import pickle
import sys
import zlib
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loeuf_cv.augment import (jitter_landmarks, remap_bboxes,  # noqa: E402
                              rescale_body, resample_series, time_warp,
                              warp_frame_index)
from loeuf_cv.hit_detection import (detect_hit_events,  # noqa: E402
                                    extract_ball_trajectory_kalman)
from loeuf_cv.racket_shape import RACKET_FEATURE_COLS  # noqa: E402
from loeuf_cv.swing_shape import (SWING3D_FEATURE_COLS,  # noqa: E402
                                  SWING_FEATURE_COLS)
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

# ฟีเจอร์รูปทรงวงสวิง (loeuf_cv/swing_shape.py) — ตอบ "มันวาดเป็นวงตีจริงไหม"
# ต่างจากชุดบนที่ตอบได้แค่ "ข้อมือเร็วไหม ณ เฟรมนั้น"
SWING_COLS = list(SWING_FEATURE_COLS)

# รูปทรงวงสวิงเดียวกันแต่คำนวณบนพิกัด 3 มิติ (world_landmarks) — ตัดผลของมุมกล้อง
SWING3D_COLS = list(SWING3D_FEATURE_COLS)

# ฟีเจอร์ของไม้ที่ "ไม่ต้องใช้ลูกบอล" — ของเดิม racket_dist วัดระยะไม้ถึงลูก
# จึงใช้ได้เฉพาะเฟรมที่เห็นลูก (4.4-22%) ทั้งที่เห็นไม้ 15-100%
RACKET_COLS = list(RACKET_FEATURE_COLS)

# ชุดที่ไม่พึ่งตำแหน่งลูก/ไม้เลย — ในคลิปจริงเห็นลูกลอยแค่ 4.4-22% ของเฟรม
# ที่เหลือ ball_dist ถูกเติมด้วย 9999 ปลอม ซึ่งเป็น "ค่าคงที่ที่ไม่ได้แปลว่า
# ไม่มีลูก" -> RF เอาไปแตกกิ่งแล้วเรียนสิ่งที่ไม่มีอยู่จริง
SPEED_ONLY_COLS = [c for c in SCALED_COLS
                   if not c.startswith(("ball_", "racket_"))]

FEATURE_SETS = {
    "base": BASE_COLS,
    "scaled": SCALED_COLS + ["speed_decel_ratio", "speed_peak_sharpness",
                             "ball_angle_change"],
    "all": BASE_COLS + SCALED_COLS,
    "swing": SWING_COLS,
    "swing+speed": SWING_COLS + SPEED_ONLY_COLS,          # <- ชุดที่ใช้อยู่
    "swing+scaled": SWING_COLS + SCALED_COLS,
    "all+swing": BASE_COLS + SCALED_COLS + SWING_COLS,
    # ── กลุ่มที่กำลังทดลอง: เพิ่มทีละกลุ่มบนชุดที่ใช้อยู่ เพื่อแยกผลของแต่ละอันได้ ──
    "+3d": SWING_COLS + SPEED_ONLY_COLS + SWING3D_COLS,
    "+racket": SWING_COLS + SPEED_ONLY_COLS + RACKET_COLS,
    "+3d+racket": SWING_COLS + SPEED_ONLY_COLS + SWING3D_COLS + RACKET_COLS,
    "3d เดี่ยว": SWING3D_COLS,
    "racket เดี่ยว": RACKET_COLS,
}
FEATURE_COLS = FEATURE_SETS["swing+speed"]

# แถวที่สร้างเก็บ "ทุก" ฟีเจอร์เสมอ ไม่ว่าจะเทรนด้วยชุดไหน — สร้างแถวคือส่วนที่
# แพง (ต้องรัน detect_hit_events ใหม่ทั้ง dataset) ส่วนการเลือกคอลัมน์ถูกมาก
# แยกกันแบบนี้ทำให้ ablation สร้างแถวรอบเดียวแล้วเทียบได้ทุกชุด
ALL_FEATURE_COLS = list(dict.fromkeys(
    BASE_COLS + SCALED_COLS + SWING_COLS + SWING3D_COLS + RACKET_COLS))
LABEL_TOL = 5   # candidate ห่าง GT impact <= 5 เฟรม ถือว่าเป็น hit จริง
MATCH_TOL = 10  # ตอนวัดผล ใช้ tolerance เดียวกับ benchmark harness


def _candidates_for(track, traj, meas, racket, w, h, fps, gt, key, person,
                    variant: str) -> list[dict]:
    """รัน detect_hit_events บน pose ชุดหนึ่ง แล้วแปลงเป็นแถวฝึก

    ⚠️ เรียก detect_hit_events ตัวจริงเสมอ ไม่คำนวณ feature เอง — augmentation
    ต้องไหลผ่านโค้ดเดียวกับตอนใช้งาน ไม่งั้นเกิด train/serve skew (โปรเจกต์นี้
    โดนมาแล้ว 3 รอบ)
    """
    cands = detect_hit_events(traj, [track], fps, w, h, racket_bboxes=racket,
                              ball_measured=meas, return_candidates=True)
    out = []
    for ev in cands:
        fe = ev.get("features", {})
        f = ev["frame"]
        # ระยะมีเครื่องหมายถึงเฉลยที่ใกล้ที่สุด (เฉลย - เฟรมนี้) = "ต้องเลื่อน
        # ไปกี่เฟรมถึงจะตรง" ใช้เป็นเป้าของตัวปรับเฟรมปะทะ
        # (scripts/train_impact_refiner.py) เก็บไว้ทุกแถวเพราะการสร้าง candidate
        # คือส่วนที่แพง จะได้ไม่ต้องรันซ้ำเมื่ออยากเทรนตัวปรับ
        nearest = min(gt, key=lambda g: abs(g - f)) if gt else None
        out.append({
            **{col: fe.get(col, 0.0) for col in ALL_FEATURE_COLS},
            "is_hit": int(any(abs(f - g) <= LABEL_TOL for g in gt)),
            "gt_offset": (nearest - f) if nearest is not None else np.nan,
            "frame": f, "clip": key, "person": person, "fps": fps,
            "n_gt": len(gt), "variant": variant, "is_aug": int(variant != "orig"),
        })
    return out


def augmented_variants(track, ball_bboxes, racket, w, h, fps, gt, seed: int,
                       n_aug: int):
    """สร้างชุด (track, traj, meas, racket, gt, ชื่อ) ที่ augment แล้ว

    ทำ 3 อย่างพร้อมกันต่อสำเนา:
      rescale   คนตัวใหญ่/เล็กลง (หรือกล้องใกล้/ไกลขึ้น)
      time warp สวิงเร็ว/ช้าลง  <- ตัวที่สร้าง "จังหวะปะทะแบบใหม่" จริง ๆ
      jitter    ความสั่นของ pose estimator (ขนาดวัดจากข้อมูลจริง)

    time warp ต้องยืดของทุกอย่างพร้อมกัน (pose + ball + racket + เฉลย) ไม่งั้น
    ฟีเจอร์ระยะห่างลูก-ข้อมือจะหลุดเฟรมกันทั้งชุด
    """
    import dataclasses

    rng = np.random.default_rng(seed)
    variants = []
    for i in range(n_aug):
        scale = float(rng.uniform(0.85, 1.20))
        warp = float(rng.uniform(0.80, 1.25))
        lm = rescale_body(track.pose.landmarks, scale)
        lm, src = time_warp(lm, warp)
        lm = jitter_landmarks(lm, rng)

        vis = track.pose.visibility
        vis_w = resample_series(vis.astype(float), src)
        ts = resample_series(track.pose.timestamps_ms.astype(float), src)
        pose = dataclasses.replace(
            track.pose, landmarks=lm, visibility=vis_w,
            world_landmarks=resample_series(track.pose.world_landmarks, src),
            timestamps_ms=ts)
        t2 = dataclasses.replace(track, pose=pose)

        n_new = len(lm)
        bb = remap_bboxes(ball_bboxes, src)
        traj, meas = extract_ball_trajectory_kalman(
            bb, n_new, return_measured=True, fps=fps)
        gt2 = [warp_frame_index(g, src) for g in gt]
        variants.append((t2, traj, meas, remap_bboxes(racket, src), gt2,
                         f"aug{i}_s{scale:.2f}_w{warp:.2f}"))
    return variants


def build_rows(cache_dir: Path, dataset_root: Path, n_aug: int = 0,
               seed: int = 0) -> pd.DataFrame:
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
        racket = c.get("racket_bboxes")
        traj, meas = extract_ball_trajectory_kalman(
            c["ball_bboxes"], len(track.pose.landmarks), return_measured=True,
            fps=fps)
        orig = _candidates_for(track, traj, meas, racket, w, h, fps, gt, key,
                               person, "orig")
        rows.extend(orig)

        n_aug_rows = 0
        # ⚠️ ห้ามใช้ hash() ของ Python ตรงนี้ — มันสุ่ม salt ใหม่ทุกโปรเซส
        # (PYTHONHASHSEED) ทำให้ augmentation ออกมาไม่เหมือนเดิมทุกครั้งที่รัน
        # เจอจริง: n_aug=6 ชุดเดิม รันสองรอบได้ F1 0.528 กับ 0.508
        # crc32 เสถียรข้ามโปรเซส/เครื่อง/เวอร์ชัน Python
        clip_seed = seed + zlib.crc32(key.encode("utf-8")) % 10_000
        for t2, tr2, me2, rb2, gt2, name in augmented_variants(
                track, c["ball_bboxes"], racket, w, h, fps, gt,
                clip_seed, n_aug):
            r = _candidates_for(t2, tr2, me2, rb2, w, h, fps, gt2, key, person,
                                name)
            rows.extend(r)
            n_aug_rows += len(r)

        extra = f" · augment +{n_aug_rows}" if n_aug else ""
        print(f"  {key:28s} candidate {len(orig):4d} · GT {len(gt):3d} · "
              f"{person}{extra}")
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
    ap.add_argument("--n-aug", type=int, default=0,
                    help="จำนวนสำเนา augment ต่อคลิป (0 = ปิด) — ใช้เฉพาะฝั่ง "
                         "train เท่านั้น ตอนวัดผลตัดออกเสมอ")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lopo-out", default=None,
                    help="โฟลเดอร์เก็บโมเดลชุด LOPO (<person>.pkl = เทรนโดย "
                         "ตัดคนนั้นออก) สำหรับ benchmark ที่ไม่ปนข้อมูลที่เคยเห็น")
    # ค่า default = ชุดที่ชนะจาก scripts/ablate_hit_features.py (LOPO, augment 3)
    #   all (เดิม)   F1 0.460 · recall 59.9% · precision 37.3%
    #   swing+speed  F1 0.665 · recall 69.8% · precision 63.5%
    # ที่ตัด ball_dist/racket_dist ออกเพราะเห็นลูกลอยแค่ 4.4-22% ของเฟรม
    # ที่เหลือถูกเติมด้วย 9999 ซึ่งเป็นค่าคงที่ที่ไม่ได้แปลว่า "ไม่มีลูก"
    ap.add_argument("--features", choices=tuple(FEATURE_SETS),
                    default="swing+speed",
                    help=" · ".join(f"{k}={len(v)} ตัว"
                                    for k, v in FEATURE_SETS.items()))
    args = ap.parse_args()

    global FEATURE_COLS
    FEATURE_COLS = FEATURE_SETS[args.features]

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import LeaveOneGroupOut

    print("=== สร้าง candidate + feature จาก cache ===")
    df = build_rows(Path(args.cache_dir), Path(args.dataset_root),
                    n_aug=args.n_aug, seed=args.seed)
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
    n_orig = int((df["is_aug"] == 0).sum())
    print(f"\nแถวทั้งหมด {len(df)} (ของจริง {n_orig} · augment "
          f"{len(df) - n_orig}) · is_hit=1 {int(df['is_hit'].sum())} "
          f"({df['is_hit'].mean():.1%}) · {df['clip'].nunique()} คลิป "
          f"· {n_person} คน")

    X, y = df[FEATURE_COLS].fillna(0), df["is_hit"]

    # ─── LOPO: กันคนเดียวกันหลุดไปอยู่ทั้ง train และ test ───
    print(f"\n=== Leave-One-Person-Out ({n_person} คน) ===")
    df["oof_prob"] = np.nan
    logo = LeaveOneGroupOut()
    is_aug = df["is_aug"].to_numpy()
    for tr, te in logo.split(X, y, df["person"]):
        # 🔴 สำเนา augment ของ "คนที่กำลังทดสอบ" ก็อยู่ใน fold ทดสอบด้วย
        # (LeaveOneGroupOut จัดกลุ่มตามคน) ถ้าปล่อยไว้จะกลายเป็นวัดผลบนข้อมูล
        # สังเคราะห์ -> ตัวเลขสวยเกินจริง จึงตัดออกจากทั้งการ fit และการวัด
        tr = tr[is_aug[tr] >= 0]          # train ใช้ได้ทั้งของจริงและ augment
        te_real = te[is_aug[te] == 0]     # test ใช้เฉพาะของจริงเท่านั้น
        clf = RandomForestClassifier(n_estimators=args.n_estimators,
                                     max_depth=args.max_depth,
                                     random_state=42, class_weight="balanced",
                                     n_jobs=-1)
        clf.fit(X.iloc[tr], y.iloc[tr])
        if len(te_real):
            df.iloc[te_real, df.columns.get_loc("oof_prob")] = \
                clf.predict_proba(X.iloc[te_real])[:, 1]

    # เก็บชุดเต็ม (รวม augment) ไว้ก่อน — ใช้เทรนโมเดล LOPO ท้ายไฟล์
    df_all = df
    # วัดผลบนแถวของจริงล้วน
    df = df[df["is_aug"] == 0].reset_index(drop=True)

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

    # ─── โมเดลชุด LOPO สำหรับ benchmark ที่ไม่ปนข้อมูลที่เคยเห็น ───
    #
    # ⚠️ ทำไมต้องมี: hit_classifier.pkl ตัวจริงเทรนจาก 16 คลิปเดียวกับที่ใช้วัด
    # benchmark → ตัวเลข recall ที่รายงานไปมีส่วนที่มาจาก "การจำคลิป" ไม่ใช่
    # ความสามารถจริง (วัดแล้ว: LOPO recall 54.7% แต่ benchmark 86.6% =
    # ช่องว่าง 31.9 pp) เอาไปเทียบโมเดลสองตัวไม่ได้ เพราะตัวที่จำเก่งกว่าชนะ
    #
    # โมเดลในโฟลเดอร์นี้: <person>.pkl = เทรนโดย **ตัดคนนั้นออกทั้งหมด**
    # benchmark เอาไปใช้กับคลิปของคนนั้น -> ได้ตัวเลข end-to-end ที่ซื่อสัตย์
    if args.lopo_out:
        out_dir = Path(args.lopo_out)
        out_dir.mkdir(parents=True, exist_ok=True)
        clip2person = (df_all[["clip", "person"]].drop_duplicates()
                       .set_index("clip")["person"].to_dict())
        for person in sorted(df_all["person"].unique()):
            tr = df_all["person"] != person
            m = RandomForestClassifier(n_estimators=args.n_estimators,
                                       max_depth=args.max_depth,
                                       random_state=42,
                                       class_weight="balanced", n_jobs=-1)
            m.fit(df_all.loc[tr, FEATURE_COLS].fillna(0),
                  df_all.loc[tr, "is_hit"])
            with open(out_dir / f"{person}.pkl", "wb") as f:
                pickle.dump((m, FEATURE_COLS), f)
            print(f"  LOPO model (ไม่มี {person}): "
                  f"{int(tr.sum())} แถว -> {person}.pkl")
        with open(out_dir / "clip_person.json", "w", encoding="utf-8") as f:
            json.dump(clip2person, f, ensure_ascii=False, indent=2)
        print(f"\nบันทึกชุด LOPO -> {out_dir}")


if __name__ == "__main__":
    main()
