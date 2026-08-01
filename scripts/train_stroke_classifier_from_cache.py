"""เทรน stroke classifier ใหม่โดยใช้ feature จาก 'ทางเดียวกับตอนใช้งานจริง'

ปัญหาที่แก้ (train/serve skew):
  ของเดิม train_model/build_training_data.py คำนวณ feature เองด้วย
  get_body_metrics() แต่ production ป้อนค่าจาก MetricsEngine ค่าที่ได้ไม่ตรงกัน
  ทั้งสเกลและเครื่องหมาย:

    feature                              ตอนเทรน   ตอนใช้งาน
    arm_backswing_depth_deg                125.3      -12.8
    contact_distance_from_body_cm           58.7       12.8
    body_shoulder_rotation_at_impact_deg     4.4       -7.4
    body_hip_rotation_at_impact_deg          5.3      -13.6

  ผลคือโมเดลเจอข้อมูลคนละแบบกับที่เรียนมา -> BH recall เหลือ 0.115
  ทั้งที่กฎเรขาคณิตล้วน ๆ ยังได้ 0.577

ตัวนี้เดินผ่าน MetricsEngine เหมือน builder เป๊ะ แล้วเรียก
classifier.build_stroke_features() ซึ่งเป็นแหล่งเดียวกับตอน inference
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

from loeuf_cv.config import PipelineConfig  # noqa: E402
from loeuf_cv.metrics import MetricsEngine  # noqa: E402
from loeuf_cv.schema_builder.builder import (  # noqa: E402
    _build_engine_keyframes, _slice_pose_for_stroke)
from loeuf_cv.schema_builder.classifier import (  # noqa: E402
    _rule_based_classify, build_stroke_features)
from loeuf_cv.config import (L_SHOULDER, L_WRIST, NOSE,  # noqa: E402
                             R_SHOULDER, R_WRIST)
from loeuf_cv.schema_builder.keyframes import extract_keyframes  # noqa: E402
from train_model.label_ingest import (find_session_labels,  # noqa: E402
                                      load_session_label)

MIN_CLASS_SAMPLES = 15   # class ที่ตัวอย่างน้อยกว่านี้ ไม่ให้ ML ตัดสิน


def build_rows(cache_dir: Path, dataset_root: Path) -> pd.DataFrame:
    sess = {}
    for p in find_session_labels(dataset_root):
        s = load_session_label(p)
        sess[f"{Path(p).parent.name}/{Path(s.video_path).stem}"] = s

    rows = []
    for pkl in sorted((cache_dir / "tracking").rglob("*.pkl")):
        key = f"{pkl.parent.name}/{pkl.stem}"
        if key not in sess:
            continue
        with open(pkl, "rb") as f:
            c = pickle.load(f)
        if c.get("cache_version") != 2:
            continue
        s = sess[key]
        dom, pose, fps = c["dominant_side"], c["track"].pose, c["fps"]
        cfg = PipelineConfig(dominant_side=dom,
                             subject_height_cm=c["subject_height_cm"] or 170.0)
        n = c["video_meta"].get("total_frames", 0)
        widx = R_WRIST if dom == "right" else L_WRIST
        wrist_path = pose.landmarks[:, widx, :2]
        head_path = pose.landmarks[:, NOSE, :2]
        _sh = np.abs(pose.landmarks[:, R_SHOULDER, 0]
                     - pose.landmarks[:, L_SHOULDER, 0])
        sw = float(np.nanmedian(_sh)) if not np.isnan(_sh).all() else None

        for st in s.strokes:
            imp = st.keyframes.get("impact")
            if imp is None or imp >= len(pose.landmarks):
                continue
            # ⚠️ ต้องใช้ keyframe ที่ 'ระบบทายเอง' ไม่ใช่ของเฉลย — builder เรียก
            # classify_stroke ก่อน refine_keyframes_for_type เสมอ ถ้าเทรนด้วย
            # keyframe จากเฉลย metric ที่คำนวณจากหน้าต่างนั้นจะคนละแบบกับตอน
            # ใช้งาน = train/serve skew รอบสอง (รอบแรกคือ get_body_metrics
            # vs MetricsEngine)
            kf = extract_keyframes(imp, wrist_path, fps, n,
                                   head_path=head_path, shoulder_width=sw)
            det = [f for f in kf.values() if f is not None]
            s0 = max(0, min(det + [imp - 30]))
            s1 = min(n - 1, max(det + [imp + 30]))
            try:
                sp = _slice_pose_for_stroke(pose, s0, s1, cfg)
                ekf = _build_engine_keyframes(kf, fps, s0)
                blocks, _, _ = MetricsEngine(sp, ekf, cfg, "FH").compute()
                feats = build_stroke_features(pose, imp, dom, blocks, kf)
            except Exception as e:
                print(f"  ข้าม {key}#{st.stroke_no}: {e}")
                continue
            rows.append({**feats, "y": st.stroke_type,
                         "player": st.player_id, "clip": key,
                         "rule": _rule_based_classify(pose.landmarks[imp],
                                                      widx, dom)})
        print(f"  {key:28s} {len(s.strokes)} stroke")
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=str(ROOT / "dataset/benchmark_cache"))
    ap.add_argument("--dataset-root", default=str(ROOT / "dataset"))
    ap.add_argument("--out", default=str(ROOT / "stroke_classifier.pkl"))
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import LeaveOneGroupOut

    print("=== สร้าง feature ผ่านทางเดียวกับ production ===")
    df = build_rows(Path(args.cache_dir), Path(args.dataset_root))
    if df.empty:
        print("ไม่มีข้อมูล")
        return

    meta = {"y", "player", "clip", "rule"}
    cols = [c for c in df.columns if c not in meta]
    X = df[cols].astype(float).fillna(0.0)
    y = df["y"]
    print(f"\n{len(df)} stroke · {df['player'].nunique()} คน · "
          f"{dict(Counter(y))}")

    counts = Counter(y)
    supported = {k for k, v in counts.items() if v >= MIN_CLASS_SAMPLES}
    print(f"class ที่ให้ ML ตัดสิน (>= {MIN_CLASS_SAMPLES} ตัวอย่าง): "
          f"{sorted(supported)}")

    # ── LOPO ──
    oof = pd.Series(index=df.index, dtype=object)
    for tr, te in LeaveOneGroupOut().split(X, y, df["player"]):
        m = RandomForestClassifier(n_estimators=400, max_depth=8,
                                   random_state=42, class_weight="balanced",
                                   n_jobs=-1)
        m.fit(X.iloc[tr], y.iloc[tr])
        oof.iloc[te] = m.predict(X.iloc[te])

    T = ["FH", "BH", "SV", "VL", "SL"]
    corner = "จริง|ทาย"   # Python 3.11: ห้ามมี backslash ใน f-string
    for name, pred in (("กฎเรขาคณิตอย่างเดียว", df["rule"]),
                       ("ML ใหม่ (LOPO)", oof)):
        acc = float((pred == y).mean())
        print(f"\n=== {name} — ความแม่นรวม {acc:.3f} ===")
        print(f"{corner:10s}" + "".join(f"{t:>6s}" for t in T) + "  recall")
        for g in T:
            sub = pred[y == g]
            n = len(sub)
            if not n:
                continue
            print(f"{g:10s}" + "".join(f"{int((sub == t).sum()):6d}" for t in T)
                  + f"  {float((sub == g).mean()):.3f} (n={n})")

    clf = RandomForestClassifier(n_estimators=400, max_depth=8, random_state=42,
                                 class_weight="balanced", n_jobs=-1)
    clf.fit(X, y)
    print("\nฟีเจอร์สำคัญ 8 อันดับ:")
    for nm, im in sorted(zip(cols, clf.feature_importances_),
                         key=lambda x: -x[1])[:8]:
        print(f"  {nm:44s} {im*100:5.1f}%")

    if args.save:
        with open(args.out, "wb") as f:
            pickle.dump((clf, cols, supported), f)
        print(f"\nบันทึก -> {args.out}")
    else:
        print("\n(ไม่ได้บันทึก — ใส่ --save)")


if __name__ == "__main__":
    main()
