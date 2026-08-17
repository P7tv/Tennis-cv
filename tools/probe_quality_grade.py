"""metric ที่เราส่งออก อธิบาย 'เกรดฝีมือ' ที่โค้ช label ไว้ได้มั้ย?

บริบท: ลูกค้าจะเอาไปทำแอปวิเคราะห์คุณภาพการตี (คล้ายแอปวิเคราะห์วงสวิงกอล์ฟ)
ให้โค้ชเอาไปคุยกับผู้เล่น — สิ่งที่แอปต้องทำได้จริงคือ "บอกว่าตีดีหรือไม่ดี
และดี/ไม่ดีตรงไหน" ไม่ใช่ "ระบุเฟรมปะทะแม่น ±1 เฟรม"

ลูกค้า label `impact_quality` (best/good/ok/bad/worst) ไว้ 171 stroke แล้ว
= ground truth ของสิ่งที่แอปต้องทำนาย แต่เรายังไม่เคยใช้เลย

สคริปต์นี้: สร้าง schema แบบ Mode B (ป้อน impact จริงให้ = แยกปัญหา hit
detection ออก) แล้ววัดว่าแต่ละ metric สัมพันธ์กับเกรดแค่ไหน
"""
import argparse
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loeuf_cv.config import PipelineConfig  # noqa: E402
from loeuf_cv.schema_builder.builder import build_loeuf_schema  # noqa: E402
from tools.label_ingest import find_session_labels  # noqa: E402

GRADE = {"worst": 1, "bad": 2, "ok": 3, "good": 4, "best": 5}


def flatten(d, prefix=""):
    """metric block ซ้อนกันหลายชั้น -> {ชื่อเต็ม: ค่าตัวเลข}"""
    out = {}
    for k, v in (d or {}).items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[key] = float(v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=str(ROOT / "dataset/benchmark_cache"))
    ap.add_argument("--dataset-root", default=str(ROOT / "dataset"))
    ap.add_argument("--min-n", type=int, default=60,
                    help="metric ต้องมีค่าจริงอย่างน้อยกี่ stroke ถึงจะรายงาน")
    args = ap.parse_args()

    labels = {}
    for p in find_session_labels(Path(args.dataset_root)):
        raw = json.loads(Path(p).read_text(encoding="utf-8"))
        key = f"{Path(p).parent.name}/{Path(raw['strokes'][0]['clip_path']).stem}"
        labels[key] = raw["strokes"]

    rows = []
    for pkl in sorted((Path(args.cache_dir) / "tracking").rglob("*.pkl")):
        key = f"{pkl.parent.name}/{pkl.stem}"
        with open(pkl, "rb") as f:
            c = pickle.load(f)
        if c.get("cache_version") != 2:
            continue
        # หา label ที่ตรงกับคลิปนี้ (ชื่อไฟล์ใน label อาจไม่ตรงเป๊ะ)
        lab = None
        for lk, lv in labels.items():
            if Path(lk).stem.split("(")[0] in pkl.stem or pkl.stem in lk:
                lab = lv
                break
        if lab is None:
            continue

        gt = [s for s in lab if s.get("impact_frame") is not None]
        if not gt:
            continue
        track = c["track"]
        vm, fps = c["video_meta"], c["fps"]
        cfg = PipelineConfig(dominant_side=c["dominant_side"],
                             subject_height_cm=c["subject_height_cm"])
        hits = [{"frame": int(s["impact_frame"]), "player_id": track.track_id,
                 "confidence": 1.0} for s in sorted(gt, key=lambda s: s["impact_frame"])]
        try:
            schema = build_loeuf_schema([track], hits, fps, vm, cfg,
                                        racket_keypoints=c.get("racket_keypoints"),
                                        ball_traj=c.get("ball_traj"))
        except Exception as e:
            print(f"  ข้าม {key}: {e}")
            continue

        srt = sorted(gt, key=lambda s: s["impact_frame"])
        for st, sc in zip(srt, schema.get("strokes", [])):
            g = GRADE.get(st.get("impact_quality"))
            if g is None:
                continue
            feats = {}
            for blk in ("metric", "kinematics", "stroke_specific"):
                feats.update(flatten(sc.get(blk), f"{blk}."))
            rows.append({"grade": g, "type": st.get("stroke_type"),
                         "player": st.get("player_id"), **feats})
        print(f"  {key:28s} {len(srt)} stroke")

    if not rows:
        print("ไม่มีข้อมูล")
        return

    grades = np.array([r["grade"] for r in rows])
    print(f"\nstroke ที่ใช้ได้ {len(rows)} · การกระจายเกรด "
          f"{dict(zip(*np.unique(grades, return_counts=True)))}")

    cols = sorted({k for r in rows for k in r
                   if k not in ("grade", "type", "player")})
    res = []
    for c in cols:
        v = np.array([r.get(c, np.nan) for r in rows], dtype=float)
        ok = ~np.isnan(v)
        if ok.sum() < args.min_n or np.nanstd(v[ok]) == 0:
            continue
        res.append((abs(np.corrcoef(v[ok], grades[ok])[0, 1]), c, ok.sum(),
                    np.corrcoef(v[ok], grades[ok])[0, 1]))
    res.sort(reverse=True)

    # ── corr แบบหักอิทธิพลของ 'ตัวบุคคล + ประเภทท่า' ออก ──
    # จำเป็นเพราะเกรดผูกกับคนมาก (pl004 ได้ worst ทั้ง 10 ครั้ง · pl011 เฉลี่ย 3.61)
    # metric ที่แค่บอกว่า "คนไหน" (เช่น body_height_cm) จะ corr สูงโดยไม่มีความหมาย
    # วิธี: ลบค่าเฉลี่ยของกลุ่ม (คน × ท่า) ออกจากทั้ง metric และเกรด แล้วค่อย corr
    groups = [(r["player"], r["type"]) for r in rows]

    def demean(vals, mask):
        out = np.full(len(vals), np.nan)
        acc = defaultdict(list)
        for i, g in enumerate(groups):
            if mask[i]:
                acc[g].append(vals[i])
        for i, g in enumerate(groups):
            if mask[i] and len(acc[g]) >= 3:
                out[i] = vals[i] - np.mean(acc[g])
        return out

    within = {}
    for _, c, n, r in res:
        v = np.array([row.get(c, np.nan) for row in rows], dtype=float)
        ok = ~np.isnan(v)
        dv, dg = demean(v, ok), demean(grades.astype(float), ok)
        m = ~np.isnan(dv) & ~np.isnan(dg)
        if m.sum() >= 30 and np.std(dv[m]) > 0 and np.std(dg[m]) > 0:
            within[c] = (float(np.corrcoef(dv[m], dg[m])[0, 1]), int(m.sum()))

    print(f"\n{'metric':50s} {'n':>4s} {'corr ดิบ':>9s} {'corr หักตัวบุคคล':>17s}")
    print("-" * 84)
    for _, c, n, r in res[:25]:
        w, wn = within.get(c, (float("nan"), 0))
        flag = "  <- เหลือสัญญาณ" if abs(w) >= 0.25 else ""
        print(f"{c:50s} {n:4d} {r:9.3f} {w:17.3f}{flag}")
    print(f"\nmetric ที่มีค่าจริง >= {args.min_n} stroke: {len(res)} / {len(cols)}")
    print("'หักตัวบุคคล' = ลบค่าเฉลี่ยของกลุ่ม (ผู้เล่น × ท่า) ออกก่อน")
    print("ถ้าค่านี้ตกลงใกล้ 0 -> metric นั้นแค่บอกว่า 'คนไหน' ไม่ได้วัดคุณภาพการตี")

    # ── คำถามที่ตรงกับสินค้า: ทำนายเกรดของ 'ผู้เล่นใหม่' ได้มั้ย ──
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import LeaveOneGroupOut

    use = [c for _, c, _, _ in res]
    X = np.array([[r.get(c, np.nan) for c in use] for r in rows], dtype=float)
    X = np.nan_to_num(X, nan=0.0)
    y = grades.astype(float)
    people = np.array([r["player"] for r in rows])

    pred = np.full(len(y), np.nan)
    for tr, te in LeaveOneGroupOut().split(X, y, people):
        m = RandomForestRegressor(n_estimators=300, max_depth=6,
                                  random_state=42, n_jobs=-1)
        m.fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])

    base = np.full(len(y), float(np.mean(y)))     # ทายค่าเฉลี่ยเสมอ
    print("\n=== ทำนายเกรด (Leave-One-Person-Out) ===")
    print(f"{'':22s} {'MAE':>7s} {'corr':>7s} {'ถูก ±1 เกรด':>13s}")
    for name, p in (("โมเดล", pred), ("ทายค่าเฉลี่ยเสมอ", base)):
        mae = float(np.mean(np.abs(p - y)))
        cr = float(np.corrcoef(p, y)[0, 1]) if np.std(p) > 0 else float("nan")
        print(f"{name:22s} {mae:7.2f} {cr:7.3f} {np.mean(np.abs(p-y) <= 1):13.1%}")
    print("\nถ้าโมเดลไม่ชนะ 'ทายค่าเฉลี่ยเสมอ' = ยังทำนายคุณภาพการตีไม่ได้")


if __name__ == "__main__":
    main()
