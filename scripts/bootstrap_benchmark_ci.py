"""ช่วงความเชื่อมั่นของตัวเลข benchmark — "ดีขึ้นเท่าไรถึงจะเชื่อได้"

ทำไมต้องมี: เราปรับปรุงระบบมาหลายรอบโดยดูแค่ค่ากลาง

    augmentation      +0.044
    ฟีเจอร์วงสวิง      +0.058
    ปรับเฟรมปะทะ      +0.032
    3D + ไม้เทนนิส     0.000

แต่ไม่เคยรู้เลยว่า "ความคลาดของการวัด" กว้างแค่ไหน ถ้าช่วงความเชื่อมั่นกว้าง
กว่าผลที่ได้ แปลว่าเรากำลังไล่ตาม noise และการจูนต่อจะยิ่งทำให้ overfit กับ
ผู้เล่น 7 คนนี้โดยที่ LOPO มองไม่เห็น (LOPO กันได้แค่ทีละคน แต่กันไม่ได้ว่า
คนเลือก "ทาง" ที่จะเดินจากการดูตัวเลขชุดเดิมซ้ำ ๆ)

วิธี: bootstrap จากผลรายลูกที่มีอยู่แล้วใน benchmark_keyframe_results.json
ไม่ต้องรันอะไรใหม่ ใช้เวลาไม่กี่วินาที

รายงาน 3 ระดับ เพราะให้คำตอบต่างกันมาก:
  รายลูก    สุ่ม stroke ใหม่ — สมมติว่าแต่ละลูกอิสระต่อกัน (มองโลกสวยเกินไป)
  รายคลิป   สุ่มคลิป — ลูกในคลิปเดียวกันสัมพันธ์กัน
  รายคน     สุ่มคน — หน่วยที่ระบบต้องย้ายข้ามจริง ๆ = ตัวเลขที่ควรใช้ตัดสิน

รัน:  python scripts/bootstrap_benchmark_ci.py
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ACCEPTANCE_KEYFRAMES = ("impact", "backswing_peak")


def load_stroke_results(results_path: Path, block: str, dataset_root: Path,
                        keyframes=ACCEPTANCE_KEYFRAMES):
    """คืน {stroke_id: [ผ่าน/ไม่ผ่าน ของแต่ละ keyframe ในเกณฑ์รับงาน]}

    keyframes: จำกัดให้เหลือเฉพาะบาง keyframe ได้ — ใช้ตอบว่า "สิ่งที่การ
    ปรับปรุงตั้งใจไปแก้ ดีขึ้นเกิน noise ไหม" แยกจากตัวเลขรับงานรวม ซึ่งเฉลี่ย
    กับ keyframe ที่การปรับปรุงนั้นไม่ได้แตะเลย (เช่นเสียงขยับแค่ impact
    ไม่ได้ขยับ backswing_peak -> ผลถูกเจือจางลงครึ่งหนึ่งโดยอัตโนมัติ)

    🔴 รายชื่อ stroke ต้องมาจาก **เฉลย** ไม่ใช่จากลิสต์ที่ตก — รายงานเก็บเฉพาะ
    รายการที่ตกไว้ ลูกที่ผ่านหมดทุก keyframe จึงไม่โผล่ในลิสต์เลย ถ้าสร้าง
    รายชื่อจากลิสต์ที่ตกจะหายไป 26 จาก 172 ลูก และทั้ง 26 ลูกนั้นคือลูกที่
    "ผ่าน" -> ค่าที่ได้เอนลงต่ำอย่างเป็นระบบ (0.342 แทนที่จะเป็น 0.442)
    """
    from loeuf_cv.benchmark_keyframes import stroke_key
    from scripts.run_keyframe_benchmark import _set_name
    from train_model.label_ingest import find_session_labels, load_session_label

    r = json.loads(results_path.read_text(encoding="utf-8"))
    fails = defaultdict(set)
    for f in r[block]["failures"]:
        if f["keyframe"] in keyframes:
            fails[f["clip"]].add(f["keyframe"])

    out = {}
    for p in find_session_labels(dataset_root):
        try:
            s = load_session_label(p)
        except Exception:
            continue
        # ต้องใช้ _set_name ตัวเดียวกับ harness — คลิปใน sessions_deferred/
        # ใช้ชื่อชุดเป็น "sessions_deferred/set4" ไม่ใช่ "set4" ถ้าใช้แค่ชื่อ
        # โฟลเดอร์ชั้นเดียว id จะไม่ตรงกัน 5 stroke แล้วถูกนับว่า "ผ่าน" ทั้งหมด
        stem = Path(s.video_path).stem
        for st in s.strokes:
            sid = stroke_key(_set_name(p), stem, st.stroke_no)
            out[sid] = [k not in fails[sid] for k in keyframes
                        if st.keyframes.get(k) is not None]
    # acceptance_n นับ keyframe ของเกณฑ์รับงานเต็ม (impact+backswing) เสมอ
    # จึงหารด้วยจำนวนของเกณฑ์เต็ม ไม่ใช่ของ subset ที่ขอมา
    n_expected = r[block]["acceptance_n"] // len(ACCEPTANCE_KEYFRAMES)
    # ค่าที่รายงานไว้เป็นของเกณฑ์เต็ม เทียบกับ subset ไม่ได้ -> คืน None ให้
    # ผู้เรียกรู้ว่าห้ามเอาไป cross-check
    reported = (r[block]["acceptance_accuracy"]
                if tuple(keyframes) == ACCEPTANCE_KEYFRAMES else None)
    return out, n_expected, reported


def ci(samples, lo=2.5, hi=97.5):
    return np.percentile(samples, lo), np.percentile(samples, hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "docs/benchmark_keyframe_results.json"))
    ap.add_argument("--dataset-root", default=str(ROOT / "dataset"))
    ap.add_argument("--clip-person", default=None,
                    help="clip_person.json สำหรับ bootstrap ระดับคน")
    ap.add_argument("--block", default="mode_a_phase",
                    choices=("mode_a_phase", "mode_b_phase", "mode_a", "mode_b"))
    ap.add_argument("--compare", default=None,
                    help="results JSON ของระบบเดิม สำหรับเทียบแบบจับคู่")
    ap.add_argument("--keyframes", default=",".join(ACCEPTANCE_KEYFRAMES),
                    help="จำกัดเฉพาะบาง keyframe (คั่นด้วยจุลภาค) — เช่น "
                         "--keyframes impact เพื่อดูเฉพาะส่วนที่การปรับด้วย"
                         "เสียงไปแตะจริง")
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    kfs = tuple(k.strip() for k in args.keyframes.split(",") if k.strip())
    strokes, n_expected, reported = load_stroke_results(
        Path(args.results), args.block, Path(args.dataset_root), kfs)
    ids = sorted(strokes)
    hits = np.array([sum(strokes[i]) for i in ids], dtype=float)
    tot = np.array([len(strokes[i]) for i in ids], dtype=float)
    point = hits.sum() / tot.sum()

    print(f"=== {args.block} · keyframe: {'+'.join(kfs)} ===")
    print(f"stroke ที่ให้คะแนนได้ {len(ids)} (คาดไว้ {n_expected}) · "
          f"keyframe {int(tot.sum())}")
    if reported is None:
        print(f"คำนวณจากรายลูก {point:.3f} (subset — ไม่มีค่าที่รายงานไว้ให้เทียบ)")
    else:
        print(f"ค่าที่รายงาน {reported:.3f} · คำนวณใหม่จากรายลูก {point:.3f}")
        if abs(point - reported) > 0.005:
            print("⚠️ ไม่ตรงกัน — ผลรายลูกอาจไม่ครบ ตัวเลขข้างล่างเชื่อไม่ได้")

    rng = np.random.default_rng(args.seed)
    n = len(ids)

    # ── ระดับลูก ──
    idx = rng.integers(0, n, size=(args.n, n))
    per_stroke = hits[idx].sum(1) / tot[idx].sum(1)

    groups = {"รายลูก (สมมติว่าอิสระ)": per_stroke}

    # ── ระดับคลิป ──
    by_clip = defaultdict(list)
    for i, sid in enumerate(ids):
        # ตัดเฉพาะส่วน "__sNN" ท้ายสุด — ห้ามสมมติว่าชื่อชุดมีชั้นเดียว เพราะ
        # คลิปใน sessions_deferred/ มีสองชั้น (sessions_deferred__set4__stem)
        by_clip["__".join(sid.split("__")[:-1])].append(i)
    clips = sorted(by_clip)
    if len(clips) > 1:
        cidx = rng.integers(0, len(clips), size=(args.n, len(clips)))
        vals = []
        for row in cidx:
            sel = np.concatenate([by_clip[clips[c]] for c in row])
            vals.append(hits[sel].sum() / tot[sel].sum())
        groups[f"รายคลิป ({len(clips)} คลิป)"] = np.array(vals)

    # ── ระดับคน ──
    by_person = defaultdict(list)
    if args.clip_person:
        mapping = json.loads(Path(args.clip_person).read_text(encoding="utf-8"))
        pass
        unknown = 0
        for clip, members in by_clip.items():
            stem = clip.split("__")[-1]
            person = next((v for k, v in mapping.items()
                           if Path(k).name == stem or k.endswith(stem)), None)
            if person is None:
                unknown += 1
                continue
            by_person[person] += members
        if unknown:
            print(f"⚠️ จับคู่คนไม่ได้ {unknown} คลิป")
        people = sorted(by_person)
        if len(people) > 1:
            pidx = rng.integers(0, len(people), size=(args.n, len(people)))
            vals = []
            for row in pidx:
                sel = np.concatenate([by_person[people[p]] for p in row])
                vals.append(hits[sel].sum() / tot[sel].sum())
            groups[f"รายคน ({len(people)} คน) ← ตัวที่ควรใช้"] = np.array(vals)

    print(f"\n{'หน่วยที่สุ่ม':<34s} {'ค่ากลาง':>8s} {'95% CI':>18s} {'ความกว้าง':>10s}")
    print("-" * 76)
    for name, v in groups.items():
        lo, hi = ci(v)
        print(f"{name:<34s} {np.median(v):8.3f}  [{lo:.3f}, {hi:.3f}] "
              f"{hi - lo:10.3f}")

    print("\n⚠️ ตัวเลขข้างบนคือความไม่แน่นอนของ **ค่าสัมบูรณ์** ซึ่งรวมคำถามว่า"
          "\n   'ผู้เล่น 7 คนนี้เป็นตัวแทนของประชากรแค่ไหน' เอาไปเทียบกับ"
          "\n   'ผลต่างระหว่างสองระบบ' ตรง ๆ ไม่ได้ — ความไม่แน่นอนส่วนนั้น"
          "\n   หักล้างกันไปเมื่อวัดสองระบบบน stroke ชุดเดียวกัน ใช้ --compare")

    # ── เทียบสองระบบแบบจับคู่ ──
    if args.compare:
        old, _, old_reported = load_stroke_results(
            Path(args.compare), args.block, Path(args.dataset_root), kfs)
        common = [i for i in ids if i in old]
        if len(common) != len(ids):
            print(f"\n⚠️ stroke ตรงกันแค่ {len(common)}/{len(ids)}")
        oh = np.array([sum(old[i]) for i in common], dtype=float)
        ot = np.array([len(old[i]) for i in common], dtype=float)
        nh = np.array([sum(strokes[i]) for i in common], dtype=float)
        nt = np.array([len(strokes[i]) for i in common], dtype=float)
        obs = nh.sum() / nt.sum() - oh.sum() / ot.sum()

        print(f"\n=== เทียบแบบจับคู่ (stroke ชุดเดียวกัน) ===")
        _rep = f" (รายงานไว้ {old_reported:.3f})" if old_reported is not None else ""
        print(f"เดิม {oh.sum()/ot.sum():.3f}{_rep} -> "
              f"ใหม่ {nh.sum()/nt.sum():.3f} · ผลต่าง {obs:+.3f}")
        up, down = int(np.sum(nh > oh)), int(np.sum(nh < oh))
        print(f"stroke ที่ผลเปลี่ยน {up + down}/{len(common)} "
              f"(ดีขึ้น {up} · แย่ลง {down})")

        # sign test — ถามคนละคำถามกับ CI: "ทิศทางของการเปลี่ยนแปลงเอนไปทางดี
        # ขึ้นเกินกว่าที่การโยนเหรียญจะอธิบายได้ไหม" ไวกว่าเพราะไม่สนขนาด
        # ⚠️ สมมติว่าแต่ละ stroke อิสระต่อกัน ซึ่งไม่จริง (ลูกในคลิปเดียวกัน
        # สัมพันธ์กัน) จึงมองโลกสวยเกินไป ใช้ประกอบ ไม่ใช่ใช้แทน CI
        if up + down:
            from math import comb
            m = up + down
            p = sum(comb(m, k) for k in range(up, m + 1)) / 2 ** m
            print(f"sign test (สมมติว่าลูกอิสระกัน): p = {p:.4f}"
                  f"{'  ← ทิศทางชัด' if p < 0.05 else ''}")

        for name, gidx in (("รายลูก", [[i] for i in range(len(common))]),
                           ("รายคลิป", None), ("รายคน", None)):
            if name == "รายคลิป":
                gidx = [[common.index(ids[i]) for i in v if ids[i] in common]
                        for v in by_clip.values()]
            elif name == "รายคน":
                if not args.clip_person:
                    continue
                gidx = [[common.index(ids[i]) for i in v if ids[i] in common]
                        for v in by_person.values()] if by_person else None
            if not gidx:
                continue
            gidx = [g for g in gidx if g]
            g_arr = np.array(gidx, dtype=object)
            sel = rng.integers(0, len(g_arr), size=(args.n, len(g_arr)))
            diffs = []
            for row in sel:
                s = np.concatenate([g_arr[j] for j in row]).astype(int)
                diffs.append(nh[s].sum() / nt[s].sum()
                             - oh[s].sum() / ot[s].sum())
            diffs = np.array(diffs)
            lo, hi = ci(diffs)
            sig = "✅ ไม่คร่อมศูนย์" if lo > 0 or hi < 0 else "⚠️ คร่อมศูนย์"
            print(f"  {name:<8s} ผลต่าง {np.median(diffs):+.3f} "
                  f"95% CI [{lo:+.3f}, {hi:+.3f}]  {sig}")


if __name__ == "__main__":
    main()
