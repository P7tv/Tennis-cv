"""เทรนโมเดลใหม่จากคลิปที่เพิ่มเข้ามา — คำสั่งเดียวจบ

สำหรับลูกค้า: วางไฟล์ใหม่ลงโฟลเดอร์ตามโครงสร้างด้านล่าง แล้วรัน

    python scripts/retrain_with_new_data.py

    dataset/sessions/<ชื่อชุด>/
        IMG_1234.MOV                                     <- วิดีโอ
        IMG_1234_stroke_labels_<คนlabel>_<วันที่>.json     <- ไฟล์ label

ทำอะไรบ้าง:
    1. ตรวจไฟล์ label ทุกไฟล์ก่อน (พังตรงไหนบอกตรงนั้น ไม่ปล่อยไปพังตอนเทรน)
    2. รัน tracking เฉพาะคลิปใหม่ (คลิปเก่าใช้ cache — ข้ามอัตโนมัติ)
    3. เทรน hit + stroke classifier ใหม่ พร้อม data augmentation
    4. ⭐ เทียบกับโมเดลเดิมด้วย Leave-One-Person-Out — **ถ้าแย่ลงจะไม่เขียนทับ**

ข้อ 4 สำคัญที่สุด: ข้อมูลเพิ่มไม่ได้แปลว่าดีขึ้นเสมอไป (label ผิด / คลิปคนละมุม
กล้อง / ผู้เล่นที่ท่าต่างจากเดิมมาก ทำให้แย่ลงได้) สคริปต์นี้จึงวัดก่อนเขียนทับ
เสมอ ถ้าอยากเขียนทับทั้งที่แย่ลงต้องใส่ --force เอง

⚠️ Leave-One-Person-Out = วัดโดยตัดผู้เล่นออกทีละคน แล้วทดสอบกับคนที่ตัดออก
   เป็นตัวเลขเดียวที่บอกได้ว่า "ใช้กับผู้เล่นคนใหม่แล้วจะเป็นยังไง" —
   ห้ามใช้ตัวเลขที่วัดจากคลิปที่โมเดลเคยเห็น เพราะมันจำได้ (สูงเกินจริง ~30%)
"""
import argparse
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train_model.label_ingest import (find_session_labels,  # noqa: E402
                                      load_session_label)

PY = sys.executable
REQUIRED_KEYFRAMES = ("impact", "backswing_peak")
KNOWN_TYPES = ("FH", "BH", "SV", "VL", "SL", "RS")


# ---------------------------------------------------------------------------
# 1. ตรวจไฟล์ label
# ---------------------------------------------------------------------------

def validate(dataset_root: Path) -> tuple[list, list[str]]:
    """คืน (sessions ที่ใช้ได้, รายการปัญหา)

    ตรวจสิ่งที่เคยพังจริงในชุดที่ผ่านมา:
      - clip_path ใน JSON ชี้ผิดไฟล์ (เจอ 6 ไฟล์)
      - video_id เสียหายจากการ copy-paste (เจอ 10 ไฟล์) -> label_ingest ไม่ใช้
      - stroke ที่ไม่มี impact/backswing = ใช้เทรนไม่ได้
      - stroke_type ที่สะกดผิด/ไม่รู้จัก
    """
    problems: list[str] = []
    sessions = []
    label_files = list(find_session_labels(dataset_root))
    if not label_files:
        problems.append(
            f"ไม่พบไฟล์ label เลยใน {dataset_root} — ตรวจว่าวางไฟล์ถูกที่ไหม "
            f"(ต้องเป็น dataset/sessions/<ชุด>/<วิดีโอ>_stroke_labels_*.json)")
        return sessions, problems

    for p in label_files:
        rel = p.relative_to(dataset_root) if p.is_relative_to(dataset_root) else p
        try:
            s = load_session_label(p)
        except Exception as e:
            problems.append(f"[อ่านไม่ได้] {rel}: {e}")
            continue

        if not s.video_path.exists():
            problems.append(
                f"[ไม่เจอวิดีโอ] {rel} -> ชี้ไปที่ {s.video_path.name} "
                f"ซึ่งไม่มีอยู่จริง (วางวิดีโอไว้โฟลเดอร์เดียวกับ label)")
            continue

        bad = [st.stroke_no for st in s.strokes
               if any(st.keyframes.get(k) is None for k in REQUIRED_KEYFRAMES)]
        if bad:
            problems.append(
                f"[keyframe ไม่ครบ] {rel}: stroke {bad} ขาด impact หรือ "
                f"backswing_peak — stroke พวกนี้จะถูกข้ามตอนเทรน")

        unknown = sorted({st.stroke_type for st in s.strokes
                          if st.stroke_type not in KNOWN_TYPES})
        if unknown:
            problems.append(f"[ประเภทท่าไม่รู้จัก] {rel}: {unknown} "
                            f"(ที่รองรับ: {', '.join(KNOWN_TYPES)})")

        no_player = [st.stroke_no for st in s.strokes if not st.player_id]
        if no_player:
            problems.append(
                f"[ไม่มี player_id] {rel}: stroke {no_player} — จำเป็นมาก "
                f"เพราะการวัดผลตัดข้อมูลเป็นราย 'คน' ไม่ใช่รายคลิป")

        sessions.append((p, s))
    return sessions, problems


def summarize(sessions) -> None:
    by_person = Counter()
    by_type = Counter()
    people_per_type = {}
    n_strokes = 0
    for _, s in sessions:
        for st in s.strokes:
            n_strokes += 1
            by_person[st.player_id] += 1
            by_type[st.stroke_type] += 1
            people_per_type.setdefault(st.stroke_type, set()).add(st.player_id)

    print(f"\n  คลิป {len(sessions)} · stroke {n_strokes} · "
          f"ผู้เล่น {len(by_person)} คน")
    print(f"\n  {'ท่า':<6}{'stroke':>8}{'จำนวนคน':>10}   หมายเหตุ")
    print("  " + "-" * 52)
    for t, n in by_type.most_common():
        k = len(people_per_type.get(t, ()))
        note = "⚠️ น้อยเกินจะวัดข้ามคนได้" if k <= 2 else ""
        print(f"  {t:<6}{n:>8}{k:>10}   {note}")

    thin = [t for t, ppl in people_per_type.items() if len(ppl) <= 2]
    if thin:
        print(f"\n  ⚠️ ท่า {', '.join(sorted(thin))} มาจากผู้เล่น <= 2 คน")
        print("     โมเดลจะยังใช้กับผู้เล่นคนใหม่ได้ไม่ดี ต่อให้เพิ่มคลิปของคนเดิม")
        print("     -> ต้องเพิ่ม **ผู้เล่นคนใหม่** ไม่ใช่เพิ่มคลิป")


# ---------------------------------------------------------------------------
# 2-3. tracking + เทรน
# ---------------------------------------------------------------------------

def run(cmd: list[str], title: str) -> str:
    print(f"\n=== {title} ===")
    print("  $ " + " ".join(str(c) for c in cmd))
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    out = (r.stdout or "") + (r.stderr or "")
    if r.returncode != 0:
        print(out[-3000:])
        raise SystemExit(f"\n❌ ขั้นตอน '{title}' ล้มเหลว (exit {r.returncode})")
    return out


_F1_RE = re.compile(r"F1 ดีสุดที่ threshold ([\d.]+) — recall ([\d.]+)% "
                    r"precision ([\d.]+)%")
_ACC_RE = re.compile(r"ML ใหม่ \(LOPO\) — ความแม่นรวม ([\d.]+)")


def parse_hit_score(out: str):
    m = _F1_RE.search(out)
    if not m:
        return None
    th, rec, pre = float(m.group(1)), float(m.group(2)), float(m.group(3))
    f1 = 2 * rec * pre / max(1e-9, rec + pre) / 100.0
    return {"threshold": th, "recall": rec, "precision": pre, "f1": round(f1, 3)}


def parse_stroke_score(out: str):
    m = _ACC_RE.search(out)
    return {"accuracy": float(m.group(1))} if m else None


def main():
    ap = argparse.ArgumentParser(
        description="เทรนโมเดลใหม่จากคลิปที่เพิ่มเข้ามา (คำสั่งเดียวจบ)")
    ap.add_argument("--dataset-root", default=str(ROOT / "dataset"))
    ap.add_argument("--n-aug", type=int, default=6,
                    help="จำนวนสำเนาสังเคราะห์ต่อคลิป (0 = ปิด)")
    ap.add_argument("--skip-tracking", action="store_true",
                    help="ข้ามการรัน YOLO (ใช้เมื่อ cache ครบแล้ว)")
    ap.add_argument("--force", action="store_true",
                    help="เขียนทับโมเดลแม้ผลจะแย่ลง (ปกติจะไม่เขียนทับ)")
    ap.add_argument("--validate-only", action="store_true",
                    help="ตรวจไฟล์อย่างเดียว ไม่เทรน")
    args = ap.parse_args()

    root = Path(args.dataset_root)
    print("=" * 60)
    print("  เทรนโมเดลใหม่จากคลิปที่เพิ่มเข้ามา")
    print("=" * 60)

    # ── 1. ตรวจไฟล์ ──
    print("\n=== ตรวจไฟล์ label ===")
    sessions, problems = validate(root)
    for p in problems:
        print(f"  {p}")
    if not sessions:
        raise SystemExit("\n❌ ไม่มีคลิปที่ใช้ได้เลย — แก้ปัญหาข้างบนก่อน")
    if problems:
        print(f"\n  ⚠️ มี {len(problems)} จุดที่ต้องดู "
              f"(ไม่ถึงกับหยุด แต่ข้อมูลส่วนนั้นจะไม่ถูกใช้)")
    else:
        print("  ✅ ไฟล์ label ผ่านทั้งหมด")
    summarize(sessions)

    if args.validate_only:
        return

    # ── 2. tracking (ข้ามคลิปที่เคยทำแล้วอัตโนมัติ) ──
    if not args.skip_tracking:
        run([PY, "scripts/run_keyframe_benchmark.py", "track",
             "--dataset-root", str(root)],
            "รัน tracking คลิปใหม่ (ขั้นนี้ช้าที่สุด ~7 นาที/นาทีวิดีโอ)")
    else:
        print("\n(ข้าม tracking ตามที่สั่ง)")

    # ── 3. เทรน + วัดผล ──
    results = {}
    for name, cmd, parser in (
        ("hit", [PY, "scripts/train_hit_classifier_from_cache.py",
                 "--dataset-root", str(root), "--n-aug", str(args.n_aug)],
         parse_hit_score),
        ("stroke", [PY, "scripts/train_stroke_classifier_from_cache.py",
                    "--dataset-root", str(root)], parse_stroke_score),
    ):
        out = run(cmd, f"เทรน {name} classifier (ยังไม่เขียนทับ)")
        score = parser(out)
        results[name] = score
        print(f"  ผลใหม่: {score}")

    # ── 4. ประตูความปลอดภัย ──
    hist_path = ROOT / "dataset/training/model_scores.json"
    history = json.loads(hist_path.read_text(encoding="utf-8")) \
        if hist_path.exists() else {}

    print("\n" + "=" * 60)
    print("  เทียบกับครั้งก่อน (Leave-One-Person-Out)")
    print("=" * 60)

    decisions = {}
    for name, score in results.items():
        prev = history.get(name)
        key = "f1" if name == "hit" else "accuracy"
        if score is None:
            print(f"  {name:8s} อ่านผลไม่ได้ -> ไม่เขียนทับ")
            decisions[name] = False
            continue
        if prev is None:
            print(f"  {name:8s} ไม่มีผลครั้งก่อนให้เทียบ ({key}={score[key]}) "
                  f"-> เขียนทับ")
            decisions[name] = True
            continue
        old, new = prev.get(key), score[key]
        delta = new - old
        mark = "✅ ดีขึ้น" if delta > 0 else ("= เท่าเดิม" if delta == 0
                                             else "❌ แย่ลง")
        print(f"  {name:8s} {key}: {old} -> {new}  ({delta:+.3f})  {mark}")
        decisions[name] = delta >= 0 or args.force

    if args.force:
        print("\n  ⚠️ --force: จะเขียนทับทุกตัวไม่ว่าผลจะเป็นยังไง")

    # ── 5. เขียนทับเฉพาะตัวที่ผ่าน ──
    print("\n" + "=" * 60)
    for name, ok in decisions.items():
        if not ok:
            print(f"  {name:8s} ❌ ไม่เขียนทับ — โมเดลเดิมยังใช้อยู่")
            print(f"           (ถ้ายืนยันจะเขียนทับ ให้รันซ้ำด้วย --force)")
            continue
        cmd = ([PY, "scripts/train_hit_classifier_from_cache.py",
                "--dataset-root", str(root), "--n-aug", str(args.n_aug), "--save"]
               if name == "hit" else
               [PY, "scripts/train_stroke_classifier_from_cache.py",
                "--dataset-root", str(root), "--save"])
        run(cmd, f"บันทึกโมเดล {name}")
        history[name] = results[name]
        print(f"  {name:8s} ✅ เขียนทับแล้ว")

    hist_path.parent.mkdir(parents=True, exist_ok=True)
    hist_path.write_text(json.dumps(history, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    print(f"\nบันทึกประวัติคะแนน -> {hist_path}")
    print("\nเสร็จแล้ว ✅")


if __name__ == "__main__":
    main()
