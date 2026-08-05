"""เสียงช่วยระบุ "เฟรมปะทะ" ได้แม่นกว่าภาพหรือเปล่า — วัดกับเฉลย 172 ลูกที่มีอยู่

คำถามที่ตอบ (ไม่ต้องสร้างเฉลยใหม่เลย)

  Q1  ถ้าให้หน้าต่างจากภาพมาแล้ว เสียงเลือกเฟรมได้ใกล้เฉลยกว่าไหม
      เทียบกับ 3 ตัว: กึ่งกลางหน้าต่าง (ไม่ใช้เสียง) · เฟรมที่ระบบทำนายจริง
      · และเฉลยเอง (เพดาน)

  Q2  ความคลาดของเสียงเป็นค่าคงที่ต่อคลิปหรือกระจายมั่ว
      ถ้าเป็นค่าคงที่ = เสียงเดินทางช้า แก้ได้ด้วยการหักออก (ดู sound_delay_frames)
      ถ้ากระจายมั่ว = เสียงที่จับได้ไม่ใช่เสียงตีจริง

  Q3  แต่ละคลิปให้ผลต่างกันแค่ไหน — คลิปที่มีคนป้อนลูกน่าจะแย่กว่า
      เพราะเสียงตีของอีกฝั่งปนเข้ามา

⚠️ นี่คือการวัดว่าเสียง "ช่วยเลือกเฟรม" ได้ไหม ไม่ใช่ "หาการตีเจอเองไหม" —
   เสียงอย่างเดียวแยกไม่ออกว่าใครตี จึงตั้งใจให้ภาพจำกัดหน้าต่างก่อนเสมอ

รัน:  python scripts/eval_audio_onset.py
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loeuf_cv.audio_onset import (best_onset_near,  # noqa: E402
                                  extract_audio, onset_envelope)
from train_model.label_ingest import (find_session_labels,  # noqa: E402
                                      load_session_label)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default=str(ROOT / "dataset"))
    ap.add_argument("--audio-dir", default=None,
                    help="ที่เก็บ wav ที่ดึงมา (default = ข้างใต้ temp ของระบบ)")
    ap.add_argument("--window", type=float, default=6.0,
                    help="ครึ่งความกว้างหน้าต่างที่ให้เสียงค้นหา (เฟรม) — "
                         "ควรใกล้เคียงความคลาดของภาพ ตอนนี้ p90 = 7 เฟรม")
    ap.add_argument("--preds-dir", default=None,
                    help="โฟลเดอร์ preds/mode_a — เปิดการทดสอบที่ใช้หน้าต่าง"
                         "จากคำทำนายจริง (ตัวเลขที่เอาไปใช้ได้)")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    audio_dir = Path(args.audio_dir) if args.audio_dir else \
        Path(__file__).resolve().parent.parent / ".audio_cache"

    rows = []
    per_clip = []
    for p in sorted(find_session_labels(Path(args.dataset_root))):
        try:
            s = load_session_label(p)
        except Exception as e:
            print(f"  ข้าม {p.name}: {e}")
            continue
        video = Path(s.video_path)
        if not video.is_file():
            print(f"  ข้าม {video.name}: ไม่พบไฟล์วิดีโอ")
            continue

        wav = audio_dir / f"{video.stem}.wav"
        if not wav.is_file() and not extract_audio(video, wav):
            print(f"  ข้าม {video.stem}: ดึงเสียงไม่ได้ (อาจไม่มีแทร็กเสียง)")
            continue

        gt = [(st.stroke_no, st.keyframes["impact"], st.fps)
              for st in s.strokes if st.keyframes.get("impact") is not None]
        if not gt:
            continue
        fps = gt[0][2] or 29.97

        t, env = onset_envelope(wav)
        deltas = []
        for _, g, _ in gt:
            f_audio, strength = best_onset_near(t, env, g, fps, args.window)
            if f_audio is None:
                continue
            deltas.append(f_audio - g)
            rows.append({"clip": video.stem, "gt": g, "audio": f_audio,
                         "delta": f_audio - g, "strength": strength})
        if deltas:
            d = np.array(deltas)
            per_clip.append((video.stem, len(d), np.median(d),
                             np.std(d - np.median(d)),
                             float(np.mean(np.abs(d - np.median(d)) <= 1))))
            print(f"  {video.stem:24s} {len(d):3d} ลูก · "
                  f"ค่ากลาง {np.median(d):+5.1f} เฟรม · "
                  f"กระจาย {np.std(d - np.median(d)):4.1f}")
        if args.limit and len(per_clip) >= args.limit:
            break

    if not rows:
        print("ไม่มีข้อมูล")
        return

    d = np.array([r["delta"] for r in rows], dtype=float)
    print(f"\n=== รวม {len(d)} ลูก จาก {len(per_clip)} คลิป ===")
    print(f"หน้าต่างที่ให้ค้นหา: ±{args.window:.0f} เฟรม\n")

    print("Q2 — ความคลาดเป็นค่าคงที่ต่อคลิปหรือกระจายมั่ว")
    print(f"  ก่อนหักค่ากลางรายคลิป: |Δ| median {np.median(np.abs(d)):.1f} "
          f"เฟรม · ภายใน 1 เฟรม {np.mean(np.abs(d) <= 1):.1%}")
    # หักค่ากลางของแต่ละคลิปออก = จำลองการชดเชยเวลาเดินทางของเสียง
    corrected = []
    for name, n, med, sd, _ in per_clip:
        sub = np.array([r["delta"] for r in rows if r["clip"] == name])
        corrected.append(sub - med)
    c = np.concatenate(corrected)
    print(f"  หลังหักค่ากลางรายคลิป:  |Δ| median {np.median(np.abs(c)):.1f} "
          f"เฟรม · ภายใน 1 เฟรม {np.mean(np.abs(c) <= 1):.1%} "
          f"· ภายใน 2 เฟรม {np.mean(np.abs(c) <= 2):.1%}")

    print("\n  ค่ากลางรายคลิป (ถ้าเป็นเวลาเดินทางของเสียงจริง ควรเป็นบวกทุกคลิป")
    print("  และแปรตามระยะห่างจากกล้อง — 10 m = 0.9 เฟรม, 20 m = 1.8 เฟรม):")
    for name, n, med, sd, within1 in sorted(per_clip, key=lambda x: x[2]):
        print(f"    {name:24s} {n:3d} ลูก  ค่ากลาง {med:+5.1f}  กระจาย {sd:4.1f}")

    print("\n⚠️ ตัวเลขข้างบนให้หน้าต่างค้นหาที่มีจุดกึ่งกลางเป็น **เฉลย** ซึ่งของจริง"
          "\n   ไม่มี = เป็นเพดานบน ไม่ใช่ผลที่ใช้ได้จริง ใช้ --preds-dir เพื่อวัด"
          "\n   ด้วยหน้าต่างที่มาจากเฟรมที่ระบบทำนายเอง")

    if args.preds_dir:
        evaluate_with_predictions(Path(args.preds_dir), Path(args.dataset_root),
                                  audio_dir, args.window)


def evaluate_with_predictions(preds_dir: Path, dataset_root: Path,
                              audio_dir: Path, window: float):
    """การทดสอบที่ใช้ได้จริง — หน้าต่างมาจากเฟรมที่ระบบทำนาย ไม่ใช่จากเฉลย

    🔴 การชดเชยเวลาเดินทางของเสียงก็ต้องทำโดยไม่ใช้เฉลยด้วย ไม่งั้นก็ยังโกงอยู่
    วิธีที่ใช้: ค่าชดเชยของคลิป = ค่ากลางของ (เฟรมที่เสียงเลือก - เฟรมที่ภาพทำนาย)
    ของทุกลูกในคลิปนั้น — ใช้ได้เพราะความคลาดของภาพไม่มีทิศทางชัด (median +0.5)
    ส่วนความช้าของเสียงเป็นค่าคงที่ทิศเดียว จึงแยกออกจากกันได้
    """
    MATCH_TOL = 10
    rows = []
    for p in sorted(find_session_labels(dataset_root)):
        try:
            s = load_session_label(p)
        except Exception:
            continue
        video = Path(s.video_path)
        wav = audio_dir / f"{video.stem}.wav"
        if not wav.is_file():
            continue
        cand = list(preds_dir.rglob(f"{video.stem}.json"))
        if not cand:
            continue
        pred = json.loads(cand[0].read_text(encoding="utf-8"))
        pf = sorted(st["keyframe"]["impact"]["frame_index"]
                    for st in pred["strokes"]
                    if st.get("keyframe", {}).get("impact", {}).get("detected"))
        gt = [(st.keyframes["impact"], st.fps) for st in s.strokes
              if st.keyframes.get("impact") is not None]
        if not gt or not pf:
            continue
        fps = gt[0][1] or 29.97
        t, env = onset_envelope(wav)

        used = set()
        for g, _ in gt:
            near = [f for f in pf if abs(f - g) <= MATCH_TOL and f not in used]
            if not near:
                continue                    # ระบบหาไม่เจอ — คนละปัญหา
            p_frame = min(near, key=lambda f: abs(f - g))
            used.add(p_frame)
            a_frame, _ = best_onset_near(t, env, p_frame, fps, window)
            if a_frame is None:
                continue
            rows.append({"clip": video.stem, "gt": g, "pred": p_frame,
                         "audio": a_frame})

    if not rows:
        print("\n(ไม่มีคำทำนายให้เทียบ)")
        return

    clips = sorted({r["clip"] for r in rows})
    pred_err, audio_err = [], []
    for c in clips:
        sub = [r for r in rows if r["clip"] == c]
        # ค่าชดเชยของคลิป คำนวณจากคำทำนาย ไม่แตะเฉลย
        off = float(np.median([r["audio"] - r["pred"] for r in sub]))
        for r in sub:
            pred_err.append(r["pred"] - r["gt"])
            audio_err.append(r["audio"] - off - r["gt"])
    pe, ae = np.abs(np.array(pred_err)), np.abs(np.array(audio_err))

    print(f"\n=== การทดสอบที่ใช้ได้จริง: หน้าต่างจากคำทำนาย ({len(pe)} ลูก) ===")
    print(f"{'':<22s} {'|Δ| median':>11s} {'ภายใน 1':>9s} {'ภายใน 2':>9s} "
          f"{'ภายใน 4':>9s}")
    print("-" * 64)
    for name, v in (("ภาพอย่างเดียว (เดิม)", pe), ("ภาพ + เสียง", ae)):
        print(f"{name:<22s} {np.median(v):11.1f} {np.mean(v<=1):9.1%} "
              f"{np.mean(v<=2):9.1%} {np.mean(v<=4):9.1%}")
    print(f"\nดีขึ้น {np.mean(ae<=2) - np.mean(pe<=2):+.1%} จุด (ภายใน 2 เฟรม) · "
          f"{np.mean(ae<=4) - np.mean(pe<=4):+.1%} จุด (ภายใน 4 เฟรม)")
    print(f"แย่ลง {np.mean(ae > pe):.1%} ของลูก")


if __name__ == "__main__":
    main()
