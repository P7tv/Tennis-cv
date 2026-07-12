#!/usr/bin/env python3
"""ตัดเฟรมรอบ impact จากทุกคลิป → dataset สำหรับ label ลูกเทนนิส

ใช้ผล impact จาก batch outputs → เซฟเฟรม ±N รอบจุดนั้นเป็น JPG
(เป็น dataset prep สำหรับ train ball detector บน VM)

ตัวอย่าง:
    python scripts/run_batch.py dataset/ -o outputs/
    python scripts/extract_impact_frames.py dataset/ outputs/ \\
        -o dataset/ball_frames --window 10 --step 2
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_root")
    ap.add_argument("outputs_dir", help="โฟลเดอร์ JSON จาก run_batch")
    ap.add_argument("-o", "--out-dir", default="dataset/ball_frames")
    ap.add_argument("--window", type=int, default=10,
                    help="±N เฟรม (ต้นทาง) รอบ impact")
    ap.add_argument("--step", type=int, default=2)
    args = ap.parse_args()

    root, outputs = Path(args.dataset_root), Path(args.outputs_dir)
    out_dir = Path(args.out_dir)
    n_clips = n_frames = 0

    for jf in sorted(outputs.rglob("*.json")):
        if jf.name in ("batch_summary.json",):
            continue
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
            stroke = data["strokes"][0]
        except (json.JSONDecodeError, KeyError, IndexError):
            continue
        imp = stroke["keyframes"]["impact"]
        if not imp["detected"]:
            continue

        rel = jf.relative_to(outputs).with_suffix("")
        video = None
        for ext in VIDEO_EXTS:
            cand = root / rel.parent / (rel.name + ext)
            if cand.exists():
                video = cand
                break
        if video is None:
            continue

        cap = cv2.VideoCapture(str(video))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # impact timestamp (ms) → เฟรมต้นทาง (output อาจถูก resample เป็น 60fps)
        impact_frame = round(imp["timestamp_ms"] * fps / 1000.0)

        dest = out_dir / rel.parent / rel.name
        dest.mkdir(parents=True, exist_ok=True)
        for f in range(impact_frame - args.window,
                       impact_frame + args.window + 1, args.step):
            if f < 0 or f >= total:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, frame = cap.read()
            if ok:
                cv2.imwrite(str(dest / f"f{f:05d}.jpg"), frame,
                            [cv2.IMWRITE_JPEG_QUALITY, 92])
                n_frames += 1
        cap.release()
        n_clips += 1

    print(f"extracted {n_frames} frames from {n_clips} clips → {out_dir}")
    print("ขั้นถัดไป: label ตำแหน่งลูก (bounding box) → train detector บน VM (T4)")


if __name__ == "__main__":
    main()
