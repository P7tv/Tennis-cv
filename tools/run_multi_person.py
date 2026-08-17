#!/usr/bin/env python3
"""Multi-person extraction CLI: แยกผู้เล่นหลายคนในคลิปเดียว

แก้ปัญหา (1) หลายคนในเฟรม (2) คนไกล/ตัวเล็ก — ด้วย background-subtraction
blob tracking + crop-zoom ก่อนส่งเข้า MediaPipe Pose (ดู multi_person.py)

ตัวอย่าง:
    python scripts/run_multi_person.py clip.mp4 --max-players 2
    python scripts/run_multi_person.py clip.mp4 --role near --stroke FH -o out.json
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from loeuf_cv import PipelineConfig
from loeuf_cv.config import CORE_LANDMARKS, STROKE_TYPES
from loeuf_cv.multi_person import extract_multi_person


def main():
    ap = argparse.ArgumentParser(description="Loeuf CV — multi-person extraction")
    ap.add_argument("video")
    ap.add_argument("--max-players", type=int, default=2)
    ap.add_argument("--role", default=None, choices=("near", "far", "other"),
                    help="ถ้าระบุ: ส่ง track นี้เข้า Phase 1 stroke pipeline ต่อ")
    ap.add_argument("--stroke", default=None, choices=STROKE_TYPES,
                    help="stroke type (ใช้คู่กับ --role เพื่อรัน stroke pipeline)")
    ap.add_argument("--height-cm", type=float, default=None)
    ap.add_argument("--auto-height", action="store_true",
                    help="ลอง auto-detect ส่วนสูงจากเส้นคอร์ท (court_calibration.py)")
    ap.add_argument("-o", "--output", default=None)
    args = ap.parse_args()

    config = PipelineConfig(subject_height_cm=args.height_cm,
                            auto_height_from_court=args.auto_height)

    def progress(i, total):
        if total and i % 60 == 0:
            print(f"\r  extracting: {i}/{total} frames", end="", flush=True)

    print(f"Processing {args.video} (max {args.max_players} players) ...")
    tracks = extract_multi_person(args.video, config, max_players=args.max_players,
                                  progress_callback=progress)
    print()

    if not tracks:
        print("  ไม่พบ track ที่ใช้ได้เลย — ลองลด MIN_TRACK_FRAMES"
              " หรือเช็คว่ากล้องนิ่งจริงไหม")
        return

    print(f"  พบ {len(tracks)} track:")
    for t in tracks:
        vis = t.pose.visibility[:, list(CORE_LANDMARKS)]
        valid = vis.mean(axis=1) > 0
        mean_vis = float(vis[valid].mean()) if valid.any() else 0.0
        coverage = t.n_frames_tracked / t.pose.n_frames * 100
        print(f"    [{t.role:5s}] track_id={t.track_id}"
              f" | tracked {t.n_frames_tracked}/{t.pose.n_frames} เฟรม"
              f" ({coverage:.1f}%) | mean visibility={mean_vis:.3f}")

    if args.role:
        target = next((t for t in tracks if t.role == args.role), None)
        if target is None:
            print(f"  ไม่พบ track role={args.role}")
            return
        if args.stroke:
            from loeuf_cv import StrokePipeline
            from loeuf_cv.gas_client import save_json
            result = StrokePipeline(config).process_timeseries(
                target.pose, args.stroke)
            out_path = args.output or str(Path(args.video).with_suffix(
                f".{args.role}.json"))
            save_json(result, out_path)
            root = result["stroke_root"]
            print(f"  stroke pipeline [{args.role}]: {root['detection_status']}"
                 f" → {root['stroke_recommended_action']}")
            print(f"  output → {out_path}")


if __name__ == "__main__":
    main()
