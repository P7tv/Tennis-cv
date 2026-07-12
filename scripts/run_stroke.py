#!/usr/bin/env python3
"""Phase 1 CLI: คลิป single stroke → JSON (17-layer schema)

ตัวอย่าง:
    python scripts/run_stroke.py clip.mp4 --stroke FH --height-cm 172 -o out.json
    python scripts/run_stroke.py clip.mp4 --stroke SV --gas-url https://script.google.com/...
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loeuf_cv import PipelineConfig, StrokePipeline
from loeuf_cv.config import STROKE_TYPES
from loeuf_cv.gas_client import save_json


def main():
    ap = argparse.ArgumentParser(description="Loeuf CV — single stroke inference")
    ap.add_argument("video", help="path ของคลิป single stroke")
    ap.add_argument("--stroke", required=True, choices=STROKE_TYPES,
                    help="stroke type (ระบุมาล่วงหน้า Phase 1)")
    ap.add_argument("--side", default="right", choices=("right", "left"),
                    dest="dominant_side", help="มือถนัด (A6)")
    ap.add_argument("--height-cm", type=float, default=None,
                    help="ส่วนสูงผู้เล่น (C1 calibration px→cm)")
    ap.add_argument("--auto-height", action="store_true",
                    help="⚠️ experimental: auto-detect ส่วนสูงจากเส้นคอร์ท"
                        " (ยังไม่เสถียรพอสำหรับ production — ดู README)")
    ap.add_argument("--session-id", default=None, help="MT2 (default: auto)")
    ap.add_argument("-o", "--output", default=None)
    ap.add_argument("--gas-url", default=None)
    args = ap.parse_args()

    config = PipelineConfig(
        dominant_side=args.dominant_side,
        subject_height_cm=args.height_cm,
        auto_height_from_court=args.auto_height,
        session_id=args.session_id,
        gas_endpoint_url=args.gas_url,
    )
    pipeline = StrokePipeline(config)

    def progress(i, total):
        if total and i % 30 == 0:
            print(f"\r  pose extraction: {i}/{total} frames", end="", flush=True)

    print(f"Processing {args.video} [{args.stroke}] ...")
    result = pipeline.process_video(
        args.video, args.stroke,
        send_to_gas=bool(args.gas_url), progress_callback=progress)
    print()

    out_path = args.output or str(Path(args.video).with_suffix(".json"))
    save_json(result, out_path)

    stroke = result["strokes"][0]
    root, cf = stroke["stroke_root"], stroke["confidence_flag"]
    detected = [k for k, v in stroke["keyframes"].items() if v["detected"]]
    print(f"  detection: {root['detection_status']}"
          f" | clean: {root['is_clean_stroke']}"
          f" | CF1: {cf['pose_estimation_confidence']}"
          f" → {root['stroke_recommended_action']}")
    print(f"  keyframes: {len(detected)}/5 — {', '.join(detected)}")
    if not config.height_calibrated:
        print("  ⚠️ ไม่ได้ระบุ --height-cm → ใช้ 170cm default"
              " (cm fields = low_confidence)")
    if "gas_delivery" in result:
        print(f"  GAS: {'OK' if result['gas_delivery']['ok'] else 'FAILED'}")
    print(f"  output → {out_path}")


if __name__ == "__main__":
    main()
