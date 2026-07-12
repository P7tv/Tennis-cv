#!/usr/bin/env python3
"""Phase 2 CLI: วิดีโอ session เต็ม → JSON (strokes + AGG/TRE/PAT/SUM)

ตัวอย่าง:
    python scripts/run_session.py match.mp4 --height-cm 172 -o session.json
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loeuf_cv import PipelineConfig, SessionPipeline
from loeuf_cv.gas_client import save_json


def main():
    ap = argparse.ArgumentParser(description="Loeuf CV — full session inference")
    ap.add_argument("video")
    ap.add_argument("--side", default="right", choices=("right", "left"),
                    dest="dominant_side")
    ap.add_argument("--height-cm", type=float, default=None)
    ap.add_argument("--session-id", default=None)
    ap.add_argument("-o", "--output", default=None)
    ap.add_argument("--gas-url", default=None)
    args = ap.parse_args()

    config = PipelineConfig(
        dominant_side=args.dominant_side,
        subject_height_cm=args.height_cm,
        session_id=args.session_id,
        gas_endpoint_url=args.gas_url,
    )
    pipeline = SessionPipeline(config)

    def progress(i, total):
        if total and i % 60 == 0:
            print(f"\r  pose extraction: {i}/{total} frames", end="", flush=True)

    print(f"Processing session {args.video} ...")
    result = pipeline.process_video(
        args.video, send_to_gas=bool(args.gas_url),
        progress_callback=progress)
    print()

    out_path = args.output or str(Path(args.video).with_suffix(".session.json"))
    save_json(result, out_path)

    mt = result["session_metadata"]
    summary = result["session_summary"]
    print(f"  strokes: {mt['usable_strokes']} usable"
          f" / {mt['total_strokes_detected']} detected")
    print(f"  distribution: {mt['stroke_type_distribution']}")
    print(f"  session_quality: {summary['session_quality']}"
          f" | fatigue: {result['trend']['fatigue_flag']}"
          f" | coach_flag: {summary['coach_flag']}")
    print(f"  output → {out_path}")


if __name__ == "__main__":
    main()
