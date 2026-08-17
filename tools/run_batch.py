#!/usr/bin/env python3
"""Batch runner: ประมวลผลทั้ง dataset (300 คลิป) แบบขนาน + resume ได้

โครง dataset ที่คาดหวัง (ตาม client dependency):
    dataset_root/
    ├── FH/*.mp4   ├── BH/*.mp4   ├── SV/*.mp4
    ├── VL/*.mp4   ├── SL/*.mp4   └── RS/*.mp4

ตัวอย่าง:
    python scripts/run_batch.py /path/to/dataset -o outputs/ --workers 6 --height-cm 172
"""

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loeuf_cv.config import STROKE_TYPES

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

_worker_config = {}


def _init_worker(config_kwargs):
    global _worker_config
    _worker_config = config_kwargs


def _process_one(job):
    """worker: (video_path, stroke_type, out_path) → status dict"""
    video_path, stroke_type, out_path = job
    from loeuf_cv import PipelineConfig, StrokePipeline

    t0 = time.time()
    try:
        pipeline = StrokePipeline(PipelineConfig(**_worker_config))
        result = pipeline.process_video(str(video_path), stroke_type)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=1)
        root = result["strokes"][0]["stroke_root"]
        return {"clip": str(video_path), "ok": True,
                "status": root["detection_status"],
                "action": root["stroke_recommended_action"],
                "sec": round(time.time() - t0, 1)}
    except Exception as exc:
        return {"clip": str(video_path), "ok": False, "error": str(exc),
                "sec": round(time.time() - t0, 1)}


def collect_jobs(root: Path, out_dir: Path, resume: bool) -> list:
    jobs = []
    for stroke in STROKE_TYPES:
        folder = root / stroke
        if not folder.is_dir():
            continue
        for video in sorted(folder.rglob("*")):
            if video.suffix.lower() not in VIDEO_EXTS:
                continue
            rel = video.relative_to(root)
            out_path = out_dir / rel.with_suffix(".json")
            if resume and out_path.exists():
                continue
            jobs.append((video, stroke, out_path))
    return jobs


def main():
    ap = argparse.ArgumentParser(description="Loeuf CV — batch processing")
    ap.add_argument("dataset_root")
    ap.add_argument("-o", "--out-dir", default="outputs")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--height-cm", type=float, default=None)
    ap.add_argument("--side", default="right", choices=("right", "left"))
    ap.add_argument("--no-resume", action="store_true",
                    help="ประมวลผลใหม่ทั้งหมดแม้มี output แล้ว")
    args = ap.parse_args()

    root, out_dir = Path(args.dataset_root), Path(args.out_dir)
    jobs = collect_jobs(root, out_dir, resume=not args.no_resume)
    if not jobs:
        print("ไม่มีคลิปค้าง — ทุกอย่างประมวลผลแล้ว (ใช้ --no-resume เพื่อรันใหม่)")
        return

    config_kwargs = {"subject_height_cm": args.height_cm,
                     "dominant_side": args.side}
    print(f"processing {len(jobs)} clips with {args.workers} workers ...")

    results, t0 = [], time.time()
    with Pool(args.workers, initializer=_init_worker,
              initargs=(config_kwargs,)) as pool:
        for i, r in enumerate(pool.imap_unordered(_process_one, jobs), 1):
            results.append(r)
            tag = (f"{r['status']}/{r['action']}" if r["ok"]
                   else f"ERROR: {r['error'][:60]}")
            print(f"[{i}/{len(jobs)}] {Path(r['clip']).name}"
                  f" ({r['sec']}s) — {tag}")

    ok = [r for r in results if r["ok"]]
    failed = [r for r in results if not r["ok"]]
    summary = {
        "total": len(results), "ok": len(ok), "failed": len(failed),
        "elapsed_min": round((time.time() - t0) / 60, 1),
        "by_action": {},
        "failures": failed,
    }
    for r in ok:
        summary["by_action"][r["action"]] = \
            summary["by_action"].get(r["action"], 0) + 1

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "batch_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\ndone: {len(ok)}/{len(results)} ok in {summary['elapsed_min']}min"
          f" | actions: {summary['by_action']}")
    if failed:
        print(f"⚠️ {len(failed)} failures — ดู batch_summary.json")


if __name__ == "__main__":
    main()
