#!/usr/bin/env python3
"""Benchmark CLI: label CSV + batch outputs → BENCHMARK.md

ตัวอย่าง:
    python scripts/run_batch.py dataset/ -o outputs/ --height-cm 172
    python scripts/run_benchmark.py --labels labels/stroke_labels.csv \\
        --outputs outputs/ --report docs/BENCHMARK.md
    # เพิ่ม Phase 2:
    ... --session-labels labels/session_labels.csv --session-json match.session.json
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loeuf_cv.benchmark import (
    accuracy_by_camera_deviation, annotator_agreement, keyframe_accuracy,
    load_predictions, load_stroke_labels, render_markdown, spotting_metrics,
)


def main():
    ap = argparse.ArgumentParser(description="Loeuf CV — benchmark report")
    ap.add_argument("--labels", required=True, help="stroke_labels.csv")
    ap.add_argument("--outputs", required=True, help="โฟลเดอร์ output จาก run_batch")
    ap.add_argument("--tolerance", type=int, default=1, help="±N เฟรมต้นทาง")
    ap.add_argument("--report", default="docs/BENCHMARK.md")
    ap.add_argument("--session-labels", default=None)
    ap.add_argument("--session-json", default=None)
    args = ap.parse_args()

    labels = load_stroke_labels(args.labels)
    preds = load_predictions(args.outputs)
    print(f"labels: {len(labels)} rows | predictions: {len(preds)} clips")

    kf_results = keyframe_accuracy(labels, preds, args.tolerance)
    agreement = annotator_agreement(labels, args.tolerance)
    deviation = accuracy_by_camera_deviation(labels, preds, args.tolerance)

    spotting = None
    if args.session_labels and args.session_json:
        with open(args.session_labels, newline="", encoding="utf-8-sig") as f:
            session_labels = list(csv.DictReader(f))
        session_output = json.loads(Path(args.session_json).read_text())
        spotting = spotting_metrics(session_labels, session_output)

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        render_markdown(kf_results, agreement, deviation, spotting),
        encoding="utf-8")

    results_path = report_path.with_name("benchmark_results.json")
    results_path.write_text(json.dumps({
        "keyframes": kf_results, "agreement": agreement,
        "camera_deviation": deviation, "spotting": spotting,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    acc = kf_results["acceptance_accuracy"]
    print(f"acceptance accuracy (impact+backswing): {acc}"
          f" {'✅ ≥0.80' if acc and acc >= 0.80 else '⚠️ ต่ำกว่าเป้า 0.80' if acc else ''}")
    print(f"report → {report_path}\nresults → {results_path}")


if __name__ == "__main__":
    main()
