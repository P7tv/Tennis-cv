"""Benchmark harness — เทียบ pipeline output กับ label CSV

Deliverable MVP 1.3: Benchmark Report (Precision/Recall + Failure
Analysis แยกต่อ stroke type)

- เกณฑ์ keyframe: ถูกต้อง = |pred − GT| ≤ ±N เฟรมต้นทาง (default ±1)
  เทียบใน time domain เสมอ (GT frame → ms ด้วย fps ต้นทางของคลิปนั้น)
- ฟังก์ชันทั้งหมด pure — รับ records คืน dict → ทดสอบได้โดยไม่ต้องมีวิดีโอ
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

KEYFRAME_COLS = {
    "unit_turn_frame": "unit_turn",
    "backswing_peak_frame": "backswing_peak",
    "impact_frame": "impact",
    "follow_through_peak_frame": "follow_through_peak",
    "ready_frame": "ready_position_restored",
}
ACCEPTANCE_KEYFRAMES = ("impact", "backswing_peak")  # ตัวที่ TOR ผูกเกณฑ์
DEVIATION_BINS = ((0, 15), (15, 30), (30, 999))


def load_stroke_labels(csv_path: str) -> list[dict]:
    rows = []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        for raw in csv.DictReader(f):
            row = {k: (v.strip() if isinstance(v, str) else v)
                   for k, v in raw.items()}
            row["fps"] = float(row.get("fps") or 30)
            row["usable"] = str(row.get("usable", "true")).lower() != "false"
            for col in KEYFRAME_COLS:
                v = row.get(col, "")
                row[col] = int(float(v)) if v not in ("", None) else None
            rr = str(row.get("ready_restored", "")).lower()
            row["ready_restored"] = (True if rr == "true"
                                     else False if rr == "false" else None)
            rows.append(row)
    return rows


def load_predictions(output_dir: str) -> dict[str, dict]:
    """โหลด output JSON จาก run_batch → {clip_path_relative: stroke_object}"""
    preds = {}
    out = Path(output_dir)
    for jf in out.rglob("*.json"):
        if jf.name == "batch_summary.json":
            continue
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
            stroke = data["strokes"][0]
        except (json.JSONDecodeError, KeyError, IndexError):
            continue
        rel = str(jf.relative_to(out).with_suffix(""))
        preds[rel] = stroke
    return preds


def _match_pred(preds: dict, clip_path: str) -> dict | None:
    key = str(Path(clip_path).with_suffix(""))
    if key in preds:
        return preds[key]
    # เผื่อ path ไม่ตรงเป๊ะ — match จากชื่อไฟล์
    name = Path(clip_path).stem
    hits = [v for k, v in preds.items() if Path(k).name == name]
    return hits[0] if len(hits) == 1 else None


def keyframe_accuracy(labels: list[dict], preds: dict,
                      tolerance_frames: int = 1,
                      keyframe_cols: dict | None = None,
                      frame_domain: bool = False) -> dict:
    """accuracy ต่อ (stroke_type × keyframe) + failure list

    keyframe_cols: mapping {label_column -> keyframe_name} — default = KEYFRAME_COLS
    (schema เดิมจาก stroke_labels.csv + StrokePipeline) ส่ง mapping อื่นเข้ามาได้
    เมื่อ label/prediction ใช้ vocabulary คนละชุด — ดู
    loeuf_cv/benchmark_keyframes.py::BENCH_KEYFRAME_COLS ที่ใช้ชื่อ B-block ของ
    schema_builder (มี trophy_position, ใช้ recovery_position แทน
    ready_position_restored)

    frame_domain: ตัดสินถูก/ผิดด้วย ``frame_index`` ตรง ๆ
    (``|pred - gt| <= tolerance_frames``) แทนการเทียบ ``timestamp_ms``

    ⚠️ ทำไมต้องมีโหมดนี้ — spec ลูกค้ากำหนด tolerance เป็น "เฟรม" แต่การเทียบแบบ
    timestamp เอา GT (``frame / fps_ใน_label``) ไปชนกับ prediction
    (``timestamp_ms`` ที่อ่านจาก cv2 = เวลาจริงในไฟล์) ซึ่งเป็น **คนละฐานเวลา**:
    fps ใน label เป็นค่าปัดเศษ (29.97) ของจริง 29.974841... ทำให้คลาดสะสมเชิงเส้น
    ~38.8 ms ต่อ 1000 เฟรม ขณะที่ tolerance 1 เฟรม ≈ 34.4 ms → พอเลยวินาทีที่ ~30
    ของคลิป prediction ที่เฟรม "ตรงเป๊ะ" ก็ถูกนับผิด (วัดจากข้อมูลจริง: 68/132
    stroke = 51.5%) คลิป session ลูกค้ายาว 1-11 นาที จึงต้องใช้ frame_domain=True
    (default False เพื่อไม่ให้ CLI เดิมเปลี่ยนพฤติกรรม)
    """
    keyframe_cols = keyframe_cols or KEYFRAME_COLS
    stats = defaultdict(lambda: {"n": 0, "correct": 0, "errors_ms": [],
                                 "errors_frames": []})
    failures, missing_pred = [], 0

    for row in labels:
        if not row["usable"]:
            continue
        pred = _match_pred(preds, row["clip_path"])
        if pred is None:
            missing_pred += 1
            continue
        tol_ms = tolerance_frames * 1000.0 / row["fps"] + 1.0

        for col, kf_name in keyframe_cols.items():
            # .get() ไม่ใช่ [] — caller ที่ส่ง keyframe_cols ชุดอื่นอาจไม่มี column
            # ครบทุกตัวใน row (เช่น GT ที่ไม่ได้ label keyframe นั้นเลย)
            gt_frame = row.get(col)
            if gt_frame is None:
                continue
            gt_ms = gt_frame / row["fps"] * 1000.0
            p = pred["keyframes"][kf_name]
            key = (row["stroke_type"], kf_name)
            stats[key]["n"] += 1

            if not p["detected"]:
                failures.append({"clip": row["clip_path"], "keyframe": kf_name,
                                 "reason": "not_detected", "gt_ms": round(gt_ms, 1)})
                continue
            err = abs(p["timestamp_ms"] - gt_ms)
            stats[key]["errors_ms"].append(err)
            if frame_domain:
                # เทียบเฟรมตรง ๆ ตาม spec — err_ms ยังเก็บไว้รายงาน MAE ได้
                # แต่ห้ามใช้ตัดสิน (คนละฐานเวลา ดู docstring)
                pf = p.get("frame_index")
                if pf is None:
                    ok = False
                else:
                    df = abs(int(pf) - int(gt_frame))
                    stats[key]["errors_frames"].append(df)
                    ok = df <= tolerance_frames
            else:
                ok = err <= tol_ms
            if ok:
                stats[key]["correct"] += 1
            else:
                failures.append({"clip": row["clip_path"], "keyframe": kf_name,
                                 "reason": "off_by_ms", "error_ms": round(err, 1)})

    table = {}
    for (stroke, kf), s in sorted(stats.items()):
        table.setdefault(stroke, {})[kf] = {
            "n": s["n"],
            "accuracy": round(s["correct"] / s["n"], 3) if s["n"] else None,
            "mae_ms": round(float(np.mean(s["errors_ms"])), 1)
            if s["errors_ms"] else None,
            # frame-domain: ไม่ปนกับ time-base drift (ดู docstring)
            "mae_frames": round(float(np.mean(s["errors_frames"])), 2)
            if s["errors_frames"] else None,
            "median_frames": round(float(np.median(s["errors_frames"])), 1)
            if s["errors_frames"] else None,
        }

    # เกณฑ์ acceptance รวม (impact + backswing_peak ทุกท่า)
    acc_n = acc_c = 0
    for (stroke, kf), s in stats.items():
        if kf in ACCEPTANCE_KEYFRAMES:
            acc_n += s["n"]
            acc_c += s["correct"]
    return {
        "tolerance_frames": tolerance_frames,
        "per_stroke": table,
        "acceptance_accuracy": round(acc_c / acc_n, 3) if acc_n else None,
        "acceptance_n": acc_n,
        "missing_predictions": missing_pred,
        "failures": failures,
    }


def annotator_agreement(labels: list[dict],
                        tolerance_frames: int = 1) -> dict | None:
    """clip ที่มี >1 annotator → % ที่ตรงกันภายใน ±N เฟรม"""
    by_clip = defaultdict(list)
    for row in labels:
        if row["usable"]:
            by_clip[row["clip_path"]].append(row)

    n = agree = 0
    for clip, rows in by_clip.items():
        if len(rows) < 2:
            continue
        a, b = rows[0], rows[1]
        for col in ("impact_frame", "backswing_peak_frame"):
            if a[col] is None or b[col] is None:
                continue
            n += 1
            if abs(a[col] - b[col]) <= tolerance_frames:
                agree += 1
    if n == 0:
        return None
    return {"pairs_compared": n, "agreement": round(agree / n, 3)}


def accuracy_by_camera_deviation(labels: list[dict], preds: dict,
                                 tolerance_frames: int = 1) -> list[dict]:
    """impact accuracy แยกตาม camera deviation → หลักฐาน Reliability Matrix"""
    bins = [{"range_deg": f"{lo}–{hi if hi < 999 else '+'}",
             "n": 0, "correct": 0} for lo, hi in DEVIATION_BINS]
    for row in labels:
        if not row["usable"] or row["impact_frame"] is None:
            continue
        pred = _match_pred(preds, row["clip_path"])
        if pred is None:
            continue
        dev = pred.get("stroke_metadata", {}).get(
            "debug_camera", {}).get("camera_deviation_deg")
        if dev is None:
            continue
        gt_ms = row["impact_frame"] / row["fps"] * 1000.0
        tol_ms = tolerance_frames * 1000.0 / row["fps"] + 1.0
        p = pred["keyframes"]["impact"]
        ok = p["detected"] and abs(p["timestamp_ms"] - gt_ms) <= tol_ms
        for (lo, hi), b in zip(DEVIATION_BINS, bins):
            if lo <= dev < hi:
                b["n"] += 1
                b["correct"] += int(ok)
    for b in bins:
        b["accuracy"] = round(b["correct"] / b["n"], 3) if b["n"] else None
    return bins


def spotting_metrics(session_labels: list[dict], session_output: dict) -> dict:
    """Phase 2: recall/precision ของ action spotting

    session_labels: [{session_path, stroke_type, start_time_s, end_time_s}]
    matched = predicted stroke ที่ impact (หรือกึ่งกลาง window) ตกใน GT segment
    """
    preds = session_output.get("strokes", [])
    pred_times = []
    for s in preds:
        imp = s["keyframes"]["impact"]
        if imp["detected"]:
            pred_times.append(imp["timestamp_ms"] / 1000.0)

    gt_matched = [False] * len(session_labels)
    pred_matched = [False] * len(pred_times)
    for gi, gt in enumerate(session_labels):
        for pi, t in enumerate(pred_times):
            if float(gt["start_time_s"]) <= t <= float(gt["end_time_s"]):
                gt_matched[gi] = True
                pred_matched[pi] = True
    recall = sum(gt_matched) / len(gt_matched) if gt_matched else None
    precision = (sum(pred_matched) / len(pred_matched)
                 if pred_matched else None)
    return {
        "gt_strokes": len(session_labels),
        "predicted_strokes": len(pred_times),
        "recall": round(recall, 3) if recall is not None else None,
        "precision": round(precision, 3) if precision is not None else None,
    }


def render_markdown(kf_results: dict, agreement: dict | None,
                    deviation_bins: list | None,
                    spotting: dict | None) -> str:
    lines = ["# Benchmark Report — Loeuf CV Pipeline", ""]
    acc = kf_results["acceptance_accuracy"]
    lines += [
        f"เกณฑ์: ถูกต้อง = ±{kf_results['tolerance_frames']} เฟรมต้นทาง",
        "",
        f"## Acceptance (impact + backswing_peak รวมทุกท่า)",
        f"**Accuracy: {acc if acc is not None else 'N/A'}**"
        f" (n={kf_results['acceptance_n']}, เป้า TOR ≥ 0.80)",
        "",
        "## Keyframe accuracy ต่อ stroke type",
        "",
        "| Stroke | Keyframe | n | Accuracy | MAE (ms) |",
        "|---|---|---|---|---|",
    ]
    for stroke, kfs in kf_results["per_stroke"].items():
        for kf, s in kfs.items():
            lines.append(f"| {stroke} | {kf} | {s['n']}"
                         f" | {s['accuracy']} | {s['mae_ms']} |")

    if agreement:
        lines += ["", "## Inter-annotator agreement",
                  f"ตรงกันภายใน ±1 เฟรม: **{agreement['agreement']}**"
                  f" ({agreement['pairs_compared']} คู่เทียบ)"]

    if deviation_bins:
        lines += ["", "## Impact accuracy vs camera deviation",
                  "", "| Deviation (deg) | n | Accuracy |", "|---|---|---|"]
        for b in deviation_bins:
            lines.append(f"| {b['range_deg']} | {b['n']} | {b['accuracy']} |")

    if spotting:
        lines += ["", "## Action spotting (Phase 2)",
                  f"- GT strokes: {spotting['gt_strokes']}"
                  f" / predicted: {spotting['predicted_strokes']}",
                  f"- Recall: **{spotting['recall']}**"
                  f" | Precision: **{spotting['precision']}**"]

    n_fail = len(kf_results["failures"])
    lines += ["", "## Failure analysis",
              f"failures ทั้งหมด: {n_fail}"
              f" | คลิปที่ไม่มี prediction: {kf_results['missing_predictions']}",
              "", "รายละเอียดใน `benchmark_results.json`"]
    return "\n".join(lines) + "\n"
