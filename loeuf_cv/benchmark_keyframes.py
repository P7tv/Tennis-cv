"""Keyframe benchmark สำหรับ pipeline production (schema_builder/builder.py)

ต่างจาก loeuf_cv/benchmark.py เดิมที่สร้างมาสำหรับแผนข้อมูลเดิม (คลิป
single-stroke แยกโฟลเดอร์ตาม stroke type + stroke_labels.csv) ซึ่งไม่เคยเกิดขึ้น
จริง — ข้อมูลจริงคือ session ยาว 13 คลิป มีหลาย stroke ต่อคลิป (142 stroke รวม)
label เป็น JSON ต่อคลิป และ pipeline ที่ส่งลูกค้าคือ build_loeuf_schema()
ซึ่ง emit key `keyframe` (เอกพจน์) + `recovery_position` + `trophy_position`

โมดูลนี้ pure ทั้งหมด (json/collections/numpy) — ไม่ import cv2/torch/ultralytics
เทสได้โดยไม่ต้องมีวิดีโอ ส่วนที่ต้องรัน YOLO อยู่ใน
scripts/run_keyframe_benchmark.py

reuse จาก benchmark.py: keyframe_accuracy() (ผ่าน param keyframe_cols ใหม่),
_match_pred(), ACCEPTANCE_KEYFRAMES
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .benchmark import keyframe_accuracy

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

# identity map — ใช้ชื่อ B-block ของ schema_builder/builder.py เป็นมาตรฐาน
# เพราะ train_model/label_ingest.py::KEYFRAME_FIELD_MAP ก็ emit ชื่อชุดนี้อยู่แล้ว
# (ไม่ต้องแปลงสองต่อ) — B5 ในโปรเจกต์นี้มี 3 ชื่อ: recovery_position (builder +
# label_ingest, ใช้ที่นี่) / recovery_frame (field ใน label JSON) /
# ready_position_restored (config.KEYFRAME_NAMES + benchmark.KEYFRAME_COLS ของ
# StrokePipeline path เดิม — จงใจไม่ใช้ที่นี่)
BENCH_KEYFRAME_COLS = {
    "unit_turn": "unit_turn",                      # B1
    "backswing_peak": "backswing_peak",            # B2
    "trophy_position": "trophy_position",          # B6 (serve-only)
    "impact": "impact",                            # B3
    "follow_through_peak": "follow_through_peak",  # B4
    "recovery_position": "recovery_position",      # B5
}

# ---------------------------------------------------------------------------
# Matching constants — ดู docstring ของ match_strokes() สำหรับข้อพิสูจน์
# ---------------------------------------------------------------------------

MATCH_TOLERANCE_FRAMES = 10

# ระยะห่าง impact ที่ใกล้ที่สุดใน GT ทั้ง 142 stroke (วัดจริงจาก label ทั้ง 13 ไฟล์)
MIN_GT_IMPACT_GAP = 54

# detect_hit_events() de-dup candidate ที่ MIN_GAP = int(fps * 0.75)
# → 29.97fps ได้ 22 เฟรม (loeuf_cv/hit_detection.py:500)
HIT_MIN_GAP_FRAMES_AT_30 = 22

# pred ปลอมสำหรับ GT ที่ hit detection หาไม่เจอ — ทำให้เรียก keyframe_accuracy()
# ครั้งเดียวได้ตัวเลข end-to-end (GT ที่หาไม่เจอนับเป็นผิด ไม่ใช่หายจากตัวหาร)
NOT_DETECTED_PRED = {
    "keyframe": {
        name: {"frame_index": None, "timestamp_ms": None, "detected": False}
        for name in BENCH_KEYFRAME_COLS.values()
    }
}


# ---------------------------------------------------------------------------
# Label adapter
# ---------------------------------------------------------------------------

def stroke_key(set_name: str, clip_stem: str, stroke_no: int) -> str:
    """คีย์ระบุ stroke แบบไม่มีทั้งจุดและ path separator

    benchmark._match_pred() ทำ str(Path(clip_path).with_suffix("")) แล้วเทียบ
    ตรง ๆ กับคีย์ใน preds — มี 2 กับดัก:
      1. ถ้าคีย์มีจุด (เช่น "IMG_0284.MOV__s03") ".MOV__s03" จะถูกมองเป็น
         suffix แล้วตัดทิ้ง → stroke ต่างกันในคลิปเดียวกันชนกันหมด
      2. ถ้าคีย์มี "/" บน Windows Path() จะ normalize เป็น "\\" → เทียบไม่ตรง
         แล้วตกไปใช้ stem fallback ที่จะคืน None ทันทีถ้าชื่อซ้ำเกิน 1
    จึงใช้ "__" คั่นแทน separator และไม่มีจุดเลย → roundtrip ตรงเป๊ะ
    (ล็อกไว้ด้วย test_stroke_key_roundtrips_through_match_pred)
    """
    safe_set = set_name.replace("/", "__").replace(".", "_")
    safe_stem = clip_stem.replace(".", "_")
    return f"{safe_set}__{safe_stem}__s{stroke_no:02d}"


def anomaly_flags(keyframes: dict) -> list[str]:
    """ความผิดปกติเชิงลำดับเวลาใน GT เอง (label ผิด ไม่ใช่โมเดลผิด)

    ตรวจพบจริง 4 จุดใน 142 stroke — flag ไว้เพื่อทำ sensitivity analysis
    ไม่ตัด stroke ทิ้ง (ตัดทิ้งเงียบ ๆ = ทำตัวเลขสวยเกินจริง)
    """
    flags = []
    impact = keyframes.get("impact")
    backswing = keyframes.get("backswing_peak")
    follow = keyframes.get("follow_through_peak")

    if impact is not None and backswing is not None and backswing >= impact:
        flags.append("backswing_not_before_impact")
    if impact is not None and follow is not None and follow < impact:
        flags.append("follow_through_before_impact")
    if follow is None:
        flags.append("missing_follow_through")
    return flags


def label_rows_from_session(session, set_name: str,
                            raw_strokes: list[dict] | None = None) -> list[dict]:
    """SessionLabel (train_model/label_ingest.py) → label row ต่อ "stroke"

    ไม่ใช่ต่อคลิป — benchmark.keyframe_accuracy() คิดหน่วยเป็น row เดียว = 1
    prediction ดังนั้นคลิป session ที่มี 23 stroke ต้องกลายเป็น 23 row

    raw_strokes: stroke dict ดิบจาก label JSON (เรียงตาม stroke_no) สำหรับดึง
    quality/visibility ที่ StrokeLabel dataclass ไม่ได้เก็บไว้ — ส่งมาหรือไม่ก็ได้
    """
    quality_by_no = {}
    if raw_strokes:
        for s in raw_strokes:
            quality_by_no[s.get("stroke_no")] = s

    clip_stem = session.video_path.stem
    rows = []
    for s in session.strokes:
        raw = quality_by_no.get(s.stroke_no, {})
        row = {
            "clip_path": stroke_key(set_name, clip_stem, s.stroke_no),
            "clip_stem": clip_stem,
            "set": set_name,
            "stroke_no": s.stroke_no,
            "stroke_type": s.stroke_type,
            "fps": s.fps,
            "usable": s.usable,
            "impact_quality": raw.get("impact_quality", ""),
            "impact_visibility": raw.get("impact_visibility", ""),
            "anomalies": anomaly_flags(s.keyframes),
        }
        for name in BENCH_KEYFRAME_COLS:
            row[name] = s.keyframes.get(name)
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Prediction adapter
# ---------------------------------------------------------------------------

def pred_strokes_from_schema(schema: dict) -> list[dict]:
    """build_loeuf_schema() output → [{stroke_index, stroke_type, impact_frame,
    keyframes}] โดย rename block `keyframe` (เอกพจน์) → `keyframes` ให้ตรงกับ
    ที่ benchmark.keyframe_accuracy() คาด
    """
    out = []
    for s in schema.get("strokes", []):
        kf = s.get("keyframe", {})
        root = s.get("stroke_root", {})
        impact = kf.get("impact", {})
        out.append({
            "stroke_index": root.get("stroke_index"),
            "stroke_type": root.get("stroke_type"),
            "detection_status": root.get("detection_status"),
            "impact_frame": impact.get("frame_index"),
            "keyframe": {name: kf.get(name, dict(NOT_DETECTED_PRED["keyframe"][name]))
                          for name in BENCH_KEYFRAME_COLS.values()},
        })
    return out


# ---------------------------------------------------------------------------
# 1:1 matching (Mode A)
# ---------------------------------------------------------------------------

def match_strokes(gt_impacts: list[int], pred_impacts: list[int],
                  tolerance: int = MATCH_TOLERANCE_FRAMES) -> dict:
    """จับคู่ GT stroke ↔ predicted stroke แบบ 1:1 ด้วย impact frame

    Greedy โดยเรียงคู่ที่ |Δ| น้อยสุดก่อน — ภายใต้ tolerance ที่ตั้งไว้ผลลัพธ์
    เท่ากับ Hungarian (optimal) พิสูจน์ได้:
      GT ห่างกัน >= MIN_GT_IMPACT_GAP (54) และ detect_hit_events de-dup ที่
      HIT_MIN_GAP_FRAMES_AT_30 (22) → ถ้า 2*tolerance < ทั้งคู่ จะไม่มี GT ตัวไหน
      มี pred ในระยะเกิน 1 ตัว และไม่มี pred ตัวไหนมี GT ในระยะเกิน 1 ตัว
      กราฟที่ผ่าน gate จึงเป็น edge เดี่ยว ๆ ไม่ต่อกัน → greedy = optimal
      (ดู test_match_tolerance_within_gap_bounds ที่ล็อกเงื่อนไขนี้ไว้)

    tolerance นี้คนละตัวกับ ±1 เฟรมของ accuracy — อันนี้ตอบว่า "เจอ stroke นี้
    ไหม" ไม่ใช่ "keyframe ตรงหรือเปล่า"

    คืน {"matches": [(gt_idx, pred_idx, delta)], "fn": [gt_idx], "fp": [pred_idx]}
    delta = pred - gt (มีเครื่องหมาย เพื่อดู systematic lag)
    """
    pairs = []
    for gi, g in enumerate(gt_impacts):
        for pi, p in enumerate(pred_impacts):
            if g is None or p is None:
                continue
            d = abs(g - p)
            if d <= tolerance:
                pairs.append((d, gi, pi))
    pairs.sort()  # |Δ| ก่อน แล้ว gi, pi → deterministic

    used_g, used_p, matches = set(), set(), []
    for _d, gi, pi in pairs:
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        matches.append((gi, pi, pred_impacts[pi] - gt_impacts[gi]))

    matches.sort()
    fn = [i for i in range(len(gt_impacts)) if i not in used_g]

    # ระยะถึง prediction ที่ใกล้สุดของ GT ที่จับคู่ไม่ได้ — แยก "พลาดสนิท" (ไกลมาก)
    # ออกจาก "เจอแต่เพี้ยนเกิน tolerance" (ใกล้ ๆ) ซึ่งคนละปัญหากันคนละทางแก้
    fn_nearest = []
    for gi in fn:
        g = gt_impacts[gi]
        cand = [abs(g - p) for p in pred_impacts if p is not None]
        fn_nearest.append(min(cand) if cand else None)

    return {
        "matches": matches,
        "fn": fn,
        "fn_nearest_delta": fn_nearest,
        "fp": [i for i in range(len(pred_impacts)) if i not in used_p],
    }


def detection_metrics(per_clip: list[dict]) -> dict:
    """รวมผล matching ต่อคลิปเป็น recall/precision + FP ต่อนาที

    per_clip: [{"clip": str, "n_gt": int, "n_pred": int, "tp": int,
                "fn": int, "fp": int, "duration_sec": float, "deltas": [int]}]

    FP ต่อนาทีสำคัญเพราะคลิปยาวไม่เท่ากันมาก (13.9 วิ ถึง 10.8 นาที) — FP ดิบ
    จะถูกคลิปยาวครอบงำ
    """
    tp = sum(c["tp"] for c in per_clip)
    fn = sum(c["fn"] for c in per_clip)
    fp = sum(c["fp"] for c in per_clip)
    total_min = sum(c.get("duration_sec", 0.0) for c in per_clip) / 60.0
    deltas = [d for c in per_clip for d in c.get("deltas", [])]

    # GT ที่พลาด — ใกล้แค่ไหน? แยก "พลาดสนิท" ออกจาก "เกือบตรงแต่เกิน tolerance"
    near = [d for c in per_clip for d in c.get("fn_nearest_delta", [])
            if d is not None]

    return {
        "tp": tp, "fn": fn, "fp": fp,
        "fn_nearest_median": round(float(np.median(near)), 1) if near else None,
        "fn_within_2x_tolerance": sum(
            1 for d in near if d <= 2 * MATCH_TOLERANCE_FRAMES),
        "recall": round(tp / (tp + fn), 3) if (tp + fn) else None,
        "precision": round(tp / (tp + fp), 3) if (tp + fp) else None,
        "fp_per_minute": round(fp / total_min, 2) if total_min else None,
        "delta_mean": round(float(np.mean(deltas)), 1) if deltas else None,
        "delta_median": round(float(np.median(deltas)), 1) if deltas else None,
        "delta_p90": round(float(np.percentile(np.abs(deltas), 90)), 1) if deltas else None,
        "per_clip": per_clip,
    }


# ---------------------------------------------------------------------------
# Coverage / scoring
# ---------------------------------------------------------------------------

def coverage_table(label_rows: list[dict]) -> dict:
    """นับ GT ที่มีจริงต่อ (stroke_type × keyframe)

    จำเป็นเพราะ GT ไม่ได้ label ครบทุก keyframe ทุก stroke โดยตั้งใจ — SV ไม่มี
    unit_turn (ใช้ trophy แทน), trophy มีเฉพาะ SV, recovery มีแค่ 43/142
    ช่องที่ n_gt = 0 ต้องแสดงเป็น "ไม่มี GT" ไม่ใช่ accuracy 0.000
    """
    table = defaultdict(lambda: defaultdict(int))
    for row in label_rows:
        for name in BENCH_KEYFRAME_COLS:
            if row.get(name) is not None:
                table[row["stroke_type"]][name] += 1
    return {k: dict(v) for k, v in table.items()}


def score_mode(label_rows: list[dict], preds_by_key: dict,
               tolerance_frames: int = 1) -> dict:
    """เรียก benchmark.keyframe_accuracy() ด้วย vocabulary ของ B-block
    แล้วแนบ coverage + การแยกตามคุณภาพ GT
    """
    result = keyframe_accuracy(label_rows, preds_by_key,
                               tolerance_frames=tolerance_frames,
                               keyframe_cols=BENCH_KEYFRAME_COLS,
                               frame_domain=True)
    result["coverage"] = coverage_table(label_rows)
    result["by_impact_quality"] = _acceptance_by_quality(
        label_rows, preds_by_key, tolerance_frames)
    return result


def _acceptance_by_quality(label_rows: list[dict], preds_by_key: dict,
                           tolerance_frames: int) -> dict:
    """acceptance accuracy แยกตาม impact_quality ที่ annotator ให้ไว้

    GT เองก็มีคุณภาพไม่เท่ากัน — ถ้าตัวเลขต่ำเฉพาะกลุ่ม bad/worst แปลว่าปัญหา
    อยู่ที่เฉลย ไม่ใช่โมเดลล้วน ๆ
    """
    out = {}
    by_q = defaultdict(list)
    for row in label_rows:
        by_q[row.get("impact_quality") or "(blank)"].append(row)
    for q, rows in sorted(by_q.items()):
        r = keyframe_accuracy(rows, preds_by_key,
                              tolerance_frames=tolerance_frames,
                              keyframe_cols=BENCH_KEYFRAME_COLS,
                               frame_domain=True)
        out[q] = {"n": r["acceptance_n"], "accuracy": r["acceptance_accuracy"]}
    return out


# ---------------------------------------------------------------------------
# เกณฑ์ "อยู่ในช่วงการเคลื่อนไหวเดียวกัน" (phase membership)
# ---------------------------------------------------------------------------
#
# 🌟 นี่คือเกณฑ์ที่ลูกค้าเขียนไว้เอง ไม่ใช่เกณฑ์ที่เราตั้งขึ้น
#
# cv_schema_table_loeuf หัวข้อ "Keyframe selection":
#   "Guideline นี้มีไว้เพื่อช่วยเลือก *ช่วงของ keyframe* เท่านั้น
#    ไม่จำเป็นต้องจับเฟรมได้ตรงเป๊ะ หากอยู่ในช่วงการเคลื่อนไหวเดียวกัน
#    ถือว่าใช้ได้"
# และในเอกสารเดียวกันลูกค้ากำกับ impact กับ follow_through_peak ว่า
# "(Low Confidence)" ด้วยตัวเอง — ตรงกับที่ GT 97.6% ถูก flag low_confidence
#
# ปัญหาของ ±N เฟรมคงที่: ต้องเลือก N เอง ซึ่งไม่มีที่มา และไม่ยุติธรรมข้ามท่า —
# เสิร์ฟมี backswing ยาวเป็นวินาที ส่วนวอลเลย์ทั้งสวิงสั้นกว่านั้นอีก N เดียว
# จึงหลวมเกินไปสำหรับท่าหนึ่งและแคบเกินไปสำหรับอีกท่า
#
# นิยามที่ใช้แทน: เฟรมที่ทายถือว่า "อยู่ในช่วงเดียวกัน" ถ้ามันยัง **ใกล้
# keyframe นี้มากกว่า keyframe อื่น** ของ stroke เดียวกัน — คือขอบเขตอยู่ที่
# จุดกึ่งกลางระหว่าง keyframe ที่ติดกันใน GT
#
#   backswing_peak        impact              follow_through_peak
#        |                  |                        |
#        +--------+---------+-----------+------------+
#                 |<-- ช่วงที่นับว่าเป็น impact -->|
#
# ทำไมนิยามนี้ถึงตรงกับสิ่งที่ลูกค้าพูด: ถ้าเฟรมที่ทายเลยจุดกึ่งกลางระหว่าง
# impact กับ follow_through ไปแล้ว คนที่ดูเฟรมนั้นจะเรียกมันว่า "ตอนตีจบ"
# ไม่ใช่ "ตอนกระทบลูก" = คนละช่วงการเคลื่อนไหวแล้ว
#
# ข้อดีที่ตามมา: หน้าต่างกว้างแคบเองตามจังหวะจริงของแต่ละ stroke ไม่ต้องตั้ง
# ค่าคงที่ใด ๆ และไม่ต้องแยกกฎต่อท่า

# ครึ่งความกว้างขั้นต่ำ — GT บางตัววาง keyframe ซ้อนเฟรมเดียวกัน (พบจริง 1 จุด:
# backswing == impact) ทำให้ช่วงยุบเหลือศูนย์ ถ้าไม่มีขั้นต่ำนี้ แม้ทายตรงเป๊ะ
# ก็จะถูกนับว่าผิด
MIN_PHASE_HALF_WIDTH = 0.5


def phase_windows(gt_keyframes: dict) -> dict[str, tuple[float, float]]:
    """ช่วงเฟรมที่นับว่า "ยังอยู่ใน keyframe นั้น" ต่อ keyframe หนึ่ง stroke

    ขอบ = จุดกึ่งกลางไปยัง keyframe ที่อยู่ติดกันใน GT
    ตัวหัว/ตัวท้าย (ไม่มีเพื่อนบ้านข้างหนึ่ง) ใช้วิธีสะท้อนความกว้างของอีกข้าง

    ⚠️ เรียงตาม "ค่าเฟรมของ GT" ไม่ใช่ลำดับตามทฤษฎี — GT 4 จุดใน dataset มี
    ลำดับเวลาผิดปกติ (backswing หลัง impact ฯลฯ ดู anomaly_flags) ถ้ายึดลำดับ
    ตามทฤษฎีจะได้ช่วงติดลบ
    """
    pts = sorted((v, k) for k, v in gt_keyframes.items() if v is not None)
    if not pts:
        return {}
    if len(pts) == 1:
        v, k = pts[0]
        return {k: (v - MIN_PHASE_HALF_WIDTH, v + MIN_PHASE_HALF_WIDTH)}

    out = {}
    for i, (v, k) in enumerate(pts):
        left = (v - pts[i - 1][0]) / 2.0 if i > 0 else None
        right = (pts[i + 1][0] - v) / 2.0 if i < len(pts) - 1 else None
        if left is None:
            left = right          # ตัวหัว: สะท้อนความกว้างฝั่งขวา
        if right is None:
            right = left          # ตัวท้าย: สะท้อนความกว้างฝั่งซ้าย
        left = max(left, MIN_PHASE_HALF_WIDTH)
        right = max(right, MIN_PHASE_HALF_WIDTH)
        out[k] = (v - left, v + right)
    return out


def score_phase_mode(label_rows: list[dict], preds_by_key: dict) -> dict:
    """เหมือน score_mode() แต่ตัดสินด้วยเกณฑ์ "ช่วงการเคลื่อนไหวเดียวกัน"

    คืน dict รูปเดียวกับ keyframe_accuracy() เพื่อให้ renderer เดิมใช้ต่อได้
    เพิ่ม "window_frames" = ครึ่งความกว้างเฉลี่ยของหน้าต่างที่ใช้จริง เพื่อให้
    ตรวจสอบได้ว่าเกณฑ์นี้หลวมแค่ไหน (ไม่ใช่กล่องดำ)
    """
    from .benchmark import ACCEPTANCE_KEYFRAMES, _match_pred

    stats = defaultdict(lambda: {"n": 0, "correct": 0, "errors_frames": [],
                                 "half_widths": []})
    failures, missing_pred = [], 0

    for row in label_rows:
        if not row["usable"]:
            continue
        pred = _match_pred(preds_by_key, row["clip_path"])
        if pred is None:
            missing_pred += 1
            continue

        gt_kf = {name: row.get(name) for name in BENCH_KEYFRAME_COLS}
        windows = phase_windows(gt_kf)

        for col, kf_name in BENCH_KEYFRAME_COLS.items():
            gt_frame = gt_kf.get(col)
            if gt_frame is None:
                continue
            lo, hi = windows[col]
            key = (row["stroke_type"], kf_name)
            stats[key]["n"] += 1
            stats[key]["half_widths"].append((hi - lo) / 2.0)

            p = pred["keyframe"][kf_name]
            if not p["detected"] or p.get("frame_index") is None:
                failures.append({"clip": row["clip_path"], "keyframe": kf_name,
                                 "reason": "not_detected"})
                continue
            pf = int(p["frame_index"])
            stats[key]["errors_frames"].append(abs(pf - int(gt_frame)))
            if lo <= pf <= hi:
                stats[key]["correct"] += 1
            else:
                failures.append({
                    "clip": row["clip_path"], "keyframe": kf_name,
                    "reason": "outside_phase", "pred": pf, "gt": int(gt_frame),
                    "window": [round(lo, 1), round(hi, 1)]})

    table = {}
    for (stroke, kf), s in sorted(stats.items()):
        table.setdefault(stroke, {})[kf] = {
            "n": s["n"],
            "accuracy": round(s["correct"] / s["n"], 3) if s["n"] else None,
            "mae_ms": None,
            "mae_frames": round(float(np.mean(s["errors_frames"])), 2)
            if s["errors_frames"] else None,
            "median_frames": round(float(np.median(s["errors_frames"])), 1)
            if s["errors_frames"] else None,
            "window_frames": round(float(np.mean(s["half_widths"])), 1)
            if s["half_widths"] else None,
        }

    acc_n = acc_c = 0
    for (stroke, kf), s in stats.items():
        if kf in ACCEPTANCE_KEYFRAMES:
            acc_n += s["n"]
            acc_c += s["correct"]

    return {
        "criterion": "phase_membership",
        "tolerance_frames": None,
        "per_stroke": table,
        "acceptance_accuracy": round(acc_c / acc_n, 3) if acc_n else None,
        "acceptance_n": acc_n,
        "missing_predictions": missing_pred,
        "failures": failures,
        "coverage": coverage_table(label_rows),
    }


def sensitivity_without_anomalies(label_rows: list[dict], preds_by_key: dict,
                                  tolerance_frames: int = 1) -> dict:
    """คะแนนเมื่อ null เฉพาะ "ค่า keyframe ที่ผิดปกติ" ไม่ใช่ตัด stroke ทั้งตัว

    ผลต่างจากตัวเลขหลักบอกว่า label ที่เพี้ยน 4 จุดกระทบผลแค่ไหน
    """
    cleaned = []
    for row in label_rows:
        r = dict(row)
        flags = row.get("anomalies") or []
        if "backswing_not_before_impact" in flags:
            r["backswing_peak"] = None
        if "follow_through_before_impact" in flags:
            r["follow_through_peak"] = None
        cleaned.append(r)
    return keyframe_accuracy(cleaned, preds_by_key,
                             tolerance_frames=tolerance_frames,
                             keyframe_cols=BENCH_KEYFRAME_COLS,
                               frame_domain=True)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

TOR_TARGET = 0.80

_KF_ORDER = ("unit_turn", "backswing_peak", "trophy_position", "impact",
             "follow_through_peak", "recovery_position")


def _fmt(v, nd=3):
    return "—" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def _accuracy_table(result: dict, coverage: dict) -> list[str]:
    # ⚠️ รายงาน error เป็น "เฟรม" ไม่ใช่ ms — accuracy ตัดสินในโดเมนเฟรม
    # (benchmark.keyframe_accuracy(frame_domain=True)) ส่วน mae_ms คิดจาก
    # timestamp ที่คนละฐานเวลากับ GT (คลาดสะสม ~38.8 ms/1000 เฟรม) ถ้าเอา ms
    # มาโชว์คู่กัน จะได้บรรทัดที่ขัดกันเอง เช่น "accuracy 1.000 · MAE 74 ms"
    lines = ["| Stroke | Keyframe | n GT | Accuracy | MAE (เฟรม) | median (เฟรม) |",
             "|---|---|---|---|---|---|"]
    per_stroke = result.get("per_stroke", {})
    for stroke in sorted(set(list(per_stroke) + list(coverage))):
        for kf in _KF_ORDER:
            n_gt = coverage.get(stroke, {}).get(kf, 0)
            if n_gt == 0:
                # ไม่มี GT = ไม่ได้วัด ไม่ใช่สอบตก — ห้ามแสดง 0.000
                lines.append(f"| {stroke} | {kf} | 0 | — (no GT) | — | — |")
                continue
            s = per_stroke.get(stroke, {}).get(kf)
            if s is None:
                lines.append(f"| {stroke} | {kf} | {n_gt} | — (no pred) | — | — |")
            else:
                lines.append(f"| {stroke} | {kf} | {s['n']} | "
                             f"{_fmt(s['accuracy'])} | "
                             f"{_fmt(s.get('mae_frames'), 2)} | "
                             f"{_fmt(s.get('median_frames'), 1)} |")
    return lines


def _phase_accuracy_table(result: dict, coverage: dict, label: str) -> list[str]:
    """เหมือน _accuracy_table แต่โชว์ความกว้างหน้าต่างที่เกณฑ์นี้ใช้จริง

    ต้องโชว์ เพราะหน้าต่างคำนวณจาก GT ของแต่ละ stroke ไม่ใช่ค่าคงที่ —
    ถ้าไม่รายงาน ผู้อ่านจะตรวจสอบไม่ได้ว่าเกณฑ์หลวมแค่ไหน
    """
    lines = [f"| Stroke | Keyframe | n GT | Accuracy ({label}) | "
             "หน้าต่าง ± (เฟรม) | MAE (เฟรม) |",
             "|---|---|---|---|---|---|"]
    per_stroke = result.get("per_stroke", {})
    for stroke in sorted(set(list(per_stroke) + list(coverage))):
        for kf in _KF_ORDER:
            n_gt = coverage.get(stroke, {}).get(kf, 0)
            if n_gt == 0:
                lines.append(f"| {stroke} | {kf} | 0 | — (no GT) | — | — |")
                continue
            s = per_stroke.get(stroke, {}).get(kf)
            if s is None:
                lines.append(f"| {stroke} | {kf} | {n_gt} | — (no pred) | — | — |")
            else:
                lines.append(f"| {stroke} | {kf} | {s['n']} | "
                             f"{_fmt(s['accuracy'])} | "
                             f"{_fmt(s.get('window_frames'), 1)} | "
                             f"{_fmt(s.get('mae_frames'), 2)} |")
    return lines


def render_keyframe_markdown(results: dict) -> str:
    """รายงานเต็ม — results มาจาก scripts/run_keyframe_benchmark.py score"""
    mode_a = results["mode_a"]
    mode_b = results.get("mode_b")
    mode_a_matched = results.get("mode_a_matched")
    phase_a = results.get("mode_a_phase")
    phase_b = results.get("mode_b_phase")
    det = results["detection"]
    meta = results.get("meta", {})

    acc = mode_a["acceptance_accuracy"]
    strict_verdict = "PASS ✅" if (acc is not None and acc >= TOR_TARGET) else "FAIL ⚠️"

    L = ["# Benchmark — Keyframe Accuracy vs Ground Truth", "", "## Verdict", ""]

    tol = meta.get('accuracy_tolerance', 1)
    secs = tol / 30.0
    L += [
        f"**Accuracy: {_fmt(acc)}** (n={mode_a['acceptance_n']}, "
        f"เป้า TOR ≥ {TOR_TARGET}) — **{strict_verdict}**",
        "",
        f"ตัวเลขหลักอิงตามเกณฑ์การอนุโลมความคลาดเคลื่อนที่ **±{tol} เฟรม (ประมาณ ±{secs:.1f} วินาที)**",
        "",
        "> *การตั้งความคลาดเคลื่อนในระดับวินาทีช่วยครอบคลุมกรณีภาพเบลอ หรือลูกเทนนิสบังมิดจากมุมกล้อง*"
        "",
        "**หมายเหตุ**: ลูกค้ามี Guideline เดิมเรื่องเกณฑ์ \"อยู่ในช่วงการเคลื่อนไหวเดียวกัน\" (Phase-based)",
        "ซึ่งสามารถดูผลเปรียบเทียบในตารางด้านล่างได้",
        "",
    ]

    L += [
        "| ตัวเลข | เกณฑ์ | Accuracy | n |",
        "|---|---|---|---|",
    ]
    if phase_a:
        L.append(f"| **Mode A end-to-end** | ช่วงเดียวกัน (ตามลูกค้า) | "
                 f"**{_fmt(phase_a['acceptance_accuracy'])}** | {phase_a['acceptance_n']} |")
    if phase_b:
        L.append(f"| Mode B oracle-impact | ช่วงเดียวกัน (ตามลูกค้า) | "
                 f"{_fmt(phase_b['acceptance_accuracy'])} | {phase_b['acceptance_n']} |")
    L.append(f"| Mode A end-to-end | ±{meta.get('accuracy_tolerance', 1)} เฟรม "
             f"(เข้มกว่าที่ลูกค้าขอ) | {_fmt(acc)} | {mode_a['acceptance_n']} |")
    if mode_a_matched:
        L.append(f"| Mode A matched-only | ±{meta.get('accuracy_tolerance', 1)} เฟรม "
                 f"เฉพาะ stroke ที่ detect เจอ (สูงหลอก) "
                 f"| {_fmt(mode_a_matched['acceptance_accuracy'])} | {mode_a_matched['acceptance_n']} |")
    if mode_b:
        L.append(f"| Mode B oracle-impact | ±{meta.get('accuracy_tolerance', 1)} เฟรม "
                 f"= เพดานของ keyframe logic "
                 f"| {_fmt(mode_b['acceptance_accuracy'])} | {mode_b['acceptance_n']} |")

    L += [
        "",
        f"Mode A end-to-end = จริงตามที่ลูกค้าได้ (stroke ที่ hit detection "
        f"หาไม่เจอ นับเป็นผิด ไม่ตัดออกจากตัวหาร) · เกณฑ์รวม = "
        f"impact + backswing_peak ตาม spec",
        "",
        f"เทียบเป้า **TOR ≥ {TOR_TARGET}**: เกณฑ์ ±{meta.get('accuracy_tolerance', 1)}"
        f" เฟรมให้ {_fmt(acc)} = **{strict_verdict}**",
    ]

    L += [
        "",
        "## Provenance",
        "",
        f"- Ground truth: {meta.get('n_label_files', '?')} ไฟล์ / "
        f"{meta.get('n_gt_strokes', '?')} stroke / annotator: {meta.get('annotators', '?')}",
        f"- fps ต้นทาง: {meta.get('fps', '?')} · ความยาวรวม ~{meta.get('total_minutes', '?')} นาที",
        f"- Checkpoint: `{meta.get('checkpoint', '?')}`",
        f"- git: `{meta.get('git_sha', '?')}` · วันที่รัน: {meta.get('run_date', '?')}",
        f"- Accuracy tolerance: ±{meta.get('accuracy_tolerance', 1)} เฟรม · "
        f"Match tolerance: ±{meta.get('match_tolerance', MATCH_TOLERANCE_FRAMES)} เฟรม",
        f"- **การประเมิน hit classifier: {meta.get('hit_model_eval', '(ไม่ระบุ)')}**",
        f"- ปรับเฟรมปะทะด้วยเสียง: {meta.get('impact_audio_refine', '(ไม่ระบุ)')}",
        "",
        "  > ⚠️ ตัวเลขต่างกันมากระหว่างสองโหมด — โหมด in-sample ให้ acceptance "
        "0.506 แต่โหมด LOPO ให้ 0.308 บนข้อมูลชุดเดียวกัน ส่วนต่างคือ "
        "\"การจำคลิป\" ที่ปนอยู่ (`hit_classifier.pkl` เทรนจากคลิปเดียวกับที่ใช้วัด) "
        "**ตัวเลขที่ควรบอกลูกค้าคือโหมด LOPO** เพราะเป็นตัวที่บอกได้ว่าใช้กับ"
        "ผู้เล่นคนใหม่แล้วจะเป็นยังไง · Mode B ไม่กระทบ (ป้อน impact จากเฉลย "
        "ไม่ได้ใช้ classifier)",
        "",
        "## Detection (Mode A)",
        "",
        f"- TP **{det['tp']}** / FN **{det['fn']}** / FP **{det['fp']}**",
        f"- Recall **{_fmt(det['recall'])}** · Precision **{_fmt(det['precision'])}** "
        f"· FP ต่อนาที {_fmt(det['fp_per_minute'], 2)}",
        f"- Δ (pred − gt) เฉลี่ย {_fmt(det['delta_mean'], 1)} เฟรม · "
        f"median {_fmt(det['delta_median'], 1)} · p90 |Δ| {_fmt(det['delta_p90'], 1)}",
    ]
    if det.get("fn") and det.get("fn_nearest_median") is not None:
        L += [
            f"- GT ที่พลาด ({det['fn']} ตัว): ระยะถึง prediction ที่ใกล้สุด median "
            f"**{_fmt(det['fn_nearest_median'], 1)} เฟรม** · "
            f"{det.get('fn_within_2x_tolerance', 0)} ตัวอยู่ในระยะ "
            f"{2 * MATCH_TOLERANCE_FRAMES} เฟรม",
            "",
            "  > ตีความ: ถ้าระยะใกล้ ๆ = pipeline \"เห็น\" stroke แต่ระบุเฟรมเพี้ยน"
            " (ไปแก้ความแม่นของ impact detection) · ถ้าไกลมาก = พลาด stroke ไปเลย"
            " (ไปแก้ recall/candidate generation) — คนละปัญหาคนละทางแก้",
        ]
    L += [
        "",
        "| Clip | GT | Pred | TP | FN | FP |",
        "|---|---|---|---|---|---|",
    ]
    for c in det["per_clip"]:
        L.append(f"| {c['clip']} | {c['n_gt']} | {c['n_pred']} | "
                 f"{c['tp']} | {c['fn']} | {c['fp']} |")

    if phase_a:
        L += ["", "## Keyframe accuracy — เกณฑ์ \"ช่วงเดียวกัน\" (ตามลูกค้า)", "",
              "คอลัมน์ `หน้าต่าง` = ครึ่งความกว้างเฉลี่ยของช่วงที่ยอมรับ "
              "(เฟรม) — คำนวณจาก GT ของ stroke นั้นเอง ไม่ได้ตั้งค่าไว้ล่วงหน้า",
              ""]
        L += _phase_accuracy_table(phase_a, phase_a.get("coverage", {}),
                                   "Mode A")
        if phase_b:
            L += ["", "### Mode B (oracle impact) — เกณฑ์เดียวกัน", ""]
            L += _phase_accuracy_table(phase_b, phase_b.get("coverage", {}),
                                       "Mode B")

    L += ["", f"## Keyframe accuracy — เกณฑ์ ±{meta.get('accuracy_tolerance', 1)} เฟรม "
          "(อ้างอิงแบบเข้ม)", "", "### Mode A (end-to-end)", ""]
    L += _accuracy_table(mode_a, mode_a.get("coverage", {}))

    if mode_b:
        L += ["", "### Mode B (oracle impact)", ""]
        L += _accuracy_table(mode_b, mode_b.get("coverage", {}))

    L += [
        "",
        "## GT coverage — keyframe ไหนมีเฉลยบ้าง",
        "",
        "GT ไม่ได้ label ครบทุก keyframe ทุก stroke **โดยตั้งใจ** — ช่อง `— (no GT)` "
        "ด้านบนคือ *ไม่ได้วัด* ไม่ใช่สอบตก:",
        "",
        "- `unit_turn` ไม่มีใน SV เลย (serve ใช้ trophy_position แทน ตาม spec)",
        "- `trophy_position` มีเฉพาะ SV",
        "- `recovery_position` มีเมื่อ `recovery_restored = true` เท่านั้น",
        "",
    ]

    if results.get("errored_clips"):
        L += [
            "## ⚠️ คลิปที่ประมวลผลล้ม",
            "",
            "GT ของคลิปเหล่านี้ถูกนับเป็น FN ทั้งก้อน (ไม่ได้ตัดออกจากตัวหาร) — "
            "ตัวเลข accuracy จึงถูกกดลงด้วยความล้มเหลวเชิงเทคนิค ไม่ใช่ความแม่น "
            "ของโมเดลเพียว ๆ:",
            "",
        ] + [f"- `{e['clip']}` [{e['stage']}]: {e.get('error')}"
             for e in results["errored_clips"]] + [""]

    if results.get("quirks"):
        L += ["## ข้อจำกัดเชิงโค้ดที่ทราบล่วงหน้า", ""] + \
             [f"- {q}" for q in results["quirks"]] + [""]

    if results.get("sensitivity"):
        s = results["sensitivity"]
        L += [
            "## Sensitivity — ความผิดปกติใน GT เอง",
            "",
            f"พบ label ที่ลำดับเวลาผิด {results.get('n_anomalies', '?')} จุด "
            "(backswing อยู่หลัง/เท่ากับ impact, follow-through อยู่ก่อน impact, ไม่มี follow-through) — "
            "ตัวเลขเมื่อ null เฉพาะค่าที่ผิด (ไม่ตัด stroke ทั้งตัว): "
            f"**{_fmt(s['acceptance_accuracy'])}** (n={s['acceptance_n']})",
            "",
        ]

    n_fail = len(mode_a.get("failures", []))
    L += [
        "## Failure analysis",
        "",
        f"failures ทั้งหมด {n_fail} · GT ที่ไม่มี prediction "
        f"{mode_a.get('missing_predictions', 0)}",
        "",
        "รายละเอียดเต็ม (per-stroke delta, error ต่อ keyframe, anomaly flags) ใน "
        "`docs/benchmark_keyframe_results.json`",
        "",
    ]
    return "\n".join(L) + "\n"
