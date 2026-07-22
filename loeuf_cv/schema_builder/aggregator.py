"""schema_builder เดิมมี AGG/TRE/PAT/SUM แบบ mock ของตัวเอง — แต่
loeuf_cv/aggregation.py (ของเดิม, ใช้กับ StrokePipeline) implement ครบ
ตาม schema doc อยู่แล้ว (improving/declining/volatile/stable trend states,
consistency score, fatigue detection, serve fault rate จาก BL) แค่คาด
stroke dict รูปแบบ nested ("metrics": {body, ..., kinematics,
stroke_specific, ball}) ต่างจากที่ build_loeuf_schema ประกอบ (flat keys:
"metric"/"kinematics"/"stroke_specific"/"ball") — ฟังก์ชันนี้แค่ reshape
เป็น view ชั่วคราวให้ตรงรูปแบบเดิมแล้วเรียก loeuf_cv/aggregation.py ตรง ๆ
ไม่เขียนตรรกะซ้ำ"""

from ..aggregation import (
    build_aggregated_metric, build_pattern, build_session_summary, build_trend,
)
from ..config import PipelineConfig


def _as_metrics_nested(stroke: dict) -> dict:
    metrics = dict(stroke.get("metric", {}))
    metrics["kinematics"] = stroke.get("kinematics", {})
    metrics["stroke_specific"] = stroke.get("stroke_specific", {})
    metrics["ball"] = stroke.get("ball", {})
    view = dict(stroke)
    view["metrics"] = metrics
    return view


def build_phase2_aggregations(strokes: list[dict], config: PipelineConfig | None = None):
    if not strokes:
        return [], {}, {}, {}

    config = config or PipelineConfig()
    view = [_as_metrics_nested(s) for s in strokes]

    agg = build_aggregated_metric(view)
    trend = build_trend(view)
    pattern = build_pattern(view)
    summary = build_session_summary(view, trend, config)
    return agg, trend, pattern, summary
