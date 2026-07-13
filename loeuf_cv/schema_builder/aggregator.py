import numpy as np

def build_phase2_aggregations(strokes):
    """
    สร้างข้อมูล Phase 2 (Session-Level Intelligence) จาก list ของ strokes
    """
    if not strokes:
        return [], {}, {}, {}
        
    agg = []
    
    # 1. Group by stroke type
    groups = {}
    for i, s in enumerate(strokes):
        stype = s.get("stroke_root", {}).get("stroke_type", "FH")
        if stype not in groups:
            groups[stype] = []
        # Inject stroke index to trace back easily
        s["_index"] = i
        groups[stype].append(s)
        
    # 2. Calculate Aggregations
    for stype, items in groups.items():
        count = len(items)
        
        # Collect shoulder rotation at impact (C4)
        c4_list = [item["metric"].get("body", {}).get("shoulder_rotation_at_impact_deg") for item in items]
        c4_list = [x for x in c4_list if x is not None]
        
        c16_list = [item["metric"].get("contact", {}).get("contact_height_cm") for item in items]
        c16_list = [x for x in c16_list if x is not None]
        
        agg.append({
            "stroke_type": stype,
            "stroke_count": count,
            "shoulder_rotation_mean_deg": round(np.mean(c4_list), 1) if c4_list else None,
            "shoulder_rotation_sd_deg": round(np.std(c4_list), 1) if c4_list else None,
            "contact_height_mean_cm": round(np.mean(c16_list), 1) if c16_list else None,
            # Placeholder for other aggregated metrics
            "split_step_rate_pct": 100.0,
            "recovery_time_mean_ms": 680.0
        })
        
    # 3. Trends (Split session into thirds)
    # Simple mock implementation for TRE
    trend = {
        "window_method": "thirds",
        "shoulder_rotation_trend": "stable",
        "tempo_ratio_trend": "stable",
        "consistency_score": 0.85,
        "fatigue_flag": False,
        "fatigue_onset_stroke_index": None
    }
    
    # 4. Pattern
    # Count dominant stroke
    dominant = max(groups.keys(), key=lambda k: len(groups[k])) if groups else "FH"
    pattern = {
        "dominant_stroke_type": dominant,
        "stroke_sequence": [s.get("stroke_root", {}).get("stroke_type", "FH") for s in strokes],
        "groundstroke_sequence_detected": False,
        "serve_fault_rate_pct": None,
        "backhand_avoidance_flag": False
    }
    
    # 5. Summary
    summary = {
        "session_quality": "good",
        "primary_issue": None,
        "highlight_stroke_index": 1,
        "recommended_drill_tag": None,
        "coach_flag": False
    }
    
    return agg, trend, pattern, summary
