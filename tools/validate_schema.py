import json
import argparse
from pathlib import Path
import sys

# Define the expected schema based on test.pdf

ROOT_KEYS = [
    'session_metadata', 'strokes', 'aggregated_metric', 
    'trend', 'pattern', 'session_summary'
]

STROKE_KEYS = [
    'stroke_metadata', 'video_quality', 'keyframe', 
    'stroke_root', 'metric', 
    'derived', 'visibility_flag', 'confidence_flag', 'visualization'
]

# Coaching metrics (C1-C41) grouped by sub-blocks in JSON
METRICS_BODY = [
    'body_height_cm', 'arm_span_cm', 'shoulder_rotation_at_unit_turn_deg', 
    'shoulder_rotation_at_impact_deg', 'shoulder_rotation_normalized_pct',
    'hip_rotation_at_unit_turn_deg', 'hip_rotation_at_impact_deg',
    'shoulder_hip_separation_at_impact_deg'
]

METRICS_ARM = ['follow_through_angle_deg', 'backswing_depth_deg']
METRICS_TIMING = ['backswing_duration_ms', 'forward_swing_duration_ms', 'total_swing_duration_ms', 'tempo_ratio', 'rhythm_class']
METRICS_CONTACT = ['contact_height_cm', 'contact_distance_from_body_cm', 'swing_path_angle_deg']
METRICS_MOVEMENT = [
    'recovery_time_ms', 'return_to_ready_frames', 'recovery_position',
    'split_step_detected', 'split_step_timing_ms', 'footwork_distance_cm',
    'movement_speed_mps', 'stance_width_at_impact_cm', 'stance_angle_deg',
    'head_still_at_contact'
]

METRICS_DEPTH = [
    'elbow_angle_at_impact_deg', 'elbow_angle_at_backswing_peak_deg',
    'wrist_angle_at_impact_deg', 'spine_tilt_at_impact_deg',
    'com_shift_distance_cm', 'com_deviation_at_impact_cm', 'com_deviation_at_finish_cm',
    'weight_transfer_timing_offset_ms', 'rear_leg_load_pct', 'front_leg_transfer_pct'
]

METRICS_KINEMATICS = [
    'hip_rotation_velocity_deg_s', 'shoulder_rotation_velocity_deg_s',
    'hip_shoulder_velocity_lag_ms', 'center_of_mass_velocity_mps',
    'racket_tip_position', 'racket_center_position', 'racket_head_speed_mps'
]

METRICS_SS = [
    'toss_deviation_cm', 'trophy_position_achieved', 'leg_drive_detected',
    'peak_contact_height_cm', 'stance_type', 'backswing_past_ear',
    'hands_height_at_contact', 'punch_forward_detected', 'swing_direction_verified'
]

ALL_METRICS = METRICS_BODY + METRICS_ARM + METRICS_TIMING + METRICS_CONTACT + METRICS_MOVEMENT + METRICS_DEPTH + METRICS_KINEMATICS + METRICS_SS

def check_keys(data, expected_keys, path):
    errors = []
    if not isinstance(data, dict):
        return [f"[{path}] Expected dict, got {type(data).__name__}"]
    
    found_keys = set(data.keys())
    missing = set(expected_keys) - found_keys
    if missing:
        errors.append(f"[{path}] Missing keys: {missing}")
        
    return errors

def validate_stroke(stroke, i):
    errors = []
    path = f"strokes[{i}]"
    
    errors.extend(check_keys(stroke, STROKE_KEYS, path))
    
    if 'metric' in stroke:
        m = stroke['metric']
        errors.extend(check_keys(m, ['body', 'arm', 'timing', 'contact', 'movement', 'depth_estimated', 'kinematics'], f"{path}.metric"))
        if 'body' in m: errors.extend(check_keys(m['body'], METRICS_BODY, f"{path}.metric.body"))
        if 'arm' in m: errors.extend(check_keys(m['arm'], METRICS_ARM, f"{path}.metric.arm"))
        if 'timing' in m: errors.extend(check_keys(m['timing'], METRICS_TIMING, f"{path}.metric.timing"))
        if 'contact' in m: errors.extend(check_keys(m['contact'], METRICS_CONTACT, f"{path}.metric.contact"))
        if 'movement' in m: errors.extend(check_keys(m['movement'], METRICS_MOVEMENT, f"{path}.metric.movement"))
        if 'depth_estimated' in m: errors.extend(check_keys(m['depth_estimated'], METRICS_DEPTH, f"{path}.metric.depth_estimated"))
        if 'kinematics' in m: errors.extend(check_keys(m['kinematics'], METRICS_KINEMATICS, f"{path}.metric.kinematics"))
        
        # Stroke specific is optional/varies but shouldn't miss if its keys are expected. We just check if it's there.
        if 'stroke_specific' in m: 
            # Note: stroke_specific is only for SV, VL, SL. We check ALL for now.
            # Real implementation might check stroke type.
            errors.extend(check_keys(m['stroke_specific'], METRICS_SS, f"{path}.metric.stroke_specific"))
        
    if 'visibility_flag' in stroke:
        vf = stroke['visibility_flag']
        errors.extend(check_keys(vf, ALL_METRICS, f"{path}.visibility_flag"))
        
        # Null Rule: If field is null, visibility must be not_detectable
        if 'metric' in stroke:
            for cat in stroke['metric'].values():
                if not isinstance(cat, dict): continue
                for key, val in cat.items():
                    if key in vf:
                        vis = vf[key]
                        if val is None and vis != 'not_detectable':
                            errors.append(f"[{path}.visibility_flag] {key} is null but visibility is '{vis}' (expected 'not_detectable')")
                        if val is not None and vis == 'not_detectable':
                            errors.append(f"[{path}.visibility_flag] {key} is NOT null but visibility is 'not_detectable'")

    return errors

def validate_schema(filepath):
    print(f"Validating {filepath} against CV Schema (test.pdf)...")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    errors = []
    
    errors.extend(check_keys(data, ROOT_KEYS, "ROOT"))
    
    if 'strokes' in data and len(data['strokes']) > 0:
        first = data['strokes'][0]
        if 'keyframe' not in first and 'keyframes' in first:
            print("WARNING: Schema expects 'keyframe' but JSON uses 'keyframes'")
        if 'metric' not in first and 'metrics' in first:
            print("WARNING: Schema expects 'metric' but JSON uses 'metrics'")
            
    if 'strokes' in data:
        strokes = data['strokes']
        for i, s in enumerate(strokes):
            errors.extend(validate_stroke(s, i))
            
    if errors:
        print(f"[FAIL] Validation Failed. Found {len(errors)} issues:")
        for e in errors:
            print("  ", e)
        return False
    else:
        print("[PASS] Validation Passed! JSON conforms to CV Schema.")
        return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('json_file')
    args = parser.parse_args()
    
    success = validate_schema(args.json_file)
    sys.exit(0 if success else 1)
