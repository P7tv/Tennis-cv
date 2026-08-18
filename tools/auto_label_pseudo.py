import os
import shutil
from pathlib import Path
import json
import subprocess
import pickle
from datetime import datetime

# Alias module for old pickle files to unpickle correctly
import loeuf_cv.pose_extractor
import sys
sys.modules['loeuf_cv.multi_person'] = loeuf_cv.pose_extractor

from loeuf_cv.hit_detection import detect_hit_events

def migrate_videos(source_dir: str, target_dir: str) -> list[str]:
    """Copies non-rally video files from source to target and returns list of target paths."""
    src = Path(source_dir)
    tgt = Path(target_dir)
    tgt.mkdir(parents=True, exist_ok=True)
    
    copied = []
    if not src.exists():
        print(f"Source dir {src} not found.")
        return copied
        
    for f in src.glob("*.*"):
        if f.suffix.lower() in [".mov", ".mp4"] and "rally" not in f.name.lower():
            target_path = tgt / f.name
            if not target_path.exists():
                shutil.copy2(f, target_path)
            copied.append(str(target_path))
    return copied

def parse_stroke_type(filename: str) -> str:
    """Extracts stroke type like FH, BH, SL, SV, VL from filename."""
    upper_name = filename.upper()
    if 'FH' in upper_name: return 'FH'
    if 'BH' in upper_name: return 'BH'
    if 'SL' in upper_name: return 'SL'
    if 'VL' in upper_name: return 'VL'
    if 'SV' in upper_name: return 'SV'
    return 'UNKNOWN'

def create_json_template(clip_name: str, clip_filename: str) -> dict:
    return {
        "clip_id": clip_name.replace("-", "_"),
        "clip_path": clip_filename,
        "video_id": clip_name,
        "players": [
            {
                "player_id": "auto",
                "name": "auto",
                "height": 170.0,
                "dominant_side": "right"
            }
        ],
        "strokes": []
    }

def create_stroke_entry(stroke_no: int, stroke_type: str, impact_frame: int) -> dict:
    return {
      "stroke_no": stroke_no,
      "player_id": "auto",
      "stroke_index_in_clip": stroke_no,
      "stroke_type": stroke_type,
      "fps": 30.0,
      "label_tier": "core",
      "impact_frame": impact_frame,
      "impact_quality": "ok",
      "impact_ambiguous": False,
      "usable": True,
      "annotator_id": "auto-pseudo",
      "label_date": datetime.now().strftime("%Y-%m-%d")
    }

def _pick_target_track(tracks):
    def coverage(t):
        vis = t.pose.visibility
        return float((vis.mean(axis=1) > 0).sum())
    if not tracks: return None
    return max(tracks, key=coverage)

def process_and_label(target_dir: str):
    tgt = Path(target_dir)
    set_name = tgt.name
    
    # 0. Create empty schema so `run_keyframe_benchmark.py` finds them
    for video_file in tgt.glob("*.*"):
        if video_file.suffix.lower() not in [".mov", ".mp4"]: continue
        clip_name = video_file.stem
        out_json = tgt / f"{clip_name}_stroke_labels_auto.json"
        # Always recreate the template to ensure player exists
        schema = create_json_template(clip_name, f"{set_name}/{video_file.name}")
        with open(out_json, "w") as f:
            json.dump(schema, f, indent=2)
                
    # 1. Run tracking
    print("Running tracking to populate cache...")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd())
    subprocess.run([sys.executable, "tools/run_keyframe_benchmark.py", "track", "--dataset-root", "dataset", "--only", "auto"], env=env)
    
    tracking_dir = Path("dataset/benchmark_cache/tracking") / set_name
    
    # 3. Process each video
    for video_file in tgt.glob("*.*"):
        if video_file.suffix.lower() not in [".mov", ".mp4"]: continue
        
        clip_name = video_file.stem
        stroke_type = parse_stroke_type(clip_name)
        
        pkl_path = tracking_dir / f"{clip_name}.pkl"
        if not pkl_path.exists():
            print(f"Warning: Cache not found for {clip_name}. Skipping.")
            continue
            
        with open(pkl_path, "rb") as f:
            cached = pickle.load(f)
            
        if "tracks" in cached:
            track = _pick_target_track(cached["tracks"])
        else:
            track = cached.get("track")
        if not track:
            print(f"No tracks found for {clip_name}")
            continue
            
        vm = cached["video_meta"]
        fps = cached.get("fps", 30.0)
        ball_traj = cached.get("ball_traj")
        
        if ball_traj is None:
            print(f"No ball_traj found for {clip_name}")
            continue
            
        print(f"Running detect_hit_events for {clip_name}...")
        
        try:
            hits = detect_hit_events(
                ball_traj, [track], fps, vm["width"], vm["height"],
                racket_bboxes=cached.get("racket_bboxes"),
                ball_measured=None, video_path=None
            )
            
            # Create stroke_labels.json
            schema = create_json_template(clip_name, f"{set_name}/{video_file.name}")
            for i, hit in enumerate(hits):
                # Filter out low confidence if needed, but for now take all
                schema["strokes"].append(create_stroke_entry(i+1, stroke_type, hit["frame"]))
            
            out_json = tgt / f"{clip_name}_stroke_labels_auto.json"
            with open(out_json, "w") as f:
                json.dump(schema, f, indent=2)
                
            print(f"Saved {len(hits)} hits to {out_json}")
            
        except Exception as e:
            print(f"Error processing {clip_name}: {e}")

if __name__ == "__main__":
    src_dir = "_incoming/new_data/extracted/260816_set"
    tgt_dir = "dataset/sessions/260816_set_auto"
    
    copied = migrate_videos(src_dir, tgt_dir)
    print(f"Migrated {len(copied)} videos.")
    
    process_and_label(tgt_dir)
