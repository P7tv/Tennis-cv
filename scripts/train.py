import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description="Loeuf CV - Unified Training Entry Point")
    parser.add_argument("--task", choices=["hit_classifier", "stroke_classifier", "impact_refiner", "yolo"], required=True, help="Model to train")
    parser.add_argument("--config", default=str(ROOT / "configs/train.yaml"), help="Path to config file")
    
    # Parse known args, pass the rest to the sub-script
    args, unknown = parser.parse_known_args()
    
    # Map tasks to their actual script names
    task_map = {
        "hit_classifier": "train_hit_classifier_from_cache.py",
        "stroke_classifier": "train_stroke_classifier_from_cache.py",
        "impact_refiner": "train_impact_refiner.py",
        "yolo": "train_yolo.py"
    }
    
    script_path = ROOT / "tools" / task_map[args.task]
    if not script_path.exists():
        print(f"Error: Script {script_path} not found.")
        sys.exit(1)
        
    cmd = [sys.executable, str(script_path)] + unknown
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    sys.exit(result.returncode)

if __name__ == "__main__":
    main()
