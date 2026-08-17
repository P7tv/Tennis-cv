import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description="Loeuf CV - Data Preprocessing Entry Point")
    parser.add_argument("--task", choices=["build_training_data", "build_pose_dataset", "label_ingest"], required=True, help="Preprocessing task to run")
    parser.add_argument("--config", default=str(ROOT / "configs/train.yaml"), help="Path to config file")
    
    # Parse known args, pass the rest to the sub-script
    args, unknown = parser.parse_known_args()
    
    script_path = ROOT / "tools" / f"{args.task}.py"
    if not script_path.exists():
        print(f"Error: Script {script_path} not found.")
        sys.exit(1)
        
    cmd = [sys.executable, str(script_path)] + unknown
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    sys.exit(result.returncode)

if __name__ == "__main__":
    main()
