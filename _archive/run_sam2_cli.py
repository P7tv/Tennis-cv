import sys
import json
import argparse
import os

# Add parent directory to path so we can import webui.sam2_track
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from webui.sam2_track import track_player_with_sam2, ClickPrompt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--checkpoint", required=True)
    # SAM 2 pre-allocates (chunk_size × 1024 × 1024 × 3 × 4 bytes) on CPU.
    # 100 frames → 1.26 GB (default, safe when ≥6 GB free RAM + reset_state between chunks).
    # 30 frames  → 377 MB  (fallback for RAM-constrained machines, but ~3× slower).
    parser.add_argument("--chunk-size", type=int, default=100,
                        help="Frames per SAM 2 inference chunk (default: 100). "
                             "Lower = less RAM, more chunks. 100 ≈ 1.26 GB CPU.")
    args = parser.parse_args()

    with open(args.prompts, "r") as f:
        prompts_data = json.load(f)
    
    prompts = [ClickPrompt(**p) for p in prompts_data]
    
    print(f"Running SAM2 tracking on {args.video} with {len(prompts)} prompts, chunk_size={args.chunk_size}...")
    
    bboxes = track_player_with_sam2(
        video_path=args.video,
        prompts=prompts,
        checkpoint=args.checkpoint,
        config_name="configs/sam2.1/sam2.1_hiera_t.yaml",
        chunk_size=args.chunk_size,
    )
    
    with open(args.out, "w") as f:
        json.dump(bboxes, f)
        
    print(f"Tracking complete. Tracked {len(bboxes)} frames. Saved to {args.out}")

if __name__ == "__main__":
    main()
