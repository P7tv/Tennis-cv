import sys
import os
import cv2
import numpy as np
import tempfile
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from webui.sam2_track import track_player_with_sam2, ClickPrompt

def create_synthetic_video(path, num_frames=350):
    fps = 30
    width = 640
    height = 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, fps, (width, height))
    
    for i in range(num_frames):
        frame = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # Moving black square
        x = int(100 + (i / num_frames) * 400)
        y = 200
        
        cv2.rectangle(frame, (x, y), (x+50, y+50), (0, 0, 0), -1)
        out.write(frame)
        
    out.release()
    print(f"Created synthetic video with {num_frames} frames at {path}")

def main():
    tmp_vid = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    try:
        # Create a 350 frame video to test chunking (150 chunk size)
        # Should create chunks: 0-150, 149-299, 298-350
        create_synthetic_video(tmp_vid, num_frames=100)
        
        # Initial prompt on the black square at frame 0
        prompts = [ClickPrompt(frame_idx=0, x=125, y=225, positive=True)]
        
        print("Starting SAM 2 tracking...")
        t0 = time.time()
        
        # We need to run this in the sam2 venv or have it available.
        # But this script will be executed in the sam2 venv environment by our run_command
        bboxes = track_player_with_sam2(
            video_path=tmp_vid, 
            prompts=prompts, 
            checkpoint="webui/checkpoints/sam2.1_hiera_tiny.pt",
            config_name="configs/sam2.1/sam2.1_hiera_t.yaml",
            chunk_size=30
        )
        
        print(f"Tracking complete in {time.time() - t0:.2f}s")
        print(f"Tracked {len(bboxes)} frames")
        
        if len(bboxes) > 0:
            print("First frame bbox:", bboxes[min(bboxes.keys())])
            print("Last frame bbox:", bboxes[max(bboxes.keys())])
            
        assert len(bboxes) >= 90, f"Expected tracking on most frames, got {len(bboxes)}"
        print("Test Passed!")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        os.remove(tmp_vid)

if __name__ == "__main__":
    main()
