"""SAM 2 click-to-track — Track 2 prototype (action plan 2026-07-10)

แก้ปัญหาหมวด B โดยตรง: "แยกผู้เล่นจากคนป้อนบอล/คนรอไม่ได้" +
"คนเดียวถูกตัด track เป็นหลายท่อน" — ที่ multi_person.py แก้ด้วย classical
CV (background subtraction + centroid tracking + clustering) มาทั้งวันและ
เจอบั๊กซ้อนบั๊ก เพราะ blob-tracking ไม่รู้ "identity" จริง รู้แค่ "มีอะไร
ขยับตรงไหน"

แนวคิด: คนคลิกจุดบนตัวผู้เล่นเป้าหมายที่เฟรมแรก (หรือกี่จุดก็ได้) → SAM 2
video predictor propagate mask ของวัตถุนั้นไปทั้งคลิป (โมเดลที่ออกแบบมา
เพื่อ track ผ่าน occlusion โดยเฉพาะ — ตรงกับปัญหาที่เราแก้ด้วยมือทั้งวัน)
→ แปลง mask เป็น bounding box ต่อเฟรม → ป้อนเข้า MediaPipe Pose เดิม
(reuse _crop_and_run_pose จาก multi_person.py) — เป็น drop-in แทน
detect_blobs()+match_tracks()+select_candidate_tracks() ทั้งชุด

⚠️ รันในเครื่องนี้ (Apple Silicon, ไม่มี CUDA) ใช้ MPS/CPU — ช้ากว่า GPU
จริงมาก ใช้เพื่อ "ตรวจสอบว่าโค้ดถูกต้อง" ก่อนย้ายไปรันจริงบนเครื่องที่มี
GPU (Windows/CUDA) — ดู webui/README.md

รันแยก venv (webui/.venv-sam2/) เพราะ SAM 2 ดึง torch เวอร์ชันเฉพาะมาด้วย
ซึ่งอาจชนกับ torch ที่ environment หลักของโปรเจกต์ใช้อยู่แล้ว (MediaPipe)
"""

from dataclasses import dataclass

import numpy as np

DEFAULT_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"  # tiny — เร็วสุด เหมาะ CPU/MPS ก่อน


@dataclass
class ClickPrompt:
    """จุดที่คนคลิกบนตัวผู้เล่นเป้าหมาย"""
    frame_idx: int
    x: int
    y: int
    positive: bool = True  # True = "ใช่คนนี้", False = "ไม่ใช่" (refine)


def mask_to_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """แปลง binary mask (H,W) เป็น (x,y,w,h) — คืน None ถ้า mask ว่างเปล่า"""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return (x0, y0, x1 - x0 + 1, y1 - y0 + 1)


def _pick_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _get_ffmpeg_path() -> str:
    import shutil
    import os
    import glob
    path = shutil.which("ffmpeg")
    if path: return path
    
    # Fallback for Windows if PATH hasn't updated after winget install
    localappdata = os.environ.get("LOCALAPPDATA", "")
    pattern = os.path.join(localappdata, "Microsoft", "WinGet", "Packages", "Gyan.FFmpeg*", "*", "bin", "ffmpeg.exe")
    matches = glob.glob(pattern)
    if matches: return matches[0]
    return "ffmpeg"

def _extract_jpeg_frames(video_path: str, out_dir: str) -> int:
    """แตกวิดีโอเป็น JPEG ต่อเฟรม (00000.jpg, 00001.jpg, ...) โดยใช้ ffmpeg (เร็วกว่า cv2 มาก)"""
    import os
    import subprocess
    import glob

    os.makedirs(out_dir, exist_ok=True)
    
    # Run FFmpeg to extract frames using optimal qscale for JPEG
    try:
        subprocess.run([
            _get_ffmpeg_path(), "-y", "-i", video_path, 
            "-qscale:v", "2", 
            "-start_number", "0",
            os.path.join(out_dir, "%05d.jpg")
        ], check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFMPEG frame extraction failed: {e.stderr.decode('utf-8', errors='replace')}")

    # Count the number of extracted frames
    frames = glob.glob(os.path.join(out_dir, "*.jpg"))
    return len(frames)


def track_player_with_sam2(video_path: str, prompts: list[ClickPrompt],
                           checkpoint: str, config_name: str = DEFAULT_CONFIG,
                           obj_id: int = 1, device: str | None = None,
                           progress_callback=None, chunk_size: int = 150) -> dict[int, tuple[int, int, int, int]]:
    """คลิกจุดบนผู้เล่นเป้าหมาย → คืน {frame_idx: (x,y,w,h)} ของทุกเฟรมที่
    SAM 2 propagate mask ได้ (ไม่ว่างเปล่า) — ทิศทางเดียว (forward จาก
    เฟรมแรกสุดที่มี prompt) ตาม use case "คลิกเฟรมแรก track ไปข้างหน้า"
    ใช้ Chunking Processing เพื่อเลี่ยงปัญหา Memory OOM:
      chunk_size=100 → ~1.26 GB CPU RAM ต่อ chunk (default, ปลอดภัยกับ RAM ≥8 GB)
      chunk_size=30  → ~377 MB ต่อ chunk (ช้ากว่าเพราะ init overhead เยอะขึ้น)
    ระหว่าง chunk จะเรียก reset_state() + empty_cache() เพื่อคืน memory ก่อน init chunk ถัดไป
    """
    import os
    import shutil
    import tempfile
    import contextlib
    import torch

    from sam2.build_sam import build_sam2_video_predictor
    
    device = device or _pick_device()
    
    if "cuda" in device:
        torch.backends.cudnn.benchmark = True
    
    # Enable autocast (half precision) to double the speed and halve the VRAM usage
    # We use float16 because GTX 1660 (Turing) supports FP16 Tensor Cores natively.
    autocast_context = torch.autocast("cuda", dtype=torch.float16) if "cuda" in device else contextlib.nullcontext()
    
    with autocast_context:
        predictor = build_sam2_video_predictor(config_name, checkpoint, device=device)
    
        # Use a temp directory inside the webui directory (on the D: drive) to avoid C: drive space limits
        temp_base_dir = os.path.join(os.path.dirname(__file__), ".tmp")
        os.makedirs(temp_base_dir, exist_ok=True)

    frames_dir = tempfile.mkdtemp(prefix="sam2_frames_", dir=temp_base_dir)
    try:
        total_frames = _extract_jpeg_frames(video_path, frames_dir)

        by_frame: dict[int, list[ClickPrompt]] = {}
        for p in prompts:
            by_frame.setdefault(p.frame_idx, []).append(p)
        
        if not by_frame:
            return {}

        bboxes: dict[int, tuple[int, int, int, int]] = {}
        CHUNK_SIZE = chunk_size
        
        start_frame = min(by_frame)
        current_chunk_start = start_frame
        last_mask = None

        while current_chunk_start < total_frames:
            chunk_end = min(current_chunk_start + CHUNK_SIZE, total_frames)
            chunk_length = chunk_end - current_chunk_start
            
            # Create chunk dir and copy frames (symlinks fail on Windows without admin)
            chunk_dir = tempfile.mkdtemp(prefix=f"sam2_chunk_{current_chunk_start}_", dir=temp_base_dir)
            with autocast_context:
                try:
                    for local_idx, global_idx in enumerate(range(current_chunk_start, chunk_end)):
                        src = os.path.join(frames_dir, f"{global_idx:05d}.jpg")
                        dst = os.path.join(chunk_dir, f"{local_idx:05d}.jpg")
                        try:
                            os.link(src, dst)
                        except OSError:
                            shutil.copy(src, dst)
                    
                    state = predictor.init_state(video_path=chunk_dir)
                    
                    # Pass mask from previous chunk if it exists
                    if last_mask is not None:
                        # add_new_mask expects boolean mask [H, W] (2 dimensions)
                        predictor.add_new_mask(state, frame_idx=0, obj_id=obj_id, mask=(last_mask.squeeze() > 0.0))
                        
                    # Add manual prompts that fall in this chunk
                    chunk_prompts = {k: v for k, v in by_frame.items() if current_chunk_start <= k < chunk_end}
                    for global_frame_idx, pts in chunk_prompts.items():
                        local_frame_idx = global_frame_idx - current_chunk_start
                        points = np.array([[p.x, p.y] for p in pts], dtype=np.float32)
                        labels = np.array([1 if p.positive else 0 for p in pts], dtype=np.int32)
                        predictor.add_new_points_or_box(
                            inference_state=state, frame_idx=local_frame_idx, obj_id=obj_id,
                            points=points, labels=labels)
                            
                    # Determine where to start propagation for this chunk
                    if last_mask is not None:
                        prop_start = 0
                    else:
                        prop_start = min(chunk_prompts.keys()) - current_chunk_start
                        
                    chunk_last_mask = None
                    for local_frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(
                            state, start_frame_idx=prop_start):
                        if obj_id not in obj_ids:
                            continue
                        i = obj_ids.index(obj_id)
                        mask_logit = mask_logits[i]
                        mask_binary = (mask_logit > 0.0).cpu().numpy().squeeze()
                        bbox = mask_to_bbox(mask_binary)
                        
                        global_frame_idx = current_chunk_start + local_frame_idx
                        if bbox is not None:
                            bboxes[global_frame_idx] = bbox
                            
                        if progress_callback:
                            progress_callback(global_frame_idx, total_frames)
                            
                        # Capture the mask at the end of the chunk to pass to the next one
                        if local_frame_idx == chunk_length - 1:
                            chunk_last_mask = mask_logit.clone()
                finally:
                    # Explicitly release SAM 2's per-chunk state before next init_state().
                    # Without this, PyTorch holds the previous chunk's feature maps in
                    # memory while trying to allocate the next chunk → OOM on chunk 2+
                    # even when chunk 1 fit fine.
                    try:
                        predictor.reset_state(state)
                    except Exception:
                        pass
                    try:
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:
                        pass
                    shutil.rmtree(chunk_dir, ignore_errors=True)
    
                # Overlap by 1 frame so we can pass the mask at local frame 0 of the next chunk
                if chunk_end == total_frames:
                    break
                    
                last_mask = chunk_last_mask
                current_chunk_start = chunk_end - 1 

    finally:
        shutil.rmtree(frames_dir, ignore_errors=True)

    return bboxes


def extract_pose_from_sam2_track(video_path: str, bboxes: dict[int, tuple],
                                 config) -> "PoseTimeseries":
    """ป้อน bbox ที่ SAM 2 track ได้เข้า MediaPipe Pose เดิม (reuse
    _crop_and_run_pose ของ multi_person.py) — คืน PoseTimeseries เดียว
    ต่อเนื่อง (ไม่ต้องมี track stitching/clustering เพราะ SAM 2 รักษา
    identity ให้แล้วตั้งแต่ต้น)"""
    import cv2
    import mediapipe as mp

    from loeuf_cv.multi_person import _crop_and_run_pose, crop_landmarks_to_frame
    from loeuf_cv.pose_extractor import FrameStats, PoseTimeseries, read_video_meta

    meta = read_video_meta(video_path)
    cap = cv2.VideoCapture(video_path)
    pose_model = mp.solutions.pose.Pose(
        static_image_mode=True, model_complexity=config.model_complexity,
        min_detection_confidence=config.min_detection_confidence)

    T = meta.frame_count
    landmarks = np.full((T, 33, 3), np.nan)
    world_landmarks = np.full((T, 33, 3), np.nan)
    visibility = np.zeros((T, 33))
    height_frac = np.zeros(T)

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        box = bboxes.get(frame_idx)
        if box is not None:
            result, crop_box = _crop_and_run_pose(pose_model, frame, box)
            if result is not None and result.pose_landmarks:
                lm = np.array([[p.x, p.y, p.z] for p in result.pose_landmarks.landmark])
                vis = np.array([p.visibility for p in result.pose_landmarks.landmark])
                wlm = np.array([[p.x, p.y, p.z]
                               for p in result.pose_world_landmarks.landmark])
                lm[:, :2] = crop_landmarks_to_frame(lm[:, :2], crop_box,
                                                    meta.width, meta.height)
                landmarks[frame_idx] = lm
                world_landmarks[frame_idx] = wlm
                visibility[frame_idx] = vis
                height_frac[frame_idx] = box[3] / meta.height
        frame_idx += 1

    cap.release()
    pose_model.close()

    return PoseTimeseries(
        landmarks=landmarks, world_landmarks=world_landmarks, visibility=visibility,
        timestamps_ms=np.arange(T) / meta.fps * 1000.0, meta=meta,
        frame_stats=FrameStats(person_height_frac=height_frac))
