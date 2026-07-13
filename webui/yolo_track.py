"""
webui/yolo_track.py

ตามรอยผู้เล่นด้วย YOLO11n + BoT-SORT (Multi-Object Tracking) แล้วสกัด
โครงกระดูกด้วย MediaPipe Pose — ใช้แทน Classical CV (MOG2) ได้ทันที

ข้อดีเหนือ MOG2:
- ทนต่อการเดินบังกัน (Occlusion) ผ่านระบบ ReID (จดจำสีเสื้อ/รูปร่าง)
- ไม่สับสนกับเงาสนามหรือการเปลี่ยนแสง (YOLO รู้จัก "คน" โดยตรง)
- กิน VRAM เพียง ~1.5GB รันได้ realtime ไม่ต้องแบ่ง Chunk แบบ SAM 2
"""

from __future__ import annotations

import cv2
import numpy as np
import torch

from loeuf_cv.config import PipelineConfig
from loeuf_cv.multi_person import (
    CROP_MARGIN_FRAC,
    UPSCALE_TARGET_PX,
    PlayerTrack,
    _crop_and_run_pose,
    crop_landmarks_to_frame,
)
from loeuf_cv.pose_extractor import FrameStats, PoseTimeseries, VideoMeta, read_video_meta


def track_players_with_yolo(
    video_path: str,
    max_players: int = 2,
    config: PipelineConfig | None = None,
    progress_callback=None,
) -> list[PlayerTrack]:
    """ใช้ YOLO11n + BoT-SORT ตามรอยคนและสกัดโครงกระดูกด้วย MediaPipe Pose

    คืน list[PlayerTrack] เหมือน extract_multi_person เป๊ะๆ เพื่อให้
    ส่งเข้า overlay pipeline ได้โดยไม่ต้องแก้โค้ดส่วนอื่น
    """
    import mediapipe as mp
    from ultralytics import YOLO

    if config is None:
        config = PipelineConfig()

    meta: VideoMeta = read_video_meta(video_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    total_frames = meta.frame_count

    # ─────────────────────────────────────────────────────────
    # Pass 1: YOLO11n + BoT-SORT → เก็บ bbox ต่อ track_id ต่อเฟรม
    # ─────────────────────────────────────────────────────────
    model = YOLO("yolo11n.pt")

    # track_id -> {frame_idx: (x, y, w, h)}
    tracks_bboxes: dict[int, dict[int, tuple[int, int, int, int]]] = {}
    tracks_cx: dict[int, list[float]] = {}
    tracks_cy: dict[int, list[float]] = {}

    ball_bboxes_per_frame: dict[int, list[tuple[int, int, int, int]]] = {}
    racket_bboxes_per_frame: dict[int, list[tuple[int, int, int, int]]] = {}

    frame_idx = 0
    for r in model.track(
        source=video_path,
        tracker="botsort.yaml",
        classes=[0, 32, 38],      # person, sports ball, tennis racket
        stream=True,
        device=device,
        verbose=False,
    ):
        boxes = r.boxes
        ball_bboxes_per_frame[frame_idx] = []
        racket_bboxes_per_frame[frame_idx] = []
        
        if len(boxes) > 0:
            xyxy_arr = boxes.xyxy.cpu().numpy()
            cls_arr = boxes.cls.cpu().numpy()
            id_arr = boxes.id.cpu().numpy() if boxes.id is not None else [None] * len(boxes)
            
            for xyxy, cls_id, tid_t in zip(xyxy_arr, cls_arr, id_arr):
                cls_id = int(cls_id)
                x1, y1, x2, y2 = xyxy
                x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
                
                if cls_id == 0: # person
                    if tid_t is not None:
                        tid = int(tid_t)
                        if tid not in tracks_bboxes:
                            tracks_bboxes[tid] = {}
                            tracks_cx[tid] = []
                            tracks_cy[tid] = []
                        tracks_bboxes[tid][frame_idx] = (x, y, w, h)
                        tracks_cx[tid].append(x + w / 2.0)
                        tracks_cy[tid].append(y + h / 2.0)
                elif cls_id == 32: # sports ball
                    ball_bboxes_per_frame[frame_idx].append((x, y, w, h))
                elif cls_id == 38: # tennis racket
                    racket_bboxes_per_frame[frame_idx].append((x, y, w, h))

        if progress_callback:
            progress_callback(frame_idx // 2, total_frames)
        frame_idx += 1

    actual_frames = frame_idx

    # ─────────────────────────────────────────────────────────
    # Filter: ต้องขยับพอ (กันผู้ชม/เก็บบอลที่ยืนนิ่ง) + เลือกยาวสุด
    # ─────────────────────────────────────────────────────────
    MIN_MOVEMENT_STD_PX = 0.03 * meta.height  # ~3% ของความสูงเฟรม
    MIN_FRAMES = 10

    candidates = {
        tid: bboxes
        for tid, bboxes in tracks_bboxes.items()
        if len(bboxes) >= MIN_FRAMES
        and float(np.std(tracks_cy[tid])) >= MIN_MOVEMENT_STD_PX
    }

    # ถ้า strict filter ตัดคนออกหมด ให้ fall back เอาคนที่ยาวสุด
    if not candidates:
        candidates = tracks_bboxes

    # ─────────────────────────────────────────────────────────
    # Stitch Broken Tracks (Re-ID fallback)
    # ถ้า YOLO หลุดแล้วกลับมาเจอใหม่ มันจะได้ ID ใหม่ ทำให้โดนตัดทิ้ง
    # เราจะเย็บ track ที่ขาดเข้าด้วยกัน ถ้าตำแหน่ง/เวลา สอดคล้องกัน
    # ─────────────────────────────────────────────────────────
    MAX_GAP_FRAMES = int(meta.fps * 60)  # ยอมให้หลุดได้นานถึง 60 วินาที
    MAX_DIST_PX = meta.height * 0.7 # ยอมให้วาร์ปไปได้ 70% ของจอ (เผื่อวิ่งไกล)

    segments = []
    for tid, bboxes in candidates.items():
        frames = sorted(bboxes.keys())
        if not frames: continue
        segments.append({
            "tid": tid,
            "start_f": frames[0],
            "end_f": frames[-1],
            "bboxes": dict(bboxes)
        })
    
    while True:
        best_pair = None
        best_score = float('inf')
        
        for i in range(len(segments)):
            for j in range(len(segments)):
                if i == j: continue
                s1, s2 = segments[i], segments[j]
                
                # s2 must start after s1 ends
                delta_t = s2["start_f"] - s1["end_f"]
                if 0 < delta_t < MAX_GAP_FRAMES:
                    # เช็คความคล้ายของขนาดตัว (กันเอาผู้เล่นไปต่อกับเด็กเก็บบอล)
                    s1_h = np.median([b[3] for b in s1["bboxes"].values()])
                    s2_h = np.median([b[3] for b in s2["bboxes"].values()])
                    if min(s1_h, s2_h) / max(s1_h, s2_h) < 0.5:
                        continue  # ขนาดตัวต่างกันเกิน 2 เท่า ข้ามเลย
                        
                    b1 = s1["bboxes"][s1["end_f"]]
                    b2 = s2["bboxes"][s2["start_f"]]
                    
                    c1x, c1y = b1[0] + b1[2]/2.0, b1[1] + b1[3]/2.0
                    c2x, c2y = b2[0] + b2[2]/2.0, b2[1] + b2[3]/2.0
                    dist = ((c1x - c2x)**2 + (c1y - c2y)**2)**0.5
                    
                    if dist < MAX_DIST_PX:
                        score = dist + (delta_t * (MAX_DIST_PX / MAX_GAP_FRAMES))
                        if score < best_score:
                            best_score = score
                            best_pair = (i, j)
                            
        if best_pair is None:
            break
            
        i, j = best_pair
        segments[i]["bboxes"].update(segments[j]["bboxes"])
        segments[i]["end_f"] = segments[j]["end_f"]
        segments.pop(j)

    segments.sort(key=lambda s: len(s["bboxes"]), reverse=True)
    kept_segments = segments[:max_players]
    
    kept_tids = []
    for s in kept_segments:
        tid = s["tid"]
        kept_tids.append(tid)
        tracks_bboxes[tid] = s["bboxes"]

    if not kept_tids:
        return [], {}, {}

    # ─────────────────────────────────────────────────────────
    # Pass 1.5: Interpolate & Smooth Bounding Boxes
    # YOLO boxes snap tightly and jitter, causing MediaPipe crop
    # to resize randomly (which creates noisy pose outputs).
    # ─────────────────────────────────────────────────────────
    from scipy.signal import savgol_filter

    for tid in kept_tids:
        frames = sorted(tracks_bboxes[tid].keys())
        if len(frames) < 5:
            continue

        start_f, end_f = frames[0], frames[-1]
        all_frames = list(range(start_f, end_f + 1))
        
        arr = np.zeros((len(all_frames), 4))
        valid_mask = np.zeros(len(all_frames), dtype=bool)
        
        for i, f in enumerate(all_frames):
            if f in tracks_bboxes[tid]:
                arr[i] = tracks_bboxes[tid][f]
                valid_mask[i] = True
                
        for col in range(4):
            arr[:, col] = np.interp(np.arange(len(all_frames)), 
                                    np.arange(len(all_frames))[valid_mask], 
                                    arr[valid_mask, col])
                                  
        window_length = min(15, len(all_frames))
        if window_length % 2 == 0:
            window_length -= 1
            
        if window_length >= 5:
            for col in range(4):
                arr[:, col] = savgol_filter(arr[:, col], window_length, 2)
                
        for i, f in enumerate(all_frames):
            tracks_bboxes[tid][f] = tuple(map(int, arr[i]))

    # ─────────────────────────────────────────────────────────
    # Pass 2: MediaPipe Pose บน crop ที่ได้จาก YOLO bbox
    # ─────────────────────────────────────────────────────────
    pose_model = mp.solutions.pose.Pose(
        static_image_mode=True,
        model_complexity=config.model_complexity,
        min_detection_confidence=config.min_detection_confidence,
    )

    # โครงสร้างเก็บผล
    track_results: dict[int, dict] = {
        tid: {
            "landmarks": np.full((actual_frames, 33, 3), np.nan),
            "world_landmarks": np.full((actual_frames, 33, 3), np.nan),
            "visibility": np.zeros((actual_frames, 33)),
            "height_frac": np.zeros(actual_frames),
            "n_frames_total": 0,
            "all_heights": [],
        }
        for tid in kept_tids
    }

    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        for tid in kept_tids:
            if frame_idx not in tracks_bboxes[tid]:
                continue

            box = tracks_bboxes[tid][frame_idx]
            x, y, w, h = box

            result, crop_box = _crop_and_run_pose(pose_model, frame, box)
            if result is not None and result.pose_landmarks:
                lm_crop = np.array([[p.x, p.y, p.z] for p in result.pose_landmarks.landmark])
                wlm = np.array([[p.x, p.y, p.z] for p in result.pose_world_landmarks.landmark])
                vis = np.array([p.visibility for p in result.pose_landmarks.landmark])

                lm = crop_landmarks_to_frame(lm_crop, crop_box, meta.width, meta.height)

                tr = track_results[tid]
                tr["landmarks"][frame_idx] = lm
                tr["world_landmarks"][frame_idx] = wlm
                tr["visibility"][frame_idx] = vis
                tr["height_frac"][frame_idx] = h / meta.height
                tr["n_frames_total"] += 1
                tr["all_heights"].append(h)

        if progress_callback:
            progress_callback(total_frames // 2 + frame_idx // 2, total_frames)
        frame_idx += 1

    cap.release()
    pose_model.close()

    # ─────────────────────────────────────────────────────────
    # Role assignment: near (ตัวใหญ่) vs far (ตัวเล็ก)
    # ─────────────────────────────────────────────────────────
    height_medians = []
    for tid in kept_tids:
        all_h = track_results[tid]["all_heights"]
        height_medians.append((tid, float(np.median(all_h)) if all_h else 0.0))
    height_medians.sort(key=lambda kv: -kv[1])

    role_by_id = {
        tid: ("near" if rank == 0 else "far" if rank == 1 else "other")
        for rank, (tid, _) in enumerate(height_medians)
    }

    result_tracks: list[PlayerTrack] = []
    for tid in kept_tids:
        tr = track_results[tid]
        pose_ts = PoseTimeseries(
            landmarks=tr["landmarks"],
            world_landmarks=tr["world_landmarks"],
            visibility=tr["visibility"],
            timestamps_ms=np.arange(actual_frames) / meta.fps * 1000.0,
            meta=meta,
            frame_stats=FrameStats(person_height_frac=tr["height_frac"]),
        )
        result_tracks.append(
            PlayerTrack(
                track_id=tid,
                role=role_by_id[tid],
                pose=pose_ts,
                n_frames_tracked=tr["n_frames_total"],
            )
        )

    order = {"near": 0, "far": 1, "other": 2}
    result_tracks.sort(key=lambda pt: order.get(pt.role, 9))
    # ─────────────────────────────────────────────────────────
    # Cleanup & Clear Resources (RAM & VRAM)
    # ─────────────────────────────────────────────────────────
    import gc
    del pose_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result_tracks, ball_bboxes_per_frame, racket_bboxes_per_frame
