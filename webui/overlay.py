"""Overlay rendering — วาด skeleton บนวิดีโอต้นฉบับสำหรับตรวจสอบด้วยตา

Reuse จาก diagnostic script ที่ใช้ debug track-selection/stitching bugs
ทั้ง session (2026-07-10) — validated ว่าช่วยจับบั๊กจริงได้ (wrong-identity,
track ขาดเป็นท่อน) ที่ตัวเลขสถิติอย่างเดียวมองไม่เห็น (เช่น "far" เกาะ
คนยืนนิ่งข้างสนามแทนคู่แข่งตัวจริง — เห็นชัดจากภาพ แต่ตัวเลข coverage
เฉย ๆ ไม่บอกอะไรผิดเลย)
"""

import subprocess
import tempfile

import cv2
import mediapipe as mp
import numpy as np

CONNECTIONS = mp.solutions.pose.POSE_CONNECTIONS
ROLE_COLORS = {"near": (0, 80, 255), "far": (255, 160, 0), "other": (0, 200, 0)}
# สีไล่ตามลำดับ label (Player 1/2/3/...) ต่างจาก ROLE_COLORS (near/far)
LABEL_COLORS = [(0, 80, 255), (255, 160, 0), (0, 200, 0), (200, 0, 200), (0, 200, 200), (128, 128, 0)]
EXCLUDED_COLOR = (140, 140, 140)
NOT_A_PLAYER_PREFIX = "ไม่ใช่ผู้เล่น"


def _track_display(tracks: list, player_labels: dict | None) -> tuple[dict, dict]:
    """คืน (track_id -> BGR color, track_id -> ข้อความที่แสดง) — ถ้ามี
    player_labels (จากหน้า Track Review) ใช้สีตาม label แทน role"""
    colors, names = {}, {}
    for i, t in enumerate(tracks):
        label = (player_labels or {}).get(t.track_id)
        if label and label.startswith(NOT_A_PLAYER_PREFIX):
            colors[t.track_id] = EXCLUDED_COLOR
            names[t.track_id] = f"excluded id={t.track_id}"
        elif label:
            colors[t.track_id] = LABEL_COLORS[i % len(LABEL_COLORS)]
            names[t.track_id] = f"{label} id={t.track_id}"
        else:
            colors[t.track_id] = ROLE_COLORS.get(t.role, (200, 200, 200))
            names[t.track_id] = f"{t.role} id={t.track_id}"
    return colors, names


def render_overlay_video(video_path: str, tracks: list,
                         player_labels: dict | None = None,
                         out_path: str | None = None,
                         ball_bboxes: dict | None = None,
                         racket_bboxes: dict | None = None,
                         court_homography: np.ndarray | None = None,
                         hit_events: list[dict] | None = None) -> str:
    """วาด skeleton ของทุก track ลงบนวิดีโอต้นฉบับ คืน path ไฟล์ output
    (H.264 mp4 — เล่นได้ใน browser ทุกตัว ต่างจาก mp4v ดิบของ OpenCV)

    จุดสี: confidence >= 0.5 = สีของ track นั้น, ต่ำกว่า = จุดเทา (บอกว่า
    เป็นค่าที่ไม่น่าเชื่อถือ เหมือน low_confidence flag ใน schema)
    """
    cap = cv2.VideoCapture(video_path)
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[Overlay Debug] Starting render. Video: {n_frames} frames.")
    print(f"[Overlay Debug] Received {len(tracks)} player tracks.")
    for pt in tracks:
        print(f"[Overlay Debug] Track {pt.track_id} ({pt.role}) - Tracked for {pt.n_frames_tracked} frames.")

    import os
    from pathlib import Path
    
    _TMP_DIR = Path(__file__).parent / ".tmp"
    _TMP_DIR.mkdir(exist_ok=True)

    if out_path is None:
        with tempfile.NamedTemporaryFile(suffix="_raw.mp4", delete=False, dir=str(_TMP_DIR)) as f:
            out_path = f.name
            
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (fw, fh))

    colors, names = _track_display(tracks, player_labels)
    
    wrist_history: dict[int, dict[str, list[tuple[int, int]]]] = {}

    for frame_idx in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        y_offset = 25
        
        # 1. Draw Ball Tracking
        if ball_bboxes and frame_idx in ball_bboxes:
            for bx, by, bw, bh in ball_bboxes[frame_idx]:
                cv2.circle(frame, (bx + bw//2, by + bh//2), max(bw, bh)//2, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, "Ball", (bx, by - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

        # 1.5 Draw Court Grid
        if court_homography is not None:
            from loeuf_cv.court_calibration import image_point_from_world, WORLD_POINTS
            try:
                # Get image coordinates for the 4 corners
                nbl = image_point_from_world(court_homography, WORLD_POINTS["near_baseline_left"])
                nbr = image_point_from_world(court_homography, WORLD_POINTS["near_baseline_right"])
                sll = image_point_from_world(court_homography, WORLD_POINTS["service_line_left"])
                slr = image_point_from_world(court_homography, WORLD_POINTS["service_line_right"])
                
                # Convert to integer pixel coordinates
                pts_bl = np.array([nbl, nbr], np.int32)
                pts_sl = np.array([sll, slr], np.int32)
                pts_left = np.array([nbl, sll], np.int32)
                pts_right = np.array([nbr, slr], np.int32)
                
                court_color = (0, 255, 0) # Green for court grid
                cv2.polylines(frame, [pts_bl], False, court_color, 2, cv2.LINE_AA)
                cv2.polylines(frame, [pts_sl], False, court_color, 2, cv2.LINE_AA)
                cv2.polylines(frame, [pts_left], False, court_color, 2, cv2.LINE_AA)
                cv2.polylines(frame, [pts_right], False, court_color, 2, cv2.LINE_AA)
            except Exception:
                pass # Fail silently if homography mapping errors out

        # 1.6 Draw Hit Events
        if hit_events:
            for hit in hit_events:
                hf = hit["frame"]
                if hf <= frame_idx <= hf + 10:
                    # Draw HIT effect
                    # Find ball position in this frame or use hit frame's ball position (approximate)
                    bx, by = int(fw/2), int(fh/2)
                    if ball_bboxes and frame_idx in ball_bboxes and ball_bboxes[frame_idx]:
                        b = ball_bboxes[frame_idx][0]
                        bx, by = b[0] + b[2]//2, b[1] + b[3]//2
                    
                    # Create ripple effect
                    radius = (frame_idx - hf + 1) * 10
                    cv2.circle(frame, (bx, by), radius, (0, 0, 255), 3, cv2.LINE_AA)
                    cv2.putText(frame, "HIT!", (bx + 20, by - 20), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)

        wrists_this_frame = []
        
        for t in tracks:
            if t.track_id not in wrist_history:
                wrist_history[t.track_id] = {"left": [], "right": []}
                
            vis = t.pose.visibility
            lm = t.pose.landmarks
            color = colors[t.track_id]
            if frame_idx >= len(vis) or vis[frame_idx].mean() <= 0:
                if frame_idx == 0: print(f"[Overlay Debug] Track {t.track_id} skipped frame 0 (no vis)")
                continue
            color = colors[t.track_id]
            pts = lm[frame_idx, :, :2]
            v = vis[frame_idx]
            
            if frame_idx == 0:
                print(f"[Overlay Debug] Track {t.track_id} DRAWING frame 0! Vis mean: {v.mean():.3f}")
            
            # 2. Extract and Draw Wrist Path
            lw, rw = None, None
            if v[15] >= 0.3 and not np.isnan(pts[15]).any():
                lw = (int(pts[15, 0] * fw), int(pts[15, 1] * fh))
                wrist_history[t.track_id]["left"].append(lw)
                wrists_this_frame.append(("left", t.track_id, lw, color))
            if v[16] >= 0.3 and not np.isnan(pts[16]).any():
                rw = (int(pts[16, 0] * fw), int(pts[16, 1] * fh))
                wrist_history[t.track_id]["right"].append(rw)
                wrists_this_frame.append(("right", t.track_id, rw, color))
                
            # Limit history to 15 frames
            wrist_history[t.track_id]["left"] = wrist_history[t.track_id]["left"][-15:]
            wrist_history[t.track_id]["right"] = wrist_history[t.track_id]["right"][-15:]
            
            # Draw trails
            if len(wrist_history[t.track_id]["left"]) >= 2:
                cv2.polylines(frame, [np.array(wrist_history[t.track_id]["left"])], False, color, 3, cv2.LINE_AA)
            if len(wrist_history[t.track_id]["right"]) >= 2:
                cv2.polylines(frame, [np.array(wrist_history[t.track_id]["right"])], False, color, 3, cv2.LINE_AA)
                
            for a, b in CONNECTIONS:
                if np.isnan(pts[a]).any() or np.isnan(pts[b]).any():
                    continue
                pa = (int(pts[a, 0] * fw), int(pts[a, 1] * fh))
                pb = (int(pts[b, 0] * fw), int(pts[b, 1] * fh))
                cv2.line(frame, pa, pb, color, 2, cv2.LINE_AA)
            for i in range(33):
                if np.isnan(pts[i]).any():
                    continue
                p = (int(pts[i, 0] * fw), int(pts[i, 1] * fh))
                conf = float(v[i])
                dot_color = color if conf >= 0.5 else (128, 128, 128)
                cv2.circle(frame, p, 4 if conf >= 0.5 else 3, dot_color, -1, cv2.LINE_AA)
            cv2.putText(frame, names[t.track_id], (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
            y_offset += 25
            
        # 3. Draw Racket Center & Tip
        if racket_bboxes and frame_idx in racket_bboxes:
            for rx, ry, rw_w, rh in racket_bboxes[frame_idx]:
                rcx, rcy = rx + rw_w//2, ry + rh//2
                best_w = None
                best_dist = float('inf')
                for w_side, w_tid, w_pt, w_color in wrists_this_frame:
                    d = ((rcx - w_pt[0])**2 + (rcy - w_pt[1])**2)**0.5
                    if d < best_dist:
                        best_dist = d
                        best_w = (w_pt, w_color)
                        
                cv2.circle(frame, (rcx, rcy), 4, (255, 0, 255), -1, cv2.LINE_AA)
                
                # Match wrist if within reasonable distance (e.g., 3x racket size)
                if best_w and best_dist < max(rw_w, rh) * 3:
                    w_pt, w_color = best_w
                    vx, vy = rcx - w_pt[0], rcy - w_pt[1]
                    v_len = (vx**2 + vy**2)**0.5
                    if v_len > 0:
                        r_len = (rw_w**2 + rh**2)**0.5
                        tip_x = int(rcx + (vx/v_len) * r_len * 0.4)
                        tip_y = int(rcy + (vy/v_len) * r_len * 0.4)
                        
                        cv2.line(frame, w_pt, (rcx, rcy), w_color, 2, cv2.LINE_AA)
                        cv2.line(frame, (rcx, rcy), (tip_x, tip_y), (255, 0, 255), 3, cv2.LINE_AA)
                        cv2.circle(frame, (tip_x, tip_y), 5, (0, 0, 255), -1, cv2.LINE_AA)
                        cv2.putText(frame, "Tip", (tip_x + 5, tip_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

        cv2.putText(frame, f"frame {frame_idx}", (fw - 160, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        writer.write(frame)
    writer.release()
    cap.release()

    h264_path = out_path.replace("_raw.mp4", ".mp4")
    
    def _get_ffmpeg_path() -> str:
        import shutil
        import glob
        path = shutil.which("ffmpeg")
        if path: return path
        localappdata = os.environ.get("LOCALAPPDATA", "")
        matches = glob.glob(os.path.join(localappdata, "Microsoft", "WinGet", "Packages", "Gyan.FFmpeg*", "*", "bin", "ffmpeg.exe"))
        if matches: return matches[0]
        return "ffmpeg"
        
    import gc
    gc.collect()
    
    try:
        result = subprocess.run(
            [_get_ffmpeg_path(), "-y", "-i", out_path, "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-crf", "23", "-preset", "ultrafast", "-threads", "2", h264_path],
            capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFMPEG failed (exit {result.returncode}):\n{result.stderr or result.stdout}")
    finally:
        if os.path.exists(out_path):
            try:
                os.unlink(out_path)
            except Exception:
                pass
                
    return h264_path
