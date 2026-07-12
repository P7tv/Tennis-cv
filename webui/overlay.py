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
                         out_path: str | None = None) -> str:
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

    if out_path is None:
        out_path = tempfile.mktemp(suffix="_raw.mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (fw, fh))

    colors, names = _track_display(tracks, player_labels)

    for frame_idx in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        y_offset = 25
        for t in tracks:
            vis = t.pose.visibility
            lm = t.pose.landmarks
            if frame_idx >= len(vis) or vis[frame_idx].mean() <= 0:
                continue
            color = colors[t.track_id]
            pts = lm[frame_idx, :, :2]
            v = vis[frame_idx]
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
        cv2.putText(frame, f"frame {frame_idx}", (fw - 160, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        writer.write(frame)
    writer.release()
    cap.release()

    h264_path = out_path.replace("_raw.mp4", ".mp4")
    subprocess.run(
        ["ffmpeg", "-y", "-i", out_path, "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-crf", "23", h264_path],
        check=True, capture_output=True)
    return h264_path
