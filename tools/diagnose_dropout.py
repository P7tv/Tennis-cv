#!/usr/bin/env python3
"""Diagnostic: หาว่า visibility dropout สัมพันธ์กับความเร็วการเคลื่อนที่
(สงสัยว่าช่วงเหวี่ยงไม้/ตี) หรือไม่ — รัน extract_multi_person แล้ว dump
per-frame visibility + wrist visibility + crop-box height + centroid speed
ของ track ที่ยาวที่สุด ไปเป็น .npz เพื่อวิเคราะห์ต่อ ไม่ต้องรัน MediaPipe ซ้ำ

ใช้:
    python scripts/diagnose_dropout.py clip.mp4 out.npz
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from loeuf_cv import PipelineConfig
from loeuf_cv.config import CORE_LANDMARKS, L_WRIST, R_WRIST
from loeuf_cv.multi_person import extract_multi_person


def main():
    video, out = sys.argv[1], sys.argv[2]
    role = sys.argv[3] if len(sys.argv) > 3 else None
    config = PipelineConfig()

    def progress(i, total):
        if total and i % 300 == 0:
            print(f"\r  {i}/{total}", end="", flush=True)

    tracks = extract_multi_person(video, config, max_players=2, progress_callback=progress)
    print()
    if not tracks:
        print("no tracks")
        return

    if role:
        target = next((t for t in tracks if t.role == role), None)
        if target is None:
            print(f"no track with role={role}; available: {[t.role for t in tracks]}")
            return
    else:
        target = max(tracks, key=lambda t: t.n_frames_tracked)
    ts = target.pose
    vis = ts.visibility
    lm = ts.landmarks

    core_vis = vis[:, list(CORE_LANDMARKS)].mean(axis=1)
    wrist_vis = vis[:, [L_WRIST, R_WRIST]].max(axis=1)  # มือถนัดไม่รู้ล่วงหน้า เอาค่าสูงสุด
    height_frac = ts.frame_stats.person_height_frac

    # centroid = hip midpoint (index 23,24), speed เป็น px normalized/frame
    hip = (lm[:, 23, :2] + lm[:, 24, :2]) / 2.0
    valid = core_vis > 0.3
    speed = np.full(len(hip), np.nan)
    d = np.diff(hip, axis=0)
    seg_speed = np.linalg.norm(d, axis=1)
    speed[1:] = seg_speed
    speed[~valid] = np.nan
    speed[np.roll(~valid, 1)] = np.nan

    np.savez(out, core_vis=core_vis, wrist_vis=wrist_vis, height_frac=height_frac,
             speed=speed, role=target.role, track_id=target.track_id,
             n_tracked=target.n_frames_tracked, n_total=ts.n_frames,
             fps=ts.fps)
    print(f"track role={target.role} tracked {target.n_frames_tracked}/{ts.n_frames}"
          f" ({target.n_frames_tracked/ts.n_frames*100:.1f}%) mean_core_vis={core_vis[valid].mean():.3f}")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
