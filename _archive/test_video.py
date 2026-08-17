from webui.yolo_track import track_players_with_yolo
from webui.overlay import render_overlay_video
import os

video_path = r'D:\Work\Tennis-cv\dataset\sessions\set2\IMG_0284.MOV'
model_path = r'D:\Work\Tennis-cv\runs\detect\runs\detect\tennis_ball_custom\weights\best.pt'
out_path = r'D:\Work\Tennis-cv\overlay_IMG_0284.mp4'

os.makedirs(os.path.dirname(out_path), exist_ok=True)
tracks, ball, racket, _ = track_players_with_yolo(video_path, model_path=model_path)
for t in tracks:
    print(f'Track {t.track_id}: {t.n_frames_tracked} frames')
render_overlay_video(video_path, tracks, out_path=out_path, ball_bboxes=ball, racket_bboxes=racket)
print('Done!')

