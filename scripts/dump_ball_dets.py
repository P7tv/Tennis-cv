"""dump raw ball detections (x,y,w,h,conf) ต่อเฟรม → npz
เพื่อวิเคราะห์ต่อว่าลูกที่เจอ 'เคลื่อนที่' หรือ 'นอนนิ่ง'
"""
import sys, pickle
from pathlib import Path
ROOT = Path("D:/Work/Tennis-cv")
sys.path.insert(0, str(ROOT))
from ultralytics import YOLO
import torch

MODEL = ROOT / "runs/pose/tennis_ball_racket_pose_finetune_best.pt"
CLIPS = [
    "dataset/sessions/set3/IMG_0281(BH).MOV",
    "dataset/sessions/set4/IMG_0300B7-SV1.mp4",
    "dataset/sessions/set2/IMG_0283A-VLfh.mov",
    "dataset/sessions/set4/IMG_0300A4(VL).mov",
]
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("scratch_ball_dets.pkl")

m = YOLO(str(MODEL))
ball_id = next(k for k, v in m.names.items() if "ball" in v.lower())
device = "cuda" if torch.cuda.is_available() else "cpu"

res = {}
for clip in CLIPS:
    p = ROOT / clip
    if not p.exists():
        print("SKIP", clip); continue
    per_frame = []
    for r in m.predict(source=str(p), stream=True, device=device, verbose=False,
                       conf=0.05, imgsz=1024, classes=[ball_id]):
        b = r.boxes
        if b is None or len(b) == 0:
            per_frame.append([])
        else:
            xy = b.xyxy.cpu().numpy(); cf = b.conf.cpu().numpy()
            per_frame.append([
                (float((x1+x2)/2), float((y1+y2)/2), float(x2-x1), float(y2-y1), float(c))
                for (x1, y1, x2, y2), c in zip(xy, cf)])
    res[p.name] = per_frame
    print(f"{p.name}: {len(per_frame)} เฟรม")

with open(ROOT / OUT, "wb") as f:
    pickle.dump(res, f)
print("wrote", OUT)
