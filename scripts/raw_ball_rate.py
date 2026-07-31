"""วัด raw ball detection rate — YOLO ดิบ ไม่ผ่าน Kalman

ตอบคำถาม: detector หาลูกเจอจริงกี่ % ของเฟรม และเจอทีละกี่ลูก
(สมมติฐาน: ลูกที่กระจายในสนาม/รถเข็น ทำให้สับสน)
"""
import sys, json
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
ROOT = Path("D:/Work/Tennis-cv")
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
import torch

MODEL = ROOT / "runs/pose/tennis_ball_racket_pose_finetune_best.pt"
CLIPS = [
    "dataset/sessions/set4/IMG_0300B7-SV1.mp4",
    "dataset/sessions/set3/IMG_0281(BH).MOV",
    "dataset/sessions/set2/IMG_0283A-VLfh.mov",
]

m = YOLO(str(MODEL))
ball_id = next((k for k, v in m.names.items() if "ball" in v.lower()), None)
print(f"names={m.names}  ball_id={ball_id}")
device = "cuda" if torch.cuda.is_available() else "cpu"

CONFS = [0.05, 0.15, 0.25, 0.40]
out = {}

for clip in CLIPS:
    p = ROOT / clip
    if not p.exists():
        print(f"SKIP {clip}")
        continue
    print(f"\n=== {p.name} ===")
    # รันที่ conf ต่ำสุด แล้วกรองเองทีหลัง จะได้รอบเดียว
    per_frame = []   # list[list[conf]]
    for r in m.predict(source=str(p), stream=True, device=device, verbose=False,
                       conf=min(CONFS), imgsz=1024, classes=[ball_id]):
        b = r.boxes
        if b is None or len(b) == 0:
            per_frame.append([])
        else:
            per_frame.append([float(c) for c in b.conf.cpu().numpy()])

    n = len(per_frame)
    row = {"n_frames": n}
    for c in CONFS:
        hits = [len([x for x in f if x >= c]) for f in per_frame]
        det_frames = sum(1 for h in hits if h > 0)
        multi = sum(1 for h in hits if h > 1)
        row[f"conf{c}"] = {
            "detect_rate": round(det_frames / n, 3),
            "multi_ball_rate": round(multi / n, 3),
            "mean_per_detected_frame": round(
                float(np.mean([h for h in hits if h > 0])) if det_frames else 0.0, 2),
            "max_in_frame": max(hits) if hits else 0,
        }
    # ความยาว gap ที่ยาวที่สุด (ที่ conf 0.15 = ค่าที่ pipeline ใช้)
    hits15 = [len([x for x in f if x >= 0.15]) > 0 for f in per_frame]
    gaps, cur = [], 0
    for h in hits15:
        if h:
            if cur: gaps.append(cur)
            cur = 0
        else:
            cur += 1
    if cur: gaps.append(cur)
    row["gap_conf0.15"] = {
        "n_gaps": len(gaps),
        "max_gap": max(gaps) if gaps else 0,
        "median_gap": float(np.median(gaps)) if gaps else 0.0,
        "gaps_over_30": sum(1 for g in gaps if g > 30),
    }
    out[p.name] = row
    print(json.dumps(row, indent=2, ensure_ascii=False))

Path(ROOT / "scratch_raw_ball_rate.json").write_text(
    json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
print("\nwrote scratch_raw_ball_rate.json")
