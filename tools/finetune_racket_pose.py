"""Fine-tune โมเดล ball+racket pose ที่มีอยู่แล้ว (runs/pose/tennis_ball_racket_pose/weights/best.pt)
ด้วยภาพจากคลิปลูกค้าจริงที่เพิ่ง label bbox เพิ่ม (train_model/../tennis-ball-racket-pose/train/images/pilot_*.jpg)
— racket detector เดิม (เทรนจาก Roboflow racket-4 เท่านั้น) แทบไม่ detect อะไรเลยบนคลิปจริง
(0.7% ของ hit candidates) เพราะมุมกล้อง/ระยะห่างต่างจากข้อมูลเทรนเดิมมาก

รัน: python train_model/finetune_racket_pose.py
"""

import sys

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

from ultralytics import YOLO

if __name__ == "__main__":
    # หมายเหตุ: รันสำเร็จบน Lightning.ai L4 (23GB VRAM) ด้วย imgsz/batch เท่าตอนเทรนเต็มรอบแรก
    # เครื่อง local (GTX 1660, 6GB) OOM ตั้งแต่ batch=4 แล้ว ต้องใช้ VM
    # ผล: racket detection rate บนเฟรมจริงลูกค้า (conf>=0.15, นอกชุด 12 ภาพที่ label เอง)
    # จาก 0/92 (0%) -> 42/92 (45.7%)
    model = YOLO("runs/pose/tennis_ball_racket_pose/weights/best.pt")
    model.train(
        data="tennis-ball-racket-pose/data.yaml",
        epochs=15,
        imgsz=1024,
        batch=32,
        lr0=0.0005,   # หมายเหตุ: optimizer='auto' (default) เมิน lr0 ที่ตั้งเองแล้วเลือกเองอัตโนมัติ
        freeze=10,    # freeze backbone ส่วนใหญ่ กัน catastrophic forgetting บน dataset เดิม (ball, val mAP เสถียร)
        fliplr=0.0,   # เหตุผลเดียวกับตอนเทรนแรก — skeleton ไม่มี flip_idx ที่ชัดเจน
        project="runs/pose",
        name="tennis_ball_racket_pose_finetune",
        device=0,
        workers=8,
    )
