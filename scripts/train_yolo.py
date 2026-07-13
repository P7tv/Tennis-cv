import os
from ultralytics import YOLO

# ==============================================================================
# คำแนะนำก่อนการใช้งาน:
# 1. ติดตั้งไลบรารี roboflow เพิ่มเติม: pip install roboflow
# 2. ไปที่เว็บ https://universe.roboflow.com/ เพื่อหา Dataset เทนนิสที่คุณต้องการ
# 3. กดปุ่ม "Export Dataset" -> เลือก Format เป็น "YOLOv8"
# 4. เลือก "Show download code" แล้วก็อปปี้โค้ดมาใส่ทับในส่วนด้านล่างนี้
# ==============================================================================

from roboflow import Roboflow
rf = Roboflow(api_key="PUibTgdxgSHRcmyOZUEI")
project = rf.workspace("test-06r5e").project("tennis-racket-r6mgq")
version = project.version(4)
dataset = version.download("yolov8")
# ------------------------------------------

DATASET_YAML_PATH = os.path.join(dataset.location, "data.yaml")

def main():
    print("🚀 กำลังเริ่มต้น Train YOLO11...")
    
    # โหลดโมเดล YOLO11n (โมเดลตัวเล็กที่สุด เร็วที่สุด) มาเป็นตัวตั้งต้น
    # ถ้าอยากให้แม่นขึ้น สามารถเปลี่ยนเป็น "yolo11s.pt" หรือ "yolo11m.pt" ได้ครับ (กินสเปคเพิ่มขึ้น)
    model = YOLO("yolo11n.pt")
    
    # ถ้าโหลด Dataset ยังไม่สำเร็จ อย่าเพิ่งรันต่อ
    if not os.path.exists(DATASET_YAML_PATH):
        print(f"❌ ไม่พบไฟล์ {DATASET_YAML_PATH}")
        print("กรุณาใส่โค้ดดาวน์โหลดจาก Roboflow ด้านบน หรือตรวจสอบ Path ก่อนรันครับ!")
        return

    # เริ่มกระบวนการ Train
    # parameters ที่สำคัญ:
    # - epochs: จำนวนรอบที่ใช้สอน (ยิ่งเยอะยิ่งดี แต่นาน) แนะนำเริ่มที่ 50-100 รอบ
    # - imgsz: ขนาดภาพที่ใช้ Train (640 เป็นมาตรฐาน)
    # - device: เลือก 0 เพื่อใช้การ์ดจอ Nvidia (CUDA)
    results = model.train(
        data=DATASET_YAML_PATH,
        epochs=100,
        imgsz=640,
        device=0,  # ใช้ GPU 0
        batch=8,   # ลด batch เป็น 8 และ
        workers=0, # ปิด workers เพื่อป้องกัน Error RAM / Pagefile เต็ม
        name="tennis_custom_yolo" # ชื่อโฟลเดอร์เซฟผลลัพธ์
    )
    
    print("✅ Train เสร็จสิ้น!")
    print("ไฟล์โมเดลที่ดีที่สุดจะถูกเซฟอยู่ที่: runs/detect/tennis_custom_yolo/weights/best.pt")
    print("นำไฟล์ best.pt ไปเลือกใช้ในหน้าเว็บ Web UI ได้เลยครับ!")

if __name__ == "__main__":
    main()
