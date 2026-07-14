from ultralytics import YOLO
import sys
import os

def main():
    print("="*50)
    print("🚀 YOLO11 Tennis Ball Custom Training")
    print("="*50)
    
    # 1. Ask for dataset path
    data_yaml = input("Enter the path to your dataset's data.yaml file: ").strip()
    if not data_yaml or not os.path.exists(data_yaml):
        print("❌ Invalid path to data.yaml!")
        sys.exit(1)
        
    print(f"\nUsing dataset config: {data_yaml}")
    print("Loading base model (yolo26s.pt)...")
    
    # 2. Load model
    try:
        # อัปเกรดมาใช้ YOLO11m (Medium) ตามที่คุณปันแนะนำ เพื่อความแม่นยำขั้นสุดในการหาลูกเทนนิส!
        # (โมเดลใหญ่ขึ้น จะกิน VRAM มากขึ้น เลยต้องลด batch size ลง)
        model = YOLO("yolo11m.pt")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)
        
    # 3. Train
    print("\n⏳ Starting training...")
    print("Parameters: imgsz=640, epochs=50, batch=8 (Adjusted for 6GB VRAM)")
    
    try:
        results = model.train(
            data=data_yaml,
            epochs=50,
            imgsz=640,
            batch=4,           # ลด Batch ลงเหลือ 4 เพื่อให้ 6GB VRAM รันรุ่น M ไหว
            workers=0,         # Fix Windows PyTorch multiprocessing DLL error (WinError 1114)
            device=0,          # GPU 0
            project="runs/detect",
            name="tennis_ball_custom",
            exist_ok=True,
            patience=10,       # Early stopping
            save=True,
            # --- 🎾 Aggressive Augmentations สำหรับลูกเทนนิส (วัตถุขนาดเล็ก) ---
            mosaic=1.0,        # เอา 4 รูปมาต่อกัน (ช่วยให้เห็นบอลในหลายมุมมอง)
            mixup=0.1,         # ซ้อนภาพบางส่วน (ช่วยลด Overfit)
            degrees=10.0,      # หมุนภาพนิดหน่อยเผื่อมุมกล้องเอียง
            hsv_s=0.5,         # ปรับความสดของสี (เผื่อถ่ายในที่ร่ม/แดดจัด)
            hsv_v=0.4,         # ปรับความสว่าง
            # --- ⚠️ Hardware Fix สำหรับ GTX 1660 ---
            amp=False          # ปิด Mixed Precision เพราะ GTX 1660 มักจะบั๊ก loss=NaN หรือ mAP=0
        )
        
        print("\n✅ Training Complete!")
        print("Your custom model is saved at:")
        print(os.path.abspath("runs/detect/tennis_ball_custom/weights/best.pt"))
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")

if __name__ == "__main__":
    main()
