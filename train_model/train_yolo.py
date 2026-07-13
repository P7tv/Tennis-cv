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
        # ใช้ YOLO26s (Small) ตัวใหม่ล่าสุด! NMS-Free, เร็วและแม่นยำกว่า YOLO11
        # (กิน VRAM เพิ่มขึ้นนิดหน่อย แต่ 1660 6GB รับไหวสบายๆ ครับ)
        model = YOLO("yolo26s.pt")
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
            batch=8,           # Fit in 6GB VRAM
            workers=0,         # Fix Windows PyTorch multiprocessing DLL error (WinError 1114)
            device=0,          # GPU 0
            project="runs/detect",
            name="tennis_ball_custom",
            exist_ok=True,
            patience=10,       # Early stopping
            save=True
        )
        
        print("\n✅ Training Complete!")
        print("Your custom model is saved at:")
        print(os.path.abspath("runs/detect/tennis_ball_custom/weights/best.pt"))
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")

if __name__ == "__main__":
    main()
