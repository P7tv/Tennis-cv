import os
import sys
import glob
import argparse
from pathlib import Path

def check_dependencies():
    """ตรวจสอบว่าได้ติดตั้งโมดูลที่จำเป็นสำหรับ TensorRT หรือไม่"""
    try:
        import tensorrt
        print(f"✅ Found TensorRT version: {tensorrt.__version__}")
    except ImportError:
        print("❌ Error: 'tensorrt' module is not installed.")
        print("\nTo export YOLO to TensorRT, you need to install the following:")
        print("    pip install ultralytics[export]")
        print("    pip install tensorrt")
        print("\nNote: You also need NVIDIA CUDA Toolkit and cuDNN installed on your system.")
        print("This script must be run on the machine with the target GPU where the model will be deployed.")
        sys.exit(1)

def find_best_models():
    """ค้นหาไฟล์โมเดล .pt ในโปรเจกต์ (เน้นโฟลเดอร์ runs ก่อน)"""
    models = glob.glob("runs/**/*.pt", recursive=True)
    if not models:
        models = glob.glob("*.pt")
    
    # กรองเอาเฉพาะไฟล์ที่มีคำว่า best หรือ custom โมเดล
    custom_models = [m for m in models if "best.pt" in m or "custom" in m]
    if custom_models:
        return custom_models
    return models

def main():
    parser = argparse.ArgumentParser(description="Export YOLOv8/11 models to TensorRT (.engine) for faster inference.")
    parser.add_argument("--model", type=str, default=None, 
                        help="Path to the YOLO .pt model. If not provided, the script will auto-detect recent models.")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size (default: 640)")
    parser.add_argument("--device", type=int, default=0, help="GPU device ID to use for export (default: 0)")
    parser.add_argument("--fp16", action="store_true", default=True, help="Export with FP16 precision for speed (default: True)")
    
    args = parser.parse_args()
    
    # 1. เช็ค Dependency
    print("Checking dependencies...")
    check_dependencies()
    
    from ultralytics import YOLO
    
    # 2. หาโมเดล
    model_path = args.model
    if not model_path:
        print("\n🔍 No model path provided. Searching for custom YOLO models...")
        candidate_models = find_best_models()
        
        if not candidate_models:
            print("❌ No .pt models found in the current directory or 'runs/' folder.")
            sys.exit(1)
            
        print("\nFound the following models:")
        for i, m in enumerate(candidate_models):
            print(f"  [{i}] {m}")
            
        if len(candidate_models) == 1:
            model_path = candidate_models[0]
            print(f"\nAuto-selecting the only model: {model_path}")
        else:
            try:
                choice = input(f"\nEnter the number [0-{len(candidate_models)-1}] to select a model (or press Enter for [0]): ")
                choice_idx = int(choice) if choice.strip() else 0
                model_path = candidate_models[choice_idx]
            except (ValueError, IndexError):
                print("Invalid selection. Exiting.")
                sys.exit(1)
                
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file '{model_path}' not found.")
        sys.exit(1)
        
    print(f"\n🚀 Loading model: {model_path}")
    model = YOLO(model_path)
    
    print(f"\n⚙️  Exporting to TensorRT...")
    print(f"   - Target format: engine")
    print(f"   - Precision: {'FP16' if args.fp16 else 'FP32'}")
    print(f"   - Image size: {args.imgsz}")
    print(f"   - Device: {args.device}")
    print("\n⏳ This process may take a while (5-15 minutes depending on GPU)...")
    
    try:
        # 3. Export
        exported_path = model.export(
            format="engine",
            imgsz=args.imgsz,
            device=args.device,
            half=args.fp16,
            dynamic=False, # Static batch size is usually faster
            simplify=True
        )
        print(f"\n✅ Export successful!")
        print(f"📦 TensorRT engine saved to: {exported_path}")
        print("\n💡 You can now run the webui and the .engine model will be automatically available in the model selector.")
    except Exception as e:
        print(f"\n❌ Export failed with error:")
        print(e)
        print("\nTroubleshooting tips:")
        print("1. Ensure your NVIDIA drivers are up to date.")
        print("2. Make sure CUDA Toolkit and cuDNN are correctly installed and match the PyTorch version.")
        print("3. Try running with admin/sudo privileges if permission is denied.")

if __name__ == "__main__":
    main()
