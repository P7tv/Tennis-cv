import argparse
from pathlib import Path
import sys

def main():
    print("🎾 AI Tennis Vision Pro - TensorRT Exporter")
    print("=" * 50)
    
    try:
        from ultralytics import YOLO
    except ImportError:
        print("❌ Error: 'ultralytics' library not found. Please install it first.")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Export YOLO .pt model to TensorRT (.engine) for 2-5x faster inference.")
    parser.add_argument("model", type=str, help="Path to the .pt model (e.g. yolo11m.pt or runs/detect/.../best.pt)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for inference (default: 640)")
    parser.add_argument("--no-half", action="store_true", help="Disable FP16 (Half Precision). By default, FP16 is ENABLED for speed.")
    parser.add_argument("--workspace", type=float, default=4, help="Workspace size in GB for TensorRT (default: 4)")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ Error: Model file '{args.model}' not found!")
        sys.exit(1)

    if not model_path.suffix == '.pt':
        print(f"⚠️ Warning: Model file '{args.model}' does not have a .pt extension.")

    use_half = not args.no_half

    print(f"🚀 Loading model: {args.model}")
    model = YOLO(args.model)
    
    print(f"\n⚙️ Exporting to TensorRT...")
    print(f"   - Image Size: {args.imgsz}")
    print(f"   - FP16 (Half Precision): {'Yes' if use_half else 'No'}")
    print(f"   - Workspace: {args.workspace} GB")
    print(f"   - Note: This process may take 5-15 minutes depending on your GPU.\n")
    
    try:
        # Export the model
        export_path = model.export(
            format="engine",
            imgsz=args.imgsz,
            half=use_half,
            workspace=args.workspace,
            device=0 # Requires GPU
        )
        
        print(f"\n✅ Success! TensorRT engine successfully saved to: {export_path}")
        print("💡 You can now select this .engine file in the Streamlit UI!")
        
    except Exception as e:
        print(f"\n❌ Export Failed: {e}")
        print("Ensure you have a compatible NVIDIA GPU and TensorRT/CUDA installed.")

if __name__ == "__main__":
    main()
