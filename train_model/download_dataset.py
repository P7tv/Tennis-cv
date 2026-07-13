import os
import sys

try:
    from roboflow import Roboflow
except ImportError:
    print("Please install roboflow: pip install roboflow")
    sys.exit(1)

def main():
    print("="*50)
    print("🎾 Tennis Ball Dataset Downloader (Roboflow)")
    print("="*50)
    
    api_key = input("Enter your Roboflow API Key (get one free at roboflow.com): ").strip()
    if not api_key:
        print("API Key is required!")
        sys.exit(1)
        
    try:
        rf = Roboflow(api_key=api_key)
        
        # We will use a highly rated public dataset for tennis balls
        print("\nDownloading dataset... This might take a while depending on your internet speed.")
        # Replace with a known good dataset, e.g., tennis-ball-v2
        project = rf.workspace("viren-dhanwani").project("tennis-ball-detection")
        version = project.version(6)
        
        # Download YOLOv8 format which is fully compatible with YOLO11
        dataset = version.download("yolov8")
        
        print("\n✅ Dataset downloaded successfully!")
        print(f"Location: {dataset.location}")
        
    except Exception as e:
        print(f"\n❌ Error downloading dataset: {e}")

if __name__ == "__main__":
    main()
