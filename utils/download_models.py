import os
import urllib.request
import sys
from pathlib import Path

def download_file(url, destination):
    print(f"Downloading {os.path.basename(destination)}...")
    
    def progress_hook(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        sys.stdout.write(f"\r[{'=' * (percent // 2)}{' ' * (50 - percent // 2)}] {percent}%")
        sys.stdout.flush()
    
    urllib.request.urlretrieve(url, destination, progress_hook)
    print("\nDownload complete!")

def setup_yolo_model():
    print("\n=== Setting up YOLOv8n Person Detector ===")
    
    try:
        from ultralytics import YOLO
        
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        onnx_path = models_dir / "yolov8n.onnx"
        
        if onnx_path.exists():
            print(f"✓ YOLOv8n ONNX model already exists at {onnx_path}")
            return
        
        print("Downloading YOLOv8n PyTorch model...")
        model = YOLO('yolov8n.pt')
        
        print("Converting to ONNX format...")
        model.export(format='onnx', simplify=True, dynamic=False, imgsz=640)
        
        if Path('yolov8n.onnx').exists():
            Path('yolov8n.onnx').rename(onnx_path)
            print(f"✓ YOLOv8n ONNX model saved to {onnx_path}")
        
    except Exception as e:
        print(f"✗ Error setting up YOLOv8n: {e}")
        print("Please install ultralytics: pip install ultralytics")

def setup_face_models():
    print("\n=== Setting up Face Models ===")
    
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    face_det_url = "https://github.com/deepinsight/insightface/releases/download/v0.7/scrfd_10g_bnkps.onnx"
    face_det_path = models_dir / "face_detection.onnx"
    
    if not face_det_path.exists():
        try:
            download_file(face_det_url, face_det_path)
            print(f"✓ Face detection model saved to {face_det_path}")
        except Exception as e:
            print(f"✗ Error downloading face detector: {e}")
            print("Manual download: https://github.com/deepinsight/insightface/tree/master/model_zoo")
    else:
        print(f"✓ Face detection model already exists")
    
    face_rec_url = "https://github.com/deepinsight/insightface/releases/download/v0.7/w600k_r50.onnx"
    face_rec_path = models_dir / "face_recognition.onnx"
    
    if not face_rec_path.exists():
        try:
            download_file(face_rec_url, face_rec_path)
            print(f"✓ Face recognition model saved to {face_rec_path}")
        except Exception as e:
            print(f"✗ Error downloading face recognizer: {e}")
            print("Manual download: https://github.com/deepinsight/insightface/tree/master/model_zoo")
    else:
        print(f"✓ Face recognition model already exists")

def main():
    print("=" * 60)
    print("Strongroom Surveillance - Model Setup")
    print("=" * 60)
    
    setup_yolo_model()
    setup_face_models()
    
    print("\n" + "=" * 60)
    print("Model setup complete!")
    print("=" * 60)
    print("\nNote: If automatic download fails, manually download models:")
    print("- YOLOv8n: https://github.com/ultralytics/assets/releases")
    print("- Face models: https://github.com/deepinsight/insightface/tree/master/model_zoo")
    print("\nRecommended models:")
    print("- Face Detection: scrfd_10g_bnkps.onnx")
    print("- Face Recognition: w600k_r50.onnx")

if __name__ == "__main__":
    main()
