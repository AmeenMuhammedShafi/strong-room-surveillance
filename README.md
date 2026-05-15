# Strongroom Surveillance System

**Production-ready surveillance system for strongroom monitoring with person detection, face recognition, and real-time alerts.**

## Features

✅ **ONNX-Optimized Models** - Fast inference using ONNX Runtime  
✅ **Person Detection** - YOLOv8n for detecting people in real-time  
✅ **Face Recognition** - ArcFace-based authentication system  
✅ **Smart Alerts** - Configurable alerts with cooldown management  
✅ **Multi-threading** - Async frame processing for smooth performance  
✅ **Production Ready** - Logging, error handling, and configuration management  

## Business Logic

The system enforces the following security rules:

| People Count | Authenticated | Unknown | Status | Action |
|-------------|---------------|---------|--------|--------|
| 0 | - | - | ✅ **NORMAL** | No alert |
| 1 | 0 or 1 | 0 or 1 | ⚠️ **ALERT** | Single person alert |
| 2 | 2 | 0 | ✅ **AUTHORIZED** | Log access |
| 2 | <2 | >0 | 🚨 **ALERT** | Unauthorized access |
| >2 | Any | Any | 🚨 **ALERT** | Excess people alert |

## System Architecture

```
strongroom_surveillance/
├── config/
│   └── config.yaml              # System configuration
├── models/                      # ONNX models (auto-downloaded)
│   └── yolov8n.onnx            # Person detection
|
├── database/
│   └── face_encodings.pkl      # User face database
├── logs/
│   └── surveillance.log        # System logs
├── utils/
│   ├── person_detector.py      # Person detection module
│   ├── face_recognizer.py      # Face recognition module
│   ├── alert_system.py         # Alert and logging system
│   └── download_models.py      # Model downloader
├── main.py                      # Main surveillance application
├── enroll_user.py              # User enrollment tool
└── requirements.txt            # Python dependencies
```

## Installation

### 1. Prerequisites

- Python 3.8+
- Webcam or IP camera
- (Optional) CUDA-enabled GPU for faster inference

### 2. Clone/Download Project

```bash
cd strongroom_surveillance
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** If you have a GPU with CUDA:
```bash
pip uninstall onnxruntime
pip install onnxruntime-gpu
```

### 4. Download Models

```bash
python download_models.py
```

This will download:
- **YOLOv8n** - Lightweight person detector (~6MB)
- **SCRFD** - Face detector (~2.5MB)
- **ArcFace** - Face recognition model (~166MB)

Place downloaded models in the `models/` directory.

## Configuration

Edit `config/config.yaml` to customize:

### Camera Settings
```yaml
camera:
  source: 0  # 0 for webcam, or "rtsp://user:pass@ip:port/stream"
  resolution_width: 1280
  resolution_height: 720
  fps: 30
```

### Detection Settings
```yaml
detection:
  person_confidence_threshold: 0.5
  face_confidence_threshold: 0.6
  face_recognition_threshold: 0.6  # Lower = stricter matching
  processing_fps: 5  # Process every Nth frame
```

### Alert Settings
```yaml
alerts:
  enable_sound: true
  enable_email: false
  enable_webhook: false
  enable_logging: true
  cooldown_seconds: 10
```

For email alerts, configure SMTP:
```yaml
alerts:
  enable_email: true
  email:
    smtp_server: "smtp.gmail.com"
    smtp_port: 587
    sender_email: "your-email@gmail.com"
    sender_password: "your-app-password"
    recipient_emails:
      - "security@company.com"
```

For webhook alerts:
```yaml
alerts:
  enable_webhook: true
  webhook:
    url: "https://your-webhook-url.com/alert"
    headers:
      Authorization: "Bearer YOUR_TOKEN"
```

## Usage

### 1. Enroll Authorized Users

**Enroll from camera (recommended):**
```bash
python enroll_user.py --mode camera --user-id emp001 --name "John Doe" --samples 5
```

This will open your camera and capture 5 face samples. Press SPACE to capture each sample.

**Enroll from images:**
```bash
# Place user's face images in a folder (e.g., enrolled_faces/john_doe/)
python enroll_user.py --mode images --user-id emp001 --name "John Doe" --image-dir enrolled_faces/john_doe
```

**List enrolled users:**
```bash
python enroll_user.py --mode list
```

**Remove user:**
```bash
python enroll_user.py --mode remove --user-id emp001
```

### 2. Run Surveillance System

```bash
python main.py
```

**Keyboard controls:**
- `Q` - Quit application
- `S` - Save snapshot to logs/

### 3. Monitor Alerts

Alerts are logged to `logs/surveillance.log`:
```
2024-01-15 14:30:22 - WARNING - SECURITY ALERT: Single person detected in strongroom - John Doe
2024-01-15 14:32:15 - INFO - Authorized access: John Doe, Jane Smith
2024-01-15 14:45:30 - ERROR - SECURITY ALERT: Unauthorized access detected! Authenticated: John Doe, Unknown: 1
```

## Production Deployment

### 1. Run as System Service (Linux)

Create `/etc/systemd/system/surveillance.service`:

```ini
[Unit]
Description=Strongroom Surveillance System
After=network.target

[Service]
Type=simple
User=your-user
WorkingDirectory=/path/to/strongroom_surveillance
ExecStart=/usr/bin/python3 /path/to/strongroom_surveillance/main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable surveillance
sudo systemctl start surveillance
sudo systemctl status surveillance
```

### 2. Use IP Camera (RTSP)

In `config/config.yaml`:
```yaml
camera:
  source: "rtsp://admin:password@192.168.1.100:554/stream1"
```

### 3. Performance Optimization

**For edge devices (Raspberry Pi, Jetson Nano):**
```yaml
detection:
  processing_fps: 2  # Reduce processing frequency
  person_confidence_threshold: 0.6  # Higher threshold = fewer detections

performance:
  enable_frame_skip: true
  num_threads: 4
```

**For GPU systems:**
```yaml
performance:
  use_gpu: true
```

And install GPU-enabled ONNX Runtime:
```bash
pip install onnxruntime-gpu
```

### 4. Email Configuration (Gmail)

1. Enable 2-factor authentication on Gmail
2. Generate App Password: Google Account → Security → App passwords
3. Use app password in config:
```yaml
alerts:
  enable_email: true
  email:
    smtp_server: "smtp.gmail.com"
    smtp_port: 587
    sender_email: "your-email@gmail.com"
    sender_password: "your-16-digit-app-password"
    recipient_emails:
      - "security@company.com"
```

## Troubleshooting

### Models not loading
```
✗ Face detector not found: models/face_detection.onnx
```
**Solution:** Run `python utils/download_models.py` or manually download models.

### Camera not opening
```
Failed to open camera: 0
```
**Solution:** 
- Check camera permissions
- Try different source: `source: 1` or `source: 2`
- For IP camera, verify RTSP URL

### Low FPS
**Solution:**
- Reduce `processing_fps` in config
- Lower camera resolution
- Use GPU acceleration
- Skip frames: `enable_frame_skip: true`

### Face recognition not accurate
**Solution:**
- Enroll more face samples (8-10 instead of 5)
- Adjust `face_recognition_threshold` (lower = stricter)
- Ensure good lighting during enrollment
- Capture faces from different angles

## API Reference

### PersonDetector
```python
from utils.person_detector import PersonDetector

detector = PersonDetector(model_path="models/yolov8n.onnx")
detections = detector.detect(frame)
# Returns: [{'box': [x1, y1, x2, y2], 'confidence': 0.95}, ...]
```

### FaceRecognizer
```python
from utils.face_recognizer import FaceRecognizer, UserDatabase

recognizer = FaceRecognizer(
    detector_path="models/face_detection.onnx",
    recognizer_path="models/face_recognition.onnx"
)

faces = recognizer.detect_faces(frame)
embedding = recognizer.get_face_embedding(frame, face_box)

db = UserDatabase("database/face_encodings.pkl")
match = db.identify_user(embedding, threshold=0.6)
```

### AlertSystem
```python
from utils.alert_system import AlertSystem

alerts = AlertSystem(config)
alerts.send_alert(
    AlertSystem.ALERT_UNAUTHORIZED,
    "Unauthorized access detected",
    data={'person_count': 2, 'unknown_count': 1}
)
```

## Security Best Practices

1. **Physical Security**: Secure the surveillance system hardware
2. **Network Security**: Use VPN for remote camera access
3. **Access Control**: Restrict who can enroll/remove users
4. **Regular Audits**: Review logs periodically
5. **Backup**: Backup user database regularly
6. **Update Models**: Keep detection models updated

## License

This project is provided as-is for surveillance and security applications.

## Support

For issues or questions:
1. Check logs in `logs/surveillance.log`
2. Verify configuration in `config/config.yaml`
3. Test components individually (person detection, face recognition)

## Credits

- **YOLOv8**: Ultralytics (https://github.com/ultralytics/ultralytics)
- **InsightFace**: Face recognition models (https://github.com/deepinsight/insightface)
- **ONNX Runtime**: Microsoft (https://onnxruntime.ai/)
