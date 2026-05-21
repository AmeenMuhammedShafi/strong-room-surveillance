import os
import sys
import codecs

# Force UTF-8 stdout/stderr encoding to prevent Windows cp1252 console unicode crashes
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import cv2
import numpy as np
import yaml
import time
import queue
import logging
import asyncio
import threading
from typing import List, Dict, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from main import SurveillanceSystem
from utils.face_recognizer import FaceRecognizer, UserDatabase

# Initialize FastAPI App
app = FastAPI(title="Strongroom Surveillance API", version="1.0.0")

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global structures for real-time logs
websocket_clients = set()
recent_logs = []
logs_lock = threading.Lock()

# Custom log handler to pipe all surveillance events to WebSockets and UI
class WebSocketLogHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    def emit(self, record):
        try:
            log_msg = self.format(record)
            level = record.levelname
            timestamp = time.strftime("%H:%M:%S")
            
            log_item = {
                "id": str(time.time_ns()),
                "timestamp": timestamp,
                "level": level,
                "message": record.getMessage()
            }
            
            # Save to recent logs buffer
            with logs_lock:
                recent_logs.append(log_item)
                if len(recent_logs) > 100:
                    recent_logs.pop(0)
            
            # Broadcast to all connected WebSockets
            if websocket_clients:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    for ws in list(websocket_clients):
                        asyncio.run_coroutine_threadsafe(ws.send_json(log_item), loop)
        except Exception as e:
            pass

# Setup logging interceptor
ws_handler = WebSocketLogHandler()
ws_handler.setLevel(logging.INFO)
root_logger = logging.getLogger()
root_logger.addHandler(ws_handler)

# Headless Surveillance Runner
class HeadlessSurveillanceRunner:
    def __init__(self):
        self.surveillance = SurveillanceSystem()
        self.latest_frame = None
        self.lock = threading.Lock()
        self.thread = None
        self.active = False
        
        # Override the alert logger specifically to ensure we capture alert details
        self.surveillance.alert_system.logger.addHandler(ws_handler)

    def start(self):
        if self.thread is None or not self.thread.is_alive():
            self.active = True
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()
            logging.info("[SYSTEM] Headless surveillance system started in background thread")

    def stop(self):
        logging.info("[SYSTEM] Stopping surveillance system and releasing camera...")
        self.active = False
        if self.thread:
            self.thread.join(timeout=2)
        logging.info("[SYSTEM] Surveillance system stopped, camera released")

    def _run(self):
        camera_source = self.surveillance.config['camera']['source']
        cap = cv2.VideoCapture(camera_source)
        
        if not cap.isOpened():
            logging.error(f"[SYSTEM] Failed to open camera source: {camera_source}")
            self.active = False
            return
            
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.surveillance.config['camera']['resolution_width'])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.surveillance.config['camera']['resolution_height'])
        cap.set(cv2.CAP_PROP_BUFFERSIZE, self.surveillance.config['camera']['buffer_size'])
        
        frame_count = 0
        last_analysis = None
        frame_read_errors = 0
        
        try:
            while self.active:
                ret, frame = cap.read()
                if not ret:
                    # Camera temporarily unavailable (browser might have it)
                    frame_read_errors += 1
                    if frame_read_errors > 10:
                        logging.warning(f"[SYSTEM] Camera unavailable for {frame_read_errors} frames. Browser may be using it.")
                        frame_read_errors = 0  # Reset counter
                    time.sleep(0.05)
                    continue
                
                # Reset error counter on successful read
                if frame_read_errors > 0:
                    frame_read_errors = 0
                    logging.info("[SYSTEM] Camera recovered")
                    
                frame_count += 1
                
                # Analyze frame at designated processing FPS
                processing_fps = self.surveillance.processing_fps
                if frame_count % max(1, (30 // processing_fps)) == 0:
                    try:
                        last_analysis = self.surveillance._analyze_frame(frame)
                        self.surveillance._evaluate_security_status(last_analysis)
                    except Exception as e:
                        logging.error(f"[DETECTION] Error during frame analysis: {e}")
                
                # Draw visual bounding box rectangles and status overlays
                if last_analysis:
                    display_frame = self.surveillance._draw_overlay(frame, last_analysis)
                else:
                    display_frame = frame
                    
                # Compress to JPG and store in memory
                ret_enc, jpeg = cv2.imencode('.jpg', display_frame)
                if ret_enc:
                    with self.lock:
                        self.latest_frame = jpeg.tobytes()
                        
                time.sleep(1.0 / 30.0) # Process/capture at standard frame rate
        finally:
            logging.info("[SYSTEM] Camera loop exiting, releasing camera...")
            cap.release()
            logging.info("[SYSTEM] Camera successfully released")

# Instantiate global runner
runner = HeadlessSurveillanceRunner()

@app.on_event("startup")
async def startup_event():
    runner.start()

@app.on_event("shutdown")
async def shutdown_event():
    runner.stop()

# Real-time Motion JPEG (MJPEG) frame streamer for high-performance frontend canvas rendering
def frame_generator():
    while True:
        if runner.latest_frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + runner.latest_frame + b'\r\n')
        time.sleep(1.0 / 15.0) # Yield up to 15 FPS to save bandwidth

@app.get("/api/video")
async def get_video_stream():
    return StreamingResponse(
        frame_generator(), 
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

# GET camera status
@app.get("/api/camera/status")
async def get_camera_status():
    return {
        "active": runner.active,
        "source": runner.surveillance.config['camera']['source']
    }

# POST start camera
@app.post("/api/camera/start")
async def start_camera():
    runner.start()
    return {"status": "success", "message": "Surveillance camera successfully started"}

# POST stop camera
@app.post("/api/camera/stop")
async def stop_camera():
    runner.stop()
    return {"status": "success", "message": "Surveillance camera successfully stopped and released"}

# WebSocket Endpoint for streaming security events
@app.websocket("/ws/logs")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    websocket_clients.add(websocket)
    
    # Catch up clients on recent history
    with logs_lock:
        for log in recent_logs:
            try:
                await websocket.send_json(log)
            except Exception:
                break
                
    try:
        while True:
            # Keep-alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        websocket_clients.remove(websocket)

# GET enrolled users
@app.get("/api/users")
async def list_users():
    try:
        users = runner.surveillance.user_db.get_all_users()
        return [{"id": uid, "name": name} for uid, name in users.items()]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# DELETE enrolled user
@app.delete("/api/users/{user_id}")
async def delete_user(user_id: str):
    try:
        success = runner.surveillance.user_db.remove_user(user_id)
        if success:
            logging.info(f"[DATABASE] Removed user ID: {user_id}")
            return {"status": "success", "message": f"User {user_id} successfully deleted"}
        else:
            raise HTTPException(status_code=404, detail="User not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# POST dynamic face enrollment with up to 5 uploaded files or base64 frames
@app.post("/api/enroll")
async def enroll_user(
    user_id: str = Form(...),
    name: str = Form(...),
    files: List[UploadFile] = File(...)
):
    if len(files) == 0:
        raise HTTPException(status_code=400, detail="Please upload at least one image file")
    if len(files) > 5:
        raise HTTPException(status_code=400, detail="A maximum of 5 images can be uploaded for enrollment")
        
    logging.info(f"[ENROLL] Processing enrollment for {name} (ID: {user_id}) with {len(files)} files")
    
    embeddings = []
    processed_count = 0
    failed_reasons = []
    
    for upload_file in files:
        try:
            file_bytes = await upload_file.read()
            nparr = np.frombuffer(file_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                failed_reasons.append(f"{upload_file.filename}: Invalid image format")
                continue
                
            faces = runner.surveillance.face_recognizer.detect_faces(img)
            
            if len(faces) == 0:
                failed_reasons.append(f"{upload_file.filename}: No face detected")
                continue
                
            # If multiple faces detected, use the primary face
            face_box = faces[0]
            embedding = runner.surveillance.face_recognizer.get_face_embedding(img, face_box)
            
            if embedding is not None:
                embeddings.append(embedding)
                processed_count += 1
            else:
                failed_reasons.append(f"{upload_file.filename}: Failed to extract face embedding")
        except Exception as ex:
            failed_reasons.append(f"{upload_file.filename}: Error: {str(ex)}")
            
    if len(embeddings) > 0:
        runner.surveillance.user_db.add_user(user_id, name, embeddings)
        logging.info(f"[ENROLL] User '{name}' (ID: {user_id}) enrolled with {len(embeddings)} face samples.")
        
        return {
            "status": "success",
            "message": f"Successfully enrolled user '{name}' with {len(embeddings)} face samples.",
            "processed": processed_count,
            "failed": len(files) - processed_count,
            "errors": failed_reasons
        }
    else:
        logging.error(f"[ENROLL] Enrollment failed for {name}. Errors: {failed_reasons}")
        raise HTTPException(
            status_code=400, 
            detail={
                "message": "Enrollment failed. No valid face embeddings could be extracted.",
                "errors": failed_reasons
            }
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("web_server:app", host="0.0.0.0", port=8000, reload=False)
