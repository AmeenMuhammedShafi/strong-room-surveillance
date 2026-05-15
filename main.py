import cv2
import numpy as np
import yaml
import time
from pathlib import Path
from typing import List, Dict, Optional
import threading
from queue import Queue

from utils.person_detector import PersonDetector
from utils.face_recognizer import FaceRecognizer, UserDatabase
from utils.alert_system import AlertSystem


class SurveillanceSystem:
    
    def __init__(self, config_path: str = "config/config.yaml"):
        print("=" * 60)
        print("Strongroom Surveillance System")
        print("=" * 60)
        
        self.config = self._load_config(config_path)
        
        self.person_detector = PersonDetector(
            model_path=self.config['models']['person_detector'],
            conf_threshold=self.config['detection']['person_confidence_threshold']
        )
        
        self.face_recognizer = FaceRecognizer(
            detector_path=self.config['models']['face_detector'],
            recognizer_path=self.config['models']['face_recognizer'],
            det_threshold=self.config['detection']['face_confidence_threshold'],
            rec_threshold=self.config['detection']['face_recognition_threshold']
        )
        
        self.user_db = UserDatabase(
            db_path=self.config['database']['face_encodings_path']
        )
        
        self.alert_system = AlertSystem(
            config=self.config['alerts']
        )
        
        self.processing_fps = self.config['detection']['processing_fps']
        self.frame_skip = self.config['performance'].get('enable_frame_skip', True)
        
        self.current_state = {
            'person_count': 0,
            'authenticated_users': [],
            'unknown_faces': 0,
            'last_check_time': 0
        }
        
        self.frame_queue = Queue(maxsize=2)
        self.result_queue = Queue(maxsize=2)
        self.processing = True
        
        print("=" * 60)
        print("✓ System initialized successfully")
        print("=" * 60)
    
    def _load_config(self, config_path: str) -> dict:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"✓ Configuration loaded from {config_path}")
        return config
    
    def _process_frame_worker(self):
        while self.processing:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                result = self._analyze_frame(frame)
                self.result_queue.put(result)
    
    def _analyze_frame(self, frame: np.ndarray) -> Dict:
        person_detections = self.person_detector.detect(frame)
        person_count = len(person_detections)
        
        authenticated_users = []
        unknown_faces = []
        
        if person_count > 0:
            faces = self.face_recognizer.detect_faces(frame)
            
            for face_box in faces:
                embedding = self.face_recognizer.get_face_embedding(frame, face_box)
                
                if embedding is not None:
                    match = self.user_db.identify_user(
                        embedding, 
                        threshold=self.config['detection']['face_recognition_threshold']
                    )
                    
                    if match:
                        user_id, name, similarity = match
                        authenticated_users.append({
                            'id': user_id,
                            'name': name,
                            'similarity': similarity,
                            'box': face_box
                        })
                    else:
                        unknown_faces.append(face_box)
        
        return {
            'person_count': person_count,
            'person_detections': person_detections,
            'authenticated_users': authenticated_users,
            'unknown_faces': unknown_faces,
            'faces_detected': len(authenticated_users) + len(unknown_faces)
        }
    
    def _evaluate_security_status(self, analysis: Dict):
        person_count = analysis['person_count']
        auth_count = len(analysis['authenticated_users'])
        unknown_count = len(analysis['unknown_faces'])
        
        self.current_state['person_count'] = person_count
        self.current_state['authenticated_users'] = [u['name'] for u in analysis['authenticated_users']]
        self.current_state['unknown_faces'] = unknown_count
        self.current_state['last_check_time'] = time.time()
        
        if person_count == 0:
            pass
        
        elif person_count == 1:
            user_name = analysis['authenticated_users'][0]['name'] if auth_count > 0 else "Unknown"
            self.alert_system.send_alert(
                AlertSystem.ALERT_SINGLE_PERSON,
                f"SECURITY ALERT: Single person detected in strongroom - {user_name}",
                data={
                    'person_count': person_count,
                    'user': user_name,
                    'authenticated': auth_count > 0
                }
            )
        
        elif person_count == 2:
            if auth_count == 2 and unknown_count == 0:
                users = ', '.join([u['name'] for u in analysis['authenticated_users']])
                self.alert_system.send_alert(
                    AlertSystem.ALERT_AUTH_SUCCESS,
                    f"Authorized access: {users}",
                    data={
                        'person_count': person_count,
                        'users': self.current_state['authenticated_users']
                    }
                )
            else:
                auth_names = ', '.join([u['name'] for u in analysis['authenticated_users']])
                self.alert_system.send_alert(
                    AlertSystem.ALERT_UNAUTHORIZED,
                    f"SECURITY ALERT: Unauthorized access detected! Authenticated: {auth_names if auth_names else 'None'}, Unknown: {unknown_count}",
                    data={
                        'person_count': person_count,
                        'authenticated_users': self.current_state['authenticated_users'],
                        'unknown_count': unknown_count
                    }
                )
        
        else:
            auth_names = ', '.join([u['name'] for u in analysis['authenticated_users']])
            self.alert_system.send_alert(
                AlertSystem.ALERT_EXCESS_PEOPLE,
                f"SECURITY ALERT: Excess people detected! Count: {person_count}, Authenticated: {auth_names if auth_names else 'None'}",
                data={
                    'person_count': person_count,
                    'authenticated_users': self.current_state['authenticated_users'],
                    'unknown_count': unknown_count
                }
            )
    
    def _draw_overlay(self, frame: np.ndarray, analysis: Dict) -> np.ndarray:
        overlay = frame.copy()
        
        for det in analysis['person_detections']:
            x1, y1, x2, y2 = det['box']
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        for user in analysis['authenticated_users']:
            x1, y1, x2, y2 = user['box']
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{user['name']} ({user['similarity']:.2f})"
            cv2.putText(overlay, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        for face_box in analysis['unknown_faces']:
            x1, y1, x2, y2 = face_box
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(overlay, "UNKNOWN", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        self._draw_status_panel(overlay, analysis)
        
        return overlay
    
    def _draw_status_panel(self, frame: np.ndarray, analysis: Dict):
        h, w = frame.shape[:2]
        panel_height = 150
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        y_offset = 30
        person_count = analysis['person_count']
        auth_count = len(analysis['authenticated_users'])
        unknown_count = len(analysis['unknown_faces'])
        
        if person_count == 0:
            status = "NORMAL"
            color = (0, 255, 0)
        elif person_count == 1:
            status = "ALERT: SINGLE PERSON"
            color = (0, 165, 255)
        elif person_count == 2 and auth_count == 2 and unknown_count == 0:
            status = "AUTHORIZED"
            color = (0, 255, 0)
        elif person_count == 2:
            status = "ALERT: UNAUTHORIZED"
            color = (0, 0, 255)
        else:
            status = "ALERT: EXCESS PEOPLE"
            color = (0, 0, 255)
        
        cv2.putText(frame, f"STATUS: {status}", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        y_offset += 35
        cv2.putText(frame, f"People Detected: {person_count}", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        y_offset += 30
        cv2.putText(frame, f"Authenticated: {auth_count}", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        y_offset += 30
        cv2.putText(frame, f"Unknown: {unknown_count}", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        if analysis['authenticated_users']:
            y_offset += 30
            users = ', '.join([u['name'] for u in analysis['authenticated_users']])
            cv2.putText(frame, f"Users: {users}", (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def run(self):
        camera_source = self.config['camera']['source']
        cap = cv2.VideoCapture(camera_source)
        
        if not cap.isOpened():
            self.alert_system.logger.error(f"Failed to open camera: {camera_source}")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config['camera']['resolution_width'])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config['camera']['resolution_height'])
        cap.set(cv2.CAP_PROP_BUFFERSIZE, self.config['camera']['buffer_size'])
        
        self.alert_system.log_event("SYSTEM", "Surveillance system started")
        
        process_thread = threading.Thread(target=self._process_frame_worker, daemon=True)
        process_thread.start()
        
        frame_count = 0
        last_analysis = None
        
        print("\n✓ Surveillance active - Press 'q' to quit, 's' to save snapshot")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    self.alert_system.logger.warning("Failed to read frame")
                    break
                
                frame_count += 1
                
                if frame_count % (30 // self.processing_fps) == 0:
                    if not self.frame_queue.full():
                        self.frame_queue.put(frame.copy())
                
                if not self.result_queue.empty():
                    last_analysis = self.result_queue.get()
                    self._evaluate_security_status(last_analysis)
                
                if last_analysis:
                    display_frame = self._draw_overlay(frame, last_analysis)
                else:
                    display_frame = frame
                
                cv2.imshow('Strongroom Surveillance', display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"logs/snapshot_{timestamp}.jpg"
                    cv2.imwrite(filename, display_frame)
                    print(f"✓ Snapshot saved: {filename}")
        
        except KeyboardInterrupt:
            print("\n✓ Shutting down...")
        
        finally:
            self.processing = False
            process_thread.join(timeout=2)
            cap.release()
            cv2.destroyAllWindows()
            self.alert_system.log_event("SYSTEM", "Surveillance system stopped")


def main():
    surveillance = SurveillanceSystem()
    surveillance.run()


if __name__ == "__main__":
    main()
