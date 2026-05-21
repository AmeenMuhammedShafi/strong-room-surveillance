import cv2
import numpy as np
import yaml
import time
from pathlib import Path
from typing import List, Dict, Optional
import threading
from queue import Queue, Empty  # Added Empty exception parsing

from utils.person_detector import PersonDetector
from utils.face_recognizer import FaceRecognizer, UserDatabase
from utils.alert_system import AlertSystem
from utils.tracker import ByteTracker
from utils.live_detector import LiveAnalyzer


class SurveillanceSystem:
    
    def __init__(self, config_path: str = "config/config.yaml"):
        print("=" * 60)
        print("Strongroom Surveillance System Engine")
        print("=" * 60)
        
        self.config = self._load_config(config_path)
        
        # Verify and secure local logging path directories exist
        Path("logs").mkdir(parents=True, exist_ok=True)
        
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
        
        # Live Detection - Production-grade with Head Pose, Blink, Gaze, and Landmark Analysis
        self.live_analyzer = LiveAnalyzer(
            consensus_frames=self.config.get('liveness', {}).get('consensus_frames', 5),
            consensus_threshold=self.config.get('liveness', {}).get('consensus_threshold', 0.60)
        )
        self.spoof_alert_cooldown = {}  # track_id -> last_spoof_alert_time
        
        self.tracker = ByteTracker(
            track_thresh=self.config['detection']['person_confidence_threshold'],
            track_buffer=30,
            frame_rate=self.config['camera']['fps']
        )
        
        self.processing_fps = self.config['detection']['processing_fps']
        self.frame_skip = self.config['performance'].get('enable_frame_skip', True)
        
        # State tracking metrics
        self.current_state = {
            'person_count': 0,
            'authenticated_users': [],
            'unknown_faces': 0,
            'last_check_time': 0
        }
        
        # Thread Synchronization Queues
        self.frame_queue = Queue(maxsize=2)
        self.result_queue = Queue(maxsize=2)
        self.processing = True
        
        print("=" * 60)
        print("[OK] Production System Engine Initialized")
        print("=" * 60)
    
    def _load_config(self, config_path: str) -> dict:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"[OK] Configuration successfully mounted from {config_path}")
        return config
    
    def _process_frame_worker(self):
        """Optimized background calculation worker utilizing non-polling blocking reads."""
        while self.processing:
            try:
                # Blocks worker thread efficiently without pinning CPU core capacity
                frame = self.frame_queue.get(block=True, timeout=0.5)
                result = self._analyze_frame(frame)
                
                # Drop older unread results if main thread encounters UI blocking lag
                if self.result_queue.full():
                    try:
                        self.result_queue.get_nowait()
                    except Empty:
                        pass
                self.result_queue.put(result)
            except Empty:
                continue
    
    def _analyze_frame(self, frame: np.ndarray) -> Dict:
        person_detections = self.person_detector.detect(frame)
        tracked_persons = self.tracker.update(person_detections)
        person_count = len(tracked_persons)
        
        authenticated_users = []
        unknown_faces = []
        spoofed_faces = []
        liveness_data = {}  # track_id -> liveness result
        
        if person_count > 0:
            face_data = self.face_recognizer.get_all_face_embeddings(frame)
            
            for face_info in face_data:
                face_box = face_info['box']
                embedding = face_info['embedding']
                x1, y1, x2, y2 = [int(v) for v in face_box]
                
                # Extract face frame for liveness analysis
                face_frame = frame[y1:y2, x1:x2].copy() if (y1 < y2 and x1 < x2) else frame
                
                track_id = self._find_matching_track(face_box, tracked_persons)
                
                # === LIVENESS DETECTION (Critical Security Gate) ===
                liveness_enabled = self.config.get('liveness', {}).get('enabled', True)
                liveness_result = None
                
                if liveness_enabled and face_frame.size > 0:
                    try:
                        liveness_result = self.live_analyzer.check_face_liveness(face_frame)
                        liveness_data[track_id or -1] = liveness_result
                        
                        is_live = liveness_result.get('is_live', False)
                        
                        if not is_live:
                            # SPOOFING DETECTED - Log and alert
                            spoof_type = "NON_LIVE_DETECTION"
                            consensus_score = liveness_result.get('consensus_score', 0)
                            spoofed_faces.append({
                                'box': face_box,
                                'track_id': track_id,
                                'spoof_type': spoof_type,
                                'liveness_score': consensus_score
                            })
                            
                            # Rate-limited spoofing alerts
                            if self._should_send_spoof_alert(track_id):
                                detection_details = liveness_result.get('detection', {})
                                method_scores = detection_details.get('method_scores', {})
                                self.alert_system.send_alert(
                                    AlertSystem.ALERT_SPOOFING_DETECTED,
                                    f"NON-LIVE DETECTION: Consensus {consensus_score:.1%} | "
                                    f"HeadPose: {method_scores.get('head_pose', 0):.2f} | "
                                    f"Blink: {method_scores.get('blink', 0):.2f} | "
                                    f"Gaze: {method_scores.get('gaze', 0):.2f}",
                                    data={
                                        'track_id': track_id,
                                        'spoof_type': spoof_type,
                                        'consensus_score': consensus_score,
                                        'method_scores': method_scores
                                    }
                                )
                            continue  # Skip recognition for non-live faces
                    except Exception as e:
                        print(f"[ERROR] Live detection check failed: {e}")
                        # Fall through to recognition if live detection check fails
                
                # === FACE RECOGNITION (Only for Live Faces) ===
                match = self.user_db.identify_user(
                    embedding, 
                    threshold=self.config['detection']['face_recognition_threshold']
                )
                
                if match:
                    user_id, name, similarity = match
                    liveness_score = liveness_result.get('consensus_score', 1.0) if liveness_result else 1.0
                    authenticated_users.append({
                        'id': user_id,
                        'name': name,
                        'similarity': similarity,
                        'box': face_box,
                        'track_id': track_id,
                        'liveness_score': liveness_score,
                        'is_live': True
                    })
                    
                    if track_id is not None:
                        self.tracker.set_track_identity(track_id, name, user_id)
                        self.tracker.store_face_embedding(track_id, embedding)
                else:
                    liveness_score = liveness_result.get('consensus_score', 1.0) if liveness_result else 1.0
                    unknown_faces.append({
                        'box': face_box,
                        'track_id': track_id,
                        'liveness_score': liveness_score,
                        'is_live': True
                    })
        
        return {
            'person_count': person_count,
            'person_detections': tracked_persons,
            'authenticated_users': authenticated_users,
            'unknown_faces': unknown_faces,
            'spoofed_faces': spoofed_faces,
            'liveness_data': liveness_data,
            'faces_detected': len(authenticated_users) + len(unknown_faces) + len(spoofed_faces)
        }
    
    def _should_send_spoof_alert(self, track_id: Optional[int]) -> bool:
        """Rate-limit spoofing alerts to prevent spam."""
        if track_id is None:
            return True
        
        spoof_cooldown = self.config.get('liveness', {}).get('spoof_cooldown', 30)
        current_time = time.time()
        
        last_alert = self.spoof_alert_cooldown.get(track_id, 0)
        if current_time - last_alert >= spoof_cooldown:
            self.spoof_alert_cooldown[track_id] = current_time
            return True
        return False
    
    def _find_matching_track(self, face_box: List, tracked_persons: List[Dict]) -> Optional[int]:
        face_cx = (face_box[0] + face_box[2]) / 2
        face_cy = (face_box[1] + face_box[3]) / 2
        
        best_match = None
        best_iou = 0
        
        for person in tracked_persons:
            x1, y1, x2, y2 = person['box']
            if x1 <= face_cx <= x2 and y1 <= face_cy <= y2:
                iou = self._calculate_iou(face_box, person['box'])
                if iou > best_iou:
                    best_iou = iou
                    best_match = person['track_id']
        
        return best_match
    
    @staticmethod
    def _calculate_iou(box1: List, box2: List) -> float:
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)
        
        if inter_xmax < inter_xmin or inter_ymax < inter_ymin:
            return 0.0
        
        inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
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
                f"SECURITY ALERT: Single person occupancy inside vault - {user_name}",
                data={'person_count': person_count, 'user': user_name, 'authenticated': auth_count > 0}
            )
        
        elif person_count == 2:
            if auth_count == 2 and unknown_count == 0:
                users = ', '.join([u['name'] for u in analysis['authenticated_users']])
                self.alert_system.send_alert(
                    AlertSystem.ALERT_AUTH_SUCCESS,
                    f"Dual-Auth Access Granted: {users}",
                    data={'person_count': person_count, 'users': self.current_state['authenticated_users']}
                )
            else:
                auth_names = ', '.join([u['name'] for u in analysis['authenticated_users']])
                self.alert_system.send_alert(
                    AlertSystem.ALERT_UNAUTHORIZED,
                    f"CRITICAL ACCESS BREACH: Authenticated: {auth_names if auth_names else 'None'}, Unknown: {unknown_count}",
                    data={'person_count': person_count, 'authenticated_users': self.current_state['authenticated_users'], 'unknown_count': unknown_count}
                )
        
        else:
            auth_names = ', '.join([u['name'] for u in analysis['authenticated_users']])
            self.alert_system.send_alert(
                AlertSystem.ALERT_EXCESS_PEOPLE,
                f"TAILGATING DETECTED: Group threshold breach! Total: {person_count}",
                data={'person_count': person_count, 'authenticated_users': self.current_state['authenticated_users'], 'unknown_count': unknown_count}
            )
    
    def _draw_overlay(self, frame: np.ndarray, analysis: Dict) -> np.ndarray:
        overlay = frame.copy()
        
        # Track map arrays to isolate and prevent double-drawing over person boundaries
        identified_track_ids = {u['track_id'] for u in analysis['authenticated_users'] if u['track_id'] is not None}
        unknown_track_ids = {f['track_id'] for f in analysis['unknown_faces'] if f['track_id'] is not None}
        spoofed_track_ids = {f['track_id'] for f in analysis.get('spoofed_faces', []) if f['track_id'] is not None}
        
        # 1. Outer Person Boundaries (Draw ONLY if identity recognition skipped/failed for track)
        for det in analysis['person_detections']:
            track_id = det.get('track_id', -1)
            if track_id not in identified_track_ids and track_id not in unknown_track_ids and track_id not in spoofed_track_ids:
                x1, y1, x2, y2 = det['box']
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 165, 0), 1)
                cv2.putText(overlay, f"Locating ID:{track_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 165, 0), 1)
        
        # 2. Confirmed Authorized Entities (Green Canvas Anchor)
        for user in analysis['authenticated_users']:
            x1, y1, x2, y2 = user['box']
            track_id = user.get('track_id', -1)
            liveness_score = user.get('liveness_score', 1.0)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{user['name']} [ID:{track_id}] ({user['similarity']:.2f}) [L:{liveness_score:.2f}]"
            cv2.putText(overlay, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 3. Unauthorized Unidentified Entities (Red Threat Canvas Anchor)
        for face_obj in analysis['unknown_faces']:
            x1, y1, x2, y2 = face_obj['box']
            track_id = face_obj.get('track_id', -1)
            liveness_score = face_obj.get('liveness_score', 1.0)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(overlay, f"BREACH [ID:{track_id}] [L:{liveness_score:.2f}]", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # 4. Spoofed/Fake Faces (Purple - Critical Security Flag)
        for spoof_obj in analysis.get('spoofed_faces', []):
            x1, y1, x2, y2 = spoof_obj['box']
            track_id = spoof_obj.get('track_id', -1)
            spoof_type = spoof_obj.get('spoof_type', 'UNKNOWN')
            liveness_score = spoof_obj.get('liveness_score', 0)
            
            # Double-thickness box for critical alert
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.putText(overlay, f"SPOOF: {spoof_type}", (x1, y1 - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            cv2.putText(overlay, f"[ID:{track_id}] [L:{liveness_score:.2f}]", (x1, y1 - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        
        self._draw_status_panel(overlay, analysis)
        return overlay
    
    def _draw_status_panel(self, frame: np.ndarray, analysis: Dict):
        h, w = frame.shape[:2]
        panel_height = 160
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, panel_height), (10, 10, 10), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        
        person_count = analysis['person_count']
        auth_count = len(analysis['authenticated_users'])
        unknown_count = len(analysis['unknown_faces'])
        spoof_count = len(analysis.get('spoofed_faces', []))
        
        # Determine status based on security posture
        if spoof_count > 0:
            status, color = "CRITICAL: SPOOFING DETECTED", (255, 0, 255)
        elif person_count == 0:
            status, color = "SECURE BASELINE", (0, 255, 0)
        elif person_count == 1:
            status, color = "VIOLATION: SINGLE OCCUPANCY", (0, 165, 255)
        elif person_count == 2 and auth_count == 2 and unknown_count == 0 and spoof_count == 0:
            status, color = "DUAL-AUTH VERIFIED", (0, 255, 0)
        elif person_count == 2:
            status, color = "CRITICAL: INTRUSION DETECTED", (0, 0, 255)
        else:
            status, color = "CRITICAL: COVERT TAILGATING", (0, 0, 255)
        
        cv2.putText(frame, f"SURVEILLANCE: {status}", (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
        
        cv2.putText(frame, f"Active Tracks: {person_count}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(frame, f"Authenticated: {auth_count}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
        cv2.putText(frame, f"Unknown: {unknown_count}", (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1)
        cv2.putText(frame, f"Spoofed: {spoof_count}", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 1)
        
        if analysis['authenticated_users']:
            users = ', '.join([u['name'] for u in analysis['authenticated_users']])
            cv2.putText(frame, f"Clearance Logs: {users}", (240, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        if spoof_count > 0:
            spoof_types = ', '.join(set([s.get('spoof_type', 'UNKNOWN') for s in analysis.get('spoofed_faces', [])]))
            cv2.putText(frame, f"Attack Types: {spoof_types}", (240, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

    def run(self):
        camera_source = self.config['camera']['source']
        cap = cv2.VideoCapture(camera_source)
        
        if not cap.isOpened():
            self.alert_system.logger.error(f"Hardware Fault - Stream down: {camera_source}")
            return
        
        # Apply camera configuration profiles
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config['camera']['resolution_width'])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config['camera']['resolution_height'])
        cap.set(cv2.CAP_PROP_BUFFERSIZE, self.config['camera']['buffer_size'])
        
        # Dynamically discover frame-rate profile properties from source metadata
        source_fps = int(cap.get(cv2.CAP_PROP_FPS))
        if source_fps <= 0 or source_fps > 120:
            source_fps = self.config['camera'].get('fps', 30)
            
        # Dynamically calculate frame processing interval loops
        skip_interval = max(1, source_fps // self.processing_fps)
        
        self.alert_system.log_event("SYSTEM", "Surveillance core active runtime context created")
        
        process_thread = threading.Thread(target=self._process_frame_worker, daemon=True)
        process_thread.start()
        
        frame_count = 0
        last_analysis = None
        
        print(f"\n[OK] Runloop Online ({source_fps} FPS Source) -> Processing at {self.processing_fps} Hz")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Push analytical frame down instances based on dynamic calculation skipping
                if frame_count % skip_interval == 0:
                    if not self.frame_queue.full():
                        self.frame_queue.put(frame.copy())
                
                # Non-blocking evaluation read checks inside standard update cycles
                if not self.result_queue.empty():
                    last_analysis = self.result_queue.get()
                    self._evaluate_security_status(last_analysis)
                
                display_frame = self._draw_overlay(frame, last_analysis) if last_analysis else frame
                cv2.imshow('Strongroom Surveillance System Terminal', display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    filename = f"logs/snapshot_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
                    cv2.imwrite(filename, display_frame)
                    print(f"[OK] High-resolution context snapshot exported: {filename}")
                    
        finally:
            self.processing = False
            process_thread.join(timeout=1.0)
            cap.release()
            cv2.destroyAllWindows()
            self.alert_system.log_event("SYSTEM", "Surveillance core cleanly decommissioned")


if __name__ == "__main__":
    surveillance = SurveillanceSystem()
    surveillance.run()