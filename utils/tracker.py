"""
ByteTrack-based object tracker for consistent person tracking across frames
Using YOLOX's implementation
"""

import numpy as np
from typing import List, Dict, Tuple, Optional


class Track:
    """Simple track object to store tracking information"""
    def __init__(self, bbox, track_id, conf=1.0):
        self.bbox = np.array(bbox, dtype=np.float32)
        self.track_id = track_id
        self.conf = conf


class PersonTracker:
    """
    Simplified ByteTrack-like tracker for consistent person tracking across frames
    Uses centroid tracking with IoU matching for association
    """
    
    def __init__(self, track_thresh: float = 0.5, track_buffer: int = 30, frame_rate: int = 30):
        """
        Initialize tracker
        
        Args:
            track_thresh: Detection threshold for tracking
            track_buffer: Maximum frames to keep track without detection
            frame_rate: Video frame rate
        """
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.frame_rate = frame_rate
        self.tracked_tracks = {}  # track_id -> Track
        self.next_track_id = 1
        self.track_history = {}  # Store track ID to identity mapping
        print(f"✓ PersonTracker initialized")
        print(f"  Track threshold: {track_thresh}, Buffer: {track_buffer}, FPS: {frame_rate}")
    
    def update(self, detections: List[Dict]) -> List[Dict]:
        """
        Update tracker with new detections and return tracked objects
        
        Args:
            detections: List of detection dicts with 'box' and 'confidence' keys
                       box format: [x1, y1, x2, y2]
        
        Returns:
            List of tracked detections with added 'track_id' key
        """
        if not detections:
            self._cleanup_old_tracks()
            return []
        
        # Match detections to existing tracks
        tracked_detections = []
        used_detections = set()
        
        for track_id, track in list(self.tracked_tracks.items()):
            best_match_idx = -1
            best_iou = 0.3  # Minimum IOU threshold
            
            for det_idx, det in enumerate(detections):
                if det_idx in used_detections:
                    continue
                
                iou = self._calculate_iou(track.bbox.astype(int).tolist(), det['box'])
                if iou > best_iou:
                    best_iou = iou
                    best_match_idx = det_idx
            
            if best_match_idx >= 0:
                # Update existing track
                det = detections[best_match_idx]
                track.bbox = np.array(det['box'], dtype=np.float32)
                track.conf = det['confidence']
                
                tracked_detections.append({
                    'box': det['box'],
                    'confidence': det['confidence'],
                    'track_id': track_id
                })
                used_detections.add(best_match_idx)
                
                # Update track history
                if track_id not in self.track_history:
                    self.track_history[track_id] = {
                        'first_seen': 0,
                        'last_seen': 0,
                        'face_embeddings': [],
                        'identified_as': None,
                        'positions': []
                    }
                
                self.track_history[track_id]['last_seen'] = 0
                self.track_history[track_id]['positions'].append(det['box'])
        
        # Create new tracks for unmatched detections
        for det_idx, det in enumerate(detections):
            if det_idx not in used_detections and det['confidence'] >= self.track_thresh:
                new_track_id = self.next_track_id
                self.next_track_id += 1
                
                track = Track(det['box'], new_track_id, det['confidence'])
                self.tracked_tracks[new_track_id] = track
                
                tracked_detections.append({
                    'box': det['box'],
                    'confidence': det['confidence'],
                    'track_id': new_track_id
                })
                
                # Initialize track history
                self.track_history[new_track_id] = {
                    'first_seen': 0,
                    'last_seen': 0,
                    'face_embeddings': [],
                    'identified_as': None,
                    'positions': [det['box']]
                }
        
        # Cleanup old tracks
        self._cleanup_old_tracks()
        
        return tracked_detections
    
    def _cleanup_old_tracks(self, max_age: int = 30):
        """Remove old tracks that haven't been seen in a while"""
        tracks_to_remove = []
        for track_id, track in self.tracked_tracks.items():
            self.track_history[track_id]['last_seen'] += 1
            if self.track_history[track_id]['last_seen'] > max_age:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.tracked_tracks[track_id]
    
    @staticmethod
    def _calculate_iou(box1: List, box2: List) -> float:
        """Calculate Intersection over Union for two boxes"""
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
    
    def store_face_embedding(self, track_id: int, embedding: np.ndarray):
        """Store face embedding for a tracked person"""
        if track_id in self.track_history:
            self.track_history[track_id]['face_embeddings'].append(embedding)
    
    def get_track_identity(self, track_id: int) -> Optional[Dict]:
        """Get stored identity information for a tracked person"""
        if track_id in self.track_history:
            return self.track_history[track_id]
        return None
    
    def set_track_identity(self, track_id: int, user_name: str, user_id: str):
        """Set identified user for a track"""
        if track_id in self.track_history:
            self.track_history[track_id]['identified_as'] = {
                'user_id': user_id,
                'user_name': user_name
            }
    
    def get_active_track_count(self) -> int:
        """Get number of currently active tracks"""
        return len(self.tracked_tracks)
    
    def reset(self):
        """Reset tracker state"""
        self.tracked_tracks = {}
        self.next_track_id = 1
        self.track_history = {}
