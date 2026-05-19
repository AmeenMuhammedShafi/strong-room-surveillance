import numpy as np
from typing import List, Dict, Tuple, Optional
import cv2


class STrack:
    """Single Track object in ByteTrack"""
    shared_counter = 0

    def __init__(self, tlwh, score, temp_feat=None, buffer_size=30):
        self.tlwh = np.asarray(tlwh, dtype=np.float32)
        self.score = score
        self.track_id = 0
        self.is_activated = False
        self.frame_id = 0
        
        self.start_frame = 0
        self.tracklet_len = 0
        self.state = TrackState.New
        
        # Store face embeddings
        self.temp_feat = temp_feat
        
        # Track history for face recognition
        self.face_embeddings = []
        self.identified_as = None
        self.positions = []
        
        self._tlwh_to_xyxy = np.array([0, 1, 2, 3], dtype=np.float32)

    def post_process(self):
        if self.state == TrackState.Tracked:
            self.tracklet_len += 1

    def activate(self, track_id, frame_id):
        self.track_id = track_id
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_strack, frame_id):
        self.tlwh = new_strack.tlwh
        self.score = new_strack.score
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id

    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_removed(self):
        self.state = TrackState.Removed

    @property
    def tlbr(self):
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @property
    def tlxy(self):
        """Get track center point"""
        tlwh = self.tlwh
        return np.array([tlwh[0], tlwh[1], tlwh[0] + tlwh[2] / 2, tlwh[1] + tlwh[3] / 2])

    @staticmethod
    def tlwh_to_xyxy(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    def xyxy_to_tlwh(xyxy):
        ret = np.asarray(xyxy).copy()
        ret[2:] -= ret[:2]
        return ret

    def __repr__(self):
        return f'Track({self.track_id}, {self.state})'


class TrackState:
    """Track state enumeration"""
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class ByteTracker:
    """ByteTrack implementation for multi-object tracking"""
    
    def __init__(self, track_thresh=0.6, track_buffer=30, frame_rate=30):
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.frame_id = 0
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []
        
        self.next_track_id = 1
        self.track_history = {}
        
        # Kalman filter initialization
        self.kalman_filter = KalmanFilterXYAH()
        
        print(f"✓ ByteTracker initialized")
        print(f"  Track threshold: {track_thresh}, Buffer: {track_buffer}, FPS: {frame_rate}")

    def update(self, dets: List[Dict]) -> List[Dict]:
        """Update tracks with detections"""
        self.frame_id += 1
        
        # Parse detections: box format [x1, y1, x2, y2] -> convert to tlwh
        if len(dets) == 0:
            self.tracked_stracks = []
        
        # Convert to STrack format
        detections = []
        for det in dets:
            x1, y1, x2, y2 = det['box']
            tlwh = np.array([x1, y1, x2 - x1, y2 - y1], dtype=np.float32)
            score = det.get('confidence', 1.0)
            detections.append(STrack(tlwh, score))

        # Perform tracking
        ''' First association, with IOU'''
        strack_pool = self.tracked_stracks + self.lost_stracks
        
        iou_distance = iou_batch(strack_pool, detections)
        iou_sim = np.maximum(0.0, iou_distance)
        
        matched, unmatched_a, unmatched_b = linear_assignment(1 - iou_sim, 0.5)
        
        for i, j in matched:
            track = strack_pool[i]
            det = detections[j]
            if track.state == TrackState.Tracked:
                track.re_activate(det, self.frame_id)
            else:
                track.re_activate(det, self.frame_id)

        ''' Second association, using appearance features'''
        detections = [detections[i] for i in unmatched_b]
        r_tracked_stracks = [strack_pool[i] for i in unmatched_a if strack_pool[i].state == TrackState.Tracked]
        
        iou_distance = iou_batch(r_tracked_stracks, detections)
        iou_sim = np.maximum(0.0, iou_distance)
        matched, unmatched_a, unmatched_b = linear_assignment(1 - iou_sim, 0.5)
        
        for i, j in matched:
            track = r_tracked_stracks[i]
            det = detections[j]
            if track.state == TrackState.Tracked:
                track.re_activate(det, self.frame_id)

        for i in unmatched_b:
            track = detections[i]
            self._new_track(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks.extend([t for t in self.lost_stracks if t.state == TrackState.Tracked])
        
        self.lost_stracks = [t for t in self.tracked_stracks + self.lost_stracks if t.state == TrackState.Lost]
        self.removed_stracks = [t for t in self.removed_stracks if t.state == TrackState.Removed]

        self._cleanup_old_tracks()

        # Return tracked detections in original format
        result = []
        for track in self.tracked_stracks:
            tlwh = track.tlwh
            x1, y1, w, h = tlwh
            box = [int(x1), int(y1), int(x1 + w), int(y1 + h)]
            
            result.append({
                'box': box,
                'confidence': track.score,
                'track_id': track.track_id
            })
            
            # Initialize track history if needed
            if track.track_id not in self.track_history:
                self.track_history[track.track_id] = {
                    'first_seen': self.frame_id,
                    'last_seen': self.frame_id,
                    'face_embeddings': [],
                    'identified_as': None,
                    'positions': [box]
                }
            else:
                self.track_history[track.track_id]['last_seen'] = self.frame_id
                self.track_history[track.track_id]['positions'].append(box)

        return result

    def _new_track(self, track):
        if track.score >= self.track_thresh:
            track.activate(self.next_track_id, self.frame_id)
            self.tracked_stracks.append(track)
            self.track_history[self.next_track_id] = {
                'first_seen': self.frame_id,
                'last_seen': self.frame_id,
                'face_embeddings': [],
                'identified_as': None,
                'positions': []
            }
            self.next_track_id += 1

    def _cleanup_old_tracks(self, max_age=30):
        for track_id, history in list(self.track_history.items()):
            if self.frame_id - history['last_seen'] > max_age:
                del self.track_history[track_id]

    def store_face_embedding(self, track_id: int, embedding: np.ndarray):
        if track_id in self.track_history:
            self.track_history[track_id]['face_embeddings'].append(embedding)

    def get_track_identity(self, track_id: int) -> Optional[Dict]:
        if track_id in self.track_history:
            return self.track_history[track_id]
        return None

    def set_track_identity(self, track_id: int, user_name: str, user_id: str):
        if track_id in self.track_history:
            self.track_history[track_id]['identified_as'] = {
                'user_id': user_id,
                'user_name': user_name
            }

    def get_active_track_count(self) -> int:
        return len(self.tracked_stracks)

    def reset(self):
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []
        self.frame_id = 0
        self.next_track_id = 1
        self.track_history = {}


class KalmanFilterXYAH:
    """Simple Kalman filter for tracking"""
    
    def __init__(self):
        ndim = 4
        dt = 1.0
        
        self.std_weight_position = 1.0 / 20
        self.std_weight_velocity = 1.0 / 160

    def predict(self, mean, covariance):
        std_pos = [
            self.std_weight_position * mean[3],
            self.std_weight_position * mean[3],
            1e-2,
            self.std_weight_position * mean[3]
        ]
        std_vel = [
            self.std_weight_velocity * mean[3],
            self.std_weight_velocity * mean[3],
            1e-5,
            self.std_weight_velocity * mean[3]
        ]

        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        return mean, covariance + motion_cov

    def update(self, mean, covariance, measurement):
        std_pos = [
            self.std_weight_position * mean[3],
            self.std_weight_position * mean[3],
            1e-1,
            self.std_weight_position * mean[3]
        ]

        innovation_cov = np.diag(np.square(std_pos))
        innovation = measurement - mean
        chol_factor, lower = np.linalg.cholesky(innovation_cov, lower=True)
        
        kalman_gain = covariance @ np.linalg.solve(chol_factor.T, np.eye(4)).T
        new_mean = mean + innovation @ kalman_gain.T
        new_covariance = covariance - kalman_gain @ innovation_cov @ kalman_gain.T
        
        return new_mean, new_covariance


def iou_batch(atracks, btracks):
    """Compute IOU distance between tracks"""
    if len(atracks) > 0 and len(btracks) > 0:
        ious_dists = np.zeros((len(atracks), len(btracks)))
        for i, atrack in enumerate(atracks):
            for j, btrack in enumerate(btracks):
                ious_dists[i, j] = iou(atrack.tlbr, btrack.tlbr)
        return ious_dists
    else:
        return np.zeros((len(atracks), len(btracks)))


def iou(box1, box2):
    """Calculate IOU between two boxes"""
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


def linear_assignment(cost_matrix, thresh):
    """Hungarian algorithm for assignment"""
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    
    matches, unmatches_a, unmatches_b = [], [], []
    
    if cost_matrix.shape[1] == 0:
        unmatches_a = list(range(cost_matrix.shape[0]))
    elif cost_matrix.shape[0] == 0:
        unmatches_b = list(range(cost_matrix.shape[1]))
    else:
        try:
            from scipy.optimize import linear_sum_assignment
            cost_matrix = np.asarray(cost_matrix)
            x, y = linear_sum_assignment(cost_matrix)
            for i, j in zip(x, y):
                if cost_matrix[i, j] > thresh:
                    unmatches_a.append(i)
                    unmatches_b.append(j)
                else:
                    matches.append([i, j])
            unmatches_a = [i for i in range(cost_matrix.shape[0]) if i not in x]
            unmatches_b = [j for j in range(cost_matrix.shape[1]) if j not in y]
        except:
            pass

    return np.asarray(matches), np.asarray(unmatches_a), np.asarray(unmatches_b)


# Alias for backward compatibility
PersonTracker = ByteTracker
