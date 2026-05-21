import cv2
import numpy as np
from typing import Dict, Tuple, Optional
import logging


class ProductionLivenessDetector:
    """
    Hardened multi-factor liveness detection engine.
    Implements mandatory spatial normalization to prevent shape mismatches.
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.threshold = self.config.get('threshold', 0.5)
        
        # Feature toggles
        self.enable_texture = self.config.get('texture_analysis', True)
        self.enable_frequency = self.config.get('frequency_analysis', True)
        self.enable_motion = self.config.get('motion_detection', True)
        
        # Target sizing for spatial normalization
        self.target_size = (256, 256)
        
        # Temporal buffers: track_id -> historical frames/metrics
        self.prev_gray_frames = {}  
        self.motion_history = {}    
        self.logger = logging.getLogger(__name__)
    
    def detect_liveness(self, face_frame: np.ndarray, track_id: int) -> Dict:
        if face_frame is None or face_frame.size == 0:
            return {"is_live": False, "liveness_score": 0.0, "confidence": 1.0, "spoof_type": "INVALID_INPUT"}

        # Normalize resolution across all frames to secure consistency in matrix operations
        resized = cv2.resize(face_frame, self.target_size)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if len(resized.shape) == 3 else resized
        
        scores = {}
        weights = {}
        
        # 1. Texture Analysis (Forbid flat surfaces/print attacks)
        if self.enable_texture:
            scores['texture'] = self._analyze_texture(gray)
            weights['texture'] = 0.35
            
        # 2. Frequency Domain Analysis (Detect screen replay Moire patterns)
        if self.enable_frequency:
            scores['frequency'] = self._analyze_frequency(gray)
            weights['frequency'] = 0.35
            
        # 3. Micro-Motion Analysis (Enforce Farneback optical flow delta check)
        if self.enable_motion and track_id is not None:
            scores['motion'] = self._analyze_motion_flow(gray, track_id)
            weights['motion'] = 0.30

        # Calculate final weighted performance
        total_weight = sum(weights.values())
        liveness_score = sum(scores[k] * weights[k] for k in scores) / total_weight if total_weight > 0 else 0.5
        
        is_live = liveness_score >= self.threshold
        confidence = abs(liveness_score - 0.5) * 2
        
        return {
            'is_live': is_live,
            'liveness_score': float(liveness_score),
            'confidence': float(confidence),
            'method_scores': scores,
            'spoof_type': self._classify_spoof(scores) if not is_live else 'GENUINE'
        }

    def _analyze_texture(self, gray: np.ndarray) -> float:
        """Evaluates surface micro-texture via Laplacian variance."""
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        # Adjusted denominator for standard webcam grain profiles
        return float(min(variance / 150.0, 1.0))

    def _analyze_frequency(self, gray: np.ndarray) -> float:
        """FFT analysis to spot artificial periodic structural frequencies."""
        dft = np.fft.fft2(gray)
        dft_shift = np.fft.fftshift(dft)
        magnitude = np.abs(dft_shift)
        
        h, w = magnitude.shape
        cx, cy = h // 2, w // 2
        
        # Safe normalized masking bounds on a fixed 256x256 frame
        center_mask = np.zeros_like(magnitude)
        center_mask[cx-12:cx+12, cy-12:cy+12] = 1
        
        center_energy = np.sum(magnitude * center_mask)
        outer_energy = np.sum(magnitude * (1 - center_mask))
        
        ratio = center_energy / (outer_energy + 1e-6)
        return float(min(ratio / 12.0, 1.0))

    def _analyze_motion_flow(self, gray: np.ndarray, track_id: int) -> float:
        """Calculates true localized optical displacement between frames."""
        score = 0.5 
        
        if track_id in self.prev_gray_frames:
            prev_gray = self.prev_gray_frames[track_id]
            
            # Guaranteed to match due to structural front-gate resizing
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            
            # Threshold micro-movements to avoid noise spikes from sensor grain
            moving_pixels = magnitude[magnitude > 0.8]
            mean_motion = np.mean(moving_pixels) if moving_pixels.size > 0 else 0.0
            
            if track_id not in self.motion_history:
                self.motion_history[track_id] = []
            self.motion_history[track_id].append(mean_motion)
            self.motion_history[track_id] = self.motion_history[track_id][-10:]
            
            if len(self.motion_history[track_id]) > 3:
                # Live targets yield fluctuating velocity curves; static targets display flatlines
                std_dev = np.std(self.motion_history[track_id])
                score = min(std_dev * 15.0, 1.0)
                if score < 0.15 and np.mean(self.motion_history[track_id]) > 2.0:
                    # Catch uniform translation attacks (moving a print paper smoothly across space)
                    score = 0.10
        
        self.prev_gray_frames[track_id] = gray
        return float(score)

    def _classify_spoof(self, scores: Dict) -> str:
        if scores.get('texture', 1.0) < 0.25: return 'PRINTED_PHOTO'
        if scores.get('frequency', 1.0) < 0.25: return 'SCREEN_REPLAY'
        if scores.get('motion', 1.0) < 0.20: return 'STATIC_SPOOF'
        return 'SUSPICIOUS'

    def cleanup_track(self, track_id: int):
        self.prev_gray_frames.pop(track_id, None)
        self.motion_history.pop(track_id, None)


class LivenessAnalyzer:
    """
    High-level analyzer wrapper for main.py compatibility.
    """
    def __init__(self, config: Dict = None):
        self.detector = ProductionLivenessDetector(config or {})
        self.logger = logging.getLogger(__name__)
        self.liveness_history = {}  
    
    def check_face_liveness(self, face_frame: np.ndarray, face_box: Tuple,
                           track_id: int, embedding: Optional[np.ndarray] = None) -> Dict:
        result = self.detector.detect_liveness(face_frame, track_id or -1)
        
        if track_id not in self.liveness_history:
            self.liveness_history[track_id] = []
        
        self.liveness_history[track_id].append(result['is_live'])
        self.liveness_history[track_id] = self.liveness_history[track_id][-5:]  # Extended pool
        
        if len(self.liveness_history[track_id]) >= 3:
            live_count = sum(self.liveness_history[track_id])
            # Require at least 60% of current window frames to evaluate as true
            consensus = live_count >= len(self.liveness_history[track_id]) * 0.6
            result['consensus_live'] = consensus
        else:
            result['consensus_live'] = result['is_live']
        
        return result
    
    def cleanup_track(self, track_id: int):
        self.detector.cleanup_track(track_id)
        self.liveness_history.pop(track_id, None)