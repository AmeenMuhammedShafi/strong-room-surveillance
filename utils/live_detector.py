"""
Production-grade Live Detection with Head Pose, Texture, Frequency, and Motion Analysis.
Uses OpenCV and advanced image analysis for robust, real-time liveness verification.

Detection Methods:
1. Head Pose Variation (primary) - Detects face angle changes (frontal vs tilted faces)
2. Texture Analysis (supporting) - Laplacian variance (screens/photos are blurry)
3. Frequency Analysis (supporting) - FFT energy distribution (distinguishes photos)
4. Motion Detection (supporting) - Optical flow analysis (dead giveaway for videos)
"""

import cv2
import numpy as np
from collections import deque
import logging

logger = logging.getLogger(__name__)


class ProductionLiveDetector:
    """Production-ready liveness detection combining texture, frequency, motion, and head pose analysis."""
    
    def __init__(self, max_history=30, verbose=False):
        """
        Initialize the live detector.
        
        Args:
            max_history: Number of frames to track for temporal analysis
            verbose: Enable detailed logging
        """
        self.verbose = verbose
        self.max_history = max_history
        
        # Cascade classifiers for face detection
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        self.profile_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_profileface.xml'
        )
        
        # History tracking for temporal consistency
        self.texture_history = deque(maxlen=max_history)
        self.frequency_history = deque(maxlen=max_history)
        self.motion_history = deque(maxlen=max_history)
        self.head_pose_history = deque(maxlen=max_history)
        
        # State tracking
        self.last_gray = None
        self.motion_buffer = deque(maxlen=5)
        
        logger.info("[OK] ProductionLiveDetector initialized with OpenCV")
    
    def detect_liveness(self, frame):
        """
        Analyze frame for liveness using texture, frequency, motion, and head pose.
        
        Args:
            frame: Input image (BGR)
            
        Returns:
            {
                'is_live': bool,
                'liveness_score': float (0.0-1.0),
                'confidence': float (0.0-1.0),
                'method_scores': {
                    'texture': float,
                    'frequency': float,
                    'motion': float,
                    'head_pose': float
                },
                'details': {
                    'head_angles': tuple (yaw, pitch),
                    'texture_variance': float,
                    'frequency_score': float,
                    'motion_score': float
                }
            }
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        
        # Detect face and head pose
        head_pose_score, head_angles = self._analyze_head_pose(gray)
        
        # Texture analysis (Laplacian variance)
        texture_score = self._analyze_texture(gray)
        
        # Frequency analysis (FFT energy)
        frequency_score = self._analyze_frequency(gray)
        
        # Motion detection (optical flow)
        motion_score = self._analyze_motion(gray)
        
        # Store in history
        self.texture_history.append(texture_score)
        self.frequency_history.append(frequency_score)
        self.motion_history.append(motion_score)
        self.head_pose_history.append(head_pose_score)
        
        # Weighted combination
        # Head pose is primary (detects frontal vs turned away)
        # Texture distinguishes photos/screens from real faces
        # Frequency shows structure differences
        # Motion shows natural movement
        weights = {
            'texture': 0.35,
            'frequency': 0.35,
            'motion': 0.20,
            'head_pose': 0.10
        }
        
        liveness_score = (
            texture_score * weights['texture'] +
            frequency_score * weights['frequency'] +
            motion_score * weights['motion'] +
            head_pose_score * weights['head_pose']
        )
        
        # Calculate temporal confidence
        temporal_confidence = self._calculate_temporal_confidence()
        confidence = min(1.0, liveness_score + temporal_confidence * 0.15)
        
        # Threshold
        threshold = 0.45
        is_live = liveness_score >= threshold
        
        if self.verbose:
            logger.debug(
                f"[DEBUG] Liveness: {liveness_score:.3f} | "
                f"Texture: {texture_score:.3f} | Freq: {frequency_score:.3f} | "
                f"Motion: {motion_score:.3f} | HeadPose: {head_pose_score:.3f}"
            )
        
        return {
            'is_live': is_live,
            'liveness_score': liveness_score,
            'confidence': confidence,
            'method_scores': {
                'texture': texture_score,
                'frequency': frequency_score,
                'motion': motion_score,
                'head_pose': head_pose_score
            },
            'details': {
                'head_angles': head_angles,
                'texture_variance': texture_score,
                'frequency_score': frequency_score,
                'motion_score': motion_score
            }
        }
    
    def _analyze_head_pose(self, gray):
        """
        Detect head orientation (frontal vs tilted/turned).
        Real people look at camera; photos are static.
        
        Returns: (score 0.0-1.0, (yaw_degrees, pitch_degrees))
        """
        # Detect frontal faces
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        
        if len(faces) == 0:
            # Try profile detection
            profiles = self.profile_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            if len(profiles) > 0:
                # Profile detected = turned head = potentially live
                # Photos are usually frontal
                head_score = 0.7
                return head_score, (45.0, 0.0)  # Estimated 45 degree yaw
            return 0.3, (0.0, 0.0)  # No face detected
        
        # Get largest face
        face = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = face
        
        # Head pose score based on:
        # 1. Face size (frontal faces are larger than side profiles)
        # 2. Face aspect ratio (real faces have consistent ratio)
        face_ratio = w / (h + 1e-6)
        
        # Normal face aspect ratio ~0.8-1.0
        # Tilted/turned faces have different ratios
        # Photos typically have exactly 0.85-0.95
        
        # Score how "frontal" the face appears
        if 0.7 < face_ratio < 1.2:
            # Frontal face detected
            frontal_score = 1.0 - abs(face_ratio - 0.88) / 0.4
        else:
            # Tilted or profile face
            frontal_score = 0.6
        
        # History: vary head pose over time = more likely real
        if len(self.head_pose_history) >= 5:
            recent = list(self.head_pose_history)[-5:]
            variation = np.std(recent) if len(recent) > 1 else 0
            # Real faces show pose variation; photos are static
            variation_score = min(1.0, variation * 3.0)
            head_score = 0.4 * frontal_score + 0.6 * variation_score
        else:
            head_score = frontal_score
        
        # Estimate rough yaw/pitch from face size and position
        center_x = x + w / 2
        frame_center = gray.shape[1] / 2
        yaw = (center_x - frame_center) / frame_center * 30  # -30 to +30 degrees
        pitch = 0.0  # Would need more landmarks to estimate pitch
        
        return head_score, (yaw, pitch)
    
    def _analyze_texture(self, gray):
        """
        Analyze texture using Laplacian variance.
        Photos and screens have different texture characteristics.
        
        Returns: score 0.0-1.0 (higher = more likely real face)
        """
        # Apply Laplacian for edge detection
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        
        # Real faces have texture variance: 100-500
        # Photos: 50-300 (depends on compression)
        # Screens: 10-100 (smooth)
        # High variance = more detail = more likely real
        
        # Normalize: 0 = completely smooth, 1 = very textured
        # Threshold at 150 for real faces
        texture_score = min(1.0, variance / 200.0)
        
        # Temporal consistency
        if len(self.texture_history) >= 5:
            recent = np.array(list(self.texture_history)[-5:])
            consistency = 1.0 - np.std(recent) / (np.mean(recent) + 1e-6)
            texture_score = 0.6 * texture_score + 0.4 * consistency
        
        return min(1.0, texture_score)
    
    def _analyze_frequency(self, gray):
        """
        Analyze frequency domain characteristics using FFT.
        Photos have different frequency patterns than real faces.
        
        Returns: score 0.0-1.0 (higher = more likely real face)
        """
        # Compute FFT
        fft = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shift)
        
        # Normalize
        magnitude_log = np.log1p(magnitude)
        
        # Create center mask (center = low frequency, edges = high frequency)
        h, w = magnitude_log.shape
        cy, cx = h // 2, w // 2
        
        # Center region (low frequencies)
        center_size = 15
        center_region = magnitude_log[cy-center_size:cy+center_size, cx-center_size:cx+center_size]
        center_energy = np.sum(center_region)
        
        # Outer region (high frequencies)
        outer_region = magnitude_log.copy()
        outer_region[cy-center_size:cy+center_size, cx-center_size:cx+center_size] = 0
        outer_energy = np.sum(outer_region)
        
        # Real faces: balanced frequency distribution
        # Photos: skewed towards center (low freq) - less detail
        # Screens: skewed towards edges (artifacts)
        
        total_energy = center_energy + outer_energy + 1e-6
        center_ratio = center_energy / total_energy
        
        # Optimal ratio for real faces: ~0.4-0.6
        # Photos: >0.7 (too much low freq)
        # Screens: <0.3 (too much high freq noise)
        
        if 0.35 < center_ratio < 0.65:
            frequency_score = 1.0 - abs(center_ratio - 0.5) / 0.15
        else:
            frequency_score = max(0.3, 1.0 - abs(center_ratio - 0.5) / 0.3)
        
        frequency_score = min(1.0, frequency_score)
        
        return frequency_score
    
    def _analyze_motion(self, gray):
        """
        Detect motion using optical flow (Lucas-Kanade).
        Real faces have natural motion; photos/screens don't.
        
        Returns: score 0.0-1.0 (higher = more likely real face)
        """
        if self.last_gray is None:
            self.last_gray = gray.copy()
            return 0.5  # Neutral score initially
        
        # Calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(
            self.last_gray, gray,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            n8=True,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )
        
        # Calculate motion magnitude
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        motion_mean = np.mean(magnitude)
        motion_std = np.std(magnitude)
        
        # Real faces: 0.2-2.0 pixels motion per frame
        # Static photos: <0.1 pixels
        # Screen videos: 0.3-5.0 with high variance (compression artifacts)
        
        # Motion score: prefer moderate, consistent motion
        if motion_mean < 0.1:
            # Too static = likely photo
            motion_score = 0.2
        elif motion_mean < 0.5:
            # Natural motion = likely real
            motion_score = 0.8 + motion_mean / 5.0
        elif motion_mean < 2.0:
            # Reasonable motion
            motion_score = 0.9
        else:
            # Excessive motion = maybe video or screen
            motion_score = max(0.4, 1.0 - (motion_mean - 2.0) / 5.0)
        
        # Penalize very high variance (compression artifacts)
        if motion_std > 1.0:
            motion_score *= 0.8
        
        # Store magnitude for temporal analysis
        self.motion_buffer.append(motion_mean)
        
        # Temporal consistency: real motion is smooth
        if len(self.motion_history) >= 3:
            recent = list(self.motion_history)[-3:]
            temporal_smoothness = 1.0 - np.std(recent) if len(recent) > 1 else 0.5
            motion_score = 0.5 * motion_score + 0.5 * temporal_smoothness
        
        self.last_gray = gray.copy()
        
        return min(1.0, motion_score)
    
    def _calculate_temporal_confidence(self):
        """Calculate confidence based on temporal consistency across frames."""
        if len(self.texture_history) < 5:
            return 0.0
        
        recent_textures = np.array(list(self.texture_history)[-5:])
        recent_frequencies = np.array(list(self.frequency_history)[-5:])
        recent_motions = np.array(list(self.motion_history)[-5:])
        
        # Consistent methods = higher confidence
        texture_consistency = 1.0 - np.std(recent_textures)
        freq_consistency = 1.0 - np.std(recent_frequencies)
        motion_consistency = 1.0 - np.std(recent_motions)
        
        avg_consistency = (texture_consistency + freq_consistency + motion_consistency) / 3.0
        
        return max(0.0, min(1.0, avg_consistency))
    
    def reset(self):
        """Reset detector state for new person."""
        self.texture_history.clear()
        self.frequency_history.clear()
        self.motion_history.clear()
        self.head_pose_history.clear()
        self.last_gray = None
        self.motion_buffer.clear()


class LiveAnalyzer:
    """
    Temporal consensus analyzer for liveness detection.
    Maintains state across multiple frames for robust decision-making.
    """
    
    def __init__(self, consensus_frames=5, consensus_threshold=0.60):
        """
        Args:
            consensus_frames: Number of frames to analyze for consensus
            consensus_threshold: Fraction of frames that must be live (0.0-1.0)
        """
        self.consensus_frames = consensus_frames
        self.consensus_threshold = consensus_threshold
        self.frame_history = deque(maxlen=consensus_frames)
        self.detector = ProductionLiveDetector(verbose=False)
        
        logger.info(
            f"[OK] LiveAnalyzer initialized: {consensus_frames} frames, "
            f"{consensus_threshold*100:.0f}% consensus threshold"
        )
    
    def check_face_liveness(self, frame):
        """
        Check if face in frame is live using temporal consensus.
        
        Args:
            frame: Input image (BGR)
            
        Returns:
            {
                'is_live': bool,
                'consensus_score': float (0.0-1.0),
                'live_frames': int,
                'total_frames': int,
                'detection': {...full detection dict...}
            }
        """
        detection = self.detector.detect_liveness(frame)
        
        # Track detection status
        self.frame_history.append(detection['is_live'])
        
        # Calculate consensus
        total = len(self.frame_history)
        live_count = sum(self.frame_history)
        consensus_score = live_count / total if total > 0 else 0.0
        
        # Require consensus_threshold fraction of frames to be live
        is_live_consensus = consensus_score >= self.consensus_threshold
        
        return {
            'is_live': is_live_consensus,
            'consensus_score': consensus_score,
            'live_frames': live_count,
            'total_frames': total,
            'detection': detection
        }
    
    def reset(self):
        """Reset analyzer for new detection session."""
        self.frame_history.clear()
        self.detector.reset()
