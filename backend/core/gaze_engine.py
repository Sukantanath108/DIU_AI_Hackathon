# ---
# CampusAI Suite Gaze and Head Pose Engine
# Owner: Member 2 (Computer vision engineer)
#
# Updated 2026-06-04: Migrated from legacy mp.solutions.face_mesh to
# the new mp.tasks.vision.FaceLandmarker API (mediapipe >=0.10.x)
# ---

import time
import logging
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List
from backend.core.config import settings

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gaze_engine")

_face_landmarker = None
_using_fallback = False

# Path to face_landmarker.task model file
_MODEL_PATH = str(Path(__file__).resolve().parent.parent.parent / "ml" / "models" / "face_landmarker.task")


def init_gaze_engine() -> None:
    """
    Initializes MediaPipe FaceLandmarker using the new Tasks API (mediapipe >=0.10.x).
    If MediaPipe fails to load or fallback is active, gracefully falls back to synthetic estimation.
    """
    global _face_landmarker, _using_fallback
    
    def _try_init():
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        
        model_path = _MODEL_PATH
        if not Path(model_path).exists():
            raise FileNotFoundError(f"FaceLandmarker model not found at {model_path}")
        
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=5,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            running_mode=vision.RunningMode.IMAGE
        )
        landmarker = vision.FaceLandmarker.create_from_options(options)
        return landmarker
    
    if not settings.USE_AI_FALLBACK:
        try:
            logger.info("Initializing MediaPipe FaceLandmarker (Tasks API)...")
            _face_landmarker = _try_init()
            _using_fallback = False
            logger.info("MediaPipe FaceLandmarker successfully loaded.")
        except Exception as err:
            logger.error(f"Failed to load MediaPipe: {err}. Fallback disabled, raising exception.")
            raise err
    else:
        try:
            logger.info("Attempting to load MediaPipe FaceLandmarker (with fallback enabled)...")
            _face_landmarker = _try_init()
            _using_fallback = False
            logger.info("MediaPipe FaceLandmarker successfully loaded.")
        except Exception as err:
            logger.warning(f"Could not initialize MediaPipe ({err}). Falling back to Synthetic Gaze Engine.")
            _using_fallback = True

# Initialize the gaze engine
try:
    init_gaze_engine()
except Exception as e:
    logger.error(f"Initial Gaze engine load error: {e}. Fallback will be used if permitted.")
    _using_fallback = True

def is_using_fallback() -> bool:
    """
    Returns True if the gaze engine is running in fallback mode.
    """
    return _using_fallback

def estimate_head_pose(
    frame: np.ndarray,
    mock_gaze_direction: str = "center"
) -> List[Dict[str, Any]]:
    """
    Estimates the head pose (yaw and pitch in degrees) for each detected face in the frame.
    
    Args:
        frame: np.ndarray, current video frame in BGR format.
        mock_gaze_direction: str, manual trigger in fallback mode ("left", "right", "up", "down", "center").
        
    Returns:
        List of dicts representing detected face poses:
        [
            {
                "bbox": [x1, y1, x2, y2],
                "yaw": 12.4,    # in degrees, negative is looking left, positive right
                "pitch": -5.2,   # in degrees, negative is looking down, positive up
                "roll": 0.0      # standard roll (optional)
            },
            ...
        ]
    """
    global _face_landmarker, _using_fallback
    
    if not _using_fallback and _face_landmarker is not None:
        try:
            import mediapipe as mp
            
            h, w = frame.shape[:2]
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # Run detection
            result = _face_landmarker.detect(mp_image)
            
            poses = []
            if result.face_landmarks:
                for face_landmarks in result.face_landmarks:
                    # Extract key landmarks for geometric head pose estimation
                    # Landmark indices (same as old FaceMesh):
                    # 1: Nose tip
                    # 33: Left eye inner corner
                    # 263: Right eye inner corner
                    # 152: Chin
                    
                    nose = np.array([face_landmarks[1].x * w, face_landmarks[1].y * h, face_landmarks[1].z * w])
                    left_eye = np.array([face_landmarks[33].x * w, face_landmarks[33].y * h, face_landmarks[33].z * w])
                    right_eye = np.array([face_landmarks[263].x * w, face_landmarks[263].y * h, face_landmarks[263].z * w])
                    chin = np.array([face_landmarks[152].x * w, face_landmarks[152].y * h, face_landmarks[152].z * w])
                    
                    # Calculate eye center and horizontal/vertical scales
                    eye_center = (left_eye + right_eye) / 2.0
                    eye_dist = np.linalg.norm(right_eye - left_eye)
                    
                    if eye_dist < 1.0:
                        continue  # Skip degenerate faces
                    
                    # Geometric approximation of Yaw (horizontal deviation)
                    dx = nose[0] - eye_center[0]
                    yaw = (dx / eye_dist) * 60.0
                    
                    # Geometric approximation of Pitch (vertical deviation)
                    dy = nose[1] - (eye_center[1] + chin[1]) / 2.0
                    pitch = (dy / eye_dist) * 60.0
                    
                    # Compute bounding box from all landmarks
                    xs = [lm.x * w for lm in face_landmarks]
                    ys = [lm.y * h for lm in face_landmarks]
                    bbox = [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]
                    
                    poses.append({
                        "bbox": bbox,
                        "yaw": float(round(yaw, 4)),
                        "pitch": float(round(pitch, 4)),
                        "roll": 0.0
                    })
                return poses if poses else []
            return []
        except Exception as err:
            logger.warning(f"MediaPipe FaceLandmarker inference failed: {err}. Routing to Fallback.")
            
    # --- SYNTHETIC FALLBACK GAZE ENGINE ---
    # Simulates head pose deviation based on mock_gaze_direction
    h, w = frame.shape[:2]
    bbox = [int(w * 0.3), int(h * 0.2), int(w * 0.7), int(h * 0.8)]
    
    yaw, pitch = 0.0, 0.0
    if mock_gaze_direction == "left":
        yaw = -30.0
    elif mock_gaze_direction == "right":
        yaw = 30.0
    elif mock_gaze_direction == "up":
        pitch = 25.0
    elif mock_gaze_direction == "down":
        pitch = -25.0
        
    return [{
        "bbox": bbox,
        "yaw": yaw,
        "pitch": pitch,
        "roll": 0.0
    }]


class StudentAnomalyTracker:
    """
    Stateful anomaly tracker for a student during an active ExamShield session.
    Accumulates time-based gaze and head pose deviations, manages multi-face occurrences,
    tracks the overall anomaly score, and triggers discrete event alerts.
    """
    def __init__(self, student_id: str):
        self.student_id: str = student_id
        self.cumulative_score: int = 0
        
        # Deviation timers (timestamps of when deviation started)
        self.gaze_deviation_start: Optional[float] = None
        self.head_deviation_start: Optional[float] = None
        
        # Event trigger locks (to prevent multiple event records for a single continuous deviation)
        self.gaze_event_fired: bool = False
        self.head_event_fired: bool = False
        
        # Identity verification flag
        self.identity_verified: bool = True  # Verified by default, can be set False

    def update(
        self,
        yaw: float,
        pitch: float,
        multi_face_detected: bool,
        current_time: float = None
    ) -> List[Dict[str, Any]]:
        """
        Updates the tracker status with the latest yaw, pitch, and multi-face readings.
        Returns a list of newly triggered anomaly events.
        
        Args:
            yaw: float, current head yaw angle.
            pitch: float, current head pitch angle.
            multi_face_detected: bool, whether more than one face is present near the seat.
            current_time: float, current timestamp (defaults to time.time()).
            
        Returns:
            List of triggered event dicts, empty if no new events are triggered.
        """
        if current_time is None:
            current_time = time.time()
            
        triggered_events = []
        
        # 1. Gaze & Head Pose Deviation Logic
        yaw_exceeded = abs(yaw) > settings.GAZE_MAX_YAW
        pitch_exceeded = abs(pitch) > settings.GAZE_MAX_PITCH
        deviation_active = yaw_exceeded or pitch_exceeded
        
        if deviation_active:
            # --- GAZE DEVIATION TIMER ---
            if self.gaze_deviation_start is None:
                self.gaze_deviation_start = current_time
            else:
                elapsed_gaze = current_time - self.gaze_deviation_start
                if elapsed_gaze > settings.GAZE_DEVIATION_TIME_LIMIT and not self.gaze_event_fired:
                    # Trigger Gaze Anomaly Event
                    self.cumulative_score += settings.SCORE_GAZE_DEVIATION
                    self.gaze_event_fired = True
                    triggered_events.append({
                        "student_id": self.student_id,
                        "event_type": "gaze_deviation",
                        "score_delta": settings.SCORE_GAZE_DEVIATION,
                        "message": f"Gaze deviation detected: Yaw {round(yaw, 1)}deg, Pitch {round(pitch, 1)}deg for >3s"
                    })
                    
            # --- HEAD POSE DEVIATION TIMER ---
            if self.head_deviation_start is None:
                self.head_deviation_start = current_time
            else:
                elapsed_head = current_time - self.head_deviation_start
                if elapsed_head > settings.HEAD_POSE_DEVIATION_TIME_LIMIT and not self.head_event_fired:
                    # Trigger Head Pose Anomaly Event
                    self.cumulative_score += settings.SCORE_HEAD_POSE_DEVIATION
                    self.head_event_fired = True
                    triggered_events.append({
                        "student_id": self.student_id,
                        "event_type": "gaze_deviation",  # standard mapping or customized
                        "score_delta": settings.SCORE_HEAD_POSE_DEVIATION,
                        "message": f"Head pose deviation detected: Yaw {round(yaw, 1)}deg, Pitch {round(pitch, 1)}deg for >4s"
                    })
        else:
            # Reset timers and event locks once student looks back at center
            self.gaze_deviation_start = None
            self.head_deviation_start = None
            self.gaze_event_fired = False
            self.head_event_fired = False
            
        # 2. Multi-Face Detection Logic (Fires immediately upon detection)
        # NOTE: Multi-face detection contributes immediately. To avoid spamming,
        # we could implement a small cool-down or simple event filtering.
        if multi_face_detected:
            self.cumulative_score += settings.SCORE_MULTI_FACE_DETECTED
            triggered_events.append({
                "student_id": self.student_id,
                "event_type": "multi_face",
                "score_delta": settings.SCORE_MULTI_FACE_DETECTED,
                "message": "Multiple faces detected near the student seat"
            })
            
        return triggered_events
        
    def trigger_object_event(self, item_label: str, score_delta: int) -> Dict[str, Any]:
        """
        Manually injects an object detection event (e.g. phone, book, cheat sheet).
        Updates cumulative score and returns the formatted event.
        """
        self.cumulative_score += score_delta
        return {
            "student_id": self.student_id,
            "event_type": item_label,
            "score_delta": score_delta,
            "message": f"Prohibited object detected: {item_label.replace('_', ' ').title()}"
        }

    def trigger_unverified_identity(self) -> Dict[str, Any]:
        """
        Triggers unverified student identity event.
        """
        self.identity_verified = False
        self.cumulative_score += settings.SCORE_UNVERIFIED_IDENTITY
        return {
            "student_id": self.student_id,
            "event_type": "unverified",
            "score_delta": settings.SCORE_UNVERIFIED_IDENTITY,
            "message": "Identity could not be verified at the start of the exam session"
        }
        
    def get_risk_status(self) -> Tuple[str, str]:
        """
        Computes the visual risk band and color.
        Score bands:
          - 0-40: Green (Normal)
          - 41-80: Amber (Caution)
          - 81+: Red (High Risk, Audible Alert)
        """
        score = self.cumulative_score
        if score <= 40:
            return "green", "Normal"
        elif score <= 80:
            return "amber", "Caution"
        else:
            return "red", "High Risk"
