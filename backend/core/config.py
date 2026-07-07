# ---
# CampusAI Suite Configuration Module
# Owner: Member 3 (Backend and database engineer)
# ---

import os
from pathlib import Path
from pydantic_settings import BaseSettings

# Base directories
BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
DATA_DIR: Path = BASE_DIR / "data"
SCREENSHOTS_DIR: Path = BASE_DIR / "screenshots"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)

class Settings(BaseSettings):
    """
    Central settings class for CampusAI Suite backend.
    """
    BASE_DIR: Path = BASE_DIR
    DATA_DIR: Path = DATA_DIR
    SCREENSHOTS_DIR: Path = SCREENSHOTS_DIR
    PROJECT_NAME: str = "CampusAI Suite API"
    VERSION: str = "1.0.0"
    
    # Database
    DATABASE_URL: str = f"sqlite:///{DATA_DIR / 'students.db'}"
    
    # Face Engine Configuration
    FACE_MODEL_NAME: str = "buffalo_l"
    FACE_DETECTION_CONFIDENCE: float = 0.50

    # Face ROI Upscaling Tiers
    # Tier 1 — too small to attempt recognition (label "TOO FAR")
    FACE_MIN_WIDTH: int = 30
    # Tier 2 — marginal; crop, upscale 2-3x with INTER_CUBIC, then ArcFace
    # Tier 3 — face_width >= this value, run standard recognition
    FACE_RECOVERY_WIDTH: int = 60
    # Target pixel size for the upscaled ROI before ArcFace embedding
    FACE_UPSCALE_SIZE: int = 200

    # SmartAttend AI Thresholds
    ATTENDANCE_PRESENT_CONFIDENCE: float = 0.60
    ATTENDANCE_LOW_CONFIDENCE: float = 0.45
    
    # ExamShield AI Proctoring Thresholds
    PROCTOR_PHONE_CONFIDENCE: float = 0.65
    PROCTOR_CHEAT_SHEET_CONFIDENCE: float = 0.65
    PROCTOR_BOOK_CONFIDENCE: float = 0.65

    # Exam Entry Verification Threshold
    # Minimum cosine similarity (ArcFace) required to match an entry photo
    # against the enrolled student database. Below this, the person is
    # treated as UNKNOWN and a 'no_match' / 'unknown_person' result is returned.
    EXAM_ENTRY_MATCH_THRESHOLD: float = 0.45

    # Per-tier cosine-similarity thresholds for classroom recognition
    # (recognize_with_upscaling).  A face that scores BELOW its tier
    # threshold is rejected as UNKNOWN, even if it is the top match.
    # Competition rule: false negative > false positive — better to leave
    # a face as UNKNOWN than to mis-identify it as the wrong student.
    FACE_MATCH_THRESHOLD_GOOD: float = 0.45
    FACE_MATCH_THRESHOLD_MARGINAL: float = 0.55

    # Margin rule: the BEST candidate must beat the SECOND-BEST candidate
    # by at least this much in cosine similarity.  This prevents the matcher
    # from picking the "least-wrong" face when several students score close
    # together.  0.08 is a conservative gap — well-separated true matches
    # easily exceed it, while confusable near-twins stay below it.
    FACE_MATCH_MARGIN: float = 0.08
    
    # Anomaly Scores (from Competition Specifications)
    SCORE_UNVERIFIED_IDENTITY: int = 60
    SCORE_PHONE_DETECTED: int = 50
    SCORE_MULTI_FACE_DETECTED: int = 45
    SCORE_CHEAT_SHEET_DETECTED: int = 35
    SCORE_BOOK_DETECTED: int = 30
    SCORE_GAZE_DEVIATION: int = 20
    SCORE_HEAD_POSE_DEVIATION: int = 15
    
    # Gaze Deviation Limits (yaw in degrees, pitch in degrees, time in seconds)
    GAZE_MAX_YAW: float = 25.0
    GAZE_MAX_PITCH: float = 20.0
    GAZE_DEVIATION_TIME_LIMIT: float = 3.0
    HEAD_POSE_DEVIATION_TIME_LIMIT: float = 4.0
    
    # Fallback configuration
    # Set to True to allow the server to mock InsightFace, MediaPipe, and YOLO detections
    # if libraries or assets are missing, enabling seamless end-to-end local testing.
    USE_AI_FALLBACK: bool = True

settings: Settings = Settings()
