# ---
# CampusAI Suite Attendance Fusion Engine
# Owner: Member 2 (Computer vision engineer) & Member 1 (ML Lead)
# ---

import logging
import numpy as np
from typing import List, Dict, Any, Tuple
from backend.core.face_engine import detect_faces, match_face_to_db
from backend.core.config import settings

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("attend_fusion")

def fuse_three_photos(
    front_left_img: np.ndarray,
    front_center_img: np.ndarray,
    front_right_img: np.ndarray,
    enrolled_students: List[Any]
) -> List[Dict[str, Any]]:
    """
    Fuses face detections and database matchings across 3 different camera angles.
    Implements the Union Logic:
      - Present: matched with confidence >= 0.60 in ANY photo.
      - Low Confidence (Review): matched with confidence between 0.45 and 0.60 in ANY photo
        (and not >= 0.60 in any other photo).
      - Absent: not matched, or matched with confidence < 0.45 in all photos.
      
    Args:
        front_left_img: np.ndarray in BGR format
        front_center_img: np.ndarray in BGR format
        front_right_img: np.ndarray in BGR format
        enrolled_students: List of Student ORM objects from the SQLite DB
        
    Returns:
        List of dicts representing attendance status for ALL enrolled students:
        [
            {
                "student_id": "S001",
                "name": "Arif",
                "status": "present" | "low_confidence" | "absent",
                "confidence": 0.8423,  # maximum confidence found across the 3 photos
                "matched_photo": "left" | "center" | "right" | "none"
            },
            ...
        ]
    """
    logger.info(f"Fusing attendance photos for {len(enrolled_students)} enrolled students...")
    
    # 1. Process each of the three images to get detected faces
    photos = {
        "left": front_left_img,
        "center": front_center_img,
        "right": front_right_img
    }
    
    # Track the highest match confidence and which photo it came from for each student_id
    student_matches: Dict[str, Tuple[float, str]] = {
        student.student_id: (0.0, "none") for student in enrolled_students
    }
    
    for angle, img in photos.items():
        if img is None:
            logger.warning(f"Image for angle '{angle}' is None. Skipping.")
            continue
            
        # Detect faces in this photo
        detected = detect_faces(img)
        logger.info(f"Detected {len(detected)} face(s) in the '{angle}' photo.")
        
        # Match each detected face against the database of enrolled students
        for face in detected:
            emb = face["embedding"]
            student_id, confidence = match_face_to_db(emb, enrolled_students)
            
            if student_id is not None:
                # If this confidence is higher than what we recorded previously, update it
                current_best_conf, _ = student_matches[student_id]
                if confidence > current_best_conf:
                    student_matches[student_id] = (confidence, angle)
                    
    # 2. Compile results for ALL enrolled students based on union thresholds
    attendance_results = []
    
    for student in enrolled_students:
        confidence, angle = student_matches[student.student_id]
        
        # Apply the Union Threshold logic
        if confidence >= settings.ATTENDANCE_PRESENT_CONFIDENCE:
            status = "present"
        elif confidence >= settings.ATTENDANCE_LOW_CONFIDENCE:
            status = "low_confidence"
        else:
            status = "absent"
            
        attendance_results.append({
            "student_id": student.student_id,
            "name": student.name,
            "status": status,
            "confidence": float(round(confidence, 4)),
            "matched_photo": angle
        })
        
    logger.info(f"Fusion complete. Results compiled for {len(attendance_results)} students.")
    return attendance_results
