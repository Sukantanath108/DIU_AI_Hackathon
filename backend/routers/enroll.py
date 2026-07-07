# ---
# CampusAI Suite Student Enrollment API Router
# Owner: Member 3 (Backend and database engineer)
# ---

import os
import cv2
import numpy as np
import logging
from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException
from sqlalchemy.orm import Session
from backend.database import get_db
from backend.models import Student
from backend.core.face_engine import detect_faces
from backend.core.config import settings

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("enroll_router")

router = APIRouter(prefix="/enroll", tags=["Enrollment"])

@router.post("")
async def enroll_student(
    student_id: str = Form(...),
    name: str = Form(...),
    section: str = Form(...),
    department: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Enrolls a student with a single ID-card-quality photo.
    Extracts face embedding and saves the record in SQLite database.
    """
    logger.info(f"Enrolling student {student_id} — {name}...")
    
    try:
        # Read file bytes
        file_bytes = await file.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file format.")
            
        # Detect faces and extract embedding
        faces = detect_faces(img)
        
        if not faces:
            return {
                "status": "error",
                "message": "No face detected in the uploaded photo. Please try again with a clearer image."
            }
            
        # Get the largest face
        largest_face = max(faces, key=lambda f: (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]))
        embedding = largest_face["embedding"]
        
        # Save photo locally
        folder_name = f"{student_id}_{name.replace(' ', '_')}"
        student_folder = settings.DATA_DIR / "students" / folder_name
        student_folder.mkdir(parents=True, exist_ok=True)
        
        photo_filename = f"enrolled_{student_id}.jpg"
        photo_path = student_folder / photo_filename
        cv2.imwrite(str(photo_path), img)
        
        # Relative path to store in database
        relative_path = str(photo_path.relative_to(settings.BASE_DIR))
        
        # Check if student already exists in DB
        db_student = db.query(Student).filter(Student.student_id == student_id).first()
        
        if db_student:
            # Update existing
            db_student.name = name
            db_student.section = section
            db_student.department = department
            db_student.set_embedding(embedding)
            db_student.photo_path = relative_path
            logger.info(f"Updated student {student_id} in database.")
        else:
            # Insert new student
            db_student = Student(
                student_id=student_id,
                name=name,
                section=section,
                department=department,
                photo_path=relative_path
            )
            db_student.set_embedding(embedding)
            db.add(db_student)
            logger.info(f"Enrolled new student {student_id} into database.")
            
        db.commit()
        db.refresh(db_student)
        
        return {
            "status": "ok",
            "data": {
                "id": db_student.id,
                "student_id": db_student.student_id,
                "name": db_student.name,
                "section": db_student.section,
                "department": db_student.department,
                "photo_path": db_student.photo_path
            },
            "message": f"Student {name} ({student_id}) enrolled successfully."
        }
        
    except HTTPException as http_ex:
        raise http_ex
    except Exception as ex:
        logger.error(f"Error during student enrollment: {ex}")
        db.rollback()
        return {
            "status": "error",
            "message": f"An internal server error occurred: {str(ex)}"
        }
