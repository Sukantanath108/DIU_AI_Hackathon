# ---
# CampusAI Suite SmartAttend Attendance Router
# Owner: Member 3 (Backend Engineer) & Member 4 (Frontend Integration)
# ---

import io
import csv
import cv2
import numpy as np
import logging
from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from typing import List, Optional
from pydantic import BaseModel

from backend.database import get_db
from backend.models import Student, AttendanceSession, AttendanceRecord
from backend.core.attend_fusion import fuse_three_photos
from backend.utils.report import generate_attendance_pdf
from backend.core.config import settings

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("attend_router")

router = APIRouter(prefix="/attend", tags=["SmartAttend AI"])

# Pydantic Schemas
class SessionCreate(BaseModel):
    section: str
    subject: str
    teacher_id: str

class OverrideRequest(BaseModel):
    session_id: int
    student_id: str
    status: str  # 'present' or 'absent' or 'low_confidence'

@router.post("/session")
def create_session(payload: SessionCreate, db: Session = Depends(get_db)):
    """
    Creates a new attendance session for a class section and subject.
    """
    try:
        session = AttendanceSession(
            section=payload.section,
            subject=payload.subject,
            teacher_id=payload.teacher_id
        )
        db.add(session)
        db.commit()
        db.refresh(session)
        
        return {
            "status": "ok",
            "data": {
                "session_id": session.id,
                "section": session.section,
                "subject": session.subject,
                "teacher_id": session.teacher_id,
                "created_at": session.created_at.isoformat()
            },
            "message": "Attendance session created successfully."
        }
    except Exception as ex:
        logger.error(f"Error creating attendance session: {ex}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(ex)}")

@router.post("/photos")
async def upload_photos(
    session_id: int = Form(...),
    front_left: UploadFile = File(...),
    front_center: UploadFile = File(...),
    front_right: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Uploads 3 photos (front-left, front-center, front-right) and processes the multi-angle union logic.
    Marks students present or flags low-confidence faces, saving results to SQLite.
    """
    logger.info(f"Processing attendance photos for session #{session_id}...")
    
    # 1. Fetch the attendance session
    session = db.query(AttendanceSession).filter(AttendanceSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Attendance session not found.")
        
    try:
        # 2. Decode the uploaded files into OpenCV images
        left_bytes = await front_left.read()
        center_bytes = await front_center.read()
        right_bytes = await front_right.read()
        
        left_img = cv2.imdecode(np.frombuffer(left_bytes, np.uint8), cv2.IMREAD_COLOR)
        center_img = cv2.imdecode(np.frombuffer(center_bytes, np.uint8), cv2.IMREAD_COLOR)
        right_img = cv2.imdecode(np.frombuffer(right_bytes, np.uint8), cv2.IMREAD_COLOR)
        
        if left_img is None or center_img is None or right_img is None:
            raise HTTPException(status_code=400, detail="One or more uploaded files are not valid images.")
            
        # 3. Retrieve students enrolled in this session's section
        students = db.query(Student).filter(Student.section == session.section).all()
        if not students:
            return {
                "status": "error",
                "message": f"No students enrolled in section '{session.section}' yet. Please enroll students first."
            }
            
        # 4. Run the attendance fusion algorithm (Union Logic)
        fusion_results = fuse_three_photos(left_img, center_img, right_img, students)
        
        # 5. Save the records in SQLite database (clear existing to allow re-runs)
        db.query(AttendanceRecord).filter(AttendanceRecord.session_id == session_id).delete()
        
        db_records = []
        for res in fusion_results:
            record = AttendanceRecord(
                session_id=session_id,
                student_id=res["student_id"],
                status=res["status"],
                confidence=res["confidence"],
                overridden=0
            )
            db.add(record)
            db_records.append(record)
            
        db.commit()
        
        # 6. Format the API response
        formatted_records = []
        for r in fusion_results:
            formatted_records.append({
                "student_id": r["student_id"],
                "name": r["name"],
                "status": r["status"],
                "confidence": float(round(r["confidence"], 4)),
                "overridden": 0,
                "matched_photo": r["matched_photo"]
            })
            
        return {
            "status": "ok",
            "data": {
                "session_id": session_id,
                "records": formatted_records
            },
            "message": "Multi-angle photo fusion complete. Attendance records calculated and stored."
        }
        
    except HTTPException as http_ex:
        raise http_ex
    except Exception as ex:
        logger.error(f"Error calculating attendance via photo upload: {ex}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(ex)}")

@router.patch("/override")
def override_attendance(payload: OverrideRequest, db: Session = Depends(get_db)):
    """
    Applies manual teacher override/correction to a student's attendance record.
    """
    try:
        # Find record
        record = db.query(AttendanceRecord).filter(
            AttendanceRecord.session_id == payload.session_id,
            AttendanceRecord.student_id == payload.student_id
        ).first()
        
        if not record:
            raise HTTPException(status_code=404, detail="Attendance record not found.")
            
        if payload.status not in ["present", "absent", "low_confidence"]:
            raise HTTPException(status_code=400, detail="Invalid status. Must be 'present', 'absent', or 'low_confidence'.")
            
        # Update record
        record.status = payload.status
        record.overridden = 1
        db.commit()
        
        return {
            "status": "ok",
            "message": f"Successfully updated attendance status for student {payload.student_id} to '{payload.status}'."
        }
    except HTTPException as http_ex:
        raise http_ex
    except Exception as ex:
        logger.error(f"Error applying override: {ex}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(ex)}")

@router.get("/export/{session_id}")
def export_attendance(
    session_id: int,
    format: str = Query("csv", regex="^(csv|pdf)$"),
    db: Session = Depends(get_db)
):
    """
    Exports a session's attendance list as a CSV file or professional PDF.
    """
    session = db.query(AttendanceSession).filter(AttendanceSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Attendance session not found.")
        
    records = db.query(AttendanceRecord).filter(AttendanceRecord.session_id == session_id).all()
    if not records:
        raise HTTPException(status_code=404, detail="No attendance records found for this session.")
        
    # Get students' names
    records_data = []
    for r in records:
        student = db.query(Student).filter(Student.student_id == r.student_id).first()
        name = student.name if student else "Unknown"
        records_data.append({
            "student_id": r.student_id,
            "name": name,
            "status": r.status,
            "confidence": r.confidence,
            "overridden": r.overridden
        })
        
    if format == "csv":
        # Create CSV memory buffer
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write headers
        writer.writerow(["Session ID", "Subject", "Section", "Teacher ID", "Date"])
        writer.writerow([session.id, session.subject, session.section, session.teacher_id, session.created_at.strftime('%Y-%m-%d %H:%M:%S')])
        writer.writerow([])
        writer.writerow(["Student ID", "Student Name", "Status", "Confidence Match", "Manually Corrected"])
        
        # Write rows
        for rd in records_data:
            writer.writerow([
                rd["student_id"],
                rd["name"],
                rd["status"].upper(),
                f"{rd['confidence']:.4f}",
                "YES" if rd["overridden"] == 1 else "NO"
            ])
            
        output.seek(0)
        
        filename = f"attendance_session_{session_id}_{session.section}_{session.subject.replace(' ', '_')}.csv"
        return StreamingResponse(
            io.BytesIO(output.getvalue().encode('utf-8')),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    elif format == "pdf":
        # Call ReportLab generator
        pdf_bytes = generate_attendance_pdf(
            session_id=session.id,
            section=session.section,
            subject=session.subject,
            teacher_id=session.teacher_id,
            created_at=session.created_at,
            records=records_data
        )
        
        filename = f"attendance_session_{session_id}_{session.section}_{session.subject.replace(' ', '_')}.pdf"
        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
