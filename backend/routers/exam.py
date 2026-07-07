# ---
# CampusAI Suite ExamShield Proctoring Router
# Owner: Member 3 (Backend Engineer) & Member 2 (Computer vision engineer)
# ---

import os
import json
import cv2
import numpy as np
import logging
from datetime import datetime
from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from typing import List, Optional
from pydantic import BaseModel
from io import BytesIO

from backend.database import get_db
from backend.models import Student, ExamSession, ExamEvent, ExamSeatAssignment, assign_camera_zone
from backend.utils.report import generate_exam_report_pdf
from backend.core.config import settings
from backend.core.face_engine import detect_faces, cosine_similarity, match_face_to_db

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("exam_router")

router = APIRouter(prefix="/exam", tags=["ExamShield AI"])


def _is_sentinel(student_id: str) -> bool:
    """True if the student_id is a UNKNOWN-* sentinel row (not a real candidate).

    These are auto-created on the fly when a face doesn't match anyone; they must
    be excluded from rosters, expected counts, and entry verification candidate
    lists to keep the UI honest.
    """
    if not student_id:
        return False
    return student_id.upper().startswith("UNKNOWN")


def _filter_real_students(students):
    """Return only students whose id does not look like a UNKNOWN-* sentinel."""
    return [s for s in students if not _is_sentinel(getattr(s, "student_id", ""))]

# Pydantic schemas
class ExamSessionCreate(BaseModel):
    course: str
    hall: str
    invigilator: str

@router.post("/session")
def start_exam_session(payload: ExamSessionCreate, db: Session = Depends(get_db)):
    """
    Starts a new proctoring exam session.
    Auto-initializes roster with ALL enrolled students as EXPECTED.
    """
    try:
        # Get all enrolled student IDs for the EXPECTED roster.
        # Strip out any UNKNOWN-* sentinel rows so they never pollute the
        # EXPECTED count shown to the invigilator.
        all_students = _filter_real_students(db.query(Student).all())
        all_ids = [s.student_id for s in all_students]

        session = ExamSession(
            course=payload.course,
            hall=payload.hall,
            invigilator=payload.invigilator,
            roster=json.dumps(all_ids),
            roster_name=f"{payload.course} Exam",
            verified_roster=json.dumps([])  # Empty — built via entry verification
        )
        db.add(session)
        db.commit()
        db.refresh(session)
        
        return {
            "status": "ok",
            "data": {
                "exam_session_id": session.id,
                "course": session.course,
                "hall": session.hall,
                "invigilator": session.invigilator,
                "started_at": session.started_at.isoformat(),
                "expected_count": len(all_ids)
            },
            "message": f"Exam session #{session.id} started. {len(all_ids)} students expected."
        }
    except Exception as ex:
        logger.error(f"Error starting exam session: {ex}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(ex)}")

@router.post("/event")
async def log_anomaly_event(
    exam_session_id: int = Form(...),
    student_id: str = Form(...),
    event_type: str = Form(...),  # 'phone', 'cheat_sheet', 'book', 'gaze_deviation', 'multi_face', 'unverified'
    screenshot: Optional[UploadFile] = File(None),
    db: Session = Depends(get_db)
):
    """
    Logs an anomaly or suspicious activity event for a student in an exam session.
    Saves flagged screenshot if provided.
    """
    logger.info(f"Logging anomaly {event_type} for student {student_id} in exam session #{exam_session_id}...")
    
    # Verify session
    session = db.query(ExamSession).filter(ExamSession.id == exam_session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Exam session not found.")
        
    # Verify student
    student = db.query(Student).filter(Student.student_id == student_id).first()
    if not student:
        raise HTTPException(status_code=404, detail="Student not found.")
        
    # Score deltas mapping
    score_mapping = {
        "phone": settings.SCORE_PHONE_DETECTED,
        "cheat_sheet": settings.SCORE_CHEAT_SHEET_DETECTED,
        "book": settings.SCORE_BOOK_DETECTED,
        "gaze_deviation": settings.SCORE_GAZE_DEVIATION,
        "multi_face": settings.SCORE_MULTI_FACE_DETECTED,
        "unverified": settings.SCORE_UNVERIFIED_IDENTITY,
        "unknown_person": 0  # Logged for evidence, no risk score impact
    }
    
    if event_type not in score_mapping:
        raise HTTPException(status_code=400, detail=f"Invalid event type: {event_type}")
        
    score_delta = score_mapping[event_type]
    
    screenshot_relative_path = None
    
    # Save screenshot if provided
    if screenshot:
        try:
            screenshot_bytes = await screenshot.read()
            nparr = np.frombuffer(screenshot_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is not None:
                # Save path: backend/utils/screenshots/{exam_id}/{student_id}_{timestamp}.jpg
                # We put screenshots in BASE_DIR / "screenshots" / {exam_id}
                session_screenshot_dir = settings.SCREENSHOTS_DIR / str(exam_session_id)
                session_screenshot_dir.mkdir(parents=True, exist_ok=True)
                
                timestamp_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                filename = f"{student_id}_{timestamp_str}.jpg"
                save_path = session_screenshot_dir / filename
                
                cv2.imwrite(str(save_path), img)
                screenshot_relative_path = str(save_path.relative_to(settings.BASE_DIR))
                logger.info(f"Saved proctoring screenshot to {screenshot_relative_path}")
        except Exception as ex:
            logger.error(f"Failed to process and save screenshot: {ex}")
            
    try:
        # Create event
        event = ExamEvent(
            exam_session_id=exam_session_id,
            student_id=student_id,
            event_type=event_type,
            score_delta=score_delta,
            screenshot_path=screenshot_relative_path
        )
        db.add(event)
        db.commit()
        db.refresh(event)
        
        return {
            "status": "ok",
            "data": {
                "event_id": event.id,
                "student_id": event.student_id,
                "event_type": event.event_type,
                "score_delta": event.score_delta,
                "screenshot_path": event.screenshot_path,
                "occurred_at": event.occurred_at.isoformat()
            },
            "message": f"Anomaly event '{event_type}' successfully logged."
        }
    except Exception as ex:
        logger.error(f"Error logging exam event: {ex}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(ex)}")


class RosterRequest(BaseModel):
    exam_session_id: int
    student_ids: List[str]  # e.g. ["S001", "S005", "S022"]
    roster_name: str = "Exam Candidates"


class UnknownPersonLogRequest(BaseModel):
    exam_session_id: int
    # Why the person could not be verified. One of:
    #   'below_threshold' — best cosine similarity < EXAM_ENTRY_MATCH_THRESHOLD
    #   'no_face'         — no face detected in the captured frame
    #   'multiple_faces'  — more than one face in the frame
    reason: str = "below_threshold"
    # Best similarity score achieved, if any (0.0–1.0). Optional.
    best_confidence: Optional[float] = None
    # Where in the frame the unknown face appeared. Optional.
    bbox: Optional[List[int]] = None
    # Snapshot for evidence. Optional, sent separately via multipart.
    note: Optional[str] = None


@router.post("/log_unknown_person")
async def log_unknown_person(
    payload: UnknownPersonLogRequest,
    screenshot: Optional[UploadFile] = File(None),
    db: Session = Depends(get_db)
):
    """
    Logs an UNKNOWN person attempt at exam entry.

    This creates an ExamEvent with event_type='unknown_person'. The score
    delta is always 0 (event is recorded for evidence, not for risk scoring).
    A 'placeholder' student record is reused if the unknown person looks
    similar to an enrolled one; otherwise the event is attributed to a
    sentinel student_id derived from the session.
    """
    logger.info(
        f"Logging UNKNOWN person for exam session #{payload.exam_session_id} "
        f"(reason={payload.reason}, best_confidence={payload.best_confidence})"
    )

    session = db.query(ExamSession).filter(ExamSession.id == payload.exam_session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Exam session not found.")

    # We need a non-null student_id for ExamEvent (FK to students.student_id).
    # Use a session-scoped sentinel: 'UNKNOWN-<session_id>'. If that student
    # doesn't exist, we create it on the fly with a zero embedding.
    sentinel_id = f"UNKNOWN-{payload.exam_session_id}"
    sentinel = db.query(Student).filter(Student.student_id == sentinel_id).first()
    if sentinel is None:
        zero_emb = np.zeros(512, dtype=np.float32)
        zero_emb[0] = 1.0
        sentinel = Student(
            student_id=sentinel_id,
            name=f"Unknown Person (Session {payload.exam_session_id})",
            section="N/A",
            department="N/A",
            embedding=zero_emb.tobytes(),
            photo_path=None
        )
        db.add(sentinel)
        try:
            db.commit()
            db.refresh(sentinel)
        except Exception as ex:
            db.rollback()
            logger.error(f"Failed to create sentinel student {sentinel_id}: {ex}")
            raise HTTPException(status_code=500, detail="Could not record unknown person.")

    # Save optional screenshot for evidence
    screenshot_relative_path = None
    if screenshot:
        try:
            screenshot_bytes = await screenshot.read()
            nparr = np.frombuffer(screenshot_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is not None:
                session_screenshot_dir = settings.SCREENSHOTS_DIR / str(payload.exam_session_id)
                session_screenshot_dir.mkdir(parents=True, exist_ok=True)
                timestamp_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                filename = f"UNKNOWN_{timestamp_str}.jpg"
                save_path = session_screenshot_dir / filename
                cv2.imwrite(str(save_path), img)
                screenshot_relative_path = str(save_path.relative_to(settings.BASE_DIR))
        except Exception as ex:
            logger.error(f"Failed to save unknown-person screenshot: {ex}")

    # Build a human-readable note describing the attempt
    note_parts = [f"reason={payload.reason}"]
    if payload.best_confidence is not None:
        note_parts.append(f"best_confidence={round(payload.best_confidence, 4)}")
    if payload.bbox is not None:
        note_parts.append(f"bbox={payload.bbox}")
    if payload.note:
        note_parts.append(payload.note)
    full_note = "; ".join(note_parts)

    event = ExamEvent(
        exam_session_id=payload.exam_session_id,
        student_id=sentinel_id,
        event_type="unknown_person",
        score_delta=0,  # logged for evidence, no risk impact
        screenshot_path=screenshot_relative_path
    )
    db.add(event)
    db.commit()
    db.refresh(event)

    return {
        "status": "ok",
        "data": {
            "event_id": event.id,
            "sentinel_student_id": sentinel_id,
            "event_type": "unknown_person",
            "score_delta": 0,
            "screenshot_path": screenshot_relative_path,
            "note": full_note,
            "occurred_at": event.occurred_at.isoformat()
        },
        "message": "Unknown person attempt logged for evidence."
    }


@router.post("/roster")
def set_exam_roster(payload: RosterRequest, db: Session = Depends(get_db)):
    """
    Sets the roster of exam candidates for a session.
    Only these students will appear in the Student Risk Grid.
    """
    session = db.query(ExamSession).filter(ExamSession.id == payload.exam_session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Exam session not found.")

    # Validate all student_ids exist
    valid_ids = []
    for sid in payload.student_ids:
        student = db.query(Student).filter(Student.student_id == sid).first()
        if student:
            valid_ids.append(sid)
        else:
            logger.warning(f"Roster: student {sid} not found in DB, skipping.")

    session.roster = json.dumps(valid_ids)
    session.roster_name = payload.roster_name
    db.commit()

    return {
        "status": "ok",
        "data": {
            "exam_session_id": session.id,
            "roster_name": payload.roster_name,
            "roster_count": len(valid_ids),
            "roster": valid_ids
        },
        "message": f"Roster set with {len(valid_ids)} candidates."
    }

@router.get("/{exam_id}/students")
def get_exam_students_status(exam_id: int, db: Session = Depends(get_db)):
    """
    Returns ALL students with their anomaly scores, risk status, and verified/unverified state.
    Verified students are fully scored; unverified students are shown grayed out.
    """
    session = db.query(ExamSession).filter(ExamSession.id == exam_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Exam session not found.")
        
    students = _filter_real_students(db.query(Student).all())

    # Determine which students are in the roster (EXPECTED)
    roster_ids = set()
    if session.roster:
        try:
            roster_ids = set(json.loads(session.roster))
            students = [s for s in students if s.student_id in roster_ids]
        except (json.JSONDecodeError, TypeError):
            pass

    # Determine which students are VERIFIED (entered the exam)
    verified_ids = set()
    if session.verified_roster:
        try:
            verified_ids = set(json.loads(session.verified_roster))
        except (json.JSONDecodeError, TypeError):
            pass

    events = db.query(ExamEvent).filter(ExamEvent.exam_session_id == exam_id).all()
    
    # Aggregate scores per student
    student_scores = {s.student_id: 0 for s in students}
    student_events_count = {s.student_id: 0 for s in students}
    student_timelines = {s.student_id: [] for s in students}
    
    for ev in events:
        if ev.student_id in student_scores:
            student_scores[ev.student_id] += ev.score_delta
            student_events_count[ev.student_id] += 1
            student_timelines[ev.student_id].append({
                "id": ev.id,
                "event_type": ev.event_type,
                "score_delta": ev.score_delta,
                "screenshot_path": ev.screenshot_path,
                "occurred_at": ev.occurred_at.isoformat()
            })
            
    # Format list
    students_list = []
    for s in students:
        score = student_scores[s.student_id]
        is_verified = s.student_id in verified_ids
        
        # Risk thresholds (only for verified students)
        if not is_verified:
            risk = "Not Verified"
            color = "gray"
        elif score <= 40:
            risk = "Normal"
            color = "green"
        elif score <= 80:
            risk = "Caution"
            color = "amber"
        else:
            risk = "High Risk"
            color = "red"
            
        students_list.append({
            "student_id": s.student_id,
            "name": s.name,
            "section": s.section,
            "department": s.department,
            "anomaly_score": score,
            "risk_status": risk,
            "color": color,
            "events_count": student_events_count[s.student_id],
            "timeline": sorted(student_timelines[s.student_id], key=lambda x: x["occurred_at"]),
            "verified": is_verified
        })
        
    # Sort: verified students first (by score desc), then unverified
    students_list.sort(key=lambda x: (not x["verified"], -x["anomaly_score"]))
    
    return {
        "status": "ok",
        "data": students_list,
        "summary": {
            "expected_count": len(students),
            "verified_count": len(verified_ids),
            "total_incidents": sum(student_events_count.values())
        }
    }


@router.get("/{exam_id}/entry_status")
def get_entry_status(exam_id: int, db: Session = Depends(get_db)):
    """
    Returns entry verification progress for the exam session.
    """
    session = db.query(ExamSession).filter(ExamSession.id == exam_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Exam session not found.")

    # Parse rosters
    expected_ids = []
    if session.roster:
        try:
            expected_ids = json.loads(session.roster)
        except (json.JSONDecodeError, TypeError):
            pass

    verified_ids = []
    if session.verified_roster:
        try:
            verified_ids = json.loads(session.verified_roster)
        except (json.JSONDecodeError, TypeError):
            pass

    not_verified = [sid for sid in expected_ids if sid not in verified_ids]

    # Get names for verified students (excluding UNKNOWN-* sentinels)
    all_students = _filter_real_students(db.query(Student).all())
    name_map = {s.student_id: s.name for s in all_students}

    verified_details = [
        {"student_id": sid, "name": name_map.get(sid, "Unknown")}
        for sid in verified_ids
    ]
    not_verified_details = [
        {"student_id": sid, "name": name_map.get(sid, "Unknown")}
        for sid in not_verified
    ]

    return {
        "status": "ok",
        "data": {
            "expected": expected_ids,
            "verified": verified_ids,
            "verified_details": verified_details,
            "not_verified": not_verified,
            "not_verified_details": not_verified_details,
            "expected_count": len(expected_ids),
            "verified_count": len(verified_ids)
        }
    }

@router.get("/report/{exam_id}")
def export_exam_report(exam_id: int, db: Session = Depends(get_db)):
    """
    Generates and returns the PDF integrity report for the exam session.
    """
    session = db.query(ExamSession).filter(ExamSession.id == exam_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Exam session not found.")
        
    students = db.query(Student).all()
    events = db.query(ExamEvent).filter(ExamEvent.exam_session_id == exam_id).all()
    
    # Calculate summaries
    student_scores = {s.student_id: 0 for s in students}
    for ev in events:
        if ev.student_id in student_scores:
            student_scores[ev.student_id] += ev.score_delta
            
    student_summaries = []
    for s in students:
        score = student_scores[s.student_id]
        if score <= 40:
            risk, color = "Normal", "green"
        elif score <= 80:
            risk, color = "Caution", "amber"
        else:
            risk, color = "High Risk", "red"
            
        student_summaries.append({
            "student_id": s.student_id,
            "name": s.name,
            "score": score,
            "risk": risk,
            "color": color
        })
        
    # Sort student summaries by score descending
    student_summaries.sort(key=lambda x: x["score"], reverse=True)
    
    # Prepare events details (incorporate student names)
    events_data = []
    student_map = {s.student_id: s.name for s in students}
    for ev in events:
        events_data.append({
            "student_id": ev.student_id,
            "name": student_map.get(ev.student_id, "Unknown"),
            "event_type": ev.event_type,
            "score_delta": ev.score_delta,
            "screenshot_path": ev.screenshot_path,
            "occurred_at": ev.occurred_at
        })
        
    # Generate PDF
    pdf_bytes = generate_exam_report_pdf(
        session_id=session.id,
        course=session.course,
        hall=session.hall,
        invigilator=session.invigilator,
        started_at=session.started_at,
        ended_at=session.ended_at,
        student_summaries=student_summaries,
        events=events_data
    )
    
    filename = f"exam_integrity_report_session_{exam_id}_{session.course.replace(' ', '_')}.pdf"
    return StreamingResponse(
        BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


@router.post("/verify_entry")
async def verify_student_entry(
    exam_session_id: int = Form(...),
    photo: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Verify a student at exam entry. Matches their face against the enrolled
    database using InsightFace. Automatically adds verified student to the
    session's verified_roster.

    Response status field:
      - "ok"            → matched an enrolled student, optionally added to verified_roster
      - "no_match"      → no enrolled student met EXAM_ENTRY_MATCH_THRESHOLD
                          (the person is treated as UNKNOWN)
      - "error"         → invalid input (bad image or no face detected)
    """
    logger.info(f"Verifying student entry for exam session #{exam_session_id}...")

    session = db.query(ExamSession).filter(ExamSession.id == exam_session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Exam session not found.")

    img_bytes = await photo.read()
    img_array = cv2.imdecode(
        np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR
    )

    if img_array is None:
        return {"status": "error", "message": "Invalid image file."}

    detections = detect_faces(img_array)
    if not detections:
        return {"status": "error", "message": "No face detected in entry photo."}

    best_face = max(detections, key=lambda d: d['det_score'])
    best_embedding = best_face['embedding']

    # Match against enrolled database, skipping UNKNOWN-* sentinel rows
    # (which hold zero-vectors and would otherwise always tie or distort results).
    # match_face_to_db applies the FACE_MATCH_MARGIN rule internally — the
    # top candidate must beat the second-best by at least the margin or the
    # match is rejected as UNKNOWN.  This enforces the competition rule
    # "false negative > false positive" at entry time.
    enrolled = _filter_real_students(db.query(Student).all())
    best_student_id, best_confidence = match_face_to_db(best_embedding, enrolled)
    best_name = None
    if best_student_id is not None:
        matched = next((s for s in enrolled if s.student_id == best_student_id), None)
        best_name = matched.name if matched else None

    match_threshold = settings.EXAM_ENTRY_MATCH_THRESHOLD
    if best_student_id is None or best_confidence < match_threshold:
        # Auto-log this as an 'unknown_person' evidence event.
        try:
            sentinel_id = f"UNKNOWN-{exam_session_id}"
            sentinel = db.query(Student).filter(Student.student_id == sentinel_id).first()
            if sentinel is None:
                zero_emb = np.zeros(512, dtype=np.float32)
                zero_emb[0] = 1.0
                sentinel = Student(
                    student_id=sentinel_id,
                    name=f"Unknown Person (Session {exam_session_id})",
                    section="N/A",
                    department="N/A",
                    embedding=zero_emb.tobytes(),
                    photo_path=None
                )
                db.add(sentinel)
                db.commit()
                db.refresh(sentinel)

            evidence_event = ExamEvent(
                exam_session_id=exam_session_id,
                student_id=sentinel_id,
                event_type="unknown_person",
                score_delta=0,
                screenshot_path=None
            )
            db.add(evidence_event)
            db.commit()
            db.refresh(evidence_event)
            logger.info(
                f"Unknown person evidence event #{evidence_event.id} created "
                f"(best_confidence={round(best_confidence, 4)}, threshold={match_threshold})"
            )
        except Exception as ex:
            # Evidence logging must never block the no_match response.
            db.rollback()
            logger.error(f"Failed to log unknown_person evidence: {ex}")

        return {
            "status": "no_match",
            "message": "Student not found in database.",
            "confidence": round(best_confidence, 4) if best_confidence > 0 else None,
            "match_threshold": match_threshold,
            "detection": {
                "det_score": best_face.get("det_score"),
                "is_synthetic": best_face.get("is_synthetic", False)
            }
        }

    # Auto-add to verified_roster
    verified_ids = []
    if session.verified_roster:
        try:
            verified_ids = json.loads(session.verified_roster)
        except (json.JSONDecodeError, TypeError):
            verified_ids = []

    entry_status = "duplicate" if best_student_id in verified_ids else "new"
    if entry_status == "new":
        verified_ids.append(best_student_id)
        session.verified_roster = json.dumps(verified_ids)
        db.commit()
        logger.info(f"Entry verified: {best_student_id} ({best_name}) added to verified roster. Total verified: {len(verified_ids)}")

    # Get expected count
    expected_count = 0
    if session.roster:
        try:
            expected_count = len(json.loads(session.roster))
        except (json.JSONDecodeError, TypeError):
            pass

    return {
        "status": "ok",
        "data": {
            "student_id": best_student_id,
            "name": best_name,
            "confidence": round(best_confidence, 4),
            "entry_status": entry_status,
            "verified_count": len(verified_ids),
            "expected_count": expected_count
        }
    }


class SeatAssignmentRequest(BaseModel):
    exam_session_id: int
    student_id: str
    seat_number: int
    total_seats: int = 24


@router.post("/assign_seat")
def assign_student_seat(payload: SeatAssignmentRequest, db: Session = Depends(get_db)):
    """
    Assigns a verified student to a seat and camera zone in the exam hall.
    """
    session = db.query(ExamSession).filter(ExamSession.id == payload.exam_session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Exam session not found.")

    student = db.query(Student).filter(Student.student_id == payload.student_id).first()
    if not student:
        raise HTTPException(status_code=404, detail="Student not found.")

    zone = assign_camera_zone(payload.seat_number, payload.total_seats)

    # Upsert logic: update if already assigned
    existing = db.query(ExamSeatAssignment).filter(
        ExamSeatAssignment.exam_session_id == payload.exam_session_id,
        ExamSeatAssignment.student_id == payload.student_id
    ).first()

    if existing:
        existing.seat_number = payload.seat_number
        existing.camera_zone = zone
    else:
        assignment = ExamSeatAssignment(
            exam_session_id=payload.exam_session_id,
            student_id=payload.student_id,
            seat_number=payload.seat_number,
            camera_zone=zone
        )
        db.add(assignment)

    db.commit()

    return {
        "status": "ok",
        "data": {
            "student_id": payload.student_id,
            "seat_number": payload.seat_number,
            "camera_zone": zone
        },
        "message": f"Student {payload.student_id} assigned to seat {payload.seat_number} (zone: {zone})."
    }
