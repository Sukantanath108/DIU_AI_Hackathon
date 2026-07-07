# ---
# CampusAI Suite ORM Models
# Owner: Member 3 (Backend and database engineer)
# ---

import datetime
import numpy as np
from typing import Optional
from sqlalchemy import Column, Integer, String, Float, LargeBinary, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from backend.database import Base

class Student(Base):
    """
    Student model representing enrolled students.
    """
    __tablename__ = "students"

    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, nullable=False)
    section = Column(String, nullable=False)
    department = Column(String, nullable=False)
    # 512-dimensional float32 vector serialized with tobytes()
    embedding = Column(LargeBinary, nullable=False)
    photo_path = Column(String, nullable=True)

    # Relationships
    attendance_records = relationship("AttendanceRecord", back_populates="student", cascade="all, delete-orphan")
    exam_events = relationship("ExamEvent", back_populates="student", cascade="all, delete-orphan")
    seat_assignments = relationship("ExamSeatAssignment", back_populates="student", cascade="all, delete-orphan")

    def get_embedding(self) -> np.ndarray:
        """
        Retrieves the 512-dimensional embedding as a numpy float32 array.
        """
        return np.frombuffer(self.embedding, dtype=np.float32)

    def set_embedding(self, emb: np.ndarray) -> None:
        """
        Serializes a numpy float32 array and stores it in the database.
        """
        self.embedding = emb.astype(np.float32).tobytes()


class AttendanceSession(Base):
    """
    Session created by a teacher for recording classroom attendance.
    """
    __tablename__ = "attendance_sessions"

    id = Column(Integer, primary_key=True, index=True)
    section = Column(String, index=True, nullable=False)
    subject = Column(String, nullable=False)
    teacher_id = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    # Relationships
    records = relationship("AttendanceRecord", back_populates="session", cascade="all, delete-orphan")


class AttendanceRecord(Base):
    """
    Individual attendance status for a student in a session.
    """
    __tablename__ = "attendance_records"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("attendance_sessions.id"), nullable=False)
    student_id = Column(String, ForeignKey("students.student_id"), nullable=False)
    # status can be 'present', 'absent', or 'low_confidence'
    status = Column(String, nullable=False, default="absent")
    confidence = Column(Float, nullable=False, default=0.0)
    # 0 = not overridden, 1 = manually overridden by teacher
    overridden = Column(Integer, nullable=False, default=0)

    # Relationships
    session = relationship("AttendanceSession", back_populates="records")
    student = relationship("Student", back_populates="attendance_records")


class ExamSession(Base):
    """
    Real-time examination session proctored by ExamShield AI.
    """
    __tablename__ = "exam_sessions"

    id = Column(Integer, primary_key=True, index=True)
    course = Column(String, nullable=False)
    hall = Column(String, nullable=False)
    invigilator = Column(String, nullable=False)
    started_at = Column(DateTime, default=datetime.datetime.utcnow)
    ended_at = Column(DateTime, nullable=True)
    # JSON list of student_ids in the exam roster e.g. '["S001","S005","S022"]'
    roster = Column(Text, nullable=True)
    # Human-readable roster label e.g. "CSE-421 Midterm"
    roster_name = Column(String, nullable=True)
    # JSON list of student_ids verified at exam entry e.g. '["S001","S007"]'
    # Built automatically as students are verified via entry photos
    verified_roster = Column(Text, nullable=True)

    # Relationships
    events = relationship("ExamEvent", back_populates="exam_session", cascade="all, delete-orphan")


class ExamEvent(Base):
    """
    Suspicious event or behavior detected during an exam session.
    """
    __tablename__ = "exam_events"

    id = Column(Integer, primary_key=True, index=True)
    exam_session_id = Column(Integer, ForeignKey("exam_sessions.id"), nullable=False)
    student_id = Column(String, ForeignKey("students.student_id"), nullable=False)
    # event_type can be: 'phone', 'cheat_sheet', 'book', 'gaze_deviation',
    # 'multi_face', 'unverified', 'unknown_person'
    # 'unknown_person' has score_delta=0 and is logged for evidence only.
    event_type = Column(String, nullable=False)
    score_delta = Column(Integer, nullable=False)
    screenshot_path = Column(String, nullable=True)
    occurred_at = Column(DateTime, default=datetime.datetime.utcnow)

    # Relationships
    exam_session = relationship("ExamSession", back_populates="events")
    student = relationship("Student", back_populates="exam_events")


class ExamSeatAssignment(Base):
    """
    Maps a student to a specific seat and camera zone in an exam session.
    Used by Full Entry Mode for YOLO zone-based event attribution.
    """
    __tablename__ = "exam_seat_assignments"

    id = Column(Integer, primary_key=True, index=True)
    exam_session_id = Column(Integer, ForeignKey("exam_sessions.id"), nullable=False)
    student_id = Column(String, ForeignKey("students.student_id"), nullable=False)
    seat_number = Column(Integer, nullable=False)
    camera_zone = Column(String, nullable=False)  # 'left' | 'center' | 'right' | 'unknown'
    verified_at = Column(DateTime, default=datetime.datetime.utcnow)

    # Relationships
    exam_session = relationship("ExamSession")
    student = relationship("Student", back_populates="seat_assignments")


def assign_camera_zone(seat_number: int, total_seats: int) -> str:
    """
    Assigns a camera zone based on seat position.
    Left third → 'left', middle third → 'center', right third → 'right'.
    """
    if total_seats <= 0:
        return 'unknown'
    third = total_seats / 3
    if seat_number <= third:
        return 'left'
    elif seat_number <= 2 * third:
        return 'center'
    else:
        return 'right'
