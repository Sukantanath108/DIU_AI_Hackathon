# ---
# CampusAI Suite Backend Entry Point
# Owner: Member 3 (Backend Engineer)
# ---

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from backend.database import engine, Base
from backend.core.config import settings
from backend.core.face_engine import init_face_engine
from backend.core.yolo_engine import init_yolo_engine
from backend.core.gaze_engine import init_gaze_engine

from backend.routers.enroll import router as enroll_router
from backend.routers.attend import router as attend_router
from backend.routers.exam import router as exam_router

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("main")

# Create tables if not exist
logger.info("Initializing database tables...")
Base.metadata.create_all(bind=engine)

# Safe migration: add verified_roster column if missing (for existing DBs)
try:
    from sqlalchemy import inspect, text
    inspector = inspect(engine)
    exam_sessions_cols = {c["name"] for c in inspector.get_columns("exam_sessions")}
    if "verified_roster" not in exam_sessions_cols:
        with engine.connect() as conn:
            conn.execute(text("ALTER TABLE exam_sessions ADD COLUMN verified_roster TEXT"))
            conn.commit()
            logger.info("Migration: added verified_roster column to exam_sessions")
    else:
        logger.info("Migration: verified_roster column already present.")
except Exception as e:
    logger.error(f"Migration failed for verified_roster: {e}")
    raise

# Initialize AI/CV Engines
logger.info("Initializing core AI wrappers (InsightFace, YOLOv8, MediaPipe)...")
try:
    init_face_engine()
except Exception as e:
    logger.error(f"Failed to initialize Face Engine: {e}")

try:
    init_yolo_engine()
except Exception as e:
    logger.error(f"Failed to initialize YOLO Engine: {e}")

try:
    init_gaze_engine()
except Exception as e:
    logger.error(f"Failed to initialize Gaze Engine: {e}")

# Create FastAPI application
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Backend API for CampusAI Suite (SmartAttend & ExamShield)"
)

# Enable CORS for all frontends (React on port 3000, Streamlit on port 8501)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Static Directories for serving files
# Serving data/ (for enrolled student photos) and screenshots/ (for proctoring incident photos)
data_dir = settings.DATA_DIR
screenshots_dir = settings.SCREENSHOTS_DIR

data_dir.mkdir(parents=True, exist_ok=True)
screenshots_dir.mkdir(parents=True, exist_ok=True)

app.mount("/data", StaticFiles(directory=str(data_dir)), name="data")
app.mount("/screenshots", StaticFiles(directory=str(screenshots_dir)), name="screenshots")

# Include Routers
app.include_router(enroll_router)
app.include_router(attend_router)
app.include_router(exam_router)

@app.get("/")
def read_root():
    return {
        "status": "ok",
        "message": "Welcome to the CampusAI Suite MVP Backend API.",
        "version": settings.VERSION,
        "config": {
            "ai_fallback_enabled": settings.USE_AI_FALLBACK,
            "database_url": settings.DATABASE_URL
        }
    }
