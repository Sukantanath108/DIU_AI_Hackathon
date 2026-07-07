# ---
# CampusAI Suite Database Engine
# Owner: Member 3 (Backend and database engineer)
# ---

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from backend.core.config import settings

# Create database engine
# connect_args={"check_same_thread": False} is required for SQLite in multithreaded environments (like FastAPI)
engine = create_engine(
    settings.DATABASE_URL, connect_args={"check_same_thread": False}
)

# Create session maker
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Declarative base
Base = declarative_base()

def get_db():
    """
    Dependency to get the database session.
    Yields database session and closes it when the request is done.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
