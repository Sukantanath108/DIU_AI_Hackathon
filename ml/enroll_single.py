# Enroll a specific student into the DB with real ArcFace embeddings
# Usage: python ml/enroll_single.py S024
import sys
import os
import cv2
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

from backend.database import SessionLocal, engine, Base
from backend.models import Student
from backend.core.face_engine import detect_faces, is_using_fallback

def enroll_student(student_id: str):
    students_dir = Path("data/students")
    
    # Find the folder
    folder = None
    for f in students_dir.iterdir():
        if f.is_dir() and f.name.startswith(student_id):
            folder = f
            break
    
    if folder is None:
        print(f"ERROR: No folder found for {student_id} in {students_dir}")
        return False
    
    parts = folder.name.split("_", 1)
    name = parts[1].replace("_", " ") if len(parts) >= 2 else student_id
    
    photos = sorted([p for p in folder.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png')])
    print(f"Student: {student_id} ({name})")
    print(f"Folder: {folder.name}")
    print(f"Photos: {len(photos)}")
    print(f"Engine: {'SYNTHETIC' if is_using_fallback() else 'InsightFace (REAL)'}")
    print()
    
    # Process photos until we get a good embedding (use best of first 5)
    best_embedding = None
    best_score = 0.0
    processed = 0
    detected_count = 0
    
    for photo_path in photos[:10]:  # Try up to 10 photos
        img = cv2.imread(str(photo_path))
        if img is None:
            print(f"  SKIP {photo_path.name}: cannot read")
            continue
        
        processed += 1
        h, w = img.shape[:2]
        
        faces = detect_faces(img, allow_synthetic=False)
        
        if faces:
            detected_count += 1
            best_face = max(faces, key=lambda f: f["det_score"])
            score = best_face["det_score"]
            is_synth = best_face["is_synthetic"]
            emb_norm = float(np.linalg.norm(best_face["embedding"]))
            
            print(f"  OK {photo_path.name}: {w}x{h} -> score={score:.4f}, synthetic={is_synth}, norm={emb_norm:.4f}")
            
            if score > best_score and not is_synth:
                best_score = score
                best_embedding = best_face["embedding"]
        else:
            print(f"  FAIL {photo_path.name}: {w}x{h} -> 0 faces")
    
    print()
    print(f"Processed: {processed}, Detected: {detected_count}")
    
    if best_embedding is None:
        print(f"FAILED: No real face embedding found for {student_id}")
        return False
    
    print(f"Best detection score: {best_score:.4f}")
    print(f"Embedding norm: {float(np.linalg.norm(best_embedding)):.4f}")
    print(f"Embedding dim: {len(best_embedding)}")
    
    # Save to DB
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    
    existing = db.query(Student).filter(Student.student_id == student_id).first()
    if existing:
        print(f"\nUpdating existing DB entry for {student_id}...")
        existing.name = name
        existing.set_embedding(best_embedding)
    else:
        print(f"\nCreating new DB entry for {student_id}...")
        student = Student(student_id=student_id, name=name, section="A", department="CSE")
        student.set_embedding(best_embedding)
        db.add(student)
    
    db.commit()
    
    # Verify
    verify = db.query(Student).filter(Student.student_id == student_id).first()
    v_emb = verify.get_embedding()
    print(f"Verified: {verify.student_id} {verify.name}, embedding norm={float(np.linalg.norm(v_emb)):.4f}")
    
    db.close()
    print(f"\nSUCCESS: {student_id} enrolled with REAL ArcFace embedding")
    return True

if __name__ == "__main__":
    sid = sys.argv[1] if len(sys.argv) > 1 else "S024"
    enroll_student(sid)
