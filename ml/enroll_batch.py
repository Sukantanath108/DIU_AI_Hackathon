# ---
# CampusAI Suite Batch Enrollment Script
# Owner: Member 1 (ML Lead) & Member 3 (Backend Engineer)
#
# Scans data/students/ for SXXX_Name folders, extracts facial embeddings
# using InsightFace (or synthetic fallback), averages up to 5 smartly
# sampled photos per student, and stores templates in SQLite.
# ---

import os
import sys
import random
import logging
import numpy as np
import cv2
from pathlib import Path
from typing import List, Tuple

# Add workspace directory to python path to allow importing from backend
sys.path.append(str(Path(__file__).resolve().parent.parent))

from backend.core.config import settings
from backend.core.face_engine import detect_faces, is_using_fallback
from backend.database import SessionLocal, engine, Base
from backend.models import Student

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("enroll_batch")

# Maximum number of photos to use per student for embedding averaging.
# InsightFace ArcFace is pre-trained — enrollment extracts embeddings, not
# trains.  Averaging more than 5 nearly-identical photos can bias the
# average embedding toward one angle and reduce matching robustness.
MAX_PHOTOS_PER_STUDENT = 5

# Competition priority students: get a higher per-student photo budget so
# their averaged embedding is more representative across angles, lighting,
# and expressions.  These IDs are mapped to data/students/SXXX_* folders.
# 30–40 photos each gives the matcher a stronger template than the default 5
# without overfitting (averaging all ~60 of S022's photos would actually
# over-smooth the embedding).
PRIORITY_STUDENT_IDS = {"S015", "S022", "S024", "S025"}
PRIORITY_MAX_PHOTOS = 40  # Upper bound -- will be clamped to actual photo count

# When True, enrollment will NOT use the synthetic grid fallback.
# Only real face detections (InsightFace or Haar Cascade on actual faces)
# will produce embeddings. Set to False only for explicit test data.
STRICT_MODE = True


def smart_sample_photos(image_paths: List[Path], target: int = MAX_PHOTOS_PER_STUDENT) -> List[Path]:
    """
    Select up to `target` photos using a spread strategy:

    1. If n <= target:               use all of them.
    2. If target < n <= 20:          first + middle + last, then 2 random from remainder.
    3. If 20 < n <= 60:              `target` evenly spread at 0%, 25%, 50%, 75%, 100%.
    4. If n > 60:                    `target` evenly spread at 0%, 100% + (target-2) stratified.

    All paths are sorted by filename before selection to ensure deterministic
    spread across angles/timestamps.  `target` is clamped to [1, len(paths)].
    """
    sorted_paths = sorted(image_paths, key=lambda p: p.name.lower())
    n = len(sorted_paths)
    target = max(1, min(target, n))

    if n <= target:
        return sorted_paths

    if target <= 3 and n <= 20:
        # Tiny target on a small pool -- first, middle, last is the best we can do
        return [sorted_paths[0], sorted_paths[n // 2], sorted_paths[-1]][:target]

    if n <= 20:
        # Strategy: first, middle, last + 2 random from remainder
        first = sorted_paths[0]
        middle = sorted_paths[n // 2]
        last = sorted_paths[-1]
        remainder = [p for p in sorted_paths if p not in (first, middle, last)]
        extras = random.sample(remainder, min(2, len(remainder)))
        selected = [first] + extras[:1] + [middle] + extras[1:] + [last]
        return selected[:target]

    # n > 20: pick `target` evenly-spaced indices across the full range,
    # always including the first and last frames (extreme angles/lighting).
    if target == 1:
        return [sorted_paths[n // 2]]
    if target == 2:
        return [sorted_paths[0], sorted_paths[-1]]
    indices = [int(round(i * (n - 1) / (target - 1))) for i in range(target)]
    # De-dup while preserving order (rounded indices can collide)
    seen = set()
    unique_indices = []
    for idx in indices:
        if idx not in seen:
            seen.add(idx)
            unique_indices.append(idx)
    return [sorted_paths[i] for i in unique_indices]


def create_mock_images_if_empty(students_dir: Path) -> None:
    """
    Creates mock student directories and synthetic dummy photos if the students directory is empty,
    so that the team and judges can test the pipeline immediately.
    """
    if any(students_dir.iterdir()):
        logger.info(f"Students directory is not empty. Proceeding with existing dataset.")
        return

    logger.info("Students directory is empty. Generating mock student photo files for 35 students (S001 to S035)...")

    mock_names = [
        "Arif", "Sadia", "Tanvir", "Nusrat", "Mahmud", "Fariha", "Imran", "Tasnim",
        "Jamil", "Ayesha", "Rahat", "Nabila", "Rakib", "Sumaiya", "Sajid", "Farhana",
        "Rifat", "Zarin", "Munir", "Humaira", "Tariq", "Laila", "Asif", "Mehnaz",
        "Sakib", "Anika", "Fahim", "Ishrat", "Nayeem", "Karima", "Habib", "Roya",
        "Zeeshan", "Shirin", "Wasif"
    ]

    for i, name in enumerate(mock_names):
        student_id = f"S{i+1:03d}"
        folder_name = f"{student_id}_{name}"
        student_folder = students_dir / folder_name
        student_folder.mkdir(parents=True, exist_ok=True)

        # Generate 3 dummy photos with different gradient colors to represent different angles
        for photo_idx in range(1, 4):
            photo_path = student_folder / f"photo{photo_idx}.jpg"
            # Create a simple 300x300 BGR color block
            img = np.zeros((300, 300, 3), dtype=np.uint8)
            # Add a colored square in the center representing a face
            cv2.rectangle(img, (75, 75), (225, 225), (100 + i * 4, 150 + photo_idx * 20, 50 + i * 5), -1)
            # Add student label text
            cv2.putText(img, f"{student_id}", (110, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imwrite(str(photo_path), img)

    logger.info("Successfully created 35 mock student profiles with synthetic images.")


def is_mock_dataset(students_dir: Path) -> bool:
    """
    Returns True if the dataset appears to be the auto-generated mock data
    (folders named S001_Arif, S002_Sadia, etc. with photo1/photo2/photo3.jpg).
    """
    first_folder = sorted([f for f in students_dir.iterdir() if f.is_dir()])
    if not first_folder:
        return True
    # Check if the first folder matches mock naming pattern
    name = first_folder[0].name
    if name.startswith("S001_") or name.startswith("S002_"):
        photos = list(first_folder[0].glob("photo*.jpg"))
        if photos:
            # Read first photo and check if it's a tiny color block (300x300)
            img = cv2.imread(str(photos[0]))
            if img is not None and img.shape == (300, 300, 3):
                return True
    return False


def enroll_all_students() -> None:
    """
    Scans the data/students/ folder, extracts facial embeddings,
    averages them if multiple photos exist for a student, and stores
    them in SQLite.
    """
    # Initialize DB tables
    logger.info("Initializing database tables...")
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()

    # Ensure students directory exists
    students_dir = settings.DATA_DIR / "students"
    students_dir.mkdir(parents=True, exist_ok=True)

    # Generate mock images if empty to make the pipeline testable out-of-the-box
    create_mock_images_if_empty(students_dir)

    # Determine if this is mock data or real data
    mock_data = is_mock_dataset(students_dir)
    # allow_synthetic = True for mock data, False for real data (unless STRICT_MODE is off)
    use_synthetic = mock_data or not STRICT_MODE

    if mock_data:
        logger.info("Detected MOCK dataset — synthetic fallback allowed for testing.")
    else:
        logger.info("Detected REAL dataset — synthetic fallback DISABLED. Only real face detections will produce embeddings.")

    # Scan subdirectories
    student_folders = sorted([f for f in students_dir.iterdir() if f.is_dir()])
    logger.info(f"Found {len(student_folders)} student folders for enrollment.")

    success_count = 0
    failed_count = 0
    summary_lines: List[str] = []

    for folder in student_folders:
        parts = folder.name.split("_", 1)
        if len(parts) < 2:
            logger.warning(f"Skipping directory with invalid naming format: {folder.name} (Expected: SXXX_Name)")
            continue

        student_id = parts[0]
        name = parts[1].replace("_", " ")

        # Gather all image files
        image_extensions = (".jpg", ".jpeg", ".png")
        all_image_paths = [p for p in folder.iterdir() if p.suffix.lower() in image_extensions]

        if not all_image_paths:
            msg = f"  SKIP   {student_id} {folder.name}: 0 photos found"
            summary_lines.append(msg)
            logger.warning(f"No valid photos found for student {student_id} ({name}). Skipping.")
            failed_count += 1
            continue

        # Smart sample to cap at MAX_PHOTOS_PER_STUDENT, with a higher cap
        # for competition priority students (S015, S022, S024, S025) so the
        # averaged embedding is more representative across angles/lighting.
        total_available = len(all_image_paths)
        if student_id in PRIORITY_STUDENT_IDS:
            target_photos = min(PRIORITY_MAX_PHOTOS, total_available)
            priority_tag = " [PRIORITY]"
        else:
            target_photos = MAX_PHOTOS_PER_STUDENT
            priority_tag = ""
        selected_paths = smart_sample_photos(all_image_paths, target=target_photos)

        logger.info(
            f"Processing student {student_id} — {name}"
            f"{priority_tag} ({len(selected_paths)}/{total_available} photos selected)..."
        )

        embeddings_list: List[np.ndarray] = []
        faces_detected_count = 0
        faces_failed_count = 0
        primary_photo_path = str(selected_paths[0].relative_to(settings.BASE_DIR))

        for img_path in selected_paths:
            img = cv2.imread(str(img_path))
            if img is None:
                logger.warning(f"Failed to read image: {img_path.name}")
                faces_failed_count += 1
                continue

            # Detect faces — allow_synthetic=False for real data
            faces = detect_faces(img, allow_synthetic=use_synthetic)

            if faces:
                # Get the face with the largest bounding box area (assumed to be the student)
                largest_face = max(faces, key=lambda f: (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]))
                
                # Check if this is a synthetic embedding — warn but still use for mock data
                if largest_face.get("is_synthetic", False) and not mock_data:
                    logger.warning(f"  {img_path.name}: face detected by Haar only (synthetic embedding) — SKIPPED for real data")
                    faces_failed_count += 1
                    continue
                
                embeddings_list.append(largest_face["embedding"])
                faces_detected_count += 1
            else:
                logger.warning(f"No face detected in photo {img_path.name} for {student_id}")
                faces_failed_count += 1

        if not embeddings_list:
            msg = f"  FAIL   {student_id} {folder.name}: {total_available} photos, 0 valid faces → NOT ENROLLED"
            summary_lines.append(msg)
            logger.error(f"Failed to extract any face embeddings for student {student_id}. Skipping.")
            failed_count += 1
            continue

        # Average the embeddings to create a single robust 512-d template
        avg_embedding = np.mean(embeddings_list, axis=0)
        norm = np.linalg.norm(avg_embedding)
        if norm > 0:
            avg_embedding = avg_embedding / norm  # Re-normalize

        # Check if student already exists in DB
        db_student = db.query(Student).filter(Student.student_id == student_id).first()

        # Standard metadata
        section = "A"
        department = "CSE"

        if db_student:
            # Update existing student
            db_student.name = name
            db_student.section = section
            db_student.department = department
            db_student.set_embedding(avg_embedding)
            db_student.photo_path = primary_photo_path
            logger.info(f"Updated student {student_id} in database.")
        else:
            # Insert new student
            new_student = Student(
                student_id=student_id,
                name=name,
                section=section,
                department=department,
                photo_path=primary_photo_path
            )
            new_student.set_embedding(avg_embedding)
            db.add(new_student)
            logger.info(f"Enrolled student {student_id} into database.")

        db.commit()
        success_count += 1

        # Build summary line with detailed stats
        embedding_type = "REAL" if not is_using_fallback() else "SYNTHETIC"
        if total_available == 1 and faces_detected_count == 1:
            msg = f"  WARN   {student_id} {folder.name}: 1 photo, 1 face → embedded ({embedding_type}) — single-photo accuracy risk{priority_tag}"
        elif faces_failed_count > 0:
            msg = f"  OK     {student_id} {folder.name}: {total_available} photos, {faces_detected_count} faces ({faces_failed_count} failed) → embedded ({embedding_type}){priority_tag}"
        else:
            msg = f"  OK     {student_id} {folder.name}: {total_available} photos, {faces_detected_count} faces → embedded ({embedding_type}){priority_tag}"
        summary_lines.append(msg)

    db.close()

    # Print final summary
    engine_type = "Synthetic Fallback Mode" if is_using_fallback() else "InsightFace buffalo_l Mode"
    dataset_type = "MOCK TEST DATA" if mock_data else "REAL STUDENT DATA"
    print()
    print("=" * 70)
    print(f"  BATCH ENROLLMENT COMPLETE — {engine_type}")
    print(f"  Dataset: {dataset_type}")
    print(f"  Synthetic fallback: {'ALLOWED (mock data)' if use_synthetic else 'DISABLED (real data)'}")
    print("=" * 70)
    print(f"  Students enrolled:  {success_count} / {len(student_folders)}")
    print(f"  Students failed:    {failed_count}")
    print(f"  Max photos/student: {MAX_PHOTOS_PER_STUDENT}")
    print(f"  Priority students:  {sorted(PRIORITY_STUDENT_IDS)}  → up to {PRIORITY_MAX_PHOTOS} photos each")
    print("-" * 70)
    for line in summary_lines:
        print(line)
    print("=" * 70)
    
    if failed_count > 0 and not mock_data:
        print()
        print("⚠️  ATTENTION: Some students could not be enrolled because no face")
        print("   was detected in any of their photos. Common causes:")
        print("   - Photo does not contain a visible face (group photo, back of head)")
        print("   - Photo is too dark, blurry, or low resolution")
        print("   - Face is heavily occluded (mask, sunglasses, hand)")
        print("   - Image file is corrupted")
        print("   Re-take photos for these students and run enrollment again.")
        print()


if __name__ == "__main__":
    enroll_all_students()
