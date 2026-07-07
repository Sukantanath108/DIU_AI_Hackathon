# ---
# CampusAI Suite — Standalone Webcam SmartAttend Test Script
# Owner: Member 1 (ML Lead) & Member 4 (Integration)
#
# Pure Python OpenCV script to test the full face recognition pipeline
# from a laptop webcam, bypassing the React frontend entirely.
# ---

import sys
import cv2
import requests
import numpy as np
from pathlib import Path

# Add workspace directory to python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# ========== CONFIGURABLE CONSTANTS ==========
BACKEND_URL = "http://localhost:8000"
DEFAULT_SECTION = "TEST-A"
DEFAULT_SUBJECT = "CSE301"
TEACHER_ID = "test_developer"
# =============================================


def main():
    print("=" * 60)
    print("  SmartAttend AI — Webcam Capture Test")
    print("  Backend:", BACKEND_URL)
    print("  Section:", DEFAULT_SECTION, "| Subject:", DEFAULT_SUBJECT)
    print("=" * 60)
    print()
    print("Instructions:")
    print("  SPACE  = Capture current frame")
    print("  Q      = Quit without submitting")
    print()

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam (cv2.VideoCapture(0)).")
        print("Make sure no other application is using the camera.")
        sys.exit(1)

    steps = [
        ("LEFT", "LEFT position - press SPACE to capture"),
        ("CENTER", "CENTER position - press SPACE to capture"),
        ("RIGHT", "RIGHT position - press SPACE to capture"),
    ]

    captured_frames = []

    for step_idx, (label, instruction) in enumerate(steps):
        print(f"  Photo {step_idx + 1}/3: {instruction}")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("ERROR: Failed to read frame from webcam.")
                cap.release()
                cv2.destroyAllWindows()
                sys.exit(1)

            # Draw instruction overlay
            display = frame.copy()
            h, w = display.shape[:2]

            # Semi-transparent banner at top
            overlay = display.copy()
            cv2.rectangle(overlay, (0, 0), (w, 70), (15, 23, 42), -1)
            cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)

            # Step text
            cv2.putText(display, f"Photo {step_idx + 1}/3: {label}",
                        (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (59, 130, 246), 2)
            cv2.putText(display, instruction,
                        (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # Bottom bar
            cv2.putText(display, "SPACE=Capture | Q=Quit",
                        (20, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

            cv2.imshow("SmartAttend Webcam Test", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                print("\nAborted by user. No photos submitted.")
                cap.release()
                cv2.destroyAllWindows()
                sys.exit(0)
            elif key == ord(' '):
                captured_frames.append(frame.copy())
                print(f"    ✓ Captured {label} photo ({frame.shape[1]}x{frame.shape[0]})")
                # Brief flash effect
                flash = np.ones_like(frame, dtype=np.uint8) * 255
                cv2.imshow("SmartAttend Webcam Test", flash)
                cv2.waitKey(150)
                break

    cap.release()
    cv2.destroyAllWindows()

    print()
    print("All 3 photos captured. Sending to backend...")
    print()

    # Step 1: Create attendance session
    try:
        r = requests.post(f"{BACKEND_URL}/attend/session", json={
            "section": DEFAULT_SECTION,
            "subject": DEFAULT_SUBJECT,
            "teacher_id": TEACHER_ID
        })
        r.raise_for_status()
        session_data = r.json()
        if session_data.get("status") != "ok":
            print(f"ERROR: Failed to create session: {session_data}")
            sys.exit(1)
        session_id = session_data["data"]["session_id"]
        print(f"  Session created: ID #{session_id}")
    except Exception as e:
        print(f"ERROR: Could not reach backend at {BACKEND_URL}")
        print(f"  Details: {e}")
        print(f"  Make sure the FastAPI backend is running.")
        sys.exit(1)

    # Step 2: Encode frames as JPEG and send to /attend/photos
    try:
        _, left_jpg = cv2.imencode('.jpg', captured_frames[0], [cv2.IMWRITE_JPEG_QUALITY, 85])
        _, center_jpg = cv2.imencode('.jpg', captured_frames[1], [cv2.IMWRITE_JPEG_QUALITY, 85])
        _, right_jpg = cv2.imencode('.jpg', captured_frames[2], [cv2.IMWRITE_JPEG_QUALITY, 85])

        files = {
            'front_left': ('left.jpg', left_jpg.tobytes(), 'image/jpeg'),
            'front_center': ('center.jpg', center_jpg.tobytes(), 'image/jpeg'),
            'front_right': ('right.jpg', right_jpg.tobytes(), 'image/jpeg'),
        }
        data = {'session_id': str(session_id)}

        print("  Uploading 3 photos to backend for AI fusion...")
        r = requests.post(f"{BACKEND_URL}/attend/photos", data=data, files=files)
        r.raise_for_status()
        result = r.json()

        if result.get("status") != "ok":
            print(f"ERROR: Backend returned error: {result}")
            sys.exit(1)

    except Exception as e:
        print(f"ERROR: Photo upload/processing failed: {e}")
        sys.exit(1)

    # Step 3: Print formatted attendance table
    records = result["data"]["records"]
    present = [r for r in records if r["status"] == "present"]
    absent = [r for r in records if r["status"] == "absent"]
    review = [r for r in records if r["status"] == "low_confidence"]

    print()
    print("=" * 60)
    print(f"  ATTENDANCE RESULTS — {DEFAULT_SUBJECT} TEST SESSION")
    print(f"  Session ID: #{session_id} | Section: {DEFAULT_SECTION}")
    print("=" * 60)

    print(f"\n  PRESENT ({len(present)}):")
    if present:
        for r in sorted(present, key=lambda x: x["student_id"]):
            matched = r.get("matched_photo", "unknown")
            print(f"    {r['student_id']} {r['name']} — confidence: {r['confidence']:.4f} — matched: {matched} angle")
    else:
        print("    (none)")

    print(f"\n  ABSENT ({len(absent)}):")
    if absent:
        for r in sorted(absent, key=lambda x: x["student_id"]):
            print(f"    {r['student_id']} {r['name']}")
    else:
        print("    (none)")

    print(f"\n  REVIEW / LOW CONFIDENCE ({len(review)}):")
    if review:
        for r in sorted(review, key=lambda x: x["student_id"]):
            print(f"    {r['student_id']} {r['name']} — confidence: {r['confidence']:.4f}")
    else:
        print("    none")

    print()
    print("=" * 60)
    print("  Test complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
